import json
import time
import queue
import traceback
import threading
from multiprocessing import Pool
import numpy as np
import paho.mqtt.client as mqtt
import io
from PIL import Image
import base64
import cv2
import asyncio
from aiohttp import ClientSession
from pyk4a import PyK4A, K4AException, FPS
from pyk4a import Config as k4aConf
from constant import K4A_DEFINITIONS
from utils import label_map_util
from utils import visualization_utils as vis_util
from constant import LOGGER_OBJECT_DETECTOR_MAIN, OBJECT_DETECTOR_CONFIG_DICT, \
    LOGGER_OBJECT_DETECTOR_MQTT_LOOP, LOGGER_OBJECT_DETECTOR_RUNNER, \
    LOGGER_OBJECT_DETECTOR_ASYNC_LOOP, LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT, \
    LOGGER_OBJECT_DETECTOR_KILL_SWITCH, LOGGER_ASYNC_RUN_DETECTION, \
    LOGGER_ASYNC_RUN_CAPTURE_LOOP, LOGGER_OBJECT_DETECTOR_SORT_TRACKED_OBJECTS
from logger import RoboLogger
from tracked_object import BoundingBox, TrackedObject, FMT_STANDARD, \
    FMT_TF_BBOX, FMT_TRACKER
log = RoboLogger.getLogger()
log.warning(LOGGER_OBJECT_DETECTOR_MAIN, msg="Libraries loaded.")


class ObjectDetector(object):
    """ main class for the object detector running on the jetson """

    def __init__(self, configuration: dict) -> None:
        """camera_index: int, frozen_graph_path: str, label_map_path: str, num_classes: int) -> None:"""
        self.configuration = configuration
        # Adjust configuration (to make strings into constants)
        self.__adjustConfigDict(self.configuration)
        self._camera_index = configuration[OBJECT_DETECTOR_CONFIG_DICT]['camera_index']
        self.num_classes = configuration[OBJECT_DETECTOR_CONFIG_DICT]['num_classes']
        self.label_map_path = configuration[OBJECT_DETECTOR_CONFIG_DICT]['label_map_path']
        self.mqtt_message_queue = queue.Queue()
        self.exception_queue = queue.Queue()
        self.ready_queue = queue.Queue()   # used to signal that there's an image ready for processing
        k4a_config_dict = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['k4a_device']
        self.k4a_device = PyK4A(
            k4aConf(color_resolution=k4a_config_dict['color_resolution'],
                    depth_mode=k4a_config_dict['depth_mode'],
                    camera_fps=k4a_config_dict['camera_fps'],
                    synchronized_images_only=k4a_config_dict['synchronized_images_only']))
        # defaults to false, put to true for debugging (need video display)
        label_map = label_map_util.load_labelmap(self.label_map_path)
        categories = label_map_util.\
            convert_label_map_to_categories(
                label_map,
                max_num_classes=self.num_classes,
                use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.show_video = False
        self.show_depth_video = False
        if k4a_config_dict['camera_fps'] == FPS.FPS_5:
            self.frame_duration = 1. / 5
        elif k4a_config_dict['camera_fps'] == FPS.FPS_15:
            self.frame_duration = 1. / 15
        elif k4a_config_dict['camera_fps'] == FPS.FPS_30:
            self.frame_duration = 1. / 30
        else:
            raise('Unsupported frame rate {}'.format(
                k4a_config_dict['camera_fps']))
        self.detection_threshold = 0.5
        self.tracked_objects = []
        self.default_tracker = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['open_cv_default_tracker']

    def run(self) -> None:
        """
        params:
        Launches the runner that runs most things (MQTT queue, etc.)
        """
        try:
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg=f'Initializing Kinect')
            self.k4a_device.connect()
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg=f'K4A device initialized...')
            # Create Process pool
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg=f'Creating Process Pool')
            self.process_pool = Pool(processes=1)

            # Launch the MQTT thread listener
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg='Launching MQTT thread.')
            self.threadMQTT = threading.Thread(
                target=self.thread_mqtt_listener, name='MQTTListener')
            self.threadMQTT.start()

            # Launch main event loop in a separate thread
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg='Launching asyncio event loop Thread')
            self.eventLoopThread = threading.Thread(
                target=self.thread_event_loop, name='asyncioEventLoop')
            self.eventLoopThread.start()

            # Launching capture loop thread
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg=f'Launching Capture Loop Thread')
            self.captureLoopThread = threading.Thread(
                target=self.thread_capture_loop, name='captureLoop')
            self.captureLoopThread.start()

            while True:
                try:
                    exc = self.exception_queue.get(block=True)
                except queue.Empty:
                    pass
                else:
                    log.error(LOGGER_OBJECT_DETECTOR_RUNNER,
                              'Exception handled in one of the spawned process : {}'.format(exc))
                    raise exc
                self.eventLoopThread.join(0.2)
                if self.eventLoopThread.isAlive():
                    continue
                else:
                    break

        except SystemExit:
            # raise the exception up the stack
            raise

        except K4AException:
            log.error(LOGGER_OBJECT_DETECTOR_RUNNER,
                      f'Error with K4A : {traceback.print_exc()}')

        except Exception:
            log.error(LOGGER_OBJECT_DETECTOR_RUNNER,
                      f'Error : {traceback.print_exc()}')

    def thread_mqtt_listener(self):
        """
        MQTT Thread launching the loop and subscripbing to the right topics
        """
        mqtt_default_qos = 2
        self.mqtt_topics = [(topic, mqtt_default_qos)
                            for topic in self.configuration['mqtt']['subscribedTopics']]

        def on_connect(client, userdata, flags, rc):
            log.warning(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                        'Connected to MQTT broker. Result code {}'.format(str(rc)))
            mqtt_connect_result, self.mqtt_connect_mid = client.subscribe(
                self.mqtt_topics)
            if mqtt_connect_result == mqtt.MQTT_ERR_SUCCESS:
                log.warning(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                            'Successfully subscribed to {}'.format(self.mqtt_topics))
            else:
                log.error(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                          'MQTT Broker subscription problem.')

        def on_message(client, userdata, message):
            """ callback function used for the mqtt client (called when
            a new message is publisehd to one of the queues we subscribe to)
            """
            log.info(LOGGER_OBJECT_DETECTOR_MQTT_LOOP, "Received MID {} : '{} on topic '{}' with QoS {} ".format(
                message.mid, str(message.payload), message.topic, str(message.qos)))
            self.mqtt_message_queue.put_nowait(message)

        def on_disconnect(client, userdata, rc=0):
            """callback for handling disconnects
            """
            log.info(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                     "Disconnected MQTT result code = {}. Should automatically re-connect to broker".format(rc))

        def on_subscribe(client, userdata, mid, granted_qos):
            if mid == self.mqtt_connect_mid:
                log.warning(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                            "Subscribed to topics. Granted QOS = {}".format(granted_qos))
            else:
                log.error(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                          "Strange... MID doesn't match self.mqtt_connect_mid")

        try:
            self.mqttClient = mqtt.Client(
                client_id="jetsonbot", clean_session=True, transport=self.configuration["mqtt"]["brokerProto"])
            self.mqttClient.enable_logger(
                logger=RoboLogger.getSpecificLogger(LOGGER_OBJECT_DETECTOR_MQTT_LOOP))
            self.mqttClient.on_subscribe = on_subscribe
            self.mqttClient.on_connect = on_connect
            self.mqttClient.on_disconnect = on_disconnect
            self.mqttClient.on_message = on_message
            self.mqttClient.connect(host=self.configuration["mqtt"]["brokerIP"],
                                    port=self.configuration["mqtt"]["brokerPort"])
            self.mqttClient.loop_start()
        except Exception:
            log.error(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                      f'Error : {traceback.print_exc()}')
            # Bump up the problem...
            raise

    def thread_event_loop(self):
        """
        Main event asyncio eventloop launched in a separate thread
        """
        try:
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                        msg=f'Launching asyncio event loop')
            self.eventLoop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.eventLoop)
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                        msg=f'Launching process MQTT message task')
            self.eventLoop.create_task(
                self.async_process_mqtt_messages(loopDelay=0.25))
            # log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
            #             msg=f'Launching async_run_capture_loop task')
            # self.eventLoop.create_task(self.async_run_capture_loop())
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                        msg=f'Launching asykc_run_detection task')
            self.eventLoop.create_task(self.async_run_detection(loopDelay=0.5))
            self.eventLoop.run_forever()
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                        msg=f'Asyncio event loop started')

        except Exception:
            log.error(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                      f'Error : {traceback.print_exc()}')
            raise
        finally:
            self.eventLoop.stop()
            self.eventLoop.close()

    async def async_run_detection(self, loopDelay=0) -> None:
        """
        returns object detection
        delay:loopDelay:delay to pause between frames scoring
        returns:task
        """
        try:
            headers = {'Content-Type': 'application/json'}
            hostname = self.configuration['object_detector']['endpoint_hostname']
            port = self.configuration['object_detector']['endpoint_port']
            path = self.configuration['object_detector']['endpoint_path']
            model_url = f'http://{hostname}:{port}/{path}'
            logging_loops = 50
            loop_time = 0
            n_loops = 0
            task_duration = 0
            # Only launch loop when ready (image on the self.rgb...)
            itm = self.ready_queue.get(block=True)
            if itm is None:
                raise('Empty Queue...')
            # Launch the loop
            while True:
                height, width = self.rgb_image_color_np.shape[:2]
                # Send image to bytes buffer
                im = Image.fromarray(self.rgb_image_color_np).resize((300, 300))
                buf = io.BytesIO()
                im.save(buf, format='PNG')
                base64_encoded_image = base64.b64encode(buf.getvalue())
                payload = {'image': base64_encoded_image.decode('ascii')}
                start_time = time.time()
                async with ClientSession() as session:
                    async with session.post(url=model_url, data=json.dumps(payload), headers=headers) as response:
                        end_time = time.time()
                        if response.status == 200:
                            body = await response.json()
                            log.debug(LOGGER_ASYNC_RUN_DETECTION,
                                      msg=f'Got model response... iteration {n_loops}')
                            # SEND BODY, SELF.DETECTION_THRESHOLD,  TO FUNCTION
                            task_start_time = time.time()
                            new_tracked_objects = self.process_pool.starmap(
                                sort_tracked_objects, [(body,
                                                       (height, width),
                                                       im,
                                                       self.detection_threshold,
                                                       self.default_tracker,
                                                       self.tracked_objects)])
                            self.tracked_objects.append(new_tracked_objects)
                            # task = pool.submit(sort_tracked_objects, height, width)
                            # task = pool.submit(sort_tracked_objects, body, height, width, im)
                            # self.tracked_objects = task.result()  # temp_tracked_objects
                            task_duration += time.time() - task_start_time
                            loop_time += (end_time - start_time)
                            if n_loops % logging_loops == 0:
                                task_duration /= logging_loops
                                loop_time /= logging_loops
                                log.debug(LOGGER_ASYNC_RUN_DETECTION,
                                          msg=f'Average loop for the past {logging_loops} '
                                              f'iteration is {loop_time:.3f}s')
                                log.debug(LOGGER_ASYNC_RUN_DETECTION,
                                          msg=f'Average task time for past {logging_loops} '
                                              f'iterations is {task_duration:.3f}s')
                                loop_time = 0
                                task_duration = 0
                            n_loops += 1
                        else:
                            raise (
                                f'HTTP response code is {response.status} for detection service...')
                # Pause for loopDelay seconds
                asyncio.sleep(loopDelay)
        except Exception:
            log.error(LOGGER_ASYNC_RUN_DETECTION,
                      f'Error : {traceback.print_exc()}')
            raise (f'Error : {traceback.print_exc()}')

    def thread_capture_loop(self) -> None:
        """
        Video capture loop (can optionally display video) for debugging purposes
        """
        try:
            # Launch the loop
            n_loops = 0
            average_duration = 0
            while True:
                start_time = time.time()
                # Read frame from camera
                bgra_image_color_np, image_depth_np = \
                    self.k4a_device.get_capture(
                        color_only=False,
                        transform_depth_to_color=True)
                log.debug(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                          msg=f'Ran capture (loop {n_loops})')
                self.rgb_image_color_np = bgra_image_color_np[:, :, :3][..., ::-1].copy()   # maybe remove copy? ONLY COPY IN THE TF LOOP to LOWER CPU USAGE
                # Update trackers at every loop
                for tracked_object in self.tracked_objects:
                    tracked_object.update(image=self.rgb_image_color_np)
                # Show video in debugging mode
                if self.show_video:
                    # Visualization of the results of a detection.
                    detection_boxes = [[list(obj.bounding_box.get_bbox(
                        fmt=FMT_TF_BBOX,
                        use_normalized_coordinates=True))
                        for obj in self.tracked_objects]]
                    detection_classes = [[obj.object_class for obj in self.tracked_objects]]
                    detection_scores = [[obj.score for obj in self.tracked_objects]]
                    bgra_image_color_np_boxes = vis_util.visualize_boxes_and_labels_on_image_array(
                        bgra_image_color_np[:, :, :3],
                        np.squeeze(detection_boxes),
                        np.squeeze(detection_classes).astype(np.int32),
                        np.squeeze(detection_scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        min_score_thresh=self.detection_threshold,
                        line_thickness=4)
                    cv2.imshow('object detection', cv2.resize(
                        bgra_image_color_np_boxes, (1024, 576)))
                if self.show_depth_video:
                    cv2.imshow('depth view', cv2.resize(
                        image_depth_np, (1024, 576)))
                if self.show_depth_video or self.show_video:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        self.show_video = False
                        self.show_depth_video = False
                duration = time.time() - start_time
                sleep_time = max(0, duration - self.frame_duration)
                average_duration += duration
                # await asyncio.sleep(sleep_time)
                time.sleep(sleep_time)
                n_loops += 1
                if n_loops % 50 == 0:
                    duration_50 = average_duration
                    average_duration /= 50
                    log.warning(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                                msg=f'Ran 50 in {duration_50:.2f}s - {average_duration:.2f}s/loop or {1/average_duration:.2f} loop/sec')
                # only do this after the first loop is done
                if n_loops == 1:
                    self.ready_queue.put_nowait('image_ready')
        except Exception:
            log.error(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                      f'Error : {traceback.print_exc()}')
            raise (f'Error : {traceback.print_exc()}')

    async def async_process_mqtt_messages(self, loopDelay=0.25):
        """
        This function receives the messages from MQTT to run various functions on the Jetson
        """
        log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                    msg='Launching MQTT processing async task')
        while True:
            try:
                if self.mqtt_message_queue.empty() is False:
                    # Remove the first in the list, will pause until there is something
                    currentMQTTMoveMessage = self.mqtt_message_queue.get()
                    # Decode message received
                    msgdict = json.loads(
                        currentMQTTMoveMessage.payload.decode('utf-8'))

                    # Check if need to shut down
                    if currentMQTTMoveMessage.topic == 'bot/kill_switch':
                        log.warning(
                            LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT, msg='Kill switch activated')
                        self.kill_switch()

                    elif currentMQTTMoveMessage.topic == 'bot/jetson/detectObjectDisplayVideo':
                        log.info(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                                 msg='Showing Video on screen (NEED A DISPLAY CONNECTED!)')
                        if 'show_video' in msgdict:
                            show_video = msgdict['show_video']
                        else:
                            show_video = True
                        self.show_video = show_video
                        if 'show_depth_video' in msgdict:
                            show_depth_video = msgdict['show_depth_video']
                        else:
                            show_depth_video = False
                        self.show_depth_video = show_depth_video

                    elif currentMQTTMoveMessage.topic == 'bot/logger':
                        # Changing the logging level on the fly...
                        log.setLevel(msgdict['logger'], lvl=msgdict['level'])
                    else:
                        raise NotImplementedError
                await asyncio.sleep(loopDelay)

            except NotImplementedError:
                log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                            f'MQTT topic not implemented.')
            except asyncio.futures.CancelledError:
                log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                            f'Cancelled the MQTT dequeing task.')
            except Exception:
                log.error(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                          f'Error : {traceback.print_exc()}')
                # raise

    def kill_switch(self):
        try:
            log.warning(LOGGER_OBJECT_DETECTOR_KILL_SWITCH,
                        'Killing process. - to be implemented.')
        except:
            pass

    def __adjustConfigDict(self, confDict):
        '''
        adjustConfigDict :
        param:confDict:parameters read in the config file for the robot
        param:confDict:type:parameter dictionary
        returns : modified confDict
        '''
        for key, value in confDict.items():
            if not isinstance(value, dict):
                if not isinstance(value, list):
                    if value in K4A_DEFINITIONS:
                        confDict[key] = K4A_DEFINITIONS[value]
            else:
                confDict[key] = self.__adjustConfigDict(confDict[key])
        return confDict


def sort_tracked_objects(detection_response: str,
                         image_shape: tuple,
                         resized_im,
                         detection_threshold: float,
                         default_tracker: str,
                         tracked_objects):
    """
    Description:
        Function run on a separate process to sort tracked objects
    Args:
        detection_Response : string containing the json response of the scored model.
            Includes bounding boxes, scores and classes of scored items.
        image_shape : tuple (height, width) in pixels
        resized_im: numpy array of resized image
        detection_threshold : float, value 0<x<1 to accept detection as an object
        default_tracker: string, representing the constant.py file for opencv
        tracked_objects: list of objects currently being tracked by the system
    Returns:
        temp_tracked_object : object's list of tracked objects.
    """
    height, width = image_shape
    detection_boxes = detection_response['boxes']
    detection_scores = detection_response['scores']
    detection_classes = detection_response['classes']
    # Find #boxes with score > thresh
    nb_bb = np.sum(np.array(detection_scores[0]) > detection_threshold)
    # Create a list of BoundingBoxes > thresh
    bb_list = [BoundingBox(box=box,
                           image_height=height,
                           image_width=width,
                           fmt=FMT_TF_BBOX)
               for box in detection_boxes[0][:nb_bb]]
    ds_list = [score for score in detection_scores[0][:nb_bb]]
    dc_list = [cla for cla in detection_classes[0][:nb_bb]]
    # Find best fitting BB for each tracked object
    temp_tracked_objects = []
    for tracked_object in tracked_objects:
        # Find best overlapping bounding box
        target_bb_idx = tracked_object.get_max_overlap_bb(bb_list)
        # Update dictionnary with outcome unless target_bb == None
        if target_bb_idx is not None:
            tracked_object.update(
                image=np.asarray(resized_im),
                box=bb_list[target_bb_idx].get_bbox(),
                fmt=FMT_TRACKER)
            temp_tracked_objects.append(tracked_object)
            bb_list.remove(bb_list[target_bb_idx])
            ds_list.remove(ds_list[target_bb_idx])
            dc_list.remove(dc_list[target_bb_idx])
        else:
            # Add object to temp list if seen in last 5 seconds
            if time.time() - tracked_object.last_seen < 5:
                temp_tracked_objects.append(
                    tracked_object)
            else:
                log.warning(LOGGER_OBJECT_DETECTOR_SORT_TRACKED_OBJECTS,
                            msg=f'Deleted tracker (id={tracked_object.id})')
    # List all unused bounding boxes and create tracker
    for idx, bb in enumerate(bb_list):
        new_obj = TrackedObject(
            object_class=dc_list[idx],
            score=ds_list[idx],
            image=np.asarray(resized_im),
            original_image_resolution=(height, width),
            box=bb.get_bbox(fmt=FMT_STANDARD),
            fmt=FMT_STANDARD,
            tracker_alg=default_tracker)
        temp_tracked_objects.append(new_obj)
        log.warning(LOGGER_OBJECT_DETECTOR_SORT_TRACKED_OBJECTS,
                    msg=f'Created tracker (id={new_obj.id})')
    # Copy temp list back to object
    return temp_tracked_objects

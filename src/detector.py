import json
import time
import queue
import traceback
import threading
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from publish_queues import PublishQueue
import os
import multiprocessing as mp
from multiprocessing import Process
import numpy as np
import paho.mqtt.client as mqtt
import io
from PIL import Image
import base64
import cv2
import asyncio
from aiohttp import ClientSession
from aiohttp.client_exceptions import ServerDisconnectedError
from pyk4a import PyK4A, K4AException, FPS
from pyk4a import Config as k4aConf
from constant import K4A_DEFINITIONS
from utils import label_map_util
from utils import visualization_utils as vis_util
from constant import LOGGER_OBJECT_DETECTOR_MAIN, \
    OBJECT_DETECTOR_CONFIG_DICT, LOGGER_OBJECT_DETECTOR_MQTT_LOOP, \
    LOGGER_OBJECT_DETECTOR_RUNNER, LOGGER_OBJECT_DETECTOR_ASYNC_LOOP, \
    LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT, \
    LOGGER_OBJECT_DETECTOR_KILL_SWITCH, LOGGER_ASYNC_RUN_DETECTION, \
    LOGGER_ASYNC_RUN_CAPTURE_LOOP, LOGGER_OBJECT_DETECTOR_SORT_TRACKED_OBJECTS, \
    OPENCV_OBJECT_TRACKERS, \
    LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT, \
    LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER, \
    LOGGER_OBJECT_DETECTION_THREAD_POLL_OBJECT_TRACKING_PROCESS_QUEUE
from logger import RoboLogger
from tracked_object import BoundingBox, TrackedObjectMP, \
    FMT_STANDARD, FMT_TF_BBOX, FMT_TRACKER
log = RoboLogger.getLogger()
log.warning(LOGGER_OBJECT_DETECTOR_MAIN, msg="Libraries loaded.")


class ObjectDetector(object):
    """ main class for the object detector running on the jetson """

    def __init__(self, configuration: dict) -> None:
        """camera_index: int, frozen_graph_path: str,
        label_map_path: str, num_classes: int) -> None:"""
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
            raise Exception('Unsupported frame rate {}'.format(
                            k4a_config_dict['camera_fps']))
        self.detection_threshold = 0.5
        self.resized_image_resolution = tuple(configuration[OBJECT_DETECTOR_CONFIG_DICT]['resized_resolution'])
        self.tracked_objects_mp = {}
        self.tracked_objects_queue = mp.Queue(maxsize=1)
        self.tracked_objects_queue_output = mp.Queue(maxsize=1)
        self.image_queue = PublishQueue()
        self.default_tracker = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['open_cv_default_tracker']
        self.max_unseen_time_for_object = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['max_unseen_time_for_object']
        self.max_tracked_objects = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['max_tracked_objects']
        self.time_between_scoring_service_calls = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['time_between_scoring_service_calls']

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

            # Create thread pool (for retrieving object tracking information)
            # log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
            #             msg=f'Creating thread pool executor for '
            #                 f'{self.max_tracked_objects} objects(s)')
            # self.object_tracking_thread_pool = ThreadPoolExecutor(
            #     max_workers=self.max_tracked_objects,
            #     thread_name_prefix='ObjTracking')

            # region Launch Processes
            # Create process to process image and tracked objects
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg=f'Creating Process to sort tracked objects')
            self.procTrackedObject = Process(
                target=process_sort_tracked_objects,
                name='ObjectTrackingSorter',
                args=(self.tracked_objects_queue,
                      self.tracked_objects_queue_output,
                      self.max_tracked_objects,))
            self.procTrackedObject.start()
            # endregion

            # region Launch Threads
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
            self.capture_loop_thread = threading.Thread(
                target=self.thread_capture_loop, name='captureLoop')
            self.capture_loop_thread.start()

            # Launching Manage Object Detection Pools
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg=f'Launching Object Detection Pool and Thread Managers')
            self.object_detection_pool_manager_thread = threading.Thread(
                target=self.thread_object_detection_pool_manager, name='objectDetectionManager',
                args=[0.25])
            self.object_detection_pool_manager_thread.start()

            # endregion

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
            raise Exception(f'Error : {traceback.print_exc()}')

        except K4AException:
            log.error(LOGGER_OBJECT_DETECTOR_RUNNER,
                      f'Error with K4A : {traceback.print_exc()}')

        except Exception:
            log.error(LOGGER_OBJECT_DETECTOR_RUNNER,
                      f'Error : {traceback.print_exc()}')

    def thread_event_loop(self):
        """
        Main event asyncio eventloop launched in a separate thread
        """
        try:
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                        msg=f'Launching asyncio event loop')
            self.eventLoop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.eventLoop)

            # region Create Async Tasks
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                        msg=f'Launching process MQTT message task')
            self.eventLoop.create_task(
                self.async_process_mqtt_messages(loopDelay=0.25))

            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                        msg=f'Launching asykc_run_detection task')
            self.eventLoop.create_task(
                self.async_run_detection(
                    loopDelay=self.time_between_scoring_service_calls))
            # endregion

            self.eventLoop.run_forever()
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                        msg=f'Asyncio event loop started')

        except Exception:
            log.error(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                      f'Error : {traceback.print_exc()}')
            raise Exception(f'Error : {traceback.print_exc()}')
        finally:
            self.eventLoop.stop()
            self.eventLoop.close()

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
            raise Exception(f'Error : {traceback.print_exc()}')

    async def async_process_mqtt_messages(self,
                                          loopDelay=0.25):
        """
        Description : This function receives the messages from MQTT to run
            various functions on the Jetson
        Args:
            loopDelay: float, delay to sleep at the end of the loop
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

    def thread_capture_loop(self) -> None:
        """
        Video capture loop (can optionally display video) for debugging purposes
        """
        try:
            # Launch the loop
            n_loops = 0
            logging_loops = 50
            average_duration = 0
            average_track_object_time_duration = 0
            average_perftimer_image_resize_time_duration = 0
            average_peftimer_image_copy_time_duration = 0
            perftimer_image_acquire_duration = 0
            k4a_errors = 0
            while True:
                start_time = time.time()
                # Read frame from camera
                try:
                    bgra_image_color_np, image_depth_np = \
                        self.k4a_device.get_capture(
                            color_only=False,
                            transform_depth_to_color=True)
                except K4AException:
                    k4a_errors += 1         # count problematic frame capture
                # log.debug(LOGGER_ASYNC_RUN_CAPTURE_LOOP, msg=f'Ran capture (loop {n_loops})')
                perftimer_image_acquire = time.time() - start_time
                perftimer_image_acquire_duration += perftimer_image_acquire

                # IMAGE COPY SECTION
                peftimer_image_copy_time_start = time.time()
                self.rgb_image_color_np = bgra_image_color_np[:, :, :3][..., ::-1]     # .copy()   # maybe remove copy? ONLY COPY IN THE TF LOOP to LOWER CPU USAGE
                peftimer_image_copy_time_duration = time.time() - peftimer_image_copy_time_start
                average_peftimer_image_copy_time_duration += peftimer_image_copy_time_duration

                # RESIZE IMAGE SECTION
                perftimer_image_resize_time_start = time.time()
                self.rgb_image_color_np_resized = np.asarray(
                    Image.fromarray(self.rgb_image_color_np).resize(
                        self.resized_image_resolution))
                self.image_queue.publish(self.rgb_image_color_np_resized)
                perftimer_image_resize_time_duration = time.time() - perftimer_image_resize_time_start
                average_perftimer_image_resize_time_duration += perftimer_image_resize_time_duration
                # only do this after the first loop is done
                if n_loops == 0:
                    self.ready_queue.put_nowait('image_ready')

                # Show video in debugging mode - move to other thread (that we can start on mqtt message...)
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
                # sleep_time = max(0, self.frame_duration - duration)
                average_duration += duration
                # await asyncio.sleep(sleep_time)
                # time.sleep(sleep_time)
                n_loops += 1
                if n_loops % logging_loops == 0:
                    duration_50 = average_duration
                    average_duration /= 50
                    average_track_object_time_duration /= 50
                    average_perftimer_image_resize_time_duration /= 50
                    average_peftimer_image_copy_time_duration /= 50
                    perftimer_image_acquire_duration /= 50
                    log.warning(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                                msg=f'Ran 50 in {duration_50:.2f}s - {average_duration:.2f}s/loop or {1/average_duration:.2f} loop/sec')
                    log.warning(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                                msg=f'Currently monitoring {len(self.tracked_objects_mp)} objects')
                    log.debug(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                              msg=f'Avg Image Acquire Time = {perftimer_image_acquire_duration:.4f}s')
                    log.debug(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                              msg=f'Avg Image Copy Time = {average_peftimer_image_copy_time_duration:.4f}s')
                    log.debug(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                              msg=f'Avg Image Resize Time = {average_perftimer_image_resize_time_duration:.4f}s')
                    log.debug(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                              msg=f'Avg Tracker Update Time = {average_track_object_time_duration:.4f}s')
                    average_duration = 0
                    average_track_object_time_duration = 0
                    average_perftimer_image_resize_time_duration = 0
                    average_peftimer_image_copy_time_duration = 0
                    perftimer_image_acquire_duration = 0
        except Exception:
            log.error(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                      f'Error : {traceback.print_exc()}')
            raise Exception(f'Error : {traceback.print_exc()}')

    async def async_run_detection(self,
                                  loopDelay=0) -> None:
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
                raise Exception('Empty Queue...')
            # Launch the loop
            while True:
                height, width = self.rgb_image_color_np_resized.shape[:2]
                # Send image to bytes buffer
                im = Image.fromarray(self.rgb_image_color_np_resized)
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
                            task_start_time = time.time()
                            self.tracked_objects_queue.put([
                                body,
                                (height, width),
                                self.detection_threshold,
                                self.tracked_objects_mp],
                                block=True)
                            # BLOCK WAIT for other PROCESS (try to wait UNBLOCKED...)
                            self.tracked_objects_mp = \
                                self.tracked_objects_queue_output.get(
                                    block=True)
                            task_duration += time.time() - task_start_time
                            loop_time += (end_time - start_time)
                            n_loops += 1
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
                        else:
                            raise Exception(f'HTTP response code is '
                                            f'{response.status} for detection'
                                            f' service...')
                # Pause for loopDelay seconds
                await asyncio.sleep(loopDelay)
        except ServerDisconnectedError:
            log.error(
                LOGGER_ASYNC_RUN_DETECTION,
                msg=f'HTTP prediction service error: {traceback.print_exc()} '
                    f'-> sleeping {loopDelay}s and continuing')
            await asyncio.sleep(loopDelay)
        except Exception:
            log.error(LOGGER_ASYNC_RUN_DETECTION,
                      f'Error : {traceback.print_exc()}')
            raise Exception(f'Error : {traceback.print_exc()}')

    def thread_object_detection_pool_manager(self,
                                             loop_delay=0.25):
        """
        Description
            Maintains a list of processes and threads per object that's being
                monitored by the robot.
                During its processing, _mapping_object_process_thread would
                look like this:
                    {
                        object_uuid_1 : [process_id, thread_id, ],
                        object_uuid_2 : [process_id, thread_id, ]
                    }
        Args:
            loop_delay : float : delay in the while true loop to ensure no high
                cpu usage for nothing.
        """
        self._mapping_object_process_thread = {}
        while True:
            try:
                temp_tracked_objects_mp = deepcopy(self.tracked_objects_mp)
                for (tracked_object_mp_id, tracked_object_mp) in \
                        temp_tracked_objects_mp.items():
                    # If it's not already monitored the monitor it
                    if tracked_object_mp_id not in self._mapping_object_process_thread.keys():
                        tracking_object_queue = mp.Manager().Queue(maxsize=1)       # tried adding a Manager...
                        kill_queue = mp.Manager().Queue(maxsize=1)
                        kill_thread = False
                        proc = Process(target=process_track_opencv_object,
                                       name='ObjectTracker',
                                       args=(tracked_object_mp,
                                             self.default_tracker,
                                             self.rgb_image_color_np_resized,
                                             self.frame_duration,
                                             self.image_queue.register(name=str(tracked_object_mp_id)),
                                             tracking_object_queue,
                                             kill_queue,))
                        proc.start()
                        log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                    msg=f'Launched proces {proc.pid} for object {tracked_object_mp_id}.')
                        thread = threading.Thread(
                            target=self.thread_poll_object_tracking_process_queue,
                            args=([tracked_object_mp,
                                   tracking_object_queue,
                                   False]),
                            name=f'ObjTrack_{str(tracked_object_mp_id)[:8]}')
                        thread.start()

                        # future_thread = self.object_tracking_thread_pool.submit(
                        #     self.thread_poll_object_tracking_process_queue,
                        #     tracked_object_mp,
                        #     tracking_object_queue,
                        #     kill_queue)
                        # # Keep track of what is under observation
                        self._mapping_object_process_thread = {
                            tracked_object_mp_id:
                                {
                                    'tracking_object_queue': tracking_object_queue,
                                    'proc': proc,
                                    'thread': thread,
                                    'kill_queue': kill_queue,
                                    'kill_thread': kill_thread
                                }
                        }
                        # self._mapping_object_process_thread[tracked_object_mp_id]['tracking_object_queue'] = tracking_object_queue
                        # self._mapping_object_process_thread[tracked_object_mp_id]['proc'] = proc
                        # self._mapping_object_process_thread[tracked_object_mp_id]['thread'] = thread
                        # self._mapping_object_process_thread[tracked_object_mp_id]['kill_queue'] = kill_queue
                        # self._mapping_object_process_thread[tracked_object_mp_id]['kill_thread'] = kill_thread
                    # Check if object related process/thread should be purged
                    # (unseen for 5 seconds)
                    if (time.time() - tracked_object_mp.last_seen) > self.max_unseen_time_for_object:
                        obj_queue = self._mapping_object_process_thread[tracked_object_mp_id]['tracking_object_queue']
                        proc = self._mapping_object_process_thread[tracked_object_mp_id]['proc']
                        thread = self._mapping_object_process_thread[tracked_object_mp_id]['thread']
                        kill_queue = self._mapping_object_process_thread[tracked_object_mp_id]['kill_queue']
                        kill_thread = self._mapping_object_process_thread[tracked_object_mp_id]['kill_thread']
                        # empty queue before killing proc
                        while not obj_queue.empty():
                            _ = obj_queue.get()
                        self.image_queue.unregister(name=str(tracked_object_mp_id))
                        if proc.is_alive():
                            log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                        msg=f'Attempting to cancel process for object {tracked_object_mp_id}.')
                            kill_queue.put(True)
                            exit_code = proc.join(1)
                            log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                        msg=f'Exit code = {exit_code} for tracked object {tracked_object_mp_id}.')
                        else:         # proc done
                            log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                        msg=f'Process for object {tracked_object_mp_id} already done.')

                        self._mapping_object_process_thread[tracked_object_mp_id]['kill_thread'] = True
                        # kill_queue.put(True)
                        thread.join(1)  # Wait 1 sec
                        if thread.is_alive():   # .running():
                            log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                        msg=f'Failed to cancel thread for object {tracked_object_mp_id}.')
                        else:
                            log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                        msg=f'Thread for object {tracked_object_mp_id} is succesfully terminated.')
                        # Remove from dict altogether
                        log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                    msg=f'Deleting object {tracked_object_mp_id} from mapping dictionnary.')
                        self._mapping_object_process_thread.pop(tracked_object_mp_id)
                        # remove from self.tracked
                        log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                    msg=f'Deleting object {tracked_object_mp_id} from list of tracked objects.')
                        # Remote item from dictionnary
                        self.tracked_objects_mp.pop(tracked_object_mp_id)
            except KeyError:
                raise Exception(f'Problem : tried removing tracked object from '
                                f'dictionnary but key not found. Traceback = '
                                f'{traceback.print_exc()}')
            except:
                raise Exception(f'Problem : {traceback.print_exc()}')
            time.sleep(loop_delay)

    def thread_poll_object_tracking_process_queue(self,
                                                  tracked_object_mp: TrackedObjectMP,
                                                  tracked_object_queue: mp.Queue,
                                                  kill=False):
        """
        Description: thread to monitor objects and retrieve updated bounding box
            coordinates - these are calculated in the associated process.
        Args:
            tracked_object_mp : <class TrackedObjectMP()>, object that this
                thread is tracking the bounding box for
            tracking_object_queue : <class 'multiprocessing.Queue()'> :
                containing the object that has the latest bounding box
            kill: bool : queue to track
                requirement to shut down process
        """
        log.warning(LOGGER_OBJECT_DETECTION_THREAD_POLL_OBJECT_TRACKING_PROCESS_QUEUE,
                    msg=f'Launching thread for object ID {tracked_object_mp.id}')
        while not kill:
            try:
                # Replace object in tracked object with what we get from the queue
                # wait max 2 second
                kill = self._mapping_object_process_thread[tracked_object_mp.id]['kill_thread']
                if kill:
                    log.warning(LOGGER_OBJECT_DETECTION_THREAD_POLL_OBJECT_TRACKING_PROCESS_QUEUE,
                                msg=f'Kill signal received for thread handling '
                                    f'object {tracked_object_mp.id}')
                    break
                new_tracked_object_mp = tracked_object_queue.get(block=True, timeout=2)
                # Get the new coordinates in the object.
                self.tracked_objects_mp[new_tracked_object_mp.id] = new_tracked_object_mp
            except queue.Empty:
                log.warning(LOGGER_OBJECT_DETECTION_THREAD_POLL_OBJECT_TRACKING_PROCESS_QUEUE,
                            msg=f'No new object was placed on the queue, process '
                                f'may have been terminated by parent process.')
            except:
                raise Exception(f'Problem: {traceback.print_exc()}')

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


def process_sort_tracked_objects(inbound_queue: mp.Queue,
                                 outbound_queue: mp.Queue,
                                 max_tracked_objects: int):
    """
    Description:
        Function run on a separate process to sort tracked objects
    Args:
        inbound_queue : <class 'multiprocessing.Queue()> : contains
            information to process new images. On the queue, you will find at
            every iteration the following arguments on a tuple:
                detection_Response : string containing the json response of
                    the scored model. Includes bounding boxes, scores and
                    classes of scored items.
                image_shape : tuple (height, width) in pixels
                detection_threshold : float, value 0<x<1 to accept detection
                    as an object
                tracked_objects_mp: [trackedObject.id (uuid), <class TrackedObjectMP>] currently being
                    tracked by the system
        outbound_queue: <class 'multiprocessing.Queue()> : contains return values :
            temp_tracked_objects_mp : [<class TrackedObjectMP>] for tracked objects as a result of
                the comparison with object tracsking versus dynamic scoring of model
        max_tracked_objects : int, max number of simulaneously tracked objects
    Returns: None, returns through multiprocessing.Queue()
    """
    log.warning(LOGGER_OBJECT_DETECTOR_SORT_TRACKED_OBJECTS,
                msg=f'Starting Tracker Sorting Process : PID = {os.getpid()}')
    logging_loops = 50
    n_loops = 0
    while True:
        try:
            detection_response, image_shape, \
                detection_threshold, \
                tracked_objects_mp = inbound_queue.get(block=True)
            height, width = image_shape
            detection_boxes = detection_response['boxes']
            detection_scores = detection_response['scores']
            detection_classes = detection_response['classes']
            # Find #boxes with score > thresh or max tracked objects
            nb_bb = min(
                max_tracked_objects,
                np.sum(np.array(detection_scores[0]) > detection_threshold))
            # Create a list of BoundingBoxes > thresh
            bb_list = [BoundingBox(box=box,
                                   image_height=height,
                                   image_width=width,
                                   fmt=FMT_TF_BBOX)
                       for box in detection_boxes[0][:nb_bb]]
            ds_list = [score for score in detection_scores[0][:nb_bb]]
            dc_list = [cla for cla in detection_classes[0][:nb_bb]]
            # Find best fitting BB for each tracked object
            temp_existing_tracked_objects_mp = {}
            for (uuid, tracked_object) in tracked_objects_mp.items():
                # Find best overlapping bounding box
                target_bb_idx = tracked_object.get_max_overlap_bb(bb_list)
                # Update dictionnary with outcome unless target_bb == None
                if target_bb_idx is not None:
                    tracked_object.update_bounding_box(
                        bb_list[target_bb_idx],
                        fmt=FMT_TRACKER)
                    bb_list.remove(bb_list[target_bb_idx])
                    ds_list.remove(ds_list[target_bb_idx])
                    dc_list.remove(dc_list[target_bb_idx])
                # add object tuple to list
                temp_existing_tracked_objects_mp[tracked_object.id] = tracked_object
            temp_new_tracked_objects_mp = {}
            # List all unused bounding boxes and create tracker
            for idx, bb in enumerate(bb_list):
                new_obj = TrackedObjectMP(
                    object_class=dc_list[idx],
                    score=ds_list[idx],
                    original_image_resolution=(height, width),
                    box=bb.get_bbox(fmt=FMT_STANDARD),
                    fmt=FMT_STANDARD)
                temp_new_tracked_objects_mp[new_obj.id] = new_obj
                log.warning(LOGGER_OBJECT_DETECTOR_SORT_TRACKED_OBJECTS,
                            msg=f'Created temp tracker (id={new_obj.id})')

            # Combine output and get only up to max_tracked_objects items
            combined_temp = {**temp_new_tracked_objects_mp,
                             **temp_existing_tracked_objects_mp}
            temp_tracked_objects_mp = {k: combined_temp[k]
                                       for k in list(combined_temp)
                                       [:max_tracked_objects]}
            # Copy temp list back to object
            outbound_queue.put(temp_tracked_objects_mp, block=True)
            if n_loops % logging_loops == 0:
                log.warning(
                    LOGGER_OBJECT_DETECTOR_SORT_TRACKED_OBJECTS,
                    msg=f'Process is monitoring '
                        f'{len(temp_tracked_objects_mp)} objects '
                        f'after {n_loops} loops.')
            n_loops += 1
        except:
            raise Exception(f'Problem : {traceback.print_exc()}')


def process_track_opencv_object(tracked_object: TrackedObjectMP,
                                tracker_alg: str,
                                initial_image: np.array,
                                frame_duration: float,
                                image_queue: PublishQueue,
                                tracking_object_queue: mp.Queue,
                                kill_queue: mp.Queue):
    """
    Description:
        Function runs a process in the executorProcessPool to keep track of
            an object using opencv's trackers.
    Args:
        tracked_object: <class 'TrackedObjectMP()'>, object tracked by process
        tracker_alg : string, one of the values in OPENCV_OBJECT_TRACKERS
        initial_image : numpy array containing the resized image that is used to
            initialize the tracker
        frame_duration : float, corresponds to the 1/FPS of input camera
        image_queue: <class 'multiprocessing.Queue()'> : contains images
            that are pushed on it in numpy format when a new image was read
            from the video capture
        tracking_object_queue: <class 'multiprocessing.Queue()'> : queue to
            publish the updated tracked_object_mp (incl. new bbox)
        kill_queue: <class 'multiprocessing.Queue()'> : queue to track
            requirement to shut down process
    """
    kill = False
    log.warning(LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT,
                f'Launching process {os.getpid()} for object ID '
                f'{tracked_object.id}')
    try:
        tracker = OPENCV_OBJECT_TRACKERS[tracker_alg]()
        tracker.init(initial_image, tracked_object.get_bbox())
        logging_loops = 50
        n_loops = 0
        while not kill:
            start_time = time.time()
            image = image_queue.get(block=True)
            if image is None:
                log.warning(LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT,
                            msg=f'Unable to retrieve image in queue for '
                                f'process for object ID {tracked_object.id}')
                raise Exception(f'Unable to retrieve image in queue for '
                                f'process for object ID {tracked_object.id}')
            height, width = image.shape[:2]
            # Costly operation - update opencv tracker information
            (success, bbox) = tracker.update(image)
            if success:
                tracked_object.update_bounding_box(
                    BoundingBox(
                        box=bbox,
                        image_height=height,
                        image_width=width,
                        fmt='tracker',
                        use_normalized_coordinates=False),
                    fmt=FMT_TRACKER)
                tracked_object.last_seen = time.time()
            else:
                log.warning(
                    LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT,
                    msg=f'Object tracking did not find object '
                        f'{tracked_object.id}')
            # Dumping tracked_object to queue
            tracking_object_queue.put_nowait(tracked_object)
            # Sleep if required
            loop_duration = time.time() - start_time
            time.sleep(max(0, frame_duration - loop_duration))
            try:
                # will pull "True" if it was placed on the queue.
                kill = kill_queue.get_nowait()
                log.warning(
                    LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT,
                    msg=f'Kill signal = {kill} received for process handling '
                        f'object {tracked_object.id}')
            except queue.Empty:
                kill = False
            if n_loops % logging_loops == 0:
                log.warning(
                    LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT,
                    msg=f'Process is stil tracking object {tracked_object.id} '
                        f'after {n_loops} loops - object class = '
                        f'{tracked_object.object_class}.')
            n_loops += 1
    except:
        raise Exception(f'Problem in process_track_opencv_object: '
                        f'{traceback.print_exc()}')
    finally:
        log.warning(LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT,
                    msg=f'Exiting obj tracking process (PID {os.getpid()}) '
                        f'for object {tracked_object.id}')

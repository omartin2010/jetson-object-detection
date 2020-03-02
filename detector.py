import queue
import traceback
import threading
import paho.mqtt.client as mqtt
import numpy as np
import tensorflow as tf
import cv2
import json
from pyk4a import PyK4A, K4AException    # , K4ATimeoutException
from pyk4a import Config as k4aConf
import time
import asyncio
from constant import K4A_DEFINITIONS
from utils import label_map_util
from utils import visualization_utils as vis_util
from constant import LOGGER_OBJECT_DETECTOR_MAIN, LOGGER_OBJECT_DETECTOR_LOAD_MODEL, \
    OBJECT_DETECTOR_CONFIG_DICT, LOGGER_OBJECT_DETECTOR_MQTT_LOOP, LOGGER_OBJECT_DETECTOR_RUNNER, \
    LOGGER_OBJECT_DETECTOR_ASYNC_LOOP, LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT, \
    LOGGER_OBJECT_DETECTOR_KILL_SWITCH, LOGGER_OBJECT_DETECTOR_RUN_DETECTION
from logger import RoboLogger
log = RoboLogger.getLogger()
log.warning(LOGGER_OBJECT_DETECTOR_MAIN,
            msg="About to initialize libraries (tensorflow + opencv)")
log.warning(LOGGER_OBJECT_DETECTOR_MAIN, msg="Libraries loaded.")


class ObjectDetector(object):
    """ main class for the object detector running on the jetson """

    def __init__(self, configuration: dict) -> None:
        """camera_index: int, frozen_graph_path: str, label_map_path: str, num_classes: int) -> None:"""
        self.configuration = configuration
        # Adjust configuration (to make strings into constants)
        self.__adjustConfigDict(self.configuration)
        self._camera_index = configuration[OBJECT_DETECTOR_CONFIG_DICT]['camera_index']
        self.frozen_graph_path = configuration[OBJECT_DETECTOR_CONFIG_DICT]['frozen_graph_path']
        self.num_classes = configuration[OBJECT_DETECTOR_CONFIG_DICT]['num_classes']
        self.label_map_path = configuration[OBJECT_DETECTOR_CONFIG_DICT]['label_map_path']
        self.mqttMessageQueue = queue.Queue()
        self.exceptionQueue = queue.Queue()
        k4a_config_dict = self.configuration['object_detector']['k4a_device']
        self.k4a_device = PyK4A(
            k4aConf(color_resolution=k4a_config_dict['color_resolution'],
                    depth_mode=k4a_config_dict['depth_mode'],
                    camera_fps=k4a_config_dict['camera_fps'],
                    synchronized_images_only=k4a_config_dict['synchronized_images_only'],
                    color_format=k4a_config_dict["color_format"]))
        self._readyForInferencing = False
        # defaults to false, put to true for debugging (need video display)
        self.show_video = False

    def run(self) -> None:
        """
        params:
        Launches the runner that runs most things (MQTT queue, etc.)
        """
        try:
            # Initialize the Kinect Device
            self.k4a_device.connect()
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg=f'K4A device initialized...')
            # Launch the MQTT thread listener
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg='Launching MQTT thread.')
            self.threadMQTT = threading.Thread(
                target=self.threadImplMQTT, name='MQTTListener')
            self.threadMQTT.start()

            # Launch main event loop in a separate thread
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg='Launching event loop')
            self.eventLoopThread = threading.Thread(
                target=self.threadImplEventLoop, name='asyncioEventLoop')
            self.eventLoopThread.start()

            # Launch detector in a separate thread
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg='Launching Detector Thread')
            self.detectorThread = threading.Thread(
                target=self.runDetection(
                    loopDelay=0.10), name='ObjectDetector')
            self.detectorThread.start()

            while True:
                try:
                    exc = self.exceptionQueue.get(block=True)
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

    def threadImplMQTT(self):
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
            self.mqttMessageQueue.put_nowait(message)

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

    def load_model(self) -> None:
        # log.warning(LOGGER_OBJECT_DETECTOR_LOAD_MODEL,
        # f'Connecting to camera {self._camera_index}')
        # self.cap = cv2.VideoCapture(self._camera_index)
        # self.cap = k4a.Capture()
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.frozen_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                log.warning(LOGGER_OBJECT_DETECTOR_LOAD_MODEL,
                            msg=f'Loading model in memory...')
                tf.import_graph_def(od_graph_def, name='')
        log.warning(LOGGER_OBJECT_DETECTOR_LOAD_MODEL,
                    msg='Model loaded in memory.')
        # Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`,
        # we know that this corresponds to `airplane`.  Here we use internal utility functions,
        # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(self.label_map_path)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.num_classes,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self._readyForInferencing = True

    def _get_tensors(self):
        """
        Helper function to getting tensors for object detection API
        """
        # Extract image tensor
        image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')
        # Extract detection boxes
        boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        # Extract detection scores
        scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0')
        # Extract detection classes
        classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        # Extract number of detections
        num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')
        return image_tensor, boxes, scores, classes, num_detections

    def threadImplEventLoop(self):
        """
        Main event asyncio eventloop launched in a separate thread
        """
        try:
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP, msg=f'Launching asyncio event loop')
            self.eventLoop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.eventLoop)
            self.eventLoop.create_task(self.asyncProcessMQTTMessages(loopDelay=0.25))
            self.eventLoop.run_forever()
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP, msg=f'Asyncio event loop started')

        except Exception:
            log.error(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                      f'Error : {traceback.print_exc()}')
        finally:
            self.eventLoop.stop()
            self.eventLoop.close()

    def runDetection(self, loopDelay=0.10) -> None:
        """
        returns object detection
        params :
        delay:float:delay to pause between frames scoring
        returns:task
        """
        self.load_model()

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                log.warning(LOGGER_OBJECT_DETECTOR_RUN_DETECTION,
                            msg='In context for detector.')
                logging_loops = 50
                loop_time = 0
                n_loops = 0
                # Launch the loop
                # if k4a.device_start_cameras(self.k4a_device, self.k4a_device_config):
                while True:
                    image_tensor, boxes, scores, classes, num_detections = self._get_tensors()
                    # Read frame from camera
                    image_color_np, image_depth_np = self.k4a_device.get_capture(color_only=False)
                    # Expand dimensions since the model expects images
                    # to have shape: [1, None, None, 3] for TF model
                    image_np_expanded = np.expand_dims(
                        image_color_np[:, :, :3], axis=0)
                    # Actual detection
                    start_time = time.time()
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    end_time = time.time()
                    loop_time += (end_time - start_time)
                    n_loops += 1
                    if n_loops % logging_loops == 0:
                        loop_time /= logging_loops
                        log.debug(LOGGER_OBJECT_DETECTOR_RUN_DETECTION,
                                  msg=f'Average loop for the past {logging_loops} iteration is {loop_time:.3f}s')
                        loop_time = 0
                    time.sleep(loopDelay)
                    # Show on screen if required
                    if self.show_video:
                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_color_np,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            self.category_index,
                            use_normalized_coordinates=True,
                            line_thickness=4)
                        # Display output
                        cv2.imshow('object detection',
                                   cv2.resize(image_color_np, (1024, 576)))
                        cv2.imshow('depth view',
                                   cv2.resize(image_depth_np, (1024, 576)), cmap='gray')
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            # cv2.destroyAllWindows()
                            cv2.destroyWindow('object detection')
                            cv2.destroyWindow('depth view')
                            self.show_video = False
                            # break

    async def asyncProcessMQTTMessages(self, loopDelay=0.25):
        """
        This function receives the messages from MQTT to run various functions on the Jetson
        """
        log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                    msg='Launching MQTT processing async task')
        while True:
            try:
                if self.mqttMessageQueue.empty() is False:
                    # Remove the first in the list, will pause until there is something
                    currentMQTTMoveMessage = self.mqttMessageQueue.get()
                    # Decode message received
                    msgdict = json.loads(
                        currentMQTTMoveMessage.payload.decode('utf-8'))

                    # Check if need to shut down
                    if currentMQTTMoveMessage.topic == 'bot/killSwitch':
                        log.warning(
                            LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT, msg='Kill switch activated')
                        self.killSwitch()

                    elif currentMQTTMoveMessage.topic == 'bot/jetson/detectSingleObject':
                        log.info(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                                 msg='Invoking object detection')
                        thresh = msgdict['threshold']
                        start_time = time.time()
                        boxes, scores, classes, num_detections = self.run_detection(
                            show_video=False, loop=False)
                        stop_time = time.time()
                        num_relevant_detections = (scores > thresh).sum()
                        boxes = boxes[0][:num_relevant_detections]
                        scores = scores[0][:num_relevant_detections]
                        classes = classes[0][:num_relevant_detections]
                        log.info(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                                 msg=f'Detector found {num_relevant_detections} objects in {(stop_time - start_time):0.2f} second.')
                        for i in range(num_relevant_detections):
                            class_name = self.category_index[int(
                                classes[i])]['name']
                            log.debug(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                                      msg=f'Object {i} : {class_name}, class_id={int(classes[i])}; Score: {scores[i]:0.4f} Box: {boxes[i]}')

                    elif currentMQTTMoveMessage.topic == 'bot/jetson/detectObjectDisplayVideo':
                        log.info(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                                 msg='Showing Video on screen (NEED A DISPLAY CONNECTED!)')
                        if 'show_video' in msgdict:
                            show_video = msgdict['show_video']
                        else:
                            show_video = True
                        self.show_video = show_video

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

    def killSwitch(self):
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

from utils import visualization_utils as vis_util
from utils import label_map_util
import asyncio
import json
import concurrent.futures
import cv2
import tensorflow as tf
import numpy as np
import paho.mqtt.client as mqtt
import threading
import traceback
import queue
from logger import RoboLogger
from constant import LOGGER_OBJECT_DETECTOR_MAIN, LOGGER_OBJECT_DETECTOR_LOAD_MODEL, \
    OBJECT_DETECTOR_CONFIG_DICT, LOGGER_OBJECT_DETECTOR_MQTT_LOOP, LOGGER_OBJECT_DETECTOR_RUNNER, \
    LOGGER_OBJECT_DETECTOR_ASYNC_LOOP, MAX_WORKER_THREADS, LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT, \
    LOGGER_OBJECT_DETECTOR_KILL_SWITCH
log = RoboLogger.getLogger()
log.warning(LOGGER_OBJECT_DETECTOR_MAIN,
            msg="About to initialize libraries (tensorflow + opencv)")
log.warning(LOGGER_OBJECT_DETECTOR_MAIN, msg="Libraries loaded.")


class ObjectDetector(object):
    """ main class for the object detector running on the jetson """

    def __init__(self, configuration: dict) -> None:
        """camera_index: int, frozen_graph_path: str, label_map_path: str, num_classes: int) -> None:"""
        self.configuration = configuration
        self._camera_index = configuration[OBJECT_DETECTOR_CONFIG_DICT]['camera_index']
        self.frozen_graph_path = configuration[OBJECT_DETECTOR_CONFIG_DICT]['frozen_graph_path']
        self.num_classes = configuration[OBJECT_DETECTOR_CONFIG_DICT]['num_classes']
        self.label_map_path = configuration[OBJECT_DETECTOR_CONFIG_DICT]['label_map_path']
        self.mqttMessageQueue = queue.Queue()
        self._readyForInferencing = False

    def run(self) -> None:
        """
        Launches the runner that runs most things (MQTT queue, etc.)
        """
        try:
            # Launch the MQTT thread listener
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER, msg='Launching MQTT thread.')
            self.threadMQTT = threading.Thread(
                target=self.threadImplMQTT, name="MQTTListener")
            self.threadMQTT.start()

            # Launch main event loop in a separate thread
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER, msg='Launching event loop')
            self.eventLoopThread = threading.Thread(
                target=self.threadImplEventLoop, name="asyncioEventLoop")
            self.eventLoopThread.start()

            # Loading model(s) in memory
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER, msg='Launching object detector load_model...')
            self.load_model()
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER, msg='Model loaded in memory.')
            self._readyForInferencing = True

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
        except Exception:
            log.error(LOGGER_OBJECT_DETECTOR_RUNNER,
                      f'Error : {traceback.print_exc()}')

    def load_model(self) -> None:
        log.warning(LOGGER_OBJECT_DETECTOR_LOAD_MODEL,
                    f'Connecting to camera {self._camera_index}')
        self.cap = cv2.VideoCapture(self._camera_index)

        log.warning(LOGGER_OBJECT_DETECTOR_LOAD_MODEL,
                    f'Rehydrating inference graph in memory...')

        # Load a (frozen) Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.frozen_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                log.warning(LOGGER_OBJECT_DETECTOR_LOAD_MODEL,
                            msg=f'Loading model in memory...')
                tf.import_graph_def(od_graph_def, name='')

        # Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`,
        # we know that this corresponds to `airplane`.  Here we use internal utility functions,
        # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(self.label_map_path)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.num_classes,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def run_detection(self, show_video=False, loop=False):
        """
        returns object detection
        params :
        show_video:bool:for debugging, requires a monitor (or x redirect over network which doesn't work)
                        only relevant if loop=True. Othrewise has no effect.
        loop:bool:if true, loops indefinitely and doesn't go out. False, sends a single pixels set back
        returns:if loop=false, returns boxes, scores, classes, num_detections. Otherwise doesn't return.
        """
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
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

                if loop:
                    while True:
                        # Read frame from camera
                        ret, image_np = self.cap.read()
                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(image_np, axis=0)
                        # Actual detection
                        (boxes, scores, classes, num_detections) = sess.run(
                            [boxes, scores, classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})

                        if show_video:
                            # Visualization of the results of a detection.
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                image_np,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                self.category_index,
                                use_normalized_coordinates=True,
                                line_thickness=8)
                            # Display output
                            cv2.imshow('object detection',
                                       cv2.resize(image_np, (1024, 576)))
                            if cv2.waitKey(5) & 0xFF == ord('q'):
                                cv2.destroyAllWindows()
                                break
                else:
                    # Read frame from camera
                    ret, image_np = self.cap.read()
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    return boxes, scores, classes, num_detections

        def __del__(self) -> None:
            pass

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

    def threadImplEventLoop(self):
        """
        Main event asyncio eventloop launched in a separate thread
        """
        try:
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP, f'Launching main event loop')
            self.eventLoop = asyncio.new_event_loop()
            self.eventLoopMainExecutors = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS, thread_name_prefix='MainThreadPool')
            self.asyncImplMoveLoopTask = self.eventLoop.create_task(self.asyncProcessMQTTMessages(0.25))
            # TODO: implement kill switch... not sure what to kill but maybe gracefull shutdown
            # self.eventLoop.run_in_executor(self.eventLoopMainExecutors, self.asyncExecutorImplKillSwitch)
            self.eventLoop.run_forever()

        except Exception:
            log.error(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP, f'Error : {traceback.print_exc()}')
        finally:
            self.eventLoop.stop()
            self.eventLoop.close()

    async def asyncProcessMQTTMessages(self, loopDelay):
        """
        This function receives the messages from MQTT to move the robot.
        It is implemented in another thread so that the killswitch can kill
        the thread without knowing which specific
        """
        while True:
            try:
                await asyncio.sleep(loopDelay)
                if self.mqttMessageQueue.empty() is False:
                    # Remove the first in the list, will pause until there is something
                    currentMQTTMoveMessage = self.mqttMessageQueue.get()
                    # Decode message received
                    msgdict = json.loads(currentMQTTMoveMessage.payload.decode('utf-8'))

                    # Check if need to shut down
                    if currentMQTTMoveMessage.topic == 'bot/killSwitch':
                        log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT, 'Kill switch activated')
                        self.killSwitch()

                    elif currentMQTTMoveMessage.topic == 'bot/jetson/detectSingleObject':
                        log.info(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT, 'Invoking object detection')
                        if not self._readyForInferencing:
                            log.error(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT, 'Model not loaded yet in memory... come back later.')
                        else:
                            boxes, scores, classes, num_detections = self.run_detection(show_video=False, loop=False)
                            # publish output to MQTT for retrieval by the caller...
                            log.info(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT, msg=f'Output : {boxes}, {scores}, {classes}, {num_detections}')

                    elif currentMQTTMoveMessage.topic == 'bot/logger':
                        # Changing the logging level on the fly...
                        log.setLevel(msgdict['logger'], lvl=msgdict['level'])
                    else:
                        raise NotImplementedError
            except NotImplementedError:
                log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT, f'MQTT topic not implemented.')
            except asyncio.futures.CancelledError:
                log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT, f'Cancelled the MQTT dequeing task.')
            except:
                log.error(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT, f'Error : {traceback.print_exc()}')
                # raise

    def killSwitch(self):
        try:
            log.warning(LOGGER_OBJECT_DETECTOR_KILL_SWITCH, 'Killing process. - to be implemented.')
        except:
            pass

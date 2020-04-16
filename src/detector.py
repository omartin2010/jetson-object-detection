import json
import time
import queue
import os
import traceback
import threading
from copy import deepcopy
from publish_queues import PublishQueue
import multiprocessing as mp
import numpy as np
import paho.mqtt.client as mqtt
import io
from PIL import Image
import base64
import cv2
import asyncio
from object_tracking_process import ObjectTrackingProcess
from aiohttp import ClientSession
from aiohttp.client_exceptions import ServerDisconnectedError
from pyk4a import PyK4A, K4AException, FPS, Calibration, CalibrationType
from pyk4a import Config as k4aConf
from constant import K4A_DEFINITIONS
from constant import LOGGER_OBJECT_DETECTOR_MAIN, \
    OBJECT_DETECTOR_CONFIG_DICT, LOGGER_OBJECT_DETECTOR_MQTT_LOOP, \
    LOGGER_OBJECT_DETECTOR_RUNNER, LOGGER_OBJECT_DETECTOR_ASYNC_LOOP, \
    LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT, \
    LOGGER_OBJECT_DETECTOR_KILL_SWITCH, LOGGER_ASYNC_RUN_DETECTION, \
    LOGGER_ASYNC_RUN_CAPTURE_LOOP, LOGGER_OBJECT_DETECTOR_SORT_TRACKED_OBJECTS, \
    LOGGER_OBJECT_DETECTION_KILL, LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER, \
    LOGGER_OBJECT_DETECTION_THREAD_POLL_OBJECT_TRACKING_PROCESS_QUEUE, \
    LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN, LOGGER_OBJECT_DETECTION_UPDATE_IMG_WITH_INFO
from logger import RoboLogger
from tracked_object import BoundingBox, TrackedObjectMP, FMT_TF_BBOX
log = RoboLogger.getLogger()
log.warning(LOGGER_OBJECT_DETECTOR_MAIN, msg=f'PID {os.getpid()} Launched. Libraries loaded.')


class ObjectDetector(object):
    """ main class for the object detector running on the jetson """

    def __init__(self,
                 configuration: dict,
                 loop: asyncio.AbstractEventLoop) -> None:
        """camera_index: int, frozen_graph_path: str,
        label_map_path: str, num_classes: int) -> None:"""
        self.configuration = configuration
        # Adjust configuration (to make strings into constants)
        self.__adjustConfigDict(self.configuration)
        self.num_classes = configuration[OBJECT_DETECTOR_CONFIG_DICT]['num_classes']
        self.label_map_path = configuration[OBJECT_DETECTOR_CONFIG_DICT]['label_map_path']
        self.mqtt_message_queue = queue.Queue()
        self.exception_queue = queue.Queue()
        self.ready_queue = queue.Queue()   # used to signal that there's an image ready for processing
        k4a_config_dict = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['k4a_device']
        self.k4a_config = k4aConf(
            color_resolution=k4a_config_dict['color_resolution'],
            depth_mode=k4a_config_dict['depth_mode'],
            camera_fps=k4a_config_dict['camera_fps'],
            synchronized_images_only=k4a_config_dict['synchronized_images_only'])
        self.k4a_device = PyK4A(self.k4a_config)
        self.k4a_device_calibration = Calibration(
            device=self.k4a_device,
            config=self.k4a_config,
            source_calibration=CalibrationType.COLOR,
            target_calibration=CalibrationType.GYRO)
        self.category_index = self.configuration['object_classes']
        self.__fix_category_index_dict()
        self.started_threads = {}
        self._show_video = False
        self._show_depth_video = False
        if k4a_config_dict['camera_fps'] == FPS.FPS_5:
            self.frame_duration = 1. / 5
        elif k4a_config_dict['camera_fps'] == FPS.FPS_15:
            self.frame_duration = 1. / 15
        elif k4a_config_dict['camera_fps'] == FPS.FPS_30:
            self.frame_duration = 1. / 30
        else:
            raise Exception('Unsupported frame rate {}'.format(
                            k4a_config_dict['camera_fps']))
        self.detection_threshold = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['detection_threshold']
        self.resized_image_resolution = tuple(configuration[OBJECT_DETECTOR_CONFIG_DICT]['resized_resolution'])
        self.lock_tracked_objects_mp = threading.Lock()
        self.tracked_objects_mp = {}
        self.image_queue = PublishQueue()
        self.default_tracker = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['open_cv_default_tracker']
        self._max_unseen_time_for_object = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['max_unseen_time_for_object']
        self.max_tracked_objects = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['max_tracked_objects']
        self._time_between_scoring_service_calls = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['time_between_scoring_service_calls']
        self.eventLoop = loop

    @property
    def max_unseen_time_for_object(self):
        return self._max_unseen_time_for_object

    @max_unseen_time_for_object.setter
    def max_unseen_time_for_object(self, value):
        if value < self.time_between_scoring_service_calls:
            self._max_unseen_time_for_object = value
        else:
            log.error('PROPERTY',
                      msg=f'Value for max_unseen_time_for_object needs to '
                          f'be less than time_between_scoring_service_calls '
                          f'({self.time_between_scoring_service_calls}). Keeping '
                          f'current value of {self._max_unseen_time_for_object}')

    @property
    def time_between_scoring_service_calls(self):
        return self._time_between_scoring_service_calls

    @time_between_scoring_service_calls.setter
    def time_between_scoring_service_calls(self, value):
        if value > self.max_unseen_time_for_object:
            self._time_between_scoring_service_calls = value
        else:
            log.error('PROPERTY',
                      msg=f'Value for time_between_scoring_service_calls needs to '
                          f'be more than max_unseen_time_for_object '
                          f'({self.max_unseen_time_for_object}). Keeping '
                          f'current value of {self._time_between_scoring_service_calls}')

    @property
    def show_video(self):
        return self._show_video

    @show_video.setter
    def show_video(self, value):
        if value is False and self._show_video:
            cv2.destroyWindow('show_video')
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                        msg=f'Deleting display window for show_video...')
        else:
            self._show_video = value

    @property
    def show_depth_video(self):
        return self._show_depth_video

    @show_depth_video.setter
    def show_depth_video(self, value):
        if value is False and self._show_depth_video:
            cv2.destroyWindow('show_depth_video')
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                        msg=f'Deleting display window for show_depth_video...')
        else:
            self._show_depth_video = value

    async def graceful_shutdown(self, s=None) -> None:
        """
        Description : Ensures a clean shutdown of the robot, including shutting down
            the Kinect camera
        Args:
            s = signal from the signal library. Could be SIGINT, SIGTERM,
                etc. if set to None, it's cancelled from another process.
        """
        try:
            if s is not None:
                log.critical(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                             msg=f'Initiating graceful shutdown now '
                                 f'from received signal {s.name}.')
            else:
                log.critical(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                             msg=f'Initiating graceful shutdown now '
                                 f'from non signal.')
            # region asyncio event loop tasks
            try:
                tasks = [t for t in asyncio.Task.all_tasks(loop=self.eventLoop)
                         if t is not asyncio.Task.current_task()]
                log.warning(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                            msg=f'Cancelling task {len(tasks)} tasks...')
                [task.cancel() for task in tasks]
                log.warning(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                            msg=f'Gaterhing out put of cancellation '
                                f'of {len(tasks)} tasks...')
                out_list = await asyncio.gather(*tasks, loop=self.eventLoop, return_exceptions=True)
                for idx, out in enumerate(out_list):
                    if isinstance(out, Exception):
                        log.warning(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                                    msg=f'Exception in stopping task {idx}')
                log.warning(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                            msg=f'Done cancelling tasks.')
            except:
                pass
            # endregion asyncio event loop tasks

            # Stop MQTT client:
            try:
                self.mqttClient.loop_stop()
                self.mqttClient.disconnect()
                if self.mqttClient.is_connected():
                    log.warning(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                                msg=f'Unable to stop MQTT client.')
                else:
                    log.warning(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                                msg=f'Stopped MQTT client.')
            except:
                log.warning(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN, msg=f'Exception in shutting down MQTT')

            # Stop object watchers
            try:
                log.warning(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                            msg=f'Stopping object watchers...')
                self.kill(all_objects=True)
                log.warning(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                            msg=f'Object watchers stopped.')
            except:
                pass

            try:
                # Stop remaining threads
                for thread, event in self.started_threads.items():
                    if thread.is_alive():
                        event.set()
                        await asyncio.sleep(0.5)
                        if thread.is_alive():
                            log.error(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                                      msg=f'Problem shutting down '
                                          f'some threads!')
            except:
                pass

            # Stop devices
            try:
                self.k4a_device.disconnect()
                log.warning(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                            msg=f'Disconnected K4A camera')
            except:
                pass

        except:
            log.warning(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                        msg=f'Problem in graceful_shutdown')
        finally:
            log.warning(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                        msg=f'Done!')
            self.eventLoop.stop()

    async def run(self) -> None:
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

            # region Launch Threads
            # Launch the MQTT thread listener
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg='Launching MQTT thread.')
            self.threadMQTT = threading.Thread(
                target=self.thread_mqtt_listener, name='MQTTListener')
            self.threadMQTT.start()
            self.started_threads[self.threadMQTT] = None

            # Launch event loop tasks in the main thread
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg='Launching asyncio tasks...')
            self.event_loop_start_main_tasks()

            # Launching capture loop thread
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg=f'Launching Capture Loop Thread')
            exit_capture_thread_event = threading.Event()
            self.capture_loop_thread = threading.Thread(
                target=self.thread_capture_loop,
                args=([exit_capture_thread_event]),
                name='captureLoop')
            self.capture_loop_thread.start()
            self.started_threads[self.capture_loop_thread] = exit_capture_thread_event

            # Launching Manage Object Detection Pools

            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg=f'Launching Object Detection Pool and Thread Managers')
            exit_detection_pool_manager_thread_event = threading.Event()
            self.object_detection_pool_manager_thread = threading.Thread(
                target=self.thread_object_detection_pool_manager,
                args=([exit_detection_pool_manager_thread_event, 0.25]),
                name='objectDetectionManager')
            self.object_detection_pool_manager_thread.start()
            self.started_threads[self.object_detection_pool_manager_thread] = exit_detection_pool_manager_thread_event

            # endregion

        except SystemExit:
            # raise the exception up the stack
            raise Exception(f'Error : {traceback.print_exc()}')

        except K4AException:
            log.error(LOGGER_OBJECT_DETECTOR_RUNNER,
                      f'Error with K4A : {traceback.print_exc()}')

        except Exception:
            log.error(LOGGER_OBJECT_DETECTOR_RUNNER,
                      f'Error : {traceback.print_exc()}')

    def event_loop_start_main_tasks(self):
        """
        Main event asyncio eventloop launched in a separate thread
        """
        try:
            # self.eventLoop = asyncio.new_event_loop()
            # asyncio.set_event_loop(self.eventLoop)

            # region Create Async Tasks
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                        msg=f'Launching asyncio TASK :"process MQTT message"')
            self.async_process_mqtt_messages_task = \
                self.eventLoop.create_task(
                    self.async_process_mqtt_messages(loopDelay=0.25))

            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                        msg=f'Launching asyncio TASK : "async_run_detection"')
            self.async_run_detection_task = \
                self.eventLoop.create_task(
                    self.async_run_detection())
            # endregion

            # self.eventLoop.run_forever()
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                        msg=f'Asyncio tasks started')

            # while True:
            #     time.sleep(1)

        except Exception:
            log.error(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                      f'Error : {traceback.print_exc()}')
            raise Exception(f'Error : {traceback.print_exc()}')
        finally:
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                        msg=f'Exiting event_loop_start_main_tasks')
            # self.eventLoop.stop()
            # self.eventLoop.close()

    def thread_mqtt_listener(self):
        """
        MQTT Thread launching the loop and subscripbing to the right topics
        """
        mqtt_default_qos = 2
        self.mqtt_topics = [(topic, mqtt_default_qos)
                            for topic in self.configuration['mqtt']['subscribedTopics']]

        def on_connect(client, userdata, flags, rc):
            log.warning(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                        msg=f'Connected to MQTT broker. Result code {str(rc)}')
            mqtt_connect_result, self.mqtt_connect_mid = client.subscribe(
                self.mqtt_topics)
            if mqtt_connect_result == mqtt.MQTT_ERR_SUCCESS:
                log.warning(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                            msg=f'Successfully subscribed to {self.mqtt_topics}')
            else:
                log.error(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                          'MQTT Broker subscription problem.')

        def on_message(client, userdata, message):
            """ callback function used for the mqtt client (called when
            a new message is publisehd to one of the queues we subscribe to)
            """
            log.info(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                     msg=f'Received MID {message.mid} : "{str(message.payload)}" '
                         f'on topic {message.topic} with QoS {message.qos}')
            self.mqtt_message_queue.put_nowait(message)

        def on_disconnect(client, userdata, rc=0):
            """callback for handling disconnects
            """
            log.info(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                     f'Disconnected MQTT result code = {rc}. '
                     f'Should automatically re-connect to broker')

        def on_subscribe(client, userdata, mid, granted_qos):
            if mid == self.mqtt_connect_mid:
                log.warning(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                            msg=f'Subscribed to topics. Granted QOS = {granted_qos}')
            else:
                log.error(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                          msg=f'Strange... MID does not match self.mqtt_connect_mid')

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
            self.mqttClient.loop_forever()
            # self.mqttClient.loop_start()
        except Exception:
            log.error(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                      f'Error : {traceback.print_exc()}')
            # Bump up the problem...
            raise Exception(f'Error : {traceback.print_exc()}')
        finally:
            log.warning(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                        msg=f'Exiting MQTT connection thread.')

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
        try:
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

                        elif currentMQTTMoveMessage.topic == 'bot/jetson/configure':
                            log.info(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                                     msg=f'Modifying configuration item...')
                            for k, v in msgdict.items():
                                if k in dir(self):
                                    log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                                                msg=f'Setting attribute self.{k} to value {v}')
                                    # Adding / changing configuration parameters for the object
                                    self.__setattr__(k, v)
                                    log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                                                msg=f'After validation, attribute self.{k} '
                                                    f'= value {self.__getattribute__(k)}')
                                else:
                                    log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                                                msg=f'Attribute self.{k} not found. Will not add it.')
                        elif currentMQTTMoveMessage.topic == 'bot/logger':
                            # Changing the logging level on the fly...
                            log.setLevel(msgdict['logger'], lvl=msgdict['level'])
                        else:
                            raise NotImplementedError
                    await asyncio.sleep(loopDelay)
                except NotImplementedError:
                    log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                                msg=f'MQTT topic not implemented.')
                except asyncio.futures.CancelledError:
                    log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                                msg=f'Cancelled the MQTT dequeing task.')
                    break
                except Exception:
                    raise
        except Exception:
            log.error(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                      msg=f'Error: {traceback.print_exc()}')

    def thread_capture_loop(self,
                            exit_thread_event: threading.Event) -> None:
        """
        Description :
            Video capture loop (can optionally display video) for debugging purposes.
                The loop also additionally calls __get_distance_from_k4a to keep objects
                updated with their respective distances to the camera
        Args:
            exit_thread_event : <class 'threading.Event()'> : used to signal
                process it's time to end.
        """
        try:
            # Launch the loop
            n_loops = 0
            logging_loops = 50
            average_duration = 0
            k4a_errors = 0
            while not exit_thread_event.is_set():
                start_time = time.time()
                # Read frame from camera
                try:
                    bgra_image_color_np, image_depth_np = \
                        self.k4a_device.get_capture(
                            color_only=False,
                            transform_depth_to_color=True)
                except K4AException:
                    k4a_errors += 1         # count problematic frame capture

                self.rgb_image_color_np = bgra_image_color_np[:, :, :3][..., ::-1]
                self.rgb_image_color_np_resized = np.asarray(
                    Image.fromarray(self.rgb_image_color_np).resize(
                        self.resized_image_resolution))
                self.image_queue.publish(self.rgb_image_color_np_resized)
                self.image_depth_np_resized = np.asarray(
                    Image.fromarray(image_depth_np).resize(
                        self.resized_image_resolution,
                        resample=Image.NEAREST))
                # only do this after the first loop is done
                if n_loops == 0:
                    self.ready_queue.put_nowait('image_ready')
                # retrieve and update distance from each object
                self.__get_distance_from_k4a()
                # Show video in debugging mode - move to other thread (that we can start on mqtt message...)
                if self.show_video:
                    # Visualization of the results of a detection.
                    with self.lock_tracked_objects_mp:
                        temp_items = deepcopy(self.tracked_objects_mp)
                    img = bgra_image_color_np[:, :, :3]
                    img = self.__update_image_with_info(img, temp_items)
                    cv2.imshow('show_video', cv2.resize(img, (1024, 576)))
                    # bgra_image_color_np_boxes, (1024, 576)))
                if self.show_depth_video:
                    cv2.imshow('show_depth_video', cv2.resize(
                        image_depth_np, (1024, 576)))
                if self.show_depth_video or self.show_video:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        self.show_video = False
                        self.show_depth_video = False
                duration = time.time() - start_time
                average_duration += duration
                n_loops += 1
                if n_loops % logging_loops == 0:
                    duration_50 = average_duration
                    average_duration /= 50
                    log.debug(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                              msg=f'Ran 50 in {duration_50:.2f}s - {average_duration:.2f}s/loop or {1/average_duration:.2f} loop/sec')
                    log.debug(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                              msg=f'Currently monitoring {len(self.tracked_objects_mp)} objects')
                    average_duration = 0
        except Exception:
            log.error(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                      f'Error : {traceback.print_exc()}')
            raise Exception(f'Error : {traceback.print_exc()}')

    async def async_run_detection(self) -> None:
        """
        returns object detection
        delay:loopDelay:delay to pause between frames scoring
        returns:task
        """
        try:
            log.warning(LOGGER_ASYNC_RUN_DETECTION,
                        msg=f'Run detection task started...')
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
                            # Reshuffle tracked objects from both capture and opencv object detection
                            self.sort_tracked_objects(body, (height, width))
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
                await asyncio.sleep(self.time_between_scoring_service_calls)
        except ServerDisconnectedError:
            log.error(LOGGER_ASYNC_RUN_DETECTION,
                      msg=f'HTTP prediction service error: '
                          f'{traceback.print_exc()} -> sleeping '
                          f'{self.time_between_scoring_service_calls}s '
                          f'and continuing')
            await asyncio.sleep(self.time_between_scoring_service_calls)
        except asyncio.futures.CancelledError:
            log.warning(LOGGER_ASYNC_RUN_DETECTION,
                        msg=f'Cancelled the detection task.')
        except Exception:
            log.error(LOGGER_ASYNC_RUN_DETECTION,
                      f'Error : {traceback.print_exc()}')
            raise Exception(f'Error : {traceback.print_exc()}')

    def thread_object_detection_pool_manager(self,
                                             exit_detection_pool_manager_thread_event: threading.Event,
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
            exit_thread_event : <class 'threading.Event()'> : used to signal
                process it's time to end.
            loop_delay : float : delay in the while true loop to ensure no high
                cpu usage for nothing.
        """
        self._mapping_object_process_thread = {}
        object_counter = 0
        while not exit_detection_pool_manager_thread_event.is_set():
            try:
                # to_be_killed_tracked_objects = {}
                with self.lock_tracked_objects_mp:
                    if len(self.tracked_objects_mp) > self.max_tracked_objects:
                        log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                    msg=f'Problem - self.max_tracked_objects > max_tracked_objects. '
                                        f'Normally this should be temporary as a result of a '
                                        f'configuration change')
                    temp_tracked_objects_mp = deepcopy(self.tracked_objects_mp)
                for (tracked_object_mp_id, tracked_object_mp) in \
                        temp_tracked_objects_mp.items():
                    # If it's not already monitored the monitor it
                    if tracked_object_mp_id not in self._mapping_object_process_thread:
                        tracking_object_queue = mp.Manager().Queue(maxsize=1)
                        new_tf_info_queue = mp.Manager().Queue(maxsize=1)
                        exit_thread_event = threading.Event()
                        proc = ObjectTrackingProcess(tracked_object=tracked_object_mp,
                                                     tracker_alg=self.default_tracker,
                                                     initial_image=self.rgb_image_color_np_resized,
                                                     frame_duration=self.frame_duration,
                                                     image_queue=self.image_queue.register(name=str(tracked_object_mp_id)),
                                                     new_tf_info_queue=new_tf_info_queue,
                                                     tracking_object_queue=tracking_object_queue)
                        thread = threading.Thread(target=self.thread_poll_object_tracking_process_queue,
                                                  args=([tracked_object_mp,
                                                        tracking_object_queue,
                                                        exit_thread_event]),
                                                  name=f'ObjTrack_{str(tracked_object_mp_id)[:8]}')
                        # # Keep track of what is under observation
                        self._mapping_object_process_thread[tracked_object_mp_id] = {
                            'object_counter': object_counter,
                            'tracking_object_queue': tracking_object_queue,
                            'new_tf_info_queue': new_tf_info_queue,
                            'proc': proc,
                            'thread': thread,
                            'exit_thread_event': exit_thread_event,
                            'flag_for_kill': False
                        }
                        object_counter += 1
                        proc.start()
                        log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                    msg=f'Manager launched process {proc.pid} for object {tracked_object_mp_id}.')
                        thread.name += f'_pid{proc.pid}'
                        thread.start()
                        log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                    msg=f'Manager launched thread for object {tracked_object_mp_id}.')
                    # Check if object related process/thread should be purged (unseen for 5 seconds)
                    proc = self._mapping_object_process_thread[tracked_object_mp_id]['proc']
                    thread = self._mapping_object_process_thread[tracked_object_mp_id]['thread']
                    # If object unseen for X seconds:
                    unseen_time = time.time() - tracked_object_mp.last_seen
                    if unseen_time > self.max_unseen_time_for_object:
                        self._mapping_object_process_thread[tracked_object_mp_id]['flag_for_kill'] = True
                        log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                    msg=f'Marking object {tracked_object_mp_id} '
                                        f'for deletion, unseen for {unseen_time:.2}s')
                    else:
                        # verify if object proc and thread are healthy - if thread or proc is dead, kill tracker
                        if not (proc.is_alive() and thread.is_alive()):
                            self._mapping_object_process_thread[tracked_object_mp_id]['flag_for_kill'] = True
                            log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                        msg=f'Marking object {tracked_object_mp_id} '
                                            f'for deletion - thread or proc has been stopped, '
                                            f'need to get back to safe state')
                # Call function that kills the threads/procs that need cleanup
                nb_obj_to_delete = len({k: v for (k, v) in self._mapping_object_process_thread.items() if v['flag_for_kill']})
                if nb_obj_to_delete > 0:
                    self.kill()
            except:
                raise Exception(f'Problem : {traceback.print_exc()}')
            time.sleep(loop_delay)
        log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                    msg=f'Exiting thread "thread_object_detection_pool_manager". '
                        f'Should not happen unless stopping robot.')

    def kill(self,
             all_objects=False) -> None:   # , tracked_objects):
        """
        Description:
            Function used to garbage collect (remove unused threads and
                processes) of tracked objects that dissapeared. The
                self._mapping_object_process_thread[tracked_object_mp_id]['flag_for_kill']
                flag will be set to True for those who need to dissapear.
        Args:
            all_objects : bool, if set to True, will stop all threads and
                processes.
        """
        try:
            # Extract list of objects to be flagged for removal
            if all_objects:
                to_be_killed_tracked_objects = \
                    {k: v for k, v in self._mapping_object_process_thread.items()}
            else:
                to_be_killed_tracked_objects = \
                    {k: v for k, v in self._mapping_object_process_thread.items() if v['flag_for_kill']}
            for (tracked_object_mp_id, tracked_object) in to_be_killed_tracked_objects.items():
                proc = self._mapping_object_process_thread[tracked_object_mp_id]['proc']
                thread = self._mapping_object_process_thread[tracked_object_mp_id]['thread']
                exit_thread_event = self._mapping_object_process_thread[tracked_object_mp_id]['exit_thread_event']
                tracking_object_queue = self._mapping_object_process_thread[tracked_object_mp_id]['tracking_object_queue']

                # Signal thread it's time to stop (use a threading.event?)
                exit_thread_event.set()
                while thread.is_alive():
                    thread.join(2)  # Wait up to X sec
                    if thread.is_alive():
                        log.warning(LOGGER_OBJECT_DETECTION_KILL,
                                    msg=f'Failed to cancel thread for object {tracked_object_mp_id}.')
                log.warning(LOGGER_OBJECT_DETECTION_KILL,
                            msg=f'Thread for object {tracked_object_mp_id} is terminated.')

                # Signal process it's time to stop
                proc.exit.set()
                while proc.is_alive():
                    log.warning(LOGGER_OBJECT_DETECTION_KILL,
                                msg=f'Attempting to cancel PID {proc.pid} for '
                                    f'object {tracked_object_mp_id}.')
                    proc.join(2)  # Wait up to X sec
                    exitcode = proc.exitcode
                    log.warning(LOGGER_OBJECT_DETECTION_KILL,
                                msg=f'PID {proc.pid} exit code = {exitcode} '
                                    f'for {tracked_object_mp_id}.'
                                    f'Proc.is_alive() = {proc.is_alive()}')
                log.warning(LOGGER_OBJECT_DETECTION_KILL,
                            msg=f'PID {proc.pid} for object '
                                f'{tracked_object_mp_id} is terminated.')

                # Remove from dict altogether
                log.warning(LOGGER_OBJECT_DETECTION_KILL,
                            msg=f'Deleting object {tracked_object_mp_id} from mapping dictionnary.')
                self._mapping_object_process_thread.pop(tracked_object_mp_id)

                # Unregister specific image_queue from PublisQueue
                self.image_queue.unregister(name=str(tracked_object_mp_id))

                # Delete object queue (force garbage collection)
                del tracking_object_queue, exit_thread_event
        except KeyError:
            raise Exception(f'Problem : tried removing tracked object from '
                            f'dictionnary but key not found. Traceback = '
                            f'{traceback.print_exc()}')
        except:
            raise Exception(f'Problem : {traceback.print_exc()}')

    def thread_poll_object_tracking_process_queue(self,
                                                  tracked_object_mp: TrackedObjectMP,
                                                  tracked_object_queue: mp.Queue,
                                                  exit_thread_event: threading.Event):
        """
        Description: thread to monitor objects and retrieve updated bounding box
            coordinates - these are calculated in the associated process.
        Args:
            tracked_object_mp : <class TrackedObjectMP()>, object that this
                thread is tracking the bounding box for
            tracked_object_queue : <class 'multiprocessing.Queue()'> :
                containing the object that has the latest bounding box
            exit_thread_event : <class 'threading.Event()'> : used to signal
                process it's time to end.
        """
        try:
            log.warning(LOGGER_OBJECT_DETECTION_THREAD_POLL_OBJECT_TRACKING_PROCESS_QUEUE,
                        msg=f'Launching thread for object ID {tracked_object_mp.id}')
            while not exit_thread_event.is_set():
                # Replace object in tracked object with what we get from the queue
                # wait max 2 second
                new_tracked_object_mp = tracked_object_queue.get(block=True, timeout=2)
                # Get the new coordinates in the object.
                with self.lock_tracked_objects_mp:
                    # Add the information in only if object already monitored
                    if new_tracked_object_mp.id in \
                            self.tracked_objects_mp.keys():
                        self.tracked_objects_mp[new_tracked_object_mp.id] = \
                            new_tracked_object_mp
                    else:
                        # If object not there, end thread
                        log.warning(LOGGER_OBJECT_DETECTION_THREAD_POLL_OBJECT_TRACKING_PROCESS_QUEUE,
                                    msg=f'Not finding object in self.tracked_objects_mp.')
                        break
        except queue.Empty:
            log.warning(LOGGER_OBJECT_DETECTION_THREAD_POLL_OBJECT_TRACKING_PROCESS_QUEUE,
                        msg=f'No new object was placed on the queue, tracking process '
                            f'may have been terminated by parent process.')
        except:
            raise Exception(f'Problem: {traceback.print_exc()}')
        finally:
            log.warning(LOGGER_OBJECT_DETECTION_THREAD_POLL_OBJECT_TRACKING_PROCESS_QUEUE,
                        msg=f'Ending thread handling object {tracked_object_mp.id}')

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

    def __fix_category_index_dict(self):
        """
        Description:
            Fixes the category index so that instead of strings describing colors,
                we have tuples
        Args:
            None
        """
        for (class_id, class_dict) in self.category_index.items():
            for (k, v) in class_dict.items():
                if k == 'rect_color' or k == 'text_color':
                    # Convert the string values to tuple
                    self.category_index[class_id][k] = tuple(map(int, v.strip('(').strip(')').split(',')))

    def __update_image_with_info(self, img, temp_items, alpha=0.5):
        """
        Description: updates images with information to display on the video
            if shown
        Args:
            img : numpy array containing original image to display
            temp_items : contains a dictionnary of items to display
            alpha : float, overlay transparency =defaults to 0.5
        Returns:
            modified image
        """
        try:
            height, width = img.shape[:2]
            output = img.copy()
            overlay = img.copy()
            for uuid, obj in temp_items.items():
                str_uuid = str(uuid).split('-')[0]
                x, y, w, h = obj.get_bbox(
                    fmt='tracker',
                    use_normalized_coordinates=True)
                rect_color = self.category_index[str(int(obj.object_class))]['rect_color']
                text_color = self.category_index[str(int(obj.object_class))]['text_color']
                class_str = self.category_index[str(int(obj.object_class))]['class_name']
                # Top section
                top_str_1 = f'Class: {class_str}'
                top_str_2 = f'UUID: {str_uuid}'
                x_size_top_str_1 = cv2.getTextSize(top_str_1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0][0] + 10
                x_size_top_str_2 = cv2.getTextSize(top_str_2, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0][0] + 10
                x_size_top_str = max(x_size_top_str_1, x_size_top_str_2)
                # Bottom section
                bottom_str_1 = f'Score: {obj.score*100:.2f}%'
                if obj.distance:
                    bottom_str_1 += f' - Dist. : {obj.distance:.1f}mm'
                else:
                    bottom_str_1 += f' - Dist. : Too close.'
                if obj.coords_3d_coordinates_center_point:
                    bottom_str_2 = f'Coords : (' + \
                        f'{obj.coords_3d_coordinates_center_point[0]:.1f}, ' + \
                        f'{obj.coords_3d_coordinates_center_point[1]:.1f}, ' + \
                        f'{obj.coords_3d_coordinates_center_point[2]:.1f})'
                else:
                    bottom_str_2 = f'Coords: Can\'t compute coords.'
                x_size_bottom_str_1 = cv2.getTextSize(bottom_str_1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0][0] + 10
                x_size_bottom_str_2 = cv2.getTextSize(bottom_str_2, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0][0] + 10
                x_size_bottom_str = max(x_size_bottom_str_1, x_size_bottom_str_2)
                # Denormalize coordinates
                x = int(x * width)
                y = int(y * height)
                w = int(w * width)
                h = int(h * height)

                # Main bounding box
                overlay = cv2.rectangle(cv2.UMat(overlay).get(), (x, y), (x + w, y + h), rect_color, 4)
                # Top left box with information
                overlay = cv2.rectangle(overlay, (x, y - 63), (x + x_size_top_str, y), rect_color, thickness=-1)
                overlay = cv2.putText(overlay, top_str_1, (x + 5, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)
                overlay = cv2.putText(overlay, top_str_2, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)
                # Bottom left box with information
                overlay = cv2.rectangle(overlay, (x, y + h), (x + x_size_bottom_str, y + h + 60), rect_color, thickness=-1)
                overlay = cv2.putText(overlay, bottom_str_1, (x + 5, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)
                overlay = cv2.putText(overlay, bottom_str_2, (x + 5, y + h + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)
            output = cv2.addWeighted(cv2.UMat(overlay).get(), alpha, cv2.UMat(output).get(), (1 - alpha), 0, cv2.UMat(output).get())
            return output
        except Exception:
            log.warning(LOGGER_OBJECT_DETECTION_UPDATE_IMG_WITH_INFO,
                        msg=f'Problem in __update_image_with_info')
            raise Exception(f'Problem in __update_image_with_info : trace = {traceback.print_exc()}')

    def sort_tracked_objects(self,
                             detection_response,
                             image_shape):
        """
        Description:
            Sort tracked objects, that is find in the list of detections from
                model scoring, which ones of the object trackings we should update
                with the detection versus open cv object tracking information.
        Args:
            detection_response : string containing the json response of
                the scored model. Includes bounding boxes, scores and
                classes of scored items.
            image_shape : tuple (height, width) in pixels
        Returns:
        ...
        """
        try:
            height, width = image_shape
            detection_boxes = detection_response['boxes']
            detection_scores = detection_response['scores']
            detection_classes = detection_response['classes']
            # Find #boxes with score > thresh or max tracked objects
            nb_bb = min(
                self.max_tracked_objects,
                np.sum(np.array(detection_scores[0]) > self.detection_threshold))
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
            for (uuid, tracked_object) in deepcopy(self.tracked_objects_mp).items():
                # Find best overlapping bounding box
                target_bb_idx = tracked_object.get_max_overlap_bb(bb_list)
                # Update dictionnary with outcome unless target_bb == None
                if target_bb_idx is not None:
                    # SEND BBOX TO QUEUE FOR PROCESSING ON THE OTHER PROC
                    if uuid in self._mapping_object_process_thread.keys():
                        self._mapping_object_process_thread[uuid]['new_tf_info_queue'].put((
                            bb_list[target_bb_idx],
                            ds_list[target_bb_idx],
                            dc_list[target_bb_idx]))
                    else:    # SHOULD NOT HAPPEN...
                        log.critical(LOGGER_OBJECT_DETECTOR_SORT_TRACKED_OBJECTS,
                                     msg=f'Problem - should not go there...')
                        # tracked_object.update_bounding_box(
                        #     bb_list[target_bb_idx])
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
                    box=bb.get_bbox(fmt=FMT_TF_BBOX),
                    fmt=FMT_TF_BBOX)
                temp_new_tracked_objects_mp[new_obj.id] = new_obj
                log.warning(LOGGER_OBJECT_DETECTOR_SORT_TRACKED_OBJECTS,
                            msg=f'Created temp tracker (id={new_obj.id})')

            # Combine output and get only up to max_tracked_objects items
            # ADD ORDERED BY SCORE
            combined_temp = {**temp_new_tracked_objects_mp,
                             **temp_existing_tracked_objects_mp}
            temp_tracked_objects_mp = {k: combined_temp[k]
                                       for k in list(combined_temp)
                                       [:self.max_tracked_objects]}
            # Order items in dict by scores
            ordered_temp_tracked_objects_mp = {
                k: v for k, v in sorted(
                    temp_tracked_objects_mp.items(),
                    key=lambda item: item[1].score,
                    reverse=True)
            }
            # Copy temp list back to object
            with self.lock_tracked_objects_mp:
                self.tracked_objects_mp = deepcopy(ordered_temp_tracked_objects_mp)
                if len(self.tracked_objects_mp) > self.max_tracked_objects:
                    log.warning(LOGGER_OBJECT_DETECTOR_SORT_TRACKED_OBJECTS,
                                msg=f'Problem - self.max_tracked_objects > max_tracked_objects. '
                                    f'Normally this should be temporary as a result of a '
                                    f'configuration change')
        except:
            raise Exception(f'Problem : {traceback.print_exc()}')

    def __get_distance_from_k4a(self):
        """
        Description:
            Retrieve the distance from a specific object to the camera lens.
        Args: None
        Returns : float, distance in mm from specific point on camera
        """
        try:
            # For each object
            for (uuid, obj) in self.tracked_objects_mp.items():
                # Get object coordinates in the image
                (x, y, w, h) = obj.get_bbox(
                    fmt='tracker',
                    use_normalized_coordinates=True)
                x = int(x * self.resized_image_resolution[0])
                y = int(y * self.resized_image_resolution[1])
                w = int(w * self.resized_image_resolution[0])
                h = int(h * self.resized_image_resolution[1])
                cropped_depth_map = self.image_depth_np_resized[y: y + h, x: x + w]
                obj.distance = np.round(np.average(cropped_depth_map))
                center_point = [int(x + w / 2), int(y + h / 2)]
                depth_center_point = self.image_depth_np_resized[center_point[1], center_point[0]]
                (valid, coords) = self.k4a_device_calibration.convert_2d_to_3d(
                    center_point,
                    depth_center_point)
                if valid:
                    obj.coords_3d_coordinates_center_point = coords
                else:
                    obj.coords_3d_coordinates_center_point = None
        except Exception:
            raise Exception(f'Problem with __get_distance_from_k4a : {traceback.print_exc()}')
        pass

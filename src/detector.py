import json
import tempfile
import time
import datetime
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
import shutil
from PIL import Image
import base64
from azure.storage.blob.aio import BlobServiceClient
from azure.storage.blob import ContentSettings
from azure.core.exceptions import ResourceExistsError
from azure.identity.aio import EnvironmentCredential
import cv2
import asyncio
from object_tracking_process import ObjectTrackingProcess
from aiohttp import ClientSession
from aiohttp.client_exceptions import ServerDisconnectedError, \
    ClientConnectorError
from pyk4a import PyK4A, K4AException, FPS, Calibration, CalibrationType
from pyk4a import Config as k4aConf
from constant import K4A_DEFINITIONS
from constant import LOGGER_OBJECT_DETECTOR_MAIN, \
    OBJECT_DETECTOR_CONFIG_DICT, \
    LOGGER_OBJECT_DETECTOR_MQTT_LOOP, \
    LOGGER_OBJECT_DETECTOR_RUNNER, LOGGER_OBJECT_DETECTOR_ASYNC_LOOP, \
    LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT, \
    OBJECT_DETECTOR_CLOUD_CONFIG_DICT, LOGGER_OBJECT_DETECTOR_KILL_SWITCH, \
    LOGGER_ASYNC_RUN_DETECTION, LOGGER_ASYNC_RUN_CAPTURE_LOOP, \
    LOGGER_OBJECT_DETECTOR_SORT_TRACKED_OBJECTS, \
    LOGGER_OBJECT_DETECTION_KILL, \
    LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER, \
    LOGGER_OBJECT_DETECTION_THREAD_POLL_OBJECT_TRACKING_PROCESS_QUEUE, \
    LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN, \
    LOGGER_OBJECT_DETECTION_UPDATE_IMG_WITH_INFO, \
    LOGGER_OBJECT_DETECTION_ASYNC_RECORD_VIDEO, \
    LOGGER_OBJECT_DETECTION_ASYNC_SAVE_PICTURE, \
    LOGGER_OBJECT_DETECTION_GET_DISTANCE_FROM_K4A, \
    LOGGER_OBJECT_DETECTION_ASYNC_DISPLAY_VIDEO, \
    LOGGER_OBJECT_DETECTION_ASYNC_UPLOAD_FILE_TO_AZURE
from logger import RoboLogger
from tracked_object import BoundingBox, TrackedObjectMP, \
    FMT_TF_BBOX, UNDETECTED, UNSEEN, UNSTABLE
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
        self.exception_queue = queue.Queue()        # PROBABLY NOT NEEDED
        self.ready_queue = queue.Queue()   # used to signal that there's an image ready for processing
        self.object_detection_result_queue = queue.Queue(maxsize=5)   # 5 allows for debugging...
        # k4a_config_dict = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['k4a_device']
        # self.k4a_config = k4aConf(
        #     color_resolution=k4a_config_dict['color_resolution'],
        #     depth_mode=k4a_config_dict['depth_mode'],
        #     camera_fps=k4a_config_dict['camera_fps'],
        #     synchronized_images_only=k4a_config_dict['synchronized_images_only'])
        # self.k4a_device = PyK4A(self.k4a_config)
        # self.k4a_device_calibration = Calibration(
        #     device=self.k4a_device,
        #     config=self.k4a_config,
        #     source_calibration=CalibrationType.COLOR,
        #     target_calibration=CalibrationType.GYRO)
        self.category_index = self.configuration['object_classes']
        self._fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.__video_file_extension = '.avi'
        self.__video_content_type = 'video/x-msvideo'
        self.__image_file_extension = '.jpg'
        self.__image_content_type = 'image/jpeg'
        self.__fix_category_index_dict()
        self.started_threads = {}
        self.show_video = False
        self.show_depth_video = False
        # if k4a_config_dict['camera_fps'] == FPS.FPS_5:
        #     self.frame_duration = 1. / 5
        #     self.fps = 5
        # elif k4a_config_dict['camera_fps'] == FPS.FPS_15:
        #     self.frame_duration = 1. / 15
        #     self.fps = 15
        # elif k4a_config_dict['camera_fps'] == FPS.FPS_30:
        #     self.frame_duration = 1. / 30
        #     self.fps = 30
        # else:
        #     raise Exception('Unsupported frame rate {}'.format(
        #                     k4a_config_dict['camera_fps']))
        self.detection_threshold = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['detection_threshold']
        self.resized_image_resolution = tuple(configuration[OBJECT_DETECTOR_CONFIG_DICT]['resized_resolution'])
        self.display_image_resolution = tuple(configuration[OBJECT_DETECTOR_CONFIG_DICT]['display_resolution'])
        self.ready_for_first_detection_event = threading.Event()
        self.lock_tracked_objects_mp = threading.Lock()
        self.tracked_objects_mp = {}
        self.image_queue = PublishQueue()
        self.default_tracker = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['open_cv_default_tracker']
        self.max_unseen_time_for_object = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['max_unseen_time_for_object']
        self.max_tracked_objects = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['max_tracked_objects']
        self.time_between_scoring_service_calls = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['time_between_scoring_service_calls']
        self.eventLoop = loop
        self.__azure_credentials = EnvironmentCredential()
        self.__blob_service_endpoint = self.configuration[OBJECT_DETECTOR_CLOUD_CONFIG_DICT]['cloud_storage_blob_endpoint']
        self.__video_cloud_container = self.configuration[OBJECT_DETECTOR_CLOUD_CONFIG_DICT]['video_blob_container']
        self.__images_cloud_container = self.configuration[OBJECT_DETECTOR_CLOUD_CONFIG_DICT]['images_blob_container']

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
                log.info(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                         msg=f'Cancelling task {len(tasks)} tasks...')
                [task.cancel() for task in tasks]
                log.info(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                         msg=f'Gaterhing out put of cancellation '
                             f'of {len(tasks)} tasks...')
                out_list = await asyncio.gather(*tasks, loop=self.eventLoop, return_exceptions=True)
                for idx, out in enumerate(out_list):
                    if isinstance(out, Exception):
                        log.error(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
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
                    log.error(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                              msg=f'Unable to stop MQTT client.')
                else:
                    log.info(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                             msg=f'Stopped MQTT client.')
            except:
                log.error(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                          msg=f'Exception in shutting down MQTT')

            # Stop pool manager thread first
            event = self.started_threads[self.object_detection_pool_manager_thread]
            if self.object_detection_pool_manager_thread.is_alive():
                event.set()
                await asyncio.sleep(0.5)
                if self.object_detection_pool_manager_thread.is_alive():
                    log.error(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                              msg=f'Problem shutting down '
                                  f'pool manager thread!')
            self.started_threads.pop(self.object_detection_pool_manager_thread)

            # Stop object watchers
            try:
                log.info(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                         msg=f'Stopping object watchers...')
                self.remove_tracked_objects(all_objects=True)
                log.info(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                         msg=f'Object watchers stopped.')
            except:
                log.info(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                         msg=f'Exception in shutting down object watchers')

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
                log.error(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                          msg=f'Exception in shutting down some '
                              f'of the remaining threads')

            # Stop devices
            try:
                self.k4a_device.disconnect()
                log.info(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                         msg=f'Disconnected K4A camera')
            except:
                log.error(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                          msg=f'Exception shutting down K4A')

        except:
            log.error(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                      msg=f'Problem in graceful_shutdown')
        finally:
            self.eventLoop.stop()
            log.warning(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                        msg=f'Done!')

    async def run(self) -> None:
        """
        params:
        Launches the runner that runs most things (MQTT queue, etc.)
        """
        try:
            # log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
            #             msg=f'Initializing Kinect')
            # self.k4a_device.connect()
            # log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
            #             msg=f'K4A device initialized...')

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
            self.started_threads[self.object_detection_pool_manager_thread] = \
                exit_detection_pool_manager_thread_event

        except SystemExit:
            # raise the exception up the stack
            raise Exception(f'Error : {traceback.print_exc()}')

        except K4AException:
            log.error(LOGGER_OBJECT_DETECTOR_RUNNER,
                      f'Error with K4A : {traceback.print_exc()}')
            raise K4AException(f'Issue with K4A - Need to stop.')

        except Exception:
            log.error(LOGGER_OBJECT_DETECTOR_RUNNER,
                      f'Error : {traceback.print_exc()}')

    def event_loop_start_main_tasks(self):
        """
        Main event asyncio eventloop launched in a separate thread
        """
        try:
            # region Create Async Tasks
            log.info(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                     msg=f'Launching asyncio TASK :"process MQTT message"')
            self.async_process_mqtt_messages_task = \
                self.eventLoop.create_task(
                    self.async_process_mqtt_messages(loopDelay=0.25))

            log.info(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                     msg=f'Launching asyncio TASK : "async_run_detection"')
            self.async_run_detection_task = \
                self.eventLoop.create_task(
                    self.async_run_detection())

            log.info(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                     msg=f'Launching asyncio TASK : "async_display_video"')
            self.async_run_detection_task = \
                self.eventLoop.create_task(
                    self.async_display_video())
            # endregion

            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                        msg=f'Asyncio tasks started')

        except Exception:
            log.error(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                      f'Error : {traceback.print_exc()}')
            raise Exception(f'Error : {traceback.print_exc()}')
        finally:
            log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_LOOP,
                        msg=f'Exiting event_loop_start_main_tasks')

    def thread_mqtt_listener(self):
        """
        MQTT Thread launching the loop and subscripbing to the right topics
        """
        mqtt_default_qos = 2
        self.mqtt_topics = [(topic, mqtt_default_qos)
                            for topic in self.configuration['mqtt']['subscribedTopics']]

        def on_connect(client, userdata, flags, rc):
            log.info(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                     msg=f'Connected to MQTT broker. Result code {str(rc)}')
            mqtt_connect_result, self.mqtt_connect_mid = client.subscribe(
                self.mqtt_topics)
            if mqtt_connect_result == mqtt.MQTT_ERR_SUCCESS:
                log.warning(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                            msg=f'Successfully subscribed to topics in input config file')
                log.debug(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                          msg=f'Topics subcribed = {self.mqtt_topics}')
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
            log.warning(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
                        f'Disconnected MQTT result code = {rc}. '
                        f'Should automatically re-connect to broker')

        def on_subscribe(client, userdata, mid, granted_qos):
            if mid == self.mqtt_connect_mid:
                log.debug(LOGGER_OBJECT_DETECTOR_MQTT_LOOP,
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
                                    log.info(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                                             msg=f'Setting attribute self.{k} to value {v}')
                                    # Adding / changing configuration parameters for the object
                                    self.__setattr__(k, v)
                                    log.warning(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                                                msg=f'After validation, attribute self.{k} '
                                                    f'= "{self.__getattribute__(k)}"')
                                else:
                                    log.error(LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT,
                                              msg=f'Attribute self.{k} not found. Will not add it.')
                        elif currentMQTTMoveMessage.topic == 'bot/logger':
                            # Changing the logging level on the fly...
                            log.setLevel(msgdict['logger'], lvl=msgdict['level'])
                        elif currentMQTTMoveMessage.topic == 'bot/logger/multiple':
                            # Changing the logging level on the fly for multiple loggers at a time
                            for logger, level in msgdict.items():
                                log.setLevel(logger, level)
                        elif currentMQTTMoveMessage.topic == 'bot/jetson/start_video':
                            if 'duration' in msgdict:
                                duration = float(msgdict['duration'])
                            else:
                                duration = 5.0
                            self.eventLoop.create_task(
                                self.async_record_video(duration=duration))
                        elif currentMQTTMoveMessage.topic == 'bot/jetson/snap_picture':
                            if 'obj_class' in msgdict:
                                obj_class = float(msgdict['obj_class'])
                            else:
                                obj_class = None
                            if 'include_visualisation_data' in msgdict:
                                include_visualisation_data = msgdict['include_visualisation_data']
                            else:
                                include_visualisation_data = False
                            self.eventLoop.create_task(
                                self.async_save_picture(
                                    obj_class=obj_class,
                                    include_visualisation_data=include_visualisation_data))
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
            raise

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
            log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                        msg=f'Initializing Kinect Thread')
            k4a_config_dict = self.configuration[OBJECT_DETECTOR_CONFIG_DICT]['k4a_device']
            self.k4a_config = k4aConf(
                color_resolution=k4a_config_dict['color_resolution'],
                depth_mode=k4a_config_dict['depth_mode'],
                camera_fps=k4a_config_dict['camera_fps'],
                synchronized_images_only=k4a_config_dict['synchronized_images_only'])
            if k4a_config_dict['camera_fps'] == FPS.FPS_5:
                self.frame_duration = 1. / 5
                self.fps = 5
            elif k4a_config_dict['camera_fps'] == FPS.FPS_15:
                self.frame_duration = 1. / 15
                self.fps = 15
            elif k4a_config_dict['camera_fps'] == FPS.FPS_30:
                self.frame_duration = 1. / 30
                self.fps = 30
            else:
                raise Exception('Unsupported frame rate {}'.format(
                                k4a_config_dict['camera_fps']))
        except:
            raise
        while not exit_thread_event.is_set():
            try:
                self.k4a_device = PyK4A(self.k4a_config)
                self.k4a_device.connect()
                log.warning(LOGGER_OBJECT_DETECTOR_RUNNER,
                            msg=f'K4A device initialized.')
                self.k4a_device_calibration = Calibration(
                    device=self.k4a_device,
                    config=self.k4a_config,
                    source_calibration=CalibrationType.COLOR,
                    target_calibration=CalibrationType.GYRO)

                # Stats counters for the loop
                n_loops = 0
                logging_loops = 50
                average_duration = 0
                k4a_errors = 0

                # Launch the capture loop
                while not exit_thread_event.is_set():
                    start_time = time.time()
                    # Read frame from camera
                    try:
                        self.bgra_image_color_np, image_depth_np = \
                            self.k4a_device.get_capture(
                                color_only=False,
                                transform_depth_to_color=True)
                        self.rgb_image_color_np = self.bgra_image_color_np[:, :, :3][..., ::-1]
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
                        # retrieve and update distance to each tracked object
                        with self.lock_tracked_objects_mp:
                            self.__get_distance_from_k4a()
                        # Visualization of the results of a detection.
                        img = self.bgra_image_color_np[:, :, :3]
                        with self.lock_tracked_objects_mp:
                            img = self.__update_image_with_info(img)
                        resized_im = cv2.resize(img, self.display_image_resolution)
                        self.resized_im_for_video = resized_im.copy()
                        self.image_depth_np = image_depth_np.copy()
                        duration = time.time() - start_time
                        average_duration += duration
                        n_loops += 1
                        if n_loops % logging_loops == 0:
                            duration_50 = average_duration
                            average_duration /= 50
                            log.debug(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                                      msg=f'Ran 50 in {duration_50:.2f}s - '
                                          f'{average_duration:.2f}s/loop '
                                          f'or {1/average_duration:.2f} '
                                          f'loop/sec')
                            average_duration = 0
                    except K4AException as err:
                        # count problematic frame capture
                        k4a_errors += 1
                        log.critical(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                                     msg=f'Error count: {k4a_errors} - '
                                         f'traceback={traceback.print_exc()} '
                                         f'Err = {err}')
                        if k4a_errors > 5:
                            raise K4AException
            except K4AException as err:
                try:
                    log.critical(LOGGER_ASYNC_RUN_CAPTURE_LOOP,
                                 msg=f'Error count too high ({k4a_errors}) '
                                     f'Trying to disconnect device:{err}')
                    self.k4a_device.disconnect()
                    log.critical(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                                 msg=f'Disconnected K4A camera - will try '
                                     f'reinitializing it now.')
                    k4a_errors = 0
                except K4AException:
                    log.error(LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN,
                              msg=f'Unable to disconnect K4A')
                    raise K4AException
            except Exception:
                raise

    async def async_display_video(self) -> None:
        """
        Description:
            Task to display video - used primarily for debugging purposes,
                when there is a screen connected to the device.
        """
        try:
            log.warning(LOGGER_OBJECT_DETECTION_ASYNC_DISPLAY_VIDEO,
                        msg=f'Launching display video background '
                            f'task.')
            prev_show_video = False
            prev_show_depth_video = False
            while True:
                start_time = time.time()
                if self.show_video:
                    log.debug(LOGGER_OBJECT_DETECTION_ASYNC_DISPLAY_VIDEO,
                              msg=f'showing: show_video... shape of image is: {self.resized_im_for_video.shape}')
                    cv2.imshow('show_video', self.resized_im_for_video)
                    cv2.waitKey(1)
                    prev_show_video = True
                if not self.show_video and prev_show_video:
                    log.debug(LOGGER_OBJECT_DETECTION_ASYNC_DISPLAY_VIDEO,
                              msg=f'IN SHOW_VIDEO - DESTROY_VIDEO')
                    cv2.destroyWindow('show_video')
                    prev_show_video = False

                if self.show_depth_video:
                    log.debug(LOGGER_OBJECT_DETECTION_ASYNC_DISPLAY_VIDEO,
                              msg=f'showing: show_depth_video...')
                    resized_depth_im = cv2.resize(
                        self.image_depth_np, self.display_image_resolution)
                    cv2.imshow('show_depth_video', resized_depth_im)
                    cv2.waitKey(1)
                    prev_show_depth_video = True
                if not self.show_depth_video and prev_show_depth_video:
                    log.debug(LOGGER_OBJECT_DETECTION_ASYNC_DISPLAY_VIDEO,
                              msg=f'IN SHOW_DEPTH_VIDEO - DESTROY_VIDEO')
                    cv2.destroyWindow('show_depth_video')
                    prev_show_depth_video = False
                duration = time.time() - start_time
                sleep_time = max(0, self.frame_duration - duration)
                log.debug(LOGGER_OBJECT_DETECTION_ASYNC_DISPLAY_VIDEO,
                          msg=f'sleep_time = {sleep_time:.4f}s - show_video = {self.show_video}, prev_show_video = {prev_show_video}')
                await asyncio.sleep(sleep_time)
        except asyncio.futures.CancelledError:
            log.warning(LOGGER_OBJECT_DETECTION_ASYNC_DISPLAY_VIDEO,
                        msg=f'Cancelled the display video task.')
        except Exception:
            log.error(LOGGER_OBJECT_DETECTION_ASYNC_DISPLAY_VIDEO,
                      msg=f'Problem in async_display_video : '
                          f'{traceback.print_exc()}')

    async def async_record_video(self,
                                 duration) -> None:
        """
        Description : Starts the recording of a video for duration seconds
            Will create a temporary file and send it to cloud storage
            eventually.
        Args:
            duration : float, represents the duration of time for which
                to enable the recording
        """
        keep_file_time = 60
        try:
            start_time = time.time()
            # Create temp dir
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Create temp file
                local_file_path = os.path.join(
                    tmpdirname,
                    next(tempfile._get_candidate_names()) + self.__video_file_extension)
                log.warning(LOGGER_OBJECT_DETECTION_ASYNC_RECORD_VIDEO,
                            msg=f'Created file {local_file_path} to store video. '
                                f'Res = {self.display_image_resolution}')
                try:
                    video_writer = cv2.VideoWriter(
                        local_file_path, self._fourcc,
                        float(self.fps), self.display_image_resolution)
                    # wait for duration seconds for recording to take place
                    log.warning(LOGGER_OBJECT_DETECTION_ASYNC_RECORD_VIDEO,
                                msg=f'Recording video to {local_file_path} now.')
                    while time.time() - start_time < duration:
                        loop_start = time.time()
                        video_writer.write(self.resized_im_for_video)
                        duration = time.time() - loop_start
                        sleep_time = max(0, self.frame_duration - duration)
                        await asyncio.sleep(sleep_time)
                except Exception:
                    raise Exception()
                finally:
                    video_writer.release()
                    log.debug(LOGGER_OBJECT_DETECTION_ASYNC_RECORD_VIDEO,
                              msg=f'Recording completed, released video writer.')
                path = shutil.copy(local_file_path, os.path.join(os.getcwd()))
                try:
                    filename = local_file_path.split('/')[-1]
                    target_filename = os.path.join(
                        datetime.datetime.now().strftime("%Y/%m/%d/%H"),
                        filename)
                    await self.async_upload_file_to_azure(
                        container_name=self.__video_cloud_container,
                        local_file_path=filename,
                        target_blob_filename=target_filename,
                        content_type=self.__video_content_type)
                    log.info(LOGGER_OBJECT_DETECTION_ASYNC_RECORD_VIDEO,
                             msg=f'Recording locally available in {path} '
                                 f'for {keep_file_time} seconds')
                    await asyncio.sleep(keep_file_time)
                    os.remove(path)
                    log.info(LOGGER_OBJECT_DETECTION_ASYNC_RECORD_VIDEO,
                             msg=f'Deleted {path}.')
                except asyncio.futures.CancelledError:
                    log.warning(LOGGER_OBJECT_DETECTION_ASYNC_RECORD_VIDEO,
                                msg=f'Cancelled the video recording at pause time.')
                    raise asyncio.futures.CancelledError()
                except:
                    log.error(LOGGER_OBJECT_DETECTION_ASYNC_RECORD_VIDEO,
                              msg=f'Uncaught Exception. Details = '
                                  f'{traceback.print_exc()}')
        except asyncio.futures.CancelledError:
            log.warning(LOGGER_OBJECT_DETECTION_ASYNC_RECORD_VIDEO,
                        msg=f'Cancelled the video recording task.')
        except:
            log.error(LOGGER_OBJECT_DETECTION_ASYNC_RECORD_VIDEO,
                      msg=f'Error in async_record_video : '
                          f'{traceback.print_exc()}')
            raise Exception(f'Error in async_record_video : '
                            f'{traceback.print_exc()}')

    async def async_save_picture(self,
                                 obj_class=None,
                                 include_visualisation_data=False) -> None:
        """
        Description : will save a picture and upload it to cloud. Will
            delete temporary file for picture.
        Args:
            obj_class : if set, will automatically upload file to the
                proper remote directory structure for retraining
            include_visualisation_data : boolean, will show distance, object
                class, etc., if set to true. If not, it will be the raw image.
        """
        keep_file_time = 60
        try:
            start_time = time.time()
            # Create temp dir
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Create temp file
                local_file_path = os.path.join(
                    tmpdirname,
                    next(tempfile._get_candidate_names()) + self.__image_file_extension)
                log.warning(LOGGER_OBJECT_DETECTION_ASYNC_SAVE_PICTURE,
                            msg=f'Created file {local_file_path} to store temp '
                                f'picture. Res = '
                                f'{self.display_image_resolution}')
                try:
                    log.warning(LOGGER_OBJECT_DETECTION_ASYNC_SAVE_PICTURE,
                                msg=f'Saving picture to {local_file_path}.')
                    if include_visualisation_data:
                        img = self.resized_im_for_video
                    else:
                        img = self.bgra_image_color_np[:, :, :3]
                    cv2.imwrite(local_file_path, img)
                    # image_to_save = Image.fromarray(self.rgb_image_color_np)
                    # image_to_save.save(local_file_path)
                except Exception:
                    raise Exception()
                finally:
                    log.debug(LOGGER_OBJECT_DETECTION_ASYNC_SAVE_PICTURE,
                              msg=f'Picture saved.')
                path = shutil.copy(local_file_path, os.path.join(os.getcwd()))
                try:
                    filename = local_file_path.split('/')[-1]
                    target_filename = os.path.join(
                        datetime.datetime.now().strftime("%Y/%m/%d/%H"),
                        filename)
                    await self.async_upload_file_to_azure(
                        container_name=self.__images_cloud_container,
                        local_file_path=filename,
                        target_blob_filename=target_filename,
                        content_type=self.__image_content_type)
                    log.info(LOGGER_OBJECT_DETECTION_ASYNC_SAVE_PICTURE,
                             msg=f'Picture locally available in {path} '
                                 f'for {keep_file_time} seconds')
                    await asyncio.sleep(keep_file_time)
                    os.remove(path)
                    log.info(LOGGER_OBJECT_DETECTION_ASYNC_SAVE_PICTURE,
                             msg=f'Deleted {path}.')
                except asyncio.futures.CancelledError:
                    log.warning(LOGGER_OBJECT_DETECTION_ASYNC_SAVE_PICTURE,
                                msg=f'Cancelled the picture saving during '
                                    f'pause. Will need to clean up.')
                    raise asyncio.futures.CancelledError()
                except:
                    log.error(LOGGER_OBJECT_DETECTION_ASYNC_SAVE_PICTURE,
                              msg=f'Uncaught Exception. Details = '
                                  f'{traceback.print_exc()}')
            duration = time.time() - start_time
            log.debug(LOGGER_OBJECT_DETECTION_ASYNC_SAVE_PICTURE,
                      msg=f'Snapping and upload of picture took {duration:.3f}s')
        except asyncio.futures.CancelledError:
            log.warning(LOGGER_OBJECT_DETECTION_ASYNC_SAVE_PICTURE,
                        msg=f'Cancelled the picture saving task.')
        except:
            log.error(LOGGER_OBJECT_DETECTION_ASYNC_SAVE_PICTURE,
                      msg=f'Error in async_save_picture : '
                          f'{traceback.print_exc()}')
            raise Exception(f'Error in async_save_picture : '
                            f'{traceback.print_exc()}')

    async def async_upload_file_to_azure(self,
                                         container_name: str,
                                         local_file_path: str,
                                         target_blob_filename: str,
                                         content_type: str) -> None:
        """
        Description :
            Sends file to cloud. Files could be video, or photos, or others
        Args:
            container_name : str, container name to be created in Azure
            local_file_path : str, absolute path of object to be uploaded
            target_blob_filename : str, target of the blob on Azure. / will be
                intepreted as a directory.
            content_type: str, mimetype, of file being uploaded, video/x-msvideo
        Returns :
            Sucess or None if failed.
        """
        try:
            metadata = {
                'create_time': datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S.%f"),
                'type': 'video_file_container'
            }
            blob_service_client = BlobServiceClient(
                account_url=self.__blob_service_endpoint,
                credential=self.__azure_credentials)
            log.debug(
                LOGGER_OBJECT_DETECTION_ASYNC_UPLOAD_FILE_TO_AZURE,
                msg=f'Blob service client created, attempting to '
                    f'create the container.')
            async with blob_service_client:
                container_client = blob_service_client.get_container_client(
                    container_name)
                assert container_client, f'container_client ' \
                                         f'is set to {container_client}'
                log.debug(
                    LOGGER_OBJECT_DETECTION_ASYNC_UPLOAD_FILE_TO_AZURE,
                    msg=f'Container client created = {container_client}.')
                try:
                    log.debug(
                        LOGGER_OBJECT_DETECTION_ASYNC_UPLOAD_FILE_TO_AZURE,
                        msg=f'About to await creation of container '
                            f'{container_name}.')
                    await container_client.create_container(metadata=metadata)
                except ResourceExistsError:
                    log.info(
                        LOGGER_OBJECT_DETECTION_ASYNC_UPLOAD_FILE_TO_AZURE,
                        msg=f'Blob container {container_name} already '
                            f'exists on storage account '
                            f'{self.__blob_service_endpoint}')
                except Exception:
                    log.error(
                        LOGGER_OBJECT_DETECTION_ASYNC_UPLOAD_FILE_TO_AZURE,
                        msg=f'Error creating blob container : '
                            f'{traceback.print_exc()}')
                else:
                    log.warning(
                        LOGGER_OBJECT_DETECTION_ASYNC_UPLOAD_FILE_TO_AZURE,
                        msg=f'Container {container_name} created.')
                # filename = local_file_path.split('/')[-1]
                # target_filename = os.path.join(
                #     datetime.datetime.now().strftime("%Y/%m/%d/%H"),
                #     filename)
                blob_client = container_client.get_blob_client(target_blob_filename)
                log.warning(LOGGER_OBJECT_DETECTION_ASYNC_UPLOAD_FILE_TO_AZURE,
                            msg=f'Uploading local file {local_file_path} to '
                                f'blob {target_blob_filename} in container '
                                f'{container_name}.')
                try:
                    metadata = {
                        'create_time': datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S.%f"),
                        'type': 'content_type'
                    }
                    with open(local_file_path, 'rb') as data:
                        await blob_client.upload_blob(
                            data,
                            blob_type='BlockBlob',
                            metadata=metadata,
                            content_settings=ContentSettings(
                                content_type=content_type))
                except Exception:
                    log.error(LOGGER_OBJECT_DETECTION_ASYNC_UPLOAD_FILE_TO_AZURE,
                              msg=f'Error uploading blob {traceback.print_exc()}')
                log.warning(LOGGER_OBJECT_DETECTION_ASYNC_UPLOAD_FILE_TO_AZURE,
                            msg=f'Upload completed.')
        except asyncio.futures.CancelledError:
            log.warning(LOGGER_OBJECT_DETECTION_ASYNC_UPLOAD_FILE_TO_AZURE,
                        msg=f'Cancelled the video uploading task.')
        except Exception:
            log.error(LOGGER_OBJECT_DETECTION_ASYNC_UPLOAD_FILE_TO_AZURE,
                      msg=f'Error uploading file to azure : '
                          f'{traceback.print_exc()}')
            raise Exception(f'Error uploading file to azure :'
                            f'{traceback.print_exc()}')

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
            back_off_retry = 0
            # Only launch loop when ready (image on the self.rgb...)
            itm = self.ready_queue.get(block=True)
            if itm is None:
                raise Exception('Empty Queue...')
            # Launch the loop
            while True:
                start_time = time.time()
                height, width = self.rgb_image_color_np_resized.shape[:2]
                # Send image to bytes buffer
                im = Image.fromarray(self.rgb_image_color_np_resized)
                buf = io.BytesIO()
                im.save(buf, format='PNG')
                base64_encoded_image = base64.b64encode(buf.getvalue())
                payload = {'image': base64_encoded_image.decode('ascii')}
                start_time = time.time()
                async with ClientSession() as session:
                    try:
                        async with session.post(
                                url=model_url,
                                data=json.dumps(payload),
                                headers=headers) as response:
                            end_time = time.time()
                            if response.status == 200:
                                body = await response.json()
                                log.debug(LOGGER_ASYNC_RUN_DETECTION,
                                          msg=f'Got model response... iteration {n_loops}')
                                task_start_time = time.time()
                                # Post results to queue to be consumed by
                                # thread_object_detection_pool_manager
                                try:
                                    self.ready_for_first_detection_event.set()
                                    self.object_detection_result_queue.put(
                                        item=(body['boxes'],
                                              body['scores'],
                                              body['classes']),
                                        block=True,
                                        timeout=2)
                                except queue.Full:
                                    log.critical(LOGGER_ASYNC_RUN_DETECTION,
                                                 msg=f'OD Queue full. Pls investigate '
                                                     f'refresh_tracked_objects. Is it running, '
                                                     f'blocked or canceled?')
                                    # raise Exception('OD Queue full, Pls investigate')
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
                    except ClientConnectorError as err:
                        if back_off_retry + 5 <= 60:
                            back_off_retry += 5
                        log.warning(LOGGER_ASYNC_RUN_DETECTION,
                                    msg=f'Cannot connect to http://{err.host}'
                                        f':{err.port}. Retrying in '
                                        f'{back_off_retry} secs.')
                        await asyncio.sleep(back_off_retry)
                    except ConnectionRefusedError:
                        if back_off_retry + 5 <= 60:
                            back_off_retry += 5
                        log.warning(LOGGER_ASYNC_RUN_DETECTION,
                                    msg=f'Connection to {model_url} refused. '
                                        f'Retying in {back_off_retry} '
                                        f'seconds')
                        await asyncio.sleep(back_off_retry)
                    except ServerDisconnectedError:
                        if back_off_retry + 5 <= 60:
                            back_off_retry += 5
                        log.error(LOGGER_ASYNC_RUN_DETECTION,
                                  msg=f'HTTP prediction service error: '
                                      f'{traceback.print_exc()} -> sleeping '
                                      f'{self.time_between_scoring_service_calls}s '
                                      f'and continuing')
                        await asyncio.sleep(back_off_retry)
                    except Exception:
                        raise
                    else:
                        back_off_retry = 0
                # Pause for loopDelay seconds
                duration = time.time() - start_time
                sleep_time = max(0, self.time_between_scoring_service_calls - duration)
                await asyncio.sleep(sleep_time)
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
                During its processing, self.tracked_objects_mp would
                look like this:
                    {
                        object_uuid_1 : {tracked_object, process_id, thread_id, },
                        object_uuid_2 : {tracked_object, process_id, thread_id, }
                    }
        Args:
            exit_thread_event : <class 'threading.Event()'> : used to signal
                process it's time to end.
            loop_delay : float : delay in the while true loop to ensure no high
                cpu usage for nothing.
        """
        object_counter = 0
        while not exit_detection_pool_manager_thread_event.is_set():
            try:
                new_objects_dict, deleted_objects_dict = \
                    self.refresh_tracked_objects(
                        image_shape=self.resized_image_resolution)
                # Create proc/thread for new items
                for (new_object_id, new_object) in \
                        new_objects_dict.items():
                    new_tf_info_queue = mp.Manager().Queue(maxsize=1)
                    tracking_object_queue = mp.Manager().Queue(maxsize=1)
                    exit_thread_event = threading.Event()
                    proc = ObjectTrackingProcess(tracked_object=new_object,
                                                 tracker_alg=self.default_tracker,
                                                 initial_image=self.rgb_image_color_np_resized,
                                                 frame_duration=self.frame_duration,
                                                 image_queue=self.image_queue.register(name=str(new_object_id)),
                                                 new_tf_info_queue=new_tf_info_queue,
                                                 tracking_object_queue=tracking_object_queue)
                    thread = threading.Thread(target=self.thread_poll_object_tracking_process_queue,
                                              args=([new_object,
                                                     tracking_object_queue,
                                                     exit_thread_event]),
                                              name=f'ObjTrack_{str(new_object_id)[:8]}')
                    # Keep track of what is under observation
                    self.tracked_objects_mp[new_object_id] = {
                        'tracked_object': new_object,
                        'object_counter': object_counter,
                        'tracking_object_queue': tracking_object_queue,
                        'new_tf_info_queue': new_tf_info_queue,
                        'proc': proc,
                        'thread': thread,
                        'exit_thread_event': exit_thread_event,
                        'flag_for_kill': False
                    }
                    log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                msg=f'Creating proc/thread mapping for new object {new_object_id[:8]}')
                    object_counter += 1
                    proc.start()
                    log.info(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                             msg=f'Manager launched process {proc.pid} for object {new_object_id[:8]}.')
                    thread.name += f'_pid{proc.pid}'
                    thread.start()
                    log.info(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                             msg=f'Manager launched thread for object {new_object_id[:8]}.')
                    new_object.monitored = True
                # Stop proc/thread for deleted items
                for (deleted_object_id, deleted_object) in \
                        deleted_objects_dict.items():
                    log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                msg=f'Marking proc/thread mapping of object {deleted_object_id[:8]} for deletion')
                    self.tracked_objects_mp[deleted_object_id]['flag_for_kill'] = True
                    self.tracked_objects_mp[deleted_object_id]['reason'] = UNDETECTED
                # Check tracked_objects sanity, flag for kill if required
                for (obj_id, tracked_object_mapping_dict) in \
                        self.tracked_objects_mp.items():
                    # Check if object related process/thread should be purged (unseen for 5 seconds)
                    proc = tracked_object_mapping_dict['proc']
                    thread = tracked_object_mapping_dict['thread']
                    tracked_object_mp = tracked_object_mapping_dict['tracked_object']
                    # If object unseen for X seconds:
                    unseen_time = time.time() - tracked_object_mp.last_seen
                    if unseen_time > self.max_unseen_time_for_object:
                        tracked_object_mapping_dict['flag_for_kill'] = True
                        tracked_object_mapping_dict['reason'] = UNSEEN
                        log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                    msg=f'Marking proc/thread mapping of object {obj_id[:8]} '
                                        f'for deletion, unseen for {unseen_time:.2f}s')
                    # verify if object proc and thread are healthy - if thread or proc is dead, kill tracker
                    elif not (proc.is_alive() and thread.is_alive()):
                        tracked_object_mapping_dict['flag_for_kill'] = True
                        tracked_object_mapping_dict['reason'] = UNSTABLE
                        log.warning(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                                    msg=f'Marking proc/thread mapping of object {obj_id[:8]} '
                                        f'for deletion, proc or thread is stopped, '
                                        f'need to get back to stable state')
                # Call function that kills the threads/procs that need cleanup
                nb_obj_to_delete = len({k: v for (k, v) in self.tracked_objects_mp.items() if v['flag_for_kill']})
                if nb_obj_to_delete > 0:
                    objs_to_remove_from_dict = self.remove_tracked_objects()
                    with self.lock_tracked_objects_mp:
                        for obj in objs_to_remove_from_dict:
                            self.tracked_objects_mp.pop(obj)
                time.sleep(loop_delay)
                log.info(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                         msg=f'Currently monitoring {len(self.tracked_objects_mp)} objects')
            except:
                raise Exception(f'Problem : {traceback.print_exc()}')
        log.critical(LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER,
                     msg=f'Exiting thread "thread_object_detection_pool_manager". '
                         f'Should not happen unless stopping robot.')

    def remove_tracked_objects(self,
                               all_objects=False) -> None:   # , tracked_objects):
        """
        Description:
            Function used to garbage collect (remove unused threads and
                processes) of tracked objects that dissapeared. The
                self.tracked_objects_mp[tracked_object_mp_id]['flag_for_kill']
                flag will be set to True for those who need to dissapear.
        Args:
            all_objects : bool, if set to True, will stop all threads and
                processes.
        Returns : [obj_ids] : list of string UUIDs to be removed by caller
        """
        try:
            list_uuids = []
            # Extract list of objects to be flagged for removal
            if all_objects:
                to_be_killed_tracked_objects = \
                    {k: v for k, v in self.tracked_objects_mp.items()}
            else:
                to_be_killed_tracked_objects = \
                    {k: v for k, v in self.tracked_objects_mp.items() if v['flag_for_kill']}
            for (obj_id, tracked_object_dict) in to_be_killed_tracked_objects.items():
                proc = tracked_object_dict['proc']
                thread = tracked_object_dict['thread']
                exit_thread_event = tracked_object_dict['exit_thread_event']
                # tracking_object_queue = tracked_object_dict['tracking_object_queue']

                # Signal thread it's time to stop (use a threading.event?)
                exit_thread_event.set()
                while thread.is_alive():
                    thread.join(2)  # Wait up to X sec
                    if thread.is_alive():
                        log.warning(LOGGER_OBJECT_DETECTION_KILL,
                                    msg=f'Failed to cancel thread for object {obj_id}.')
                log.warning(LOGGER_OBJECT_DETECTION_KILL,
                            msg=f'Thread for object {obj_id[:8]} is terminated.')

                # Signal process it's time to stop
                proc.exit.set()
                while proc.is_alive():
                    log.warning(LOGGER_OBJECT_DETECTION_KILL,
                                msg=f'Attempting to cancel PID {proc.pid} for '
                                    f'object {obj_id[:8]}.')
                    proc.join(2)  # Wait up to X sec
                    exitcode = proc.exitcode
                    log.warning(LOGGER_OBJECT_DETECTION_KILL,
                                msg=f'PID {proc.pid}\'s exit code is {exitcode} '
                                    f'for {obj_id[:8]}.')
                log.warning(LOGGER_OBJECT_DETECTION_KILL,
                            msg=f'PID {proc.pid} for object '
                                f'{obj_id[:8]} is terminated.')

                # Remove from dict altogether
                log.warning(LOGGER_OBJECT_DETECTION_KILL,
                            msg=f'Adding object {obj_id} to removal list (for mapping dictionnary).')
                list_uuids.append(obj_id)

                # Unregister specific image_queue from PublisQueue
                self.image_queue.unregister(name=str(obj_id))

        except KeyError:
            raise Exception(f'Problem : tried removing tracked object from '
                            f'dictionnary but key not found. Traceback = '
                            f'{traceback.print_exc()}')
        except:
            raise Exception(f'Problem : {traceback.print_exc()}')
        finally:
            return list_uuids

    def thread_poll_object_tracking_process_queue(self,
                                                  tracked_object_mp: TrackedObjectMP,
                                                  tracked_object_queue: mp.Queue,
                                                  exit_thread_event: threading.Event):
        """
        Description: thread to monitor objects and retrieve updated bounding box
            coordinates - these are calculated in the associated process. Need
            to run one per tracked object.
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
                        msg=f'Launching thread for object ID '
                            f'{str(tracked_object_mp.id)[:8]}')
            while not exit_thread_event.is_set():
                # Replace object in tracked object with what we get from the queue
                # wait max 2 second
                new_tracked_object_mp = tracked_object_queue.get(
                    block=True, timeout=2)
                obj_id = str(new_tracked_object_mp.id)
                # Get the new coordinates in the object.
                if obj_id in self.tracked_objects_mp.keys():
                    obj = self.tracked_objects_mp[obj_id]['tracked_object']
                    obj.set_bbox(new_tracked_object_mp.get_bbox_object())
                    obj.score = new_tracked_object_mp.score
                    obj.object_class = new_tracked_object_mp.object_class
                else:
                    # If object not there, end thread
                    log.warning(LOGGER_OBJECT_DETECTION_THREAD_POLL_OBJECT_TRACKING_PROCESS_QUEUE,
                                msg=f'Not finding object {obj_id[:8]}.. '
                                    f'in self.tracked_objects_mp.')
                    break
        except queue.Empty:
            log.warning(LOGGER_OBJECT_DETECTION_THREAD_POLL_OBJECT_TRACKING_PROCESS_QUEUE,
                        msg=f'No new object was placed on the queue, '
                            f'tracking process may have been terminated '
                            f'by parent process.')
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

    def __update_image_with_info(self,
                                 img,
                                 alpha=0.5):
        """
        Description: updates images with information to display on the video
            if shown
        Args:
            img : numpy array containing original image to display
            alpha : float, overlay transparency =defaults to 0.5
        Returns:
            modified image
        """
        try:
            height, width = img.shape[:2]
            output = img.copy()
            overlay = img.copy()
            tmp_obj_dict = {}
            # Create temp list of objects
            for (uuid, obj_dict) in self.tracked_objects_mp.items():
                tmp_obj_dict[uuid] = deepcopy(obj_dict['tracked_object'])
            for uuid, obj in tmp_obj_dict.items():
                # for uuid, obj_dict in self.tracked_objects_mp.items():
                # obj = obj_dict['tracked_object']
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

    def refresh_tracked_objects(self,
                                image_shape):
        """
        Description:
            Refresh tracked objects, that is make sure the list of tracked_objects
                by the bot is reflecting current detections and tracking outputs.
        Args:
            image_shape : tuple (height, width) in pixels
        Returns:
            new_objects and deleted_objects : 2 dictionnaries containing objects to
                create/track or delete/untrack. Format of dict is
                uuid:tracked_object_mp.
        """
        try:
            height, width = image_shape
            new_tracked_objects_mp = {}
            new_objects = {}
            deleted_objects = {}
            # Obtain from queue the latest scoring information
            self.ready_for_first_detection_event.set()
            detection_boxes, detection_scores, detection_classes = \
                self.object_detection_result_queue.get_nowait()
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
            # Find best fitting BB for each tracked object and
            # add to temp list
            temp_existing_tracked_objects_mp = {}
            for (obj_id, tracked_object_dict) in self.tracked_objects_mp.items():
                # Find best overlapping bounding box
                target_bb_idx = tracked_object_dict['tracked_object']\
                    .get_max_overlap_bb(bb_list)
                # Update dictionnary with outcome unless target_bb == None
                if target_bb_idx is not None:
                    tracked_object_dict['tracked_object'].set_bbox(
                        bb_list[target_bb_idx])
                    tracked_object_dict['tracked_object'].score = \
                        ds_list[target_bb_idx]
                    tracked_object_dict['tracked_object'].object_class = \
                        dc_list[target_bb_idx]
                    # Send updated TF info for processing on the other process
                    tracked_object_dict['new_tf_info_queue'].put((
                        bb_list[target_bb_idx],
                        ds_list[target_bb_idx],
                        dc_list[target_bb_idx]))
                    bb_list.remove(bb_list[target_bb_idx])
                    ds_list.remove(ds_list[target_bb_idx])
                    dc_list.remove(dc_list[target_bb_idx])
                # add object tuple to list
                temp_existing_tracked_objects_mp[obj_id] = \
                    tracked_object_dict['tracked_object']

            # Go through list of untapped boxes and add to temp list
            temp_new_tracked_objects_mp = {}
            # List all unused bounding boxes and create tracker
            for idx, bb in enumerate(bb_list):
                new_obj = TrackedObjectMP(
                    object_class=dc_list[idx],
                    score=ds_list[idx],
                    original_image_resolution=(height, width),
                    box=bb.get_bbox(fmt=FMT_TF_BBOX),
                    fmt=FMT_TF_BBOX)
                temp_new_tracked_objects_mp[str(new_obj.id)] = new_obj

            # Combine output and get only up to max_tracked_objects items
            combined_temp = {**temp_existing_tracked_objects_mp,
                             **temp_new_tracked_objects_mp}

            # Order outputs by scores
            ordered_combined_temp = {k: v
                                     for k, v in sorted(
                                         combined_temp.items(),
                                         key=lambda item: item[1].score,
                                         reverse=True)}

            # Keep top "self.max_tracked_objects" only
            new_tracked_objects_mp = {k: ordered_combined_temp[k]
                                      for k in list(ordered_combined_temp)
                                      [:self.max_tracked_objects]}

            # List objects that were created as warnings
            new_objects_ids = set(new_tracked_objects_mp).difference(
                set(self.tracked_objects_mp))
            new_objects = {
                k: v for k, v in new_tracked_objects_mp.items()
                if k in new_objects_ids}

            # # List objects that sere deleted as warnings
            deleted_objects_ids = set(self.tracked_objects_mp).difference(
                set(new_tracked_objects_mp))
            for obj in list(deleted_objects_ids):
                log.warning(LOGGER_OBJECT_DETECTOR_SORT_TRACKED_OBJECTS,
                            msg=f'Deleted tracker for object {obj}')
            deleted_objects = {
                k: v for k, v in self.tracked_objects_mp.items()
                if k in deleted_objects_ids}

        except queue.Empty:
            pass
        except Exception:
            raise Exception(f'Problem : {traceback.print_exc()}')
        finally:
            return new_objects, deleted_objects

    def __get_distance_from_k4a(self):
        """
        Description:
            Retrieve the distance from a specific object to the camera lens.
        Args: None
        Returns : float, distance in mm from specific point on camera
        """
        try:
            # For each object
            for (uuid, obj_dict) in self.tracked_objects_mp.items():
                # Get object coordinates in the image
                obj = obj_dict['tracked_object']
                (x, y, w, h) = obj.get_bbox(
                    fmt='tracker',
                    use_normalized_coordinates=True)
                x_res = self.resized_image_resolution[0]
                y_res = self.resized_image_resolution[1]
                x = int(x * x_res)
                y = int(y * y_res)
                w = int(w * x_res)
                h = int(h * y_res)
                cropped_depth_map = \
                    self.image_depth_np_resized[y: max(y + h, y_res - 1),
                                                x: max(x + w, x_res - 1)]
                distance = np.round(np.average(cropped_depth_map))
                obj.distance = distance
                log.debug(LOGGER_OBJECT_DETECTION_GET_DISTANCE_FROM_K4A,
                          msg=f'****** DIST CALCULATED {uuid[:8]} = {obj.distance}')
                center_point = [max(int(x + w / 2), x_res - 1),
                                max(int(y + h / 2), y_res - 1)]
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

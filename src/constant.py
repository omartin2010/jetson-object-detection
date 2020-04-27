import pyk4a
import cv2
# region LOGGERS
LOGGER_OBJECT_DETECTOR_STARTUP = 'startup'
LOGGER_OBJECT_DETECTOR_ERROR_HANDLER = 'err_handler'
LOGGER_OBJECT_DETECTOR_MAIN = 'object_py'
LOGGER_OBJECT_DETECTOR_LOAD_MODEL = 'object_py_load_model'
LOGGER_ASYNC_RUN_DETECTION = 'run_detection'
LOGGER_ASYNC_RUN_CAPTURE_LOOP = 'async_run_capture_loop'
LOGGER_OBJECT_DETECTOR_ASYNC_RUN_DETECTION = 'async_run_detection'
LOGGER_OBJECT_DETECTOR_MQTT_LOOP = 'mqtt_loop'
LOGGER_OBJECT_DETECTOR_RUNNER = 'runner'
LOGGER_OBJECT_DETECTOR_ASYNC_LOOP = 'async_loop'
LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT = 'async_process_mqtt_messages'
LOGGER_OBJECT_DETECTOR_KILL_SWITCH = 'kill_sw'
LOGGER_OBJECT_DETECTOR_SORT_TRACKED_OBJECTS = 'sort_tracked_objects'
LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT = 'process_track_opencv_object'
LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER = 'thread_object_detection_pool_manager'
LOGGER_OBJECT_DETECTION_THREAD_POLL_OBJECT_TRACKING_PROCESS_QUEUE = 'thread_poll_object_tracking_process_queue'
LOGGER_OBJECT_DETECTION_KILL = 'thread_poll_kill'
LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN = 'soft_shutdown'
LOGGER_OBJECT_DETECTION_UPDATE_IMG_WITH_INFO = '__update_image_with_info'
LOGGER_OBJECT_DETECTION_ASYNC_RECORD_VIDEO = 'async_record_video'
LOGGER_OBJECT_DETECTION_ASYNC_SAVE_PICTURE = 'async_save_picture'
LOGGER_OBJECT_DETECTION_TRACKED_OBJECT_PROPERTY_UPDATE = 'tracked_object_property'
LOGGER_OBJECT_DETECTION_GET_DISTANCE_FROM_K4A = '__get_distance_from_k4a'
LOGGER_OBJECT_DETECTION_PUBLISH_QUEUES_UNREGISTER = 'publish_queue_unregister'
LOGGER_OBJECT_DETECTION_PUBLISH_QUEUES_REGISTER = 'publish_queue_register'
LOGGER_OBJECT_DETECTION_ASYNC_DISPLAY_VIDEO = 'async_display_video'
LOGGER_OBJECT_DETECTION_ASYNC_UPLOAD_FILE_TO_AZURE = 'async_upload_file_to_azure'
LOGGER_OBJECT_DETECTION_ASYNC_WATCHDOG = 'async_watchdog'
LOGGER_OBJECT_DETECTION_ASYNC_STREAM_VIDEO = 'async_stream_video'
# endregion

K4A_DEFINITIONS = {
    "K4A_COLOR_RESOLUTION_720P": pyk4a.ColorResolution.RES_720P,
    "K4A_COLOR_RESOLUTION_1080P": pyk4a.ColorResolution.RES_1080P,
    "K4A_COLOR_RESOLUTION_1440P": pyk4a.ColorResolution.RES_1440P,
    "K4A_COLOR_RESOLUTION_1536P": pyk4a.ColorResolution.RES_1536P,
    "K4A_COLOR_RESOLUTION_2160P": pyk4a.ColorResolution.RES_2160P,
    "K4A_COLOR_RESOLUTION_3072P": pyk4a.ColorResolution.RES_3072P,
    "K4A_DEPTH_MODE_NFOV_2X2BINNED": pyk4a.DepthMode.NFOV_2X2BINNED,
    "K4A_DEPTH_MODE_NFOV_UNBINNED": pyk4a.DepthMode.NFOV_UNBINNED,
    "K4A_DEPTH_MODE_WFOV_UNBINNED": pyk4a.DepthMode.WFOV_UNBINNED,
    "K4A_DEPTH_MODE_WFOV_2X2BINNED": pyk4a.DepthMode.WFOV_2X2BINNED,
    "K4A_DEPTH_MODE_OFF": pyk4a.DepthMode.OFF,
    "K4A_DEPTH_MODE_PASSIVE_IR": pyk4a.DepthMode.PASSIVE_IR,
    "K4A_FRAMES_PER_SECOND_5": pyk4a.FPS.FPS_5,
    "K4A_FRAMES_PER_SECOND_15": pyk4a.FPS.FPS_15,
    "K4A_FRAMES_PER_SECOND_30": pyk4a.FPS.FPS_30,
    "True": True,
    "False": False,
}

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

# region various constants
OBJECT_DETECTOR_CONFIG_DICT = 'object_detector'
OBJECT_DETECTOR_CLOUD_CONFIG_DICT = 'cloud_connection'

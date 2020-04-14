import pyk4a
import cv2
# region LOGGERS
LOGGER_OBJECT_DETECTOR_STARTUP = 'obj_detector_startup'
LOGGER_OBJECT_DETECTOR_MAIN = 'obj_detector_object_py'
LOGGER_OBJECT_DETECTOR_LOAD_MODEL = 'obj_detector_object_py_load_model'
LOGGER_ASYNC_RUN_DETECTION = 'obj_detector_run_detection'
LOGGER_ASYNC_RUN_CAPTURE_LOOP = 'async_run_capture_loop'
LOGGER_OBJECT_DETECTOR_ASYNC_RUN_DETECTION = 'obj_detector_async_run_detection'
LOGGER_OBJECT_DETECTOR_MQTT_LOOP = 'obj_detector_mqtt_loop'
LOGGER_OBJECT_DETECTOR_RUNNER = 'obj_detector_runner'
LOGGER_OBJECT_DETECTOR_ASYNC_LOOP = 'obj_detector_async_loop'
LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT = 'obj_detector_proc_mqtt'
LOGGER_OBJECT_DETECTOR_KILL_SWITCH = 'obj_detector_kill_sw'
LOGGER_OBJECT_DETECTOR_SORT_TRACKED_OBJECTS = 'obj_detector_sort_tracked_objects'
LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT = 'obj_detector_process_track_opencv_object'
LOGGER_OBJECT_DETECTION_OBJECT_DETECTION_POOL_MANAGER = 'obj_detector_thread_object_detection_pool_manager'
LOGGER_OBJECT_DETECTION_THREAD_POLL_OBJECT_TRACKING_PROCESS_QUEUE = 'obj_detector_thread_poll_object_tracking_process_queue'
LOGGER_OBJECT_DETECTION_KILL = 'obj_detector_thread_poll_kill'
LOGGER_OBJECT_DETECTION_SOFTSHUTDOWN = 'obj_detector_soft_shutdown'
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

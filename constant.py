import pyk4a
# region LOGGERS
LOGGER_OBJECT_DETECTOR_STARTUP = 'obj_detector_startup'
LOGGER_OBJECT_DETECTOR_MAIN = 'obj_detector_object_py'
LOGGER_OBJECT_DETECTOR_LOAD_MODEL = 'obj_detector_object_py_load_model'
LOGGER_OBJECT_DETECTOR_RUN_DETECTION = 'obj_detector_run_detection'
LOGGER_OBJECT_DETECTOR_ASYNC_RUN_DETECTION = 'obj_detector_async_run_detection'
LOGGER_OBJECT_DETECTOR_MQTT_LOOP = 'obj_detector_mqtt_loop'
LOGGER_OBJECT_DETECTOR_RUNNER = 'obj_detector_runner'
LOGGER_OBJECT_DETECTOR_ASYNC_LOOP = 'obj_detector_async_loop'
LOGGER_OBJECT_DETECTOR_ASYNC_PROCESS_MQTT = 'obj_detector_proc_mqtt'
LOGGER_OBJECT_DETECTOR_KILL_SWITCH = 'obj_detector_kill_sw'
# endregion

K4A_DEFINITIONS = {
    "K4A_IMAGE_FORMAT_COLOR_MJPG": pyk4a.ImageFormat.COLOR_MJPG,
    "K4A_IMAGE_FORMAT_COLOR_BGRA32": pyk4a.ImageFormat.COLOR_BGRA32,
    "K4A_IMAGE_FORMAT_COLOR_NV12": pyk4a.ImageFormat.COLOR_NV12,
    "K4A_IMAGE_FORMAT_COLOR_YUY2": pyk4a.ImageFormat.COLOR_YUY2,
    "K4A_IMAGE_FORMAT_CUSTOM": pyk4a.ImageFormat.CUSTOM,
    "K4A_IMAGE_FORMAT_CUSTOM16": pyk4a.ImageFormat.CUSTOM16,
    "K4A_IMAGE_FORMAT_CUSTOM8": pyk4a.ImageFormat.CUSTOM8,
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

# region various constants
OBJECT_DETECTOR_CONFIG_DICT = 'object_detector'
MAX_WORKER_THREADS = 5

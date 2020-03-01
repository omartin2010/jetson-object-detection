import pyk4a
import cv2
import numpy as np

print(pyk4a.Device.get_installed_count())
device=pyk4a.Device.open(pyk4a.K4A_DEVICE_DEFAULT)
print(device.get_serialnum())
config = pyk4a.Configuration()
config.depth_mode = pyk4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
config.color_resolution = pyk4a.K4A_COLOR_RESOLUTION_720P
config.camera_fps = pyk4a.K4A_FRAMES_PER_SECOND_30
config.color_format = pyk4a.K4A_IMAGE_FORMAT_COLOR_BGRA32
config.synchronized_images_only = True

calibration = device.get_calibration(config.depth_mode, config.color_resolution)
transformation = pyk4a.Transformation(calibration)
device.start_cameras(config)
device.start_imu()
capture=pyk4a.Capture()

a=100
while a>0:
    a-=1
    print(a)

    if device.get_capture(capture,1000):

        img=capture.get_color_image()
        img=transformation.depth_image_to_color_camera(img)
        # print(img.get_device_timestamp())
        print(device.get_raw_calibration())
        img=img.numpy()
        # state=device.get_imu_sample()
        # print(state)
        cv2.imshow('t',img)
        cv2.waitKey(1)
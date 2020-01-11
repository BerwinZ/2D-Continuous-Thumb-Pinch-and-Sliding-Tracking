'''
This script is used to test the Pi Camera
'''

import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep
from color_map import rgb2ycbcr

if __name__ == "__main__":
    # Set up camera resolution, MAX is 1920*1080. 
    # The framerate needs to be set to 15 to enable this maximum resolution
    IM_LENGTH = 1920
    IM_WIDTH = 1080
    camera = PiCamera(resolution=(IM_LENGTH, IM_WIDTH), framerate=15)

    # Grab reference to the raw capture (3-d RGB Array)
    rawCapture = PiRGBArray(camera, size=(IM_LENGTH, IM_WIDTH))
    rawCapture.truncate(0)

    try:
        # Capture an image from camera and write it to the rawCapture
        for frame1 in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
            rgb_image = frame1.array
            ycbcr_image = rgb2ycbcr(rgb_image)

            # height, width, _ = rgb_image.shape
            # frame = frame[:int(height * 2 / 3), int(width / 2) - 213: int(width / 2) + 213, :]
            # frame.setflags(write=1)
            
            cv2.imshow('RGB Image', rgb_image)
            cv2.imshow('yCbCr Image', ycbcr_image)
            cv2.waitKey(1)
            rawCapture.truncate(0)
    except KeyboardInterrupt:
        camera.close()
        cv2.destroyAllWindows()
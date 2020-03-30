'''
This script is used to set up the Pi Camera.

It includes:
1. configure_camera
    Configure the Pi Camera and get the reference for the raw data
2. live_video
    Show the live video stream of the camera
'''

import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep

def configure_camera(IM_LENGTH=480, IM_WIDTH=480, FRAME_RATE=25):
    """Set up camera resolution, framerate and get the reference for raw data

    Max resolution is 1920*1080. (Optional 1080*720, 640*480). 
    The framerate needs to be set to 15 to enable this maximum resolution

    Returns:
        PiCamera, PiRGBArray
    """
    camera = PiCamera(resolution=(IM_LENGTH, IM_WIDTH), framerate=FRAME_RATE)

    # Grab reference to the raw capture (3-d RGB Array)
    rawCapture = PiRGBArray(camera, size=(IM_LENGTH, IM_WIDTH))
    rawCapture.truncate(0)

    sleep(0.1)

    return camera, rawCapture


def live_video(camera, rawCapture):
    """Show Live Video

    Arguments:
        camera {PiCamera}
        rawCapture {PiGRBArray}
    """
    capture_index = 0

    # Capture an image from camera and write it to the rawCapture
    for frame in camera.capture_continuous(rawCapture,
                                           format="bgr",
                                           use_video_port=True):
        bgr_image = frame.array
        cv2.imshow('Live Video', bgr_image)

        # Detect key action
        keypress = cv2.waitKey(25) & 0xFF
        if keypress == 27:
            break
        elif keypress == ord('s'):
            cv2.imwrite("./capture/hand_" + str(capture_index) + ".jpg",
                        bgr_image)
            capture_index += 1

        rawCapture.truncate(0)

    camera.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    Get the rawcapture of pi camera and show the live video
    """
    import sys, traceback

    try:
        camera, rawCapture = configure_camera(640, 480)
        live_video(camera, rawCapture)
    except Exception as e:
        camera.close()
        cv2.destroyAllWindows()
        print("Exception in user code:")
        print('-' * 60)
        traceback.print_exc(file=sys.stdout)
        print('-' * 60)

'''
This script is used to segment the hand from the video stream with the edges

It includes:

'''

import cv2
import numpy as np
from time import sleep
import picamera_control
import sys, traceback
from draw_tools import draw_vertical_lines

def edge_masking(img):
    pass



if __name__ == '__main__':
    """
    This function get the frame from the camera, and use thresholding to segment the hand part
    """
    try:
        camera, rawCapture = picamera_control.configure_camera(640,
                                                               480,
                                                               FRAME_RATE=35)

        for frame in camera.capture_continuous(rawCapture,
                                               format="bgr",
                                               use_video_port=True):
            bgr_image = frame.array

            # Get the mask using the Otsu thresholding method
            mask, _ = edge_masking(bgr_image)

            # Apply the mask to the image
            segment = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)

            # Display
            image_joint = np.concatenate((bgr_image, segment), axis=1)
            draw_vertical_lines(image_joint, 1)
            cv2.imshow('Image', image_joint)

            # if the user pressed ESC, then stop looping
            keypress = cv2.waitKey(25) & 0xFF
            if keypress == 27:
                break

            rawCapture.truncate(0)

        camera.close()
        cv2.destroyAllWindows()
    except Exception as e:
        camera.close()
        cv2.destroyAllWindows()
        print("Exception in user code:")
        print('-' * 60)
        traceback.print_exc(file=sys.stdout)
        print('-' * 60)

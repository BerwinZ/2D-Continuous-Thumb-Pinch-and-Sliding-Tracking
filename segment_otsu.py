'''
This script is used to segment the hand from the video stream with the skin color
'''

import cv2
import numpy as np
from time import sleep
import picamera_control
import os


def threshold_masking(img):
    """Get the mask for the img
    1. Use Otsu thresholding
    2. Erode and dilate to remove noise
    3. Get the area with the max contour 

    Arguments:
        img {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    # Convert to YCrCb 
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    # Cr and Cb Channels
    # mask_ycrcb = cv2.inRange(frame_ycrcb, np.array(
    #     [0, 145, 85]), np.array([255, 185, 155]))

    # Just use Y channel
    # mask = cv2.inRange(img_ycrcb[:,:,0], np.array([0]), np.array([150]))

    # Otsu Thresholding
    _, mask = cv2.threshold(img_ycrcb[:,:,1], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Erode or dilate the edges that has been removed
    kernel_size = min(img.shape[0], img.shape[1]) // 50
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.erode(mask, element)
    mask = cv2.dilate(mask, element)

    # Get the max contours
    _, contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    max_index = 0
    max_val = -1
    for idx, c in enumerate(contours):
        if cv2.contourArea(c) > max_val:
            max_val = cv2.contourArea(c)
            max_index = idx

    # Draw the max contours and fill it
    canvas = np.zeros(mask.shape).astype('uint8')
    mask = cv2.drawContours(canvas, contours, max_index, 255, -1)

    return mask


if __name__ == '__main__':
    """
    This function get the frame from the camera, and use thresholding to segment the hand part
    """
    try:
        camera, rawCapture = picamera_control.configure_camera(640, 480)

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            bgr_image = frame.array

            # img_ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCR_CB)
            # cv2.imshow("Y", img_ycrcb[:,:,0])
            # cv2.imshow("Cr", img_ycrcb[:,:,1])
            # cv2.imshow("Cb", img_ycrcb[:,:,2])

            mask = threshold_masking(bgr_image)
            segment = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)
            cv2.imshow("original", bgr_image)
            cv2.imshow('Mask', mask)
            cv2.imshow('Segment', segment)

            # if the user pressed ESC, then stop looping
            keypress = cv2.waitKey(25) & 0xFF
            if keypress == 27:
                break

            rawCapture.truncate(0)

        camera.close()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        camera.close()
        cv2.destroyAllWindows()

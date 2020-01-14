'''
This script is used to segment the hand from the video stream.

The steps are as follows.
1. Set up the histogram for the hand image
2. Use backprojection for the target image to extract the hand image
3. Use filter and other tools to remove the noise
'''

import cv2
import numpy as np
from time import sleep
import picamera_control
import os

# -----------------
# Read images
# -----------------
def read_images(folder_path):
    imgs = []
    for file in os.listdir(folder_path):
        imgs.append(cv2.imread(os.path.join(folder_path, file)))
    return imgs

# -----------------
# Generate the histogram for hand images
# -----------------
def generate_histogram(img_list):
    # Convert to HSV space
    hsv_imgs = []
    for im in img_list:
        hsv_imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
    # H: 0-180, S: 0-256
    hist = cv2.calcHist(hsv_imgs, [0, 1], None, [180, 256], [0, 180, 0, 256])
    # Normalize
    return cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)


# -----------------
# Use histogram backprojection to get mask for target image
# -----------------
def hist_masking(target, hist):
    # Get the backprojection result
    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([target_hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    # Filter to smooth the img
    kernel_size = 25
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dst = cv2.filter2D(dst, -1, disc)

    # Use threshold to make remove more noise
    _, mask = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    # Erode or dilate the edges that has been removed
    mask = cv2.erode(dst, None, iterations=5)
    mask = cv2.dilate(dst, None, iterations=5)

    # Count max area of contour
    # _, contour_list, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # max_area = 0
    # for cont in contour_list:
    #     area_cnt = cv2.contourArea(cont)
    #     max_area = max(area_cnt, max_area)
    
    # if max_area < 10000:
    #     return np.zeros(dst.shape, dtype='uint8')
    # else:
    #     return dst

    return mask



if __name__ == '__main__':
    try:
        hand_imgs = read_images('./images')
        hand_hist = generate_histogram(hand_imgs)

        camera, rawCapture = picamera_control.configure_camera()

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            bgr_image = frame.array

            mask = hist_masking(bgr_image, hand_hist)
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

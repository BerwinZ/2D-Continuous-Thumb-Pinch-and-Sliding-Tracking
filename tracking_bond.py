'''
This script is used to track the touch position

It includes:
1. Call Otsu threshold to finger_image the hand part and get the contour of hand
2. Get 2 Convexity Defects with largest distance from the contour
3. Segment the contour to up and bottom finger contour
'''

import cv2
import numpy as np
from time import sleep
import sys
import traceback
import picamera_control
from draw_board import draw_board, draw_vertical_lines
from segment_otsu import threshold_masking
from relative_mov_tracker import point_trakcer
from tracking_convdef import get_defect_points, configure_kalman_filter


def segment_diff_fingers(contour, defect_points, touch_point=None):
    """Segment the contour to the up finger and down finger

    Arguments:
        contour {[type]} -- [description]
        defect_points {[type]} -- [description]
        touch_point {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    if contour is None or defect_points is None:
        return None, None

    (x1, y1), (x2, y2) = defect_points

    if abs(x2 - x1) < 1e-6:
        up_finger = contour[contour[:, 0, 0] <= x1]
        down_finger = contour[contour[:, 0, 0] >= x1]
    else:
        grad_direc = (y2 - y1) / (x2 - x1)
        offset = y1 - grad_direc * x1
        up_finger = contour[grad_direc * contour[:, 0, 0] + offset -
                            contour[:, 0, 1] >= 0]
        down_finger = contour[grad_direc * contour[:, 0, 0] + offset -
                              contour[:, 0, 1] <= 0]

    if touch_point is not None:
        to_add = np.reshape(touch_point, [1, 1, 2])

        index1 = np.where((up_finger[:, 0, 0] == x1)
                          & (up_finger[:, 0, 1] == y1))[0]
        if index1 is not None and len(index1) != 0:
            up_finger = np.insert(up_finger, index1[-1] + 1, to_add, axis=0)

        down_finger = np.insert(down_finger,
                                down_finger.shape[0],
                                to_add,
                                axis=0)

    return up_finger, down_finger


def get_bound_points(up_contour, down_contour, height, width):
    if up_contour is None or down_contour is None:
        return None

    top_left = None
    left_bd = np.where(up_contour[:, 0, 0] == 0)[0]
    if len(left_bd) > 0:
        y_list = up_contour[left_bd, 0, 1]
        top_left = (0, max(y_list))
    else:
        top_bd = np.where(up_contour[:, 0, 1] == 0)[0]
        if len(top_bd) > 0:
            x_list = up_contour[top_bd, 0, 0]
            top_left = (min(x_list), 0)
        else:
            top_left = None

    top_right = None
    right_bd = np.where(up_contour[:, 0, 0] == width - 1)[0]
    if len(right_bd) > 0:
        y_list = up_contour[right_bd, 0, 1]
        top_right = (width - 1, max(y_list))
    else:
        top_bd = np.where(up_contour[:, 0, 1] == 0)[0]
        if len(top_bd) > 0:
            x_list = up_contour[top_bd, 0, 0]
            top_right = (max(x_list), 0)
        else:
            top_right = None

    bottom_left = None
    left_bd = np.where(up_contour[:, 0, 0] == 0)[0]
    if len(left_bd) > 0:
        y_list = up_contour[left_bd, 0, 1]
        bottom_left = (0, min(y_list))
    else:
        bottom_bd = np.where(up_contour[:, 0, 1] == height - 1)[0]
        if len(bottom_bd) > 0:
            x_list = up_contour[bottom_bd, 0, 0]
            bottom_left = (min(x_list), height - 1)
        else:
            bottom_left = None

    bottom_right = None
    right_bd = np.where(up_contour[:, 0, 0] == width - 1)[0]
    if len(right_bd) > 0:
        y_list = up_contour[right_bd, 0, 1]
        bottom_right = (0, min(y_list))
    else:
        bottom_bd = np.where(up_contour[:, 0, 1] == height - 1)[0]
        if len(bottom_bd) > 0:
            x_list = up_contour[bottom_bd, 0, 0]
            bottom_right = (max(x_list), height - 1)
        else:
            bottom_right = None

    return top_left, top_right, bottom_left, bottom_right


if __name__ == '__main__':
    """
    This function get the frame from the camera, and use thresholding to finger_image the hand part
    """
    try:
        WIDTH, HEIGHT = 640, 480
        # Note: Higher framerate will bring noise to the segmented image
        camera, rawCapture = picamera_control.configure_camera(WIDTH,
                                                               HEIGHT,
                                                               FRAME_RATE=35)

        # Kalman filter to remove noise from the point movement
        kalman_filter = configure_kalman_filter()

        print('-' * 60)
        print("Press F to turn ON/OFF the kalman filter")
        print('-' * 60)

        for frame in camera.capture_continuous(rawCapture,
                                               format="bgr",
                                               use_video_port=True):
            bgr_image = frame.array

            # Get the mask and its contour using the Otsu thresholding method and apply the mask to image
            mask, contour = threshold_masking(bgr_image)
            finger_image = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)

            # Get defect points from the contour
            defect_points, _ = get_defect_points(
                contour, MIN_CHECK_AREA=100000, MIN_DEFECT_DISTANCE=5000)

            # Segment the two fingers
            up_finger_contour, down_finger_contour = segment_diff_fingers(
                contour, defect_points)

            # if up_finger_contour:
            #     cv2.drawContours(finger_image, [up_finger_contour], 0,
            #                      [0, 0, 255], 3)
            #     cv2.drawContours(finger_image, [down_finger_contour], 0,
            #                      [255, 0, 0], 3)

            # Get four points
            bound_points = get_bound_points(up_finger_contour,
                                            down_finger_contour,
                                            bgr_image.shape[0],
                                            bgr_image.shape[1])
            if bound_points:
                for point in list(bound_points):
                    if point:
                        cv2.circle(finger_image, point, 5, [0, 0, 255], -1)

            # Display
            cv2.imshow('Finger', finger_image)

            # if the user pressed ESC, then stop looping
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == 27:
                break
            elif keypress == ord('f'):
                if kalman_filter is None:
                    kalman_filter = configure_kalman_filter()
                    print("Kalman Filter ON")
                else:
                    kalman_filter = None
                    print("Kalman Filter OFF")

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

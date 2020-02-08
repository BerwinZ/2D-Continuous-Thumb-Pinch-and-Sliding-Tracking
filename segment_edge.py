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
from segment_otsu import threshold_masking
from tracking_convdef import get_defect_points
from draw_tools import draw_points
from scipy import ndimage



def edge_masking(img):
    pass


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1,M-1):
        for j in range(1,N-1):
            q = 255
            r = 255
            
            #angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = img[i, j+1]
                r = img[i, j-1]
            #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]
            #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]
            #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]

            if (img[i,j] >= q) and (img[i,j] >= r):
                Z[i,j] = img[i,j]
            else:
                Z[i,j] = 0

    return Z


if __name__ == '__main__':
    """
    This function get the frame from the camera, and use thresholding to segment the hand part
    """
    try:
        IM_WIDTH, IM_HEIGHT = 480, 640
        camera, rawCapture = picamera_control.configure_camera(IM_HEIGHT,
                                                               IM_WIDTH,
                                                               FRAME_RATE=35)

        threshold1 = 17
        threshold2 = 23
        power = 1

        for frame in camera.capture_continuous(rawCapture,
                                               format="bgr",
                                               use_video_port=True):
            bgr_image = frame.array
            mask, contour = threshold_masking(bgr_image)
            finger_image = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)

            defect_points, _ = get_defect_points(contour,
                                                 MIN_CHECK_AREA=100000,
                                                 MIN_DEFECT_DISTANCE=5000)

            draw_points(bgr_image, defect_points)

            WIN_HEIGHT = 100

            if defect_points is not None:
                (left, y1), (right, y2) = defect_points
                up = max(0, min(y1 - WIN_HEIGHT//2, y2 - WIN_HEIGHT//2))
                down = min(IM_HEIGHT - 1, max(y1 + WIN_HEIGHT//2, y2 + WIN_HEIGHT//2))
                window_img = cv2.cvtColor(bgr_image[up:down, left: right, :], cv2.COLOR_BGR2GRAY)

                edge_image = cv2.Canny(window_img,
                                    threshold1=threshold1,
                                    threshold2=threshold2)

                sample_points = np.where(edge_image == 255)
                line = np.poly1d(np.polyfit(sample_points[0], sample_points[1], power))

                draw_img = window_img.copy()
                touch_line_x = np.arange(0, draw_img.shape[1])
                touch_line_y = np.array(list(map(int, line(touch_line_x))))
                touch_line_x = np.reshape(touch_line_x, (touch_line_x.shape[0], 1))
                touch_line_y = np.reshape(touch_line_y, (touch_line_y.shape[0], 1))
                touch_line = np.concatenate((touch_line_x, touch_line_y), axis=1)
                draw_points(draw_img, list(touch_line), 3)
 
                window_joint = np.concatenate(
                    (window_img, edge_image, draw_img), axis=1
                )
                draw_vertical_lines(window_joint, 2)
                cv2.imshow("Window", window_joint)

            # Display
            cv2.imshow("Original", bgr_image)

            # if the user pressed ESC, then stop looping
            keypress = cv2.waitKey(25) & 0xFF
            if keypress == 27:
                break
            elif keypress == ord('j'):    # LEFT
                threshold1 -= 1
                print("Thres1", threshold1, "Thres2", threshold2)
            elif keypress == ord('l'):    # RIGHT
                threshold1 += 1
                print("Thres1", threshold1, "Thres2", threshold2)
            elif keypress == ord('k'):    # UP
                threshold2 -= 1
                print("Thres1", threshold1, "Thres2", threshold2)
            elif keypress == ord('i'):    # DOWN
                threshold2 += 1
                print("Thres1", threshold1, "Thres2", threshold2)
            elif keypress == ord('m'):
                power += 1
                print("Fitting Power", power)
            elif keypress == ord('n'):
                power -= 1
                print("Fitting Power", power)   
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

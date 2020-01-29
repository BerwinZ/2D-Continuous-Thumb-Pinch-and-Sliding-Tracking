'''
This script is used to track the touch position with amendment

It includes:
1. Call Otsu threshold to finger_image the hand part and get the contour of hand
2. Get 2 Convexity Defects with largest distance from the contour
3. Calculate the middle point of convexity defects, find the touch point and use Kalman filter to correct it
4. Segment the contour to up and bottom finger contour
5. Use point tracker to calculate movements with amendment
6. Draw movements in drawing board
'''

import cv2
import numpy as np
from time import sleep
import sys
import traceback
import picamera_control
from segment_otsu import threshold_masking
from tracking_convdef import get_defect_points, get_touch_point, configure_kalman_filter
from tracking_bound import segment_diff_fingers
from move_tracker import touch_trakcer
from draw_tools import draw_board, draw_vertical_lines, draw_points, draw_contours

if __name__ == '__main__':
    """
    This function get the frame from the camera, and use thresholding to finger_image the hand part
    """
    try:
        WIDTH, HEIGHT = 640, 480
        # Note: Higher framerate will bring noise to the segmented image
        camera, rawCapture = picamera_control.configure_camera(WIDTH,
                                                               HEIGHT,
                                                               FRAME_RATE=40)

        # Kalman filter to remove noise from the point movement
        kalman_filter = configure_kalman_filter()

        # Tracker to convert point movement in image coordinate to the draw board coordinate
        tracker = touch_trakcer()

        # Drawing boards
        DRAW_SCALER = 0.5
        DR_WIDTH, DR_HEIGHT = 320, 320
        hv_board = draw_board(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=5)
        hor_board = draw_board(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)
        ver_board = draw_board(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)

        print('-' * 60)
        print(
            "To calibrate, press 'C' and move the hand in IMAGE LEFT, RIGHT, UP, DOWN"
        )
        print("Press F to turn ON/OFF the kalman filter")
        print('-' * 60)

        for frame in camera.capture_continuous(rawCapture,
                                               format="bgr",
                                               use_video_port=True):
            bgr_image = frame.array

            # ---------------------------------------------
            # 1. Calculation
            # ---------------------------------------------

            # ---------------------------------------------
            # 1.1 Get the mask and its contour using the Otsu thresholding method and apply the mask to image
            # ---------------------------------------------
            mask, contour = threshold_masking(bgr_image)
            finger_image = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)
            finger_image2 = finger_image.copy()

            # ---------------------------------------------
            # 1.2 Get touch point
            # ---------------------------------------------
            defect_points, _ = get_defect_points(contour,
                                                 MIN_CHECK_AREA=100000,
                                                 MIN_DEFECT_DISTANCE=5000)
            touch_point, filter_touch_point = get_touch_point(
                defect_points, finger_image, kalman_filter=kalman_filter)
            # ---------------------------------------------
            # 1.3 Get up and down finger contour
            # ---------------------------------------------
            up_contour, down_contour = segment_diff_fingers(
                contour, defect_points, touch_point)

            # ---------------------------------------------
            # 1.4 Draw elements
            # ---------------------------------------------
            # Two defect points (Green)
            draw_points(finger_image, defect_points, color=[0, 255, 0])
            # Raw touch point (Red)
            draw_points(finger_image, touch_point, color=[0, 0, 255])
            # Filter touch point (Blue)
            draw_points(finger_image, filter_touch_point, color=[255, 0, 0])
            # Draw contour
            draw_contours(finger_image2,
                          up_contour,
                          thickness=3,
                          color=[0, 0, 255])
            draw_contours(finger_image2,
                          down_contour,
                          thickness=3,
                          color=[255, 0, 0])
            # ---------------------------------------------
            # 1.5 Show image
            # ---------------------------------------------
            image_joint = np.concatenate((finger_image, finger_image2), axis=1)
            draw_vertical_lines(image_joint, 1)
            cv2.imshow('Finger', image_joint)

            # ---------------------------------------------
            # 2. Application
            # ---------------------------------------------

            # ---------------------------------------------
            # 2.1 Use tracker to calculate the movements
            # ---------------------------------------------
            if filter_touch_point:
                touch_point = filter_touch_point

            dx, dy = tracker.calc_scaled_move(touch_point,
                                              MOVE_SCALE_RANGE=[-100, 100])

            # ---------------------------------------------
            # 2.2 Draw the movments in the drawing board
            # ---------------------------------------------
            if dx is not None:
                dx = -dx * DRAW_SCALER
                dy = dy * DRAW_SCALER
            hor_board.draw_filled_point((dx, 0))
            ver_board.draw_filled_point((0, dy))
            hv_board.draw_filled_point((dx, dy))

            # ---------------------------------------------
            # 2.3 Display
            # ---------------------------------------------
            move_joint = np.concatenate(
                (hv_board.board, hor_board.board, ver_board.board), axis=1)
            draw_vertical_lines(move_joint, 2)
            cv2.imshow('Finger Track', move_joint)

            # ---------------------------------------------
            # 3. User Input
            # ---------------------------------------------
            # if the user pressed ESC, then stop looping
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == 27:
                break
            elif keypress == ord('c'):
                tracker.calibrate_touch_point(touch_point)
                hv_board.reset_board()
                hor_board.reset_board()
                ver_board.reset_board()
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

'''
1. get_touch_point
    get the touch point from the defect points and the gray scale of the image
'''

import cv2
import numpy as np
from math_tools import get_min_gray_point, points_distance

def get_touch_point(defect_points, finger_img, kalman_filter=None):
    """Get the touch points based on the defect points and gray value of the image

    Arguments:
        defect_points {list of tuple} -- [Two convexity defects points]
        finger_img {np.array} -- [BGR Image of fingers]

    Keyword Arguments:
        kalman_filter {[type]} -- [description] (default: {None})

    Returns:
        touchPoint [tuple] -- [The touch coordinate of the fingertips]
        filter_touchPoint [tuple] -- [The touch coordinate of the fingertips after kalman filter]
    """
    if defect_points is None or finger_img is None:
        return None, None

    # Calculate the middle point
    (x1, y1), (x2, y2) = defect_points
    middle_point = ((x1 + x2) // 2, (y1 + y2) // 2)

    # Calculate the touch point according to the gradient change
    gray_img = cv2.cvtColor(finger_img, cv2.COLOR_BGR2GRAY)
    search_distance = points_distance(defect_points[0], defect_points[1]) / 5

    if abs(y2 - y1) < 1e-9:
        touch_point = get_min_gray_point(gray_img,
                                   middle_point,
                                   distance=search_distance)
    else:
        grad_direc = -(x2 - x1) / (y2 - y1)
        touch_point = get_min_gray_point(gray_img,
                                   middle_point,
                                   distance=search_distance,
                                   slope=grad_direc)

    # If kalman filter adopted, use it to correct the observation
    filter_touch_point = None
    if kalman_filter is not None:
        kalman_filter.correct(
            np.array([[np.float32(touch_point[0])],
                      [np.float32(touch_point[1])]]))
        predict_point = kalman_filter.predict()
        filter_touch_point = (int(predict_point[0]), int(predict_point[1]))

    return touch_point, filter_touch_point


if __name__ == '__main__':
    """
    This script is used to track the touch position with touch points

    It includes:
    1. Call Otsu threshold to finger_image the hand part and get the contour of hand
    2. Get 2 Convexity Defects with largest distance from the contour
    3. Calculate the middle point of convexity defects, find the touch point and use Kalman filter to correct it
    4. Use point_tracker to track the touch points movements
    5. Draw the relative movements in a drawing board
    """
    from time import sleep
    import sys, traceback
    import picamera_control
    from draw_tools import DrawBoard, draw_vertical_lines, draw_points
    from math_tools import configure_kalman_filter
    from image_segment import threshold_masking
    from coord_calculator import BaseCalculator
    from feature_extraction import get_defect_points

    try:
        WIDTH, HEIGHT = 640, 480
        # Note: Higher framerate will bring noise to the segmented image
        camera, rawCapture = picamera_control.configure_camera(WIDTH,
                                                               HEIGHT,
                                                               FRAME_RATE=35)

        # Kalman filter to remove noise from the point movement
        kalman_filter = configure_kalman_filter()

        # Tracker to convert point movement in image coordinate to the draw board coordinate
        tracker = BaseCalculator()

        # Drawing boards
        DR_WIDTH, DR_HEIGHT = 320, 320
        hv_board = DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=5)
        hor_board = DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)
        ver_board = DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)

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

            # Get the mask and its contour using the Otsu thresholding method and apply the mask to image
            mask, contour, finger_image = threshold_masking(bgr_image)

            # Get convexity defects points from the contour
            defect_points, _ = get_defect_points(contour,
                                                 MIN_VALID_CONT_AREA=100000,
                                                 MIN_DEFECT_DISTANCE=5000)

            # Get touch point from the defect points and the img
            touch_point, filter_touch_point = get_touch_point(
                defect_points, finger_image, kalman_filter=kalman_filter)

            # Two defect points (Green)
            draw_points(finger_image, defect_points, color=[0, 255, 0])
            # Raw touch point (Red)
            draw_points(finger_image, touch_point, color=[0, 0, 255])
            # Filter touch point (Blue)
            draw_points(finger_image, filter_touch_point, color=[255, 0, 0])

            # Display
            cv2.imshow('Finger', finger_image)

            # ---------------------------------------------
            # 2. Application
            # ---------------------------------------------

            # Track the touch point
            if filter_touch_point:
                touch_point = filter_touch_point
            dx, dy = tracker.calc_coord(touch_point)

            # Draw the touch point track
            DRAW_SCALER = 50
            if dx is not None:
                dx = -dx * DRAW_SCALER
                dy = dy * DRAW_SCALER
            hor_board.update_dot((dx, 0))
            ver_board.update_dot((0, dy))
            hv_board.update_dot((dx, dy))

            # Display
            H_V_joint = np.concatenate(
                (hv_board.board, hor_board.board, ver_board.board), axis=1)
            draw_vertical_lines(H_V_joint, 2)
            cv2.imshow('H V Movement', H_V_joint)

            # if the user pressed ESC, then stop looping
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == 27:
                break
            elif keypress == ord('c'):
                tracker.calibrate_touch_scale(touch_point)
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

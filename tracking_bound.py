'''
This script is used to track the touch position

It includes:
1. Call Otsu threshold to finger_image the hand part and get the contour of hand
2. Get 2 Convexity Defects with largest distance from the contour
3. Segment the contour to up and bottom finger contour
4. Get the top_left and bottom_right boundary points
5. Use bound_tracker to track the movements
6. Draw the relative movements in a drawing board
'''

import cv2
import numpy as np
from time import sleep
import sys
import traceback
import picamera_control
from draw_tools import draw_vertical_lines, draw_points, draw_board
from segment_otsu import threshold_masking
from move_tracker import bound_trakcer
from tracking_convdef import get_defect_points, get_min_gray


def get_touch_line(finger_img, defect_points, line_points_num=10):

    if defect_points is None:
        return None

    (x1, y1), (x2, y2) = defect_points
    gray_img = cv2.cvtColor(finger_img, cv2.COLOR_BGR2GRAY)
    search_distance = np.sqrt(
        np.sum(
            np.square(np.array(defect_points[0]) -
                      np.array(defect_points[1])))) / 5

    # Get the check points
    check_points = []
    for i in range(1, line_points_num):
        check_points.append((((x2 - x1) // line_points_num * i + x1),
                             ((y2 - y1) // line_points_num * i + y1)))

    # Get the touch line
    if abs(y2 - y1) < 1e-6:
        touch_line = map(get_min_gray, [gray_img] * (line_points_num - 1),
                         check_points,
                         [search_distance] * (line_points_num - 1))
    else:
        grad_direc = -(x2 - x1) / (y2 - y1)
        touch_line = map(get_min_gray, [gray_img] * (line_points_num - 1),
                         check_points,
                         [search_distance] * (line_points_num - 1),
                         [grad_direc] * (line_points_num - 1))

    return list(touch_line)


def segment_diff_fingers(contour, defect_points):
    """Segment the contour to the up finger and down finger

    Arguments:
        contour {[type]} -- [description]
        defect_points {[type]} -- [description]
        touch_points {[type]} -- [description]

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
        segment_line = lambda x, y: grad_direc * (x - x1) - (y - y1)
        up_finger = contour[segment_line(contour[:, 0, 0], contour[:, 0,
                                                                   1]) >= 0]
        down_finger = contour[segment_line(contour[:, 0, 0], contour[:, 0,
                                                                     1]) <= 0]

    return up_finger, down_finger


def add_touch_line(is_up, contour, defect_points, touch_line):
    """Add the touch line to the up and down contour
    
    Arguments:
        is_up {bool} -- [description]
        contour {[type]} -- [description]
        defect_points {[type]} -- [description]
        touch_line {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    if contour is None or touch_line is None or defect_points is None:
        return None

    (x1, y1), (x2, y2) = defect_points

    if type(touch_line) == list:
        to_add = np.reshape(touch_line, [len(touch_line), 1, 2])
    elif type(touch_line) == tuple:
        to_add = np.reshape(touch_line, [1, 1, 2])

    if is_up:
        index = np.where((contour[:, 0, 0] == x1)
                         & (contour[:, 0, 1] == y1))[0]
        if index is not None and len(index) != 0:
            contour = np.insert(contour, index[-1] + 1, to_add, axis=0)
    else:
        contour = np.insert(contour, contour.shape[0], to_add[::-1], axis=0)

    return contour


def get_bound_points(up_contour, down_contour, height, width):
    """Get the four boundary points of the hand contour
    
    Arguments:
        up_contour {[type]} -- [description]
        down_contour {[type]} -- [description]
        height {[type]} -- [description]
        width {[type]} -- [description]
    
    Returns:
        Boundary Points [list of tuple] -- [top_left, top_right, bottom_left, bottom_right]
    """
    if up_contour is None or down_contour is None or height is None or width is None:
        return None, None, None, None

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
    left_bd = np.where(down_contour[:, 0, 0] == 0)[0]
    if len(left_bd) > 0:
        y_list = down_contour[left_bd, 0, 1]
        bottom_left = (0, min(y_list))
    else:
        bottom_bd = np.where(down_contour[:, 0, 1] == height - 1)[0]
        if len(bottom_bd) > 0:
            x_list = down_contour[bottom_bd, 0, 0]
            bottom_left = (min(x_list), height - 1)
        else:
            bottom_left = None

    bottom_right = None
    right_bd = np.where(down_contour[:, 0, 0] == width - 1)[0]
    if len(right_bd) > 0:
        y_list = down_contour[right_bd, 0, 1]
        bottom_right = (width - 1, min(y_list))
    else:
        bottom_bd = np.where(down_contour[:, 0, 1] == height - 1)[0]
        if len(bottom_bd) > 0:
            x_list = down_contour[bottom_bd, 0, 0]
            bottom_right = (max(x_list), height - 1)
        else:
            left_bd = np.where(down_contour[:, 0, 0] == 0)[0]
            if len(left_bd) > 0:
                y_list = down_contour[left_bd, 0, 1]
                bottom_right = (0, max(y_list))
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

        DRAW_CONTOUR = True

        # Point tracker
        tracker = bound_trakcer(HEIGHT, WIDTH)

        # Drawing boards
        DR_WIDTH, DR_HEIGHT = 320, 320
        hv_board = draw_board(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=5)
        hor_board = draw_board(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)
        ver_board = draw_board(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)

        print('-' * 60)
        print(
            "To calibrate, press 'C' and move the hand in IMAGE LEFT, RIGHT, UP, DOWN"
        )
        print("Press 'D' to turn on/off the contour drawing")
        print('-' * 60)

        for frame in camera.capture_continuous(rawCapture,
                                               format="bgr",
                                               use_video_port=True):
            bgr_image = frame.array

            # ---------------------------------------------
            # 1. Calculation
            # ---------------------------------------------

            # Get the mask and its contour using the Otsu thresholding method and apply the mask to image
            mask, contour = threshold_masking(bgr_image)
            finger_image = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)

            # Get defect points from the contour
            defect_points, _ = get_defect_points(contour,
                                                 MIN_CHECK_AREA=100000,
                                                 MIN_DEFECT_DISTANCE=5000)
            draw_points(finger_image, defect_points)

            # Get the touch lines
            touch_line = get_touch_line(finger_image,
                                        defect_points,
                                        line_points_num=10)
            draw_points(finger_image, touch_line)

            # Segment the two fingers
            up_finger_contour, down_finger_contour = segment_diff_fingers(
                contour, defect_points)
            up_finger_contour = add_touch_line(True, up_finger_contour,
                                               defect_points, touch_line)
            down_finger_contour = add_touch_line(False, down_finger_contour,
                                                 defect_points, touch_line)

            if up_finger_contour is not None and DRAW_CONTOUR:
                cv2.drawContours(finger_image, [up_finger_contour], 0,
                                 [0, 0, 255], 3)
                cv2.drawContours(finger_image, [down_finger_contour], 0,
                                 [255, 0, 0], 3)

            # Get four points
            bound_points = get_bound_points(up_finger_contour,
                                            down_finger_contour,
                                            bgr_image.shape[0],
                                            bgr_image.shape[1])

            to_draw = [bound_points[0], bound_points[3]]
            draw_points(finger_image, to_draw, radius=10, color=[0, 0, 255])

            # Display
            cv2.imshow('Finger', finger_image)

            # ---------------------------------------------
            # 2. Application
            # ---------------------------------------------

            dx, dy = tracker.calc_scaled_move(bound_points[0], bound_points[3])

            # Draw the touch point track
            DRAW_SCALER = 50
            if dx is not None:
                dx = -dx * DRAW_SCALER
                dy = dy * DRAW_SCALER
            hor_board.draw_filled_point((dx, 0))
            ver_board.draw_filled_point((0, dy))
            hv_board.draw_filled_point((dx, dy))

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
                tracker.calibrate_boundary_point(bound_points[0],
                                                 bound_points[3])
                hv_board.reset_board()
                hor_board.reset_board()
                ver_board.reset_board()
            elif keypress == ord('d'):
                DRAW_CONTOUR = not DRAW_CONTOUR
                print("Draw contour: ", DRAW_CONTOUR)

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

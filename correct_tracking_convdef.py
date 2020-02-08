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
from scipy.optimize import fsolve, curve_fit
import picamera_control
from segment_otsu import threshold_masking
from segment_edge import sobel_filters
from tracking_convdef import get_defect_points
from tracking_bound import segment_diff_fingers, add_touch_line, get_bound_points
from move_tracker import correct_tracker
from draw_tools import draw_board, draw_vertical_lines, draw_points, draw_contours
from math_tools import points_distance


def get_centroid(contour):
    if contour is None:
        return None, None

    moment = cv2.moments(contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None, None


def calc_touch_angle(base, target):
    """Calculate the degree angle from base to target
    
    Arguments:
        base {[type]} -- [description]
        target {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    if base is None or target is None:
        return None

    x1, y1 = base
    x2, y2 = target
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None

    if abs(x1 - x2) < 1e-9:
        if y2 > y1:
            return 90
        else:
            return -90
    else:
        slope = (y1 - y2) / (x1 - x2)
        angle = np.degrees(np.arctan(slope))
        if y2 > y1 and angle < 0:
            angle = 180 + angle

        return angle


def fit_lost_contour(is_up,
                     contour,
                     bound_points,
                     fitting_curve,
                     defect_points,
                     draw_image=None):
    """Fit the up contour with polyfit using the points of the contour within the bound points
    
    Arguments:
        contour {[type]} -- [description]
        bound_points {[type]} -- [description]
        is_up {bool} -- [description]
        fitting_curve {[type]} -- [description]
        defect_points {[type]} -- [description]
    
    Keyword Arguments:
        draw_image {[type]} -- [description] (default: {None})
    
    Returns:
        touch_line_points [2-d list] -- [description]
        curve [lambda function]
    """
    if contour is None or fitting_curve is None or defect_points is None:
        return None, None, None

    # Get the points used to fit
    if bound_points is None or bound_points[0] is None or bound_points[
            1] is None:
        return None, None, None

    (x1, y1), (x2, y2) = bound_points[0], bound_points[1]
    contour = np.reshape(contour, (contour.shape[0], 2))
    theta = None

    if is_up:
        f = lambda x, y: (y2 - y1) / (x2 - x1) * (x - x1) - (y - y1)
        contour = contour[f(contour[:, 0], contour[:, 1]) < 0]
    else:
        # Rotate points
        if abs(x1 - x2) < 1e-9:
            theta = -1 * 90 * np.pi / 180
            contour = contour[contour[:, 0] > 0]
        else:
            theta = -1 * np.arctan((y2 - y1) / (x2 - x1))
            f = lambda x, y: (y2 - y1) / (x2 - x1) * (x - x1) - (y - y1)
            contour = contour[f(contour[:, 0], contour[:, 1]) > 0]

        rotate = rotate_array(theta)
        contour = rotate.dot(contour.T).T
        defect_points = rotate.dot(np.array(defect_points).T).T

    # Draw it
    if draw_image is not None:
        color = [0, 0, 255] if is_up else [255, 0, 0]
        draw_points(draw_image, contour.astype("int"), radius=3, color=color)

    # Fit the function
    X = contour[:, 0]
    Y = contour[:, 1]
    curve = fitting_curve(X, Y)

    # Get the touch line x range
    p_x = np.arange(defect_points[0][0], defect_points[1][0])
    p_y = np.array(curve(p_x))
    # Reshape
    num = p_x.shape[0]
    p_x = np.reshape(p_x, (num, 1))
    p_y = np.reshape(p_y, (num, 1))
    # Joint
    points = np.concatenate((p_x, p_y), axis=1)

    if is_up is False:
        draw_points(draw_image, points.astype("int"), 3, [255, 0, 0])
        rotate = rotate_array(-theta)
        points = rotate.dot(points.T).T

    points = points.astype("int")

    return points, curve, theta


def get_intersection(defect_points, curve):
    # Calculate the middle point
    (x1, y1), (x2, y2) = defect_points
    middle_point = ((x1 + x2) // 2, (y1 + y2) // 2)

    if abs(y2 - y1) < 1e-9:
        touch_point = (middle_point[0], curve(middle_point[0]))
    else:
        k = -(x2 - x1) / (y2 - y1)
        line = lambda x: k * x + middle_point[1] - k * middle_point[0]
        intersec = fsolve(lambda x: line(x) - curve(x), middle_point[0])[0]
        touch_point = (intersec, line(intersec))

    return touch_point


def rotate_array(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def get_touch_point(defect_points,
                    up_centroid,
                    down_centroid,
                    up_touch_line,
                    down_touch_line,
                    theta,
                    finger_image=None):
    """Get the touch point according to the defect points and touch line.
    
    Arguments:
        defect_points {[type]} -- [description]
        up_touch_line {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    if defect_points is None or up_centroid is None or down_centroid is None or up_touch_line is None or down_touch_line is None or theta is None:
        return None

    # Calculate the middle point
    (x1, y1), (x2, y2) = defect_points
    middle_point = ((x1 + x2) // 2, (y1 + y2) // 2)

    distance_ratio = points_distance(middle_point,
                                     up_centroid) / points_distance(
                                         middle_point, down_centroid)
    if distance_ratio >= 1:
        touch_point = get_intersection(defect_points, up_touch_line)
    else:
        rotate = rotate_array(theta)
        r_defect_points = rotate.dot(np.array(defect_points).T).T
        touch_point = get_intersection(r_defect_points, down_touch_line)

        rotate = rotate_array(-theta)
        touch_point = rotate.dot(np.array([touch_point]).T).T[0]

        if distance_ratio >= 0.65:
            tmp = get_intersection(defect_points, up_touch_line)
            touch_point = ((touch_point[0] + tmp[0]) / 2,
                           (touch_point[1] + tmp[1]) / 2)

    touch_point = tuple(map(int, touch_point))

    if finger_image is not None:
        G, _ = sobel_filters(cv2.cvtColor(finger_image, cv2.COLOR_BGR2GRAY))

    return touch_point


if __name__ == '__main__':
    """
    This function get the frame from the camera, and use thresholding to finger_image the hand part
    """
    try:
        IM_WIDTH, IM_HEIGHT = 640, 480
        # Note: Higher framerate will bring noise to the segmented image
        camera, rawCapture = picamera_control.configure_camera(IM_WIDTH,
                                                               IM_HEIGHT,
                                                               FRAME_RATE=40)
        # Show image
        SHOW_IMAGE = True

        # Tracker to convert point movement in image coordinate to the draw board coordinate
        tracker = correct_tracker()

        # Drawing boards
        DRAW_SCALER = 50
        DR_WIDTH, DR_HEIGHT = 320, 320
        hv_board = draw_board(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=5)
        hor_board = draw_board(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)
        ver_board = draw_board(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)

        print('-' * 60)
        print(
            "To calibrate, press 'C' and move the hand in IMAGE LEFT, RIGHT, UP, DOWN"
        )
        print("Press A to turn ON/OFF the finger image")
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
            # 1.2 Get defect points
            # ---------------------------------------------
            defect_points, _ = get_defect_points(contour,
                                                 MIN_CHECK_AREA=100000,
                                                 MIN_DEFECT_DISTANCE=5000)

            # ---------------------------------------------
            # 1.3 Get up and down finger contour
            # ---------------------------------------------
            up_contour, down_contour = segment_diff_fingers(
                contour, defect_points)

            # ---------------------------------------------
            # 1.4 Get touch lines
            # ---------------------------------------------
            top_left, top_right, bottom_left, bottom_right = get_bound_points(
                up_contour, down_contour, IM_HEIGHT, IM_WIDTH)

            up_LC_points, up_touch_line, _ = fit_lost_contour(
                is_up=True,
                contour=up_contour,
                bound_points=(top_left, top_right),
                fitting_curve=lambda X, Y: np.poly1d(np.polyfit(X, Y, 4)),
                defect_points=defect_points,
                draw_image=None)
            draw_points(finger_image, up_LC_points, 3, [0, 0, 255])

            down_LC_points, down_touch_line, theta = fit_lost_contour(
                is_up=False,
                contour=down_contour,
                bound_points=(bottom_left, bottom_right),
                fitting_curve=lambda X, Y: np.poly1d(np.polyfit(X, Y, 3)),
                defect_points=defect_points,
                draw_image=None)
            draw_points(finger_image, down_LC_points, 3, [255, 0, 0])

            # ---------------------------------------------
            # 1.5 Get centroids of contours
            # ---------------------------------------------
            up_centroid = get_centroid(up_contour)
            down_centroid = get_centroid(down_contour)

            # ---------------------------------------------
            # 1.7 Get touch point and get the touch angle of touch point to the centroid
            # ---------------------------------------------
            touch_point = get_touch_point(defect_points=defect_points,
                                          up_centroid=up_centroid,
                                          down_centroid=down_centroid,
                                          up_touch_line=up_touch_line,
                                          down_touch_line=down_touch_line,
                                          theta=theta)
            touch_angle = calc_touch_angle(up_centroid, touch_point)

            # ---------------------------------------------
            # 1.8 Draw elements
            # ---------------------------------------------
            # Two defect points (Green)
            draw_points(finger_image, defect_points, color=[0, 255, 0])
            # Raw touch point (Red)
            draw_points(finger_image, touch_point, color=[255, 255, 255])
            # Draw centroid points (Pink)
            draw_points(finger_image, up_centroid, color=[255, 0, 255])
            draw_points(finger_image, down_centroid, color=[255, 0, 255])

            # Draw up contour
            draw_contours(finger_image2,
                          up_contour,
                          thickness=3,
                          color=[0, 0, 255])
            draw_contours(finger_image2,
                          down_contour,
                          thickness=3,
                          color=[255, 0, 0])

            # ---------------------------------------------
            # 1.9 Show image
            # ---------------------------------------------
            if SHOW_IMAGE:
                image_joint = np.concatenate(
                    (bgr_image, finger_image, finger_image2), axis=1)
                draw_vertical_lines(image_joint, 2)
                cv2.imshow('Finger', image_joint)

            # ---------------------------------------------
            # 2. Application
            # ---------------------------------------------

            # ---------------------------------------------
            # 2.1 Use tracker to calculate the movements
            # ---------------------------------------------
            dx, dy = tracker.calc_scaled_move(touch_angle, up_centroid[1])

            # ---------------------------------------------
            # 2.2 Show parameters
            # ---------------------------------------------
            # x1 = points_distance(touch_point, up_centroid)
            # x2 = points_distance(touch_point, down_centroid)
            # if x1 is not None and x2 is not None:
            #     x1 = round(x1, 0)
            #     x2 = round(x2, 0)
            #     x3 = round(x1 + x2, 0)
            #     print(x1, x2, x3)

            # if dx is not None and up_contour is not None and defect_points is not None:
            #     print(round(dx, 2), round(dy, 2),
            #          cv2.contourArea(up_contour),
            #          cv2.contourArea(down_contour),
            #          round(points_distance(defect_points[0], defect_points[1]), 1))

            # ---------------------------------------------
            # 2.3 Draw the movments in the drawing board
            # ---------------------------------------------
            if dx is not None and dy is not None:
                dx = -dx * DRAW_SCALER
                dy = dy * DRAW_SCALER
            hor_board.draw_filled_point((dx, 0))
            ver_board.draw_filled_point((0, dy))
            hv_board.draw_filled_point((dx, dy))

            # ---------------------------------------------
            # 2.4 Display
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
                tracker.calibrate_touch_point(touch_angle, up_centroid[1])
                hv_board.reset_board()
                hor_board.reset_board()
                ver_board.reset_board()
            elif keypress == ord('s'):
                cv2.imwrite('screenshot.jpg', finger_image)
            elif keypress == ord('a'):
                if SHOW_IMAGE:
                    cv2.destroyWindow("Finger")
                SHOW_IMAGE = not SHOW_IMAGE

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

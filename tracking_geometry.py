"""
1. get_touch_point
    Get touch point in the touch line curve
2. show_whole_finger_motion
    Show the motion for the whole finger
"""

def __get_intersection(curve, defect_points, theta=None):
    """Get intersection point between the vertical bisector and the curve
    
    Arguments:
        defect_points {[type]} -- [description]
        curve {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    # Calculate the middle point
    if theta is None:
        (x1, y1), (x2, y2) = defect_points
        middle_point = ((x1 + x2) // 2, (y1 + y2) // 2)

        if abs(y2 - y1) < 1e-9:
            touch_point = (middle_point[0], curve(middle_point[0]))
        else:
            k = -(x2 - x1) / (y2 - y1)
            line = lambda x: k * (x - middle_point[0]) + middle_point[1]
            intersec = fsolve(lambda x: line(x) - curve(x), middle_point[0])[0]
            touch_point = (intersec, line(intersec))
    else:
        # Rotate the defect points
        rotate = __rotate_array(theta)
        r_defect_points = rotate.dot(np.array(defect_points).T).T
        t_touch_point = __get_intersection(curve, r_defect_points)
        # Rotate the touch point back
        rotate = __rotate_array(-theta)
        touch_point = rotate.dot(np.array([t_touch_point]).T).T[0]

    return tuple(touch_point)


def __rotate_array(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def get_touch_point(defect_points,
                    up_centroid,
                    down_centroid,
                    up_touch_line,
                    down_touch_line,
                    theta,
                    image_for_grad=None):
    """Get the touch point according to the defect points and touch line.
    
    Arguments:
        defect_points {[type]} -- [description]
        up_centroid {[type]} -- [description]
        down_centroid {[type]} -- [description]
        up_touch_line {[type]} -- [description]
        down_touch_line {[type]} -- [description]
        theta {[type]} -- [description]
    
    Keyword Arguments:
        image_for_grad {[type]} -- [description] (default: {None})
    """
    if defect_points is None or up_centroid is None or down_centroid is None or up_touch_line is None or down_touch_line is None or theta is None:
        return None

    # Calculate the middle point of defect points
    (x1, y1), (x2, y2) = defect_points
    defect_middle = ((x1 + x2) // 2, (y1 + y2) // 2)

    distance_ratio = points_distance(defect_middle,
                                     up_centroid) / points_distance(
                                         defect_middle, down_centroid)

    if distance_ratio >= 1:
        touch_point = __get_intersection(up_touch_line, defect_points)
    elif 0.65 <= distance_ratio <= 1:
        touch_point1 = __get_intersection(up_touch_line, defect_points)
        touch_point2 = __get_intersection(down_touch_line, defect_points,
                                          theta)
        touch_point = ((touch_point1[0] + touch_point2[0]) / 2,
                       (touch_point1[1] + touch_point2[1]) / 2)
    elif distance_ratio < 0.65:
        touch_point = __get_intersection(down_touch_line, defect_points, theta)

    touch_point = tuple(map(int, touch_point))

    if image_for_grad is not None:
        touch_point = __get_min_grad(
            cv2.cvtColor(image_for_grad, cv2.COLOR_BGR2GRAY), defect_points,
            touch_point)

    return touch_point


## TODO: Need to improve the effiency. Can crop the image
def __get_min_grad(gray_img, defect_points, start_point):
    height, width = gray_img.shape[0], gray_img.shape[1]

    def __IsValid(x, y):
        """Check whether x and y is in the boundary of img
        """
        return 0 <= x < width and 0 <= y < height

    ck_d = points_distance(defect_points[0], defect_points[1]) / 10
    G, _ = sobel_filters(gray_img)
    (x1, y1), (x2, y2) = defect_points
    defect_middle = ((x1 + x2) // 2, (y1 + y2) // 2)

    if abs(y1 - y2) < 1e-9:
        min_y = max(0, start_point[1] - ck_d // 2)
        max_y = min(height - 1, start_point[1] + ck_d // 2)
        Y = np.arange(min_y, max_y)
        X = np.array([start_point[0]] * Y.shape[0])
    else:
        k = -(x2 - x1) / (y2 - y1)
        line = lambda x: k * (x - defect_middle[0]) + defect_middle[1]

        dx = (int)(abs(ck_d / 2 * np.cos(np.arctan(k))))
        min_x = max(0, start_point[0] - dx)
        max_x = min(width - 1, start_point[0] + dx)
        X = np.arange(min_x, max_x)
        Y = line(X).astype("int")
        X = X[(Y >= 0) & (Y < height)]
        Y = Y[(Y >= 0) & (Y < height)]

    if X.size > 0:
        index = np.where(G[Y, X] == max(G[Y, X]))[0][0]
        return (X[index], Y[index])
    else:
        return start_point


def show_whole_finger_motion(up_direcs, down_direcs):
    if up_direcs is None or down_direcs is None:
        return None

    vector_up = sum(up_direcs)
    vector_down = sum(down_direcs)

    # Normalization (?)

    norm_up = np.linalg.norm(vector_up)
    norm_down = np.linalg.norm(vector_down)

    if norm_up > 15 and norm_down > 30:
        print('Angle', (int)(my_arctan_degrees(*vector_up)), 'Dis', (int)(norm_up))
        print('Angle', (int)(my_arctan_degrees(*vector_down)), 'Dis', (int)(norm_down))
        print('-' * 60)

    return None


import cv2
import numpy as np
from scipy.optimize import fsolve, curve_fit

if __name__ == '__main__':
    '''
    This script is used to track the touch position based on geometry features

    It includes:
    1. Call Otsu threshold to finger_image the hand part and get the contour of hand
    2. Get 2 Convexity Defects with largest distance from the contour
    3. Segment the contour to up and bottom finger contour
    4. Fit the curve of up and bottom finger touch line
    5. Find the touch point
    6. Use optical flow to detect the movements of finger
    '''
    from time import sleep
    import sys, traceback
    import picamera_control
    from segment_otsu import threshold_masking
    from segment_edge import sobel_filters
    from contour_tools import get_defect_points, segment_diff_fingers, add_touch_line_to_contour, get_boundary_points, get_centroid, get_touch_line_curve
    from math_tools import points_distance, my_arctan_degrees, get_circle
    from draw_tools import DrawBoard, draw_vertical_lines, draw_points, draw_contours
    from opt_flow_LK import OpticalFlowLK
    from coord_calculator import GeometryCalculator

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

        # Optical FLow calculator
        opt_flow_up = OpticalFlowLK(IM_WIDTH, IM_HEIGHT, step=50)
        opt_flow_down = OpticalFlowLK(IM_WIDTH, IM_HEIGHT, step=50)

        # Tracker to convert point movement in image coordinate to the draw board coordinate
        tracker = GeometryCalculator()

        # Drawing boards
        DRAW_SCALER = 50
        DR_WIDTH, DR_HEIGHT = 320, 320
        hv_board = DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=5)
        hor_board = DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)
        ver_board = DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)

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

            # ---------------------------------------------
            # 1.2 Get defect points
            # ---------------------------------------------
            defect_points, _ = get_defect_points(contour,
                                                 MIN_VALID_CONT_AREA=100000,
                                                 MIN_DEFECT_DISTANCE=5000)

            # ---------------------------------------------
            # 1.3 Get up and down finger contour
            # ---------------------------------------------
            up_contour, down_contour = segment_diff_fingers(
                contour, defect_points)

            # ---------------------------------------------
            # 1.4 Get touch lines
            # ---------------------------------------------
            top_left, top_right, bottom_left, bottom_right = get_boundary_points(
                up_contour, down_contour, IM_HEIGHT, IM_WIDTH)

            up_touch_line, _ = get_touch_line_curve(
                IS_UP=True,
                contour=up_contour,
                bound_points=(top_left, top_right),
                fitting_curve=lambda X, Y: np.poly1d(np.polyfit(X, Y, 4)),
                defect_points=defect_points,
                draw_image=None)

            down_touch_line, theta = get_touch_line_curve(
                IS_UP=False,
                contour=down_contour,
                bound_points=(bottom_left, bottom_right),
                fitting_curve=lambda X, Y: np.poly1d(np.polyfit(X, Y, 3)),
                defect_points=defect_points,
                draw_image=None)

            # ---------------------------------------------
            # 1.5 Get centroids of contours
            # ---------------------------------------------
            up_centroid = get_centroid(up_contour)
            down_centroid = get_centroid(down_contour)

            # ---------------------------------------------
            # 1.6 Get touch point
            # ---------------------------------------------
            touch_point = get_touch_point(defect_points=defect_points,
                                          up_centroid=up_centroid,
                                          down_centroid=down_centroid,
                                          up_touch_line=up_touch_line,
                                          down_touch_line=down_touch_line,
                                          theta=theta,
                                          image_for_grad=None)

            # ---------------------------------------------
            # 1.7 Calculate optical flow
            # ---------------------------------------------
            gray_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

            up_direcs = opt_flow_up.calc_contour(gray_img,
                                                 up_contour,
                                                 draw_img=None)

            down_direcs = opt_flow_down.calc_contour(gray_img,
                                                     down_contour,
                                                     draw_img=None)

            show_whole_finger_motion(up_direcs, down_direcs)

            # real_x = tracker.coor_to_real_len(up_touch_line, up_centroid, touch_point)

            # ---------------------------------------------
            # 1.8 Draw elements
            # ---------------------------------------------
            # Two defect points (Green), touch point (Red), centroid points (Pink)
            draw_points(finger_image, defect_points, color=[0, 255, 0])
            draw_points(finger_image, touch_point, color=[255, 255, 255])
            draw_points(finger_image, up_centroid, color=[255, 0, 0])
            draw_points(finger_image, down_centroid, color=[255, 0, 0])

            # ---------------------------------------------
            # 1.9 Show image
            # ---------------------------------------------
            if SHOW_IMAGE:
                image_joint = np.concatenate((bgr_image, finger_image), axis=1)
                draw_vertical_lines(image_joint, 1)
                cv2.imshow('Finger', image_joint)

            # ---------------------------------------------
            # 2. Application
            # ---------------------------------------------

            # ---------------------------------------------
            # 2.1 Use tracker to calculate the movements
            # ---------------------------------------------
            dx, dy = tracker.calc_coord(up_touch_line, up_centroid,
                                              touch_point)

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
                tracker.calibrate_touch_scale(up_touch_line, up_centroid,
                                              touch_point)
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

'''
This script is used to track the touch position

It includes:
1. Call Otsu threshold to finger_image the hand part and get the contour of hand
2. Get 2 Convexity Defects with largest distance from the contour
3. Calculate the middle point of convexity defects, find the touch point and use Kalman filter to correct it
4. Use point_tracker to track the touch points movements
5. Draw the relative movements in a drawing board
'''

def configure_kalman_filter():
    """Configure the kalman filter

    Returns:
        kalman_filter [cv2.KalmanFilter]
    """
    # State number: 4, including (x，y，dx，dy) (position and velocity)
    # Measurement number: 2, (x, y) (position)
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]],
                                        np.float32)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        np.float32) * 0.03

    return kalman


def get_defect_points(contour, MIN_CHECK_AREA=0, MIN_DEFECT_DISTANCE=0):
    """Get the two convex defect points

    Arguments:
        contour {cv2.contour} -- [the contour of the finger]

    Returns:
        defect_points [list of tuple] -- [(left_x, left_y), (right_x, right_y)]
        hull_line [list of tuple] -- [(start1, end1), (start2, end2)]
    """
    # In case no contour or the contour area is too small (single fingertip)
    if contour is None or cv2.contourArea(contour) < MIN_CHECK_AREA:
        return None, None

    # Get the convex defects
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    # Get the defects with the largest 2 distance
    if defects is None:
        return None, None

    # Filter the defects with threshold
    defects = defects[defects[:, 0, 3] > MIN_DEFECT_DISTANCE]
    if defects.shape[0] < 2:
        return None, None

    sorted_defects = sorted(defects[:, 0, :], key=lambda x: x[3])

    defect_points = []
    hull_lines = []
    for s, e, f, d in sorted_defects[-2:]:
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        defect_points.append(far)
        hull_lines.append((start, end))

    if defect_points[0][0] > defect_points[1][0]:
        defect_points[0], defect_points[1] = defect_points[1], defect_points[0]
        hull_lines[0], hull_lines[1] = hull_lines[1], hull_lines[0]

    return defect_points, hull_lines


def get_min_gray(gray_img, start_pos, distance=0, slope=None):
    """Get the min gray in the slope direction and within the distance

    Arguments:
        gray_img {[type]} -- [description]
        start_pos {[type]} -- [description]

    Keyword Arguments:
        distance {int} -- [description] (default: {0})
        vertical {bool} -- [description] (default: {True})
        slope {int} -- [description] (default: {0})

    Returns:
        point_max_gradient {tuple} -- [description]
    """
    # x is from left to right, related to width
    # y is from up to down, related to height
    x, y = start_pos
    min_gray = float('inf')
    min_x, min_y = 0, 0
    height, width = gray_img.shape[0], gray_img.shape[1]

    def __IsValid(x, y):
        """Check whether x and y is in the boundary of img
        """
        return 0 <= x < width and 0 <= y < height

    if slope is None:
        # Let column(x) to be fixed and change the row(y)
        for dy in range(int(-distance / 2), int(distance / 2)):
            if __IsValid(x, y + dy) and gray_img[y + dy, x] < min_gray:
                min_gray = gray_img[y + dy, x]
                min_x, min_y = x, y + dy
    else:
        c_x, c_y = x, y
        # up
        while __IsValid(c_x, c_y) and points_distance((c_x, c_y), (x, y)) < distance / 2:
            if gray_img[c_y, c_x] < min_gray:
                min_gray = gray_img[c_y, c_x]
                min_x, min_y = c_x, c_y
            c_x = c_x + 1
            c_y = int(c_y + 1 * slope)

        c_x, c_y = x, y
        # down
        while __IsValid(c_x, c_y) and points_distance((c_x, c_y), (x, y)) < distance / 2:
            if gray_img[c_y, c_x] < min_gray:
                min_gray = gray_img[c_y, c_x]
                min_x, min_y = c_x, c_y
            c_x = c_x - 1
            c_y = int(c_y - 1 * slope)

    # Check the ans
    if min_gray == float('inf'):
        return start_pos
    else:
        return (min_x, min_y)


def get_touch_point(defect_points,
                    finger_img,
                    kalman_filter=None):
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
        touch_point = get_min_gray(gray_img,
                                   middle_point,
                                   distance=search_distance)
    else:
        grad_direc = -(x2 - x1) / (y2 - y1)
        touch_point = get_min_gray(gray_img,
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


import cv2
import numpy as np
from time import sleep
import sys
import traceback

if __name__ == '__main__':
    import picamera_control
    from draw_tools import draw_board, draw_vertical_lines, draw_points
    from math_tools import points_distance
    from segment_otsu import threshold_masking
    from move_tracker import touch_trakcer

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

        # Tracker to convert point movement in image coordinate to the draw board coordinate
        tracker = touch_trakcer()

        # Drawing boards
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

            # Get the mask and its contour using the Otsu thresholding method and apply the mask to image
            mask, contour = threshold_masking(bgr_image)
            finger_image = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)

            # Get convexity defects points from the contour
            defect_points, _ = get_defect_points(contour,
                                                 MIN_CHECK_AREA=100000,
                                                 MIN_DEFECT_DISTANCE=5000)

            # Get touch point from the defect points and the img
            touch_point, filter_touch_point = get_touch_point(
                                                            defect_points,
                                                            finger_image,
                                                            kalman_filter=kalman_filter)

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
            dx, dy = tracker.calc_scaled_move(touch_point)

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

'''
This script is used to track the touch position

It includes:
1. Call Otsu threshold to segment the hand part and get the contour of hand
2. Use Convexity Defects to get feature points from the contour
3. Calculate the middle point and use Kalman filter to correct it
4. Draw the relative movements in a drawing board
'''

import cv2
import numpy as np
from time import sleep
from enum import IntEnum
import picamera_control
from draw_board import draw_board
import segment_otsu


def _get_defect_points(contour, CNT_AREA_THRES=0, draw_img=None):
    """Get the two convex defect points

    Arguments:
        contour {cv2.contour} -- [the contour with the max area]

    Returns:
        defect_points [list] -- [left and right defect points]
    """
    # In case no contour or the contour area is too small (single fingertip)
    if contour is None or cv2.contourArea(contour) < CNT_AREA_THRES:
        return None

    # Get the convex defects
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    # Get the defects with the largest 2 distance
    sorted_defects = sorted(defects[:, 0], key=lambda x: x[3])
    if len(sorted_defects) < 2:
        return None

    defect_points = []
    for s, e, f, d in sorted_defects[-2:]:
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        defect_points.append(far)
        if draw_img is not None:
            cv2.line(draw_img, start, end, [0, 255, 0], 2)
            cv2.circle(draw_img, far, 5, [0, 255, 0], -1)

    defect_points.sort(key=lambda x: x[0])

    return defect_points


def _calculate_max_gradient(gray_img, start_pos, distance=0, vertical=True, slope=0):
    """Get the max gradient in the slope direction and within the distance

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
    max_grad = -1
    grad_x, grad_y = 0, 0

    if vertical:
        # Let column(x) to be fixed and change the row(y)
        for dy in range(int(-distance / 2), int(distance / 2)):
            if _IsValid(x, y + dy + 1, gray_img) and _IsValid(x, y + dy, gray_img) and abs(gray_img[y + dy + 1, x] - gray_img[y + dy, x]) > max_grad:
                max_grad = abs(gray_img[y + dy + 1, x] - gray_img[y + dy, x])
                grad_x, grad_y = x, y + dy
    else:
        last_x, last_y = x, y
        c_x, c_y = x, y
        # up
        while _IsValid(c_x, c_y, gray_img) and np.sqrt(np.sum(np.square(np.array([c_x, c_y]) - np.array([x, y])))) < distance / 2:
            if abs(gray_img[c_y, c_x] - gray_img[last_y, last_x]) > max_grad:
                max_grad = abs(gray_img[c_y, c_x] - gray_img[last_y, last_x])
                grad_x, grad_y = c_x, c_y
            last_x, last_y = c_x, c_y
            c_x = c_x + 1
            c_y = int(c_y + 1 * slope)

        last_x, last_y = x, y
        c_x, c_y = x, y
        # down
        while _IsValid(c_x, c_y, gray_img) and np.sqrt(np.sum(np.square(np.array([c_x, c_y]) - np.array([x, y])))) < distance / 2:
            if abs(gray_img[c_y, c_x] - gray_img[last_y, last_x]) > max_grad:
                max_grad = abs(gray_img[c_y, c_x] - gray_img[last_y, last_x])
                grad_x, grad_y = c_x, c_y
            last_x, last_y = c_x, c_y
            c_x = c_x - 1
            c_y = int(c_y - 1 * slope)

    # Check the ans
    if max_grad == -1:
        return start_pos
    else:
        return (grad_x, grad_y)


def _get_min_gray(gray_img, start_pos, distance=0, vertical=True, slope=0):
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
    min_gray = 255
    grad_x, grad_y = 0, 0

    if vertical:
        # Let column(x) to be fixed and change the row(y)
        for dy in range(int(-distance / 2), int(distance / 2)):
            if _IsValid(x, y + dy, gray_img) and gray_img[y + dy, x] < min_gray:
                min_gray = gray_img[y + dy, x]
                grad_x, grad_y = x, y + dy
    else:
        c_x, c_y = x, y
        # up
        while _IsValid(c_x, c_y, gray_img) and np.sqrt(np.sum(np.square(np.array([c_x, c_y]) - np.array([x, y])))) < distance / 2:
            if gray_img[c_y, c_x] < min_gray:
                min_gray = gray_img[c_y, c_x]
                grad_x, grad_y = c_x, c_y
            c_x = c_x + 1
            c_y = int(c_y + 1 * slope)

        c_x, c_y = x, y
        # down
        while _IsValid(c_x, c_y, gray_img) and np.sqrt(np.sum(np.square(np.array([c_x, c_y]) - np.array([x, y])))) < distance / 2:
            if gray_img[c_y, c_x] < min_gray:
                min_gray = gray_img[c_y, c_x]
                grad_x, grad_y = c_x, c_y
            c_x = c_x - 1
            c_y = int(c_y - 1 * slope)

    # Check the ans
    if min_gray == -1:
        return start_pos
    else:
        return (grad_x, grad_y)


def _IsValid(x, y, img):
    """Check whether x and y is in the boundary of img

    Arguments:
        x {[type]} -- [description]
        y {[type]} -- [description]
        img {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    height, width = img.shape[0], img.shape[1]
    return 0 <= x < width and 0 <= y < height


def get_touch_point(max_contour, CNT_AREA_THRES=0, kalman_filter=None, draw_img=None):
    """Extract feature points with the max contour

    Arguments:
        draw_img {np.array} -- [Target image to draw the results. The segmented image]
        max_contour {cv2.contour} -- [Parameter to generate the touch point]

    Returns:
        touchPoint [tuple] -- [The touch coordinate of the fingertips]
        defectPoints [list] -- [A list of tuple which indicate the coordinate of the defects]
    """
    # Get the convex defects point
    defect_points = _get_defect_points(max_contour, CNT_AREA_THRES, draw_img)
    if defect_points is None:
        return None

    # Calculate the middle point
    middle_point = ((defect_points[0][0] + defect_points[1][0]) // 2,
                    (defect_points[0][1] + defect_points[1][1]) // 2)

    # Calculate the touch point according to the gradient change
    if draw_img is None:
        touch_point = middle_point
    else:
        gray_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
        search_distance = np.sqrt(
            np.sum(np.square(np.array(defect_points[0]) - np.array(defect_points[1])))) / 5

        if abs(defect_points[1][1] - defect_points[0][1]) < 1e-6:
            grad_direc = 0
            touch_point = _get_min_gray(
                gray_img, middle_point, distance=search_distance, vertical=True)
        else:
            grad_direc = -(defect_points[1][0] - defect_points[0]
                           [0]) / (defect_points[1][1] - defect_points[0][1])
            touch_point = _get_min_gray(
                gray_img, middle_point, distance=search_distance, vertical=False, slope=grad_direc)

        cv2.circle(draw_img, touch_point, 5, [0, 0, 255], -1)

    # Adopt kalman filter to the touch point
    if kalman_filter is None:
        return touch_point

    kalman_filter.correct(
        np.array([[np.float32(touch_point[0])], [np.float32(touch_point[1])]]))
    predict_point = kalman_filter.predict()
    filter_touch_point = (int(predict_point[0]), int(predict_point[1]))

    if draw_img is not None:
        cv2.circle(draw_img, filter_touch_point, 5, [255, 0, 0], -1)

    return filter_touch_point


def configure_kalman_filter():
    """Configure the kalman filter

    Returns:
        [type] -- [description]
    """
    # State number: 4, including (x，y，dx，dy) (position and velocity)
    # Measurement number: 2, (x, y) (position)
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

    return kalman


class point_type(IntEnum):
    MIN_X = 0
    MAX_X = 1
    MIN_Y = 2
    MAX_Y = 3


CALIBRATE_BASE = [181, 353, 87, 300]


class point_trakcer:
    def __init__(self):
        self.cur_point_type = point_type.MIN_X
        self.base_point = CALIBRATE_BASE

    def calibrate_base_point(self, point):
        """Reset the old touch point

        Arguments:
            point {[type]} -- [description]
        """
        if self.cur_point_type == point_type.MIN_X or self.cur_point_type == point_type.MAX_X:
            self.base_point[int(self.cur_point_type)] = point[0]
        else:
            self.base_point[int(self.cur_point_type)] = point[1]
        print("Store base point", self.cur_point_type)
        print("Current base points", self.base_point)
        self.cur_point_type = point_type((int(self.cur_point_type) + 1) % 4)

    def calculate_movements(self, point):
        """Canculate the relative movements of current touch points to the old touch points

        Arguments:
            point {tuple} -- [current touch position]

        Returns:
            dx {float} -- [relative movement in x direction]
            dy {float}  -- [relative movement in y direction]
        """
        MAX_UNIT = 100

        dx = self._scaler(point[0], [self.base_point[int(point_type.MIN_X)], -
                                     MAX_UNIT], [self.base_point[int(point_type.MAX_X)], MAX_UNIT])
        dy = self._scaler(point[1], [self.base_point[int(point_type.MIN_Y)], -
                                     MAX_UNIT], [self.base_point[int(point_type.MAX_Y)], MAX_UNIT])

        return dx, dy

    def _scaler(self, value, min_base_target, max_base_target):
        """Project value from [min_base, max_base] to [min_target, max_target]

        Arguments:
            value {float} -- [description]
            min_base_target {list} -- [min_base, min_target]
            max_base_target {list} -- [max_base, max_target]

        Returns:
            value -- [projected value]
        """
        min_base, min_target = min_base_target
        max_base, max_target = max_base_target
        return (value - min_base) / (max_base - min_base) * (max_target - min_target) + min_target


if __name__ == '__main__':
    """
    This function get the frame from the camera, and use thresholding to segment the hand part
    """
    try:
        WIDTH, HEIGHT = 640, 480
        # Note: Higher framerate will bring noise to the segmented image
        camera, rawCapture = picamera_control.configure_camera(
            WIDTH, HEIGHT, FRAME_RATE=35)

        # Kalman filter to remove noise from the point movement
        kalman_filter = configure_kalman_filter()
        kalman_filter_on = True

        # Tracker to convert point movement in image coordinate to the draw board coordinate
        tracker = point_trakcer()

        # Drawing boards
        DR_WIDTH, DR_HEIGHT = 320, 320
        hv_board = draw_board(DR_WIDTH, DR_HEIGHT, MAX_POINTS=5)
        hor_board = draw_board(DR_WIDTH, DR_HEIGHT, MAX_POINTS=1)
        ver_board = draw_board(DR_WIDTH, DR_HEIGHT, MAX_POINTS=1)

        print("To calibrate, press 'C' and follow the order LEFT, RIGHT, UP, DOWN")
        print("Press F to turn ON/OFF the kalman filter")

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            bgr_image = frame.array

            # Get the mask using the Otsu thresholding method
            mask, contour = segment_otsu.threshold_masking(bgr_image)

            # Apply the mask to the image
            segment = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)

            # Get touch from the contour and draw points in the segment image
            if kalman_filter_on:
                touch_point = get_touch_point(
                    contour, CNT_AREA_THRES=100000, kalman_filter=kalman_filter, draw_img=segment)
            else:
                touch_point = get_touch_point(
                    contour, CNT_AREA_THRES=100000, kalman_filter=None, draw_img=segment)

            if touch_point is not None:
                # Track the touch point
                dx, dy = tracker.calculate_movements(touch_point)
                k = 1
                point_size = 10
                hor_board.draw_filled_point((-dx * k, 0), radius=point_size)
                ver_board.draw_filled_point((0, dy * k), radius=point_size)
                hv_board.draw_filled_point(
                    (-dx * k, dy * k), radius=point_size)

            # Display
            cv2.imshow('Segment', segment)

            H_V_joint = np.concatenate(
                (hv_board.board, hor_board.board, ver_board.board), axis=1)

            joint_width, joint_height = H_V_joint.shape[1], H_V_joint.shape[0]
            cv2.line(H_V_joint, (joint_width // 3, 0),
                     (joint_width // 3, joint_height), [255, 255, 255], 3)
            cv2.line(H_V_joint, (joint_width * 2 // 3, 0),
                     (joint_width * 2 // 3, joint_height), [255, 255, 255], 3)
            cv2.imshow('H V Movement', H_V_joint)

            # if the user pressed ESC, then stop looping
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == 27:
                break
            elif keypress == ord('c'):
                tracker.calibrate_base_point(touch_point)
                hv_board.reset_board()
                hor_board.reset_board()
                ver_board.reset_board()
            elif keypress == ord('f'):
                kalman_filter_on = not kalman_filter_on
                if kalman_filter_on:
                    print("Kalman Filter ON")
                else:
                    print("Kalman Filter OFF")


            rawCapture.truncate(0)

        camera.close()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        camera.close()
        cv2.destroyAllWindows()

'''
This script is used to track the touch position

It includes:
1. Call Otsu threshold to finger_image the hand part and get the contour of hand
2. Use Convexity Defects to get feature points from the contour
3. Calculate the middle point and use Kalman filter to correct it
4. Draw the relative movements in a drawing board
'''

import cv2
import numpy as np
from time import sleep
import picamera_control
from draw_board import draw_board
import segment_otsu
from relative_mov_tracker import point_trakcer
import sys, traceback


def __get_defect_points(contour, MIN_CHECK_AREA=0, MIN_DEFECT_DISTANCE=0):
    """Get the two convex defect points

    Arguments:
        contour {cv2.contour} -- [the contour of the finger]

    Returns:
        defect_points [list of tuple] -- [left and right defect points]
    """
    # In case no contour or the contour area is too small (single fingertip)
    if contour is None or cv2.contourArea(contour) < MIN_CHECK_AREA:
        return None

    # Get the convex defects
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    # Get the defects with the largest 2 distance
    if defects is None:
        return None
    defects = defects[defects[:, 0, 3] > MIN_DEFECT_DISTANCE]
    if defects.shape[0] < 2:
        return None

    sorted_defects = sorted(defects[:, 0, :], key=lambda x: x[3])

    defect_points = []
    for s, e, f, d in sorted_defects[-2:]:
        # start = tuple(contour[s][0])
        # end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        defect_points.append(far)
        
    defect_points.sort(key=lambda x: x[0])

    return defect_points


def __calculate_max_gradient(gray_img, start_pos, distance=0, vertical=True, slope=0):
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
            if __IsValid(x, y + dy + 1, gray_img) and __IsValid(x, y + dy, gray_img) and abs(gray_img[y + dy + 1, x] - gray_img[y + dy, x]) > max_grad:
                max_grad = abs(gray_img[y + dy + 1, x] - gray_img[y + dy, x])
                grad_x, grad_y = x, y + dy
    else:
        last_x, last_y = x, y
        c_x, c_y = x, y
        # up
        while __IsValid(c_x, c_y, gray_img) and np.sqrt(np.sum(np.square(np.array([c_x, c_y]) - np.array([x, y])))) < distance / 2:
            if abs(gray_img[c_y, c_x] - gray_img[last_y, last_x]) > max_grad:
                max_grad = abs(gray_img[c_y, c_x] - gray_img[last_y, last_x])
                grad_x, grad_y = c_x, c_y
            last_x, last_y = c_x, c_y
            c_x = c_x + 1
            c_y = int(c_y + 1 * slope)

        last_x, last_y = x, y
        c_x, c_y = x, y
        # down
        while __IsValid(c_x, c_y, gray_img) and np.sqrt(np.sum(np.square(np.array([c_x, c_y]) - np.array([x, y])))) < distance / 2:
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


def __get_min_gray(gray_img, start_pos, distance=0, vertical=True, slope=0):
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
            if __IsValid(x, y + dy, gray_img) and gray_img[y + dy, x] < min_gray:
                min_gray = gray_img[y + dy, x]
                grad_x, grad_y = x, y + dy
    else:
        c_x, c_y = x, y
        # up
        while __IsValid(c_x, c_y, gray_img) and np.sqrt(np.sum(np.square(np.array([c_x, c_y]) - np.array([x, y])))) < distance / 2:
            if gray_img[c_y, c_x] < min_gray:
                min_gray = gray_img[c_y, c_x]
                grad_x, grad_y = c_x, c_y
            c_x = c_x + 1
            c_y = int(c_y + 1 * slope)

        c_x, c_y = x, y
        # down
        while __IsValid(c_x, c_y, gray_img) and np.sqrt(np.sum(np.square(np.array([c_x, c_y]) - np.array([x, y])))) < distance / 2:
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


def __IsValid(x, y, img):
    """Check whether x and y is in the boundary of img

    Arguments:
        x {[type]} -- [description]
        y {[type]} -- [description]
        img {[type]} -- [description]

    Returns:
        flag [bool] -- [description]
    """
    height, width = img.shape[0], img.shape[1]
    return 0 <= x < width and 0 <= y < height


def get_touch_point(finger_contour, finger_img, MIN_CHECK_AREA=0, MIN_DEFECT_DISTANCE=0, kalman_filter=None, DRAW_POINTS=True):
    """Extract feature points with the max contour

    Arguments:
        finger_contour {cv2.contour} -- [Parameter to generate the touch point]
        finger_img {np.array} -- [BGR Image of fingers]

    Keyword Arguments:
        MIN_CHECK_AREA {int} -- [description] (default: {0})
        kalman_filter {[type]} -- [description] (default: {None})
        DRAW_POINTS {bool} -- [description] (default: {True})

    Returns:
        touchPoint [tuple] -- [The touch coordinate of the fingertips]
        defectPoints [list of tuple] -- [A list of tuple which indicate the coordinate of the defects]]
    """
    # Get the convex defects point
    defect_points = __get_defect_points(finger_contour, MIN_CHECK_AREA, MIN_DEFECT_DISTANCE)

    if defect_points is None:
        return None, None

    # Calculate the middle point
    (x1, y1), (x2, y2) = defect_points
    middle_point = ((x1 + x2) // 2, (y1 + y2) // 2)

    # Calculate the touch point according to the gradient change
    gray_img = cv2.cvtColor(finger_img, cv2.COLOR_BGR2GRAY)
    search_distance = np.sqrt(
        np.sum(np.square(np.array(defect_points[0]) - np.array(defect_points[1])))) / 5

    if abs(y2 - y1) < 1e-6:
        touch_point = __get_min_gray(
            gray_img, middle_point, distance=search_distance, vertical=True)
    else:
        grad_direc = - (x2 - x1) / (y2 - y1)
        touch_point = __get_min_gray(
            gray_img, middle_point, distance=search_distance, vertical=False, slope=grad_direc)

    # If kalman filter adopted, use it to correct the observation
    filter_touch_point = None
    if kalman_filter is not None:
        kalman_filter.correct(
            np.array([[np.float32(touch_point[0])], [np.float32(touch_point[1])]]))
        predict_point = kalman_filter.predict()
        filter_touch_point = (int(predict_point[0]), int(predict_point[1]))

    if DRAW_POINTS:
        # Two defect points (Green)
        cv2.circle(finger_img, defect_points[0], 5, [0, 255, 0], -1)
        cv2.circle(finger_img, defect_points[1], 5, [0, 255, 0], -1)
        # Raw touch point (Red)
        cv2.circle(finger_img, touch_point, 5, [0, 0, 255], -1)
        # Filter touch point (Blue)
        if filter_touch_point is not None:
            cv2.circle(finger_img, filter_touch_point, 5, [255, 0, 0], -1)

    final_touch_point = touch_point if kalman_filter is None else filter_touch_point

    return final_touch_point, defect_points


def configure_kalman_filter():
    """Configure the kalman filter

    Returns:
        kalman_filter [cv2.KalmanFilter]
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

def segment_diff_fingers(contour, defect_points, touch_point):
    """Segment the contour to the up finger and down finger
    
    Arguments:
        contour {[type]} -- [description]
        defect_points {[type]} -- [description]
        touch_point {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    to_add = np.reshape(touch_point, [1, 1, 2])
    (x1, y1), (x2, y2) = defect_points
    
    if abs(x2 - x1) < 1e-6:
        up_finger = contour[contour[:, 0, 0] <= x1]
        down_finger = contour[contour[:, 0, 0] >= x1]
    else:
        grad_direc = (y2 - y1) / (x2 - x1)
        offset = y1 - grad_direc * x1
        up_finger = contour[grad_direc * contour[:, 0, 0] + offset - contour[:, 0, 1] >= 0]
        down_finger = contour[grad_direc * contour[:, 0, 0] + offset - contour[:, 0, 1] <= 0]
    
    index1 = np.where((up_finger[:, 0, 0] == x1) & (up_finger[:, 0, 1] == y1))[0]
    if index1 is not None and len(index1) != 0:
        up_finger = np.insert(up_finger, index1[-1] + 1, to_add, axis=0)

    down_finger = np.insert(down_finger, down_finger.shape[0], to_add, axis=0)
    
    return up_finger, down_finger

if __name__ == '__main__':
    """
    This function get the frame from the camera, and use thresholding to finger_image the hand part
    """
    try:
        WIDTH, HEIGHT = 640, 480
        # Note: Higher framerate will bring noise to the segmented image
        camera, rawCapture = picamera_control.configure_camera(
            WIDTH, HEIGHT, FRAME_RATE=35)

        # Kalman filter to remove noise from the point movement
        kalman_filter = configure_kalman_filter()

        # Tracker to convert point movement in image coordinate to the draw board coordinate
        tracker = point_trakcer()

        # Drawing boards
        mov_scaler = 1
        DR_WIDTH, DR_HEIGHT = 320, 320
        hv_board = draw_board(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=5)
        hor_board = draw_board(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)
        ver_board = draw_board(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)

        print('-'*60)
        print("To calibrate, press 'C' and follow the order LEFT, RIGHT, UP, DOWN")
        print("Press F to turn ON/OFF the kalman filter")
        print('-'*60)

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            bgr_image = frame.array

            # Get the mask and its contour using the Otsu thresholding method and apply the mask to image
            mask, contour = segment_otsu.threshold_masking(bgr_image)
            finger_image = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)
            two_finger_image = finger_image.copy()

            # Get touch point and defect points from the contour and draw points in the finger_image image
            touch_point, defect_points = get_touch_point(
                contour, finger_image, MIN_CHECK_AREA=100000, MIN_DEFECT_DISTANCE=5000, kalman_filter=kalman_filter, DRAW_POINTS=True)

            
            if touch_point is not None:
                # Track the touch point
                dx, dy = tracker.calculate_movements(touch_point)
                
                # Draw the touch point track
                hor_board.draw_filled_point((-dx * mov_scaler, 0))
                ver_board.draw_filled_point((0, dy * mov_scaler))
                hv_board.draw_filled_point((-dx * mov_scaler, dy * mov_scaler))
                
                # Segment the two fingers
                up_finger_contour, down_finger_contour = segment_diff_fingers(contour, defect_points, touch_point)
                
                cv2.drawContours(two_finger_image, [up_finger_contour], 0, [0, 0, 255], 3)
                cv2.drawContours(two_finger_image, [down_finger_contour], 0, [255, 0, 0], 3)
                cv2.circle(two_finger_image, touch_point, 5, [255, 0, 0], -1)

                defect_points = np.sqrt(
                    np.sum(np.square(np.array(defect_points[0]) - np.array(defect_points[1]))))
                print(cv2.contourArea(up_finger_contour), cv2.contourArea(down_finger_contour), defect_points)
                

            # Display
            cv2.imshow('Finger', finger_image)
            cv2.imshow('2 Fingers Segment', two_finger_image)

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
        print('-'*60)
        traceback.print_exc(file=sys.stdout)
        print('-'*60)

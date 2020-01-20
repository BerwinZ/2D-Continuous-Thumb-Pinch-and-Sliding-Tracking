'''
This script is used to track the touch position

It includes:
1. Call Otsu threshold to segment the hand part
2. Use Convexity Defects to get touch point
3. Draw the movements in a drawing board
'''

import cv2
import numpy as np
from time import sleep
import picamera_control
from draw_board import draw_board
import segment_otsu


def get_touch_point(img, max_contour):
    """Extract feature points with the max contour

    Arguments:
        img {np.array} -- [Target image to draw the results. The segmented image]
        max_contour {cv2.contour} -- [Parameter to generate the touch point]

    Returns:
        touchPoint [tuple] -- [The touch coordinate of the fingertips]
        defectPoints [list] -- [A list of tuple which indicate the coordinate of the defects]
    """
    if max_contour is None:
        return None, None

    # Get the convex defects
    hull = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull)

    # Get the defects with the largest 2 distance
    sorted_defects = sorted(defects[:, 0], key=lambda x: x[3])
    if len(sorted_defects) < 2:
        return None, None

    defect_points = []
    for s, e, f, d in sorted_defects[-2:]:
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])
        defect_points.append(far)
        cv2.line(img, start, end, [0, 255, 0], 2)
        cv2.circle(img, far, 5, [0, 0, 255], -1)

    # Get the touch point
    defect_points.sort(key=lambda x: x[0])
    touch_point = ((defect_points[0][0] + defect_points[1][0]) // 2,
                   (defect_points[0][1] + defect_points[1][1]) // 2)
    cv2.circle(img, touch_point, 5, [255, 0, 0], -1)

    return touch_point, defect_points


initial_touch = None


def reset_tracking(point):
    """Reset the old touch point

    Arguments:
        point {[type]} -- [description]
    """
    global initial_touch
    initial_touch = point


def calculate_movements(point):
    """Canculate the relative movements of current touch points to the old touch points

    Arguments:
        point {tuple} -- [current touch position]

    Returns:
        horizonal {float} -- [relative movement in x direction]
        vertical {float}  -- [relative movement in y direction]
    """
    global initial_touch

    if initial_touch == None:
        initial_touch = point
        return 0, 0

    horizonal = point[0] - initial_touch[0]
    vertical = point[1] - initial_touch[1]

    # TODO: Should include some filter part here

    return horizonal, vertical


if __name__ == '__main__':
    """
    This function get the frame from the camera, and use thresholding to segment the hand part
    """
    try:
        WIDTH, HEIGHT = 640, 480
        camera, rawCapture = picamera_control.configure_camera(WIDTH, HEIGHT)

        hv_board = draw_board(WIDTH, HEIGHT)
        hor_board = draw_board(WIDTH, HEIGHT)
        ver_board = draw_board(WIDTH, HEIGHT)

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            bgr_image = frame.array

            # Get the mask using the Otsu thresholding method
            mask, max_contour = segment_otsu.threshold_masking(bgr_image)

            # Apply the mask to the image
            segment = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)

            # Get touch point in the segment image
            touch_point, _ = get_touch_point(segment, max_contour)

            if touch_point is not None:
                # Track the touch point
                dx, dy = calculate_movements(touch_point)
                center_x, center_y = WIDTH / 2, HEIGHT / 2
                k = 1
                size = 10
                hor_board.draw_filled_point(
                    (int(-dx * k + center_x), int(center_y)), size)
                ver_board.draw_filled_point(
                    (int(center_x), int(dy * k + center_y)), size)
                hv_board.draw_filled_point(
                    (int(-dx * k + center_x), int(dy * k + center_y)), size)

            # Display
            # cv2.imshow("original", bgr_image)
            # cv2.imshow('Mask', mask)
            cv2.imshow('Segment', segment)
            cv2.imshow('HV Board', hv_board.board)
            cv2.imshow('H Board', hor_board.board)
            cv2.imshow('V Board', ver_board.board)

            # if the user pressed ESC, then stop looping
            keypress = cv2.waitKey(25) & 0xFF
            if keypress == 27:
                break
            elif keypress == ord('r'):
                reset_tracking(touch_point)
                hv_board.reset_board()
                hor_board.reset_board()
                ver_board.reset_board()

            rawCapture.truncate(0)

        camera.close()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        camera.close()
        cv2.destroyAllWindows()

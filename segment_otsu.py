'''
This script is used to segment the hand from the video stream with the skin color
'''

import cv2
import numpy as np
from time import sleep
import picamera_control
import draw_board


def threshold_masking(img):
    """Get the mask for the img
    1. Use Otsu thresholding
    2. Erode and dilate to remove noise
    3. Get the area with the max contour 

    Arguments:
        img {np.array} -- [BGR Image]

    Returns:
        Mask [np.array] -- [0/255 Mask]
        Max_contour [3-d array] 
    """
    # Convert to YCrCb
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    # Cr and Cb Channels
    # mask_ycrcb = cv2.inRange(frame_ycrcb, np.array(
    #     [0, 145, 85]), np.array([255, 185, 155]))

    # Just use Y channel
    # mask = cv2.inRange(img_ycrcb[:,:,0], np.array([0]), np.array([150]))

    # Otsu Thresholding
    _, mask = cv2.threshold(
        img_ycrcb[:, :, 1], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Erode or dilate the edges that has been removed
    kernel_size = min(img.shape[0], img.shape[1]) // 50
    element = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.erode(mask, element)
    mask = cv2.dilate(mask, element)

    # Get the max contours, CHAIN_APPROX_NONE means get all the points
    _, contours, _ = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    max_index = 0
    max_val = -1
    for idx, c in enumerate(contours):
        if cv2.contourArea(c) > max_val:
            max_val = cv2.contourArea(c)
            max_index = idx

    # Draw the max contours and fill it
    canvas = np.zeros(mask.shape).astype('uint8')
    mask = cv2.drawContours(canvas, contours, max_index, 255, -1)

    return mask, contours[max_index]


def get_touch_point(img, max_contour):
    """Extract feature points with the max contour

    Arguments:
        img {[type]} -- [description]
        max_contour {[type]} -- [description]

    Returns:
        touchPoint [tuple] -- [The touch coordinate of the fingertips]
        defectPoints [list] -- [A list of tuple which indicate the coordinate of the defects]
    """
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


def draw_points(frame, points):
    """Draw points circles in the frame

    Arguments:
        frame {np.array} -- Image
        points {list} -- Points need to be drawn
    """
    for p in points:
        cv2.circle(frame, tuple(p), 1, [0, 0, 255], -1)


old_touch_point = None


def reset_tracking(point):
    """Reset the old touch point

    Arguments:
        point {[type]} -- [description]
    """
    global old_touch_point
    old_touch_point = point


def track_touch_point(point):
    """Canculate the relative movements of current touch points to the old touch points

    Arguments:
        point {tuple} -- [description]

    Returns:
        [type] -- [description]
    """
    global old_touch_point

    if old_touch_point == None:
        old_touch_point = point
        return 0, 0

    horizonal = point[0] - old_touch_point[0]
    vertical = point[1] - old_touch_point[1]

    # TODO: Should include some filter part here

    return horizonal, vertical


if __name__ == '__main__':
    """
    This function get the frame from the camera, and use thresholding to segment the hand part
    """
    try:
        camera, rawCapture = picamera_control.configure_camera(640, 480)
        hor_board = draw_board.configure_board(640, 480)
        ver_board = draw_board.configure_board(640, 480)

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            bgr_image = frame.array

            # img_ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCR_CB)
            # cv2.imshow("Y", img_ycrcb[:,:,0])
            # cv2.imshow("Cr", img_ycrcb[:,:,1])
            # cv2.imshow("Cb", img_ycrcb[:,:,2])

            # Get the mask using the Otsu thresholding method
            mask, max_contour = threshold_masking(bgr_image)

            # Apply the mask to the image
            segment = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)
            # cv2.drawContours(segment, [max_contour], 0, [0, 0, 255])

            # Get touch point in the segment image
            touch_point, _ = get_touch_point(segment, max_contour)

            if touch_point != None:
                # Track the touch point
                hor, ver = track_touch_point(touch_point)
                k = 1
                draw_board.draw_point(hor_board,
                                      (int(-hor * k + bgr_image.shape[1] / 2), int(bgr_image.shape[0] / 2)), 30)
                draw_board.draw_point(ver_board,
                                      (int(bgr_image.shape[1] / 2), int(ver * k + bgr_image.shape[0] / 2)), 30)

            # Display
            cv2.imshow("original", bgr_image)
            cv2.imshow('Mask', mask)
            cv2.imshow('Segment', segment)
            cv2.imshow('H Board', hor_board)
            cv2.imshow('V Board', ver_board)

            # if the user pressed ESC, then stop looping
            keypress = cv2.waitKey(25) & 0xFF
            if keypress == 27:
                break
            elif keypress == ord('r'):
                reset_tracking(touch_point)
                hor_board = draw_board.reset_board(hor_board)
                ver_board = draw_board.reset_board(ver_board)

            rawCapture.truncate(0)

        camera.close()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        camera.close()
        cv2.destroyAllWindows()

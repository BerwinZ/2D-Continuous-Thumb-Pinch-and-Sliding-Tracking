"""
1. get_touch_line_samples
    get the sample points in the touch line according to searching gray scale
"""

def get_touch_line_samples(finger_img, defect_points, line_points_num=10):

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
        touch_line = map(get_min_gray_point, [gray_img] * (line_points_num - 1),
                         check_points,
                         [search_distance] * (line_points_num - 1))
    else:
        grad_direc = -(x2 - x1) / (y2 - y1)
        touch_line = map(get_min_gray_point, [gray_img] * (line_points_num - 1),
                         check_points,
                         [search_distance] * (line_points_num - 1),
                         [grad_direc] * (line_points_num - 1))

    return list(touch_line)


import cv2
import numpy as np
from time import sleep
import sys
import traceback

if __name__ == '__main__':
    '''
    This script is used to track the touch position with two boundary points

    It includes:
    1. Call Otsu threshold to finger_image the hand part and get the contour of hand
    2. Get 2 Convexity Defects with largest distance from the contour
    3. Segment the contour to up and bottom finger contour
    4. Get the top_left and bottom_right boundary points
    5. Use bound_tracker to track the movements
    6. Draw the relative movements in a drawing board
    '''

    import picamera_control
    from draw_tools import draw_vertical_lines, draw_points, DrawBoard
    from image_segment import threshold_masking
    from coord_calculator import BoundCalculator
    from feature_extraction import get_defect_points, segment_diff_fingers, add_touch_line_to_contour, get_boundary_points
    from math_tools import get_min_gray_point

    try:
        WIDTH, HEIGHT = 640, 480
        # Note: Higher framerate will bring noise to the segmented image
        camera, rawCapture = picamera_control.configure_camera(WIDTH,
                                                               HEIGHT,
                                                               FRAME_RATE=35)

        DRAW_CONTOUR = True

        # Point tracker
        tracker = BoundCalculator(HEIGHT, WIDTH)

        # Drawing boards
        DR_WIDTH, DR_HEIGHT = 320, 320
        hv_board = DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=5)
        hor_board = DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)
        ver_board = DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)

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
            mask, contour, finger_image = threshold_masking(bgr_image)

            # Get defect points from the contour
            defect_points, _ = get_defect_points(contour,
                                                 MIN_VALID_CONT_AREA=100000,
                                                 MIN_DEFECT_DISTANCE=5000)
            draw_points(finger_image, defect_points)

            # Get the touch lines
            touch_line = get_touch_line_samples(finger_image,
                                        defect_points,
                                        line_points_num=10)
            draw_points(finger_image, touch_line)

            # Segment the two fingers
            up_finger_contour, down_finger_contour = segment_diff_fingers(
                contour, defect_points)
            up_finger_contour = add_touch_line_to_contour(True, up_finger_contour,
                                               defect_points, touch_line)
            down_finger_contour = add_touch_line_to_contour(False, down_finger_contour,
                                                 defect_points, touch_line)

            if up_finger_contour is not None and DRAW_CONTOUR:
                cv2.drawContours(finger_image, [up_finger_contour], 0,
                                 [0, 0, 255], 3)
                cv2.drawContours(finger_image, [down_finger_contour], 0,
                                 [255, 0, 0], 3)

            # Get four points
            bound_points = get_boundary_points(up_finger_contour,
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

            dx, dy = tracker.calc_coord(bound_points[0], bound_points[3])

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

"""
[summary]
"""

import cv2
import numpy as np
from segment_otsu import threshold_masking
from contour_tools import get_defect_points, segment_diff_fingers, \
    get_centroid, get_boundary_points, get_touch_line_curve
from draw_tools import draw_points


def extract_features(bgr_image, output_image=None):
    # ---------------------------------------------
    # 1.1 Get the mask and its contour and apply the mask to image
    # ---------------------------------------------
    mask, contour = threshold_masking(bgr_image)
    finger_image = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)

    # ---------------------------------------------
    # 1.2 Get defect points
    # ---------------------------------------------
    # defect_points, _ = get_defect_points(contour,
    #                                     MIN_VALID_CONT_AREA=100000,
    #                                     MIN_DEFECT_DISTANCE=5000)
    defect_points, _ = get_defect_points(contour)
    if defect_points is None:
        return None

    # ---------------------------------------------
    # 1.3 Divide up and down finger contour
    # ---------------------------------------------
    up_contour, down_contour = segment_diff_fingers(contour, defect_points)
    up_centroid = get_centroid(up_contour)
    down_centroid = get_centroid(down_contour)

    # ---------------------------------------------
    # 1.4 Get boundary points
    # ---------------------------------------------
    im_height, im_width, _ = bgr_image.shape
    top_left, top_right, bottom_left, bottom_right = get_boundary_points(
        up_contour, down_contour, im_height, im_width)

    # ---------------------------------------------
    # 1.5 Get touch line then lowest up point and rightest down point
    # ---------------------------------------------
    up_touch_line, _ = get_touch_line_curve(
        IS_UP=True,
        contour=up_contour,
        bound_points=(top_left, top_right),
        fitting_curve=lambda X, Y: np.poly1d(np.polyfit(X, Y, 4)),
        defect_points=defect_points,
        draw_image=None)

    lowest_up, rightest_down = None, None
    if up_touch_line is not None and up_contour is not None and down_contour is not None:
        index_list = np.where(up_contour[:, 0, 1] == max(up_contour[:, 0,
                                                                    1]))[0]
        tmp1 = tuple(up_contour[index_list[0], 0, :])
        tmp2 = ((int)(up_centroid[0]), (int)(up_touch_line(up_centroid[0])))
        if tmp1[1] > tmp2[1]:
            lowest_up = tmp1
        else:
            lowest_up = tmp2
        index_list = np.where(down_contour[:, 0, 0] == max(down_contour[:, 0,
                                                                        0]))[0]
        rightest_down = tuple(down_contour[index_list[0], 0, :])

    # ---------------------------------------------
    # 1.6 Check None and form the feature data
    # ---------------------------------------------
    features = np.array([
        defect_points[0], defect_points[1], up_centroid, down_centroid,
        top_left, top_right, bottom_left, bottom_right, lowest_up,
        rightest_down
    ])

    if None in features:
        return None

    # ---------------------------------------------
    # 1.7 If show the points
    # ---------------------------------------------
    if output_image is not None:
        # Two defect points (Green), centroid points (Blue), boundary points (Green-blue)
        draw_points(output_image, defect_points, color=[0, 255, 0])
        draw_points(output_image, up_centroid, color=[255, 0, 0])
        draw_points(output_image, down_centroid, color=[255, 0, 0])
        draw_points(output_image, top_left, radius=10, color=[255, 255, 0])
        draw_points(output_image, top_right, radius=10, color=[255, 255, 0])
        draw_points(output_image, bottom_left, radius=10, color=[255, 255, 0])
        draw_points(output_image, bottom_right, radius=10, color=[255, 255, 0])
        draw_points(output_image, lowest_up, color=[0, 255, 255])
        draw_points(output_image, rightest_down, color=[0, 255, 255])
        # draw_contours(output_image, down_contour)
    
    return features

if __name__ == '__main__':
    '''
    
    '''
    from time import sleep
    import sys, traceback
    import picamera_control
    from draw_tools import DrawBoard
    from math_tools import configure_kalman_filter
    import joblib
    # import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import RandomForestRegressor

    try:
        print('Initializing...')
        IM_WIDTH, IM_HEIGHT = 640, 480
        # Note: Higher framerate will bring noise to the segmented image
        camera, rawCapture = picamera_control.configure_camera(IM_WIDTH,
                                                               IM_HEIGHT,
                                                               FRAME_RATE=40)
        # Show image
        SHOW_IMAGE = True

        # Drawing boards
        DR_WIDTH, DR_HEIGHT = 320, 320
        hv_board = DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=5)
        # hor_board = DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)
        # ver_board = DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)
        kalman_filter = configure_kalman_filter()

        # Model
        print('-' * 60)
        print('Start Loading Model...')

        model_path = "./models/large_models/0_862_RandomForestRegressor.joblib"
        model = joblib.load(model_path)

        print(model.get_params())
        print("\nLoad model successfully!")

        print('-' * 60)
        print("Press A to turn ON/OFF the finger image")
        print('-' * 60)

        for frame in camera.capture_continuous(rawCapture,
                                               format="bgr",
                                               use_video_port=True):
            bgr_image = frame.array
            out_image = bgr_image.copy()
            features = extract_features(bgr_image, out_image)

            if features is not None:
                coord = model.predict(features.flatten().reshape(-1, 20))[0]
                kalman_filter.correct(np.float32(coord))
                filter_ans = kalman_filter.predict()
                filtered_coord = np.array((int(filter_ans[0]), int(filter_ans[1])))
                print(filtered_coord)
                DRAW_SCALER = 5
                hv_board.draw_filled_point(filtered_coord * DRAW_SCALER)

            if SHOW_IMAGE:
                cv2.imshow('Finger', out_image)

            cv2.imshow('Board', hv_board.board)


            # ---------------------------------------------
            # 3. User Input
            # ---------------------------------------------
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == 27:
                break
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
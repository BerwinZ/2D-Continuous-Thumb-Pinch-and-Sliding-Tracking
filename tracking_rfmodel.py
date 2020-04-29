"""
Track the finger motion with random forest regression model
"""

if __name__ == '__main__':
    '''
    
    '''
    import cv2
    import numpy as np
    from time import sleep
    import sys, traceback
    import picamera_control
    from image_segment import threshold_masking
    from feature_extraction import extract_features
    from feature_mapping import MlrmMapping
    from draw_tools import DrawBoard
    from math_tools import configure_kalman_filter
    # from sklearn.multioutput import MultiOutputRegressor

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
        # DR_WIDTH, DR_HEIGHT = 320, 320
        # hv_board = DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=5)
        # hor_board = DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)
        # ver_board = DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)
    
        # model_path1 = "./models/large_models/0_862_RandomForestRegressor.joblib"
        path1 = "./models/large_models_2/0_89_RandomForestRegressor_x.joblib"
        path2 = "./models/large_models_2/0_926_RandomForestRegressor_y.joblib"

        model = MlrmMapping(path1, path2)

        kalman_filter = configure_kalman_filter()

        print('-' * 60)
        print("Press A to turn ON/OFF the finger image")
        print('-' * 60)

        for frame in camera.capture_continuous(rawCapture,
                                               format="bgr",
                                               use_video_port=True):
            bgr_image = frame.array
            out_image = bgr_image.copy()

            # ---------------------------------------------
            # 1.1 Get the mask and its contour and apply the mask to image
            # ---------------------------------------------
            mask, contour, finger_image = threshold_masking(bgr_image)

            # ---------------------------------------------
            # 1.2 Extract features
            # ---------------------------------------------
            features = extract_features(contour, IM_HEIGHT, IM_WIDTH, out_image)

            # ---------------------------------------------
            # 1.3 Map features
            # ---------------------------------------------
            coord = model.predict(features)

            if coord is not None:
                kalman_filter.correct(np.float32(coord))
                filter_ans = kalman_filter.predict()
                filtered_coord = np.array((int(filter_ans[0]), int(filter_ans[1])))
                print(filtered_coord)

            # DRAW_SCALER = 5
            # hv_board.update_dot(filtered_coord * DRAW_SCALER)

            if SHOW_IMAGE:
                cv2.imshow('Finger', out_image)

            # cv2.imshow('Board', hv_board.board)


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
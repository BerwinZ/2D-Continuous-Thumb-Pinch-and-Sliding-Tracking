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
    import draw_tools as dtl
    from math_tools import KalmanFilter

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
        DR_WIDTH, DR_HEIGHT = 300, 300
        hv_board  = dtl.DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=5)
        hor_board = dtl.DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)
        ver_board = dtl.DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)
    
        # model_path1 = "./models/large_models/0_862_RandomForestRegressor.joblib"
        path1 = "./models/large_models_2/0_89_RandomForestRegressor_x.joblib"
        path2 = "./models/large_models_2/0_926_RandomForestRegressor_y.joblib"

        model = MlrmMapping(path1, path2)

        kalman = KalmanFilter()

        print('-' * 60)
        print("Press A to turn ON/OFF the finger image")
        print('-' * 60)

        for frame in camera.capture_continuous(rawCapture,
                                               format="bgr",
                                               use_video_port=True):
            bgr_image = frame.array

            # ---------------------------------------------
            # 1.1 Get the mask and its contour and apply the mask to image
            # ---------------------------------------------
            mask, contour, finger_image = threshold_masking(bgr_image)
            out_image = finger_image.copy()

            # ---------------------------------------------
            # 1.2 Extract features
            # ---------------------------------------------
            features = extract_features(contour, IM_HEIGHT, IM_WIDTH, out_image)

            # ---------------------------------------------
            # 1.3 Map features
            # ---------------------------------------------
            coord = model.predict(features)

            if SHOW_IMAGE:
                cv2.imshow('Finger', out_image)

            # ---------------------------------------------
            # 2. Application
            # ---------------------------------------------
            if coord is not None:
                coord = kalman.predict(coord)
            else:
                coord = kalman.predict((0, 0))
            hv_board.update_dot(coord, scaler=[3, 3])
            cv2.imshow('Drawboard', hv_board.board)


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
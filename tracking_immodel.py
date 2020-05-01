"""
Track the finger motion with finger imaging model
"""


if __name__ == '__main__':
    import cv2
    import numpy as np
    from time import sleep
    import sys, traceback

    import picamera_control
    from image_segment import threshold_masking
    from feature_extraction import extract_features
    from feature_mapping import ImmMapping

    from math_tools import KalmanFilter
    import draw_tools as dtl
    # from opt_flow_LK import OpticalFlowLK

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
        # opt_flow_up = OpticalFlowLK(IM_WIDTH, IM_HEIGHT, step=50)
        # opt_flow_down = OpticalFlowLK(IM_WIDTH, IM_HEIGHT, step=50)

        # Tracker to convert point movement in image coordinate to the draw board coordinate
        # tracker = GeometryCalculator()

        kalman = KalmanFilter()

        # Drawing boards
        DR_WIDTH, DR_HEIGHT = 300, 300
        hv_board  = dtl.DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=5)
        hor_board = dtl.DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)
        ver_board = dtl.DrawBoard(DR_WIDTH, DR_HEIGHT, RADIUS=10, MAX_POINTS=1)

        model = ImmMapping()

        print('-' * 60)
        print(
            "To calibrate, press 'C'"
        )
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
            # 1.3 Feature Mapping
            # ---------------------------------------------
            coord = None
            if features is not None:
                coord = model.predict(features[-2], features[-3])
                # coord = model.predict(features[6] * 2, features[5] * 2)
                # print(coord)

            # ---------------------------------------------
            # 1.9 Show image
            # ---------------------------------------------
            if SHOW_IMAGE:
                # image_joint = np.concatenate((bgr_image, finger_image), axis=1)
                # dtl.draw_vertical_lines(image_joint, 1)
                cv2.imshow('Finger', out_image)

            # ---------------------------------------------
            # 2. Application
            # ---------------------------------------------
            if coord is not None:
                coord = kalman.predict(coord)
            else:
                coord = kalman.predict((0, 0))
            # make the y scale larger
            hv_board.update_dot(coord, scaler=[3, 9])
            cv2.imshow('Drawboard', hv_board.board)

            # ---------------------------------------------
            # 3. User Input
            # ---------------------------------------------
            # if the user pressed ESC, then stop looping
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == 27:
                break
            elif keypress == ord('c'):
                if features is not None:
                    model.calibrate(features[-2], features[-3])
            elif keypress == ord('r'):
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

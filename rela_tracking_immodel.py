"""
Track the finger's relative motion with finger imaging model
"""


if __name__ == '__main__':
    import cv2
    import numpy as np
    from time import sleep
    import sys, traceback

    import picamera_control
    from image_segment import threshold_masking
    from feature_extraction import extract_features_simple
    from feature_mapping import ImmMapping

    from math_tools import KalmanFilter
    import draw_tools as dtl

    """
    This function get the frame from the camera, and use thresholding to finger_image the hand part
    """
    try:
        IM_WIDTH, IM_HEIGHT = 640, 480
        # Note: Higher framerate will bring noise to the segmented image
        camera, rawCapture = picamera_control.configure_camera(IM_WIDTH,
                                                               IM_HEIGHT,
                                                               FRAME_RATE=40)
        # Show image flag
        SHOW_IMAGE = False

        # Mapping model
        model = ImmMapping()

        # Start pos
        ini_touch_pos = None
        dlt_touch_vec = (0, 0)

        ini_board_pos = (0, 0)
        cur_board_pos = (0, 0)

        wait_frame = 2

        # Filter
        kalman = KalmanFilter()

        # Drawing boards
        DR_SIZE = 500
        # board = dtl.DrawBoard(DR_SIZE, DR_SIZE, RADIUS=10, MAX_POINTS=10)
        board = dtl.TargetDotBoard(DR_SIZE, DR_SIZE, RADIUS=10, MAX_POINTS=1)

        print('-' * 60)
        print("To calibrate, press 'C'")
        print("To turn ON/OFF the finger image, press 'A'")
        print('-' * 60)

        for frame in camera.capture_continuous(rawCapture,
                                               format="bgr",
                                               use_video_port=True):
            bgr_image = frame.array

            # ---------------------------------------------
            # 1.1 Get the mask and its contour and apply the mask to image
            # ---------------------------------------------
            mask, contour, finger_image = threshold_masking(bgr_image)
            out_image = finger_image.copy() if SHOW_IMAGE else None

            # ---------------------------------------------
            # 1.2 Extract features
            # ---------------------------------------------
            features = extract_features_simple(contour, IM_HEIGHT, IM_WIDTH, out_image)

            # ---------------------------------------------
            # 1.3 Feature Mapping and kalman filter
            # ---------------------------------------------
            if features is not None:
                coord = model.predict(features[0], features[1], features[2])
                k_coord = kalman.predict(coord)
            # print(coord)

            # ---------------------------------------------
            # 1.4 Calculate relative move
            # ---------------------------------------------
            if features is None:
                if ini_touch_pos is not None:
                    ini_board_pos = cur_board_pos

                ini_touch_pos = None
                dlt_touch_vec = (0, 0)
                wait_frame = 2
            else:
                if wait_frame >= 0:
                    wait_frame -= 1
                else:
                    if ini_touch_pos is None:
                        ini_touch_pos = k_coord
                        dlt_touch_vec = (0, 0)
                    else:
                        dlt_touch_vec = (k_coord[0] - ini_touch_pos[0],
                                         k_coord[1] - ini_touch_pos[1])
            
            cur_board_pos = (ini_board_pos[0] + dlt_touch_vec[0],
                             ini_board_pos[1] + dlt_touch_vec[1])

            # print(cur_board_pos)

            # ---------------------------------------------
            # 1.9 Show image
            # ---------------------------------------------
            if SHOW_IMAGE:
                cv2.imshow('Finger', out_image)

            # ---------------------------------------------
            # 2. Application
            # ---------------------------------------------
            # make the y scale larger
            board.update_dot(cur_board_pos, scaler=[8 * 1.6, 11 * 1.6])
            cv2.imshow('Drawboard', board.board)

            # ---------------------------------------------
            # 3. User Input
            # ---------------------------------------------
            # if the user pressed ESC, then stop looping
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == 27:
                break
            elif keypress == ord('c'):
                ini_board_pos = (0, 0) 
            elif keypress == ord('r'):
                board.reset_board()
                hor_board.reset_board()
                ver_board.reset_board()
            elif keypress == ord('s'):
                # cv2.imwrite('screenshot.jpg', finger_image)
                board.start()
            elif keypress == ord('a'):
                if SHOW_IMAGE:
                    cv2.destroyWindow("Finger")
                SHOW_IMAGE = not SHOW_IMAGE

            rawCapture.truncate(0)

        camera.close()
        cv2.destroyAllWindows()
        board.stop()
    except Exception as e:
        camera.close()
        cv2.destroyAllWindows()
        board.stop()
        print("Exception in user code:")
        print('-' * 60)
        traceback.print_exc(file=sys.stdout)
        print('-' * 60)

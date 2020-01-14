import cv2
import numpy as np
from time import sleep
import picamera_control
import os

if __name__ == '__main__':
    try:
        camera, rawCapture = picamera_control.configure_camera(640, 480)

        # For MOG Mask
        # mog = cv2.bgsegm.createBackgroundSubtractorMOG()

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            bgr_image = frame.array

            # yCrCb mask
            frame_ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCR_CB)
            mask_ycrcb = cv2.inRange(frame_ycrcb, np.array(
                [0, 145, 85]), np.array([255, 185, 155]))

            # Mask_mog
            # mask_mog = mog.apply(bgr_image)

            segment = cv2.bitwise_and(bgr_image, bgr_image, mask=mask_ycrcb)

            cv2.imshow("original", bgr_image)
            cv2.imshow('Mask', mask_ycrcb)
            cv2.imshow('Segment', segment)

            # if the user pressed ESC, then stop looping
            keypress = cv2.waitKey(25) & 0xFF
            if keypress == 27:
                break

            rawCapture.truncate(0)
            
        camera.close()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        camera.close()
        cv2.destroyAllWindows()
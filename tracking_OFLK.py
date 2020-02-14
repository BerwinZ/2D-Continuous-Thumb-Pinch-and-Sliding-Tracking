'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. 
-----
'''

import numpy as np
import cv2
import picamera_control
import sys, traceback

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class optical_flow_LK:
    def __init__(self):
        self.old_gray = None

    def calc_optical_flow(self, new_gray, old_points, draw_img=None):
        """[summary]
        
        Arguments:
            gray {[type]} -- [description]
            old_points {[type]} -- [description]
        
        Keyword Arguments:
            draw_img {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [description]
        """
        if self.old_gray is None or old_points is None:
            self.old_gray = new_gray
            return None

        new_points, st, _err = cv2.calcOpticalFlowPyrLK(self.old_gray, new_gray, old_points, None, **lk_params)
        self.old_gray = new_gray

        direc = new_points - old_points
        direc[st.ravel() == 1] = (0, 0)
        
        if draw_img is not None:

            for new, old in zip(new_points[st.ravel() == 1], old_points[st.ravel() == 1]):

                x1, y1 = new.ravel()
                x0, y0 = old.ravel()
                
                cv2.line(draw_img, (x0, y0), (x1, y1), [0, 255, 0], 3)

        return direc

if __name__ == '__main__':
    """
    This function get the frame from the camera, and use thresholding to segment the hand part
    """
    try:
        camera, rawCapture = picamera_control.configure_camera(640,
                                                               480,
                                                               FRAME_RATE=35)

        tracks_points = np.float32([(100, 100), (200, 200), (300, 300), (400, 400)])
        opt_flow = optical_flow_LK()

        for frame in camera.capture_continuous(rawCapture,
                                               format="bgr",
                                               use_video_port=True):
            bgr_image = frame.array            
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

            direc = opt_flow.calc_optical_flow(gray, tracks_points, bgr_image)
            print(direc)
            
            cv2.imshow('lk_track', bgr_image)

            keypress = cv2.waitKey(1) & 0xFF
            if keypress == 27:
                break

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

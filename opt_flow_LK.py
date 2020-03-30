'''
This script is used to show the optical flow of fingers
'''

import numpy as np
import cv2
import sys, traceback

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,
                           0.03))


class OpticalFlowLK:
    def __init__(self, IM_WIDTH=None, IM_HEIGHT=None, step=50):
        self._old_gray = None
        self._grid_points = None

        if IM_HEIGHT is not None or IM_HEIGHT is not None:
            X, Y = np.mgrid[0:IM_WIDTH:step, 0:IM_HEIGHT:step]
            all_points = np.vstack((X.ravel(), Y.ravel())).T
            all_points = list(map(tuple, all_points))
            self._grid_points = all_points

    def calc_on_points(self, new_gray, track_points, draw_img=None):
        """Calculate the optical flow on the given points
        
        Arguments:
            gray {[type]} -- [description]
            track_points {[type]} -- [description]
        
        Keyword Arguments:
            draw_img {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [description]
        """
        if self._old_gray is None or track_points is None or track_points.size == 0:
            self._old_gray = new_gray
            return None

        new_points, st, err = cv2.calcOpticalFlowPyrLK(
            self._old_gray, new_gray, track_points, None, **lk_params)
        self._old_gray = new_gray

        direc = new_points - track_points
        direc = np.delete(direc, np.where((st.ravel() == 0) | (err.ravel() > 1)), axis=0)
        if direc.size == 0:
            direc = None

        if draw_img is not None:

            for new, old in zip(new_points[(st.ravel() == 1) & (err.ravel() <= 1)],
                                track_points[(st.ravel() == 1) & (err.ravel() <= 1)]):

                x1, y1 = new.ravel()
                x0, y0 = old.ravel()

                cv2.circle(draw_img, (x0, y0), 5, [255, 0, 0])
                cv2.line(draw_img, (x0, y0), (x1, y1), [0, 255, 0], 3)

        return direc

    def calc_in_contour(self, new_gray, contour, draw_img=None):
        """Calculate the optical flow on the points inside the contour
        
        Arguments:
            contour {[type]} -- [description]
            all_points {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        if contour is None:
            return None
        
        if self._grid_points is None:
            print("Haven't configured the grid points")
            return None

        f = lambda p: cv2.pointPolygonTest(contour, p, True)
        flag = map(f, self._grid_points)
        flag = np.fromiter(flag, dtype=np.float)
        track_points = np.float32(self._grid_points)[flag > 0]

        # track_points = []
        # for p in self._grid_points:
        #     if cv2.pointPolygonTest(contour, p, True) > 0:
        #         track_points.append(p)
        # track_points = np.float32(track_points)

        return self.calc_points(new_gray, track_points, draw_img)



if __name__ == '__main__':
    """
    This function get the frame from the camera, and use thresholding to segment the hand part
    """
    try:
        import picamera_control
        camera, rawCapture = picamera_control.configure_camera(640,
                                                               480,
                                                               FRAME_RATE=35)

        tracks_points = np.float32([(100, 100), (200, 200), (300, 300),
                                    (400, 400)])
        opt_flow = OpticalFlowLK()

        for frame in camera.capture_continuous(rawCapture,
                                               format="bgr",
                                               use_video_port=True):
            bgr_image = frame.array
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

            direc = opt_flow.calc_on_points(gray, tracks_points, bgr_image)
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

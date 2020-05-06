# -----------------
# Extract the hand region from the images using the skin color
# -----------------
import cv2
import numpy as np
# from skimage.color import rgb2ycbcr, rgb2gray

if __name__ == "__main__":
    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # For MOG Mask
    mog = cv2.bgsegm.createBackgroundSubtractorMOG()

    # For MOG2 Mask
    mog2 = cv2.createBackgroundSubtractorMOG2()

    # For GMG Mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gmg = cv2.bgsegm.createBackgroundSubtractorGMG()

    try:
        # keep looping, until interrupted
        while True:
            # get the current frame, the format is BGR
            (ret, frame) = camera.read()

            # yCbCr maskB
            frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            mask_ycrcb = cv2.inRange(frame_ycrcb, np.array(
                [0, 145, 85]), np.array([255, 185, 155]))

            # MOG
            mask_mog = mog.apply(frame)

            # MOG2
            mask_mog2 = mog2.apply(frame)

            # GMG
            mask_gmg = gmg.apply(frame)
            mask_gmg = cv2.morphologyEx(mask_gmg, cv2.MORPH_OPEN, kernel)

            # filtering remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_mog = cv2.filter2D(mask_mog, -1, kernel)

            # display the frame
            cv2.imshow("Original",      frame)
            cv2.imshow("Mask_yCrCb",    mask_ycrcb.astype("float"))
            cv2.imshow("Mask_mog",      mask_mog)
            cv2.imshow("Mask_mog2",     mask_mog2)
            cv2.imshow("Mask_gmg",      mask_gmg)
            cv2.imshow("ROI",           cv2.bitwise_and(
                frame, frame, mask=mask_mog))

            # in case stuck
            keypress = cv2.waitKey(25) & 0xFF

            # if the user pressed "q", then stop looping
            if keypress == ord("q"):
                break

        camera.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        camera.release()
        cv2.destroyAllWindows()

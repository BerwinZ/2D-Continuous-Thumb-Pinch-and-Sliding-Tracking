import cv2
import numpy as np

# Get original histogram
roi = cv2.imread('hand_crop.jpg')
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# Get target histogram
target = cv2.imread('hand.jpg')
target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
target_hist = cv2.calcHist([target_hsv], [0, 1], None, [
                           180, 256], [0, 180, 0, 256])

# Calculate the M/I
R = np.true_divide(roi_hist, np.add(target_hist, 0.0001))

# Get the backprojection value to B
h, s, v = cv2.split(target_hsv)
B = R[h.ravel(), s.ravel()]
B = np.minimum(B, 1)
B = B.reshape(target_hsv.shape[:2])

cv2.imshow('Naive', B)

# Apply a convolution with a circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
cv2.filter2D(B, -1, disc, B)
cv2.normalize(B, B, 0, 255, cv2.NORM_MINMAX)
B = np.uint8(B)

cv2.imshow('After convolution', B)

# Threshold
ret, B = cv2.threshold(B, 50, 255, 0)

cv2.imshow('After thresholding', B)

cv2.waitKey(0)
cv2.destroyAllWindows()

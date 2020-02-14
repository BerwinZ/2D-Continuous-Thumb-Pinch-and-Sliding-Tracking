# MicroPinch

This repo is used to track the thumb's touch position relatived to other fintertips. It runs in Raspberry Pi with a Pi Camera.

## Requirements

* Raspberry Pi 3B+
* Pi Camera
* Python 3.7.6
* OpenCV-Python 3.4.2.16

## Content

| Script   | Content   |
|---|---|
| picamera_control.py | configure the pi camera and get raw image data (BGR) |
| segment_hbp.py | segment the hand part from the image with histogram backprojection method |
| segment_otsu.py | segment the hand part from the image with otsu thresholding method |
| tracking_convdef.py | track the touch point with convexcity defects from the contour of segmented finger |
| tracking_bound.py | track the boundary points from the up and down finger contour |
| move_tracker.py | transferred the movements of tracked feature points to the movements of finger's real movements |
| draw_tools.py | draw points in a board to indicate the movements, and some useful drawing functions |

## Raspberry Pi Folder

pi@192.168.137.209:/home/pi/mcrpnh

## Steps

### Find the touch line between two fingers

#### Fit a curve using the contour points

Up contour can work with a 4 ladder poly, but Down contour cannot because it is not a function.

#### Rotate the down contour points, then rotate the fitted curve back

Work only when the thumb move up.

#### Fit a ellipse for the down contour

Cannot work, shape is not fitted and cannot find the intersection between the ellipse with the line

#### Detect the touch line with the Canny algorithm

Cannot work because it is very hard to detect

#### Fit a curve, then find the nearest points, then fit again

May work

### Decide the touch point

When thumb is up, use the intersection point of up touch line generated from thumb contour. When the thumb is down, use the intersection point of down touch line generated from index finger contour.

### Calculate movements

#### X



## Reference

1. [Histogram Backprojection](https://docs.opencv.org/master/dc/df6/tutorial_py_histogram_backprojection.html)
2. [Otsu thresholding](http://www.kevinlt.top/2018/10/23/hand_segment/)
3. [Convex Hull and Convex Defect](https://docs.opencv.org/3.4.2/d5/d45/tutorial_py_contours_more_functions.html)
4. [Kalman Filter](https://blog.csdn.net/lwplwf/article/details/74295801)
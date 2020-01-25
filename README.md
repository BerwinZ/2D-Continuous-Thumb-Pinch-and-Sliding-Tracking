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
| relative_mov_tracker.py | calculate the scaled relative movements of the touch point |
| draw_board.py | draw points in a board to indicate the movements |
| tracking_bond.py | calculate the movement of hand with boundary points |
| main.py | TBD |

## Raspberry Pi Folder

pi@192.168.137.209:/home/pi/mcrpnh

## Reference

1. [Histogram Backprojection](https://docs.opencv.org/master/dc/df6/tutorial_py_histogram_backprojection.html)
2. [Otsu thresholding](http://www.kevinlt.top/2018/10/23/hand_segment/)
3. [Convex Hull and Convex Defect](https://docs.opencv.org/3.4.2/d5/d45/tutorial_py_contours_more_functions.html)
4. [Kalman Filter](https://blog.csdn.net/lwplwf/article/details/74295801)
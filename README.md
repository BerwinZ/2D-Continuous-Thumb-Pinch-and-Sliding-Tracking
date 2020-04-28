# MicroPinch

This repo is used to track the thumb's touch position relative to other fingertips. It runs in Raspberry Pi with a Pi Camera Module.

## Requirements

* Raspberry Pi 3B+
* Pi Camera Module
* Python 3.7.6
* OpenCV-Python 3.4.2.16 and other libaries in requirement.txt

## How to Run

1. Clone or download this repo to Raspberry Pi
2. Install Python 3.7.6 in Raspberry Pi
3. Install the required packages by
    ```Bash
    pip3 install -r requirements/run_requirement.txt
    ```
4. Run the script by
    ```Bash
    python3 tracking_models.py 
    ```

## Raspberry Pi Folder

pi@192.168.137.209:/home/pi/mcrpnh

## Scripts Analysis

### Scripts Description

 Script | Content |
--------|----------
draw_tools.py | draw points in a board to indicate the movements, and some useful drawing functions
math_tools.py | math calculations
feature_extraction.py | methods that is realted to contours
picamera_control.py | configure the pi camera and get raw image data (BGR)
coord_calculator.py | transferred the movements of tracked feature points to the movements of finger's real movements
segment_hbp.py | segment the hand part from the image with histogram backprojection method
image_segment.py | segment the hand part from the image with otsu thresholding method
tracking_defect.py | track the touch point with convexcity defects from the contour of segmented finger
tracking_bound.py | track the boundary points from the up and down finger contour
tracking_geometry.py | track the points in some geometric features
tracking_bound.py | track features with a model


### Pipeline

TBD


## Reference

1. [Histogram Backprojection](https://docs.opencv.org/master/dc/df6/tutorial_py_histogram_backprojection.html)
2. [Otsu thresholding](http://www.kevinlt.top/2018/10/23/hand_segment/)
3. [Convex Hull and Convex Defect](https://docs.opencv.org/3.4.2/d5/d45/tutorial_py_contours_more_functions.html)
4. [Kalman Filter](https://blog.csdn.net/lwplwf/article/details/74295801)
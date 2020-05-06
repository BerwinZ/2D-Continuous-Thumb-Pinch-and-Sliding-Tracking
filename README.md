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
    python3 tracking_immodel.py 
    ```

## Functions of scripts

 Script                 | Content |
------------------------|------------------------------
draw_tools.py           | drawing board, user experiments board, and some useful drawing functions
math_tools.py           | math calculations, kalman filter
picamera_control.py     | configure the pi camera and get raw image data in BGR format
image_segment.py        | segment the hand part from the image with otsu thresholding method
feature_extraction.py   | functions to extract features from contour of fingers
feature_mapping.py      | classes to map feature data to pose coordinate
train_knn_models.py     | train the knn models
train_re_models.py      | train random forest models
train_re_models_win.ipynb       | train random forest models. Run in `Windows` 
prepare_fea_data_pca_win.ipynb  | process generated image and build a feature dataset. Use PCA. Run in `Windows`
tracking_knnmodel.py    | tracking demo using the PCA and KNN
tracking_rfmodel.py     | tracking demo using random forest regression model
**tracking_immodel.py**     | tracking demo using imaging model. User experiment using absolute pose coordinate tracking
**rela_tracking_immodel.py** | user experiment using relative motion tracking

## Reference

1. [Histogram Backprojection](https://docs.opencv.org/master/dc/df6/tutorial_py_histogram_backprojection.html)
2. [Otsu thresholding](http://www.kevinlt.top/2018/10/23/hand_segment/)
3. [Convex Hull and Convex Defect](https://docs.opencv.org/3.4.2/d5/d45/tutorial_py_contours_more_functions.html)
4. [Kalman Filter](https://blog.csdn.net/lwplwf/article/details/74295801)
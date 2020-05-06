'''
example to show optical flow

USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch

Keys:
    ESC    - exit
'''

import numpy as np
import cv2
import picamera_control
import sys, traceback

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

if __name__ == '__main__':
    """
    This function get the frame from the camera, and use thresholding to segment the hand part
    """
    try:
        camera, rawCapture = picamera_control.configure_camera(640,
                                                               480,
                                                               FRAME_RATE=35)

        prevgray = None
        show_hsv = False
        show_glitch = False

        #-----------------------------------------------
        # Method 2
        # inst = cv2.optflow.createOptFlow_DeepFlow()
        #-----------------------------------------------

        #-----------------------------------------------
        # Method 5
        # inst = cv2.optflow.createOptFlow_PCAFlow()
        #-----------------------------------------------
        
        #-----------------------------------------------
        # Method 6
        inst = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)
        inst.setUseSpatialPropagation(True)
        #-----------------------------------------------
    
        for frame in camera.capture_continuous(rawCapture,
                                               format="bgr",
                                               use_video_port=True):
            img = frame.array

            gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray_full.shape
            gray = gray_full[height // 2:, : width // 2]
            if prevgray is None:
                prevgray = gray
                cur_glitch = img.copy()
            else:
                #-----------------------------------------------
                # Method 1
                # flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                #-----------------------------------------------

                #-----------------------------------------------
                # Method 2 / 5 / 6 (Work good)
                # flow = inst.calc(prevgray, gray, None)
                #-----------------------------------------------

                #-----------------------------------------------
                # Method 3
                # flow = cv2.optflow.calcOpticalFlowSF(prevgray, gray, 2, 2, 4)
                #-----------------------------------------------

                #-----------------------------------------------
                # Method 4 (Work good)
                flow = cv2.optflow.calcOpticalFlowSparseToDense(prevgray, gray)
                #-----------------------------------------------

            
                prevgray = gray

                cv2.imshow('flow', draw_flow(gray, flow))
                cv2.imshow('Full', gray_full)
                if show_hsv:
                    cv2.imshow('flow HSV', draw_hsv(flow))
                if show_glitch:
                    cur_glitch = warp_flow(cur_glitch, flow)
                    cv2.imshow('glitch', cur_glitch)

            ch = cv2.waitKey(1) & 0xFF
            if ch == 27:
                break
            if ch == ord('1'):
                show_hsv = not show_hsv
                print('HSV flow visualization is', ['off', 'on'][show_hsv])
            if ch == ord('2'):
                show_glitch = not show_glitch
                if show_glitch:
                    cur_glitch = img.copy()
                print('glitch is', ['off', 'on'][show_glitch]) 
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
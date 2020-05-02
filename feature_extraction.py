"""
This script provide methods to extract the features from the images. The extractions are based on the contour of the fingers.

It also includes several functions which can be used indepently
1. extract_features
    extract all 10 features 
2. get_defect_points
    Get left and right defect points from contour
3. segment_diff_fingers
    Segment whole contour into up and down contours according to defect points
4. add_touch_line_to_contour
    Add touch line points to the contour
5. get_boundary_points
    Get 4 boundary points of the up and down contour 
6. get_centroid
    Get the centroid of a boundary
7. get_touch_line_curve
    Get touch line curve and curve points
8. get_special_pt
    Get the lowest thumb, rightest index finger points 
"""

import numpy as np
import cv2
import draw_tools as dtl

def extract_features(contour, im_height, im_width, output_image=None):
    """Extract features

    Arguments:
        contour {[type]} -- [description]
        im_height {[type]} -- [description]
        im_width {[type]} -- [description]

    Keyword Arguments:
        output_image {[type]} -- [description] (default: {None})

    Returns:
        array {1 * 20 array} -- left defect points, right defect points, thumb_centroid, index_centroid, top_left, top_right, bottom_left, bottom_right, lowest_thumb, rightest_index
    """
    # ---------------------------------------------
    # 1.1 Get 2 defect points
    # ---------------------------------------------
    # dft_pts, _ = get_defect_points(contour,
    #                                     MIN_VALID_CONT_AREA=100000,
    #                                     MIN_DEFECT_DISTANCE=5000)
    dft_pts, _ = get_defect_points(contour)

    # ---------------------------------------------
    # 1.2 Divide up and down finger contour and get 2 centroids
    # ---------------------------------------------
    thumb_cnt, index_cnt = segment_diff_fingers(contour, dft_pts)
    thumb_cent = get_centroid(thumb_cnt)
    index_cent = get_centroid(index_cnt)

    # ---------------------------------------------
    # 1.3 Get 4 boundary points
    # ---------------------------------------------
    top_left, top_right, bottom_left, bottom_right = get_boundary_points(
        thumb_cnt, index_cnt, im_height, im_width)

    # ---------------------------------------------
    # 1.4 Get touch line, then lowest thumb point and rightest index finger point
    # ---------------------------------------------
    _, _, thumb_touch_pts = get_touch_line_curve(
        IS_UP=True,
        contour=thumb_cnt,
        bound_points=(top_left, top_right),
        fitting_curve=lambda X, Y: np.poly1d(np.polyfit(X, Y, 2)),
        defect_points=dft_pts)

    _, _, index_touch_pts = get_touch_line_curve(
        IS_UP=False,
        contour=index_cnt,
        bound_points=(bottom_left, bottom_right),
        fitting_curve=lambda X, Y: np.poly1d(np.polyfit(X, Y, 3)),
        defect_points=dft_pts)

    lowest_thumb   = get_special_pt(thumb_cnt, thumb_touch_pts, im_width, 'lowest')
    rightest_index = get_special_pt(index_cnt, index_touch_pts, im_width, 'rightest')

    # ---------------------------------------------
    # 1.5 Check None and form the feature data
    # ---------------------------------------------
    if dft_pts is None:
        return None
    
    # dft pts is list of tuple, others are all tuple
    features = np.array([
        dft_pts[0], dft_pts[1], thumb_cent, index_cent,
        top_left, top_right, bottom_left, bottom_right, 
        lowest_thumb, rightest_index
    ])
    
    if None in features:
        return None

    features = features.flatten()

    # ---------------------------------------------
    # 1.6 If show the points
    # ---------------------------------------------
    if output_image is not None:
        # dtl.draw_contours(output_image, thumb_cnt, color=[255, 0, 0])
        # dtl.draw_contours(output_image, index_cnt, color=[0, 0, 255])
        # Two defect points (Green), centroid points (Blue), boundary points (Green-blue)
        dtl.draw_points(output_image, dft_pts, color=[0, 255, 0])
        dtl.draw_points(output_image, thumb_cent, color=[255, 0, 0])
        dtl.draw_points(output_image, index_cent, color=[255, 0, 0])
        dtl.draw_points(output_image, top_left, color=[255, 255, 0])
        dtl.draw_points(output_image, top_right, color=[255, 255, 0])
        dtl.draw_points(output_image, bottom_left, color=[255, 255, 0])
        dtl.draw_points(output_image, bottom_right, color=[255, 255, 0])
        dtl.draw_points(output_image, lowest_thumb, radius=10, color=[0, 255, 255])
        dtl.draw_points(output_image, rightest_index, radius=10, color=[0, 255, 255])
    
    return features

def get_defect_points(contour, MIN_VALID_CONT_AREA=0, MIN_DEFECT_DISTANCE=0):
    """Get the two convex defect points

    Arguments:
        contour {cv2.contour} -- [the contour of the finger]

    Returns:
        defect_points [list of tuple] -- [(left_x, left_y), (right_x, right_y)]
        hull_line [list of tuple] -- [(start1, end1), (start2, end2)]
    """
    # In case no contour or the contour area is too small (single fingertip)
    if contour is None or cv2.contourArea(contour) < MIN_VALID_CONT_AREA:
        return None, None

    # Get the convex defects
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    # Get the defects with the largest 2 distance
    if defects is None:
        return None, None

    # Filter the defects with threshold
    defects = defects[defects[:, 0, 3] > MIN_DEFECT_DISTANCE]
    if defects.shape[0] < 2:
        return None, None

    sorted_defects = sorted(defects[:, 0, :], key=lambda x: x[3])

    defect_points = []
    hull_lines = []
    for s, e, f, d in sorted_defects[-2:]:
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        defect_points.append(far)
        hull_lines.append((start, end))

    if defect_points[0][0] > defect_points[1][0]:
        defect_points[0], defect_points[1] = defect_points[1], defect_points[0]
        hull_lines[0], hull_lines[1] = hull_lines[1], hull_lines[0]

    return defect_points, hull_lines

def segment_diff_fingers(contour, defect_points):
    """Segment the contour to the up finger and down finger

    Arguments:
        contour {[type]} -- ((m+n) * 1 * 2) array
        defect_points {[type]} -- left and right defect points

    Returns:
        thumb contour -- (m * 1 * 2) array
        index finger contour -- (n * 1 * 2) array
    """
    if contour is None or defect_points is None:
        return None, None

    d1 = np.linalg.norm(contour[:, 0, :] - np.array(defect_points[0]),
                        axis=1)
    d2 = np.linalg.norm(contour[:, 0, :] - np.array(defect_points[1]),
                        axis=1)

    idx_left = np.where(d1 == min(d1))[0][0]
    idx_right = np.where(d2 == min(d2))[0][0]

    thumb_cnt = np.concatenate((contour[:idx_left+1],
                                contour[idx_right:]))
    index_cnt = contour[idx_left:idx_right+1]

    return thumb_cnt, index_cnt


def add_touch_line_to_contour(is_up, contour, defect_points, touch_line):
    """Add the touch line to the up and down contour
    
    Arguments:
        is_up {bool} -- [description]
        contour {[type]} -- [description]
        defect_points {[type]} -- [description]
        touch_line {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    if contour is None or touch_line is None or defect_points is None:
        return None

    (x1, y1), (x2, y2) = defect_points

    if type(touch_line) == list:
        to_add = np.reshape(touch_line, [len(touch_line), 1, 2])
    elif type(touch_line) == tuple:
        to_add = np.reshape(touch_line, [1, 1, 2])

    if is_up:
        index = np.where((contour[:, 0, 0] == x1)
                         & (contour[:, 0, 1] == y1))[0]
        if index is not None and len(index) != 0:
            contour = np.insert(contour, index[-1] + 1, to_add, axis=0)
    else:
        contour = np.insert(contour, contour.shape[0], to_add[::-1], axis=0)

    return contour

# TODO: some error here, 179938 image in step=2 dataset
def get_boundary_points(up_contour, down_contour, height, width):
    """Get the four boundary points of the hand contour
    
    Arguments:
        up_contour {[type]} -- [description]
        down_contour {[type]} -- [description]
        height {[type]} -- [description]
        width {[type]} -- [description]
    
    Returns:
        Boundary Points [list of tuple] -- [top_left, top_right, bottom_left, bottom_right]
    """
    if up_contour is None or down_contour is None or height is None or width is None:
        return None, None, None, None

    up_cnt   = np.reshape(up_contour,   (up_contour.shape[0],   2))
    down_cnt = np.reshape(down_contour, (down_contour.shape[0], 2))

    def get_pt(contour, bd_info, pt_info):
        bd_basis, bd_val = bd_info
        bd_idx = np.where(contour[:, bd_basis] == bd_val)[0]
        if len(bd_idx) > 0:
            bd     = contour[bd_idx]
            pt_basis, pt_f = pt_info
            pt_idx = np.where(bd[:, pt_basis] == pt_f(bd[:, pt_basis]))[0][0]
            return tuple(bd[pt_idx])
        else:
            return None

    bd_dict = {'top'  : [1, 0],
               'down' : [1, height-1],
               'left' : [0, 0],
               'right': [0, width-1]}

    top_left = get_pt(up_cnt, bd_dict['left'], mode_dict['lowest'])
    if top_left is None:
        top_left = get_pt(up_cnt, bd_dict['top'], mode_dict['leftest'])

    top_right = get_pt(up_cnt, bd_dict['right'], mode_dict['lowest'])
    if top_right is None:
        top_right = get_pt(up_cnt, bd_dict['top'], mode_dict['rightest'])
    
    bottom_left = get_pt(down_cnt, bd_dict['left'], mode_dict['upest'])
    if bottom_left is None:
        bottom_left = get_pt(down_cnt, bd_dict['down'], mode_dict['leftest'])

    bottom_right = get_pt(down_cnt, bd_dict['right'], mode_dict['upest'])
    if bottom_right is None:
        bottom_right = get_pt(down_cnt, bd_dict['down'], mode_dict['rightest'])
    if bottom_right is None:
        bottom_right = get_pt(down_cnt, bd_dict['left'], mode_dict['lowest'])

    return top_left, top_right, bottom_left, bottom_right

def get_centroid(contour):
    """Get the centroid of the contour
    
    Arguments:
        contour {[type]} -- [description]
    
    Returns:
        centroid -- tuple
    """
    if contour is None:
        return None

    moment = cv2.moments(contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return (cx, cy)
    else:
        return None

def get_touch_line_curve(IS_UP,
                     contour,
                     bound_points,
                     fitting_curve,
                     defect_points,
                     draw_image=None):
    """Fit the up contour with polyfit using the points of the contour within the bound points
    
    Arguments:
        IS_UP {bool} -- [description]
        contour {[type]} -- [description]
        bound_points {[type]} -- [description]
        fitting_curve {[type]} -- [description]
        defect_points {[type]} -- [description]
    
    Keyword Arguments:
        draw_image {[type]} -- [description] (default: {None})
    
    Returns:
        curve [lambda function]
        theta  [float]
        points [n * 2 array]
    """
    if contour is None or fitting_curve is None or defect_points is None:
        return None, None, None

    # Get the points used to fit
    if bound_points is None or bound_points[0] is None or bound_points[
            1] is None:
        return None, None, None

    def __rotate_array(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

    (x1, y1), (x2, y2) = bound_points
    contour = np.reshape(contour, (contour.shape[0], 2))
    theta = None

    if IS_UP:
        f = lambda x, y: (y2 - y1) / (x2 - x1) * (x - x1) - (y - y1)
        contour = contour[f(contour[:, 0], contour[:, 1]) < 0]
    else:
        # Rotate points
        if abs(x1 - x2) < 1e-9:
            theta = -1 * 90 * np.pi / 180
            contour = contour[contour[:, 0] > 0]
        else:
            theta = -1 * np.arctan((y2 - y1) / (x2 - x1))
            f = lambda x, y: (y2 - y1) / (x2 - x1) * (x - x1) - (y - y1)
            contour = contour[f(contour[:, 0], contour[:, 1]) > 0]

        rotate = __rotate_array(theta)
        contour = rotate.dot(contour.T).T
        defect_points = rotate.dot(np.array(defect_points).T).T

    # Fit the function
    X = contour[:, 0]
    Y = contour[:, 1]
    if len(X) == 0 or len(Y) == 0:
        return None, None, None
    
    curve = fitting_curve(X, Y)

    # Get the touch line x range
    p_x = np.arange(defect_points[0][0], defect_points[1][0])
    p_y = np.array(curve(p_x))
    # Reshape
    num = p_x.shape[0]
    p_x = np.reshape(p_x, (num, 1))
    p_y = np.reshape(p_y, (num, 1))
    # Joint
    points = np.concatenate((p_x, p_y), axis=1)

    if not IS_UP:
        rotate = __rotate_array(-theta)
        points = rotate.dot(points.T).T

    points = points.astype("int")

    if draw_image is not None:
        # Draw the points used to fit
        color = [0, 0, 255] if IS_UP else [255, 0, 0]
        dtl.draw_points(draw_image, contour.astype("int"), radius=3, color=color)
        dtl.draw_points(draw_image, points, radius=3, color=color)

    return curve, theta, points


mode_dict = {'lowest': [1, max], 
             'upest':  [1, min],
             'leftest':[0, min],
             'rightest':[0, max]}

def get_special_pt(contour, touch_pts, im_width, mode='lowest'):
    """Return the special point

    Arguments:
        contour {[type]} -- [description]
        touch_pts {[type]} -- [description]

    Keyword Arguments:
        mode {str} -- [description] (default: {'lowest'})

    Returns:
        pt -- tuple
    """

    if contour is None or touch_pts is None or len(touch_pts) == 0:
        return None

    if mode not in mode_dict:
        return None

    contour = np.reshape(contour, (contour.shape[0], 2))
    pt = None
    basis, f = mode_dict[mode]

    if mode == 'rightest':
        index = np.where(contour[:, basis] == f(contour[:, basis]))[0]
        bo = tuple(contour[index[0] ])
        up = tuple(contour[index[-1]])
        if up[0] == im_width - 1:
            pt = (up[0] + bo[1] - up[1], (bo[1] + up[1]) // 2)
        else:
            pt = (up[0], (bo[1] + up[1]) // 2)
    elif mode == 'lowest':
        index = np.where(contour[:, basis] == f(contour[:, basis]))[0][0]
        p1 = tuple(contour[index])

        index = np.where(touch_pts[:, basis] == f(touch_pts[:, basis]))[0][0]
        p2 = tuple(touch_pts[index])

        pt = p1 if p1[basis] == f(p1[basis], p2[basis]) else p2

    return pt

if __name__ == '__main__':
    import cv2
    import numpy as np
    from time import sleep
    import sys, traceback

    import picamera_control
    from image_segment import threshold_masking

    try:
        IM_WIDTH, IM_HEIGHT = 640, 480
        # Note: Higher framerate will bring noise to the segmented image
        camera, rawCapture = picamera_control.configure_camera(IM_WIDTH,
                                                               IM_HEIGHT,
                                                               FRAME_RATE=40)

        for frame in camera.capture_continuous(rawCapture,
                                               format="bgr",
                                               use_video_port=True):
            bgr_image = frame.array
            out_image = bgr_image.copy()

            mask, contour, finger_image = threshold_masking(bgr_image)

            features = extract_features(contour, IM_HEIGHT, IM_WIDTH, out_image)

            cv2.imshow('Finger', out_image)

            keypress = cv2.waitKey(1) & 0xFF
            if keypress == 27:
                break
            elif keypress == ord('s'):
                cv2.imwrite('screenshot.jpg', finger_image)

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
"""
This script provide methods to operate the contours
1. get_defect_points
    Get left and right defect points from contour
2. segment_diff_fingers
    Segment whole contour into up and down contours according to defect points
3. add_touch_line_to_contour
    Add touch line points to the contour
4. get_boundary_points
    Get 4 boundary points of the up and down contour 
5. get_centroid
    Get the centroid of a boundary
6. get_touch_line_curve
    Get touch line curve
"""

import numpy as np
import cv2

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
        contour {[type]} -- [description]
        defect_points {[type]} -- [description]
        touch_points {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    if contour is None or defect_points is None:
        return None, None

    (x1, y1), (x2, y2) = defect_points

    # TODO: some error here, 179938 image in step=2 dataset
    if abs(x2 - x1) < 1e-6:
        up_finger = contour[contour[:, 0, 0] <= x1]
        down_finger = contour[contour[:, 0, 0] >= x1]
    else:
        grad_direc = (y2 - y1) / (x2 - x1)
        segment_line = lambda x, y: grad_direc * (x - x1) - (y - y1)
        up_finger = contour[segment_line(contour[:, 0, 0], contour[:, 0,
                                                                   1]) >= 0]
        down_finger = contour[segment_line(contour[:, 0, 0], contour[:, 0,
                                                                     1]) <= 0]

    return up_finger, down_finger


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

    top_left = None
    left_bd = np.where(up_contour[:, 0, 0] == 0)[0]
    if len(left_bd) > 0:
        y_list = up_contour[left_bd, 0, 1]
        top_left = (0, max(y_list))
    else:
        top_bd = np.where(up_contour[:, 0, 1] == 0)[0]
        if len(top_bd) > 0:
            x_list = up_contour[top_bd, 0, 0]
            top_left = (min(x_list), 0)
        else:
            top_left = None

    top_right = None
    right_bd = np.where(up_contour[:, 0, 0] == width - 1)[0]
    if len(right_bd) > 0:
        y_list = up_contour[right_bd, 0, 1]
        top_right = (width - 1, max(y_list))
    else:
        top_bd = np.where(up_contour[:, 0, 1] == 0)[0]
        if len(top_bd) > 0:
            x_list = up_contour[top_bd, 0, 0]
            top_right = (max(x_list), 0)
        else:
            top_right = None

    bottom_left = None
    left_bd = np.where(down_contour[:, 0, 0] == 0)[0]
    if len(left_bd) > 0:
        y_list = down_contour[left_bd, 0, 1]
        bottom_left = (0, min(y_list))
    else:
        bottom_bd = np.where(down_contour[:, 0, 1] == height - 1)[0]
        if len(bottom_bd) > 0:
            x_list = down_contour[bottom_bd, 0, 0]
            bottom_left = (min(x_list), height - 1)
        else:
            bottom_left = None

    bottom_right = None
    right_bd = np.where(down_contour[:, 0, 0] == width - 1)[0]
    if len(right_bd) > 0:
        y_list = down_contour[right_bd, 0, 1]
        bottom_right = (width - 1, min(y_list))
    else:
        bottom_bd = np.where(down_contour[:, 0, 1] == height - 1)[0]
        if len(bottom_bd) > 0:
            x_list = down_contour[bottom_bd, 0, 0]
            bottom_right = (max(x_list), height - 1)
        else:
            left_bd = np.where(down_contour[:, 0, 0] == 0)[0]
            if len(left_bd) > 0:
                y_list = down_contour[left_bd, 0, 1]
                bottom_right = (0, max(y_list))
            else:
                bottom_right = None

    return top_left, top_right, bottom_left, bottom_right

def get_centroid(contour):
    """Get the centroid of the contour
    
    Arguments:
        contour {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    if contour is None:
        return None, None

    moment = cv2.moments(contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None, None

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
        theta [float]
    """
    if contour is None or fitting_curve is None or defect_points is None:
        return None, None

    # Get the points used to fit
    if bound_points is None or bound_points[0] is None or bound_points[
            1] is None:
        return None, None

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
        return None, None
    
    curve = fitting_curve(X, Y)

    if draw_image is not None:
        # Draw the points used to fit
        color = [0, 0, 255] if IS_UP else [255, 0, 0]
        draw_points(draw_image, contour.astype("int"), radius=3, color=color)

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
        draw_points(draw_image, points, radius=3, color=color)

    return curve, theta
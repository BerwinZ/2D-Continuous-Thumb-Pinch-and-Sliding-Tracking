import cv2
import numpy as np
from scipy.optimize import fsolve, curve_fit

def __get_intersection(curve, defect_points, theta=None):
    """Get intersection point between the vertical bisector and the curve
    
    Arguments:
        defect_points {[type]} -- [description]
        curve {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    # Calculate the middle point
    if theta is None:
        (x1, y1), (x2, y2) = defect_points
        middle_point = ((x1 + x2) // 2, (y1 + y2) // 2)

        if abs(y2 - y1) < 1e-9:
            touch_point = (middle_point[0], curve(middle_point[0]))
        else:
            k = -(x2 - x1) / (y2 - y1)
            line = lambda x: k * (x - middle_point[0]) + middle_point[1]
            intersec = fsolve(lambda x: line(x) - curve(x), middle_point[0])[0]
            touch_point = (intersec, line(intersec))
    else:
        # Rotate the defect points
        rotate = __rotate_array(theta)
        r_defect_points = rotate.dot(np.array(defect_points).T).T
        t_touch_point = __get_intersection(curve, r_defect_points)
        # Rotate the touch point back
        rotate = __rotate_array(-theta)
        touch_point = rotate.dot(np.array([t_touch_point]).T).T[0]

    return tuple(touch_point)


def __rotate_array(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def get_touch_point(defect_points,
                    up_centroid,
                    down_centroid,
                    up_touch_line,
                    down_touch_line,
                    theta,
                    image_for_grad=None):
    """Get the touch point according to the defect points and touch line.
    
    Arguments:
        defect_points {[type]} -- [description]
        up_centroid {[type]} -- [description]
        down_centroid {[type]} -- [description]
        up_touch_line {[type]} -- [description]
        down_touch_line {[type]} -- [description]
        theta {[type]} -- [description]
    
    Keyword Arguments:
        image_for_grad {[type]} -- [description] (default: {None})
    """
    if defect_points is None or up_centroid is None or down_centroid is None or up_touch_line is None or down_touch_line is None or theta is None:
        return None

    # Calculate the middle point of defect points
    (x1, y1), (x2, y2) = defect_points
    defect_middle = ((x1 + x2) // 2, (y1 + y2) // 2)

    distance_ratio = points_distance(defect_middle,
                                     up_centroid) / points_distance(
                                         defect_middle, down_centroid)

    if distance_ratio >= 1:
        touch_point = __get_intersection(up_touch_line, defect_points)
    elif 0.65 <= distance_ratio <= 1:
        touch_point1 = __get_intersection(up_touch_line, defect_points)
        touch_point2 = __get_intersection(down_touch_line, defect_points,
                                          theta)
        touch_point = ((touch_point1[0] + touch_point2[0]) / 2,
                       (touch_point1[1] + touch_point2[1]) / 2)
    elif distance_ratio < 0.65:
        touch_point = __get_intersection(down_touch_line, defect_points, theta)

    touch_point = tuple(map(int, touch_point))

    if image_for_grad is not None:
        touch_point = __get_min_grad(
            cv2.cvtColor(image_for_grad, cv2.COLOR_BGR2GRAY), defect_points,
            touch_point)

    return touch_point


## TODO: Need to improve the effiency. Can crop the image
def __get_min_grad(gray_img, defect_points, start_point):
    height, width = gray_img.shape[0], gray_img.shape[1]

    def __IsValid(x, y):
        """Check whether x and y is in the boundary of img
        """
        return 0 <= x < width and 0 <= y < height

    ck_d = points_distance(defect_points[0], defect_points[1]) / 10
    G, _ = sobel_filters(gray_img)
    (x1, y1), (x2, y2) = defect_points
    defect_middle = ((x1 + x2) // 2, (y1 + y2) // 2)

    if abs(y1 - y2) < 1e-9:
        min_y = max(0, start_point[1] - ck_d // 2)
        max_y = min(height - 1, start_point[1] + ck_d // 2)
        Y = np.arange(min_y, max_y)
        X = np.array([start_point[0]] * Y.shape[0])
    else:
        k = -(x2 - x1) / (y2 - y1)
        line = lambda x: k * (x - defect_middle[0]) + defect_middle[1]

        dx = (int)(abs(ck_d / 2 * np.cos(np.arctan(k))))
        min_x = max(0, start_point[0] - dx)
        max_x = min(width - 1, start_point[0] + dx)
        X = np.arange(min_x, max_x)
        Y = line(X).astype("int")
        X = X[(Y >= 0) & (Y < height)]
        Y = Y[(Y >= 0) & (Y < height)]

    if X.size > 0:
        index = np.where(G[Y, X] == max(G[Y, X]))[0][0]
        return (X[index], Y[index])
    else:
        return start_point

def show_whole_finger_motion(up_direcs, down_direcs):
    if up_direcs is None or down_direcs is None:
        return None

    vector_up = sum(up_direcs)
    vector_down = sum(down_direcs)

    # Normalization (?)

    norm_up = np.linalg.norm(vector_up)
    norm_down = np.linalg.norm(vector_down)

    if norm_up > 15 and norm_down > 30:
        print('Angle', (int)(my_arctan_degrees(*vector_up)), 'Dis', (int)(norm_up))
        print('Angle', (int)(my_arctan_degrees(*vector_down)), 'Dis', (int)(norm_down))
        print('-' * 60)

    return None
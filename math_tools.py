import numpy as np


def points_distance(point1, point2):
    """Return the distance between point 1 and point2
    
    Arguments:
        point1 {[type]} -- [description]
        point2 {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    if type(point1) != tuple or type(point2) != tuple:
        return None
    
    if point1[0] is None or point1[1] is None or point2[0] is None or point2[1] is None:
        return None

    d = np.sqrt(np.sum(np.square(np.array(point1) - np.array(point2))))
    return d


def scaler(value, old_range, new_range):
    """Project value from [min_old, max_old] to [min_new, max_new]

    Arguments:
        value {float} -- [description]
        min_base_target {list} -- [min_old, min_new]
        max_base_target {list} -- [max_old, max_new]

    Returns:
        value -- [projected value]
    """
    if value is None or all(old_range) is False or all(new_range) is False:
        return None

    min_old, max_old = old_range
    min_new, max_new = new_range
    return (value - min_old) / (max_old - min_old) * (max_new -
                                                      min_new) + min_new
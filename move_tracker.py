from enum import IntEnum


class point_type(IntEnum):
    MIN_X = 0
    MAX_X = 1
    MIN_Y = 2
    MAX_Y = 3


def scaler(value, old_range, new_range):
    """Project value from [min_old, max_old] to [min_new, max_new]

    Arguments:
        value {float} -- [description]
        min_base_target {list} -- [min_old, min_new]
        max_base_target {list} -- [max_old, max_new]

    Returns:
        value -- [projected value]
    """
    min_old, max_old = old_range
    min_new, max_new = new_range
    return (value - min_old) / (max_old - min_old) * (max_new -
                                                      min_new) + min_new


# ------------------------------------------------
# Parameters for touch point tracker
# ------------------------------------------------
CALIBRATE_BASE_TOUCH = [181, 353, 87, 300]


class touch_trakcer:
    def __init__(self):
        """Used for tracking the touch point. Transfer the movement of touch point in image coordinate to the movement's of finger in real world.
        """
        self.cur_point_type = point_type.MIN_X
        self.touch_base_point = CALIBRATE_BASE_TOUCH

    def calibrate_touch_point(self, point):
        """Reset the old touch point

        Arguments:
            point {[type]} -- [description]
        """
        if point is None:
            return

        if self.cur_point_type == point_type.MIN_X or self.cur_point_type == point_type.MAX_X:
            self.touch_base_point[int(self.cur_point_type)] = point[0]
        else:
            self.touch_base_point[int(self.cur_point_type)] = point[1]
        print("Store base touch point", self.cur_point_type)
        print("Current base touch points", self.touch_base_point)
        self.cur_point_type = point_type((int(self.cur_point_type) + 1) % 4)

    def calc_scaled_move(self, point, MOVE_SCALE_RANGE=[-1, 1]):
        """Canculate the scaled movements of current touch points to the base points

        Arguments:
            point {tuple} -- [current touch position]

        Returns:
            dx {float} -- [scaled movement in x direction]
            dy {float}  -- [scaled movement in y direction]
        """
        if point is None:
            return None, None

        dx = scaler(point[0], (self.touch_base_point[int(
            point_type.MIN_X)], self.touch_base_point[int(point_type.MAX_X)]),
                      MOVE_SCALE_RANGE)
        dy = scaler(point[1], (self.touch_base_point[int(
            point_type.MIN_Y)], self.touch_base_point[int(point_type.MAX_Y)]),
                      MOVE_SCALE_RANGE)

        return dx, dy


# ------------------------------------------------
# Parameters for boundary point tracker
# ------------------------------------------------
CALIBRATE_BASE_BOUNDARY = [-101, 544, 56, -135]


class bound_trakcer:
    def __init__(self, IMG_HEIGHT=None, IMG_WIDTH=None):
        """Used for tracking the boundary points. Transfer the movement of boundary points in image coordinate to the movement's of finger in real world.
        """
        self.cur_point_type = point_type.MIN_X
        self.img_height = IMG_HEIGHT
        self.img_width = IMG_WIDTH
        self.bound_base_point = CALIBRATE_BASE_BOUNDARY

    def calibrate_boundary_point(self, up_bound, down_bound):
        """Reset the old touch point

        Arguments:
            point {[type]} -- [description]
        """
        if up_bound is None or down_bound is None:
            return

        if self.cur_point_type == point_type.MIN_X or self.cur_point_type == point_type.MAX_X:
            if down_bound[0] == 0:
                x_value = 0 - (self.img_height - down_bound[1])
            else:
                x_value = down_bound[0] + self.img_height - down_bound[1]
            self.bound_base_point[int(self.cur_point_type)] = x_value
        else:
            y_value = up_bound[0] - up_bound[1]
            self.bound_base_point[int(self.cur_point_type)] = y_value

        print("Store base boundary point", self.cur_point_type)
        print("Current base boundary points", self.bound_base_point)
        self.cur_point_type = point_type((int(self.cur_point_type) + 1) % 4)

    def calc_scaled_move(self, up_bound, down_bound, MOVE_SCALE_RANGE=[-1, 1]):
        """Canculate the scaled movements of current boundary points to the base points 
        
        Arguments:
            up_bound {[type]} -- [description]
            down_bound {[type]} -- [description]
        
        Keyword Arguments:
            MOVE_SCALE_RANGE {list} -- [description] (default: {[-1, 1]})
        
        Returns:
            [type] -- [description]
        """
        if up_bound is None or down_bound is None:
            return None, None

        y_value = up_bound[0] - up_bound[1]
        if down_bound[0] == 0:
            x_value = 0 - (self.img_height - down_bound[1])
        else:
            x_value = down_bound[0] + self.img_height - down_bound[1]

        dx = scaler(x_value, (self.bound_base_point[int(
            point_type.MIN_X)], self.bound_base_point[int(point_type.MAX_X)]),
                      MOVE_SCALE_RANGE)
        dy = scaler(y_value, (self.bound_base_point[int(
            point_type.MIN_Y)], self.bound_base_point[int(point_type.MAX_Y)]),
                      MOVE_SCALE_RANGE)

        return dx, dy

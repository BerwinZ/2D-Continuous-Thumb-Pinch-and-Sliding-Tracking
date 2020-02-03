from enum import IntEnum
import numpy as np
from math_tools import scaler

class point_type(IntEnum):
    MIN_X = 0
    MAX_X = 1
    MIN_Y = 2
    MAX_Y = 3

# ------------------------------------------------
# Parameters for touch point tracker
# ------------------------------------------------
# [Left(X_value), Right(X_value), Up(Y_value), Down(Y_value)]
CALIBRATE_BASE_TOUCH = [144, 340, 250, 349]


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
            # Store the touch position
            self.touch_base_point[int(self.cur_point_type)] = point[0]
        else:
            # Store the touch position
            self.touch_base_point[int(self.cur_point_type)] = point[1]

        # Print updated calibration data
        print("Store base touch point", self.cur_point_type)
        print("Current base touch points", self.touch_base_point)

        # Update current storing state
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
# Parameters for pi camera and user
# ------------------------------------------------
# Dot per inch (2.54 cm)
CAMERA_DPI = 96
# Pixel to length
pixel_length = 2.54 / CAMERA_DPI
# Length (cm) of the first knuckle of the thumb
THUMB_LENGTH = 2.54
# Imaging coefficient (Roughly)
CAMERA_COEFF = 480 / CAMERA_DPI / (THUMB_LENGTH / 2)
# ------------------------------------------------
# Parameters for correct tracker
# ------------------------------------------------
# [Left(Angle), Right(Angle), Up(Y_value), Down(Y_value)]
CALIBRATE_BASE_CORRECT = [151, 101, 73, 139]


class correct_tracker:
    def __init__(self):
        """Used for tracking the touch point. Transfer the movement of touch point in image coordinate to the movement's of finger in real world.
        """
        self.touch_base_correct = CALIBRATE_BASE_CORRECT

    def calibrate_touch_point(self, angle, up_cent_y):
        """Reset the old touch point

        Arguments:
            point {[type]} -- [description]
        """
        if angle is None or up_cent_y is None:
            return

        if self.cur_point_type == point_type.MIN_X or self.cur_point_type == point_type.MAX_X:
            self.touch_base_correct[int(self.cur_point_type)] = angle
        else:
            self.touch_base_correct[int(self.cur_point_type)] = up_cent_y

        # Print updated calibration data
        print("Store base touch point", self.cur_point_type)
        print("Current base angle", self.touch_base_correct)

        # Update current storing state
        self.cur_point_type = point_type((int(self.cur_point_type) + 1) % 4)

    def calc_scaled_move(self,
                         touch_angle,
                         up_cent_y,
                         MOVE_SCALE_RANGE=[-1, 1]):
        """Calculate the horizontal and vertical movements with correction
        
        Arguments:
            touch_angle {[type]} -- [description]
            up_cent_y {[type]} -- [description]
        
        Keyword Arguments:
            MOVE_SCALE_RANGE {list} -- [description] (default: {[-1, 1]})
        
        Returns:
            [type] -- [description]
        """
        if touch_angle is None or up_cent_y is None:
            return None, None

        dx = scaler(touch_angle, self.touch_base_correct[:2], MOVE_SCALE_RANGE)
        dy = scaler(self.__coor_to_real_pos(up_cent_y),
                    (self.__coor_to_real_pos(self.touch_base_correct[int(
                        point_type.MIN_Y)]),
                     self.__coor_to_real_pos(self.touch_base_correct[int(
                         point_type.MAX_Y)])), MOVE_SCALE_RANGE)

        return dx, dy

    def __coor_to_real_pos(self, y_value):
        """Convert the y_value coordinate to the real length in world
        
        Arguments:
            y_value {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        if y_value is None:
            return None

        value = (2 * y_value * pixel_length) / THUMB_LENGTH / CAMERA_COEFF
        # print(value)
        if abs(value) > 1:
            return None
        else:
            return np.sin(np.arccos(value)) * THUMB_LENGTH


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

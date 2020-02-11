from enum import IntEnum
import numpy as np
from math_tools import scaler
from scipy.optimize import fsolve
from draw_tools import draw_points


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
# Dot per inch (1 inch = 2.54 cm)
# Camera para
CAMERA_DPI = 96

# How long (cm) of 1 pixel in image worth in real world
PIXEL_TO_LEN = 2.54 / CAMERA_DPI

# Length (cm) of the first knuckle of the thumb
# Average length of human
THUMB_LENGTH = 2.54

# Imaging coefficient (Roughly)
# Length in image / CAMERA_COEFF = Length in real / Distance to the camera
# 1.064 is measured in the 3d Model of Fusion 360
CAMERA_COEFF = (480 * PIXEL_TO_LEN) / (THUMB_LENGTH / 2) * 1.064

# Distance (cm) 
# From origin point to the camera (M)
# Measured in 3d Model of Fusion 360
ORIGIN_TO_CAMERA = 1.746

# Distance (cm) 
# From origin point to the joint of thumb (N)
# Measured in 3d Model Fusion 360
ORIGIN_TO_JOINT = 1.7

# ------------------------------------------------
# Parameters for correct tracker
# ------------------------------------------------
# [Left(Angle), Right(Angle), Up(Y_value), Down(Y_value)]
CALIBRATE_BASE_CORRECT = [-0.47993001423007386, -0.022311212016247174, 2.5028753838888025, 2.3770159537675566]


class correct_tracker:
    def __init__(self):
        """Used for tracking the touch point. Transfer the movement of touch point in image coordinate to the movement's of finger in real world.
        """
        self.touch_base_correct = CALIBRATE_BASE_CORRECT
        self.cur_point_type = point_type.MIN_X

    def calibrate_touch_point(self, 
                             up_curve, 
                             up_centroid, 
                             touch_point):
        """Reset the old touch point

        Arguments:
            point {[type]} -- [description]
        """
        if up_curve is None or up_centroid is None or touch_point is None:
            return None

        if self.cur_point_type == point_type.MIN_X or self.cur_point_type == point_type.MAX_X:
            self.touch_base_correct[int(self.cur_point_type)] = self.coor_to_real_len(up_curve, up_centroid, touch_point)
        else:
            self.touch_base_correct[int(self.cur_point_type)] = self.__coor_to_real_pos(up_centroid[1])

        # Print updated calibration data
        print("Store base touch point", self.cur_point_type)
        print("Current base angle", self.touch_base_correct)

        # Update current storing state
        self.cur_point_type = point_type((int(self.cur_point_type) + 1) % 4)

    def calc_scaled_move(self,
                         up_curve, 
                         up_centroid, 
                         touch_point,
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
        if up_curve is None or up_centroid is None or touch_point is None:
            return None, None

        dx = scaler(self.coor_to_real_len(up_curve, up_centroid, touch_point),
                    self.touch_base_correct[:2], 
                    MOVE_SCALE_RANGE)
        dy = scaler(self.__coor_to_real_pos(up_centroid[1]),
                    self.touch_base_correct[2:],
                    MOVE_SCALE_RANGE)

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

        value = (2 * y_value * PIXEL_TO_LEN) / THUMB_LENGTH / CAMERA_COEFF
        # print(value)
        if abs(value) > 1:
            return None
        else:
            return np.sin(np.arccos(value)) * THUMB_LENGTH

    def coor_to_real_len(self, up_curve, up_centroid, touch_point):

        if up_curve is None or up_centroid is None or touch_point is None:
            return None

        m_x = up_centroid[0]
        im_x = touch_point[0] - m_x
        im_y = up_curve(m_x)

        f = lambda theta: CAMERA_COEFF * (THUMB_LENGTH * np.cos(
            theta) - ORIGIN_TO_CAMERA) / (THUMB_LENGTH * np.sin(
                theta) + ORIGIN_TO_JOINT) - im_y * PIXEL_TO_LEN
        thumb_theta = fsolve(f, 0)[0]
        real_x = im_x / im_y * (THUMB_LENGTH * np.cos(thumb_theta) - ORIGIN_TO_CAMERA)

        return real_x


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

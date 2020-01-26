from enum import IntEnum


class point_type(IntEnum):
    MIN_X = 0
    MAX_X = 1
    MIN_Y = 2
    MAX_Y = 3


CALIBRATE_BASE = [181, 353, 87, 300]


class point_trakcer:
    def __init__(self):
        self.cur_point_type = point_type.MIN_X
        self.base_point = CALIBRATE_BASE

    def calibrate_base_point(self, point):
        """Reset the old touch point

        Arguments:
            point {[type]} -- [description]
        """
        if point is None:
            return

        if self.cur_point_type == point_type.MIN_X or self.cur_point_type == point_type.MAX_X:
            self.base_point[int(self.cur_point_type)] = point[0]
        else:
            self.base_point[int(self.cur_point_type)] = point[1]
        print("Store base point", self.cur_point_type)
        print("Current base points", self.base_point)
        self.cur_point_type = point_type((int(self.cur_point_type) + 1) % 4)

    def calc_scaled_rela_move(self, point, MOVE_SCALE_RANGE=1):
        """Canculate the relative movements of current touch points to the old touch points

        Arguments:
            point {tuple} -- [current touch position]

        Returns:
            dx {float} -- [relative movement in x direction]
            dy {float}  -- [relative movement in y direction]
        """
        if point is None:
            return None, None

        dx = self._scaler(
            point[0],
            [self.base_point[int(point_type.MIN_X)], -MOVE_SCALE_RANGE],
            [self.base_point[int(point_type.MAX_X)], MOVE_SCALE_RANGE])
        dy = self._scaler(
            point[1],
            [self.base_point[int(point_type.MIN_Y)], -MOVE_SCALE_RANGE],
            [self.base_point[int(point_type.MAX_Y)], MOVE_SCALE_RANGE])

        return dx, dy

    def _scaler(self, value, min_base_target, max_base_target):
        """Project value from [min_base, max_base] to [min_target, max_target]

        Arguments:
            value {float} -- [description]
            min_base_target {list} -- [min_base, min_target]
            max_base_target {list} -- [max_base, max_target]

        Returns:
            value -- [projected value]
        """
        min_base, min_target = min_base_target
        max_base, max_target = max_base_target
        return (value - min_base) / (max_base - min_base) * (
            max_target - min_target) + min_target

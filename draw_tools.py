"""
A board that allow user to draw points on the board
"""

import cv2
import numpy as np


def draw_vertical_lines(img, line_num=0):
    """Draw white vertical lines in img

    Arguments:
        img {[type]} -- [description]
        line_num {[type]} -- [description]
    """
    if img is None:
        return

    width, height = img.shape[1], img.shape[0]

    for i in range(line_num):
        cv2.line(img, (width // (line_num + 1) * (i + 1), 0),
                 (width // (line_num + 1) * (i + 1), height), [255, 255, 255],
                 3)


def draw_points(img, points, radius=5, color=[0,255,0]):
    """Draw point(s) in the img
    
    Arguments:
        img {[type]} -- [description]
        points {[type]} -- [description]
        radius {[type]} -- [description]
        color {[type]} -- [description]
    """
    if points is None:
        return
    elif type(points) == tuple:
        cv2.circle(img, points, radius, color, -1)
    elif type(points) == list:
        for p in points:
            if p:
                cv2.circle(img, p, radius, color, -1)


class draw_board:
    def __init__(self, WIDTH=480, HEIGHT=480, RADIUS=10, MAX_POINTS=10):
        """Draw board for showing the points

        Keyword Arguments:
            LENGTH {int} -- [description] (default: {480})
            WIDTH {int} -- [description] (default: {480})
            RADIUS {int} -- [description] (default: {10})
            MAX_POINTS {int} -- [description] (default: {10})
        """
        self.width = WIDTH
        self.height = HEIGHT
        self.radius = RADIUS
        self.board = np.zeros((HEIGHT, WIDTH, 3))
        self.points = []
        self.max_points = MAX_POINTS

    def reset_board(self):
        """Reset the board to all 0 array
        """
        self.board = self.board * 0
        self.points = []

    def draw_filled_point(self, point, middle=True, color=[255, 0, 0]):
        """Draw points on the board
        
        Arguments:
            point {[type]} -- [description]
        
        Keyword Arguments:
            middle {bool} -- [description] (default: {True})
            color {list} -- [description] (default: {[255, 0, 0]})
        """
        if point[0] is None or point[1] is None:
            self.points.append(None)
        else:
            if middle:
                new_point = (int(point[0] + self.width / 2),
                             int(point[1] + self.height / 2))
            else:
                new_point = tuple(point)
            self.points.append(new_point)

        if len(self.points) > self.max_points:
            if self._IsValid(self.points[0]):
                cv2.circle(self.board, self.points[0], self.radius, [0, 0, 0],
                           -1)
            self.points.pop(0)
            for p in self.points:
                if self._IsValid(p):
                    cv2.circle(self.board, p, self.radius, color, -1)
        else:
            if self._IsValid(self.points[-1]):
                cv2.circle(self.board, new_point, self.radius, color, -1)

    def _IsValid(self, point):
        return point is not None and 0 <= point[0] < self.width and 0 <= point[
            1] < self.height


if __name__ == '__main__':
    """Test the draw_board class
    """
    board = draw_board()
    board.draw_filled_point((0, 0), color=[255, 0, 0])
    board.draw_filled_point((100, 0), color=[0, 255, 0])
    cv2.imshow("Board", board.board)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

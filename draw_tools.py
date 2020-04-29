"""
Drawing tools collections
1. Draw vertical lines in image
2. Draw points on image
3. Draw contours on image
4. DrawBoard (class)
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
    """Draw point(s) in the img. Supporting type of points include
         1. None
         2. Tuple
         3. List
         4. np.ndarray
    
    Arguments:
        img {[type]} -- [description]
        points {[type]} -- [description]
        radius {[type]} -- [description]
        color {[type]} -- [description]
    """
    if points is None:
        return
    elif type(points) == tuple:
        if points[0] is not None and points[1] is not None:
            cv2.circle(img, tuple(points), radius, color, -1)
    elif type(points) == list:
        for p in points:
            if p is not None:
                cv2.circle(img, tuple(p), radius, color, -1)
    elif type(points) == np.ndarray:
        for i in range(points.shape[0]):
            cv2.circle(img, tuple(points[i, :]), radius, color, -1)


def draw_contours(img, contours, thickness=3, color=[0,255,0]):
    """Draw contour(s) in the img
    
    Arguments:
        img {[type]} -- [description]
        contours {[type]} -- [description]
    
    Keyword Arguments:
        thickness {int} -- [description] (default: {3})
        color {list} -- [description] (default: {[0,255,0]})
    """
    if contours is None:
        return
    
    if type(contours) == list:
        for c in contours:
            if c is not None:
                cv2.drawContours(img, [c], 0, color=color, thickness=thickness)
    else:
        cv2.drawContours(img, [contours], 0, color=color, thickness=thickness)


class DrawBoard:
    def __init__(self, WIDTH=480, HEIGHT=480, RADIUS=10, MAX_POINTS=10):
        """Draw board for points with points persistence

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
        self.pts_queue = []
        self.max_points = MAX_POINTS

    def reset_board(self):
        """Reset the board to all 0 array
        """
        self.board = self.board * 0
        self.pts_queue = []

    def update_dot(self, point, middle=True, scaler=[1, 1], color=[255, 0, 0]):
        """Draw points on the board
        
        Arguments:
            point {[type]} -- [description]
        
        Keyword Arguments:
            middle {bool} -- [description] (default: {True})
            color {list} -- [description] (default: {[255, 0, 0]})
        """
        if point is None or point[0] is None or point[1] is None:
            self.pts_queue.append(None)
        else:
            if middle:
                new_point = (int(point[0] * scaler[0] + self.width / 2),
                             int(point[1] * scaler[1] + self.height / 2))
            else:
                new_point = tuple(point)

            self.pts_queue.append(new_point)

        if len(self.pts_queue) > self.max_points:
            if self._IsValid(self.pts_queue[0]):
                cv2.circle(self.board, self.pts_queue[0], self.radius, [0, 0, 0],
                           -1)
            self.pts_queue.pop(0)
            for p in self.pts_queue:
                if self._IsValid(p):
                    cv2.circle(self.board, p, self.radius, [209, 206, 0], -1)
            if self._IsValid(self.pts_queue[-1]):
                cv2.circle(self.board, p, self.radius, color, -1)
        else:
            if self._IsValid(self.pts_queue[-1]):
                cv2.circle(self.board, new_point, self.radius, color, -1)

    def _IsValid(self, point):
        return point is not None and 0 <= point[0] < self.width and 0 <= point[
            1] < self.height


if __name__ == '__main__':
    """Test the DrawBoard class
    """
    board = DrawBoard()
    board.update_dot((0, 0), color=[255, 0, 0])
    board.update_dot((100, 0), color=[0, 255, 0])
    cv2.imshow("Board", board.board)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

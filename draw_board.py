"""
A board that allow user to draw points on the board
"""

import cv2
import numpy as np

class draw_board:
    """Draw board for showing the points
    """
    def __init__(self, LENGTH=480, WIDTH=480, DIMEN=3, MAX_POINTS=10):
        """Configure the board

        Keyword Arguments:
            LENGTH {int} -- [description] (default: {480})
            WIDTH {int} -- [description] (default: {480})
            DIMEN {int} -- [description] (default: {3})
        """
        self.board = np.zeros((WIDTH, LENGTH, DIMEN))
        self.points = []
        self.max_points = MAX_POINTS


    def reset_board(self):
        """Reset the board to all 0 array
        """
        self.board = self.board * 0
        self.points = []


    def draw_filled_point(self, point, radius=10, color=[255, 0, 0]):
        """Draw points on the board

        Arguments:
            point {[type]} -- [description]
            color {[type]} -- [description]
        """
        self.points.append(point)
        if len(self.points) > self.max_points:
            cv2.circle(self.board, self.points[0], radius, [0, 0, 0], -1)
            self.points.pop(0)
    
        cv2.circle(self.board, tuple(point), radius, color, -1)


if __name__ == '__main__':
    board = draw_board()
    board.draw_filled_point((0, 0), color=[255, 0, 0])
    board.draw_filled_point((100, 0), color=[0, 255, 0])
    cv2.imshow("Board", board.board)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

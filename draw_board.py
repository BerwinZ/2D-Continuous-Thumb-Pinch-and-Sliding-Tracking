"""
A board that allow user to draw points on the board
"""

import cv2
import numpy as np

class draw_board:
    def __init__(self, WIDTH=480, HEIGHT=480, DIMEN=3, MAX_POINTS=10):
        """Draw board for showing the points

        Keyword Arguments:
            LENGTH {int} -- [description] (default: {480})
            WIDTH {int} -- [description] (default: {480})
            DIMEN {int} -- [description] (default: {3})
            MAX_POINTS {int} -- [description] (default: {10})
        """
        self.board = np.zeros((HEIGHT, WIDTH, DIMEN))
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
            for p in self.points:
                cv2.circle(self.board, p, radius, color, -1)
        else:
            cv2.circle(self.board, tuple(point), radius, color, -1)


if __name__ == '__main__':
    """Test the draw_board class
    """
    board = draw_board()
    board.draw_filled_point((0, 0), color=[255, 0, 0])
    board.draw_filled_point((100, 0), color=[0, 255, 0])
    cv2.imshow("Board", board.board)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

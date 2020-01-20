"""
A board that allow user to draw points on the board
"""

import cv2
import numpy as np


def configure_board(LENGTH=480, WIDTH=480, DIMEN=3):
    """Configure the board

    Keyword Arguments:
        LENGTH {int} -- [description] (default: {480})
        WIDTH {int} -- [description] (default: {480})
        DIMEN {int} -- [description] (default: {3})
    """
    board = np.zeros((WIDTH, LENGTH, DIMEN))
    return board


def reset_board(board):
    """Reset the board to all 0 array
    """
    board = board * 0
    return board


def draw_point(board, point, radius=10, color=[255, 0, 0]):
    """Draw points on the board

    Arguments:
        point {[type]} -- [description]
        color {[type]} -- [description]
    """
    cv2.circle(board, tuple(point), radius, color, -1)


if __name__ == '__main__':
    board = configure_board()
    draw_point(board, (0, 0), color=[255, 0, 0])
    draw_point(board, (100, 0), color=[0, 255, 0])
    cv2.imshow("Board", board)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

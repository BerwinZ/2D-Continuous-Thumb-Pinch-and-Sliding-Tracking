"""
Drawing tools collections
1. Draw vertical lines in image
2. Draw points on image
3. Draw contours on image
4. DrawBoard (class)
"""

import cv2
import numpy as np
import random
from time import sleep
import threading
import timeit


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
        # self.pts_queue = []

    def update_dot(self, point, middle=True, scaler=[1, 1]):
        """Draw points on the board
        
        Arguments:
            point {[type]} -- [description]
        
        Keyword Arguments:
            middle {bool} -- [description] (default: {True})
            scaler {list} -- [scaler of x and y] (default: [1, 1])
        """
        if point is None or point[0] is None or point[1] is None:
            self.pts_queue.append(None)
        else:
            if middle:
                new_point = (int(point[0] * scaler[0] + self.width / 2),
                             int(-point[1] * scaler[1] + self.height / 2))
            else:
                new_point = (int(point[0]), int(point[1]))

            self.pts_queue.append(new_point)

        if len(self.pts_queue) > self.max_points:
            pt = self.pts_queue.pop(0)
            if self.__IsValid(pt):
                cv2.circle(self.board, pt, self.radius, [0, 0, 0], -1)
            
        for idx in range(len(self.pts_queue)):
            pt = self.pts_queue[idx]
            if self.__IsValid(pt):
                if idx == len(self.pts_queue) - 1:
                    cv2.circle(self.board, pt, self.radius, [255, 0, 0], -1)
                else:
                    cv2.circle(self.board, pt, self.radius, [209, 206, 0], -1)            
        
    def __IsValid(self, point):
        return point is not None and 0 <= point[0] < self.width and 0 <= point[
            1] < self.height


class TargetDotBoard:
    def __init__(self, WIDTH=480, HEIGHT=480, RADIUS=10, MAX_POINTS=10):
        self.drawboard = DrawBoard(WIDTH, HEIGHT, RADIUS, MAX_POINTS)
        self.board = self.drawboard.board
        self.width = WIDTH
        self.height = HEIGHT

        self.cur_pos = None

        self.target_radius = 20
        self.target_count = 10
        self.target_pos = None
        
        self.thread_sign = True
        
        self.task_cost_time = []
        
    def update_dot(self, point, middle=True, scaler=[1, 1]):
        if self.target_pos != None:
             cv2.circle(self.drawboard.board, self.target_pos, self.target_radius, [0, 255, 255], -1)
        
        if middle:
            self.cur_pos = (int(point[0] * scaler[0] + self.width / 2),
                            int(-point[1] * scaler[1] + self.height / 2))
        else:
            self.cur_pos = (int(point[0]), int(point[1]))

        self.drawboard.update_dot(point, middle, scaler)
        self.board = self.drawboard.board

    def start(self):
        self.target_count = 10
        self.thread_sign = True
        self.task_cost_time = []
        threading.Thread(target=self.__task).start()
    
    def stop(self):
        self.thread_sign = False

    def __task(self):
        while self.target_count > 0 and self.thread_sign:
            # generate new target pos
            if self.target_pos is not None:
                cv2.circle(self.drawboard.board, self.target_pos, self.target_radius, [0, 0, 0], -1)
            self.target_count = self.target_count - 1
            self.target_pos = self.__generate_target_dot()

            # draw the target pos
            cv2.circle(self.drawboard.board, self.target_pos, self.target_radius, [0, 255, 255], -1)
            self.board = self.drawboard.board

            # check current dot position
            arrived = False

            # wait the user touch the dot
            print('\n')
            print('-'*60)
            print("Target Dot Changed!")
            start_time = timeit.default_timer()
            
            # -------------------------------------------------------------
            # 1. Have a countdown
            # countdown = 5
            # while countdown > 0 and self.thread_sign:
            #     print("Countdown:", countdown)
            #     countdown -= 1
            #     for i in range(1000):
            #         if not arrived and self.__check_arrive():
            #             print("Arrive!")
            #             arrived = True
            #             stop_time = timeit.default_timer()
            #         sleep(1e-3)
            # print("Countdown:", 0)
            # -------------------------------------------------------------
            # 2. No countdown
            while self.thread_sign and not arrived:
                if self.__check_arrive():
                    print("Arrived!")
                    arrived = True
                    stop_time = timeit.default_timer()
                sleep(1e-3)
            # ---------------------------------------------------------------

            # show result
            print('-'*60)
            if not arrived:
                print("Not arrived")
                self.task_cost_time.append(None)
            else:
                cost_time = round(stop_time-start_time, 2)
                print("Arrive in", cost_time, 'seconds')
                self.task_cost_time.append(cost_time)
            
            sleep(1)
        
        self.target_pos = None
        self.drawboard.reset_board()
        print("Task finished. Cost time is")
        print(self.task_cost_time)
        
        print('-'*60)
        print("Mean cost time:")
        print(np.mean(self.task_cost_time))
        print('-'*60)

    def __generate_target_dot(self, offset = 100):
        x = random.randint(0 + offset, self.width - 1 - offset)
        y = random.randint(0 + offset, self.height - 1 - offset)
        return x, y

    def __check_arrive(self):
        if self.cur_pos is None or self.target_pos is None:
            return False
        
        return np.sqrt(
        (self.cur_pos[0] - self.target_pos[0])** 2 + (self.cur_pos[1]-self.target_pos[1])**2) < self.target_radius

if __name__ == '__main__':
    """Test the DrawBoard class
    """
    board = TargetDotBoard()
    def draw_f(event, x, y, flags, param):
        board.update_dot((x, y), middle=False)

    cv2.imshow("Board", board.board)
    cv2.setMouseCallback("Board", draw_f)
    while True:
        cv2.imshow("Board", board.board)
            
        keypress = cv2.waitKey(25) & 0xFF
        if keypress == 27:
            break
        elif keypress == ord('s'):
            board.start()

    cv2.destroyAllWindows()
    board.stop()

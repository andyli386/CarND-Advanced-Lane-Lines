from enum import Enum

import numpy as np
from collections import deque

import cv2

import matplotlib.pyplot as plt

class LineType(Enum):
    left = 0
    right = 1

class Line(object):
    def __init__(self, lineType):
        self.__frame_num = 15
        self.lineType = lineType
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = []
        #y values for detected line pixels
        self.ally = []
        self.x = []
        self.y = []
        self.current_bottom_x = None
        self.bottom_x = deque(maxlen=self.__frame_num)

        self.current_top_x = None
        self.top_x = deque(maxlen=self.__frame_num)

        # Record radius of curvature
        self.radius = None

        # Polynomial coefficients: x = A*y**2 + B*y + C
        self.A = deque(maxlen=self.__frame_num)
        self.B = deque(maxlen=self.__frame_num)
        self.C = deque(maxlen=self.__frame_num)
        self.fit = None
        self.fitx = None
        self.fity = None


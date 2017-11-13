import numpy as np
import matplotlib.pyplot as plt
import cv2
from Line import Line, LineType
from ImageUtils import ImageUtils


class Detector(object):
    def __init__(self):
        self.__leftLine, self.__rightLine = Line(LineType.left), Line(LineType.right)
        self.__nwindows = 8
        # Set the width of the windows +/- margin
        self.__margin = 45
        # Set minimum number of pixels found to recenter window
        self.__minpix = 50

        self.__imageUtils = ImageUtils()

    def detect(self, binary_image, plot=False):

        warped_result = self.__imageUtils.perspective(binary_image)

        result = self.__imageUtils.luv_lab_filter(warped_result)

        self.__set_binary_image(result)
        if self.__leftLine.detected:
            self.quick_search(self.__leftLine)
        else:
            self.blind_search(self.__leftLine, plot)
        if self.__rightLine.detected:
            self.quick_search(self.__rightLine)
        else:
            self.blind_search(self.__rightLine, plot)


        left_fitx, left_fity = self.get_fit(self.__leftLine)
        right_fitx, right_fity = self.get_fit(self.__rightLine)


        result = self.__imageUtils.draw_on_origin_image(binary_image, left_fitx, right_fitx, left_fity, right_fity, plot)


        offset, mean_curv = self.car_pos()

        result = self.__imageUtils.wirte_on_processed_image(result, offset, mean_curv)


        return result

    def quick_search(self, line):
        """
        Assuming in last frame, lane has been detected. Based on last x/y coordinates, quick search current lane.
        https://github.com/uranus4ever/Advanced-Lane-Detection/blob/master/Project.py
        """
        allx = []
        ally = []

        if line.detected:
            win_bottom = 720
            win_top = 630
            while win_top >= 0:
                yval = np.mean([win_top, win_bottom])
                xval = (np.median(line.current_fit[0])) * yval ** 2 + (np.median(line.current_fit[1])) * yval + (np.median(line.current_fit[2]))
                x_idx = np.where((((xval - 50) < self.__nonzerox)
                                  & (self.__nonzerox < (xval + 50))
                                  & ((self.__nonzeroy > win_top) & (self.__nonzeroy < win_bottom))))
                x_window, y_window = self.__nonzerox[x_idx], self.__nonzeroy[x_idx]
                if np.sum(x_window) != 0:
                    np.append(allx, x_window)
                    np.append(ally, y_window)
                win_top -= 90
                win_bottom -= 90
            line.allx = allx
            line.ally = ally
        if np.sum(allx) == 0:
            self.detected = False  # If no lane pixels were detected then perform blind search


    def blind_search(self, line, debug=False):
        allx = []
        ally = []
        base, window_bottom, window_top = self.__get_base(line.lineType)
        window_x_high, window_x_low = self.__get_x_low_high(base)

        x_idx, x_window, y_window = self.__get_xy_window(window_bottom, window_top, window_x_high, window_x_low)

        if debug:
            print(base, window_bottom, window_top,window_x_low, window_x_high)
            cv2.rectangle(self.__binary_image, (window_x_low, window_top), (window_x_high, window_bottom),
                          (0, 255, 0), 2)
        if np.sum(x_window) != 0:
            allx.extend(x_window)
            ally.extend(y_window)
        if len(x_idx[0]) > self.__minpix:
            base = np.int(np.mean(x_window))

        for window in range(1, self.__nwindows):
            window_bottom = window_top
            window_top = window_top - self.__window_height

            histogram = np.sum(self.__binary_image[window_top:window_bottom, :], axis=0)
            search_high = min(base + 100, 1280)
            search_low = max(base - 100, 0)
            x_move = np.argmax(histogram[search_low:search_high])
            base = x_move if x_move > 0 else (search_high - search_low) // 2
            base += search_low

            window_x_high, window_x_low = self.__get_x_low_high(base)

            x_idx, x_window, y_window = self.__get_xy_window(window_bottom, window_top, window_x_high, window_x_low)


            if np.sum(x_window) != 0:
                allx.extend(x_window)
                ally.extend(y_window)
            if len(x_idx[0]) > self.__minpix:
                base = np.int(np.mean(x_window))


        if np.sum(allx) > 0:
            self.detected = True

            line.allx = allx
            line.ally = ally


    def get_fit(self, line):

        line.current_fit = np.polyfit(line.ally, line.allx, 2)
        line.current_bottom_x = line.current_fit[0] * 720 ** 2 + line.current_fit[1] * 720 + line.current_fit[2]

        line.current_top_x = line.current_fit[2]

        line.bottom_x.append(line.current_bottom_x)
        #print("\nline.bottom_x = ", line.bottom_x);
        line.current_bottom_x = np.median(line.bottom_x)

        line.top_x.append(line.current_top_x)
        line.current_top_x = np.median(line.top_x)

        line.allx = np.append(line.allx, line.current_bottom_x)
        line.ally = np.append(line.ally, 720)

        line.allx = np.append(line.allx, line.current_top_x)
        line.ally = np.append(line.ally, 0)

        #print(line.lineType, " ", line.allx, " ", line.ally)

        sorted_idx = np.argsort(line.ally)
        #print(sorted_idx)
        line.allx = line.allx[sorted_idx]
        line.ally = line.ally[sorted_idx]


        line.current_fit = np.polyfit(line.ally, line.allx, 2)
        line.A.append(line.current_fit[0])
        line.B.append(line.current_fit[1])
        line.C.append(line.current_fit[2])
        line.fity = line.ally
        line.current_fit = [np.median(line.A), np.median(line.B), np.median(line.C)]
        line.fitx = line.current_fit[0] * line.fity ** 2 + line.current_fit[1] * line.fity + line.current_fit[2]


        return line.fitx, line.fity

    def curvature(self, line):
        """
        calculate curvature from fit parameter
        :param fit: [A, B, C]
        :return: radius of curvature (in meters unit)
        https://github.com/uranus4ever/Advanced-Lane-Detection/blob/master/Project.py
        """

        ym_per_pix = 18 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        fitx = line.current_fit[0] * self.__ploty ** 2 + line.current_fit[1] * self.__ploty + line.current_fit[2]
        y_eval = np.max(self.__ploty)
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.__ploty * ym_per_pix, fitx * xm_per_pix, 2)

        curved = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / \
                   np.absolute(2 * fit_cr[0])

        return curved

    def car_pos(self):
        """
        Calculate the position of car on left and right lane base (convert to real unit meter)
        :param left_fit:
        :param right_fit:
        :return: distance (meters) of car offset from the middle of left and right lane
        https://github.com/uranus4ever/Advanced-Lane-Detection/blob/master/Project.py
        """

        xleft_eval = self.__get_eval(self.__leftLine)
        xright_eval = self.__get_eval(self.__rightLine)

        ym_per_pix = 18 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / abs(xleft_eval - xright_eval)  # meters per pixel in x dimension
        xmean = np.mean((xleft_eval, xright_eval))
        offset = (self.__binary_image.shape[1] / 2 - xmean) * xm_per_pix  # +: car in right; -: car in left side

        left_curved = self.__get_curved(self.__leftLine.current_fit, xm_per_pix, ym_per_pix)

        right_curved = self.__get_curved(self.__rightLine.current_fit, xm_per_pix, ym_per_pix)

        mean_curv = np.mean([left_curved, right_curved])

        return offset, mean_curv


    def __get_eval(self, line):
        return line.current_fit[0] * np.max(self.__ploty) ** 2 + line.current_fit[1] * np.max(self.__ploty) + line.current_fit[2]

    def __get_curved(self, fit, xm_per_pix, ym_per_pix):
        y_eval = np.max(self.__ploty)

        fitx = fit[0] * self.__ploty ** 2 + fit[1] * self.__ploty + fit[2]
        fit_cr = np.polyfit(self.__ploty * ym_per_pix, fitx * xm_per_pix, 2)
        curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / \
                        np.absolute(2 * fit_cr[0])
        return curverad

    def __get_x_low_high(self, base):
        window_x_low = max(base - self.__margin, 0)
        window_x_high = min(base + self.__margin, 1280)
        return window_x_high, window_x_low

    def __get_xy_window(self, window_bottom, window_top, window_x_high, window_x_low):
        x_idx = np.where(((window_x_low < self.__nonzerox) & (self.__nonzerox < window_x_high)
                          & ((self.__nonzeroy > window_top) & (self.__nonzeroy < window_bottom))))
        x_window, y_window = self.__nonzerox[x_idx], self.__nonzeroy[x_idx]
        return x_idx, x_window, y_window

    def __get_base(self, lineType):
        small_window_bottom = self.__binary_image.shape[0]
        small_window_top = self.__binary_image.shape[0] - self.__window_height

        small_window_histogram = np.sum(self.__binary_image[small_window_top:small_window_bottom, :], axis=0)
        all_histogram = np.sum(self.__binary_image[200:, :], axis=0)


        if lineType == LineType.right:
            base = (np.argmax(small_window_histogram[self.__midpoint:-60]) + self.__midpoint) \
                if np.argmax(small_window_histogram[self.__midpoint:-60]) > 0 \
                else (np.argmax(all_histogram[self.__midpoint:]) + self.__midpoint)
        else:
            base = np.argmax(small_window_histogram[:self.__midpoint]) \
                if np.argmax(small_window_histogram[:self.__midpoint]) > 0 \
                else np.argmax(all_histogram[:self.__midpoint])
        return base, small_window_bottom, small_window_top

    def __set_binary_image(self, binary_image):
        self.__binary_image = binary_image

        nonzero = self.__binary_image.nonzero()
        self.__nonzeroy = np.array(nonzero[0])
        self.__nonzerox = np.array(nonzero[1])
        self.__window_height = np.int(self.__binary_image.shape[0] / self.__nwindows)
        self.__midpoint = np.int(self.__binary_image.shape[1] / 2)
        self.__ploty = np.linspace(0, self.__binary_image.shape[0] - 1, self.__binary_image.shape[0])



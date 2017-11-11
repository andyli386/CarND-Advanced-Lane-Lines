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

        if plot:
            self.__imageUtils.draw(warped_result, result)

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

        return result

    def quick_search(self, line):
        """
        Assuming in last frame, lane has been detected. Based on last x/y coordinates, quick search current lane.
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


        line.fit = np.polyfit(line.ally, line.allx, 2)
        line.A.append(line.fit[0])
        line.B.append(line.fit[1])
        line.C.append(line.fit[2])
        line.fity = line.ally
        line.fit = [np.median(line.A), np.median(line.B), np.median(line.C)]
        line.fitx = line.fit[0] * line.fity ** 2 + line.fit[1] * line.fity + line.fit[2]


        return line.fitx, line.fity

    def measuring_curvature(self):
        ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
        #print(ploty)
        quadratic_coeff = 3e-4  # arbitrary quadratic coefficient
        # For each y position generate random x position within +/-50 pix
        # of the line base position in each case (x=200 for left, and x=900 for right)
        leftx = np.array([200 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                          for y in ploty])
        rightx = np.array([900 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                           for y in ploty])
        #print(leftx.shape)

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # Fit a second order polynomial to pixel positions in each fake lane line
        left_fit = np.polyfit(ploty, leftx, 2)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fit = np.polyfit(ploty, rightx, 2)
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Plot up the fake data
        mark_size = 3
        plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
        plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(left_fitx, ploty, color='green', linewidth=3)
        plt.plot(right_fitx, ploty, color='green', linewidth=3)
        plt.gca().invert_yaxis()  # to visualize as we do the images
        plt.show()

        y_eval = np.max(ploty)
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
        #print(left_curverad, right_curverad)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                             (1 + (
                             2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        #print(left_curverad, 'm', right_curverad, 'm')



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


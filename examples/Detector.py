import numpy as np
import matplotlib.pyplot as plt
import cv2
from Line import Line, LineType



class Detector(object):
    def __init__(self):
        self.__leftLine, self.__rightLine = Line(LineType.left), Line(LineType.right)
        self.__nwindows = 9
        # Set the width of the windows +/- margin
        self.__margin = 45
        # Set minimum number of pixels found to recenter window
        self.__minpix = 50


    def set_binary_image(self, binary_image):
        self.__binary_image = binary_image

        nonzero = self.__binary_image.nonzero()
        self.__nonzeroy = np.array(nonzero[0])
        self.__nonzerox = np.array(nonzero[1])
        self.__window_height = np.int(self.__binary_image.shape[0] / self.__nwindows)
        self.__midpoint = np.int(self.__binary_image.shape[1] / 2)

    def blind_search(self, line):
        base = self.__get_base_new(line.lineType)
        window_x_low = base - self.__margin
        window_x_high = base + self.__margin



        pass

    def __get_base_new(self, lineType):
        small_window_bottom = self.__binary_image.shape[0]
        small_window_top = self.__binary_image.shape[0] - self.__window_height

        small_window_histogram = np.sum(self.__binary_image[small_window_top:small_window_bottom, :], axis=0)
        all_histogram = np.sum(self.__binary_image[200:, :], axis=0)

        if lineType == LineType.right:
            base = (np.argmax(small_window_histogram[self.mid_point:-60]) + self.mid_point) \
                if np.argmax(small_window_histogram[self.mid_point:-60]) > 0 \
                else (np.argmax(all_histogram[self.mid_point:]) + self.mid_point)
        else:
            base = np.argmax(small_window_histogram[:self.mid_point]) \
                if np.argmax(small_window_histogram[:self.mid_point]) > 0 \
                else np.argmax(all_histogram[:self.mid_point])
        return base


    def blind_search(self):
        leftx_base, rightx_base = self.__get_base()

        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.__nwindows):
            good_left_inds, good_right_inds = self.__get_blind_inds(leftx_current, rightx_current, window)
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.__minpix:
                leftx_current = np.int(np.median(self.__nonzerox[good_left_inds]))
            if len(good_right_inds) > self.__minpix:
                rightx_current = np.int(np.median(self.__nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        self.__leftLine.allx = self.__nonzerox[left_lane_inds]
        self.__leftLine.ally = self.__nonzeroy[left_lane_inds]
        self.__rightLine.allx = self.__nonzerox[right_lane_inds]
        self.__rightLine.ally = self.__nonzeroy[right_lane_inds]

    def get_fit(self):
        # Fit a second order polynomial to each
        self.__leftLine.current_fit = np.polyfit(self.__leftLine.ally, self.__leftLine.allx, 2)
        self.__rightLine.current_fit = np.polyfit(self.__rightLine.ally, self.__rightLine.allx, 2)
        ploty = np.linspace(0, self.__binary_image.shape[0] - 1, self.__binary_image.shape[0])
        self.__leftLine.current_fitx = self.__leftLine.current_fit[0] * ploty ** 2 + self.__leftLine.current_fit[1] * ploty + self.__leftLine.current_fit[2]
        self.__rightLine.current_fitx = self.__rightLine.current_fit[0] * ploty ** 2 + self.__rightLine.current_fit[1] * ploty + self.__rightLine.current_fit[2]
        return self.__leftLine.current_fitx, self.__rightLine.current_fitx, ploty

    def __get_blind_inds(self, leftx_current, rightx_current, window):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = self.__binary_image.shape[0] - (window + 1) * self.__window_height
        win_y_high = self.__binary_image.shape[0] - window * self.__window_height
        win_xleft_low = leftx_current - self.__margin
        win_xleft_high = leftx_current + self.__margin
        win_xright_low = rightx_current - self.__margin
        win_xright_high = rightx_current + self.__margin

        good_left_inds = ((self.__nonzeroy >= win_y_low) & (self.__nonzeroy < win_y_high) &
                          (self.__nonzerox >= win_xleft_low) & (self.__nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((self.__nonzeroy >= win_y_low) & (self.__nonzeroy < win_y_high) &
                           (self.__nonzerox >= win_xright_low) & (self.__nonzerox < win_xright_high)).nonzero()[0]

        return good_left_inds, good_right_inds

    def __get_base(self):
        histogram = np.sum(self.__binary_image[int(self.__binary_image.shape[0] / 2):, :], axis=0)

        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        return leftx_base, rightx_base

    def quick_search(self):

        left_lane_inds, right_lane_inds = self.__get_quick_inds()

        # Again, extract left and right line pixel positions
        self.__leftLine.allx = self.__nonzerox[left_lane_inds]
        self.__leftLine.ally = self.__nonzeroy[left_lane_inds]
        self.__rightLine.allx = self.__nonzerox[right_lane_inds]
        self.__rightLine.ally = self.__nonzeroy[right_lane_inds]

    def __get_quick_inds(self):
        left_lane_inds = ((self.__nonzerox > (self.__get_line_values(self.__leftLine) - self.__margin)) &
                          (self.__nonzerox < (self.__get_line_values(self.__leftLine) + self.__margin)))

        right_lane_inds = ((self.__nonzerox > (self.__get_line_values(self.__rightLine) - self.__margin)) &
                           (self.__nonzerox < (self.__get_line_values(self.__rightLine) + self.__margin)))

        return left_lane_inds, right_lane_inds

    def __get_line_values(self, line):
        return line.current_fit[0] * (self.__nonzeroy ** 2) + line.current_fit[1] * self.__nonzeroy + line.current_fit[2]

    def measuring_curvature(self):
        ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
        print(ploty)
        quadratic_coeff = 3e-4  # arbitrary quadratic coefficient
        # For each y position generate random x position within +/-50 pix
        # of the line base position in each case (x=200 for left, and x=900 for right)
        leftx = np.array([200 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                          for y in ploty])
        rightx = np.array([900 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                           for y in ploty])
        print(leftx.shape)

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
        print(left_curverad, right_curverad)

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
        print(left_curverad, 'm', right_curverad, 'm')
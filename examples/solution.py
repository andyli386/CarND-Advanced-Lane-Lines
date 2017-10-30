# coding: utf-8
import numpy as np
import cv2

import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from skimage import data, exposure, img_as_float
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from IPython import get_ipython

from ImageUtils import ImageUtils
import Line

def get_obj_img_points():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)


    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('../camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
    return objpoints, imgpoints




def save_to_pickle(objpoints, imgpoints, imageName, pickleName="../camera_cal/wide_dist_pickle.p"):
    img = cv2.imread(imageName)
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open(pickleName, "wb" ) )


def get_from_pickle(pickleName="../camera_cal/wide_dist_pickle.p"):
    dist_pickle = pickle.load( open(pickleName , "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    return mtx, dist


def process_thresh(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    imageUtils = ImageUtils()
    img = np.copy(img)

    img = imageUtils.reset_gamma(img)

    l_channel = imageUtils.get_hls(img, 'l')
    sobelx_binary = imageUtils.abs_sobel_thresh(l_channel, 'x', sobel_kernel=3, toGray=False, thresh=sx_thresh )

    s_channel = imageUtils.get_hls(img, 's')
    s_binary = imageUtils.apply_thresh(s_channel, s_thresh)

    #color_binary = np.dstack((np.zeros_like(sobelx_binary), sobelx_binary, s_binary))

    color_binary = np.zeros_like(s_binary)
    color_binary[(s_binary > 0) | (sobelx_binary > 0)] = 1
    return color_binary

#def pipeline(img,  mtx, dist, src, dst, s_thresh, sx_thresh):
#    processed_image = process_thresh(img, s_thresh, sx_thresh)
#    return perspective(processed_image, mtx, dist, src, dst)

def blind_search(warped_image):
    binary_warped = warped_image
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    #print(binary_warped.shape, histogram.shape)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    #draw(binary_warped, out_img)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.median(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.median(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    global left_fit, right_fit
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)





    return left_fitx, right_fitx, ploty

def quick_search(warped_image):
    binary_warped = warped_image
    global left_fit, right_fit
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()




def measuring_curvature():
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
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')

def pipeline(image):
    src = np.float32([[190, 720], [589, 457], [698, 457], [1145, 720]])
    dst = np.float32([[340, 720], [340, 0], [995, 0], [995, 720]])
    objpoints, imgpoints = get_obj_img_points()
    mtx, dist = get_from_pickle()

    #test_straight_lines1 = mpimg.imread('../test_images/straight_lines1.jpg')
    #test_ch1 = mpimg.imread('../test_images1/test_ch1.jpg')
    #test_ch5 = mpimg.imread('../test_images1/test_ch5.jpg')
    #test_ch6 = mpimg.imread('../test_images1/test_ch6.jpg')

    # gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(20, 100))
    # grady = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(20, 100))
    #
    # result, M = perspective(gradx, mtx, dist, src, dst)
    # result, M = pipeline(test_ch6, mtx, dist, src, dst,s_thresh=(170, 255), sx_thresh=(20, 100))
    # draw(test_ch6, result)
    # result, M = pipeline(test_ch5, mtx, dist, src, dst,s_thresh=(60, 200), sx_thresh=(20, 100))
    # draw(test_ch1, result)

    #result, M = pipeline(image, mtx, dist, src, dst, s_thresh=(60, 200), sx_thresh=(20, 100))
    processed_image = process_thresh(image, s_thresh=(60, 200), sx_thresh=(20, 100))
    result = imageUtils.perspective(processed_image, mtx, dist, src, dst)
    #left_fitx, right_fitx, lefty, righty = find_lines(result)
    #print(lefty.shape)

    left_fitx, right_fitx, ploty = blind_search(result)
    #quick_search(result)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image[:,:,-1]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, imageUtils.M1, imageUtils.img_size)
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    #plt.show()

    return result
    #origin_image = imageUtils.perspective(image, mtx, dist, src, dst)
    #f, (ax) = plt.subplots(1, 1, figsize=(16, 9))
    #ax.plot(left_fitx, ploty, color='yellow')
    #ax.plot(right_fitx, ploty, color='yellow')
    #ax.imshow(origin_image)
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.show()


    #
    #
    # mag_image = mag_thresh(image, sobel_kernel=3, mag_thresh=(20, 100))
    # result, M = perspective(mag_image, mtx, dist, src, dst)
    # draw(mag_image, result)
    #
    #
    # dir_image = dir_threshold(image, sobel_kernel=3, thresh=(0.6, 1.3))
    # result, M = perspective(dir_image, mtx, dist, src, dst)
    # draw(dir_image, result)
    #

    #
    #return find_lines(result)

#test_straight_lines1 = mpimg.imread('../test_images/straight_lines1.jpg')
#pipelined = pipeline(test_straight_lines1)

#test3 = mpimg.imread('../test_images/test6.jpg')
imageUtils = ImageUtils()
#pipelined = pipeline(test3)

#blind_search(test3)
#quick_search(test3)
#measuring_curvature()
#print(pipelined.nonzero())
#draw(test3, pipelined)

white_output = '../test_videos_output/test.mp4'
clip1 = VideoFileClip("../project_video.mp4")
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

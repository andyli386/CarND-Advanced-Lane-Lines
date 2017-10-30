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

def apply_thresh(processed_binary, thresh):
    binary = np.zeros_like(processed_binary)
    binary[(processed_binary >= thresh[0]) & (processed_binary <= thresh[1])] = 1

    return binary

def get_abs_soble(img, orient, sobel_kernel, toGray):
    if toGray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scale_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    return scale_sobel


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, toGray=True, thresh=(0, 255)):
    return apply_thresh(get_abs_soble(img, orient, sobel_kernel, toGray), thresh)


def get_mag(img, sobel_kernel, toGray):
    if toGray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)

    scale_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

    return scale_sobel

def mag_thresh(img, sobel_kernel=3, toGray=True, thresh=(0, 255)):
    return apply_thresh(get_mag(img, sobel_kernel, toGray), thresh)


def get_dir(img, sobel_kernel, toGray):
    if toGray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize= sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize= sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    direction = np.arctan2(abs_sobely, abs_sobelx)

    return direction

def dir_threshold(img, sobel_kernel=3, toGray=True, thresh=(0, np.pi/2)):
    return apply_thresh(get_dir(img, sobel_kernel, toGray), thresh)

def get_hls(img, channel):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    if channel == 'h':
        cha = hls[:,:,0]
    elif channel == 'l':
        cha = hls[:,:,1]
    elif channel == 's':
        cha = hls[:,:,2]

    return cha

def hls_threshold(img, channel='s', thresh=(0, 255)):
    return apply_thresh(get_hls(img, channel), thresh)


def get_ycbcr(img, channel):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float)

    if channel == 'y':
        cha = ycrcb[:, :, 0]
    elif channel == 'cr':
        cha = ycrcb[:, :, 1]
    elif channel == 'cb':
        cha = ycrcb[:, :, 2]

    return cha

def ycbcr_threshold(img, channel='y', thresh=(0, 255)):
    return apply_thresh(get_ycbcr(img, channel), thresh)

def get_luv(img, channel):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV).astype(np.float)
    if channel == 'l':
        cha = luv[:, :, 0]
    elif channel == 'u':
        cha = luv[:, :, 1]
    elif channel == 'v':
        cha = luv[:, :, 2]
    return cha

def luv_threshold(img, channel='l', thresh=(0, 255)):
    return apply_thresh(get_luv(img, channel), thresh)


def get_lab(img, channel):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float)
    if channel == 'l':
        cha = lab[:, :, 0]
    elif channel == 'a':
        cha = lab[:, :, 1]
    elif channel == 'b':
        cha = lab[:, :, 2]
    return cha

def lab_threshold(img, channel='l', thresh=(0, 255)):
    return apply_thresh(get_lab(img, channel), thresh)

def draw(image, processed_image):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    f.tight_layout()

    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image', fontsize=20)

    ax2.imshow(processed_image, cmap='gray')
    ax2.set_title('Processed Result', fontsize=20)
    #plt.subplots_adjust(left=0., right=1, top=2., bottom=0.)
    plt.show()


def reset_gamma(img, gamma=0.3):
    return exposure.adjust_gamma(img, gamma)


def perspective(img, mtx, dist, src, dst):
    global M
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    img_size = (img.shape[1], img.shape[0])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, img_size)
    return warped


def process_thresh(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)

    img = reset_gamma(img)

    l_channel = get_hls(img, 'l')
    sobelx_binary = abs_sobel_thresh(l_channel, 'x', sobel_kernel=3, toGray=False, thresh=sx_thresh )

    s_channel = get_hls(img, 's')
    s_binary = apply_thresh(s_channel, s_thresh)

    #color_binary = np.dstack((np.zeros_like(sobelx_binary), sobelx_binary, s_binary))

    color_binary = np.zeros_like(s_binary)
    color_binary[(s_binary > 0) | (sobelx_binary > 0)] = 1
    return color_binary

#def pipeline(img,  mtx, dist, src, dst, s_thresh, sx_thresh):
#    processed_image = process_thresh(img, s_thresh, sx_thresh)
#    return perspective(processed_image, mtx, dist, src, dst)

def find_lines(warped_image):
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

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    print(left_fitx.shape)
    tmp = left_fit[0] * 0 ** 2 + left_fit[1] * 0+ left_fit[2]
    print(tmp)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    #f, (ax) = plt.subplots(1, 1, figsize=(16, 9))
    #ax.plot(left_fitx, ploty, color='yellow')
    #ax.plot(right_fitx, ploty, color='yellow')
    #ax.xlim(0, 1280)
    #ax.ylim(720, 0)
    #ax.imshow(out_img)
    #plt.show()
    #return out_img

    #return left_fitx, right_fitx, lefty, righty

   # warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
   # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
    print(pts_left)
   # pts_right = np.array([np.transpose(np.vstack([right_fitx, righty]))])
   # pts = np.hstack((pts_left, pts_right))
   # cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)
   # cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
   # newwarp = cv2.warpPerspective(color_warp, M, (binary_warped.shape[1], binary_warped.shape[0]))
   # result = cv2.addWeighted((test3), 1, newwarp, 0.5, 0)

   # f, (ax) = plt.subplots(1, 1, figsize=(16, 9))
   # #ax.plot(left_fitx, ploty, color='yellow')
   # #ax.plot(right_fitx, ploty, color='yellow')
   # ax.imshow(result)
   # plt.xlim(0, 1280)
   # plt.ylim(720, 0)
   # plt.show()

    return left_fitx, right_fitx, ploty


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
    result = perspective(processed_image, mtx, dist, src, dst)
    #left_fitx, right_fitx, lefty, righty = find_lines(result)
    #print(lefty.shape)

    left_fitx, right_fitx, ploty = find_lines(result)

    origin_image = perspective(image, mtx, dist, src, dst)
    f, (ax) = plt.subplots(1, 1, figsize=(16, 9))
    ax.plot(left_fitx, ploty, color='yellow')
    ax.plot(right_fitx, ploty, color='yellow')
    ax.imshow(origin_image)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


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
test3 = mpimg.imread('../test_images/test4.jpg')
pipelined = pipeline(test3)
#print(pipelined.nonzero())
#draw(test3, pipelined)

#white_output = '../test_videos_output/test.mp4'
#clip1 = VideoFileClip("../project_video.mp4")
#white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
#white_clip.write_videofile(white_output, audio=False)

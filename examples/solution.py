# coding: utf-8
import numpy as np
import cv2

import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from skimage import data, exposure, img_as_float
#from moviepy.editor import VideoFileClip
from IPython.display import HTML

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

def get_abs_soble(img, orient='x', sobel_kernel=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scale_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    return scale_sobel


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    return apply_thresh(get_abs_soble(img, orient, sobel_kernel), thresh)


def get_mag(img, sobel_kernel=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)

    scale_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

    return scale_sobel

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    return apply_thresh(get_mag(img, sobel_kernel), mag_thresh())


def get_dir(img, sobel_kernel):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize= sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize= sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    direction = np.arctan2(abs_sobely, abs_sobelx)

    return direction

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    return apply_thresh(get_dir(img, sobel_kernel), thresh)

def get_hls(img, channel):
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == 's':
        cha = hls_img[:,:,2]
    elif channel == 'h':
        cha = hls_img[:,:,0]
    elif channel == 'l':
        cha = hls_img[:,:,1]

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
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    img_size = (img.shape[1], img.shape[0])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, img_size)
    return warped, M


def process_thresh(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    img = reset_gamma(img)

    l_channel = get_hls(img, 'l')
    s_channel = get_hls(img, 's')

    sobelx_binary = abs_sobel_thresh(l_channel, 'x', sobel_kernel=3, thresh=sx_thresh )
    s_binary = apply_thresh(s_channel, s_thresh)

    color_binary = np.dstack(( np.zeros_like(sobelx_binary), sobelx_binary, s_binary))
    return color_binary

def pipeline(img,  mtx, dist, src, dst, s_thresh, sx_thresh):
    processed_image = process_thresh(img, s_thresh, sx_thresh)
    return perspective(processed_image, mtx, dist, src, dst)

src = np.float32([[190, 720], [589, 457], [698, 457], [1145,720]])
dst = np.float32([[340, 720], [340, 0], [995, 0], [995, 720]])

objpoints, imgpoints = get_obj_img_points()

mtx,dist = get_from_pickle()

#image = mpimg.imread('../test_images/straight_lines1.jpg')

test_ch1 = mpimg.imread('../test_images1/test_ch1.jpg')
test_ch6 = mpimg.imread('../test_images1/test_ch6.jpg')

#gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(20, 100))
#grady = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(20, 100))
#
#result, M = perspective(gradx, mtx, dist, src, dst)
#result, M = pipeline(test_ch6, mtx, dist, src, dst,s_thresh=(170, 255), sx_thresh=(20, 100))
#draw(test_ch6, result)
result, M = pipeline(test_ch1, mtx, dist, src, dst,s_thresh=(170, 255), sx_thresh=(20, 100))
#draw(test_ch1, result)

#draw(test_ch1, get_luv(test_ch1, channel='l'))
#draw(test_ch1, luv_threshold(test_ch1, channel='u'))
#draw(test_ch1, luv_threshold(test_ch1, channel='v'))

draw(test_ch1, get_lab(test_ch1, channel='l'))
draw(test_ch1, get_lab(test_ch1, channel='a'))
draw(test_ch1, get_lab(test_ch1, channel='b'))

#result, M = perspective(grady, mtx, dist, src, dst)
#draw(grady, result)
#
#
#mag_image = mag_thresh(image, sobel_kernel=3, mag_thresh=(20, 100))
#result, M = perspective(mag_image, mtx, dist, src, dst)
#draw(mag_image, result)
#
#
#dir_image = dir_threshold(image, sobel_kernel=3, thresh=(0.6, 1.3))
#result, M = perspective(dir_image, mtx, dist, src, dst)
#draw(dir_image, result)
#
#
#s_binary = hls_select(image, channel='s', thresh=(90, 255))
#h_binary = hls_select(image, channel='h', thresh=(90, 255))
#l_binary = hls_select(image, channel='l', thresh=(90, 255))
#
#
#result, M = perspective(s_binary, mtx, dist, src, dst)
#draw(s_binary, result)
#
#
#result, M = perspective(h_binary, mtx, dist, src, dst)
#draw(h_binary, result)
#
#
#result, M = perspective(l_binary, mtx, dist, src, dst)
#draw(l_binary, result)
#
#
#img1 = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
#
#
#img1.shape
#
#
#YCrCb0 = img1[:,:,0]
#YCrCb1 = img1[:,:,1]
#YCrCb2 = img1[:,:,2]
#
#
#draw(YCrCb0, YCrCb1)
#
#
#draw(YCrCb0, YCrCb2)
#
#
#
#
#
#
#test_ch6_gamma = reset_gamma(test_ch6)
#
#
#test_ch6_ycrcb = cv2.cvtColor(test_ch6_gamma, cv2.COLOR_RGB2YCrCb)
#
#
#test_ch6_0 = test_ch6_ycrcb[:, :, 0]
#test_ch6_1 = test_ch6_ycrcb[:, :, 1]
#test_ch6_2 = test_ch6_ycrcb[:, :, 2]
#
#
#draw(test_ch6_0, test_ch6_1)
#
#
#draw(test_ch6_0, test_ch6_2)
#
#
#test_ch6_lab = cv2.cvtColor(test_ch6, cv2.COLOR_RGB2LAB)
#
#
#draw(test_ch6_gamma, test_ch6_lab)
#
#
#test_ch6_lab0 = test_ch6_lab[:, :, 0]
#test_ch6_lab1 = test_ch6_lab[:, :, 1]
#test_ch6_lab2 = test_ch6_lab[:, :, 2]
#
#
#draw(test_ch6_lab, test_ch6_lab0)
#
#
#draw(test_ch6_lab, test_ch6_lab1)
#
#
#
#
#draw(test_ch6_lab, test_ch6_lab2)
#
#
#
#
#white_output = 'test_videos_output/solidWhiteRight.mp4'
### To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
### To do so add .subclip(start_second,end_second) to the end of the line below
### Where start_second and end_second are integer values representing the start and end of the subclip
### You may also uncomment the following line for a subclip of the first 5 seconds
###clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
##white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
#
##get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')
#
#
#
#
#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format(white_output))
#
#
#
#
#def draw_rect(warped_image):
#    binary_warped = warped_image[:,:,0]
#    # Assuming you have created a warped binary image called "binary_warped"
#    # Take a histogram of the bottom half of the image
#    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
#    # Create an output image to draw on and  visualize the result
#    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
#    # Find the peak of the left and right halves of the histogram
#    # These will be the starting point for the left and right lines
#    midpoint = np.int(histogram.shape[0]/2)
#    leftx_base = np.argmax(histogram[:midpoint])
#    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
#
#    # Choose the number of sliding windows
#    nwindows = 9
#    # Set height of windows
#    window_height = np.int(binary_warped.shape[0]/nwindows)
#    # Identify the x and y positions of all nonzero pixels in the image
#    nonzero = binary_warped.nonzero()
#    nonzeroy = np.array(nonzero[0])
#    nonzerox = np.array(nonzero[1])
#    # Current positions to be updated for each window
#    leftx_current = leftx_base
#    rightx_current = rightx_base
#    # Set the width of the windows +/- margin
#    margin = 100
#    # Set minimum number of pixels found to recenter window
#    minpix = 50
#    # Create empty lists to receive left and right lane pixel indices
#    left_lane_inds = []
#    right_lane_inds = []
#
#    # Step through the windows one by one
#    for window in range(nwindows):
#        # Identify window boundaries in x and y (and right and left)
#        win_y_low = binary_warped.shape[0] - (window+1)*window_height
#        win_y_high = binary_warped.shape[0] - window*window_height
#        win_xleft_low = leftx_current - margin
#        win_xleft_high = leftx_current + margin
#        win_xright_low = rightx_current - margin
#        win_xright_high = rightx_current + margin
#        # Draw the windows on the visualization image
#        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
#        (0,255,0), 2)
#        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
#        (0,255,0), 2)
#        # Identify the nonzero pixels in x and y within the window
#        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
#        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
#        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
#        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
#        # Append these indices to the lists
#        left_lane_inds.append(good_left_inds)
#        right_lane_inds.append(good_right_inds)
#        # If you found > minpix pixels, recenter next window on their mean position
#        if len(good_left_inds) > minpix:
#            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
#        if len(good_right_inds) > minpix:
#            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
#
#    # Concatenate the arrays of indices
#    left_lane_inds = np.concatenate(left_lane_inds)
#    right_lane_inds = np.concatenate(right_lane_inds)
#
#    # Extract left and right line pixel positions
#    leftx = nonzerox[left_lane_inds]
#    lefty = nonzeroy[left_lane_inds]
#    rightx = nonzerox[right_lane_inds]
#    righty = nonzeroy[right_lane_inds]
#
#    # Fit a second order polynomial to each
#    left_fit = np.polyfit(lefty, leftx, 2)
#    right_fit = np.polyfit(righty, rightx, 2)
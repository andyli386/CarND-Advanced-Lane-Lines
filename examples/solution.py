# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from ImageUtils import ImageUtils
from Calibration import Calibration
from Detector import Detector




def pipeline(image):
    detector = Detector()
    src = np.float32([[190, 720], [589, 457], [698, 457], [1145, 720]])
    dst = np.float32([[340, 720], [340, 0], [995, 0], [995, 720]])
    calibration =  Calibration()
    mtx, dist = calibration.get_from_pickle()

    #result, M = pipeline(image, mtx, dist, src, dst, s_thresh=(60, 200), sx_thresh=(20, 100))

    #processed_image = process_thresh(image, s_thresh=(60, 200), sx_thresh=(20, 100))
    processed_image = imageUtils.luv_lab_filter(image)
    result = imageUtils.perspective(processed_image, mtx, dist, src, dst)

    detector.set_binary_image(result)
    detector.blind_search()
    left_fitx, right_fitx, ploty = detector.get_fit()
    result = imageUtils.draw_on_origin_image(image, imageUtils, left_fitx, right_fitx, ploty, False)


    #detector.set_binary_image(result)
    #detector.quick_search()
    #left_fitx, right_fitx, ploty = detector.get_fit()
    #result = imageUtils.drawOnNormalPic(image, imageUtils, left_fitx, right_fitx, ploty, True)



    return result

#test_straight_detector1 = mpimg.imread('../test_images/straight_detector1.jpg')
#pipelined = pipeline(test_straight_detector1)

#test_straight_detector1 = mpimg.imread('../test_images/straight_detector1.jpg')
#test_ch1 = mpimg.imread('../test_images1/test_ch1.jpg')
#test_ch5 = mpimg.imread('../test_images1/test_ch5.jpg')
#test_ch6 = mpimg.imread('../test_images1/test_ch6.jpg')

#test3 = mpimg.imread('../test_images/test3.jpg')
imageUtils = ImageUtils()
#pipelined = pipeline(test3)

#blind_search(test3)
#quick_search(test3)
#measuring_curvature()
#print(pipelined.nonzero())
#draw(test3, pipelined)

#white_output = '../test_videos_output/project_video_output_full2.mp4'
#clip1 = VideoFileClip("../project_video.mp4")

white_output = '../test_videos_output/challenge_video_output.mp4'
clip1 = VideoFileClip("../challenge_video.mp4")
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

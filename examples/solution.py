# coding: utf-8

from moviepy.editor import VideoFileClip
from Detector import Detector
from ImageUtils import ImageUtils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def pipeline(image):
    detector = Detector()
    return detector.detect(image, debug)

#test_straight_detector1 = mpimg.imread('../test_images/straight_detector1.jpg')
#pipelined = pipeline(test_straight_detector1)

#test_straight_detector1 = mpimg.imread('../test_images/straight_detector1.jpg')
#test_ch1 = mpimg.imread('../test_images1/test_ch1.jpg')
#test_ch5 = mpimg.imread('../test_images1/test_ch5.jpg')
#test_ch6 = mpimg.imread('../test_images1/test_ch6.jpg')

#test1 = mpimg.imread('../test_images/test1.jpg')
#test2 = mpimg.imread('../test_images/test2.jpg')
#test3 = mpimg.imread('../test_images/test3.jpg')
#test4 = mpimg.imread('../test_images/test4.jpg')
#test5 = mpimg.imread('../test_images/test5.jpg')
#test6 = mpimg.imread('../test_images/test6.jpg')
#pipelined = pipeline(test1)
#ImageUtils().draw(test1, pipelined)
#pipelined = pipeline(test2)
#ImageUtils().draw(test2, pipelined)
#pipelined = pipeline(test3)
#ImageUtils().draw(test3, pipelined)
#pipelined = pipeline(test4)
#ImageUtils().draw(test4, pipelined)
#pipelined = pipeline(test5)
#ImageUtils().draw(test5, pipelined)
#pipelined = pipeline(test6)
#ImageUtils().draw(test6, pipelined)



debug = False
#test0 = mpimg.imread('../test_images1/test_ch0.jpg')
#test1 = mpimg.imread('../test_images1/test_ch1.jpg')
#test2 = mpimg.imread('../test_images1/test_ch2.jpg')
#test3 = mpimg.imread('../test_images1/test_ch3.jpg')
#test4 = mpimg.imread('../test_images1/test_ch4.jpg')
#test5 = mpimg.imread('../test_images1/test_ch5.jpg')
#test6 = mpimg.imread('../test_images1/test_ch6.jpg')
#pipelined = pipeline(test0)
##ImageUtils().draw(test0, pipelined)
#
#pipelined = pipeline(test1)
##ImageUtils().draw(test1, pipelined)
##pipelined = pipeline(test2)
##ImageUtils().draw(test2, pipelined)
##pipelined = pipeline(test3)
##ImageUtils().draw(test3, pipelined)
##pipelined = pipeline(test4)
##ImageUtils().draw(test4, pipelined)
#pipelined = pipeline(test5)
##ImageUtils().draw(test5, pipelined)
#pipelined = pipeline(test6)
#ImageUtils().draw(test6, pipelined)

#blind_search(test3)
#quick_search(test3)
#measuring_curvature()
#print(pipelined.nonzero())
white_output = '../test_videos_output/project_video_output_full3.mp4'
clip = VideoFileClip("../project_video.mp4")

#white_output = '../test_videos_output/challenge_video_output.mp4'
#clip = VideoFileClip("../challenge_video.mp4")
white_clip = clip.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

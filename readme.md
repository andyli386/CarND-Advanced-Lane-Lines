## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image2_un]: ./test_images/test_and_undist.png "Road Transformed and undistort"
[image2_bin]: ./test_images/test_and_bin.png "Road Transformed and binary"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./test_images/challenge_video_output.mp4.png "Output"
[video1]: ./output_images/project_video_output_full3.mp4 "Project Video output"
[video2]: ./output_images/challenge_video_output.mp4 "Challenge Video output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code located in "./examples/Calibration.py". 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
Since I have saved the calibration parameters in "camera_cal/wide_dist_pickle.p", it's easy to load them with the `get_from_pickle` function in Calibration.py. I calibrate the image with `cv2.undistort` function.
![alt text][image2_un]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of `l`channel in luv color space with the threshhold `(215, 255)` and `b` channel in lab color space with the threshhold `(145, 200)`  to generate a binary image (thresholding steps at line 139# in `./examples/ImageUtils.py`).  Here's an example of my output for this step.

![alt text][image2_bin]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective()`, which appears in lines 167 through 178 in the file `ImageUtils.py`. The `perspective()` function takes as inputs an image (`image`), and the source (`src`) and destination (`dst`) points are saved in the class. I chose the hardcode the source and destination points in the following manner:

```python
self.__src = np.float32([[190, 720], [589, 457], [698, 457], [1145, 720]])
self.__dst = np.float32([[340, 720], [340, 0], [995, 0], [995, 720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 340, 720      | 
| 589, 457      | 340, 0        |
| 698, 457      | 995, 0        |
| 1145, 720     | 995, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:First I did blind search. I split 8 windows in y-axis, and sum all the elements in y-axis, then find the the max value in one window as base. Then I used this base point to calculate the area bounds. After getting the bounds, I choose all the nonzeros points in this restricted area as alternative points for fitting. Then I iterated this steps in one frame. If I found any nonzeros point in the restricted areas of x-axis, it means I found a lane in this frame and I would do quick search base on this frame in next frame. The `blind_search()` is in the lines 78 through 123 in `Detector.py`

In the quick search I did some stuff like this: since I have found a lane in the blind search I would fit a 2nd order polynomial. With this polynomial, I can calculate the base point in quick search and then the same way to find nonzero points in a restrcited area with 8 iterations. If I found nothing in this frame, I would do the blind search in the next frame. The `quick_search()` is in the lines 50 through 75 in `Detector.py`

And this is how I fit to get a 2nd order polynomial: I did np.polyfit with the found nonzerors point. Then I calculate the bottom x value and the top x value. And append them in a queue with the length 15 separately. I would get the median of the top and bottom points and append them to the found points. Then I did a new polyfit with these points and got the parameters of the polynomial and also append them in a queue and get the median of them as the final fit parameters.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 165 through 206 in my code in `Detector.py`. In the `curvature()` function I calculate the meters per pixel in x and y axis. Then get the fit x and y evaluation, then fit a new polynomials in real world space.  Last, calculate the curved by the given formula. In the `car_pos()` function, I calculate the left and right evaluation, then calculate the meters per pixel in x and y axis and mean the left evaluation and right evaluation in x axis, by that I get the offset.  

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 19 through 48 in my code in `Detector.py` in the function `detect()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1], this is the project output video.
Here's a [link to my video result][video2], this is the challenge output video.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
I have 4 classes in this project, Calibration, Detector, ImageUtils and Line. The Calibration class is for something calibrate. The ImageUtils stores the image processing functions. The Detector stores the functions that related to the basic algorithms of finding lanes and fitting. The Line is the basic struct of a lane. The luv and lab color spaces are very important for filtering the yellow and white . The quick and blind search is very effecient and acurrate. The fit method I used is also very robust. Without them I can't achive the goal.

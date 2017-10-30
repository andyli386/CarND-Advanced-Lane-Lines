from unittest import TestCase
from ImageUtils import ImageUtils
import matplotlib.image as mpimg


imageUtils = ImageUtils()
test6 = mpimg.imread('../test_images/test6.jpg')
processed = imageUtils.get_abs_sobel(test6, 'x', sobel_kernel=3, toGray=False)
processed1 = imageUtils.apply_thresh(processed, (20, 100))
imageUtils.draw(test6, processed1)
from unittest import TestCase
from .ImageUtils import ImageUtils
import matplotlib.image as mpimg


class TestImageUtils(TestCase):
    test6 = mpimg.imread('../test_images/test6.jpg')
    imageUtils = ImageUtils()

    def test_get_abs_sobel(self):
        processed = self.imageUtils.get_abs_sobel(self.test6, 'x', sobel_kernel=3, toGray=False)
        processed1 = self.imageUtils.apply_thresh(processed, thresh=(20, 100))
        self.imageUtils.draw(self.test6, processed1)

'''

    def test_abs_sobel_thresh(self):
        processed = self.imageUtils.abs_sobel_thresh(self.test6, 'x', sobel_kernel=3, toGray=False, thresh=(20, 100))

        self.imageUtils.draw(self.test6, processed)
        #self.fail()


    def test_mag_thresh(self):
        self.imageUtils.mag_thresh(self.test6)
        self.fail()


    def test_dir_threshold(self):
        self.fail()

    def test_get_hls(self):
        self.fail()

    def test_hls_threshold(self):
        self.fail()

    def test_get_ycbcr(self):
        self.fail()

    def test_ycbcr_threshold(self):
        self.fail()

    def test_get_luv(self):
        self.fail()

    def test_luv_threshold(self):
        self.fail()

    def test_get_lab(self):
        self.fail()

    def test_lab_threshold(self):
        self.fail()

    def test_reset_gamma(self):
        self.fail()

    def test_perspective(self):
        self.fail()
'''

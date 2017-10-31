from unittest import TestCase
from .ImageUtils import ImageUtils
import matplotlib.image as mpimg


class TestImageUtils(TestCase):
    test6 = mpimg.imread('../test_images/test6.jpg')
    imageUtils = ImageUtils()

    def test_get_abs_sobel(self):
        processed = self.imageUtils.get_abs_sobel(self.test6, 'x', sobel_kernel=3, toGray=True)
        #self.imageUtils.draw(self.test6, processed)


    def test_abs_sobel_thresh(self):
        processed = self.imageUtils.abs_sobel_thresh(self.test6, 'x', sobel_kernel=3, toGray=True, thresh=(20, 100))
        #self.imageUtils.draw(self.test6, processed)
        #self.fail()

    def test_mag_thresh(self):
        processed = self.imageUtils.mag_thresh(self.test6, sobel_kernel=3, toGray=True, thresh=(20, 100))
        #self.imageUtils.draw(self.test6, processed)
        #self.fail()

    def test_dir_threshold(self):
        processed = self.imageUtils.dir_threshold(self.test6, sobel_kernel=3, toGray=True, thresh=(0.6, 1.3))
        #self.imageUtils.draw(self.test6, processed)
        #self.fail()

    def test_get_hls(self):
        pass
        #processed = self.imageUtils.get_hls(self.test6, channel='h')
        #self.imageUtils.draw(self.test6, processed)
        #processed = self.imageUtils.get_hls(self.test6, channel='l')
        #self.imageUtils.draw(self.test6, processed)
        #processed = self.imageUtils.get_hls(self.test6, channel='s')
        #self.imageUtils.draw(self.test6, processed)
        #self.fail()

    def test_hls_threshold(self):
        pass
        #self.fail()

    def test_get_ycbcr(self):
        pass
        #processed = self.imageUtils.get_ycbcr(self.test6, channel='y')
        #self.imageUtils.draw(self.test6, processed)
        #processed = self.imageUtils.get_ycbcr(self.test6, channel='cb')
        #self.imageUtils.draw(self.test6, processed)
        #processed = self.imageUtils.get_ycbcr(self.test6, channel='cr')
        #self.imageUtils.draw(self.test6, processed)
        #self.fail()
    def test_ycbcr_threshold(self):
        pass
        #self.fail()

    def test_get_luv(self):
        pass
        #processed = self.imageUtils.get_luv(self.test6, channel='l')
        #self.imageUtils.draw(self.test6, processed)
        #processed = self.imageUtils.get_luv(self.test6, channel='u')
        #self.imageUtils.draw(self.test6, processed)
        #processed = self.imageUtils.get_luv(self.test6, channel='v')
        #self.imageUtils.draw(self.test6, processed)

        #self.fail()
    def test_luv_threshold(self):
        pass
        #self.fail()

    def test_get_lab(self):
        pass
        #processed = self.imageUtils.get_lab(self.test6, channel='l')
        #self.imageUtils.draw(self.test6, processed)
        #processed = self.imageUtils.get_lab(self.test6, channel='a')
        #self.imageUtils.draw(self.test6, processed)
        #processed = self.imageUtils.get_lab(self.test6, channel='b')
        #self.imageUtils.draw(self.test6, processed)
        #self.fail()

    def test_lab_threshold(self):
        pass
        #self.fail()

    def test_reset_gamma(self):
        pass
        #processed = self.imageUtils.reset_gamma(self.test6)
        #self.imageUtils.draw(self.test6, processed)
        #self.fail()

    def test_perspective(self):
        pass
        #self.fail()

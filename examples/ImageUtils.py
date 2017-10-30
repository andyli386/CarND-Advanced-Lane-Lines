import numpy as np
import cv2
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt


class ImageUtils(object):
    def __init__(self):
        pass


    def apply_thresh(self, processed_binary, thresh):
        binary = np.zeros_like(processed_binary)
        #self.draw(processed_binary, binary)
        binary[(processed_binary >= thresh[0]) & (processed_binary <= thresh[1])] = 1
        #print(binary)

        return binary

    def get_abs_sobel(self, image, orient, sobel_kernel, toGray):
        if toGray:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scale_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        return scale_sobel

    def abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, toGray=True, thresh=(0, 255)):
        return self.apply_thresh(self.get_abs_sobel(image, orient, sobel_kernel, toGray), thresh)

    def get_mag(self, image, sobel_kernel, toGray):
        if toGray:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)

        scale_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

        return scale_sobel

    def mag_thresh(self, image, sobel_kernel=3, toGray=True, thresh=(0, 255)):
        return self.apply_thresh(self.get_mag(image, sobel_kernel, toGray), thresh)

    def get_dir(self, image, sobel_kernel, toGray):
        if toGray:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        direction = np.arctan2(abs_sobely, abs_sobelx)

        return direction

    def dir_threshold(self, image, sobel_kernel=3, toGray=True, thresh=(0, np.pi / 2)):
        return self.apply_thresh(self.get_dir(image, sobel_kernel, toGray), thresh)

    def get_hls(self, image, channel):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
        if channel == 'h':
            cha = hls[:, :, 0]
        elif channel == 'l':
            cha = hls[:, :, 1]
        elif channel == 's':
            cha = hls[:, :, 2]

        return cha

    def hls_threshold(self, image, channel='s', thresh=(0, 255)):
        return self.apply_thresh(self.get_hls(image, channel), thresh)

    def get_ycbcr(self, image, channel):
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb).astype(np.float)

        if channel == 'y':
            cha = ycrcb[:, :, 0]
        elif channel == 'cr':
            cha = ycrcb[:, :, 1]
        elif channel == 'cb':
            cha = ycrcb[:, :, 2]

        return cha

    def ycbcr_threshold(self, image, channel='y', thresh=(0, 255)):
        return self.apply_thresh(self.get_ycbcr(image, channel), thresh)

    def get_luv(self, image, channel):
        luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV).astype(np.float)
        if channel == 'l':
            cha = luv[:, :, 0]
        elif channel == 'u':
            cha = luv[:, :, 1]
        elif channel == 'v':
            cha = luv[:, :, 2]
        return cha

    def luv_threshold(self, image, channel='l', thresh=(0, 255)):
        return self.apply_thresh(self.get_luv(image, channel), thresh)

    def get_lab(self, image, channel):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float)
        if channel == 'l':
            cha = lab[:, :, 0]
        elif channel == 'a':
            cha = lab[:, :, 1]
        elif channel == 'b':
            cha = lab[:, :, 2]
        return cha

    def lab_threshold(self, image, channel='l', thresh=(0, 255)):
        return self.apply_thresh(self.get_lab(image, channel), thresh)


    def reset_gamma(self, image, gamma=0.3):
        return exposure.adjust_gamma(image, gamma)

    def perspective(self, image, mtx, dist, src, dst):
        self.undist = cv2.undistort(image, mtx, dist, None, mtx)

        self.img_size = (image.shape[1], image.shape[0])

        self.M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(self.undist, self.M, self.img_size)
        return warped

    def draw(self, image, processed_image):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
        f.tight_layout()
        print(processed_image)

        ax1.imshow(image, cmap='gray')
        ax1.set_title('Original Image', fontsize=20)

        ax2.imshow(processed_image, cmap='gray')
        ax2.set_title('Processed Result', fontsize=20)
        # plt.subplots_adjust(left=0., right=1, top=2., bottom=0.)
        plt.show()
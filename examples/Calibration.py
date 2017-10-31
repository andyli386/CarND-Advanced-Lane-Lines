import numpy as np
import cv2, glob, pickle


class Calibration(object):
    def __init__(self, regeneratePickle=False):
        if regeneratePickle:
            objpoints, imgpoints = self.get_obj_img_points()
            self.save_to_pickle(objpoints, imgpoints, '../camera_cal/calibration3.jpg')

    def get_obj_img_points(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('../camera_cal/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        return objpoints, imgpoints

    def save_to_pickle(self, objpoints, imgpoints, imageName, pickleName="../camera_cal/wide_dist_pickle.p"):
        img = cv2.imread(imageName)
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open(pickleName, "wb"))

    def get_from_pickle(self, pickleName="../camera_cal/wide_dist_pickle.p"):
        dist_pickle = pickle.load(open(pickleName, "rb"))
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
        return mtx, dist
import cv2
import numpy as np
import glob
import os
import math
"""
Script written for CV-Aid Design project at McGill
@author jamestang
@version 1.1

This script undistort images

"""
# "k1": "0.2199999988079071",
# "k2": "-0.03999999910593033",
# "k3": "-0.05000000074505806",
# "k4": "0.014999999664723873",
# D = np.array([[0.2199999988079071],[-0.03999999910593033],[-0.05000000074505806],[0.014999999664723873]])
# focal_length = 8

# class util:
#     def undistort(self,img):
#         h,w = img.shape[:2]
#         print(img.shape[:2])
#         # calculate the focal length fx,fy using estimation
#         # assumption: CMOS sensor specification similar to: http://www.superpix.com.cn/cn/xiazai/SP5508.pdf
#         # assumption: focal length is 8mm - 10mm 
#         # fx = Fx / W * w fy = Fy / H * h
#         # W: is the sensor width expressed in world units, let's say mm
#         # w: is the image width expressed in pixel
#         # fx: is the focal length expressed in pixel units (as is in the camera matrix )
#         H = 1944 * 1.12 / 100
#         h = 1944
#         W = 2592 * 1.12 / 100
#         w = 2592
#         fx = 8 / W * w
#         fy = 8 / H * h
#         # print("Fx: {} Fy:{}".format(fx,fy))
#         h,w = img.shape[:2]
#         cx = w / 2
#         cy = h / 2
#         K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
#         print(K)
#         DIM = (w,h)
#         print(DIM)
#         map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
#         undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#         return undistorted_img

class util:
    def __init__(self):
        w = 1080
        h = 1920
        DIM = (w,h)
        cx = w / 2
        cy = h / 2
        # "k1": "0.2199999988079071",
        # "k2": "-0.03999999910593033",
        # "k3": "-0.05000000074505806",
        # "k4": "0.014999999664723873",
        D = np.array([[0.2199999988079071],[-0.03999999910593033],[-0.05000000074505806],[0.014999999664723873]])
        focal_length = 6.5 # focal length in mm
        # calculate the focal length fx,fy using estimation
        # assumption: CMOS sensor specification similar to: http://www.superpix.com.cn/cn/xiazai/SP5508.pdf
        # assumption: focal length is 8mm - 10mm 
        # fx = Fx / W * w fy = Fy / H * h
        # W: is the sensor width expressed in world units, let's say mm
        # w: is the image width expressed in pixel
        # fx: is the focal length expressed in pixel units (as is in the camera matrix )
        H = 1944 * 1.12 / 100
        h = 1944
        W = 2592 * 1.12 / 100
        w = 2592
        fx = focal_length / W * w
        fy = focal_length / H * h

        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        print("===================================")
        print("Initialized initUndistortRectifyMap")
        print("Camera matrix K: ")
        print(K)
        print("Distortion coeffecients D: ")
        print(D)
        print("Focal length: {}".format(focal_length))
        print("===================================")

        # rotation matrix
        rvec_left = np.array([ 0.8176273703575134, 1.429028034210205 ,-1.6050984859466553 ])
        rvec_lefteye = np.array([ 2.4510138034820557, 1.5082813501358032 , 1.6038360595703125 ])
        rvec_righteye = np.array([ 3.8327078819274902, 1.4986587762832642 ,-4.765188217163086 ])
        rvec_right = np.array( [ 5.4923787117004395, 1.5711150169372559 ,-7.820102691650391 ])

        # # calcuate euler angle of rotation
        # self.rot_angle_left = self.rotationMatrixToEulerAngles(rvec_left)
        # self.rot_angle_lefteye = self.rotationMatrixToEulerAngles(rvec_lefteye)
        # self.rot_angle_righteye = self.rotationMatrixToEulerAngles(rvec_righteye)
        # self.rot_angle_right = self.rotationMatrixToEulerAngles(rvec_right)
                # R, jacobian = cv2.Rodrigues(R)
        self.rot_angle_left, jacobian = cv2.Rodrigues(rvec_left)
        self.rot_angle_lefteye, jacobian = cv2.Rodrigues(rvec_lefteye)
        self.rot_angle_righteye, jacobian = cv2.Rodrigues(rvec_righteye)
        self.rot_angle_right, jacobian = cv2.Rodrigues(rvec_right)

        # the inversion of rotation matrix is its transposition
        self.rot_angle_left = np.transpose(self.rot_angle_left)
        self.rot_angle_lefteye = np.transpose(self.rot_angle_lefteye)
        self.rot_angle_righteye = np.transpose(self.rot_angle_righteye)
        self.rot_angle_right = np.transpose(self.rot_angle_right)

    def undistort(self,img):
        undistorted_img = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img

    def get_rotation_matrix(self):
        return self.rot_angle_left, self.rot_angle_lefteye, self.rot_angle_righteye, self.rot_angle_right
    
    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    # source: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self,R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self,R) :
        R, jacobian = cv2.Rodrigues(R)
        print(R)
        assert(self.isRotationMatrix(R))
         
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
         
        singular = sy < 1e-6
     
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
     
        return np.array([x, y, z])

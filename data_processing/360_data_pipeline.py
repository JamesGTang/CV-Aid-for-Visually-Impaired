import shutil
import fnmatch
import os
import numpy as np
import cv2
import imutils
import subprocess
import subprocess
from label_cam_img import IMG_LABELLER
from undistort import util
"""
Script written for CV-Aid Design project at McGill
@author jamestang
@version 1.1
This script preprocess the images obtained from 360 glass, and apply synchronize stiching
Modifies only the parameter ORBI_DATA_ROOT for orginal dataset.
and PROCESSED_ORBI_ROOT for where to store the processed images. 

"""

# modify these two parameters according to folder structure
ORBI_DATA_ROOT = '../../CV_VI_original_data/201901-201904/'
# ORBI_DATA_ROOT = '../orbi_sample/'
# PROCESSED_ORBI_ROOT = '../orbi_sample_processed/'
PROCESSED_ORBI_ROOT = './labelled_360/'
# PROCESSED_ORBI_ROOT = '../orbi_360_processed/'
# PROCESSED_ORBI_ROOT = '../orbi_to_cam_processed/'
# modify this file to specify the size of the final image
ADJ_SIZE = (426,240)
ADJ_SIZE2 = (1920,1080)
ADJ_SIZE3 = (600,600)


# git add .; git commit -m "more data"; git push origin master
class DATA_PIPELINE():
    def __init__(self):
        # determine if we are using OpenCV v3.X
        # self.isv3 = imutils.is_cv3(or_better=True)    
        pass
    # rotate image and replace original image
    def rotate_img(self,img,rot_deg):
        # rotate ccw
        out=cv2.transpose(img)
        out=cv2.flip(out,flipCode=rot_deg)
        return out
    
    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
        # detect and extract features from the image
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return kps, features
    
    def bf_match(self,keypoints1,keypoints2):
        """
        Match all keypoints of the reference image to the transformed images using
        a brute-force method.
        """
        bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf_matcher.match(keypoints1,keypoints2)
        return matches

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
        ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

                # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None
    
    def stitch(self, images, ratio=0.75, reprojThresh=2.0):
        # self.cachedH = None
        # unpack the images
        imageB = images[0]
        imageA = images[1]
        # if the cached homography matrix is None, then we need to
        # apply keypoint matching to construct it
        # if self.cachedH is None:
        #     # detect keypoints and extract
        #     kpsA, featuresA = self.detectAndDescribe(imageA)
        #     kpsB, featuresB = self.detectAndDescribe(imageB)
 
        #     # match features between the two images
        #     M = self.matchKeypoints(kpsA, kpsB,
        #         featuresA, featuresB, ratio, reprojThresh)
 
        #     # if the match is None, then there aren't enough matched
        #     # keypoints to create a panorama
        #     if M is None:
        #         return None
 
        #     # cache the homography matrix
        #     self.cachedH = M[1]
        kpsA, featuresA = self.detectAndDescribe(imageA)
        kpsB, featuresB = self.detectAndDescribe(imageB)
        matches = self.bf_match(featuresA,featuresB)
        matches = sorted(matches, key = lambda x:x.distance)[:10]
        #-- Localize the object
        occ_pts = np.empty((len(matches),2), dtype=np.float32)
        crop_pts = np.empty((len(matches),2), dtype=np.float32)
        for i in range(len(matches)):
            occ_pts[i,:] = kpsA[matches[i].queryIdx].pt
            crop_pts[i,: ] = kpsB[matches[i].trainIdx].pt
        M, mask = cv2.findHomography(crop_pts,occ_pts, cv2.RANSAC)

        # apply a perspective transform to stitch the images together
        # using the cached homography matrix
        result = cv2.warpPerspective(imageA, M,
            (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        # return the stitched image
        return result

    def process_all_undistort_stitch(self):
        undistort_util = util()
        # remove cam directory and recreate
        if not os.path.exists(PROCESSED_ORBI_ROOT):
            os.makedirs(PROCESSED_ORBI_ROOT)
        else:  
            print(PROCESSED_ORBI_ROOT+' exists, will be removed')
            shutil.rmtree(PROCESSED_ORBI_ROOT, ignore_errors=True)
            os.makedirs(PROCESSED_ORBI_ROOT)

        matches = []
        folder_idx = 1
        file_idx = 0
        for root, dirnames, filenames in os.walk(ORBI_DATA_ROOT):
            # for each folder find videos of 4 angles
            all_files_in_dir = []
            fp = ""
            dest_dir = ""
            matches = fnmatch.filter(filenames, '*.MP4')
            for filename in matches:
                all_files_in_dir.append(os.path.join(root, filename))
            if matches != []:
                fp = os.path.join(root, filename)
                print("Processing directory: "+root+" | destination directory: "+PROCESSED_ORBI_ROOT)

            # if length of files in each folder are less than 4 then the folder doesnt have footage of all angles
            if len(all_files_in_dir) < 4:
                # if not root folder
                if folder_idx != 1:
                    print("Folder: "+root+" is corrputed, is skipped")
            else:
                # make a directory in destination folder
                # print("Files to be processed"+all_files_in_dir)
                # leftVid = cv2.VideoCapture(all_files_in_dir[3])
                print(all_files_in_dir)
                leftEyeVid = cv2.VideoCapture(all_files_in_dir[2])
                # rightVid = cv2.VideoCapture(all_files_in_dir[0])
                rightEyeVid = cv2.VideoCapture(all_files_in_dir[0])
                # since all videos have same framerate, only sample one video for FPS
                fps = leftEyeVid.get(cv2.CAP_PROP_FPS)
                framecount = 0
                
                # count the frames for capture
                while(True):
                    # Capture frame-by-frame
                    # success_left, leftIMG = leftVid.read()
                    success_leftEye, leftEyeIMG = leftEyeVid.read()
                    # success_right, rightIMG = rightVid.read()
                    success_rightEye, rightEyeIMG = rightEyeVid.read()

                    framecount += 1

                    # Check if this is the frame closest to 1 seconds
                    if success_leftEye==True:
                        if framecount == (fps * 0.2) and success_rightEye==True:  #and success_rightEye==True and success_right==True:
                            #resize all images
                            # leftIMG = cv2.resize(leftIMG,ADJ_SIZE2)
                            leftEyeIMG = cv2.resize(leftEyeIMG,ADJ_SIZE2)
                            # rightIMG = cv2.resize(rightIMG,ADJ_SIZE2)
                            rightEyeIMG = cv2.resize(rightEyeIMG,ADJ_SIZE2)                  
                            #rotate images
                            # leftIMG = self.rotate_img(leftIMG,0) #left image rotate 90 deg CCW
                            leftEyeIMG = self.rotate_img(leftEyeIMG,+1) #left eye image rotate 90 deg CCW
                            rightEyeIMG = self.rotate_img(rightEyeIMG,0) #right eye image rotate 90 deg CCW
                            
                            # undistort
                            leftEyeIMG = undistort_util.undistort(leftEyeIMG)                            
                            rightEyeIMG = undistort_util.undistort(rightEyeIMG)
                            cv2.imshow("l", leftEyeIMG)
                            cv2.imshow("r",rightEyeIMG)
                            # rightIMG = self.rotate_img(rightIMG,0) #right image rotate 90 deg CW
                            framecount = 0
                            left_fp = PROCESSED_ORBI_ROOT+str(file_idx)+"1"+".jpg"
                            right_fp = PROCESSED_ORBI_ROOT+str(file_idx)+"0"+".jpg"
                            # cv2.imwrite(PROCESSED_ORBI_ROOT+str(file_idx)+"0"+".jpg",leftIMG)
                            cv2.imwrite(left_fp,leftEyeIMG)
                            # # cv2.imwrite(PROCESSED_ORBI_ROOT+str(file_idx)+"2"+".jpg",rightIMG)
                            cv2.imwrite(right_fp,rightEyeIMG)
                            
                            cmd = "./image-stitching " +  left_fp + " " + right_fp
                            res = subprocess.call(cmd,shell=True)
                            # print("---------------------------------")
                            # print(res)
                            if res == 0:
                                move_to_fp = PROCESSED_ORBI_ROOT + str(file_idx) + ".jpg"
                                shutil.move("out.jpg", move_to_fp)
                                img = cv2.imread(move_to_fp, cv2.IMREAD_UNCHANGED)
                                img= cv2.resize(img,ADJ_SIZE3)
                                cv2.imwrite(move_to_fp,img)

                            else:
                                print("error when processing: " + left_fp + " , " + right_fp)
                            

                            os.remove(left_fp)
                            os.remove(right_fp)
                            
                            file_idx += 1 
                            # # Check end of video
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                  break
                    else:
                        break
                folder_idx += 1
        cv2.destroyAllWindows()        
        print("Finished proprocessing images, total images: ",file_idx)

    # process all images and undistort without stiching 
    def process_all_undistort(self):
        undistort_util = util()
        # remove cam directory and recreate
        if not os.path.exists(PROCESSED_ORBI_ROOT):
            os.makedirs(PROCESSED_ORBI_ROOT)
        # else:  
        #     print(PROCESSED_ORBI_ROOT+' exists, will be removed')
        #     shutil.rmtree(PROCESSED_ORBI_ROOT, ignore_errors=True)
        #     os.makedirs(PROCESSED_ORBI_ROOT)

        matches = []
        folder_idx = 1
        file_idx = 0
        for root, dirnames, filenames in os.walk(ORBI_DATA_ROOT):
            # for each folder find videos of 4 angles
            all_files_in_dir = []
            fp = ""
            dest_dir = ""
            matches = fnmatch.filter(filenames, '*.MP4')
            for filename in matches:
                all_files_in_dir.append(os.path.join(root, filename))
            if matches != []:
                fp = os.path.join(root, filename)
                print("Processing directory: "+root+" | destination directory: "+PROCESSED_ORBI_ROOT)

            # if length of files in each folder are less than 4 then the folder doesnt have footage of all angles
            if len(all_files_in_dir) < 4:
                # if not root folder
                if folder_idx != 1:
                    print("Folder: "+root+" is corrputed, is skipped")
            else:
                # make a directory in destination folder
                leftVid = cv2.VideoCapture(all_files_in_dir[3])
                leftEyeVid = cv2.VideoCapture(all_files_in_dir[2])
                rightVid = cv2.VideoCapture(all_files_in_dir[0])
                rightEyeVid = cv2.VideoCapture(all_files_in_dir[1])
                # since all videos have same framerate, only sample one video for FPS
                fps = leftVid.get(cv2.CAP_PROP_FPS)
                framecount = 0
                
                # count the frames for capture
                while(True):
                    # Capture frame-by-frame
                    success_left, leftIMG = leftVid.read()
                    success_leftEye, leftEyeIMG = leftEyeVid.read()
                    success_right, rightIMG = rightVid.read()
                    success_rightEye, rightEyeIMG = rightEyeVid.read()

                    framecount += 1

                    # Check if this is the frame closest to 1 seconds
                    if success_left==True:
                        if framecount == (fps * 0.1) and success_right==True and success_leftEye==True and success_rightEye==True:
                            #resize all images
                            leftIMG = cv2.resize(leftIMG,ADJ_SIZE2)
                            leftEyeIMG = cv2.resize(leftEyeIMG,ADJ_SIZE2)
                            rightIMG = cv2.resize(rightIMG,ADJ_SIZE2)
                            rightEyeIMG = cv2.resize(rightEyeIMG,ADJ_SIZE2)                  

                            #rotate images
                            leftIMG = self.rotate_img(leftIMG,0) #left image rotate 90 deg CCW
                            leftEyeIMG = self.rotate_img(leftEyeIMG,+1) #left eye image rotate 90 deg CCW
                            rightEyeIMG = self.rotate_img(rightEyeIMG,+1) #right eye image rotate 90 deg CCW
                            rightIMG = self.rotate_img(rightIMG,0) #right image rotate 90 deg CW
                            framecount = 0
                            
                            leftIMG = undistort_util.undistort(leftIMG)
                            leftEyeIMG = undistort_util.undistort(leftEyeIMG)                            
                            rightEyeIMG = undistort_util.undistort(rightEyeIMG)
                            rightIMG = undistort_util.undistort(rightIMG)

                            # cv2.imwrite(PROCESSED_ORBI_ROOT+str(file_idx)+"0"+".jpg",leftIMG)
                            # cv2.imwrite(PROCESSED_ORBI_ROOT+str(file_idx)+"1"+".jpg",leftEyeIMG)
                            # cv2.imwrite(PROCESSED_ORBI_ROOT+str(file_idx)+"2"+".jpg",rightIMG)
                            # cv2.imwrite(PROCESSED_ORBI_ROOT+str(file_idx)+"3"+".jpg",rightEyeIMG)
                            
                            # cv2.imwrite(PROCESSED_ORBI_ROOT+str(file_idx)+"0"+".jpg",leftIMG)
                            cv2.imwrite(PROCESSED_ORBI_ROOT+str(file_idx) + ".jpg",leftEyeIMG)
                            # cv2.imwrite(PROCESSED_ORBI_ROOT+str(file_idx)+"2"+".jpg",rightIMG)
                            # cv2.imwrite(PROCESSED_ORBI_ROOT+str(file_idx)+"3"+".jpg",rightEyeIMG)

                            file_idx += 1 
                            
                            # # Check end of video
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                  break
                    else:
                        break
                folder_idx += 1
        cv2.destroyAllWindows()        
        print("Finished proprocessing images, total images: ",file_idx*4)
    
    def label_all_img(self):
        print("start labelling now")
        # init image utility object
        iu = IMG_LABELLER()
        print("labeling images")
        iu.label_all("./labelled_cam")

dp = DATA_PIPELINE()
# dp.process_all_undistort()
dp.process_all_undistort_stitch()
# dp.label_all_img()
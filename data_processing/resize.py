import shutil
import fnmatch
import os
import numpy as np
import cv2

"""
Script written for CV-Aid Design project at McGill
@author jamestang
@version 1.1
This script converts all the images given in parameter TRAIN_DATA_ROOT into squared images with 
size 224 pixel by 224 pixel, the size reduction allows the images to work with machine learnin model as it reduces the 
number of neurons in the network.

"""

# only modify this parameter
TRAIN_DATA_ROOT = './training_data_orbi/'

img_list = []
for root, dirnames, filenames in os.walk(TRAIN_DATA_ROOT):
    matches = fnmatch.filter(filenames, '*.jpg')
    for filename in matches:
        img_list.append(os.path.join(root, filename))

def convert_to_square(img):
        #get size
        height, width, channels = img.shape
        # Create a black image
        x = height if height > width else width
        y = height if height > width else width
        square= np.zeros((x,y,3), np.uint8)
        square[int((y-height)/2):y-int((y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = img
        return square

# report the image processed 
print("Image list processed: ")
print(img_list)

for filename in img_list:
    origin_img = cv2.imread(filename)
    # print(filename)
    square = convert_to_square(origin_img)
    # skip the files that is corrupted, which OpenCV will return an empty image
    if square is None:
        print("File: "+filename+" is corrupted, skipped")
    else:
        square = cv2.resize(square,(224,224))
        cv2.imwrite(filename,square)
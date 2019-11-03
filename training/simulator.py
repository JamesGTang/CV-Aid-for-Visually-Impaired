from tensorflow.python.framework import graph_io, graph_util
from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json
import json
import glob
import cv2
import os
import numpy as np
"""

# Script written for CV-Aid Design project at McGill
# @author jamestang
# @version 1.0
# This script loads model and run the simulation prediction from video file

"""

# get the RUN_ID
with open("./model/RUN_ID.log","r+") as f:
    RUN_ID = str(f.readline())

# refer to train_result.json
architecture_fp = './model/architecture/'+"CV_aid_mobilenet_cam-2.json"
# weights_fp = './model/weights/'+RUN_ID+"/"+"CV_aid_mobilenet_cam-1-01-0.08.hdf5"
weights_fp = './model/mobilenetV1.h5'
# image size
IMG_SIZE = (224,224)

# where to get and store video
ORBI_DATA_ROOT = '../orbi_sample/'
PROCESSED_ORBI_ROOT = '../orbi_sample_simulation/'
OUT_VIDEO_FP = "./simulation/"

class simulator:
    def __init__(self):
        with open(architecture_fp,'r') as f:
            model_json = json.load(f)
        # get the RUN_ID
        with open("./model/RUN_ID.log","r+") as f:
            RUN_ID = str(f.readline())
        
        self.OUT_VIDEO = OUT_VIDEO_FP + RUN_ID +".mp4"
        self.model = model_from_json(model_json)
        self.model.load_weights(weights_fp)

        session = K.get_session()

        print("------------------------ All layers ------------------------")
        print(self.model.layers)

        print("\n\n------------------------ Input layers ------------------------\n\n")
        print(self.model.inputs)

        print("\n\n------------------------ Output layers ------------------------\n\n")
        print(self.model.outputs)
        print("\n\n------------------------ End ------------------------")

    def rotate_img(self,img,rot_deg):
        # rotate ccw
        out=cv2.transpose(img)
        out=cv2.flip(out,flipCode=rot_deg)
        return out
    
    def convert_to_square(self,img):
        #get size
        height, width, channels = img.shape
        # Create a black image
        x = height if height > width else width
        y = height if height > width else width
        square= np.zeros((x,y,3), np.uint8)
        square[int((y-height)/2):y-int((y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = img
        return square

    def create_simulation(self):
        video_out = cv2.VideoWriter(self.OUT_VIDEO, cv2.VideoWriter_fourcc(*'PIM1'), 10.0, (1920, 1080), True)
        for filename in sorted(glob.glob(PROCESSED_ORBI_ROOT+"*.jpg")):
            img = cv2.imread(filename)
            img = self.convert_to_square(img)
            cv2.imshow('image',img)
            resized_img = cv2.resize(img,IMG_SIZE)
            height, width, layers = img.shape
            size = (width,height)
            print(height, width, layers)
            resized_img= np.expand_dims(resized_img, axis=0)
            print(resized_img.shape)
            print(self.model.predict(resized_img))










sim = simulator()
sim.create_simulation()
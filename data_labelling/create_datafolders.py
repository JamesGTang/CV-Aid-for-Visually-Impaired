import numpy as np
from shutil import copyfile
import glob
import re
import os

LABELLED_DIR = "labelled_data/"
SPLIT_DIR = "data_split/"
NUM_ACTIONS = 8

if not os.path.isdir(SPLIT_DIR):
    os.mkdir(SPLIT_DIR)

folder_types = ['train/', 'val/', 'test/']
for folder in folder_types:
    if not os.path.isdir(SPLIT_DIR+folder):
        os.mkdir(SPLIT_DIR+folder)
    for i in range(NUM_ACTIONS):
        if not os.path.isdir(SPLIT_DIR+folder+str(i)):
            os.mkdir(SPLIT_DIR+folder+str(i))

for i in range(NUM_ACTIONS):
    all_fnames_in_folder = glob.glob(LABELLED_DIR+str(i)+'/*.jpg')
    if len(all_fnames_in_folder) == 0:
        continue
    random_choices = np.random.choice(3, len(all_fnames_in_folder), p=[0.8, 0.1, 0.1])
    for j, fname in enumerate(all_fnames_in_folder):
        new_fname = re.sub(LABELLED_DIR, SPLIT_DIR+folder_types[random_choices[j]], fname)
        copyfile(src=fname, dst=new_fname)

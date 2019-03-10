# labelling_street_veer

Implemented code from https://github.com/roggirg/labelling_street_veer for data labelling
Credit: Roger https://github.com/roggirg/labelling_street_veer

Labelling individual images to one of 8 classes.

Place all your images in a directory named "all_frames/". 
Next, run the [main script](https://github.com/roggirg/labelling_street_veer/blob/master/label_img.py), and an opencv
image window will open with vertical lines splitting the image into different slices.
Note that you may need to adjust you image file type in the script (set up for .jpg file).
Each slice maps to a value between 1 and 7 on your keyboard, where 1 is the leftmost slice and 7 is the rightmost slice.
If the image is such that you cannot see the curb on the other side of the road, you should select the "0" key, which
represents the unknown action.

Your images will be stored in a directory named "labelled_data/[corresponding_numerical_label]/", as 224x224 images.
In addition, the code exploits data augmentation by vertically flipping images and assigning the corresponding label
according to the label you provided.
For example, if an image was labelled with "2", the corresponding flipped label is "6".

After being labelled, the raw images are moved to "all_frames/archives/" directory.

Once you have labelled all the images, i.e., all images are in "labelled_data/[corresponding_numerical_label]",
you should run the [data split script](https://github.com/roggirg/labelling_street_veer/blob/master/create_datafolders.py)
which will split the data in train(80%)-val(10%)-test(10%).
These splits will be stored in "data_split/".

# Dependencies

All you need (for now):
- Python 3
- Numpy
- Shutil
- OpenCV

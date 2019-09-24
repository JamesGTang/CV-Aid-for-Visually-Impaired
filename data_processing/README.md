# Data preprocessing

## Credit
Implemented code from https://github.com/roggirg/labelling_street_veer for data labelling

Credit: Roger @ https://github.com/roggirg/labelling_street_veer

## Description 
Labelling individual images to one of 8 classes.

## Procedures
 1. Place all your images in a directory and name it accordingly, depends on the type of data you want to preprocess, run the following two scripts accordingly
	 2.  360 camera data
		 `Python3 360_data_pipeline.py`
	 3. Normal camera data
	 	 `Python3 cam_data_pipeline.py`
 2. an opencv image window will open with vertical lines splitting the image into different slices. Note that you may need to adjust you image file type in the script (set up for .jpg file). 
	 3. Note: the labelling script has the ability for team collaboration, which means the left-over work can be pushed to GitHub and another person can restart the labelling by pull from the repository again.
	 4. If you restarting the labelling or the date has been already processed. run
		 4. 360 camera data
		 `Python3 label_360_img.py`
		 5. Normal camera data
	 	 `Python3 label_cam_img.py` 
 3. Each slice maps to a value between 1 and 7 on your keyboard, where 1 is the leftmost slice and 7 is the rightmost slice.
 4. If the image is such that you cannot see the curb on the other side of the road, you should select the "0" key, which
represents the unknown action.
5. Your images will be stored in a directory given in the parameter "[given directory] /[corresponding_numerical_label]/", as 224x224 images.
6. In addition, the code exploits data augmentation by vertically flipping images and assigning the corresponding label
according to the label you provided. For example, if an image was labelled with "2", the corresponding flipped label is "6".
7. After being labelled, the raw images are moved to "[given directory] /archives/" directory.
8. Once you have labelled all the images, i.e., all images are in "labelled_data/[corresponding_numerical_label]",
you should modify the parameters
`LABELLED_DIR` ( where the script should get the labelled images) 
`SPLIT_DIR` ( where the split sets should be stored )
Train_Valid_Test_Split ( how the data will be split, default is train(80%)-val(10%)-test(10% )
and run `python3 create_datafolders.py`
These splits will be stored in "[SPLIT_DIR]".

## Support modules
`data_convert.py` provides some generalized utilities for working with dataset, especially for working with cam data
`resize.py` provides resize functions for working with 360 images, it converts image size and dimensions. 

## Dependencies

All you need (for now):
- Python 3
- Numpy
- Shutil
- OpenCV

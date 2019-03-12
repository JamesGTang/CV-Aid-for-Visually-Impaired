# train_crossing_model

Code to retrain a Mobilenet classifier on a new task.

# Instructions

## Train
To start, you should place your labelled folder inside this directory (i.e., train_crossing_model/labelled_data/).
In the train.py file, modify [this line](https://github.com/roggirg/train_crossing_model/blob/db08f0e74c8aa407ea5e358968053805c8a9db49/train.py#L11) to point to the correct folder (i.e., data_dir=train_crossing_model/labelled_data/)
Run the program. At the end of training, some results will be printed and two files (.pb and .h5) will be stored in saved_ckpt.

## Test
A folder with some test images is included. So you can test your model with those images by running predict_test_imgs.py.

# Dependencies
- Python 3.5
- Numpy
- Tensorflow 1.11.0
- Keras 2.2.2

I suggest installing an nvidia docker image with these dependencies or use a virtualenv.

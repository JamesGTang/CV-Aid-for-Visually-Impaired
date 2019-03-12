import os

import keras
from keras.preprocessing import image
import numpy as np
from keras.applications import imagenet_utils
import matplotlib.pyplot as plt
import itertools
from scipy.misc import imsave

import scipy.stats


def prepare_image(file, model_type):
    img = image.load_img(file, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    img_arr_expanded = np.expand_dims(img_arr, axis=0)
    if model_type != 'OV1':
        return keras.applications.mobilenetv2.preprocess_input(img_arr_expanded)
    else:
        return img_arr_expanded


def predict_single_image(model, file, model_type='V1'):
    processed_img = prepare_image(file, model_type)
    # return imagenet_utils.decode_predictions(model.predict(processed_img))
    return model.predict(processed_img)


def calculate_test_results(y_true, y_pred):

    # Accuracy and Precision Calculation
    diff_rel = []
    above_threshold = []
    max_diff = 0.
    idx_max_diff = None
    threshold = 2
    pred_unknown = 0
    true_unknown = 0

    for idx in range(len(y_true)):
        # Removing the indices where the label was unknown for either pred or true
        if y_true[idx] != 0 and np.argmax(y_pred[idx]) != 0:
            # Append the differences
            diff_rel.append(y_true[idx] - np.argmax(y_pred[idx]))

            # Check for the maximum difference
            if np.abs(y_true[idx] - np.argmax(y_pred[idx])) > max_diff:
                max_diff = abs(y_true[idx] - np.argmax(y_pred[idx]))
                idx_max_diff = idx

            # Record indices of all predictions above the threshold
            if np.abs(diff_rel[-1]) > threshold:
                above_threshold.append(idx)
        else:
            if y_true[idx] == 0:
                true_unknown += 1
                # imsave("results/all_unknown/true_unknown_" + str(idx) + ".jpeg", X_[idx])
            elif np.argmax(y_pred[idx]) == 0:
                pred_unknown += 1
                # imsave("results/all_unknown/pred_unknown_" + str(idx) + ".jpeg", X_[idx])

    diff_arr = np.array(diff_rel)

    # Precision Metrics
    mean_diff = np.mean(np.abs(diff_arr))
    stdev_diff = np.std(np.abs(diff_arr))

    # Accuracy calculation
    accuracy = float((len(diff_arr)) - float(np.count_nonzero(diff_arr))) / len(diff_arr)

    print("Accuracy: ", accuracy)
    print("Mean Difference: ", mean_diff)
    print("Standard Deviation: ", stdev_diff)
    print("Max Difference: ", max_diff)
    print("Index of Max Diff.: ", idx_max_diff)
    print("Total Number of Images Above threshold: ", len(above_threshold))
    print("Number of Unknowns {0}, True unknowns {1}, Pred Unknowns {2}".
          format(true_unknown+pred_unknown, true_unknown, pred_unknown))

    # return accuracy, mean_diff, stdev_diff, max_diff, idx_max_diff


def calculate_test_results_Regression(y_true, y_pred):

    # Accuracy and Precision Calculation
    diff_rel = []
    above_threshold = []
    max_diff = 0.
    idx_max_diff = None
    threshold = 2
    pred_unknown = 0
    true_unknown = 0

    for idx in range(len(y_true)):
        # Removing the indices where the label was unknown for either pred or true
        # Append the differences
        diff_rel.append(y_true[idx] - round(y_pred[idx][0]))

        # Check for the maximum difference
        if np.abs(y_true[idx] - round(y_pred[idx][0])) > max_diff:
            max_diff = abs(y_true[idx] - round(y_pred[idx][0]))
            idx_max_diff = idx

        # Record indices of all predictions above the threshold
        if np.abs(diff_rel[-1]) > threshold:
            above_threshold.append(idx)

    diff_arr = np.array(diff_rel)

    # Precision Metrics
    mean_diff = np.mean(np.abs(diff_arr))
    stdev_diff = np.std(np.abs(diff_arr))

    # Accuracy calculation
    accuracy = float((len(diff_arr)) - float(np.count_nonzero(diff_arr))) / len(diff_arr)

    print("Accuracy: ", accuracy)
    print("Mean Difference: ", mean_diff)
    print("Standard Deviation: ", stdev_diff)
    print("Max Difference: ", max_diff)
    print("Index of Max Diff.: ", idx_max_diff)
    print("Total Number of Images Above threshold: ", len(above_threshold))
    print("Number of Unknowns {0}, True unknowns {1}, Pred Unknowns {2}".
          format(true_unknown+pred_unknown, true_unknown, pred_unknown))
    print("95% Confidence Interval", mean_confidence_interval(np.abs(diff_arr)))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def plot_confusion_matrix(cm, classes, model_type,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if not os.path.isdir("test_results"):
        os.mkdir("test_results")
    plt.savefig("test_results/confusion_matrix"+model_type+".png")

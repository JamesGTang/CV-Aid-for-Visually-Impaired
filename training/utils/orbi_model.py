import keras
from keras import Model
from keras.layers import Convolution2D, Activation, GlobalAveragePooling2D, Reshape, Dropout, Dense, ReLU
import numpy as np
import tensorflow as tf
import keras.backend as K


def mobilenetv1_transfer_learning(num_classes):
    ft_layers = [15, 22]
    mobilenet = keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=(960,426,3)
    )
    x = mobilenet.layers[-1].output  # -3
    x = Convolution2D(64, (4, 4), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5, name='dropout')(x)
    x = Dense(1)(x)
    predictions = ReLU(max_value=6.0)(x)

    model = Model(inputs=mobilenet.input, outputs=predictions)

    # model = initial_training(model)
    print(model.summary())
    return model, ft_layers


def mobilenetv2_transfer_learning(num_classes):
    ft_layers = [2] # , 14, 22]
    mobilenet = keras.applications.mobilenet.MobileNetV2(
        include_top=False,
        input_shape=(960,426,3)
    )
    x = mobilenet.layers[-3].output  # -14

    x = Convolution2D(64, (3, 3), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5, name='dropout')(x)
    x = Dense(1)(x)
    predictions = ReLU(max_value=6.0)(x)

    # x = GlobalAveragePooling2D()(x)
    # predictions = Dense(num_classes, activation='softmax', name='act_softmax')(x)

    model = Model(inputs=mobilenet.input, outputs=predictions)

    # model = initial_training(model)
    print(model.summary())
    return model, ft_layers


def fine_tune(model, ft_depth):
    for layer in model.layers[-ft_depth:]:
        layer.trainable = True
    for layer in model.layers[:-ft_depth]:
        layer.trainable = False

    # Show which layers are now trainable
    for layer in model.layers:
        if layer.trainable:
            print(layer)
    return model

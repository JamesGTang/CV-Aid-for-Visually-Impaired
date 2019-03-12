from keras.optimizers import Adam, SGD, RMSprop
from utils import models, prediction_utils, save_model_pb
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error
from utils.load_data import load_data


def main():
    # Training Params
    batch_size = 64
    model_type = 'V1'  # V1, V2
    data_dir = "./data_labelling/all_frames_cam/labelled_data/"

    # load the right pre-trained model and set the corresponding optimizer
    if model_type == 'V1':
        model, ft_layers = models.mobilenetv1_transfer_learning(num_classes=8)
        init_optimizer = Adam(lr=0.00025)
        model.compile(init_optimizer, loss='mean_absolute_error', metrics=['mae'])
        preprocess = True
    elif model_type == 'V2':
        model, ft_layers = models.mobilenetv2_transfer_learning(num_classes=8)
        init_optimizer = SGD(lr=0.005)
        model.compile(init_optimizer, loss='mean_squared_error', metrics=['mae'])
        preprocess = True

    # Get my data loaders
    train_batches, val_batches, test_batches = load_data(batch_size=batch_size, head_dir=data_dir,
                                                         preprocess=preprocess, class_mode="sparse")
    print(len(test_batches))

    # Initial training of final layer
    model.fit_generator(train_batches, steps_per_epoch=train_batches.n//batch_size, validation_data=val_batches,
                        validation_steps=val_batches.n//batch_size, epochs=50, verbose=2)

    # for ft_layer_depth in ft_layers:
    #     model = models.fine_tune(model, ft_layer_depth)
    #     model.compile(SGD(lr=0.01), loss='mean_squared_error', metrics=['mae'])
    #     model.fit_generator(train_batches, steps_per_epoch=train_batches.n // batch_size, validation_data=val_batches,
    #                         validation_steps=val_batches.n // batch_size, epochs=10, verbose=2)

    test_labels = test_batches.classes
    predictions = model.predict_generator(test_batches, verbose=2)
    print("Mean Absolute Error", mean_absolute_error(test_labels, predictions))

    # save model h5 and frozen pb
    model.save_weights("saved_ckpt/mobilenet"+model_type+".h5")
    save_model_pb.save_pb(model_type)


if __name__ == '__main__':
    main()

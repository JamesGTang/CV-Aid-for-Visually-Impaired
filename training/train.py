from keras.optimizers import Adam, SGD, RMSprop
from utils import models, prediction_utils, save_model_pb
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error
from utils.load_data import load_data

from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from matplotlib import pyplot as plt

import time

RUN_ID = 1
start_time = 0
end_time = 0

def main():
    # Training Params
    batch_size = 64
    model_type = 'V1'  # V1, V2
    data_dir = "../data_labelling/training_data/"
    MODEL_NAME = "CV_aid_mobilenet_transfer_learning"

    # # load the right pre-trained model and set the corresponding optimizer
    # if model_type == 'V1':
    #     model, ft_layers = models.mobilenetv1_transfer_learning(num_classes=8)
    #     init_optimizer = Adam(lr=0.00025)
    #     model.compile(init_optimizer, loss='mean_absolute_error', metrics=['mae'])
    #     preprocess = True
    # elif model_type == 'V2':
    #     model, ft_layers = models.mobilenetv2_transfer_learning(num_classes=8)
    #     init_optimizer = SGD(lr=0.005)
    #     model.compile(init_optimizer, loss='mean_squared_error', metrics=['mae'])
    #     preprocess = True

    # load the right pre-trained model and set the corresponding optimizer
    if model_type == 'V1':
        # model, ft_layers = models.mobilenetv1_transfer_learning(num_classes=8)
        model, ft_layers = orbi_model.mobilenetv1_transfer_learning(num_classes=8)
        init_optimizer = Adam(lr=0.00025)
        model.compile(
            init_optimizer, 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        preprocess = True
    elif model_type == 'V2':
        # model, ft_layers = models.mobilenetv2_transfer_learning(num_classes=8)
        model, ft_layers = orbi_model.mobilenetv2_transfer_learning(num_classes=8)
        init_optimizer = SGD(lr=0.005)
        model.compile(
            init_optimizer, 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        preprocess = True
    
    # save model
    print("Saving model visual to file")
    if not os.path.isdir('./graph'):
        os.mkdir('./graph')
    plot_model(model, to_file='./graph/'+MODEL_NAME+'.png')
    
    # checkpoint
    if not os.path.isdir('./model'):
        os.mkdir('./model')
    filepath="./model/model-"+str(RUN_ID)+"{epoch:02d}-{val_acc:.2f}.hdf5"
    mode_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # Get my data loaders
    train_batches, val_batches, test_batches = load_data(
        batch_size=batch_size, 
        head_dir=data_dir,
        preprocess=preprocess, 
        class_mode="sparse"
    )

    print("==================================")   
    print("Start timing module:")
    start_time = time.time()
    print("Program starts at time: ",start_time)
    print("Size of training data: ",len(train_batches))
    print("Size of validation data: ",len(val_batches))
    print("Size of test data: ",len(test_batches))

    # Initial training of final layer
    history = model.fit_generator(
        train_batches, 
        steps_per_epoch=train_batches.n//batch_size, 
        validation_data=val_batches,
        validation_steps=val_batches.n//batch_size, 
        epochs=50, 
        callbacks=model_checkpoint,
        verbose=1,
    )

    print(history)
    
    end_time = time.time()
    print("Program ends at time",end_time)
    total_time = (end_time - start_time)
    print("Total time elapsed in training(s): " +str("%.3f" %total_time))

    # save history
    print("Saving history to file")
    if not os.path.isdir('./graph'):
        os.mkdir('./graph')
    with open('./graph/history_r'+str(RUN_ID)+'.log','w+') as f:
        f.write(history)

    # # for ft_layer_depth in ft_layers:
    # #     model = models.fine_tune(model, ft_layer_depth)
    # #     model.compile(SGD(lr=0.01), loss='mean_squared_error', metrics=['mae'])
    # #     model.fit_generator(train_batches, steps_per_epoch=train_batches.n // batch_size, validation_data=val_batches,
    # #                         validation_steps=val_batches.n // batch_size, epochs=10, verbose=2)

    # test_labels = test_batches.classes
    # predictions = model.predict_generator(test_batches, verbose=2)
    # print("Mean Absolute Error", mean_absolute_error(test_labels, predictions))

    # # save model h5 and frozen pb
    # model.save_weights("saved_ckpt/mobilenet"+model_type+".h5")
    # save_model_pb.save_pb(model_type)


if __name__ == '__main__':
    main()

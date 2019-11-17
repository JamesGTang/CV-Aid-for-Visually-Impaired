import tensorflow as tf
# from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.optimizers import Adam, SGD, RMSprop
# from utils import models, prediction_utils, save_model_pb
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error
from utils.load_data import load_data

# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from matplotlib import pyplot as plt
from utils import orbi_model
import time
import os
import pickle
import json
from datetime import date
import numpy

# run_param = {'RUN_ID':0,'Time',"Params","Hyperparameters","Files"}
JSON = {}
Time = {}
Parameters = {}
Hyperparameters = {}
Files = {}
Results = {}
Graphs = {}

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

# update run id in the file
with open("./model/RUN_ID.log","r+") as f:
    RUN_ID = str(int(f.readline()) + 1)
    f.seek(0)
    f.write(RUN_ID)
    f.truncate()
print("+++++++++++++++++++++++++++++++++++++++++++++\n\n\n\n")
print("Initiating training run with RUN_ID: "+RUN_ID)
print("\n\n\n\n+++++++++++++++++++++++++++++++++++++++++++++")
# add run id to JSON
JSON["RUN_ID"] = str(RUN_ID)

# add time to JSON
training_date = str(date.today())
Time["Training date"]=training_date
# with open("./model/train_result.json","w") as f:
#     json.dump(run_param,f)

"""
#
# parameter to identify each training
# MODEL_NAME: the name of the model
# data_dir: where to get training data
# 
"""
# MODEL_NAME = "CV_aid_mobilenet_with_360_dataset"
# data_dir = "../data_processing/training_data_360/"

MODEL_NAME = "CV_aid_mobilenet_cam"
data_dir = "../data_processing/training_data_cam/"
model_type = 'V1'  # V1, V2
# add parameter to JSON
Parameters["Model name"] = MODEL_NAME
Parameters["Training data directory"] = data_dir
Parameters["Model Type"] = model_type

# print(JSON)
"""
# hyperparameter related to training
# batch_size, batch size to feed into traning
# lr: learning rate for ADAM optimizer

"""
batch_size = 64
learning_rate = 0.00025
epoch = 50
loss = "mean_squared_error"

# add parameter to JSON
Hyperparameters["Batch Size"] = batch_size
Hyperparameters["Learning rate(ADAMS)"] = learning_rate
Hyperparameters["Epoch"] = epoch
Hyperparameters["Loss"] = loss

# general variable to hold training information
start_time = 0
end_time = 0


def main():
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
        init_optimizer = Adam(lr=learning_rate)
        model.compile(
            init_optimizer, 
            loss=loss,
            metrics=['acc']
        )
        preprocess = True
    elif model_type == 'V2':
        # model, ft_layers = models.mobilenetv2_transfer_learning(num_classes=8)
        model, ft_layers = orbi_model.mobilenetv2_transfer_learning(num_classes=8)
        init_optimizer = SGD(lr=learning_rate)
        model.compile(
            init_optimizer, 
            loss=loss,
            metrics=['acc']
        )
        preprocess = True
    
    # save model
    model_visual_fp = './graph/'+MODEL_NAME+"-"+RUN_ID+'.png'
    print("Saving model visual to file: "+ model_visual_fp)
    if not os.path.isdir('./graph'):
        os.mkdir('./graph')
    plot_model(model, to_file=model_visual_fp)
    Graphs["Model visual"] = model_visual_fp
    
    architecture_fp = "./model/architecture/"+ MODEL_NAME + "-"+RUN_ID+'.json'
    print("Saving model architecture to file: "+architecture_fp)
    model_json = model.to_json()
    with open(architecture_fp, "w") as json_file:
        json.dump(model_json, json_file)
    Files["Architecture"]=architecture_fp

    # checkpoint
    if not os.path.isdir('./model/weights/'+RUN_ID):
        os.mkdir('./model/weights/'+RUN_ID)
    filepath='./model/weights/'+ RUN_ID + "/" + MODEL_NAME + "-" + RUN_ID +"-{epoch:02d}-{val_acc:.2f}.hdf5"
    Files["Weights"] = './model/weights/'+ RUN_ID + "/" + MODEL_NAME + "-" + RUN_ID + "****.hdf5"
    model_checkpoint = ModelCheckpoint(
        filepath, 
        monitor='val_acc', 
        verbose=1, 
        save_weights_only=True,
        save_best_only=True, 
        mode='max'
    )

    # Get my data loaders
    train_batches, val_batches, test_batches = load_data(
        batch_size=batch_size, 
        head_dir=data_dir,
        preprocess=preprocess, 
        class_mode="sparse"
    )

    print("============== Session Parameters ============")   
    # print("Start timing module:")
    start_time = time.time()
    print("Program starts at time: ",start_time)
    Time["Start time"] = start_time
    print("Model name:  ",MODEL_NAME)
    print("RUN_ID: ", RUN_ID)
    print("Data directory: ",data_dir)
    print("Model type: "+model_type)
    
    print("============== Session Hyperparameters =============")   
    # print("Start timing module:")
    print("Batch size: " + str(batch_size))
    print("Epoch: " + str(epoch))
    print("learning_rate: "+ str(learning_rate))
    print("Training data batches: ",len(train_batches))
    print("Validation data batches: ",len(val_batches))
    print("Test data batches: ",len(test_batches))

    Hyperparameters["Training data batches"] = len(train_batches)
    Hyperparameters["Validation data batches"] = len(val_batches)
    Hyperparameters["Test data batches"] = len(test_batches)
    # Initial training of final layer
    history = model.fit_generator(
        train_batches, 
        steps_per_epoch=train_batches.n//batch_size, 
        validation_data=val_batches,
        validation_steps=val_batches.n//batch_size, 
        epochs=epoch, 
        callbacks=[model_checkpoint],
        verbose=1,
    )    
    end_time = time.time()
    print("Program ends at time",end_time)
    Time["End time"] = end_time
    total_time = (end_time - start_time)
    Time["Total time(s)"] = str("%.3f" %total_time)
    print("Total time elapsed in training(s): " +str("%.3f" %total_time))

    # save history
    print("Saving history to file")
    if not os.path.isdir('./graph'):
        os.mkdir('./graph')
    with open('./graph/history_RUN_'+str(RUN_ID)+'.pkl','wb') as f:
        pickle.dump(history.history,f)
    Results["History"] = history.history
    print(history.history)
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

    # plot
    plt.figure(figsize=[16,9])
    plt.plot(history.history['loss'],'r',linewidth=2.0)
    plt.plot(history.history['val_loss'],'b',linewidth=2.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=16)
    plt.xlabel('Epochs ',fontsize=14)
    plt.ylabel('Loss',fontsize=14)
    plt.title('Loss Curves for Model'+MODEL_NAME+" Run ID: "+str(RUN_ID),fontsize=20)
    plt.savefig('./graph/RUN_'+str(RUN_ID)+'_loss_curve.png')
    Graphs["Loss graph"]='./graph/RUN_'+str(RUN_ID)+'_loss_curve.png'
     
    # Accuracy Curves
    plt.figure(figsize=[16,9])
    plt.plot(history.history['acc'],'r',linewidth=2.0)
    plt.plot(history.history['val_acc'],'b',linewidth=2.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=16)
    plt.xlabel('Epochs ',fontsize=14)
    plt.ylabel('Accuracy',fontsize=14)
    plt.title('Accuracy Curves for Model '+MODEL_NAME+" Run ID: "+str(RUN_ID),fontsize=20)
    plt.savefig('./graph/RUN_'+str(RUN_ID)+'_acc_curve.png')
    Graphs["Accuracy graph"]='./graph/RUN_'+str(RUN_ID)+'_acc_curve.png'

    # add all to JSON
    JSON["Time"]=Time
    JSON["Parameters"] = Parameters
    JSON["Hyperparameters"] = Hyperparameters
    JSON["Files"] = Files
    JSON["Results"] = Results
    JSON["Graphs"] = Graphs
    DATA = {}
    DATA[RUN_ID] = JSON

    with open('./model/train_result.json') as f:
        data = json.load(f)

    # data.update(JSON.replace("\'", "\""))
    data.update(DATA)

    with open('./model/train_result.json', 'w') as f:
        json.dump(obj=data, indent=4, fp=f, cls=MyEncoder)
    
    print("+++++++++++++++++++++++++++++++++++++++++++++\n\n\n\n")
    print("Training result for RUN_ID: ")
    print(JSON)
    print("\n\n\n\n+++++++++++++++++++++++++++++++++++++++++++++")

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
if __name__ == '__main__':
    main()

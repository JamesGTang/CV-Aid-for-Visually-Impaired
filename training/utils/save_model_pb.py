from tensorflow.python.framework import graph_io, graph_util
from keras import backend as K
from keras.models import load_weights

"""

# Script written for CV-Aid Design project at McGill
# @author jamestang
# @version 1.0
# This script converts saved weights from training to .pb file for use in android app
# Modifies only the model_filepath.
# and the model version should be the same as train.py

"""

"""
# RUN_ID: run ID that identify each training
# MODEL_NAME: the name of the model
# model_filepath: where to retrieve the saved model
# model_type: which version of the model is being loaded
# 
"""
RUN_ID = 0 
# MODEL_NAME = "CV_aid_mobilenet_with_360_dataset"
MODEL_NAME = "CV_aid_mobilenet_with_cam_dataset"
model_filepath = "./model/CV_aid_mobilenet_with_cam_dataset-0V1-36-0.84.hdf5"
model_type = 'V1'  # V1, V2



def convert_to_pb(model_type,model_filepath):

    if model_type == 'V1':
        final_layer = 're_lu_1/Relu'
    elif model_type == 'V2':
        final_layer = 're_lu_1/Relu'
    elif model_type == 'OV1' or model_type == 'unknown':
        final_layer = "loss/Softmax"

    model = load_weights(model_filepath)
    session = K.get_session()
    for key in session.graph._names_in_use.keys():
        print(key)

    print(model_filepath.split("/"))
    # minimal_graph = graph_util.convert_variables_to_constants(session, session.graph.as_graph_def(), [final_layer])
    # filename = model_filepath.split("/") + ".pb"
    # print(filename)
    # graph_io.write_graph(minimal_graph, '.', filename, as_text=False)
    # print(conversion success)

convert_to_pb(model_type,model_filepath)
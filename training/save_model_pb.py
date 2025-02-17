# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import load_model
# from tensorflow.keras.models import model_from_json
# import json

from tensorflow.python.framework import graph_io, graph_util
from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json
import json

"""

# Script written for CV-Aid Design project at McGill
# @author jamestang
# @version 1.0
# This script converts saved weights from training to .pb file for use in android app
# and the model version should be the same as train.py

"""

# get the RUN_ID
with open("./model/RUN_ID.log","r+") as f:
    RUN_ID = str(f.readline())

# refer to train_result.json
architecture_fp = './model/architecture/CV_aid_mobilenet_cam-41.json'
# weights_fp = './model/weights/'+RUN_ID+"/"+"CV_aid_mobilenet_cam-1-01-0.08.hdf5"
weights_fp = './model/weights/41/CV_aid_mobilenet_cam-41-33-0.56.hdf5'
save_fp = './model/saved_pb/'+RUN_ID+".pb"
model_type = 'V1'  # V1, V2

def convert_to_pb():

    if model_type == 'V1':
        final_layer = 'RELU_FINAL/Relu6'
    elif model_type == 'V2':
        final_layer = 'RELU_FINAL/Relu6'
    elif model_type == 'OV1' or model_type == 'unknown':
        final_layer = "loss/Softmax"
    
    with open(architecture_fp,'r') as f:
        model_json = json.load(f)
    

    K.clear_session()
    K.set_learning_phase(0)

    model = model_from_json(model_json)
    model.load_weights(weights_fp)

    session = K.get_session()

    print("------------------------ All layers ------------------------")
    print(model.layers)

    print("\n\n------------------------ Input layers ------------------------\n\n")
    print(model.inputs)

    print("\n\n------------------------ Output layers ------------------------\n\n")
    print(model.outputs)
    print("\n\n------------------------ End ------------------------")
   
    model.summary()
    
    # print(model_filepath.split("/"))
    minimal_graph = graph_util.convert_variables_to_constants(session, session.graph.as_graph_def(), [final_layer])
    print("Saving to: " + save_fp)
    graph_io.write_graph(minimal_graph, '.', save_fp, as_text=False)
    print("conversion success")

convert_to_pb()
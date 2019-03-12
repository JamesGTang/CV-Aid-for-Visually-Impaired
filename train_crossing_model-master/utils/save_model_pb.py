from tensorflow.python.framework import graph_io, graph_util
from keras import backend as K


def save_pb(model_type):

    if model_type == 'V1':
        final_layer = 're_lu_1/Relu'
    elif model_type == 'V2':
        final_layer = 're_lu_1/Relu'
    elif model_type == 'OV1' or model_type == 'unknown':
        final_layer = "loss/Softmax"

    session = K.get_session()
    for key in session.graph._names_in_use.keys():
        print(key)

    minimal_graph = graph_util.convert_variables_to_constants(session, session.graph.as_graph_def(), [final_layer])
    filename = "saved_ckpt/mobilenet" + model_type + ".pb"
    graph_io.write_graph(minimal_graph, '.', filename, as_text=False)

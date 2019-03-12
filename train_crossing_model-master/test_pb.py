import glob

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from utils.prediction_utils import prepare_image


model_type = 'V1'
with tf.Graph().as_default() as graph:
    with tf.Session() as sess:
        print("Load graph")

        with gfile.FastGFile("saved_ckpt/mobilenet"+model_type+".pb", 'rb') as f:

            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

            tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="",
                                op_dict=None, producer_op_list=None)

            for op in graph.get_operations():
                print("Operation Name: ", op.name)
                print("Tensor Stats: ", str(op.values()))

            l_input = graph.get_tensor_by_name('input_1:0')
            l_output = graph.get_tensor_by_name('re_lu_1/Relu:0')
            print("Shape of input:", tf.shape(l_input))
            tf.global_variables_initializer()

            all_fnames = glob.glob("test_imgs/*.png") + glob.glob("test_imgs/*.jpg")
            for fname in all_fnames:
                img_input = prepare_image(fname, model_type=model_type)
                Session_out = sess.run(l_output, feed_dict={l_input: img_input})
                # print(np.argmax(Session_out, axis=1))
                print(fname, Session_out)

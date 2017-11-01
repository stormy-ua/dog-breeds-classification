import os
import sys
import urllib2

import numpy as np
import pandas as pd
import tensorflow as tf

from src.common import consts
from src.data_preparation import dataset
from src.freezing import freeze
from src.common import paths

def convert(model_name, export_dir):
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir=export_dir)

    with tf.Graph().as_default(), tf.Session().as_default() as sess:
        tensors = freeze.unfreeze_into_current_graph(
            os.path.join(paths.FROZEN_MODELS_DIR, model_name + '.pb'),
            tensor_names=[consts.INCEPTION_INPUT_TENSOR, consts.OUTPUT_TENSOR_NAME])

        raw_image_proto_info = tf.saved_model.utils.build_tensor_info(tensors[consts.INCEPTION_INPUT_TENSOR])
        probs_proto_info = tf.saved_model.utils.build_tensor_info(tensors[consts.OUTPUT_TENSOR_NAME])

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'image_raw': raw_image_proto_info},
                outputs={'probs': probs_proto_info},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={
                                                 tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
                                             })

    builder.save()

if __name__ == '__main__':
    convert(consts.CURRENT_MODEL_NAME, export_dir='/tmp/dogs_1')
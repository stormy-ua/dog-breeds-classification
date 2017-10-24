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


def infer(model_name, img_raw):
    with tf.Graph().as_default(), tf.Session().as_default() as sess:
        tensors = freeze.unfreeze_into_current_graph(
            os.path.join(paths.FROZEN_MODELS_DIR, model_name + '.pb'),
            tensor_names=[consts.INCEPTION_INPUT_TENSOR, consts.OUTPUT_TENSOR_NAME])

        _, one_hot_decoder = dataset.one_hot_label_encoder()

        probs = sess.run(tensors[consts.OUTPUT_TENSOR_NAME],
                         feed_dict={tensors[consts.INCEPTION_INPUT_TENSOR]: img_raw})

        breeds = one_hot_decoder(np.identity(consts.CLASSES_COUNT)).reshape(-1)

        # print(breeds)

        df = pd.DataFrame(data={'prob': probs.reshape(-1), 'breed': breeds})
        print(df.sort_values(['prob'], ascending=False).take(range(5)))


if __name__ == '__main__':
    src = sys.argv[1]
    path = sys.argv[2] # uri to a dog image to classify
    if src == 'uri':
        response = urllib2.urlopen(path)
        img_raw = response.read()
    else:
        with open(path, 'r') as f:
            img_raw = f.read()

    infer(consts.CURRENT_MODEL_NAME, img_raw)

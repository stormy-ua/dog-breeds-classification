import os

import numpy as np
import pandas as pd
import tensorflow as tf

import consts
import dataset
import models
from src.common import paths


def infer_train(model_name, output_probs, x):
    BATCH_SIZE = 20000

    with tf.Session().as_default() as sess:
        ds, filename = dataset.features_dataset()
        ds_iter = ds.batch(BATCH_SIZE).make_initializable_iterator()
        sess.run(ds_iter.initializer, feed_dict={filename: paths.TRAIN_TF_RECORDS})

        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        lines = open(os.path.join(paths.CHECKPOINTS_DIR, model_name + '_latest')).read().split('\n')
        last_checkpoint = [l.split(':')[1].replace('"', '').strip() for l in lines if 'model_checkpoint_path:' in l][0]
        saver.restore(sess, os.path.join(paths.CHECKPOINTS_DIR, last_checkpoint))

        _, one_hot_decoder = dataset.one_hot_label_encoder()

        breeds = one_hot_decoder(np.identity(consts.CLASSES_COUNT))
        agg_test_df = None

        try:
            while True:
                test_batch = sess.run(ds_iter.get_next())

                inception_output = test_batch['inception_output']
                labels = test_batch['label']

                pred_probs = sess.run(output_probs, feed_dict={x: inception_output.T})
                pred_probs_max = pred_probs >= np.max(pred_probs, axis=0)
                pred_breeds = one_hot_decoder(pred_probs_max.T)

                test_df = pd.DataFrame(data={'pred': pred_breeds, 'actual': labels})

                if agg_test_df is None:
                    agg_test_df = test_df
                else:
                    agg_test_df = agg_test_df.append(test_df)

        except tf.errors.OutOfRangeError:
            print('End of the dataset')

        print(agg_test_df.take(range(0, 10)))

        agg_test_df.to_csv(paths.TRAIN_CONFUSION, index_label='id', float_format='%.17f')

        print('predictions saved to %s' % paths.TRAIN_CONFUSION)


if __name__ == '__main__':
    with tf.Graph().as_default():
        x = tf.placeholder(dtype=tf.float32, shape=(consts.INCEPTION_CLASSES_COUNT, None), name="x")
        _, output_probs, _, _ = models.denseNNModel(
            x, consts.HEAD_MODEL_LAYERS, gamma=0.01)
        infer_train(consts.CURRENT_MODEL_NAME, output_probs, x)

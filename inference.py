import tensorflow as tf
import paths
import pandas as pd
from sklearn import preprocessing
import numpy as np
import dataset
import models
import consts
import train
import os

def infer_test(model_name, output_probs, x):
    BATCH_SIZE = 20000

    with tf.Session().as_default() as sess:
        ds, filename = dataset.test_features_dataset()
        ds_iter = ds.batch(BATCH_SIZE).make_initializable_iterator()
        sess.run(ds_iter.initializer, feed_dict={ filename: paths.TEST_TF_RECORDS })

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
                ids = test_batch['id']

                pred_probs = sess.run(output_probs, feed_dict={x: inception_output.T})

                #print(pred_probs.shape)

                test_df = pd.DataFrame(data=pred_probs.T, columns=breeds)
                test_df.index = ids

                if agg_test_df is None:
                    agg_test_df = test_df
                else:
                    agg_test_df = agg_test_df.append(test_df)

        except tf.errors.OutOfRangeError:
            print('End of the dataset')

        agg_test_df.to_csv(paths.TEST_PREDICTIONS, index_label='id', float_format='%.17f')

        print('predictions saved to %s'%paths.TEST_PREDICTIONS)

if __name__ == '__main__':
    with tf.Graph().as_default():
        #_, output_probs, x, _, _ = models.logisticRegressionModel([2048, consts.CLASSES_COUNT])
        x = tf.placeholder(dtype=tf.float32, shape=(consts.INCEPTION_CLASSES_COUNT, None), name="x")
        _, output_probs, _, _ = models.denseNNModel(
            x, [consts.INCEPTION_CLASSES_COUNT, 1024, consts.CLASSES_COUNT], gamma=0.01)
        infer_test('stanford_5_64_0001', output_probs, x)








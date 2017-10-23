import tensorflow as tf
import paths
import pandas as pd
from sklearn import preprocessing
import numpy as np
import dataset
import pyprind
import consts
import models
import os

def train_dev_split(sess, tf_records_path, dev_set_size=2000, batch_size=64, train_sample_size=2000):
    ds_, filename = dataset.features_dataset()

    ds = ds_.shuffle(buffer_size=20000)

    train_ds = ds.skip(dev_set_size).repeat()
    train_ds_iter = train_ds.shuffle(buffer_size=20000) \
        .batch(batch_size) \
        .make_initializable_iterator()

    train_sample_ds = ds.skip(dev_set_size)
    train_sample_ds_iter = train_sample_ds.shuffle(buffer_size=20000) \
        .take(train_sample_size) \
        .batch(train_sample_size) \
        .make_initializable_iterator()

    dev_ds_iter = ds.take(dev_set_size).batch(dev_set_size).make_initializable_iterator()

    sess.run(train_ds_iter.initializer, feed_dict={filename: tf_records_path})
    sess.run(dev_ds_iter.initializer, feed_dict={filename: tf_records_path})
    sess.run(train_sample_ds_iter.initializer, feed_dict={filename: tf_records_path})

    return train_ds_iter.get_next(), dev_ds_iter.get_next(), train_sample_ds_iter.get_next()


def error(x, output_probs, name):
    expected = tf.placeholder(tf.float32, shape=(consts.CLASSES_COUNT, None), name='expected')
    exp_vs_output = tf.equal(tf.argmax(output_probs, axis=0), tf.argmax(expected, axis=0))
    accuracy = 1. - tf.reduce_mean(tf.cast(exp_vs_output, dtype=tf.float32))
    summaries = [tf.summary.scalar(name, accuracy)]

    merged_summaries = tf.summary.merge(summaries)

    def run(sess, output, expected_):
        acc, summary_acc = sess.run([accuracy, merged_summaries],
                                    feed_dict={x: output, expected: expected_})

        return acc, summary_acc

    return run


def make_model_name(prefix, batch_size, learning_rate):
    return '%s_%d_%s' % (prefix, batch_size, str(learning_rate).replace('0.', ''))


if __name__ == '__main__':
    BATCH_SIZE = 64
    DEV_SET_SIZE = 500
    TRAIN_SAMPLE_SIZE = 1000
    EPOCHS_COUNT = 5000
    LEARNING_RATE = 0.0001

    model_name = make_model_name(prefix=consts.CURRENT_MODEL_NAME, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

    with tf.Graph().as_default() as g, tf.Session().as_default() as sess:
        next_train_batch, get_dev_ds, get_train_sample_ds = \
            train_dev_split(sess, paths.TRAIN_TF_RECORDS,
                            dev_set_size=DEV_SET_SIZE,
                            batch_size=BATCH_SIZE,
                            train_sample_size=TRAIN_SAMPLE_SIZE)

        dev_set = sess.run(get_dev_ds)
        dev_set_inception_output = dev_set[consts.INCEPTION_OUTPUT_FIELD]
        dev_set_y_one_hot = dev_set[consts.LABEL_ONE_HOT_FIELD]

        train_sample = sess.run(get_train_sample_ds)
        train_sample_inception_output = train_sample[consts.INCEPTION_OUTPUT_FIELD]
        train_sample_y_one_hot = train_sample[consts.LABEL_ONE_HOT_FIELD]

        x = tf.placeholder(dtype=tf.float32, shape=(consts.INCEPTION_CLASSES_COUNT, None), name="x")
        cost, output_probs, y, nn_summaries = models.denseNNModel(
            x, [consts.INCEPTION_CLASSES_COUNT, 1024, consts.CLASSES_COUNT], gamma=0.001)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

        dev_error_eval = error(x, output_probs, name='dev_error')
        train_error_eval = error(x, output_probs, name='train_error')

        nn_merged_summaries = tf.summary.merge(nn_summaries)
        tf.global_variables_initializer().run()

        writer = tf.summary.FileWriter(os.path.join(paths.SUMMARY_DIR, model_name))

        bar = pyprind.ProgBar(EPOCHS_COUNT, update_interval=1, width=60)

        saver = tf.train.Saver()

        for epoch in range(0, EPOCHS_COUNT):
            batch_features = sess.run(next_train_batch)
            batch_inception_output = batch_features[consts.INCEPTION_OUTPUT_FIELD]
            batch_y = batch_features[consts.LABEL_ONE_HOT_FIELD]

            _, summary = sess.run([optimizer, nn_merged_summaries], feed_dict={
                                      x: batch_inception_output.T,
                                      y: batch_y.T
                                  })

            writer.add_summary(summary, epoch)

            _, dev_summaries = dev_error_eval(sess, dev_set_inception_output.T, dev_set_y_one_hot.T)
            writer.add_summary(dev_summaries, epoch)

            _, train_sample_summaries = train_error_eval(sess, train_sample_inception_output.T, train_sample_y_one_hot.T)
            writer.add_summary(train_sample_summaries, epoch)

            writer.flush()

            if epoch % 10 == 0 or epoch == EPOCHS_COUNT:
                saver.save(sess, os.path.join(paths.CHECKPOINTS_DIR, model_name), latest_filename=model_name+'_latest')

            bar.update()

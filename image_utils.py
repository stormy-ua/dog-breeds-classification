import tensorflow as tf
import paths
import pandas as pd
from sklearn import preprocessing
import numpy as np
import dataset
import pyprind
import consts
import train
import matplotlib.pyplot as plt

# makes sense to have k = {1, 3}
def rotate_ccw(img, k = 1):
    return tf.image.rot90(img, k)

def flip_along_width(img):
    return tf.image.flip_left_right(img)

#delta = {-0.1, 0.1}
def adjust_brightness(img, delta=-0.1):
    return tf.image.adjust_brightness(img, delta=delta)

def adjust_contrast(img, contrast_factor=0.8):
    return tf.image.adjust_contrast(img, contrast_factor=contrast_factor)


if __name__ == '__main__':
    with tf.Graph().as_default() as g, tf.Session().as_default() as sess:
        next_train_batch, _, _ = train.train_dev_split(sess, paths.TRAIN_TF_RECORDS, batch_size=1)

        batch = sess.run(next_train_batch)
        img_raw = batch[consts.IMAGE_RAW_FIELD]
        img = adjust_brightness(tf.image.decode_jpeg(img_raw[0]), 0.1).eval()

        print(img.shape)

        plt.imshow(img)
        plt.show()

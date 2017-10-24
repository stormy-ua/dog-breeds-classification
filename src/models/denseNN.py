import numpy as np
import tensorflow as tf

from src.common import consts


def denseNNModel(input_node, layers, gamma=0.1):
    n_x = layers[0]
    n_y = layers[-1]
    L = len(layers)
    summaries = []
    Ws = []

    with tf.name_scope("placeholders"):
        # x = tf.placeholder(dtype=tf.float32, shape=(n_x, None), name="x")
        y = tf.placeholder(dtype=tf.float32, shape=(n_y, None), name="y")

    a = input_node

    with tf.name_scope("hidden_layers"):
        for l in range(1, len(layers) - 1):
            W = tf.Variable(np.random.randn(layers[l], layers[l - 1]) / tf.sqrt(layers[l - 1] * 1.0), dtype=tf.float32,
                            name="W" + str(l))
            Ws.append(W)
            summaries.append(tf.summary.histogram('W' + str(l), W))
            b = tf.Variable(np.zeros((layers[l], 1)), dtype=tf.float32, name="b" + str(l))
            summaries.append(tf.summary.histogram('b' + str(l), b))
            z = tf.matmul(W, a) + b
            a = tf.nn.relu(z)

    W = tf.Variable(np.random.randn(layers[L - 1], layers[L - 2]) / tf.sqrt(layers[L - 2] * 1.0), dtype=tf.float32,
                    name="W" + str(L - 1))
    summaries.append(tf.summary.histogram('W' + str(L - 1), W))
    b = tf.Variable(np.zeros((layers[L - 1], 1)), dtype=tf.float32, name="b" + str(L - 1))
    summaries.append(tf.summary.histogram('b' + str(L - 1), b))
    z = tf.matmul(W, a) + b

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf.transpose(y), logits=tf.transpose(z)))  # +\
        # gamma * tf.reduce_sum([tf.nn.l2_loss(w) for w in Ws])
        summaries.append(tf.summary.scalar('cost', cost))

    output = tf.nn.softmax(z, dim=0, name=consts.OUTPUT_NODE_NAME)

    return cost, output, y, summaries

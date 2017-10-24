import tensorflow as tf

from src.common import consts
import freeze
from src.common import paths


def inception_model():
    tensors = freeze.unfreeze_into_current_graph(paths.IMAGENET_GRAPH_DEF,
                                                 tensor_names=[
                                                     consts.INCEPTION_INPUT_TENSOR,
                                                     consts.INCEPTION_OUTPUT_TENSOR])

    def forward(sess, image_raw):
        out = sess.run(tensors[consts.INCEPTION_OUTPUT_TENSOR], {tensors[consts.INCEPTION_INPUT_TENSOR]: image_raw})
        return out

    return forward


if __name__ == '__main__':
    with tf.Session().as_default() as sess:
        image_raw = tf.read_file('../../images/airedale.jpg').eval()

    g = tf.Graph()
    sess = tf.Session(graph=g)

    with g.as_default():
        model = inception_model()

    with g.as_default():
        out = model(sess, image_raw)
        print(out.shape)

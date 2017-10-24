import os

import tensorflow as tf
from tensorflow.python.tools import freeze_graph

from src.common import consts
from src.models import denseNN
from src.common import paths


def _freeze_graph(graph_path, checkpoint_path, output_node_names, output_path):
    restore_op_name = 'save/restore_all'
    filename_tensor_name = 'save/Const:0'

    saved_path = freeze_graph.freeze_graph(
        graph_path, '', True, checkpoint_path,
        output_node_names, restore_op_name, filename_tensor_name,
        output_path, False, '', '')

    print('Frozen model saved to ' + output_path)

    return saved_path


def freeze_current_model(model_name, output_node_names):
    lines = open(os.path.join(paths.CHECKPOINTS_DIR, model_name + '_latest')).read().split('\n')
    last_checkpoint = [l.split(':')[1].replace('"', '').strip() for l in lines if 'model_checkpoint_path:' in l][0]

    checkpoint_path = os.path.join(paths.CHECKPOINTS_DIR, last_checkpoint)
    graph_path = os.path.join(paths.GRAPHS_DIR, model_name + '.pb')
    output_graph_path = os.path.join(paths.FROZEN_MODELS_DIR, model_name + '.pb')

    #saver = tf.train.Saver()
    #checkpoint_path = saver.save(sess, checkpoint_prefix, global_step=0, latest_filename=model_name)
    tf.train.write_graph(g, paths.GRAPHS_DIR, os.path.basename(graph_path), as_text=False)

    _freeze_graph(graph_path, checkpoint_path, output_node_names=output_node_names, output_path=output_graph_path)


def freeze_model(model_name, checkpoint, output_node_names):
    checkpoint_path = os.path.join(paths.CHECKPOINTS_DIR, checkpoint)
    graph_path = os.path.join(paths.GRAPHS_DIR, model_name + '.pbtext')
    output_graph_path = os.path.join(paths.FROZEN_MODELS_DIR, model_name + '.pb')

    _freeze_graph(graph_path, checkpoint_path, output_node_names=output_node_names, output_path=output_graph_path)


def unfreeze_into_current_graph(model_path, tensor_names):
    with tf.gfile.FastGFile(name=model_path, mode='rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        g = tf.get_default_graph()

        tensors = {t: g.get_tensor_by_name(t) for t in tensor_names}

        return tensors


if __name__ == '__main__':
    with tf.Graph().as_default() as g, tf.Session().as_default() as sess:
        tensors = unfreeze_into_current_graph(paths.IMAGENET_GRAPH_DEF,
                                              tensor_names=[
                                                  consts.INCEPTION_INPUT_TENSOR,
                                                  consts.INCEPTION_OUTPUT_TENSOR])

        _, output_probs, y, _ = denseNN.denseNNModel(
            tf.reshape(tensors[consts.INCEPTION_OUTPUT_TENSOR], shape=(-1, 1), name=consts.HEAD_INPUT_NODE_NAME),
                consts.HEAD_MODEL_LAYERS,gamma=0.01)

        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(paths.CHECKPOINTS_DIR, consts.CURRENT_MODEL_NAME))

        freeze_current_model(consts.CURRENT_MODEL_NAME, output_node_names=consts.OUTPUT_NODE_NAME)
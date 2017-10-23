import os

JPEG_EXT = '.jpg'
ROOT = '/Users/panarky/Projects/tensorflow_docker/share/data/'
TRAIN_DIR = os.path.join(ROOT, 'train')
TEST_DIR = os.path.join(ROOT, 'test')
#TRAIN_TF_RECORDS = os.path.join(ROOT, 'dogs_train.tfrecords')
TRAIN_TF_RECORDS = os.path.join('data', 'stanford.tfrecords')
TEST_TF_RECORDS = os.path.join(ROOT, 'dogs_test.tfrecords')
LABELS = os.path.join(ROOT, 'train', 'labels.csv')
IMAGENET_GRAPH_DEF = '/tmp/imagenet/classify_image_graph_def.pb'
TEST_PREDICTIONS = os.path.join(ROOT, 'predictions.csv')
METRICS_DIR = 'metrics'
TRAIN_CONFUSION = os.path.join(METRICS_DIR, 'training_confusion.csv')
FROZEN_MODELS_DIR = 'frozen'
CHECKPOINTS_DIR = 'checkpoints'
GRAPHS_DIR = 'graphs'
SUMMARY_DIR = 'summary'
STANFORD_DS_DIR = 'data/stanford_ds'
STANFORD_DS_TF_RECORDS = os.path.join('data', 'stanford.tfrecords')


import os

JPEG_EXT = '.jpg'
DATA_ROOT = 'data/'
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
TEST_DIR = os.path.join(DATA_ROOT, 'test')
#TRAIN_TF_RECORDS = os.path.join(ROOT, 'dogs_train.tfrecords')
TRAIN_TF_RECORDS = os.path.join(DATA_ROOT, 'stanford.tfrecords')
TEST_TF_RECORDS = os.path.join(DATA_ROOT, 'dogs_test.tfrecords')
LABELS = os.path.join(DATA_ROOT, 'train', 'labels.csv')
IMAGENET_GRAPH_DEF = 'frozen/inception/classify_image_graph_def.pb'
TEST_PREDICTIONS = 'predictions.csv'
METRICS_DIR = 'metrics'
TRAIN_CONFUSION = os.path.join(METRICS_DIR, 'training_confusion.csv')
FROZEN_MODELS_DIR = 'frozen'
CHECKPOINTS_DIR = 'checkpoints'
GRAPHS_DIR = 'graphs'
SUMMARY_DIR = 'summary'
STANFORD_DS_DIR = os.path.join(DATA_ROOT, 'stanford_ds')
STANFORD_DS_TF_RECORDS = os.path.join(DATA_ROOT, 'stanford.tfrecords')


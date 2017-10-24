import pyprind
import tensorflow as tf

from src.common import consts
import dataset
import image_utils
from src.freezing import inception
from src.common import paths
from tf_record_utils import *

def read_example(id, breed):
    image_str = tf.read_file(tf.string_join([paths.TRAIN_DIR, '/', id, paths.JPEG_EXT], separator=''))
    return id, image_str, breed


def read_test_example(file_id):
    image_str = tf.read_file(tf.string_join([paths.TEST_DIR, '/', file_id], separator=''))
    parts = tf.string_split(tf.expand_dims(file_id, 0), '.').values
    return image_str, parts[0]


def parse_row(line):
    vals = tf.string_split(tf.expand_dims(line, 0), ',').values
    return vals[0], vals[1]


def build_train_example(img, one_hot_label, breed_label, inception_output):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': bytes_feature(breed_label.encode()),
        consts.IMAGE_RAW_FIELD: bytes_feature(img),
        consts.LABEL_ONE_HOT_FIELD: float_feature(one_hot_label),
        consts.INCEPTION_OUTPUT_FIELD: float_feature(inception_output)}))

    return example


def image_augmenter():
    img = tf.placeholder(tf.string)

    img_decoded = tf.image.decode_jpeg(img)

    # img1_tf = tf.image.encode_jpeg(image_utils.rotate_ccw(img_decoded, 1))
    # img2_tf = tf.image.encode_jpeg(image_utils.rotate_ccw(img_decoded, 3))
    img1_tf = tf.image.encode_jpeg(image_utils.adjust_contrast(img_decoded))
    img2_tf = tf.image.encode_jpeg(image_utils.adjust_brightness(img_decoded, -0.15))
    img3_tf = tf.image.encode_jpeg(image_utils.flip_along_width(img_decoded))

    # examples.append(build_train_example(
    #    tf.image.encode_jpeg(image_utils.adjust_brightness(img_decoded, 0.1)).eval(),
    #                         one_hot_label, breed_label, inception_output))

    # examples.append(build_train_example(
    #    tf.image.encode_jpeg(image_utils.adjust_brightness(img_decoded, -0.1)).eval(),
    #                         one_hot_label, breed_label, inception_output))

    # examples.append(build_train_example(
    #    tf.image.encode_jpeg(image_utils.adjust_contrast(img_decoded)).eval(),
    #                         one_hot_label, breed_label, inception_output))

    sess = tf.get_default_session()

    def augment(image):
        img1, img2, img3 = sess.run([img1_tf, img2_tf, img3_tf], feed_dict={img: image})

        return [img1, img2, img3]

    return augment


def convert_train(tfrecords_path):
    one_hot_encoder, _ = dataset.one_hot_label_encoder()

    inception_graph = tf.Graph()
    inception_sess = tf.Session(graph=inception_graph)

    with inception_graph.as_default(), inception_sess.as_default():
        incept_model = inception.inception_model()

    with tf.Graph().as_default(), tf.Session().as_default() as sess:

        labels_path = tf.placeholder(dtype=tf.string)

        images_ds = tf.contrib.data.TextLineDataset(labels_path) \
            .skip(1) \
            .map(parse_row) \
            .map(read_example)

        labels_iter = images_ds.make_initializable_iterator()
        next_label = labels_iter.get_next()

        sess.run(labels_iter.initializer, feed_dict={labels_path: paths.LABELS})

        print('Writing ', tfrecords_path)

        bar = pyprind.ProgBar(13000, update_interval=1, width=60)

        augmenter = image_augmenter()

        with tf.python_io.TFRecordWriter(tfrecords_path, tf.python_io.TFRecordCompressionType.NONE) as writer:
            try:
                while True:
                    id, img, breed_label = sess.run(next_label)
                    one_hot_label = one_hot_encoder([breed_label]).reshape(-1).tolist()

                    def get_inception_ouput(img):
                        with inception_graph.as_default():
                            inception_output = incept_model(inception_sess, img).reshape(-1).tolist()
                        return inception_output
                        # print(inception_output.shape)

                    # print('writing %s - %s' % (len(img), breed_label))

                    images = [img]
                    images.extend(augmenter(img))

                    for image in images:
                        example = build_train_example(image, one_hot_label, breed_label, get_inception_ouput(image))
                        writer.write(example.SerializeToString())

                    bar.update()

            except tf.errors.OutOfRangeError:
                print('End of the dataset')

            writer.flush()
            writer.close()

        print('Finished')


def convert_test(tfrecords_path):
    inception_graph = tf.Graph()
    inception_sess = tf.Session(graph=inception_graph)

    with inception_graph.as_default(), inception_sess.as_default():
        incept_model = inception.inception_model()

    with tf.Graph().as_default() as g, tf.Session().as_default() as sess:

        labels_path = tf.placeholder(dtype=tf.string)

        images_ds = tf.contrib.data.Dataset.from_tensor_slices(tf.constant(tf.gfile.ListDirectory(paths.TEST_DIR))) \
            .map(read_test_example)

        labels_iter = images_ds.make_initializable_iterator()
        next_label = labels_iter.get_next()

        sess.run(labels_iter.initializer, feed_dict={labels_path: paths.LABELS})

        print('Writing ', tfrecords_path)

        with tf.python_io.TFRecordWriter(tfrecords_path, tf.python_io.TFRecordCompressionType.NONE) as writer:
            try:
                while True:
                    img, id = sess.run(next_label)

                    with inception_graph.as_default():
                        inception_output = incept_model(inception_sess, img).reshape(-1).tolist()
                        # print(inception_output.shape)

                    print('writing %s - %s' % (len(img), id))
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'id': bytes_feature(id.encode()),
                        consts.IMAGE_RAW_FIELD: bytes_feature(img),
                        consts.INCEPTION_OUTPUT_FIELD: float_feature(inception_output)}))

                    writer.write(example.SerializeToString())
            except tf.errors.OutOfRangeError:
                print('End of the dataset')

            writer.flush()

        print('Finished')


#if __name__ == '__main__':
    #convert_train(paths.TRAIN_TF_RECORDS)
    #convert_test(paths.TEST_TF_RECORDS)

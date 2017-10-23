import tensorflow as tf
import numpy as np
import os
import paths
import pandas as pd
import xml.etree.ElementTree
import images_to_tfrecords
import consts
import dataset
import inception

images_root_dir = os.path.join(paths.STANFORD_DS_DIR, 'Images')
annotations_root_dir = os.path.join(paths.STANFORD_DS_DIR, 'Annotation')


def parse_annotation(path):
    xml_root = xml.etree.ElementTree.parse(path).getroot()
    object = xml_root.findall('object')[0]
    name = object.findall('name')[0].text.lower()
    bound_box = object.findall('bndbox')[0]

    return {
        'breed': name,
        'bndbox_xmin': bound_box.findall('xmin')[0].text,
        'bndbox_ymin': bound_box.findall('ymin')[0].text,
        'bndbox_xmax': bound_box.findall('xmax')[0].text,
        'bndbox_ymax': bound_box.findall('ymax')[0].text
    }


def parse_image(breed_dir, filename):
    path = os.path.join(images_root_dir, breed_dir, filename + '.jpg')
    img_raw = open(path, 'r').read()

    return img_raw


def build_stanford_example(img_raw, inception_output, one_hot_label, annotation):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': images_to_tfrecords._bytes_feature(annotation['breed'].encode()),
        consts.IMAGE_RAW_FIELD: images_to_tfrecords._bytes_feature(img_raw),
        consts.LABEL_ONE_HOT_FIELD: images_to_tfrecords._float_feature(one_hot_label),
        consts.INCEPTION_OUTPUT_FIELD: images_to_tfrecords._float_feature(inception_output)}))

    return example


if __name__ == '__main__':
    one_hot_encoder, _ = dataset.one_hot_label_encoder()

    with tf.Graph().as_default(), \
         tf.Session().as_default() as sess, \
            tf.python_io.TFRecordWriter(paths.STANFORD_DS_TF_RECORDS,
                                        tf.python_io.TFRecordCompressionType.NONE) as writer:

        incept_model = inception.inception_model()


        def get_inception_ouput(img):
            inception_output = incept_model(sess, img).reshape(-1).tolist()
            return inception_output


        for breed_dir in [d for d in os.listdir(annotations_root_dir)]:
            print(breed_dir)
            for annotation_file in [f for f in os.listdir(os.path.join(annotations_root_dir, breed_dir))]:
                print(annotation_file)
                annotation = parse_annotation(os.path.join(annotations_root_dir, breed_dir, annotation_file))

                # print(annotation)

                one_hot_label = one_hot_encoder([annotation['breed']]).reshape(-1).tolist()
                image = parse_image(breed_dir, annotation_file)

                example = build_stanford_example(image, get_inception_ouput(image), one_hot_label, annotation)

                writer.write(example.SerializeToString())

        writer.flush()
        writer.close()

        print('Finished')

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


def get_files(file_dir):
    '''
    :param file_dir:
    :return: list of images and labels
    '''
    image_list = []
    label_list = []
    dir_num = 0
    for dir_name in os.listdir(file_dir):
        for file in os.listdir(os.path.join(file_dir, dir_name)):
            image_list.append(os.path.join(file_dir, dir_name, file))
            label_list.append(dir_num)
        dir_num = dir_num + 1
    print("Total %d data, %d class" % (len(image_list), dir_num))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(num) for num in label_list]

    return image_list, label_list


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecord(images, labels, save_name):
    '''
    :param images: list type
    :param labels: list type
    :param save_dir_name: dir+name.tfrecords
    :return:
    '''
    sample_num = len(labels)
    writer = tf.python_io.TFRecordWriter(save_name)
    print("\nTransform start.....")
    print("Total %d data" % sample_num)
    for i in range(sample_num):
        try:
            image = plt.imread(images[i])
            image_raw = image.tobytes()
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": int64_feature(label),
                "image_raw": bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
            if (i + 1) % 1000 == 0:
                print("Process %d finished" % (i + 1))
        except IOError:
            print("Could not read:", images[i])
    writer.close()
    print("Transform done!")


def read_tfrecord(tfrecords_file, batch_size, image_H, image_W):
    '''
    :param tfrecords_file:
    :param batch_size:
    :return: image_batch and label_batch
    '''
    # make an input queue from tfrecord file
    input_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(input_queue)
    data_features = tf.parse_single_example(serialized=serialized_example,
                                            features={
                                                "label": tf.FixedLenFeature([], tf.int64),
                                                "image_raw": tf.FixedLenFeature([], tf.string)
                                            })
    image = tf.decode_raw(data_features["image_raw"], tf.uint8)
    label = tf.cast(data_features["label"], tf.int32)

    image = tf.reshape(image, [224, 224, 3])
    image = tf.cast(image, tf.float32) / 255.0

    ###################################
    # data argumentation should go here
    ###################################
    image = tf.image.resize_images(image, [image_H, image_W])
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size)

    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch

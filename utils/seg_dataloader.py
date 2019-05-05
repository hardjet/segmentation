# import pdb
import numpy as np
import tensorflow as tf
# import random

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor


class SegDataLoader(object):
    def __init__(self, main_dir, batch_size, resize_shape, paths_file, buffer_size=100, split='train'):
        self.main_dir = main_dir
        self.resize_shape = resize_shape
        self.paths_file = paths_file

        self.imgs_files = []
        self.labels_files = []

        # Read image and label paths from file and fill in self.images, self.labels
        self.parse_file(self.paths_file)
        self.shuffle_lists()

        img = convert_to_tensor(self.imgs_files, dtype=dtypes.string)
        label = convert_to_tensor(self.labels_files, dtype=dtypes.string)
        data_tr = tf.data.Dataset.from_tensor_slices((img, label))
        self.data_len = len(self.imgs_files)

        if split == 'train':
            data_tr = data_tr.map(self.parse_train, num_parallel_calls=16)  #, num_threads=8, output_buffer_size=100*self.batch_size)
        else:
            data_tr = data_tr.map(self.parse_val, num_parallel_calls=16)  #, num_threads=8, output_buffer_size=100*self.batch_size)

        data_tr = data_tr.shuffle(buffer_size)
        self.data_tr = data_tr.batch(batch_size)

    def shuffle_lists(self):
        imgs = self.imgs_files
        labels = self.labels_files

        permutation = np.random.permutation(len(self.imgs_files))
        self.imgs_files = []
        self.labels_files = []
        for i in permutation:
            self.imgs_files.append(imgs[i])
            self.labels_files.append(labels[i])

    def parse_train(self, im_path, label_path):
        # Load image
        img = tf.read_file(im_path)
        img = tf.image.decode_png(img, channels=3)
        # last_image_dim = tf.shape(img)[-1]

        # Load label
        label = tf.read_file(label_path)
        label = tf.image.decode_png(label, channels=1)

        # Scale
        img = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.BICUBIC)
        label = tf.image.resize_images(label, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return img, label

    def parse_val(self, im_path, label_path):
        # Load image
        img = tf.read_file(im_path)
        img = tf.image.decode_png(img, channels=3)
        # last_image_dim = tf.shape(img)[-1]

        # Load label
        label = tf.read_file(label_path)
        label = tf.image.decode_png(label, channels=1)

        # Scale
        img = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.BICUBIC)
        label = tf.image.resize_images(label, self.resize_shape, method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return img, label

    def parse_file(self, path):
        ff = open(path, 'r')
        for line in ff:
            tokens = line.strip().split(' ')
            self.imgs_files.append(self.main_dir+tokens[0])
            tokens[1] = tokens[1].replace('labelIds', 'labelIds_proc')
            self.labels_files.append(self.main_dir+tokens[1])

    def print_files(self):
        for x, y in zip(self.imgs_files, self.labels_files):
            print(x, y)


if __name__ == "__main__":
    # import cv2
    # import matplotlib.pyplot as plt
    # import scipy

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    with tf.device('/cpu:0'):
        segdl = SegDataLoader('/home/anson/sda/dataset/cityscapes/', 4, (512, 1024), 'trainImage.txt')
        # segdl = SegDataLoader('/home/anson/sda/dataset/cityscapes/', 4, (512, 1024), 'val.txt', split='val')

        iterator = tf.data.Iterator.from_structure(segdl.data_tr.output_types, segdl.data_tr.output_shapes)
        next_batch = iterator.get_next()

        training_init_op = iterator.make_initializer(segdl.data_tr)
        session.run(training_init_op)

    for i in range(10):
        img_batch, label_batch = session.run(next_batch)



"""
Trainer class to train Segmentation models
"""

from test.basic_test import BasicTest
from metrics.metrics import Metrics
from utils.reporter import Reporter
from utils.misc import timeit
from utils.average_meter import FPSMeter

from tqdm import tqdm
import numpy as np
# import tensorflow as tf
import time
# import h5py
# import pickle
# from utils.augmentation import flip_randomly_left_right_image_with_annotation, \
#     scale_randomly_image_with_annotation_with_fixed_size_output
import scipy.misc as misc
from utils.img_utils import decode_labels
# from utils.seg_dataloader import SegDataLoader
from utils.postprocess import postprocess
import os
#
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt


class Test(BasicTest):
    """
    Trainer class
    """
    name = 'Test'

    def __init__(self, args, sess, model):
        """
        Call the constructor of the base class
        init summaries
        init loading data
        :param args:
        :param sess:
        :param model:
        :return:
        """
        super().__init__(args, sess, model)
        # Init load data and generator
        self.generator = None
        self.run = None

        # 加载数据
        if self.args.data_mode == "realsense":
            self.test_data = None
            self.test_data_len = None
            self.num_iterations_testing_per_epoch = None
            self.load_realsence_data()
        elif self.args.data_mode == "cityscapes_val":
            self.test_data = None
            self.test_data_len = None
            self.num_iterations_testing_per_epoch = None
            self.load_val_data()
        elif self.args.data_mode == "cityscapes_test":
            self.test_data = None
            self.test_data_len = None
            self.num_iterations_testing_per_epoch = None
            self.load_test_data()
        elif self.args.data_mode == "video":
            self.test_data = None
            self.test_data_len = None
            self.num_iterations_testing_per_epoch = None
            self.load_vid_data()

        if self.args.task == "test":
            self.run = self.test
        elif self.args.task == "realsense":
            self.run = self.realsense_inference
        elif self.args.task == "realsense_imgs":
            self.run = self.realsense_imgs
        else:
            print("ERROR Please select a proper data_mode BYE")
            exit(-1)

        # Init metrics class
        self.metrics = Metrics(self.args.num_classes)
        # Init reporter class
        self.reporter = Reporter(self.args.out_dir + 'report_test.json', self.args)

    def resize(self, data):
        X = []
        Y = []
        for i in range(data['X'].shape[0]):
            X.append(misc.imresize(data['X'][i, ...], (self.args.img_height, self.args.img_width)))
            Y.append(misc.imresize(data['Y'][i, ...], (self.args.img_height, self.args.img_width), 'nearest'))
        data['X'] = np.asarray(X)
        data['Y'] = np.asarray(Y)
        return data

    @timeit
    def load_vid_data(self):
        print("Loading Video data..")
        self.test_data = {'X': np.load(self.args.data_dir + "X_vid.npy")}
        self.test_data['Y'] = np.zeros(self.test_data['X'].shape[:3])
        self.test_data_len = self.test_data['X'].shape[0]
        print("Vid-shape-x -- " + str(self.test_data['X'].shape))
        print("Vid-shape-y -- " + str(self.test_data['Y'].shape))
        self.num_iterations_testing_per_epoch = (self.test_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("Video data is loaded")

    @timeit
    def load_val_data(self):
        print("Loading Validation data..")
        self.test_data = {'X': np.load(self.args.data_dir + "X_val.npy"),
                          'Y': np.load(self.args.data_dir + "Y_val.npy")}
        self.test_data = self.resize(self.test_data)
        self.test_data['Y_large'] = self.test_data['Y']

        self.test_data_len = self.test_data['X'].shape[0] - self.test_data['X'].shape[0] % self.args.batch_size
        print("Validation-shape-x -- " + str(self.test_data['X'].shape))
        print("Validation-shape-y -- " + str(self.test_data['Y'].shape))
        self.num_iterations_testing_per_epoch = (self.test_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("Validation data is loaded")

    @timeit
    def load_realsence_data(self):
        print("Loading RealSense data..")
        self.test_data = {'X': np.load(self.args.data_dir + "/realsense/x_inference.npy"),
                          'names': np.load(self.args.data_dir + "/realsense/name_inference.npy")}

        self.test_data_len = self.test_data['X'].shape[0] - self.test_data['X'].shape[0] % self.args.batch_size
        print("RealSense-shape-x -- " + str(self.test_data['X'].shape))
        print("RealSense-shape-name -- " + str(self.test_data['names'].shape))
        self.num_iterations_testing_per_epoch = (self.test_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("RealSense data is loaded")

    @timeit
    def load_test_data(self):
        print("Loading Testing data..")
        self.test_data = {'X': np.load(self.args.data_dir + "X_test.npy")}
        self.names_mapper = {'X': np.load(self.args.data_dir + "xnames_test.npy"),
                             'Y': np.load(self.args.data_dir + "ynames_test.npy")}
        self.test_data_len = self.test_data['X'].shape[0] - self.test_data['X'].shape[0] % self.args.batch_size
        print("Test-shape-x -- " + str(self.test_data['X'].shape))
        self.num_iterations_testing_per_epoch = (self.test_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("Test data is loaded")

    def test_generator(self):
        start = 0
        new_epoch_flag = True
        idx = None
        while True:
            # init index array if it is a new_epoch
            if new_epoch_flag:
                if self.args.shuffle:
                    idx = np.random.choice(self.test_data_len, self.test_data_len, replace=False)
                else:
                    idx = np.arange(self.test_data_len)
                new_epoch_flag = False

            # select the mini_batches
            mask = idx[start:start + self.args.batch_size]
            x_batch = self.test_data['X'][mask]
            y_batch = self.test_data['Y'][mask]

            # update start idx
            start += self.args.batch_size

            if start >= self.test_data_len:
                start = 0
                new_epoch_flag = True

            yield x_batch, y_batch

    @staticmethod
    def linknet_postprocess(gt):
        gt2 = gt - 1
        gt2[gt == -1] = 19
        return gt2

    def test(self, pkl=False):
        print("Testing will begin NOW..")

        # init tqdm and get the epoch value
        tt = tqdm(range(self.test_data_len))
        # naming = np.load(self.args.data_dir + 'names_train.npy')

        # init acc and loss lists
        acc_list = []
        img_list = []

        # idx of image
        idx = 0

        # reset metrics
        self.metrics.reset()

        # loop by the number of iterations
        for cur_iteration in tt:
            # load mini_batches
            x_batch = self.test_data['X'][idx:idx + 1]
            y_batch = self.test_data['Y'][idx:idx + 1]

            idx += 1

            # Feed this variables to the network
            if self.args.random_cropping:
                feed_dict = {self.model.x_pl_before: x_batch,
                             self.model.y_pl_before: y_batch,
                             self.model.is_training: False,
                             }
            else:
                feed_dict = {self.model.x_pl: x_batch,
                             self.model.y_pl: y_batch,
                             self.model.is_training: False
                             }

            # run the feed_forward
            out_argmax, acc, segmented_imgs = self.sess.run(
                [self.model.out_argmax, self.model.accuracy,
                 # self.model.merged_summaries, self.model.segmented_summary],
                 self.model.segmented_summary],
                feed_dict=feed_dict)

            if pkl:
                out_argmax[0] = self.linknet_postprocess(out_argmax[0])
                segmented_imgs = decode_labels(out_argmax, 20)

            # print('mean preds ', out_argmax.mean())
            # np.save(self.args.out_dir + 'npy/' + str(cur_iteration) + '.npy', out_argmax[0])

            misc.imsave(self.args.out_dir + 'imgs/' + 'test_' + str(cur_iteration) + '.png', segmented_imgs[0])

            # log loss and acc
            acc_list += [acc]

            # log metrics
            if self.args.random_cropping:
                y1 = np.expand_dims(y_batch[0, :, :512], axis=0)
                y2 = np.expand_dims(y_batch[0, :, 512:], axis=0)
                y_batch = np.concatenate((y1, y2), axis=0)
                self.metrics.update_metrics(out_argmax, y_batch, 0, 0)
            else:
                self.metrics.update_metrics(out_argmax[0], y_batch[0], 0, 0)

        # mean over batches
        total_loss = 0
        total_acc = np.mean(acc_list)
        mean_iou = self.metrics.compute_final_metrics(self.test_data_len)

        # print in console
        tt.close()
        print("Here the statistics")
        print("Total_loss: " + str(total_loss))
        print("Total_acc: " + str(total_acc)[:6])
        print("mean_iou: " + str(mean_iou))

        print("Plotting imgs")
        for i in range(len(img_list)):
            misc.imsave(self.args.imgs_dir + 'test_' + str(i) + '.png', img_list[i])

    def realsense_imgs(self):
        print("realsense_imgs will begin NOW..")

        # init tqdm and get the epoch value
        tt = tqdm(range(self.test_data_len))

        # idx of image
        idx = 0

        # loop by the number of iterations
        for cur_iteration in tt:
            # load mini_batches
            x_batch = self.test_data['X'][idx:idx + 1]

            idx += 1
            feed_dict = {self.model.x_pl: x_batch,
                         self.model.is_training: False
                         }

            # run the feed_forward
            segmented_imgs = self.sess.run([self.model.segmented_summary], feed_dict=feed_dict)

            # plt.imsave(self.args.out_dir + 'imgs/' + 'test_' + str(cur_iteration) + '.png', segmented_imgs[0][0])
            misc.imsave(self.args.out_dir + 'imgs/' + 'test_' + str(cur_iteration) + '.png', segmented_imgs[0][0])

        tt.close()
        print("realsense_imgs finished~")

    def test_eval(self, pkl=False):
        print("Testing will begin NOW..")

        # load the best model checkpoint to test on it
        if not pkl:
            self.load_best_model()

        # init tqdm and get the epoch value
        tt = tqdm(range(self.test_data_len))

        # idx of image
        idx = 0

        # loop by the number of iterations
        for cur_iteration in tt:
            # load mini_batches
            x_batch = self.test_data['X'][idx:idx + 1]

            # Feed this variables to the network
            if self.args.random_cropping:
                feed_dict = {self.model.x_pl_before: x_batch,
                             self.model.is_training: False,
                             }
            else:
                feed_dict = {self.model.x_pl: x_batch,
                             self.model.is_training: False
                             }

            # run the feed_forward
            out_argmax, segmented_imgs = self.sess.run(
                [self.model.out_argmax,
                 self.model.segmented_summary],
                feed_dict=feed_dict)

            if pkl:
                out_argmax[0] = self.linknet_postprocess(out_argmax[0])
                segmented_imgs = decode_labels(out_argmax, 20)

            # Colored results for visualization
            colored_save_path = self.args.out_dir + 'imgs/' + str(self.names_mapper['Y'][idx])
            if not os.path.exists(os.path.dirname(colored_save_path)):
                os.makedirs(os.path.dirname(colored_save_path))
            misc.imsave(colored_save_path, segmented_imgs[0])

            # Results for official evaluation
            save_path = self.args.out_dir + 'results/' + str(self.names_mapper['Y'][idx])
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            output = postprocess(out_argmax[0])
            misc.imsave(save_path, misc.imresize(output, [1024, 2048], 'nearest'))

            idx += 1

        # print in console
        tt.close()

    def realsense_inference(self):
        print("INFERENCE will begin NOW..")

        # init tqdm and get the epoch value
        tt = tqdm(range(self.test_data_len))

        # idx of image
        idx = 0

        # create the FPS Meter
        fps_meter = FPSMeter()

        # loop by the number of iterations
        for cur_iteration in tt:
            # load mini_batches
            x_batch = self.test_data['X'][idx:idx + 1]
            # y_batch = self.test_data['Y'][idx:idx + 1]

            # update idx of mini_batch
            idx += 1

            # Feed this variables to the network
            feed_dict = {self.model.x_pl: x_batch,
                         self.model.is_training: False}

            # calculate the time of one inference
            start = time.time()

            # run the feed_forward
            _ = self.sess.run(
                [self.model.out_argmax],
                feed_dict=feed_dict)

            # update the FPS meter
            fps_meter.update(time.time() - start)

        fps_meter.print_statistics()

    def test_inference(self):
        """
        Like the testing function but this one is for calculate the inference time
        and measure the frame per second
        """
        print("INFERENCE will begin NOW..")

        # init tqdm and get the epoch value
        tt = tqdm(range(self.test_data_len))

        # idx of image
        idx = 0

        # create the FPS Meter
        fps_meter = FPSMeter()

        # loop by the number of iterations
        for cur_iteration in tt:
            # load mini_batches
            x_batch = self.test_data['X'][idx:idx + 1]
            y_batch = self.test_data['Y'][idx:idx + 1]

            # update idx of mini_batch
            idx += 1

            # Feed this variables to the network
            if self.args.random_cropping:
                feed_dict = {self.model.x_pl_before: x_batch,
                             self.model.y_pl_before: y_batch,
                             self.model.is_training: False,
                             }
            else:
                feed_dict = {self.model.x_pl: x_batch,
                             self.model.y_pl: y_batch,
                             self.model.is_training: False
                             }

            # calculate the time of one inference
            start = time.time()

            # run the feed_forward
            _ = self.sess.run(
                [self.model.out_argmax],
                feed_dict=feed_dict)

            # update the FPS meter
            fps_meter.update(time.time() - start)

        fps_meter.print_statistics()

    def finalize(self):
        self.reporter.finalize()


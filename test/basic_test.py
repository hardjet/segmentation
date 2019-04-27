"""
The Basic class to train any Model
"""

import tensorflow as tf
from utils.misc import timeit


class BasicTest(object):
    """
    A Base class for test classes of the models
    Contain all necessary functions for testing
    """

    def __init__(self, args, sess, model):
        print("\nTraining is initializing itself\n")

        self.args = args
        self.sess = sess
        self.model = model

        # shortcut for model params
        self.params = self.model.params

        # To initialize all variables
        self.init = None
        self.init_model()

        # Get the description of the graph
        # self.get_all_variables_in_graph()
        # exit(0)

        # Create a saver object
        self.saver_best = tf.train.Saver(max_to_keep=1,
                                         save_relative_paths=True)

        # Load from latest checkpoint if found
        if model is not None:
            self.load_best_model()

    @timeit
    def init_model(self):
        print("Initializing the variables of the model")
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        print("Initialization finished")

    def load_best_model(self):
        """
        Load the best model checkpoint
        :return:
        """
        print("loading a checkpoint for BEST ONE")
        latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_best_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver_best.restore(self.sess, latest_checkpoint)
        else:
            print("ERROR NO best checkpoint found")
            exit(-1)
        print("BEST MODEL LOADED..")

    def finalize(self):
        raise NotImplementedError("finalize function is not implemented in the trainer")

    @staticmethod
    def get_all_variables_in_graph():
        print('################### Variables of the graph')
        from collections import OrderedDict
        var_dict = OrderedDict()
        for var in tf.all_variables():
            if var.op.name not in var_dict.keys() and var.shape != () and "Adam" not in var.op.name:
                print("'" + str(var.op.name) + "': " + str(var.op.name).replace('/', '_') + "# " + str(var.shape))
                key = var.op.name
                # x = var.eval(self.sess)
                var_dict[key] = var.shape
        print('Finished Display all variables')

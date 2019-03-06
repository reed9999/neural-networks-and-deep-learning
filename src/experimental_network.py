#####
# See the README.md
# At the moment this is inadequately documented. In particular it's not obvious
# how to compare outcomes with the standard NN from Nielsen's original code
# with results following my "helps". For now you'd have to comment out around
# line 63, the call to initial_seed or initial_seed_for_numeral

# So far I've found no evidence that seeding the original weights has much
# appreciable effect on final performance. Indeed for the first 2-3 epochs it
# seems to hurt performance, but the numbers catch up.


from sys import argv
import os
# import cPickle as pickle  # Not in my venv -- maybe I chose another way to serialize?
import argparse
import numpy as np
import mnist_loader
import network, network2
import main_part_2

# DRY
DEVELOPMENT = True
ORIGINAL = True
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.join(THIS_FILE_DIR, '..')

training_data, validation_data, test_data = \
  mnist_loader.load_data_wrapper()

# end DRY

class ExperimentalNetwork(network.Network):
    DEFAULT_SHAPE = [784, 30, 10]
    def __init__(self, shape=None, output=None):
        self.shape = shape or self.DEFAULT_SHAPE
        network.Network.__init__(self, shape)

    def initial_seed_for_numeral(self, numeral):
        """
        Arbitrarily we take the first 10 neurons of the second layer
        and make them into a representation of whatever heatmap we want as a
        starting, biased state. For example, some idealized figure of what we
        expect the numeral to look like.

        :param net:
        :param numeral:
        :return:
        """
        try:
            input = os.path.join('..', 'experiments','starting-weights')
            try:

                canonical = np.loadtxt(os.path.join(input, 'canonical-{}.csv'.format(numeral)), delimiter=',')
            except ValueError:
                canonical = np.loadtxt(os.path.join(input, 'canonical-{}.csv'.format(numeral)), delimiter=',')
            reshaped = canonical.reshape(28 * 28, )
            self.weights[0][numeral] = reshaped
        except IOError:
            print ("""The numeral {} does not have a readable CSV file. Therefore we will fall
            back to standard behavior with random initial seeds.""".format(numeral))

    def initial_seed(self):
        for i in range(0,10):
            self.initial_seed_for_numeral(i)

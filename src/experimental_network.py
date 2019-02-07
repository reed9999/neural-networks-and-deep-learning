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

def initial_seed_for_numeral(net, numeral):
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
        try:

            canonical = np.loadtxt('../starting-weights/canonical-{}.csv'.format(numeral), delimiter=' ')
        except ValueError:
            canonical = np.loadtxt('../starting-weights/canonical-{}.csv'.format(numeral), delimiter=',')
        reshaped = canonical.reshape(28 * 28, )
        net.weights[0][numeral] = reshaped
    except IOError:
        print ("The numeral {} does not have a readable CSV file".format(
            numeral))



def initial_seed(net):
    for i in range(0,10):
        initial_seed_for_numeral(net, i)


#### Below are some less enduring attempts at seeding the weights.
def pjr_silly_seeding_eight(net):
    """enhancement of silly_seeding_seven reading from a file of the final dot product, with a bit of cleaning up."""
    second_level = output = 8
    canonical8 = np.loadtxt('../pjr-images/canonical-8.csv', delimiter=' ')
    reshaped8 = canonical8.reshape(28 * 28,)
    net.weights[0][second_level] = reshaped8


def pjr_silly_seeding_seven(net):
    #What I'm going to try to do is weight a few first-level pixel neurons that might be expected to make a seven,
    # making them feed into second level neuron 7 (arbitrary choice but may as well make it match), and weighting that one
    # bit higher for an actual output of 7. Point is I'm curious if starting with a prior makes it more predictive.

    second_level = output = 7
    upper_leftish_pixel = 9*28+9
    upper_rightish_pixel = 9*28+18
    lower_leftish_pixel = 18*28+9
    lower_rightish_pixel = 18*28+18
    #The -2 thru +2: just a lazy way to initialize more pixels w/o proper loop
    net.weights[0][second_level][upper_leftish_pixel-2:upper_leftish_pixel+2] = -1
    net.weights[0][second_level][upper_rightish_pixel-2:upper_rightish_pixel+2] = -1
    net.weights[0][second_level][lower_leftish_pixel-2:lower_leftish_pixel+2] = +1
    net.weights[0][second_level][lower_rightish_pixel-2:lower_rightish_pixel+2] = -1

def pjr_intermediate_level_seeds(net):
    #Does it help or hurt our predictions to give a strong hint via starting weights that
    # the first 10 intermediate neurons COULD be tuned to more or less correspond to output neurons?
    # In other words what role would 11 through 30 play? Would this ruin the point of NN?

    #LEARN TO DO THIS THE TRUELY NUMPYISH WAY
    for i in range(0,10):
        net.weights[1][i][0:10] = -1
        net.weights[1][i][i] = +1
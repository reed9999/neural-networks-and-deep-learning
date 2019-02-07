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
import cPickle as pickle
import argparse
import numpy as np
import mnist_loader
import network, network2
from network import Network as OriginalNetwork
from experimental_network import ExperimentalNetwork
import main_part_2

DEVELOPMENT = True
ORIGINAL = False
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.join(THIS_FILE_DIR, '..')

if os.getcwd() != THIS_FILE_DIR:
    raise RuntimeError("""For now, you have to start in the src directory.
    Although it appears to be legacy issue, it should really be remedied.""")
training_data, validation_data, test_data = \
  mnist_loader.load_data_wrapper()

def main():
    ALTERNATIVE_STRUCTURE = [784, 90, 30, 10]
    DEFAULTS = [784, 30, 10]

    if ORIGINAL:
        net = network.Network(DEFAULTS)
    else:
        net = ExperimentalNetwork(DEFAULTS)
    # net = network.Network([784, 30, 25, 10])

    if DEVELOPMENT:
        print ("""DEVELOPMENT=True

        This is intentionally a small number of epochs.run to troubleshoot or 
        verify functionality."""
               )
        num_epochs = 2
    else:
        # A real run...
        num_epochs = 30

    print(
        "Eventually it will be clearer how to tweak when we do the initial seed")
    print("For now we comment or uncomment initial_seed() or "
          "initial_seed_for_numeral()")
    # initial_seed(net)
    initial_seed_for_numeral(net, 8)
    net.SGD(training_data, num_epochs, 10, 3.0, test_data=test_data)

    grand_dot_product = net.grand_dot_product()

    # weights=net.weights
    # import numpy as np

    # Not 100% sure why I wanted to do this, but I think the idea was to
    # save off the dot product matrix so as to do analysis on it later without
    # rerunning the entire NN.


    pickle.dumps(grand_dot_product)
    with open('grand_dot_product.pkl', 'w') as f:
        pickle.dump(grand_dot_product, f)

    #pjr_intermediate_level_seeds(net)

    #an example...
    net.show_product_for_numeral(7)


def get_argparser():
    parser = argparse.ArgumentParser(prog='neural-networks-exploration',
                                     description='See README.md; no options being parsed yet.')
    return parser

if __name__ == '__main__':
    parser = get_argparser()
    main()

###
#I had added this code to network.py
# def pjr_graph(self, which=0):
#     # As best I can tell, each of these 30 nodes (of which I here graph 28 and 29
#     # corresponds to the 2nd dimension of weights[1], which means for example that if
#     # weights[1][4] has a high number for element 28, then the heatmap for 28 should show
#     # pixels that are likely associated with a 4. I don't know this.
#     a = np.resize(self.weights[0][28], (28, 28))
#     plt.imshow(a, cmap='hot', interpolation='nearest')
#     plt.show()
#     a = np.resize(self.weights[0][29], (28, 28))
#     plt.imshow(a, cmap='hot', interpolation='nearest')
#     plt.show()


"""
pre better canonical 8
Epoch 1: 9269 / 10000
Epoch 2: 9416 / 10000
Epoch 3: 9424 / 10000
Epoch 4: 9458 / 10000
Epoch 5: 9421 / 10000
Epoch 6: 9529 / 10000
Epoch 7: 9543 / 10000
Epoch 8: 9521 / 10000
Epoch 9: 9552 / 10000
Epoch 10: 9550 / 10000
Epoch 11: 9570 / 10000
Epoch 12: 9578 / 10000
Epoch 13: 9570 / 10000
Epoch 14: 9564 / 10000
Epoch 15: 9579 / 10000
Epoch 16: 9593 / 10000
Epoch 17: 9546 / 10000
Epoch 18: 9609 / 10000
Epoch 19: 9573 / 10000
Epoch 20: 9593 / 10000
Epoch 21: 9588 / 10000
Epoch 22: 9593 / 10000
Epoch 23: 9610 / 10000
Epoch 24: 9598 / 10000
Epoch 25: 9618 / 10000
Epoch 26: 9614 / 10000
Epoch 27: 9596 / 10000
Epoch 28: 9608 / 10000
Epoch 29: 9621 / 10000
"""

"""
with the better canonical 8
Epoch 0: 9112 / 10000
Epoch 1: 9308 / 10000
Epoch 2: 9408 / 10000
Epoch 3: 9479 / 10000
Epoch 4: 9410 / 10000
Epoch 5: 9489 / 10000
Epoch 6: 9520 / 10000


Epoch 7: 9512 / 10000
Epoch 8: 9566 / 10000
Epoch 9: 9470 / 10000
Epoch 10: 9567 / 10000
Epoch 11: 9548 / 10000
Epoch 12: 9507 / 10000
Epoch 13: 9567 / 10000
Epoch 14: 9580 / 10000
Epoch 15: 9602 / 10000
Epoch 16: 9580 / 10000
Epoch 17: 9608 / 10000
Epoch 18: 9589 / 10000
Epoch 19: 9593 / 10000
Epoch 20: 9609 / 10000
Epoch 21: 9617 / 10000
Epoch 22: 9617 / 10000
Epoch 23: 9627 / 10000
Epoch 24: 9615 / 10000
Epoch 25: 9611 / 10000
Epoch 26: 9624 / 10000
Epoch 27: 9619 / 10000
Epoch 28: 9628 / 10000
Epoch 29: 9615 / 10000

"""
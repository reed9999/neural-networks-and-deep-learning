#####
# See the README.md
# At the moment this is inadequately documented. In particular it's not obvious
# how to compare outcomes with the standard NN from Nielsen's original code
# with results following my "helps". For now you'd have to comment out around
# line 63, the call to initial_seed or initial_seed_for_numeral

# So far I've found no evidence that seeding the original weights has much
# appreciable effect on final performance. Indeed for the first 2-3 epochs it
# seems to hurt performance, but the numbers catch up.


import cPickle as pickle
import numpy as np
import mnist_loader
import network, network2
import main_part_2

training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

#net = network.Network([784, 30, 10])
net = network.Network([784, 90, 30, 10])
#net = network.Network([784, 30, 25, 10])

DEVELOPMENT = False
if DEVELOPMENT:
    print ("ATTENTION: Ridiculously small run to troubleshoot or verify functionality.")
    num_epochs = 5
else:
# A real run...
    num_epochs = 30


def initial_seed_for_numeral(net, numeral):
    """
    Arbitrarily we take the first 10 neurons of the second layer
    and make them into a representation of whatever heatmap we want as a
    starting, biased state. For example, some idealized figure of what we
    expect the numeral to look like.

    :param net:
    :param numeral:
    :return:

    What's hilarious: The numbers are abysmal.
    Epoch 0: 8219 / 10000
Epoch 1: 9217 / 10000
Epoch 2: 9255 / 10000
Epoch 3: 9333 / 10000
Epoch 4: 9322 / 10000
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

print("Eventually it will be clearer how to tweak when we do the initial seed")
print("For now we comment or uncomment initial_seed() or "
      "initial_seed_for_numeral()")
#initial_seed(net)
initial_seed_for_numeral(net, 8)
net.SGD(training_data, num_epochs, 10, 3.0, test_data=test_data)

grand_dot_product=net.grand_dot_product()


#weights=net.weights
#import numpy as np

pickle.dumps(grand_dot_product)
with open('grand_dot_product.pkl', 'w') as f:
    pickle.dump(grand_dot_product, f)


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

#pjr_intermediate_level_seeds(net)

#an example...
net.show_product_for_numeral(7)


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
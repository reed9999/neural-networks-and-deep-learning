import numpy as np
import mnist_loader
import network, network2
import pjr_more

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
    What I'm doing here is rather arbitrarily taking the first 10 neurons of the 30 (or whatever) in the second layer
    and making them into a representation of our heatmap, or of my tweaked heatmap,
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

            canonical = np.loadtxt('../pjr-images/canonical-{}.csv'.format(numeral), delimiter=' ')
        except ValueError:
            canonical = np.loadtxt('../pjr-images/canonical-{}.csv'.format(numeral), delimiter=',')
        reshaped = canonical.reshape(28 * 28, )
        net.weights[0][numeral] = reshaped
    except IOError:
        print ("Oh well, {} doesn't have a file yet".format(numeral))



def initial_seed(net):
    for i in range(0,10):
        initial_seed_for_numeral(net, i)

#initial_seed(net)
initial_seed_for_numeral(net, 8)
net.SGD(training_data, num_epochs, 10, 3.0, test_data=test_data)

grand_dot_product=net.grand_dot_product()


#weights=net.weights
#import numpy as np

import cPickle as pickle
pickle.dumps(grand_dot_product)
with open('grand_dot_product.pkl', 'w') as f:
    pickle.dump(grand_dot_product, f)


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
    #The -2 thru +2 is just a way to initialize more pixels without a proper loop because I'm lazy
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

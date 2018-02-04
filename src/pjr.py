import mnist_loader
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

import network, network2
import pjr_more

net = network.Network([784, 30, 10])
#net = network.Network([784, 30, 25, 10])

#come back to this concept
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

pjr_intermediate_level_seeds(net)
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)



weights=net.weights
import numpy as np
#grand_dot_product=np.dot(np.dot(weights[2],weights[1]),weights[0])
grand_dot_product=np.dot(weights[1],weights[0])

import cPickle as pickle
pickle.dumps(grand_dot_product)
with open('grand_dot_product.pkl', 'w') as f:
    pickle.dump(grand_dot_product, f)


for i in range(0,net.num_layers-1):
    print "Shape of the {0}th weight array is {1}".format(i, weights[i].shape)
# x=w[1]



numeral_of_interest=7
a=np.resize(grand_dot_product[numeral_of_interest],(28,28))
plt.imshow(a, cmap='hot', interpolation='nearest'); plt.show()
# a=np.resize(p[2],(28,28))
# plt.imshow(a, cmap='hot', interpolation='nearest'); plt.show()



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

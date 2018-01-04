import mnist_loader
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

import network
#net = network.Network([784, 30, 10])
net = network.Network([784, 30, 25, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)



weights=net.weights
for i in range(0,3):
    print "Shape of the {0}th array is {1}".format(i, weights[i].shape)
# x=w[1]
import numpy as np
grand_dot_product=np.dot(np.dot(weights[2],weights[1]),weights[0])
numeral_of_interest=0
a=np.resize(grand_dot_product[numeral_of_interest],(28,28))
plt.imshow(a, cmap='hot', interpolation='nearest'); plt.show()
# a=np.resize(p[2],(28,28))
# plt.imshow(a, cmap='hot', interpolation='nearest'); plt.show()
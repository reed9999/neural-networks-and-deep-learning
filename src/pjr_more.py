import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt


def main():
    gdp = None
    with open('grand_dot_product.pkl', 'r') as f:
        gdp = pickle.load(f)
    print(gdp)
    numeral_of_interest = 7
    a = np.resize(gdp[numeral_of_interest], (28, 28))
    plt.imshow(a, cmap='hot', interpolation='nearest');
    plt.show()


if __name__=="__main__":
    main()
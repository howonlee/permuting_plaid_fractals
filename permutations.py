import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

def rand_permutation(size):
    arr = np.eye(size)
    # apply knuth shuffle?
    return arr

def total_shuffle(arr):
    pass

def kron_net(order):
    pass

def sample_net(arr):
    """
    Sample a network from a network ensemble
    """
    new_arr = np.zeros_like(arr)
    for x in xrange(arr.shape[0]):
        for y in xrange(arr.shape[1]):
            if npr.rand() < arr[x,y]:
                new_arr[x,y] = 1
    return new_arr

if __name__ == "__main__":
    pass

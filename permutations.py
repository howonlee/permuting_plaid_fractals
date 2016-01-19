import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

def rand_permutation(size):
    arr = np.identity(size)
    npr.shuffle(arr) # inplace
    return arr

def total_shuffle(arr):
    new_arr = arr.copy()
    npr.shuffle(new_arr.flat)
    return new_arr

def kron_net(order):
    generator = np.array([[0.99, 0.7], [0.7, 0.15]])
    arr = generator.copy()
    for x in xrange(order-1):
        arr = np.kron(arr, generator)
    return arr

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
    net = kron_net(11)
    l_perm_mat = rand_permutation(2 ** 11)
    r_perm_mat = rand_permutation(2 ** 11)
    net = np.dot(l_perm_mat, net)
    net = np.dot(net, r_perm_mat)
    net = sample_net(net)
    # net = total_shuffle(net)
    plt.plot(sorted(net.sum(axis=0)))
    #plt.imshow(net)
    #plt.colorbar()
    plt.show()

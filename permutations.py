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

def row_col_shuffle_experiment():
    print "This is to show that row and column shuffling will get you something like a plaid network"
    print "If that previous statement is actually true, the task of finding a gradient with the fractal hypothesis is finding and applying a permutation of those two shuffles"
    print "But getting those two lists to obey a permutation, by the decision tree analysis, is O(n log n)"
    print "How much time is finding the permutation in the first place? I do not know, but it can't be that much, can it?"
    net = kron_net(10)
    l_perm_mat = rand_permutation(2 ** 10)
    r_perm_mat = rand_permutation(2 ** 10)
    net = np.dot(l_perm_mat, net)
    net = np.dot(net, r_perm_mat)
    plt.close()
    plt.plot(sorted(net.sum(axis=0)))
    plt.title("inequality of sorted ensemble weight")
    plt.xlabel("ensemble weight rank")
    plt.ylabel("ensemble weight")
    plt.show()
    plt.close()
    plt.imshow(net)
    plt.title("view of unsampled plaid matrix ensemble")
    plt.show()
    plt.close()
    print "if you want to sample the ensemble and look at it that way, there's a function in the code"

def total_shuffle_experiment():
    print "This is to show that _total_ shuffling, eg., construing the matrix as a vector and shuffling the members of the vector, loses you the nice properties of the fractal matrix"
    net = kron_net(10)
    net = total_shuffle(net)
    plt.close()
    plt.plot(sorted(net.sum(axis=0)))
    plt.title("inequality of sorted ensemble weight")
    plt.xlabel("ensemble weight rank")
    plt.ylabel("ensemble weight")
    plt.show()
    plt.close()
    plt.imshow(net)
    plt.title("view of unsampled shuffled 'ensemble'")
    plt.show()
    plt.close()

if __name__ == "__main__":
    row_col_shuffle_experiment()
    total_shuffle_experiment()

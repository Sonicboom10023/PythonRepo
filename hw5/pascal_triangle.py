#!/usr/bin/env python3
# CS103; john k johnstone; jkj at uab dot edu; mit license

import numpy as np

def pascal (n):
    """ Build a Pascal triangle of (n k),
    where (n k) is the number of ways of choosing a subset of size k
    from a set of size n.
    A formula for (n k) is n!/(k!(n-k)!), 
    but we will compute it more robustly
    using an incremental algorithm, using just addition.

    Params: n (int): # of rows of the Pascal triangle to define
    Returns: (np.ndarray of shape (n,n)) Pascal triangle P;
                                         P[i,j] = (i j) for j<=i
    """

    P = np.zeros ((n,n), dtype=np.int32)
    for i in range(n): # define the first and last element of each row
        P[i,0] = 1 # one way to choose nothing (0 elt from set of size n)
        P[i,i] = 1 # one way to choose entire set (n elt from set of size n)
    for i in range(1,n):
        for j in range(1,i):
            P[i,j] = P[i-1,j-1] + P[i-1,j]
    return P

# print (pascal (10))
print (pascal (30))

# try computing it naively using factorial
# if we have time later, I can do more examples
# another numpy example: de Casteljau triangle 
# another numpy example: read a point cloud matrix and draw it 
# in turtle and in pyglet

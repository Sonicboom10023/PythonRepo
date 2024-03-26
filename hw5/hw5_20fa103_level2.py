#!/usr/bin/env python3
# CS103; john k johnstone; jkj at uab dot edu; mit license

# only 4 modules are necessary, and only 4 are allowed: numpy/turtle/random/math
import numpy as np
from turtle import *
import random
import math

def randomPt (BL, TR):

    """Build a random point in the cell
    with bottom left corner and top right corner specified.
    Params:
        BL (point as np.ndarray of shape (2,)): bottom left corner of cell
        TR (point as np.ndarray of shape (2,)): top right corner of cell
    Returns: (np.ndarray)
    """

    assert type(BL) == np.ndarray and len(BL) == 2 
    assert type(TR) == np.ndarray and len(TR) == 2
    # ADD CODE HERE
    a = np.zeros(2)
    x1 = BL[0:1]
    x2 = TR[0:1]
    y1 = BL[1:]
    y2 = TR[1:]
    x3 = random.randrange(x1,x2)
    y3 = random.randrange(y1,y2)
    a = np.array([x3,y3])
    return a

def randomTri (i,j,n):

    """ Build a random triangle in the cell (i,j),
    where top left cell is (0,0), like a matrix.
    Let's define the top left point of the grid to have turtle coordinates
    (-320,320) and bottom right point of the grid as (320,-320).
    What is the left   side of the (i,j) cell? 
    What is the right  side of the (i,j) cell? 
    What is the top    side of the (i,j) cell? 
    What is the bottom side of the (i,j) cell? 
    This will help you define the bottom left corner and top right corner
    of the (i,j) cell, as needed by randomPt.
    Hint: matrix coordinates (i,j) behave differently 
    from Cartesian coordinates (x,y).

    Params:
        i: index of row of cell    (0 <= i < n)
        j: index of column of cell (0 <= j < n)
        n: grid has n rows and n columns of cells (grid is n x n)
    Returns: (np.ndarray of shape (6,)) 3 points of the random triangle
                         array([x1 y1 x2 y2 x3 y3])
    """
    
    assert n%2 == 0
    assert i>=0 and i<n
    assert j>=0 and j<n
    a = np.zeros((6))
    w = 640/n
    w = int(w)
    BL = np.array( [  [(-320 + j*w)], [(320 -(i+1)*w) ] ])
    TR = np.array( [ [-320 + (j+1)*w], [320 -i*w] ] )
    x = 0
    for x in range(0,5,2):
        P = randomPt(BL,TR)
        if a[x] == 0 and a[x+1] == 0:
            a[x] = P[0]
            a[x+1] = P[1]
    return a

def buildTriData (n, tri_data):

    """Build n^2 random triangles, one in each cell of an nxn grid,
    and store these n^2 triangles in a data matrix, one per row.
    The triangles are stored in a data matrix format.
    Each row of the matrix represents a triangle by 6 floats 
    x1 y1 x2 y2 x3 y3
    representing the 3 vertices (x1,y1), (x2,y2), (x3,y3) of a triangle.

    Store the triangles row by row, left to right (row-major order).
    For example, for 4 triangles in a 2x2 grid, 
    the first row of the data matrix stores the top left triangle, 
    the second row stores the top right triangle, 
    the third row stores the bottom left triangle, 
    and the 4th row stores the bottom right triangle.
    
    Build one triangle per cell, so that the triangles do not overlap.
    The grid is nxn.  Each cell is a square of the same size as the other cells.
    The triangle in a cell is built from 3 random points in that cell.

    Useful function: randomRobustTri.

    Params: 
        n (int) grid is nxn, one triangle per cell of this grid
                n is even (to make cell sizes simpler
        tri_data (np.ndarray of shape (n*n, 6)) triangle data matrix
               enters empty (full of 0's), leaves populated with triangles
               (leveraging aliasing)
    """

    assert n%2 == 0 
    assert type(tri_data) == np.ndarray and tri_data.shape == (n*n,6)
    # ADD CODE HERE

def writeData (fn, data_matrix):

    """Write a data matrix to a file, prefacing it by its shape.
    Params: 
        fn (string): filename 
        data_matrix (np.ndarray of shape (m,n)) data matrix 
    """

    assert type(fn) == str
    assert type(data_matrix) == np.ndarray
    # ADD CODE HERE

def writeTriData (fn, tri_data):

    """Write a triangle data matrix to a file, prefacing it by its shape.
    Params: 
        fn (string): filename 
        tri_data (np.ndarray of shape n^2 x 6) triangle data matrix 
              (with the same structure as the output of buildTriData)
    """

    assert type(fn) == str
    assert type(tri_data) == np.ndarray and len(tri_data.shape) == 2
    assert tri_data.shape[1] == 6

    # ADD CODE HERE

def readTriData (fn):

    """Read a triangle data matrix from a file.
    Params: fn (string): filename 
    Returns: (np.ndarray of shape (n*n,6)) triangle data matrix,
             with the same structure as the output of buildTriData
    """
    
    assert type(fn) == str
    # ADD CODE HERE

def drawTri (v):

    """Draw a single triangle.
    Note that this is different than the earlier triangle drawing functions,
    since the input parameter is a row of a triangle data matrix.

    Params: v (np.ndarray of shape (6,)) triangle as x1 y1 x2 y2 x3 y3
    """
    
    assert type(v) == np.ndarray and len(v) == 6
    # ADD CODE HERE

def drawManyTri (tri_data):

    """Draw all the triangles in a triangle data matrix.
    Hint: use slicing.
    Returns: (np.ndarray of shape (n*n,6)) triangle data matrix
    """

    assert type(tri_data) == np.ndarray and len(tri_data.shape) == 2
    assert tri_data.shape[1] == 6
    # ADD CODE HERE

def driver (n):

    """Driver routine.
    Builds and draws a grid of triangles.
    Also tests your read/write functions.
    Params: n (int) width of the square grid, even number 
    """

    # DO NOT CHANGE THIS FUNCTION: IT IS COMPLETE
    assert n%2 == 0
    tri_data_1 = np.zeros ((n*n, 6)) # empty, will be filled by buildTriData
    buildTriData (n, tri_data_1)
    writeTriData ('tri.txt', tri_data_1)
    tri_data = readTriData ('tri.txt')
    assert np.array_equal (tri_data_1, tri_data)
    drawManyTri (tri_data)

# uncomment once you have implemented
# driver(2)
# driver(4)

# temporarily comment out some calls in driver
# to build up gradually;
# test individual functions

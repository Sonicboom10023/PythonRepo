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
    a = np.array([x3,y3],dtype=np.int32)
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
    # ADD CODE HERE
    a = np.zeros((6), dtype=np.int32)
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
    x = 0
    c = 0
    for i in range(0,n):
        for j in range(0,n):
            a = randomTri(i,j,n)
            for x in range(len(a)):
                tri_data[c, x] = a[x]
            c+=1
    

def drawTri (v):

    """Draw a single triangle.
    Note that this is different than the earlier triangle drawing functions,
    since the input parameter is a row of a triangle data matrix.

    Params: v (np.ndarray of shape (6,)) triangle as x1 y1 x2 y2 x3 y3
    """
    assert type(v) == np.ndarray and len(v) == 6
    # ADD CODE HERE
    pencolor('black')
    up()
    for j in range(0, 3):
        goto(v[2*j:(2*j)+2])
        down()
        if j == 2:
            goto(v[0:2])
    up()

def drawManyTri (tri_data):

    """Draw all the triangles in a triangle data matrix.
    Hint: use slicing.
    Returns: (np.ndarray of shape (n*n,6)) triangle data matrix
    """

    assert type(tri_data) == np.ndarray and len(tri_data.shape) == 2
    assert tri_data.shape[1] == 6
    # ADD CODE HERE
    for i in range(0,np.size(tri_data,axis=0)):
        tri_data[drawTri(tri_data[i,:])]
        

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
    drawManyTri (tri_data_1)

# uncomment once you have implemented
speed(0)
#driver(2)

driver(4)
hideturtle()
done()
# temporarily comment out some calls in driver
# to build up gradually;
# test individual functions

#!/usr/bin/env python3
# CS103; john k johnstone; jkj at uab dot edu; mit license
#Eric Dollar 
#BlazerID: ERIC6249
# only 4 modules are necessary, and only 4 are allowed: numpy/turtle/random/math
import numpy as np
from turtle import *
import random
import math

def randomPt (BL, TR):


    assert type(BL) == np.ndarray and len(BL) == 2
    assert type(TR) == np.ndarray and len(TR) == 2
    
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


    assert n%2 == 0
    assert i>=0 and i<n
    assert j>=0 and j<n

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


def vtx (i,T):


    assert 0 <= i and i < 3
    assert type(T) == np.ndarray and len(T) == 6
    i*2
    a = T.array[i]
    np.append(a, T.ndarray[i+1])
    return a


def angle (i, T):


    assert 0 <= i and i < 3
    assert type(T) == np.ndarray and len(T) == 6
    if i == 0:
        #1st point to 2nd
        v = np.array([T[i+2] - T[i], T[i+3] - T[i+1]])
        #1st point to 3rd
        w = np.array([T[i+4] - T[i], T[i+5] - T[i+1]])
    elif i == 1:
        #2nd point to 3rd
        v = np.array([T[i+3] - T[i+1], T[i+4] - T[i+2]])
        #2nd point to 1st
        w = np.array([T[i-1] - T[i+1], T[i] - T[i+2]])
    elif i == 2:
        #3rd point to 1st 
        v = np.array([T[i-2] - T[i+2], T[i-1] - T[i+3]])
        #3rd point to 2nd
        w = np.array([T[i] - T[i+2], T[i+1] - T[i+3]])
    
    w1 = math.sqrt(w[0]**2 + w[1]**2)
    v1 = np.linalg.norm(v)
    w1 = np.linalg.norm(w)
    a = np.inner(v,w)/(v1*w1)
    if -1 < a < 1:
        theta = math.acos(a)
        a = math.degrees(theta)
        return a
    else:
        a = 0
        return a


def randomRobustTri (i,j,n,minAngle=30):


    assert n%2 == 0
    assert i>=0 and i<n
    assert j>=0 and j<n
    assert minAngle < 60 

    robust = False
    while robust == False:
        T = randomTri(i,j,n)
        a = []
        for k in range(0,3):
            a.append(angle(k,T))
        if a[0] > 30 and a[1] > 30 and a[2] > 30:
            robust = True
    return T


def buildTriData (n, tri_data):
    
    
    assert n%2 == 0 
    assert type(tri_data) == np.ndarray and tri_data.shape == (n*n,6)

    x = 0
    c = 0
    for i in range(0,n):
        for j in range(0,n):
            a = randomRobustTri(i,j,n)
            for x in range(len(a)):
                tri_data[c, x] = a[x]
            c+=1


def writeData (fn, data_matrix):


    assert type(fn) == str
    assert type(data_matrix) == np.ndarray

    f = open(fn, 'w')
    mat = np.matrix(data_matrix)
    with open(fn, 'w') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')
    f.close()


def writeTriData (fn, tri_data):


    assert type(fn) == str
    assert type(tri_data) == np.ndarray and len(tri_data.shape) == 2
    assert tri_data.shape[1] == 6
    # ADD CODE HERE
    f = open(fn, 'w')
    writeData(fn, tri_data)
    f.close()


def readTriData (fn):

    
    assert type(fn) == str
    f = open(fn, 'r')
    a = np.loadtxt(fn)
    f.close()
    return a


def drawTri (v):


    assert type(v) == np.ndarray and len(v) == 6

    speed(0)
    up()
    l=["black","yellow","orange","blue","purple","red","green","brown","gold","maroon","violet","magenta","navy","gray"]
    color('black',random.choice(l))
    begin_fill()
    for j in range(0, 3):
        goto(v[2*j:(2*j)+2])
        down()
        if j == 2:
            goto(v[0:2])
    up()
    end_fill()


def drawManyTri (tri_data):


    assert type(tri_data) == np.ndarray and len(tri_data.shape) == 2
    assert tri_data.shape[1] == 6

    for i in range(0,np.size(tri_data,axis=0)):
        tri_data[drawTri(tri_data[i,:])]


def driver (n):

    # DO NOT CHANGE THIS FUNCTION: IT IS COMPLETE
    assert n%2 == 0
    tri_data_1 = np.zeros ((n*n, 6)) # empty, will be filled by buildTriData
    buildTriData (n, tri_data_1)
    writeTriData ('tri.txt', tri_data_1)
    tri_data = readTriData ('tri.txt')
    assert np.array_equal (tri_data_1, tri_data)
    drawManyTri (tri_data)

# uncomment once you have implemented
#driver(2)
driver(10)
done()
# temporarily comment out some calls in driver
# to build up gradually;
# test individual functions

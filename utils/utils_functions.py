'''
 Before running call:
export PATH=Packages/anaconda/bin\:$PATH
export PYTHONPATH=Caffe/caffe-dilation/build_master_release/python:$PYTHONPATH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:Caffe/caffe-dilation/build_master_release/lib"
export PATH=$PATH\:Packages/anaconda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH\:/usr/local/cuda-7.5/lib64
export PATH=$PATH\:/usr/local/cuda-7.5/bin

Bodyparts in order: Standing    Biking
1. head             Upper 30%   Upper 50%
2. neck             Upper 40%   Upper 50%
3. R. shoulder      Upper 40%   Upper 60%
4. R. elbow         Upper 50%   Upper 75%
5. R. wrist         Middle 60%  Middle 60%
6. L. shoulder      Upper 40%
7. L. elbow         Upper 40%
8. L. wrist         Upper 50%
9. R. hip           Middle 50%  Middle 60%
10. R. knee         Lower 50%   Lower 75%
11. R. ankle        Lower 20%   Lower 50%
12. L. hip          Middle 50%
13. L. knee         Lower 50%
14. L. ankle        Lower 20%
'''

# The script evaluates the approximate position of the joints by assuming teh above

# from PyQt4.QtCore import QObject
# matplotlib.use("Qt4Agg")
# matplotlib.use('Agg')# " Uncomment when ssh"

# import matplotlib.pyplot as plt
# import pylab

import math
import numpy as np
from scipy import ndimage

# Checks if two lists overlap. buf is precision
def overlap(cluster1, cluster2, buf):
    # Buffer length relative to the length of the square.
    # y_dist = math.ceil(max(cluster1[1] - cluster1[0], cluster2[1] - cluster2[0]) * buf)
    # x_dist = math.ceil(max(cluster1[3] - cluster1[2], cluster2[3] - cluster2[2]) * buf)
    #

    if len(cluster1)==6:
        return overlap_3D(cluster1, cluster2)
    vals = [False, False]
    for dim in range(2):
        i = dim * 2
        # print (str(cluster1[i:i + 2])+" "+str(cluster2[i:i + 2])+" dim "+str(i))
        # print ("Between "+str(between(cluster1[i:i + 2], cluster2[i:i + 2]))+" "+str(between(cluster2[i:i + 2], cluster1[i:i + 2])))
        # print ("Inside " + str( inside( cluster1[i:i + 2], cluster2[i:i + 2]))+" "+str(inside(cluster1[i:i + 2], cluster2[i:i + 2])))
        if between(cluster1[i:i + 2], cluster2[i:i + 2]) or between(cluster2[i:i + 2], cluster1[i:i + 2]) or inside(
                cluster1[i:i + 2], cluster2[i:i + 2]) or inside(cluster1[i:i + 2], cluster2[i:i + 2]):
            vals[dim] = True
    #         print (vals)
    # print (vals[0] and vals[1])
    return vals[0] and vals[1]
    #return len(intersection(cluster1, cluster2, x_dist, y_dist))

def overlap_3D(cluster1, cluster2):

    # Buffer length relative to the length of the square.
    vals=[False,False, False]
    for dim in range(3):
        i=dim*2
        #print str(cluster1[i:i + 2])+" "+str(cluster2[i:i + 2])
        if between(cluster1[i:i+2], cluster2[i:i+2]) or between(cluster2[i:i + 2], cluster1[i:i + 2]) or inside(cluster1[i:i+2], cluster2[i:i+2])or inside(cluster1[i:i+2], cluster2[i:i+2]):
            vals[dim]=True
            #print "overlap"
    #print vals[0] and vals[1] and vals[2]

    return vals[0] and vals[1] and vals[2]

def between (pair1, pair2):
    return round(pair1[0])<=round(pair2[0])<=round(pair1[1]) or round(pair1[0])<=round(pair2[1])<=round(pair1[1])

def inside (pair1, pair2):
    return round(pair1[0])<=round(pair2[0]) and round(pair2[1])<=round(pair1[1])

# Union of two rectangles
def union(a, b):
    y_min = min(a[0], b[0])
    y_max = max(a[1], b[1])
    x_min = min(a[2], b[2])
    x_max = max(a[3], b[3])
    return (y_min, y_max, x_min, x_max)


# Intersection of two rectangles
def intersection(a, b, x_dist, y_dist):
    x = max(a[0], b[0])
    y = max(a[2], b[2])
    w = min(a[1], b[1])
    h = min(a[3], b[3])
    if w - x < -y_dist or h - y < -x_dist: return ()
    return (x, y, w, h)

# creates float range.
def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def save_obj(obj, name ):
    import pickle
    with open( name[:-4] + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    import pickle
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding="latin1", fix_imports=True)

def shortest_dist(positions_pers, pred_indx):
    min_dist = np.square(pred_indx[0] - positions_pers[0][0]) + np.square(pred_indx[1] - positions_pers[1][0])
    # print len(positions_pers[0])
    for k in range(0, len(positions_pers[0])):
        d = np.square(pred_indx[0] - positions_pers[0][k]) + np.square(pred_indx[1] - positions_pers[1][k])
        if (d < min_dist):
            min_dist = d
    return np.sqrt(min_dist)


def label_show(label):
    colors = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                       [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                       [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])
    colors = np.uint8(colors)
    label_r = colors[label.ravel()]
    label_r = label_r.reshape((label.shape[0], label.shape[1], 3))
    return label_r

# Checks if point x, y is in square sq.

def in_square(sq, x, y, x_dist, y_dist):
    return (sq[0] <= y + y_dist and sq[1] + y_dist >= y) and (sq[2] <= x + x_dist and sq[3] + x_dist >= x)

def find_border(v_pos):
    test_positions=[]
    # for x in range(v_pos.shape[0]):
    #     for y in range(v_pos.shape[1]):
    #         if x > 0 and y > 0 and( v_pos[x - 1, y]!=v_pos[x, y] or v_pos[x , y-1]!=v_pos[x, y]):
    #             if v_pos[x , y] == 1:
    #                 test_positions.append([x, y])
    #             if v_pos[x - 1, y] == 1:
    #                 test_positions.append([x-1, y])
    #             if v_pos[x, y - 1] == 1:
    #                 test_positions.append([x, y - 1])
    difference=v_pos-ndimage.grey_erosion(v_pos, size=(3,3))
    test_positions=np.transpose(np.nonzero(difference))
    # print "Border "#+str(test_positions.shape)
    # print test_positions
    return test_positions
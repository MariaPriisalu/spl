import numpy as np


def dist(long_1, lat_1, long_2, lat_2):
    from geopy.distance import vincenty
    dest2 = (lat_2, long_2)
    dest1_xdiff = (lat_1, long_2)
    dest1_ydiff = (lat_2, long_1)
    return (np.sign(lat_1 - lat_2) * vincenty(dest2, dest1_xdiff).meters,
            np.sign(long_1 - long_2) * vincenty(dest2, dest1_ydiff).meters)


def rotation(bearing_2):
    deg = -(bearing_2)
    M = [[np.cos(deg), -np.sin(deg), 0],
         [np.sin(deg), np.cos(deg), 0],
         [0, 0, 1]]
    return np.array(M)
import json
import math
import numpy as np


def readCameraM(data_file):
    data = json.load(data_file)
    inD = data["intrinsic"]
    K = np.array([[inD["fx"], 0, inD["u0"]], [0, inD["fy"], inD["v0"]], [0, 0, 1]])

    exD = data["extrinsic"]
    s_y = (math.sin(exD["yaw"]), math.cos(exD["yaw"]))
    s_p = (math.sin(exD["pitch"]), math.cos(exD["pitch"]))
    s_r = (math.sin(exD["roll"]), math.cos(exD["roll"]))
    R = [[s_y[1] * s_p[1], s_y[1] * s_p[0] * s_r[0] - (s_y[0] * s_r[1]), s_y[1] * s_p[0] * s_r[1] + (s_y[0] * s_r[0]),
          exD["x"]],
         [s_y[0] * s_p[1], s_y[0] * s_p[0] * s_r[0] + (s_y[1] * s_r[1]), s_y[0] * s_p[0] * s_r[1] - (s_y[1] * s_r[0]),
          exD["y"]],
         [-s_p[0], s_p[1] * s_r[0], s_p[1] * s_r[1], exD["z"]]]
    R = np.array(R)
    Q = [[1.0 / inD["fx"], 0, 0, -inD["u0"] / inD["fx"]],
         [0, 1 / inD["fy"], 0, -inD["v0"] / inD["fy"]],
         [0, 0, 0, 1],
         [0, 0, 1.0 / (exD['baseline'] * inD["fx"]), 0]]
    return R, K, Q, exD, inD


def readCamera_inverse(data_file):
    data = json.load(data_file)
    inD = data["intrinsic"]
    exD = data["extrinsic"]
    baseline = exD["baseline"]
    s_y = (math.sin(exD["yaw"]), math.cos(exD["yaw"]))
    s_p = (math.sin(exD["pitch"]), math.cos(exD["pitch"]))
    s_r = (math.sin(exD["roll"]), math.cos(exD["roll"]))
    R = [[s_y[1] * s_p[1], s_y[1] * s_p[0] * s_r[0] - (s_y[0] * s_r[1]), s_y[1] * s_p[0] * s_r[1] + (s_y[0] * s_r[0])],
         [s_y[0] * s_p[1], s_y[0] * s_p[0] * s_r[0] + (s_y[1] * s_r[1]), s_y[0] * s_p[0] * s_r[1] - (s_y[1] * s_r[0])],
         [-s_p[0], s_p[1] * s_r[0], s_p[1] * s_r[1]]]
    R_inv = np.zeros((3, 4))
    R_inv[:, :-1] = np.array(R).T
    t = -np.dot(R, np.array([exD["x"], exD["y"], exD["x"]]).T)
    R_inv[:, 3] = t
    return R_inv


def estimateR(yaw, exD):
    s_y = (math.sin(yaw), math.cos(yaw))
    s_p = (math.sin(exD["pitch"]), math.cos(exD["pitch"]))
    s_r = (math.sin(exD["roll"]), math.cos(exD["roll"]))
    R = [[s_y[1] * s_p[1], s_y[1] * s_p[0] * s_r[0] - (s_y[0] * s_r[1]), s_y[1] * s_p[0] * s_r[1] + (s_y[0] * s_r[0]),
          exD["x"]],
         [s_y[0] * s_p[1], s_y[0] * s_p[0] * s_r[0] + (s_y[1] * s_r[1]), s_y[0] * s_p[0] * s_r[1] - (s_y[1] * s_r[0]),
          exD["y"]],
         [-s_p[0], s_p[1] * s_r[0], s_p[1] * s_r[1], exD["z"]]]
    R = np.array(R)
    return R
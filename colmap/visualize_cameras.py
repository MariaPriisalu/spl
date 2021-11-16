# COLMAP - Structure-from-Motion and Multi-View Stereo.
# Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# This script exports inlier image pairs from a COLMAP database to a text file.

import sqlite3

import os
from scipy import misc
from PIL import Image as Image
import json
import locale

# Set to users preferred locale:
locale.setlocale(locale.LC_ALL, '')
# Or a specific locale:
locale.setlocale(locale.LC_NUMERIC, "en_DK.UTF-8")

import matplotlib.pyplot as plt

import numpy as np
from colmap.reconstruct import get_colmap_camera_matrixes, get_camera_matrix_IMU
from .reconstruct_all import get_camera_matrices

def main():
    # Set paths!.
    mode_name="train"
    path_colmap_dir = "Datasets/colmap2/"
    img_path = "cityscapes_dataset/cityscapes_videos/leftImg8bit_sequence"  # images
    gt_path = "Datasets/cityscapes/gtFine/"  # Ground truth
    results_path = "Datasets/cityscapes/colmap/FRCNN/"  # Where to find joint positions.
    vis_path = "Datasets/cityscapes/visual"
    plot=False
    city="weimar"
    seq_nbr="000120"
    camera_m, translation, camera_params = get_colmap_camera_matrixes(city, seq_nbr, path_colmap_dir)
    #K, R = get_camera_matrix_IMU(city, mode_name, seq_nbr)
    camera_locations, camera_locations_colmap, camera_locations_right, camera_locations_right_colmap = order_colmap_camera_matrices(
        camera_m, city, mode_name, seq_nbr, translation)

    recentered_cam_pos=np.zeros_like(camera_locations)
    recentered_cam_pos_colmap = np.zeros_like(camera_locations)
    for i,cam_pos in enumerate(camera_locations):
        pos=camera_locations[i]-camera_locations[0]
        recentered_cam_pos[i]=camera_locations[i]-camera_locations[0]
        pos_colmap = camera_locations_colmap[i] - camera_locations_colmap[0]
        recentered_cam_pos_colmap[i] = camera_locations_colmap[i] - camera_locations_colmap[0]
        print("Pos in cityscapes: "+str(pos)+ " Pos in colmap: " + str(pos_colmap)+"Difference in distance: "+str(np.linalg.norm(pos_colmap)-np.linalg.norm(pos)))
    print("Right side ")
    for i,cam_pos in enumerate(camera_locations_right ):
        pos=camera_locations_right[i]-camera_locations_right[0]
        pos_colmap = camera_locations_right_colmap[i] - camera_locations_right_colmap[0]
        print("Pos in cityscapes: "+str(pos)+ " Pos in colmap: " + str(pos_colmap)+str(pos_colmap)+"Difference in distance: "+str(np.linalg.norm(pos_colmap)-np.linalg.norm(pos)))

    scale = get_avg_colmap_baseline(camera_locations, camera_locations_colmap, camera_locations_right,
                                    camera_locations_right_colmap, camera_m)
    #print "Scaling: "+ str(scale)

    for i, cam_pos in enumerate(camera_locations):
        pos = camera_locations[i] - camera_locations[0]
        pos_colmap = (camera_locations_colmap[i] - camera_locations_colmap[0])*scale
        print("Pos in cityscapes: " + str(pos) + " Pos in colmap: " + str(
            pos_colmap) + "Difference in distance: " + str(np.linalg.norm(pos_colmap) - np.linalg.norm(pos)))
    print("Right side ")
    for i, cam_pos in enumerate(camera_locations_right):
        pos = camera_locations_right[i] - camera_locations_right[0]
        pos_colmap = (camera_locations_right_colmap[i] - camera_locations_right_colmap[0])*scale
        print("Pos in cityscapes: " + str(pos) + " Pos in colmap: " + str(pos_colmap) + str(
            pos_colmap) + "Difference in distance: " + str(np.linalg.norm(pos_colmap) - np.linalg.norm(pos)))

        # after re-scaling:


    for i, cam_pos in enumerate(camera_locations):  # Should be  0.209313
        pos = camera_locations[i] - camera_locations_right[i]
        print("Pos in cityscapes: " + str(pos) + " Distance: " + str(np.linalg.norm(pos)))
        pos_colmap = (camera_locations_colmap[i] - camera_locations_right_colmap[i])*scale
        print("Pos in colmap: " + str(pos_colmap) + " Distance: " + str(np.linalg.norm(pos_colmap)))
    import matplotlib.pyplot as plt
    # fig1 = plt.figure()
    # plt.plot(camera_locations[:, 0], camera_locations[:, 2], label="camera")
    # plt.plot(camera_locations_colmap[:, 0], camera_locations_colmap[:, 2], label="colmap")
    # plt.plot(camera_locations_colmap[:, 0] * scale, camera_locations_colmap[:, 2] * scale, label="colmap scaled")
    #
    # # plt.ylim((-0.5,2))
    # # plt.xlim((0, 45))
    # # plt.axis('equal')
    # plt.legend()
    # plt.show()

    fig1 = plt.figure()
    plt.plot(recentered_cam_pos[:, 0], recentered_cam_pos[:, 2], label="camera")
    plt.plot(recentered_cam_pos_colmap[:, 0], recentered_cam_pos_colmap[:, 2], label="colmap")
    plt.plot(recentered_cam_pos_colmap[:, 0]*scale, recentered_cam_pos_colmap[:, 2]*scale, label="colmap scaled")

    # plt.ylim((-0.5,2))
    # plt.xlim((0, 45))
    #plt.axis('equal')
    plt.legend()
    plt.show()
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(recentered_cam_pos[:, 0], -recentered_cam_pos[:, 1], recentered_cam_pos[:, 2], label='camera')
    ax.plot(recentered_cam_pos_colmap[:, 0], -recentered_cam_pos_colmap[:, 1], recentered_cam_pos_colmap[:, 2],
            label='colmap')
    ax.plot(recentered_cam_pos_colmap[:, 0] * scale, -recentered_cam_pos_colmap[:, 1] * scale,
            recentered_cam_pos_colmap[:, 2] * scale, label='colmap scaled')
    ax.legend()

    plt.show()
    print("__________________________________________________________________")
    print(camera_locations)
    print("__________________________________________________________________")
    print(camera_locations_colmap)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # ax.plot(camera_locations[:, 0], -camera_locations[:, 1], camera_locations[:, 2], label='camera')
    # ax.plot(camera_locations_colmap[:, 0], -camera_locations_colmap[:, 1], camera_locations_colmap[:, 2], label='colmap')
    # ax.plot(camera_locations_colmap[:, 0]*scale, -camera_locations_colmap[:, 1]*scale, camera_locations_colmap[:, 2]*scale, label='colmap scaled')
    # ax.legend()
    #
    # plt.show()


def get_avg_colmap_baseline(camera_locations, camera_locations_colmap, camera_locations_right,
                            camera_locations_right_colmap, camera_m):
    # baseline
    avg_dist = 0
    count = 0
    for i, cam_pos in enumerate(camera_locations):  # Should be  0.209313
        pos = camera_locations[i] - camera_locations_right[i]
        # print "Pos in cityscapes: " + str(pos) + " Distance: " + str( np.linalg.norm(pos))
        pos_colmap = camera_locations_colmap[i] - camera_locations_right_colmap[i]
        # print "Pos in colmap: " + str(pos_colmap) + " Distance: " + str(np.linalg.norm(pos_colmap))
        filename_left = "weimar_000120_%06d_leftImg8bit.png" % (i)
        filename_right = "weimar_000120_%06d_rightImg8bit.png" % (i)
        if filename_left in camera_m and filename_right in camera_m:
            print(filename_left +str(np.linalg.norm(pos_colmap)))
            avg_dist += np.linalg.norm(pos_colmap)
            count += 1
        print(avg_dist)
        print(count)
    # print "Avg dist "+str(avg_dist/count)
    scale = 0.209313 / (avg_dist / count)
    return scale


def order_colmap_camera_matrices(camera_m, city, mode_name, seq_nbr, translation):
    camera_locations, camera_locations_right = get_camera_matrices(city, seq_nbr, mode_name)
    camera_locations_colmap = np.ones((30, 3))
    camera_locations_right_colmap = np.ones((30, 3))
    camera_locations_colmap_dict = {}
    for camera_filename in camera_m:
        parts = camera_filename.split('_')
        frame_nbr = int(parts[2])
        R_loc = camera_m[camera_filename]
        t = translation[camera_filename].reshape((3, 1))
        pos_colmap =  np.matmul(-np.matrix(R_loc).T, t)
        if "right" in parts[3]:
            camera_locations_right_colmap[frame_nbr, :] = pos_colmap.T
        else:
            camera_locations_colmap[frame_nbr, :] = pos_colmap.T

        camera_locations_colmap_dict[camera_filename] = pos_colmap
    extract_camera_ply(camera_locations_colmap_dict, camera_m)
    return camera_locations, camera_locations_colmap, camera_locations_right, camera_locations_right_colmap


def extract_camera_ply(camera_locations_colmap_dict,  camera_m):
    file = open("cameras.ply", "w")
    file.write("ply \nformat ascii 1.0 \n")
    file.write("element vertex %d \nproperty float x\nproperty float y \nproperty float z\n" % (len(camera_m)))#* 2))
    #file.write("element edge %d\n" % len(camera_m))
    #file.write("property int vertex1\nproperty int vertex2\nend_header\n")
    #file.write("property uchar red \nproperty uchar green\n property uchar blue")
    file.write("end_header\n")
    for filename_m in camera_m:
        cam_pos = camera_locations_colmap_dict[filename_m].reshape((3,))
        strN = "%.6f %.6f %.6f \n" % (cam_pos[0, 0], cam_pos[0, 1], cam_pos[0, 2])
        file.write(strN.replace('.', ','))

    # for i, cam_pos in enumerate(camera_m):
    #     file.write("%d %d 255 0 0\n" % (i * 2, i * 2 + 1))
    file.flush()


if __name__ == "__main__":
    main()

from plyfile import PlyData
import numpy as np

#
# Read pointcloud from ply file.
#
def read_3D_pointcloud(filename, file_end='/dense/fused_text.ply'):
    plydata_3Dmodel = PlyData.read(filename + file_end)
    nr_points = plydata_3Dmodel.elements[0].count
    pointcloud_3D = np.array([plydata_3Dmodel['vertex'][k] for k in range(nr_points)])
    return pointcloud_3D


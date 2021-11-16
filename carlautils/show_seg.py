import pickle, os
from plyfile import PlyData, PlyElement
from commonUtils.ReconstructionUtils import save_to_disk, get_label_mapping
import numpy as np

filepath="Packages/CARLA_0.8.2/PythonClient/_out/test_0"
reconstruction_path=os.path.join(filepath, 'combined_carla_moving.ply')
centering = pickle.load(open(os.path.join(filepath, "centering.p"), "rb"), encoding="latin1", fix_imports=True)

colours = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
            (111, 74, 0), (81, 0, 81), (128, 64, 128),(244, 35, 232), (250, 170, 160),
           (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),(180, 165, 180),
           (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
            (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
           (255, 0, 0),(0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90),
           (0, 0, 110), (0, 80, 100), (0, 0, 230),(119, 11, 32), (0, 0, 142)]


plydata = PlyData.read(reconstruction_path)
nr_points = plydata.elements[0].count
pointcloud_3D = np.array([plydata['vertex'][k] for k in range(nr_points)])
combined_reconstruction=[]
for point in pointcloud_3D:
    rgb=colours[point[6]]
    combined_reconstruction.append(
        (point[0], point[1], point[2], rgb[2], rgb[1], rgb[0], point[6]))

save_to_disk(combined_reconstruction, filepath+"/combined_seg.ply")


filepath="Packages/CARLA_0.8.2/PythonClient/_out/test_0"
reconstruction_path=os.path.join(filepath, '00010_seg.ply')
centering = pickle.load(open(os.path.join(filepath, "centering.p"), "rb"), encoding="latin1", fix_imports=True)

colours = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
            (111, 74, 0), (81, 0, 81), (128, 64, 128),(244, 35, 232), (250, 170, 160),
           (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),(180, 165, 180),
           (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
            (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
           (255, 0, 0),(0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90),
           (0, 0, 110), (0, 80, 100), (0, 0, 230),(119, 11, 32), (0, 0, 142)]


plydata = PlyData.read(reconstruction_path)
nr_points = plydata.elements[0].count
pointcloud_3D = np.array([plydata['vertex'][k] for k in range(nr_points)])
label_map=get_label_mapping()
combined_reconstruction=[]
for point in pointcloud_3D:
    labels = [point[3], point[4], point[5]]
    counts = np.bincount(labels)
    label = np.argmax(counts)
    rgb=colours[label_map[label]]
    combined_reconstruction.append(
        (point[0], point[1], point[2], rgb[2], rgb[1], rgb[0], label))

save_to_disk(combined_reconstruction, filepath+"/00010_colseg.ply")
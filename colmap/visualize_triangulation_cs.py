import os, pickle, glob, cv2, copy, re
external=True

import matplotlib.pyplot as plt
from .reconstruct import read_array, calculate_mode_depth_real, triangulate_point, get_bbx_from_csv, reconstruct_ppl
import numpy as np
from pyquaternion import Quaternion
#from commonUtils.ReconstructionUtils import reconstruct3D_ply import get_camera_rotationmatrix, get_camera_intrinsic_matrix, get_cars_list
folder_path="Datasets/colmap2/bremen_000014"#"Packages/CARLA_0.8.2/PythonClient/recon2/"
segmentation_path=os.path.join(folder_path,"segmentation", 'left')
images_path=os.path.join(folder_path,"images", 'left')
depth_path=os.path.join(folder_path,"images", 'left')

camera_file = open(folder_path + '/cameras.txt', 'r')
content = camera_file.readlines()
camera_file.close()
camera_params = [0, 1]
for line in content:
    if not '#' == line[0]:
        params = line.split(" ")
        t = []
        t.append([float(params[4]), 0, float(params[6])])
        t.append([0, float(params[5]), float(params[7][:-1])])
        t.append([0, 0, 1])
        if len(camera_params)<int(line[0]) :
            camera_params.append(np.array(t))
        else:
            camera_params[int(line[0]) - 1] = np.array(t)
print(camera_params)



camera_rotation_file = open(folder_path + '/images.txt', 'r')
content = camera_rotation_file.readlines()
camera_rotation_file.close()
camera_rotation = {}
translation = {}
# Update camera positions.
for line in content:
    if not '#' == line[0] and ".png" in line:
        params = line.split(" ")
        basename = os.path.basename(params[9].rstrip())
        v = Quaternion(float(params[1]), float(params[2]), float(params[3]), float(params[4])) #w x,y,z
        camera_rotation[basename] = v.rotation_matrix
        translation[basename] = np.array((float(params[5]), float(params[6]), float(params[7])))
        if float(params[5])==0 and float(params[6])==0 and float(params[7])==0:
            print("center "+basename)
print(camera_rotation)
bbox_dict = get_bbx_from_csv(os.path.join("Datasets/cityscapes/pannet2/", 'people_bboxes.csv'), os.path.basename(folder_path))
#reconstrcution_dir, scale_vector, middle, bbox_dict, path_masks,cars_flag,colmap_path, image_size=(1024, 2048)
people = reconstruct_ppl(os.path.basename(folder_path), (1, 1, 1), (0,0,0), bbox_dict,  path_masks="Datasets/cityscapes/pannet2/", cars_flag=False,
                         colmap_path="Datasets/colmap2/", to_cs=False)

bbox_cars = get_bbx_from_csv(os.path.join("Datasets/cityscapes/pannet2/", 'cars_bboxes.csv'),os.path.basename(folder_path))
#print "cars "+str(bbox_cars)

cars=reconstruct_ppl(os.path.basename(folder_path),
                     (1, 1, 1), (0, 0, 0), bbox_cars, path_masks="Datasets/cityscapes/pannet2/",
                     cars_flag=False,
                     colmap_path="Datasets/colmap2/", to_cs=False)


file_content = ""

file_name = os.path.join(folder_path, "agent.ply")
print(file_name)
nbr_points_cpy = 0
init_text = "ply\nformat ascii 1.0\nelement vertex 290219\nproperty float32 x\nproperty float32 y\nproperty float32 z\nproperty uchar diffuse_red\nproperty uchar diffuse_green\nproperty uchar diffuse_blue\nend_header\n"
with open(file_name, 'w+') as frame_file:
    for frame in range(len(people)):#range(28):#len(people_bounding_boxes)):
        for person in people[frame]:

            for x in np.arange(person[0][0]+3, person[0][1] -3, 0.1):
                for y in np.arange(person[1][0]+3, person[1][1] -3, 0.1):
                    for z in np.arange(person[2][0]+3, person[2][1] -3, 0.1):
                        s = '%f %f %f %d %d %d \n' % (x, y, z, 0, 255, 0)
                        file_content = file_content + s
                        # frame_file.write('%f %f %f %d %d %d %d \n'%(x,y,z,255,0,0, 4))
                        nbr_points_cpy = nbr_points_cpy + 1

        for person in cars[frame]:
            for x in np.arange(person[0][0]+3, person[0][1] -3, 0.1):
                for y in np.arange(person[1][0]+3, person[1][1] -3, 0.1):
                    for z in np.arange(person[2][0]+3, person[2][1] -3, 0.1):
                        s = '%f %f %f %d %d %d \n' % (x, y, z, 0,0, 255)
                        file_content = file_content + s
                        # frame_file.write('%f %f %f %d %d %d %d \n'%(x,y,z,255,0,0, 4))
                        nbr_points_cpy = nbr_points_cpy + 1





    line = re.sub(
        r"element vertex \d+",
        "element vertex %d" % nbr_points_cpy,
        init_text)

    frame_file.write(line + file_content )



for key in list(camera_rotation.keys()):
    print(key)



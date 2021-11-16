import os, pickle, glob, cv2, copy, re
external=True

import matplotlib.pyplot as plt
from .reconstruct import read_array, calculate_mode_depth_real, triangulate_point
import numpy as np
from pyquaternion import Quaternion
#from commonUtils.ReconstructionUtils import get_camera_rotationmatrix, get_camera_intrinsic_matrix, get_cars_list
folder_path="Packages/CARLA_0.8.2/PythonClient/recon2/"
segmentation_path=os.path.join(folder_path,"segmentation", 'left')
images_path=os.path.join(folder_path,"images", 'left')
depth_path=os.path.join(folder_path,"images", 'left')


cars_path = os.path.join(folder_path, 'cars.p')
cars_dict = pickle.load(open(cars_path, "rb"), encoding="latin1", fix_imports=True)

people_path = os.path.join(folder_path, 'people.p')
people_dict = pickle.load(open(people_path, "rb"), encoding="latin1", fix_imports=True)

#R, middle, frames= get_camera_rotationatrix(folder_path)
# K=get_camera_intrinsic_matrix(folder_path)


cameras_dict = pickle.load(open(os.path.join(folder_path, 'cameras.p'), "rb"), encoding="latin1", fix_imports=True)
frames=sorted(cameras_dict.keys())

frame = frames[0]#np.min(cameras_dict.keys())
R_world_to_car = cameras_dict[frame]['inverse_rotation']
R_car_to_world=cameras_dict[frame]['rotation']



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

people_bounding_boxes=[]
for frame, filename in enumerate(sorted(glob.glob(images_path+'/*'))):
    people_bounding_boxes.append([])

    img = destRGB = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2RGB)
    fig = plt.figure(frameon=False)

    sem_filename = os.path.join(segmentation_path, os.path.basename(filename))
    sem_rgb = cv2.cvtColor(cv2.imread(sem_filename),cv2.COLOR_BGR2RGB)
    sem=sem_rgb[:,:,0]
    ped_mask=sem==4
    depth_map_colmap_path = os.path.join(folder_path,"dense/stereo/depth_maps/left/" + os.path.basename(filename) + ".geometric.bin")
    if os.path.exists(depth_map_colmap_path):
        print(depth_map_colmap_path)
        depth_map_colmap = read_array(depth_map_colmap_path)
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        ax.imshow(img)
        for person_id, person_list in people_dict[frame+1].items():
            if "bounding_box_2D" in list(person_list.keys()):
                bbox=[0,0,0,0]
                for pos in  person_list["bounding_box_2D"]:
                    if sum(bbox)==0:
                        bbox[2]=int(pos[0][0][0])#y-min
                        bbox[3]=int(pos[0][0][0])#y-max
                        bbox[0]=int(pos[1][0][0])#x-min
                        bbox[1] = int(pos[1][0][0])#x-max

                    if pos[0][0][0]>bbox[3]:
                        bbox[3]=int(pos[0][0][0])
                    if pos[1][0][0] > bbox[1]:
                        bbox[1] = int(pos[1][0][0])

                    if pos[0][0][0] <bbox[2]:
                        bbox[2] = int(pos[0][0][0])
                    if pos[1][0][0] < bbox[0]:
                        bbox[0] = int(pos[1][0][0])
                if bbox[1] - bbox[0] > 20 and bbox[3] - bbox[2] > 20:
                    # print bbox
                    ax.add_patch(
                        plt.Rectangle((bbox[0], bbox[2]),
                                      bbox[1] - bbox[0], bbox[3] - bbox[2], fill=False, edgecolor='g',
                                      linewidth=0.5))

                    est_depth_1 = person_list['bounding_box_2D_avg_depth']
                    est_depth=calculate_mode_depth_real(bbox, depth_map_colmap, ped_mask[bbox[2]:bbox[3], bbox[0]:bbox[1]])

                    est_depth=est_depth_1#middle_camera_internal[2]
                    # Now find the point in carla camera coordinates
                    if est_depth > 0:

                        # Find middle of bbox.
                        x = (bbox[0] + bbox[1]) / 2.0
                        y = (bbox[2] + bbox[3]) / 2.0
                        u = np.array([800-x, 600-y, 1])  # 2D point in camera coordinate system.
                        #u = np.array([x, y, 1])


                        p_v = []


                        #p_v = triangulate_point(K, R_car_to_world, est_depth, middle, 1,u, b,colmap=True)  # b7

                        if os.path.basename(filename) in camera_rotation :
                            local_filename=os.path.basename(filename)


                            tran = -np.matmul(np.transpose(camera_rotation[local_filename]),
                                              np.reshape(translation[local_filename], (3, 1)))
                            rotational_matrix = np.concatenate((np.transpose(camera_rotation[local_filename]), tran), axis=1)
                            #print rotational_matrix
                            p_v = triangulate_point(camera_params[0], rotational_matrix, est_depth, u)  # b
                            #p_v = camera_to_cityscapes_coord(p_v, middle, scale)

                        print("Triangulated point "+str(p_v)+" ")
                        people_bounding_boxes[-1].append(p_v)

        height=0
        # people, people_2D = get_cars_list(people_bounding_boxes, {}, frames, R, middle, height, 1, K, folder_path,
        #                                   people_flag=True, find_poses=False)
        plt.show()
        fig.savefig("weimar_120/images_" + str(frame) + "_.jpg")

file_content = ""

file_name = os.path.join(folder_path, "agent.ply")
print(file_name)
nbr_points_cpy = 0
init_text = "ply\nformat ascii 1.0\nelement vertex 290219\nproperty float32 x\nproperty float32 y\nproperty float32 z\nproperty uchar diffuse_red\nproperty uchar diffuse_green\nproperty uchar diffuse_blue\nend_header\n"
with open(file_name, 'w+') as frame_file:
    for frame in [28]:#range(28):#len(people_bounding_boxes)):
            for person in people_bounding_boxes[frame]:
                for x in range(int(person[0])-1, int(person[0])+1 ):
                    for y in range(int(person[1])-1, int(person[1]) +1):
                        for z in range(int(person[2])-1, int(person[2]) + 1):
                            s = '%f %f %f %d %d %d \n' % (x, y, z, 0, 255, 0)
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



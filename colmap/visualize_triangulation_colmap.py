import os, pickle, glob, cv2, copy, re
external=True

import matplotlib.pyplot as plt
from .reconstruct import read_array, calculate_mode_depth_real, triangulate_point
import numpy as np
from commonUtils.ReconstructionUtils import get_camera_matrix, get_camera_intrinsic_matrix, get_cars_list
folder_path="Packages/CARLA_0.8.2/PythonClient/recon2/"
segmentation_path=os.path.join(folder_path,"segmentation", 'left')
images_path=os.path.join(folder_path,"images", 'left')
depth_path=os.path.join(folder_path,"images", 'left')


cars_path = os.path.join(folder_path, 'cars.p')
cars_dict = pickle.load(open(cars_path, "rb"), encoding="latin1", fix_imports=True)

people_path = os.path.join(folder_path, 'people.p')
people_dict = pickle.load(open(people_path, "rb"), encoding="latin1", fix_imports=True)

R, middle, frames= get_camera_matrix(folder_path)
K=get_camera_intrinsic_matrix(folder_path)


cameras_dict = pickle.load(open(os.path.join(folder_path, 'cameras.p'), "rb"), encoding="latin1", fix_imports=True)
frames=sorted(cameras_dict.keys())

frame = frames[0]#np.min(cameras_dict.keys())
R_world_to_car = cameras_dict[frame]['inverse_rotation']
R_car_to_world=cameras_dict[frame]['rotation']

debug_triangulation=False

# Rt = car_to_world_transform.matrix
# Rt_inv = car_to_world_transform.inverse().matrix <---- world to car
# # R_inv=world_transform.inverse().matrix
# if frame>50:
#     cameras_dict[frame-50] = {}
#     cameras_dict[frame-50]['inverse_rotation'] = Rt_inv[:]
#     cameras_dict[frame-50]['rotation'] = Rt[:]
#     cameras_dict[frame-50]['translation'] = Rt_inv[0:3, 3]
#     cameras_dict[frame-50]['camera_to_car'] = camera_to_car_transform.matrix

people_bounding_boxes=[]
for frame, filename in enumerate(sorted(glob.glob(images_path+'/*'))):
    people_bounding_boxes.append([])
    if frame<10:
        img =  cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2RGB)


        depth_filename=os.path.join(depth_path, os.path.basename(filename))

        #depth_rgb = cv2.cvtColor(cv2.imread(depth_filename),cv2.COLOR_BGR2RGB).astype(np.float)
        array = cv2.imread(depth_filename,  cv2.IMREAD_UNCHANGED)
        array = array.astype(np.float32)
        # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        depth_map = np.dot(array[:, :, :3], [256.0 * 256.0, 256.0, 1.0])
        depth_map /= (256.0 * 256.0 * 256.0 - 1.0)
        depth_map=depth_map*1000


        sem_filename = os.path.join(segmentation_path, os.path.basename(filename))

        sem_rgb = cv2.cvtColor(cv2.imread(sem_filename),cv2.COLOR_BGR2RGB)
        sem=sem_rgb[:,:,0]
        ped_mask=sem==4
        print("All pedestrian masks "+str(np.sum(ped_mask[:])))



        depth_map_colmap_path = os.path.join(folder_path,"dense/stereo/depth_maps/left/" + os.path.basename(filename) + ".geometric.bin")
        if os.path.exists(depth_map_colmap_path):
            depth_map_colmap = read_array(depth_map_colmap_path)
        #
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        ax.imshow(img)
        indx=0
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

                #print bbox
                ax.add_patch(
                    plt.Rectangle(( bbox[0], bbox[2]),
                                    bbox[1] - bbox[0],bbox[3] - bbox[2], fill=False, edgecolor='g',
                                  linewidth=0.5))
                if bbox[1] - bbox[0] > 20 and bbox[3] - bbox[2] > 20:

                    est_depth_1 = person_list['bounding_box_2D_avg_depth']
                    m=ped_mask[bbox[2]:bbox[3], bbox[0]:bbox[1]]
                    print("Mask "+str(np.sum(m)))
                    est_depth=calculate_mode_depth_real(bbox, depth_map, ped_mask[bbox[2]:bbox[3], bbox[0]:bbox[1]])
                    person_area = depth_map[bbox[2]:bbox[3], bbox[0]:bbox[1]]

                    if len(ped_mask[bbox[2]:bbox[3], bbox[0]:bbox[1]]) > 0 and len(ped_mask[0]) > 0:
                        person_area = person_area[ped_mask[bbox[2]:bbox[3], bbox[0]:bbox[1]]]
                    if person_area.size:
                        person_area = person_area.flatten()
                        person_area = person_area[np.nonzero(person_area)]
                    fig, ax = plt.subplots()
                    alpha = 0.5

                    plot_values=depth_map[bbox[2]:bbox[3], bbox[0]:bbox[1]]
                    plot_values[~ped_mask[bbox[2]:bbox[3], bbox[0]:bbox[1]]]=0
                    plot_values[plot_values>200]=200
                    img_h = plt.imshow(plot_values, cmap='viridis')
                    right_cb = fig.colorbar(img_h)

                    plt.show()
                    fig.savefig("weimar_120/carla_depth_"+str(indx)+".jpg", bbox_inches='tight')
                    indx=indx+1

                    print("Average depth "+str(est_depth)+" "+str(est_depth_1))
                    if os.path.exists(depth_map_colmap_path):
                        d=calculate_mode_depth_real(bbox, depth_map_colmap, ped_mask[bbox[2]:bbox[3], bbox[0]:bbox[1]])
                        print("Colmap depth  "+ str(d))



                        middle_x=[]
                        middle_y=[]
                        middle_z=[]
                        camera_points=[]
                        for points in zip(person_list['bounding_box_coord'], person_list["bounding_box_2D"]):
                            middle_x.append(points[0][0, 0])
                            middle_y.append(points[0][0, 1])
                            middle_z.append(points[0][0, 2])
                            if debug_triangulation:
                                print("--------------------------------------")
                                point_in_3D=points[0]
                                point_in_2D=points[1]

                                print("Point "+str(point_in_3D)+" "+str(point_in_2D))
                                point_in_3D_4=np.array([[point_in_3D[0,0]], [point_in_3D[0,1]], [point_in_3D[0,2]], [1]])
                                point_3d_in_camera = np.matmul(R_world_to_car, point_in_3D_4)

                                point_3d_in_camera=np.reshape(point_3d_in_camera[:3], (3, 1))
                                print("Point in camera: " + str(point_3d_in_camera)+ " "+str(point_3d_in_camera.shape))
                                point_3d_camera_internal= np.matmul(K, np.reshape(point_3d_in_camera[:3], (3, 1)))
                                print("Point in camera intrinsic :"+str(point_3d_camera_internal)+" "+str(point_3d_camera_internal.item(2))+" "+str(1/point_3d_camera_internal.item(2)))
                                point_2D=point_3d_camera_internal*(1/point_3d_camera_internal.item(2))
                                print("Point 2D:"+str(point_2D))
                                print("Pixels: "+str(800-point_2D[0])+" "+str(600-point_2D[1])+" "+str(point_in_2D[1])+" "+str(point_in_2D[0]))

                                # Find middle of bbox.
                                x = 800-point_in_2D[1][0,0]
                                y = 600-point_in_2D[0][0,0]
                                u = np.array([x, y, 1])  # 2D point in camera coordinate system.
                                b = 0.2  # baselines[int(frame_nbr)]

                                p_v = []
                                # if filename in camera_m and b > 0:


                                p_v = triangulate_point(K, R_car_to_world, point_3d_camera_internal[2], u, b, colmap=True)  # b

                                # np.array([[p_v[0] + 4, p_v[1]- 4, p_v[2] - 4 ],
                                #       [p_v[0] + 4, p_v[1]+ 4, p_v[2] - 4 ],
                                #       [p_v[0] - 4, p_v[1]+ 4,p_v[2] - 4 ],
                                #       [p_v[0] - 4, p_v[1]- 4, p_v[2] - 4 ],
                                #       [p_v[0] + 4, p_v[1]- 4,p_v[2] + 4 ],
                                #       [p_v[0] + 4, p_v[1]+ 4,p_v[2] + 4 ],
                                #       [p_v[0] - 4, p_v[1]+ 4, p_v[2] + 4 ],
                                #       [p_v[0] - 4, p_v[1]- 4,p_v[2] + 4 ]]))
                                print("Triangulated point " + str(p_v) + " ")
                                print(person_list['bounding_box_coord'])
                                # Then multiply with ['rotation']



                        middle_in_3D_4=[np.average(middle_x),np.average(middle_y),np.average(middle_z), 1]
                        print("Middle point "+str(middle_in_3D_4))
                        middle_in_camera=np.matmul(R_world_to_car, np.reshape(middle_in_3D_4, (4, 1)))
                        middle_in_camera = np.reshape(middle_in_camera[:3], (3, 1))
                        print("Point in camera : " + str(middle_in_camera) + " " + str(middle_in_camera.shape))
                        middle_camera_internal = np.matmul(K, np.reshape(middle_in_camera[:3], (3, 1)))
                        print("Point in camera intrinsic :" + str(middle_camera_internal))
                        point_2D = middle_camera_internal * (1 / middle_camera_internal.item(2))
                        print("Point 2D:" + str(point_2D))
                        print("Pixels: " + str(800-point_2D[0])+" "+str(600-point_2D[1]))

                        print("Middle in camera coordinates: "+str(middle_camera_internal))
                        print("Average depth from file:" + str(est_depth) + " " + str(middle_camera_internal[2])+" "+str(middle_camera_internal[2]/est_depth)+" "+str(1/K[0,0]))
                        print("Average depth from depth file:"    + str(est_depth_1) + " "+str(middle_camera_internal[2])+" "+str(middle_camera_internal[2]/est_depth_1)+" "+str(1/K[0,0]))
                        str()
                        est_depth=est_depth_1#middle_camera_internal[2]
                        # Now find the point in carla camera coordinates
                        if est_depth > 0:

                            # Find middle of bbox.
                            x = (bbox[0] + bbox[1]) / 2.0
                            y = (bbox[2] + bbox[3]) / 2.0
                            u = np.array([800-x, 600-y, 1])  # 2D point in camera coordinate system.
                            b = 0.2#baselines[int(frame_nbr)]

                            p_v = []


                            p_v = triangulate_point(K, R_car_to_world, est_depth,u, b,colmap=True)  # b
                            print("Triangulated point "+str(p_v)+" ")
                            people_bounding_boxes[-1].append(p_v)
                            #p_v = triangulate_point(K, R_car_to_world, est_depth, middle, 1, u, b, colmap=True)  # b

                            #Then multiply with ['rotation']
        height=0
        # people, people_2D = get_cars_list(people_bounding_boxes, {}, frames, R, middle, height, 1, K, folder_path,
        #                                   people_flag=True, find_poses=False)
        plt.show()
        fig.savefig("weimar_120/images_" + str(frame) + "_.jpg")

        for frame in range(1):
            file_content = ""

            file_name = os.path.join("weimar_120/", "agent.ply")
            print(file_name)
            nbr_points_cpy = 0
            init_text="ply\nformat ascii 1.0\nelement vertex 290219\nproperty float32 x\nproperty float32 y\nproperty float32 z\nproperty uchar diffuse_red\nproperty uchar diffuse_green\nproperty uchar diffuse_blue\nend_header\n"

            with open(file_name, 'w+') as frame_file:

                for person in people_bounding_boxes[frame]:

                    for x in range(int(person[0][0,0])-1, int(person[0][0,0])+1 ):
                        for y in range(int(person[0][0][0,1])-1, int(person[0][0][0,1]) +1):
                            for z in range(int(person[0][0][0,2])-1, int(person[0][0][0,2]) + 1):
                                s = '%f %f %f %d %d %d \n' % (x, y, z, 0, 255, 0)
                                file_content = file_content + s
                                # frame_file.write('%f %f %f %d %d %d %d \n'%(x,y,z,255,0,0, 4))
                                nbr_points_cpy = nbr_points_cpy + 1





                line = re.sub(
                    r"element vertex \d+",
                    "element vertex %d" % nbr_points_cpy,
                    init_text)

                frame_file.write(line + file_content )








import json
import numpy as np
import os

import sys

sys.path.append('reconstruct3D')
sys.path.append('utils')
from .camera import readCameraM
from .movement import rotation
from colmap.reconstruct import get_colmap_camera_matrixes

def get_camera_matrices(city, seq_nbr, mode_name):
    frame_nbr='000019'

    # Determine file paths to: camera matrix, and vehicle accelerometer/GPS data and the frame's timestamp.
    camera_file = city + "_" + seq_nbr + "_" + frame_nbr + "_camera.json"
    path_camera = os.path.join("Datasets/cityscapes/camera_trainvaltest/camera", mode_name,
                               city, camera_file)

    vehicle_file = city + "_" + seq_nbr + "_" + frame_nbr + "_vehicle.json"
    vehicle_path = os.path.join("Datasets/cityscapes/vehicle_sequence/vehicle_sequence",
                                mode_name, city, vehicle_file)

    timestamp_file = city + "_" + seq_nbr + "_" + frame_nbr + "_timestamp.txt"
    timestamp_path = os.path.join("Datasets/cityscapes/timestamp_sequence/", mode_name, city,
                                  timestamp_file)

    # Open vehicle and camera files.
    vehicle_data = json.load(open(vehicle_path))
    camera = open(path_camera)
    timestamp_file = open(timestamp_path, 'r')

    # read camera matrix and parameters.
    R, K, Q, exD, inD = readCameraM(camera)
    R_inv=np.transpose(R[0:3,0:3])

    # Coordinate change between vehicle and camera.
    P = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    P_inv = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

    M=np.matmul(P,R_inv)

    camera_p=np.matmul(P,-np.matmul(R_inv, R[:,3])) # vechicle center in camera coord.
    # Set 19-th frame's reference vehicle position.
    vech_ref_pos = [vehicle_data["gpsLongitude"], vehicle_data["gpsLatitude"]]



    # Read timestamp file.
    timestamp_prev = 0

    translation_x = 0
    translation_y = 0

    people = []
    cars = []

    yaw_est = 0

    # Go through the frames in the order 19, 18, 17, ...1, 0, 20, 21,...29, 30.
    Rs = {}
    yaws = []
    deltas = []
    accum_angle=0
    frames = list(range(0, 30))
    camera_locations = np.zeros((30, 3))
    camera_locations_right = np.zeros((30, 3))
    for frame in frames:
        people.append([])
        cars.append([])

        yaws.append([])
        deltas.append([])
        # frames=range(30)

    # For all frames.
    for frame_nbr in frames:
        # Find path to vehicle accelerometer data.
        vehicle_file = city + "_" + seq_nbr + "_%06d_vehicle.json" % (frame_nbr)
        vehicle_path = os.path.join("Datasets/cityscapes/vehicle_sequence/vehicle_sequence",
                                    mode_name, city, vehicle_file)
        # Load vehicle accelerometer data.
        car = json.load(open(vehicle_path))

        # File path to timestamp
        timestamp_file = os.path.join("Datasets/cityscapes/timestamp_sequence/", mode_name,
                                      city, city + "_" + seq_nbr + "_%06d_timestamp.txt" % (frame_nbr))
        # Read timestamp from file.
        with open(timestamp_file, 'r') as timestampf:
            timestamp = int(timestampf.read())
        timestampf.closed

        # Difference in time between this and previous frame.
        time_dif=(timestamp - timestamp_prev)
        delta_t = (timestamp - timestamp_prev) * 1e-9

        timestamp_prev=timestamp
        # Difference in angle between this and previous frame.
        yaw_delta = car["yawRate"] * delta_t
        accum_angle += yaw_delta

        # Change in the car's position since last frame.
        dist=car["speed"] * delta_t
        dist_x=car["speed"] * delta_t * np.cos(accum_angle)
        dist_y=car["speed"] * delta_t * np.sin(accum_angle)
        translation_x += car["speed"] * delta_t * np.cos(accum_angle)
        translation_y += car["speed"] * delta_t * np.sin(accum_angle)

        vehicle_pos=np.array([translation_x, translation_y,0])
        camera_left_pos=np.matmul(rotation(accum_angle),R[:,3]) # Rotated camera pos in vehicle coord
        camera_pos_right=np.matmul(rotation(accum_angle)+[0,-exD['baseline'],0],R[:,3])
        # Convert translation into first camera coordinates.
        camera_locations[frame_nbr,:]=np.matmul(M,camera_left_pos+vehicle_pos)+camera_p
        camera_locations_right[frame_nbr, :] =np.matmul(M,camera_pos_right + vehicle_pos)+camera_p

    return camera_locations, camera_locations_right



def reconstruct3D(file):
    filei=open('images.txt', 'w')
    mode_name="train"
    # Split name of label.
    basename = os.path.basename(file)
    parts = basename.split('_')
    city = parts[0]
    seq_nbr = parts[1]
    frame_nbr = parts[2]

    if int(frame_nbr)!=19:
        return None

    # Determine file paths to: camera matrix, and vehicle accelerometer/GPS data and the frame's timestamp.
    camera_file = city + "_" + seq_nbr + "_" + frame_nbr + "_camera.json"
    path_camera = os.path.join("Datasets/cityscapes/camera_trainvaltest/camera", mode_name,
                               city, camera_file)

    vehicle_file = city + "_" + seq_nbr + "_" + frame_nbr + "_vehicle.json"
    vehicle_path = os.path.join("Datasets/cityscapes/vehicle_sequence/vehicle_sequence",
                                mode_name, city, vehicle_file)

    timestamp_file = city + "_" + seq_nbr + "_" + frame_nbr + "_timestamp.txt"
    timestamp_path = os.path.join("Datasets/cityscapes/timestamp_sequence/",mode_name, city,
                                  timestamp_file)


    # Open vehicle and camera files.
    vehicle_data = json.load(open(vehicle_path))
    camera = open(path_camera)
    timestamp_file = open(timestamp_path, 'r')

    # read camera matrix and parameters.
    R, K, Q, exD, inD = readCameraM(camera)

    # Set 19-th frame's reference vehicle position.
    vech_ref_pos = [vehicle_data["gpsLongitude"], vehicle_data["gpsLatitude"]]

    # Coordinate change between vehicle and camera.
    P = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    P_inv = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

    # Read timestamp file.
    timestamp_prev = int(timestamp_file.read())
    timestamp_file.close()
    translation_x = 0
    translation_y = 0

    people = []
    cars = []

    yaw_est=0

    # Go through the frames in the order 19, 18, 17, ...1, 0, 20, 21,...29, 30.
    Rs = {}
    yaws=[]
    deltas=[]
    frames = list(range(0,30))
    camera_locations=np.zeros((30,3))
    vehicle_locations = np.zeros((30, 3))
    for frame in frames:
        people.append([])
        cars.append([])

        yaws.append([])
        deltas.append([])
        # frames=range(30)


    # For all frames.
    for frame_nbr in frames:
        # Find path to vehicle accelerometer data.
        vehicle_file = city + "_" + seq_nbr + "_%06d_vehicle.json" % (frame_nbr)
        vehicle_path = os.path.join("Datasets/cityscapes/vehicle_sequence/vehicle_sequence",
                                    mode_name, city, vehicle_file)
        # Load vehicle accelerometer data.
        car = json.load(open(vehicle_path))

        # File path to timestamp
        timestamp_file = os.path.join("Datasets/cityscapes/timestamp_sequence/", mode_name,
                                      city, city + "_" + seq_nbr + "_%06d_timestamp.txt" % (frame_nbr))
        # Read timestamp from file.
        with open(timestamp_file, 'r') as timestampf:
            timestamp = int(timestampf.read())
        timestampf.closed

        # Difference in time between this and previous frame.
        delta_t = (timestamp - timestamp_prev) * 1e-9
        # Difference in angle between this and previous frame.
        yaw_delta = car["yawRate"] * delta_t
        # Change in the car's position since last frame.
        translation_x += car["speed"] * delta_t * np.cos(yaw_delta)
        translation_y += car["speed"] * delta_t * np.sin(yaw_delta)
        print(translation_x)
        print(translation_y)
        img_name = city + "_" + seq_nbr + "_%06d" % (frame)
        right_img = "right/" + img_name + "_rightImg8bit.png"
        left_img = "left/" + img_name + "_leftImg8bit.png"
        filei.write(right_img+" "+str(translation_y)+ " 0 "+str(translation_x)+ "\n")
        filei.write(left_img + " " + str(translation_y+0.22) + " 0 " + str(translation_x) + "\n")
        if frame_nbr != 0:
            # Change in the car's direction.
            yaw_est = +yaw_delta
        else:
            yaw_est = 0
            translation_x = 0
            translation_y = 0
        timestamp_prev = timestamp


        R_c=rotation(-yaw_est)

        R_new=np.matmul(P, (R[0:3, 0:3]).T)
        R_new=np.matmul(R_new, R_c)
        R_new=np.matmul(R_new,R[0:3, 0:3] )
        R_new=np.matmul(R_new, P_inv)
        #R_new=np.matmul(np.matmul(P,(R[0:3, 0:3]).T), R_c)

        t_cv=[exD["x"], exD["y"], exD["x"]]
        t=[translation_x, translation_y, 0]
        t_new=np.matmul(R_c,t_cv )+t
        t_new=np.matmul((R[0:3,0:3]).T,t_new)-np.matmul(R[0:3,0:3],t_cv)
        t_new=np.matmul(P,t_new)
        #t_new=np.matmul(np.matmul(P,(R[0:3, 0:3]).T),t)

        if frame_nbr==0:
            print(R_new)
        Rs["Rotation_matrix_frame_"+str(frame_nbr)]=R_new
        Rs["translation_frame"+str(frame_nbr)]=t_new
        camera_locations[frame_nbr, :]=-np.matmul(R_new.T, t_new)
        vehicle_locations[frame_nbr,:]=[translation_x, translation_y, 0]#np.matmul(P,t)
        print(camera_locations[frame_nbr,:])
        # points[0, :] -= translation_x
        # points[1, :] -= translation_y
    path_colmap_dir= "Datasets/colmap/"
    camera_m, translation, camera_params = get_colmap_camera_matrixes(city, seq_nbr, path_colmap_dir)

    import matplotlib.pyplot as plt
    fig1 = plt.figure()
    plt.plot(camera_locations[:,0],camera_locations[:,2]  ,label="camera" )
    plt.plot(vehicle_locations[:, 0], vehicle_locations[:, 1], label="vehicle")
    plt.plot(translation[:, 0], translation[:, 2], label="colmap")

    # plt.ylim((-0.5,2))
    # plt.xlim((0, 45))
    plt.axis('equal')
    plt.legend()
    plt.show()

    return Rs




file="cityscapes_dataset/cityscapes_videos/leftImg8bit_sequence/train/aachen/aachen_000000_000019_leftImg8bit.png"


#file="cityscapes_dataset/cityscapes_videos/leftImg8bit_sequence/train/weimar/weimar_000120_000019_leftImg8bit.png"


R_s=reconstruct3D(file)
import scipy.io
scipy.io.savemat("camera_matrices_aachen.mat",R_s)
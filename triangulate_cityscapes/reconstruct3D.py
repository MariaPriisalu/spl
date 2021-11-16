from datetime import datetime
import json
import numpy as np
import os
from PIL import Image as Image
from scipy import io as sio
import cv2

from .camera import readCameraM
from .input import calcDepth,find_disparity
from poses.find3Dpose import  find_3D_poses
from .utils_3D import framepoints_to_collection, refine_voxel_values, find_cars
from .rescale import points_to_same_coord, rescale_pedestrians, rescale_cars, rescale3D



def reconstruct3D(file, constants, log_file=None):
    #import hotshot
    #prof = hotshot.Profile("stones.prof")
    #prof.start()

    print("Reconstruct 3D")
    #log_file.write(datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f') + " Reconstruct 3D \n")
    label_e = ""
    matlab_file = ""
    matlab_filename = ""
    path_label = ""
    left_img_path = ""

    points3D = {}
    points3D_RGB = {}
    label_hist = {}
    from collections import defaultdict
    if not constants.reconstruct_each_frame:
        points3D =  defaultdict(list)
        points3D_RGB = []
        label_hist = []

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
    path_camera = os.path.join("Datasets/cityscapes/camera_trainvaltest/camera", constants.mode_name,
                               city, camera_file)
    if not os.path.isfile(path_camera):
        #log_file.write(datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f') + " No camera file " + path_camera+ "\n")
        print("No camera file")
        print(path_camera)
        return None
    vehicle_file = city + "_" + seq_nbr + "_" + frame_nbr + "_vehicle.json"
    vehicle_path = os.path.join("Datasets/cityscapes/vehicle_sequence/vehicle_sequence",
                                constants.mode_name, city, vehicle_file)
    if not os.path.isfile(vehicle_path):
        #log_file.write(datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f') + " No vehicle file " + vehicle_path + "\n")
        print("No vehicle file")
        print(vehicle_path)
        return None
    timestamp_file = city + "_" + seq_nbr + "_" + frame_nbr + "_timestamp.txt"
    timestamp_path = os.path.join("Datasets/cityscapes/timestamp_sequence/", constants.mode_name, city,
                                  timestamp_file)
    if not os.path.isfile( timestamp_path):
        #log_file.write(datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f') + " No timestamp file " + timestamp_path + "\n")
        print("No timestamp file")
        print(timestamp_path)
        return None
    reconstruction_file_tensor = os.path.join("Datasets/cityscapes/pointClouds/", constants.mode_name,
                                                                  "tensor_" + city + "_" + seq_nbr + "_" + frame_nbr + ".npy")
    reconstruction_file=city + "_" + seq_nbr + "_" + frame_nbr + ".npz"
    reconstruction_path = os.path.join("Datasets/cityscapes/pointClouds/", constants.mode_name,
                                       reconstruction_file)

    reconstruction_old = os.path.join("Datasets/cityscapes/pointClouds/", constants.mode_name,
                                       city + "_" + seq_nbr + "_" + frame_nbr + "-full.npz")

    if os.path.isfile(reconstruction_path) and os.path.isfile(reconstruction_file_tensor):
        tensor=np.load(reconstruction_file_tensor)[()]
        loaded = np.load(reconstruction_path)
        print(reconstruction_path)
        if 'people' in loaded and 'cars' in loaded and 'scale' in loaded:
            print("ALl variables saved")
            if type(tensor) is dict:
                # log_file.write(
                #     datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f') + " ALl variables saved  \n")
                print("dictionary")
                return tensor, loaded['people'], loaded['cars'], loaded['scale']  # points, out_colors_seg, out_colors, poses_3d_dict

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

    range_reconstruction = np.array(
        [[float("inf"), -float("inf")], [float("inf"), -float("inf")], [float("inf"), -float("inf")]])
    # Go through the frames in the order 19, 18, 17, ...1, 0, 20, 21,...29, 30.
    if constants.same_coordinates:
        frames = list(range(19, -1, -1))
        frames += list(range(20, 30))
        for frame in frames:
            people.append([])
            cars.append([])
            # frames=range(30)
    else:
        frames = [int(frame_nbr)]

    # For all frames.
    for frame_nbr in frames:
        # Find path to vehicle accelerometer data.
        vehicle_file = city + "_" + seq_nbr + "_%06d_vehicle.json" % (frame_nbr)
        vehicle_path = os.path.join("Datasets/cityscapes/vehicle_sequence/vehicle_sequence",
                                    constants.mode_name, city, vehicle_file)
        # Load vehicle accelerometer data.
        car = json.load(open(vehicle_path))

        # File path to timestamp
        timestamp_file = os.path.join("Datasets/cityscapes/timestamp_sequence/", constants.mode_name,
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

        if frame_nbr != 19:
            # Change in the car's direction.
            yaw_est = +yaw_delta
        else:
            yaw_est = 0
            translation_x = 0
            translation_y = 0
        timestamp_prev = timestamp

        # Find the ground truth label path. for frame 20.
        path_label = os.path.join(constants.gt_path, constants.mode_name, city, file)
        # Find segmentation label.
        if not frame_nbr == 19:
            if constants.model == 1:
                label_e = os.path.join(constants.main_path, constants.mode_name, city,
                                       city + "_" + seq_nbr + "_" + "0000" + str(
                                           frame_nbr) + "_leftImg8bit.png")  # estimated label
            elif constants.model == 2:
                label_name = "%s_%s_%06d.png" % (city, seq_nbr, int(frame_nbr))
                label_e = os.path.join(constants.main_path, label_name)  # file[:-20]+".png")
                if constants.mode == 2:
                    label_e = os.path.join('cityscapes_dataset/cityscapes_prepared/results',
                                           city + "_" + seq_nbr + "_" + "0000" + str(frame_nbr) + ".png")
        # Find path to right image.
        right_img_path = os.path.join(constants.right_path, constants.mode_name, city,
                                      city + "_" + str(seq_nbr) + "_" + "0000" + str(frame_nbr) + "_rightImg8bit.png")
        # Find path to left image
        left_img_path = os.path.join(constants.img_path, constants.mode_name, city, basename)
        # If using Instance level labels as GT, then file mst be opened with Image.
        if constants.instance_level:
            iml = Image.open(path_label, mode='r')
            label = np.array(iml)
            classes = np.unique(label)
            iml.close()
        else:
            label = path_label

        # Decide on the end of the filename.
        if not constants.all_frames:
            left_img_path = left_img_path[:-len(constants.label_name)] + "_leftImg8bit.png"

        # Calculate of read disparity from file.
        left_img = cv2.imread(left_img_path)
        disp_path, disparity = find_disparity(city, constants, frame_nbr, left_img, right_img_path,
                                                                   seq_nbr)#segmentation_colors = visualize_segmentation(label, constants)
        valid_disp_mask = disparity > 0  # disparity 0 means not a valid value

        # Re-project image to 3D.
        img_3d, out_colors, out_seg, points = reproject_to_3D(K, P_inv, Q, R, constants, disp_path, disparity, exD, inD,
                                                              label, left_img_path, right_img_path, valid_disp_mask)

        #### Done with 3D reconstruction!!!!!!!!!!!
        # If needed convert to same coordinate system.
        if constants.same_coordinates:
            points, range_reconstruction = points_to_same_coord(points, range_reconstruction, translation_x, translation_y, yaw_est)


        depth, disparity = calcDepth(disp_path, exD['baseline'], inD)
        ## Go through cars.
        if not constants.instance_level:
            cars=find_cars(K, P_inv, R, cars, constants, depth, disparity, frame, label)#print "After Same coordinates "+str(datetime.datetime.now() - prev_timestamp)
        #prev_timestamp = datetime.datetime.now()

        #### Go through joint position poses_2d_dict.
        # Where to save poses_2d_dict?
        pred_file_name = os.path.join(constants.results_path, city + "_" + seq_nbr + "_pred.mat")
        poses_2d_dict = {}  # Holder for 2D joint positions.
        triangulated_poses_dict = {}


        if os.path.isfile(pred_file_name):
            #print "reading from file " + pred_file_name
            poses_2d_dict = sio.loadmat(pred_file_name)
        else :
            # log_file.write(
            #     datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')+ " Could not find file "+pred_file_name+ "\n")
            print("Could not find file "+pred_file_name)
        people = find_3D_poses(K, P_inv, R, constants, depth, frame, frame_nbr, img_3d, people, poses_2d_dict,
                               translation_x, translation_y, triangulated_poses_dict, yaw_est, log_file)

        points = np.around(points, decimals=1)
        # Add color and segmentation to each frame

        if constants.same_coordinates and constants.reconstruct_each_frame:
            add_points_to_label_histogram(constants, label_hist, out_colors, out_seg, points, points3D_RGB)
        if not constants.reconstruct_each_frame and constants.same_coordinates:
            points3D = framepoints_to_collection( out_colors, out_seg, points, points3D)

    scale=1.0#0.2
    rescale=False
    if constants.same_coordinates and rescale:
        tensor, scale_x, scale_y, scale_z= rescale3D(constants, range_reconstruction, points3D, points3D_RGB, label_hist)
        people=rescale_pedestrians(people, range_reconstruction, scale_x, scale_y, scale_z)
        cars=rescale_cars(cars, range_reconstruction, scale_x, scale_y, scale_z)
        np.savez_compressed(reconstruction_path, tensor=tensor, people=people,cars=cars , scale=[scale_x, scale_y, scale_z])
        return tensor, people, cars, range_reconstruction # points, out_colors_seg, out_colors, poses_3d_dict

    tensor = refine_voxel_values(points3D)
    np.save(reconstruction_file_tensor, tensor)
    np.savez_compressed(reconstruction_path,people=people, cars=cars, scale=range_reconstruction)
    return tensor,people,cars, range_reconstruction# points3D_RGB, label_hist


def add_points_to_label_histogram(constants, label_hist, out_colors, out_seg, points, points3D_RGB):
    i = 0
    for point in points.T:
        point = (point[0], point[1], point[2])
        if not out_seg[i] in constants.dynamic_labels:
            if not point in points3D_RGB:
                points3D_RGB[point] = []
                label_hist[point] = {}
            if out_seg[i] in label_hist[point]:
                label_hist[point][out_seg[i]] += 1
            else:
                label_hist[point][out_seg[i]] = 1
            points3D_RGB[point].append(out_colors[i])

        i += 1


def reproject_to_3D(K, P_inv, Q, R, constants, disp_path, disparity, exD, inD, label, left_img_path, right_img_path,
                    valid_disp_mask):
    right_img=cv2.imread(right_img_path)
    left_img = cv2.imread(left_img_path)
    if constants.opencv:
        img_3d = cv2.reprojectImageTo3D(disparity, np.float32(Q)) # x,y,z coord for each pixel
        out_points = img_3d[valid_disp_mask]
        # out_colors_seg = segmentation_colors[valid_disp_mask]
        out_colors = left_img[valid_disp_mask]
        if not constants.instance_level:
            label_val = cv2.imread(label)
            out_seg = label_val[valid_disp_mask][:, 0]
        else:
            out_seg = label[valid_disp_mask][:, 0]
        points = out_points.reshape(-1, 3)
        points = points.T
        points = np.dot(P_inv, points)
        points = np.dot(R, np.r_[points, np.ones((1, points.shape[1]))])
    else:
        depth, disparity = calcDepth(disp_path, exD['baseline'], inD)
        points = []
        out_colors_seg = []
        out_seg = []
        out_colors = left_img[valid_disp_mask]
        for y in range(0, disparity.shape[0]):
            for x in range(0, disparity.shape[1]):
                if disparity[y, x] > 0:
                    u = np.array([x, y, 1])
                    p_c = np.linalg.solve(K, u)
                    z_v = depth[y, x] * 1.0 / p_c[2]
                    p_c = p_c * z_v
                    p_c = np.dot(P_inv, p_c)
                    p_c = np.append(p_c, 1)
                    p_v = np.dot(R, np.array(p_c))
                    points.append(p_v)  # x, y, z
                    # out_colors_seg.append(segmentation_colors[y][x])
                    out_seg.append(label[y][x][0])
                    out_colors.append((left_img[y][x] + right_img[y][x + disparity[y, x]]) * .5)
    return img_3d, out_colors, out_seg, points
import numpy as np
import os, re, cv2
from scipy import stats

from plyfile import PlyData
from collections import Counter
from pyquaternion import Quaternion

#from PIL import Image

# Used methods.
from utils.constants import Constants
from utils.utils_functions import overlap
from poses.boundingboxes import find_people_bd
from triangulate_cityscapes.input import find_disparity
from triangulate_cityscapes.camera import readCameraM
from commonUtils.ReconstructionUtils import read_3D_pointcloud
from PIL import Image

# Set paths!.
# path_colmap_dir = "Datasets/colmap2/"
# img_path = "cityscapes_dataset/cityscapes_videos/leftImg8bit_sequence"  # images
# gt_path = "Datasets/cityscapes/gtFine/"  # Ground truth
# #results_path = "Datasets/cityscapes/colmap/FRCNN/"  # Where to find joint positions.
# vis_path = "Datasets/cityscapes/visual"
constants = Constants()

# initialize paths
path_label = ""
path_img = ""
mode_name = ""

# Coordinate change between vehicle and camera.
P = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]) # vehicle coord to camera
P_inv = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) # camera to vehicle

##
# Get camera matrix from colmap.
#
def get_colmap_camera_matrixes(city, seq_nbr, path_colmap_dir):
    camera_params, path_seq = get_colmap_camera_inner_matrix(city, path_colmap_dir,
                                                             seq_nbr)  # Get a mapping between image ids and image names
    camera_rotation, translation = get_colmap_camera_rotation_matrix(path_seq)  # Gather all feature positions
    return camera_rotation, translation, camera_params


##
# Get rotation matrices from colmap
# camera_m dictionary: key-filename value-rotation matrix
# translation - dictionary: key-filename value-translation vector
def get_colmap_camera_rotation_matrix(path_seq):
    camera_rotation_file = open(path_seq + '/images.txt', 'r')
    content = camera_rotation_file.readlines()
    camera_rotation_file.close()
    camera_m = {}
    translation = {}
    # Update camera positions.
    for line in content:
        if not '#' == line[0] and ".png" in line:
            params = line.split(" ")
            basename = os.path.basename(params[9].rstrip())
            v = Quaternion(float(params[1]), float(params[2]), float(params[3]), float(params[4])) #w x,y,z
            camera_m[basename] = v.rotation_matrix
            translation[basename] = np.array((float(params[5]), float(params[6]), float(params[7])))
            if float(params[5])==0 and float(params[6])==0 and float(params[7])==0:
                print("center "+basename)

    return camera_m, translation


##
# Get camera inner matrix parameters from colmap from colmap
#
def get_colmap_camera_inner_matrix(city, path_colmap_dir, seq_nbr):
    path_seq = path_colmap_dir + city + '_' + seq_nbr + '/'
    camera_file = open(path_seq + '/cameras.txt', 'r')
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
            camera_params[int(line[0]) - 1] = np.array(t)

    return camera_params, path_seq


#
# Extracts bounding boxes from csv file.
# Returns a dictionary with filename as key.
#
def get_bbx_from_csv(csv_path, reconstruction_dir):
    csv_file = open(csv_path, 'r')
    text = csv_file.read()
    rows = re.findall("^.*" + reconstruction_dir + ".*$", text, re.MULTILINE)

    csv_file.close()
    bbox_dict = {}
    if len(rows) > 0:
        [city, seq_nbr] = reconstruction_dir.split('_')
        for row in rows:
            bboxes=[_f for _f in re.split('[,\]\[ "\r]', row) if _f]#bboxes = row.split(',')
            filename = bboxes[0]#.split('"')
            bboxes_array = []
            counter = 0
            if len(bboxes) > 1:
                while counter * 6 + 1 < len(bboxes):#counter * 5 + 1 < len(bboxes):
                    bbox = bboxes[counter * 6 + 1:(counter + 1) * 6 + 1]

                    bboxes_array.append(
                        [int(float(bbox[0])), int(float(bbox[2])), int(float(bbox[1])), int(float(bbox[3])), float(bbox[4]),float(bbox[5])])#1,3,0,2
                    counter += 1
            bbox_dict[filename] = bboxes_array#bbox_dict[filename[1]] = bboxes_array
    return bbox_dict


def get_avg_colmap_baseline(camera_locations_colmap, camera_locations_right_colmap, camera_m, city, seq_nbr):
    # baseline
    avg_dist = 0
    count = 0
    baselines=[]
    for i in range(30):
        baselines.append(0)
    for i, cam_pos in enumerate(camera_locations_colmap):  # Should be  0.209313
        pos_colmap = camera_locations_colmap[i] - camera_locations_right_colmap[i]

        filename_left =  "%s_%s_%06d_leftImg8bit.png" % (city, seq_nbr,i)
        filename_right = "%s_%s_%06d_rightImg8bit.png" % (city, seq_nbr,i)
        if filename_left in camera_m and filename_right in camera_m:
            avg_dist += np.linalg.norm(pos_colmap)
            count += 1
            baselines[i]=np.linalg.norm(pos_colmap)

    if count==0:
        print("Missing stereo: "+str(camera_m))
        return 1.0, baselines
    # print "Avg dist "+str(avg_dist/count)
    #print baselines
    scale = 1.0 / (avg_dist / count)

    return scale, baselines

#
# Finds camera centers in the colmap coordinate system.
#
def order_colmap_camera_matrices(camera_m, translation):
    camera_locations_colmap = np.ones((30, 3))
    camera_locations_right_colmap = np.ones((30, 3))
    # camera_locations_colmap_dict = {}
    for camera_filename in camera_m:
        parts = camera_filename.split('_')
        frame_nbr = int(parts[2])
        R_loc = camera_m[camera_filename] # colmap camera matrix
        t = translation[camera_filename].reshape((3, 1)) # colmap rotation
        pos_colmap = - np.matmul(np.matrix(R_loc).T, t) # position in colmap coordinates
        if "right" in parts[3]:
            camera_locations_right_colmap[frame_nbr, :] = pos_colmap.T
        else:
            camera_locations_colmap[frame_nbr, :] = pos_colmap.T

            # camera_locations_colmap_dict[camera_filename] = pos_colmap
    return camera_locations_colmap, camera_locations_right_colmap  # camera_locations_colmap_dict,


# TO DO: triangulate the whole bounding box! Then average?

#
# Reconstruct people's positions, given bounding boxes in bbox.
#
def reconstruct_ppl(reconstrcution_dir, scale_vector, middle, bbox_dict, path_masks,cars_flag,colmap_path, image_size=(1024, 2048), to_cs=True):
    obj_list = []
    for frame in range(30):
        obj_list.append([])
    [city, seq_nbr] = reconstrcution_dir.split('_')
    camera_rotation,  translation, camera_params = get_colmap_camera_matrixes(city, seq_nbr, colmap_path)


    # Go through all files in this directory.
    for filename in bbox_dict:
        if len(bbox_dict[filename]) > 0 :
            mask_path=os.path.join(path_masks,filename)
            mask_global = cv2.imread(mask_path)
            frame_nbr = get_frame_nbr(filename)
            depth_map = get_depth_map(city, seq_nbr, frame_nbr, colmap_path)
            # Go though all bounding boxes.
            for bbox in bbox_dict[filename]:
                if bbox[1]-bbox[0]>20 and bbox[3]-bbox[2]>20:
                    #K, R, baseline = get_camera_matrix_IMU(city, mode_name, seq_nbr)
                    mask = get_sem_mask(bbox, cars_flag, mask_global)
                    est_depth=calculate_mode_depth_real(bbox, depth_map, mask)#calculate_mean_disparity(bbox, depth_map, right_bboxes)

                    if est_depth>0:
                        # Find middle of bbox.
                        x = (bbox[0] + bbox[1]) / 2.0
                        y = (bbox[2] + bbox[3]) / 2.0

                        u = np.array([x,y, 1])  # 2D point in camera coordinate system.

                        #b = baselines[int(frame_nbr)]
                        p_v = []
                        if filename in camera_rotation:

                            tran=-np.matmul(np.transpose(camera_rotation[filename]),np.reshape(translation[filename], (3, 1)))
                            rotational_matrix = np.concatenate((np.transpose(camera_rotation[filename]), tran),axis=1)

                            #rotational_matrix= np.concatenate((camera_rotation[filename], np.reshape(translation[filename], (3, 1))),axis=1)

                            p_v = triangulate_point(camera_params[0], rotational_matrix, est_depth, u)
                            if to_cs:
                                p_v=camera_to_cityscapes_coord(p_v, middle, scale_vector)

                        if len(p_v)>0:
                            if cars_flag :
                                obj_list[int(frame_nbr)].append(
                                    [int(p_v[0] - 4), int(p_v[0] + 4), int(p_v[1] - 4), int(p_v[1] + 4),
                                     int(p_v[2] - 4), int(p_v[2] + 4)])
                            else:
                                obj_list[int(frame_nbr)].append(np.array(
                                    [[int(p_v[0]) - 4, int(p_v[0]) + 4], [int(p_v[1]) - 4, int(p_v[1]) + 4], [int(p_v[2]) - 4, int(p_v[2]) + 4]]).reshape(3, 2))

    return obj_list


def get_sem_mask(bbox, cars_flag, mask_global):
    mask = []
    if bbox[-1] > 0:
        bbox_mask = mask_global[bbox[2]:bbox[3], bbox[0]:bbox[1], :]
        if cars_flag:
            bbox_mask = bbox_mask[:, :, 1]
        else:
            bbox_mask = bbox_mask[:, :, 0]
        mask = np.where(bbox_mask == int(bbox[-1]))
    return mask


#
# Triangulates a point
# returns in cityscapes coordinate system
def triangulate_point(K, R, est_disparity,  u):
    p_c = np.linalg.solve(K, u)  # 3D point in camera coordinate system.
    if est_disparity==0:
        return []
    #if colmap:
    p_c[2]=est_disparity
    p_c[0]=p_c[0]*p_c[2]#/K[0,0]
    p_c[1] = p_c[1] * p_c[2] #/ K[1,1]
    # else:
    #     p_c[2] = baseline*K[0,0]*1.0/est_disparity
    #     p_c[0] = p_c[0] * p_c[2] / K[0, 0]
    #     p_c[1] = p_c[1] * p_c[2] / K[1, 1]

    p_c = np.append(p_c, 1)
    p_v = np.dot(R, np.array(p_c))  # Middle of human.
    return p_v



#
# Returns the inner and rotation of camera matrices from cityscapes.
#
def get_camera_matrix_IMU(city, mode_name, seq_nbr):
    # Determine file paths to: camera matrix, and vehicle accelerometer/GPS data and the frame's timestamp.
    camera_file = city + "_" + seq_nbr + "_000019_camera.json"

    path_camera = os.path.join("Datasets/cityscapes/camera_trainvaltest/camera",
                               mode_name,
                               city, camera_file)
    if not os.path.exists(""):
        path_camera=os.path.join("Datasets/cityscapes/cameras",
                               mode_name,
                               city, camera_file)
    # Open vehicle and camera files.
    camera = open(path_camera)
    # read camera matrix and parameters.
    R, K, Q, exD, inD = readCameraM(camera)
    R_inv = np.transpose(R[0:3, 0:3])
    return K, R, exD['baseline']

#
# Return colmap's depth map
#
def get_depth_map(city, seq_nbr, frame_nbr, colmap_path):
    file_name_colmap ="left/"+ city + "_" + seq_nbr + "_" + frame_nbr + "_leftImg8bit.png"
    depth_map_path=os.path.join(colmap_path, city+"_"+seq_nbr,"dense/stereo/depth_maps/"+file_name_colmap+".geometric.bin")
    if not os.path.exists(depth_map_path):
        print("Does not exist: "+depth_map_path)
        return None
    depth_map = read_array(depth_map_path)
    return depth_map


#
# Colmap demo code.
#
def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

#
# Finds the mean disparity
#
import matplotlib.pyplot as plt
def calculate_mean_disparity(bbox, disparity, right_bboxes):
    person_area = disparity[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    mode_struct = stats.mode(person_area[person_area > 0], axis=None)
    if len(mode_struct.mode)==0:
        return 0

    avg_d = mode_struct.mode[0]
    return avg_d


def calculate_mode_depth(bbox, depth, mask):
    if depth is None:
        return 0
    person_area = depth[ bbox[2]:bbox[3],bbox[0]:bbox[1]]
    if len(mask) > 0 and len(mask[0]) > 0:
        person_area = person_area[mask]
        if person_area.size:
            person_area = person_area.flatten()
            person_area = person_area[np.nonzero(person_area)]
        else:
            return 0
        return statistics.mode(person_area)



import statistics
def calculate_mode_depth_real(bbox, depth, mask):
    if depth is None:
        return 0
    person_area = depth[ bbox[2]:bbox[3],bbox[0]:bbox[1]]

    if len(mask)>0 and len(mask[0])>0:
        person_area=person_area[mask]
    if person_area.size:
        person_area=person_area.flatten()
        person_area=person_area[np.nonzero(person_area)]
        if len(person_area)==0:
            return 0
        return statistics.median_grouped(person_area)
    else:
        return 0





#
# Get the disparity of a file
#
def get_disparity(city, frame_nbr, mode_name, disparity_dir, seq_nbr):


    # Read from file.
    disp_file = city + "_" + seq_nbr + "_" + frame_nbr+"_disparity.png" #% frame_nbr
    disp_path = os.path.join(disparity_dir,mode_name, city, disp_file)

    if os.path.exists(disp_path):
        #print disp_path
        disparity = np.array(Image.open(disp_path))
        disparity[disparity != 0] = (disparity[disparity != 0].astype(np.float) - 1.0) / 256
        return disparity

    return None
#
# Get file number
#
def get_frame_nbr(filename):
    parts_filename = filename.split('_')
    city = parts_filename[0]
    seq_nbr = parts_filename[1]
    frame_nbr = parts_filename[2]
    return frame_nbr


#
# Read reconstruction from colmap.
# Returns points, people and cars in cityscapes coordinate system
#
def reconstruct3D_ply(filename, settings,  training):
    mode_name = "train"
    [city, seq_nbr]=os.path.basename(filename).split('_')
    if city not in ['darmstadt', 'tubingen', 'weimar', 'bremen', 'krefeld',
                  'ulm', 'jena', 'stuttgart', 'erfurt', 'strasbourg',
                  'cologne', 'zurich', 'hanover', 'hamburg', 'aachen',
                  'monchengladbach', 'bochum', 'dusseldorf']:
        mode_name="val"
    # if not training:
    #    mode_name="val"
    csv_path=settings.path_masks
    # Read point cloud in colmap coordinate system.
    pointcloud_3D = read_3D_pointcloud(filename)
    try:
        x, y, z, nx, ny, nz, r, g, b, label = list(zip(*pointcloud_3D))
    except ValueError:
        #print pointcloud_3D
        print("Incorrect file" + filename + '/dense/fused_text.ply')
        return [], [], [], []

    # Rescale points.
    baselines, camera_locations_colmap, camera_locations_right_colmap, scale = get_scaling_factor(filename, mode_name,settings.colmap_path, settings.camera_path)
    print("Scale " + str(scale))
    print (" Semantic labels "+str(set(label)))

    # Find middle
    points = list(zip(x, y, z))

    middle = get_middle_point(camera_locations_colmap, camera_locations_right_colmap, label, y,points, settings.scale_x * scale)
    print("Middle:")
    print(middle)
    middle[2]=0
    #middle[1] =middle[1]

    # Rescale and center the reconstruction
    reconstruction_rescaled = rescaled_3D_pointcloud(middle, pointcloud_3D, scale, settings)

    # Reconstruct people

    #print os.path.basename(filename)
    bbox_dict = get_bbx_from_csv(os.path.join(csv_path,'people_bboxes.csv'), os.path.basename(filename))
    #print "people " + str(bbox_dict)


    people = reconstruct_ppl(os.path.basename(filename),
                             (settings.scale_y * scale, settings.scale_x * scale, settings.scale_z * scale), middle,
                             bbox_dict,path_masks=settings.path_masks, cars_flag=False,
                             colmap_path=settings.colmap_path)



    bbox_cars = get_bbx_from_csv(os.path.join(csv_path,'cars_bboxes.csv'),os.path.basename(filename))
    #print "cars "+str(bbox_cars)

    cars=reconstruct_ppl(os.path.basename(filename),
                             (settings.scale_y * scale, settings.scale_x * scale, settings.scale_z * scale), middle,
                             bbox_cars,path_masks=settings.path_masks, cars_flag=True,
                             colmap_path=settings.colmap_path)

    print("Done reconstruction")
    return reconstruction_rescaled, people, cars, scale, camera_locations_colmap, middle#, bbox_dict, bbox_cars


def rescaled_3D_pointcloud(middle, pointcloud_3D, scale, settings):
    rescaled_label_dict, rescaled_rgb_points_dict = rescale_points(pointcloud_3D, middle, (
    settings.scale_y * scale, settings.scale_x * scale, settings.scale_z * scale))
    reconstruction_rescaled = extract_most_popular_label_per_point(rescaled_label_dict, rescaled_rgb_points_dict)
    return reconstruction_rescaled


def get_middle_point(camera_locations_colmap, camera_locations_right_colmap, label, y,points, scale_total):
    i = 0
    mean_z = find_height_of_pavement(label, y)
    middle = camera_locations_colmap[i, :]
    middle[1] =mean_z#middle[1] #min(mean_z, middle[1])
    return middle


def get_scaling_factor(filename, mode_name, path_colmap_dir, camera_path):
    [city, seq_nbr] = os.path.basename(filename).split('_')
    camera_m, translation, camera_params = get_colmap_camera_matrixes(city, seq_nbr, path_colmap_dir)
    camera_locations_colmap, camera_locations_right_colmap = order_colmap_camera_matrices(camera_m, translation)
    scale, baselines = get_avg_colmap_baseline(camera_locations_colmap, camera_locations_right_colmap, camera_m, city,
                                               seq_nbr)
    print(scale)
    scale = rescale_scale(city, mode_name, scale, seq_nbr, camera_path)
    print(scale)
    return baselines, camera_locations_colmap, camera_locations_right_colmap, scale

def find_start_of_reconstruction(points, scale_total, middle):
    border = [128 / 2 * (1 / scale_total), 32 * (1 / scale_total), 256 * (1 / scale_total)]
    filtered=[]
    for p in points:
        if middle[0] - border[0] < p[0] and middle[0] + border[0] > p[0] and middle[1] < p[1] and middle[1] + border[
            1] > p[1]:
            filtered.append(p[2])
    # print "Filtered points:"+str(len(points))
    # print np.percentile(filtered, 1)
    return np.percentile(filtered, 1)


def find_height_of_pavement(label, y):
    indx = []
    for i, l in enumerate(label):
        if l == 22 or (5 < l and l < 11):
            indx.append(i)
    median_z = []
    for i in indx:
        median_z.append(y[i])
    mean_z = np.median(median_z)
    return mean_z


def rescale_scale(city, mode_name, scale, seq_nbr, camera_path):
    camera_file = city + "_" + seq_nbr + "_000019_camera.json"
    print(camera_path)
    path_camera = os.path.join(camera_path, mode_name, city,
                               camera_file)
    if not os.path.exists(path_camera):
        path_camera=os.path.join(camera_path+"/val", city,
                               camera_file)
    if not os.path.exists(path_camera):
        path_camera=os.path.join(camera_path+"/train", city,
                               camera_file)
    if not os.path.exists(path_camera):
        print("No file")
        return scale*0.22
    camera = open(path_camera)
    R, K, Q, exD, inD = readCameraM(camera)
    scale = scale * exD["baseline"]
    return scale

#
# Extract the most popular label and RGB after rescaling for each voxel. If tehre are multiple points then choose the most popular.
#
def extract_most_popular_label_per_point(rescaled_label_dict, rescaled_rgb_points_dict):
    reconstruction = {}
    for point in rescaled_rgb_points_dict:
        tuple = Counter(rescaled_rgb_points_dict[point]).most_common()[0][0]
        count_labels = Counter(rescaled_label_dict[point]).most_common()[0]
        label = count_labels[0]
        reconstruction[point] = ((tuple[0], tuple[1], tuple[2], label))
    return reconstruction


#
# Rescale points
# takes points in camera coordinate system.
#
def rescale_points(pointcloud_3D, middle, scale_vector, visualize=False):
    # middle, scale = find_middle(settings, x, y, z)

    rescaled_rgb_points_dict = {}
    rescaled_label_dict = {}
    for point in pointcloud_3D:
        p=np.array((point['x'], point['y'],point['z']))
        indx=tuple(camera_to_cityscapes_coord(p, middle, scale_vector[0]).astype(int))
        if indx not in rescaled_rgb_points_dict:
            rescaled_rgb_points_dict[indx] = []
            rescaled_label_dict[indx] = []
        rescaled_rgb_points_dict[indx].append((point['red'], point['green'], point['blue']))
        rescaled_label_dict[indx].append(point['label'])
    return rescaled_label_dict, rescaled_rgb_points_dict,


def visualize_distribution_of_points(border, filtered, middle, points_x, points_y, points_z):
    print("Middle calculated:" + str(np.mean(points_x)) + " " + str(np.mean(points_y)) + " " + str(np.mean(points_z)))
    print("Min calculated:" + str(np.min(points_x)) + " " + str(np.min(points_y)) + " " + str(np.min(points_z)))
    print("Max calculated:" + str(np.max(points_x)) + " " + str(np.max(points_y)) + " " + str(np.max(points_z)))
    fig1 = plt.figure()
    plt.hist(points_x, bins=50)
    plt.axvline(x=middle[0] - border[0], color='b', linestyle='dashed', linewidth=2)
    plt.axvline(x=middle[0] + border[0], color='b', linestyle='dashed', linewidth=2)
    plt.axvline(x=middle[0], color='r', linestyle='dashed', linewidth=2)
    plt.title("x")
    fig1.savefig("Valid_pavement_x.png")
    fig2 = plt.figure()
    plt.hist(points_y, bins=50)
    plt.axvline(x=middle[1] - border[1], color='b', linestyle='dashed', linewidth=2)
    plt.axvline(x=middle[1], color='r', linestyle='dashed', linewidth=2)
    plt.title("y")
    fig2.savefig("Valid_pavement_y.png")
    fig3 = plt.figure()
    plt.hist(points_z, bins=50)
    plt.axvline(x=middle[1] + border[2], color='b', linestyle='dashed', linewidth=2)
    plt.axvline(x=middle[2], color='r', linestyle='dashed', linewidth=2)
    plt.title("z")
    fig3.savefig("Valid_pavement_z.png")
    fig4 = plt.figure()
    plt.hist(filtered, bins=50)
    plt.axvline(x=middle[1] + border[2], color='b', linestyle='dashed', linewidth=2)
    plt.axvline(x=middle[2], color='r', linestyle='dashed', linewidth=2)
    plt.title("z")
    fig4.savefig("Filtered_z.png")
    fig5 = plt.figure()
    plt.scatter(points_x, points_y)
    plt.xlabel("x")
    plt.ylabel("y")
    fig5.savefig("Scatter_x_y.png")
    fig6 = plt.figure()
    plt.scatter(points_x, points_z)
    plt.xlabel("x")
    plt.ylabel("z")
    fig6.savefig("Scatter_x_z.png")
    fig7 = plt.figure()
    plt.scatter(points_z, points_y)
    plt.xlabel("z")
    plt.ylabel("y")
    fig7.savefig("Scatter_z_y.png")
    plt.show()
    print("Middle camera:" + str(middle))


def camera_to_cityscapes_coord(point, middle, scale_x):
    return np.matmul(P_inv,(point-middle))*scale_x
    #return np.matmul(P_inv,p)

#
# Find the center of the reconstruction.
#
def find_middle(settings, x, y, z):
    scale = (np.min(y), np.max(y))
    middle = [np.median(y) * settings.scale_y - (settings.height / 2),
              np.median(z) * settings.scale_z - (settings.depth / 2),
              np.median(x) * settings.scale_x - (settings.width / 2)]
    return middle, scale

from utils.constants import Constants
import glob, os, subprocess, sqlite3, json
from scipy import misc
import numpy as np
from PIL import Image as Image
from triangulate_cityscapes.movement import rotation
from triangulate_cityscapes.camera import readCameraM





def get_camera_matrices(city, seq_nbr, mode_name, frame_nbr='000019', nbr_frames=30):

    path_camera, timestamp_path, vehicle_path = get_paths(city, frame_nbr, mode_name, seq_nbr)

    # Open vehicle and camera files.
    vehicle_data = json.load(open(vehicle_path))
    camera = open(path_camera)
    timestamp_file = open(timestamp_path, 'r')

    M, R, camera_p, exD = get_camera_matrix_IMU(camera)
    # Set 19-th frame's reference vehicle position.
    vech_ref_pos = [vehicle_data["gpsLongitude"], vehicle_data["gpsLatitude"]]

    # Initialize.
    timestamp_prev = 0
    translation_x = 0
    translation_y = 0
    people = []
    cars = []
    yaws = []
    deltas = []
    accum_angle=0


    frames = list(range(0, nbr_frames))
    camera_locations = np.zeros((nbr_frames, 3))
    camera_locations_right = np.zeros((nbr_frames, 3))
    for frame in frames:
        people.append([])
        cars.append([])
        yaws.append([])
        deltas.append([])
    vehicle_poses=np.zeros((nbr_frames,3))
    camera_poses = np.zeros((nbr_frames, 3))
    camera_poses_r = np.zeros((nbr_frames, 3))



    # For all frames.
    for frame_nbr in frames:
        # Find path to vehicle accelerometer data.
        vehicle_file = city + "_" + seq_nbr + "_%06d_vehicle.json" % (frame_nbr)
        vehicle_path = os.path.join("Datasets/cityscapes/vehicle_sequence/vehicle_sequence",
                                    mode_name, city, vehicle_file)
        # File path to timestamp
        timestamp_file = os.path.join("Datasets/cityscapes/timestamp_sequence/", mode_name,
                                      city, city + "_" + seq_nbr + "_%06d_timestamp.txt" % (frame_nbr))

        # Load vehicle accelerometer data.
        if os.path.isfile(vehicle_path) and os.path.isfile(timestamp_file):
            car = json.load(open(vehicle_path))

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
            vehicle_poses[frame_nbr,:]=vehicle_pos[:]
            camera_left_pos=np.matmul(rotation(accum_angle),R[:,3]) # Rotated camera pos in vehicle coord
            camera_pos_right=np.matmul(rotation(accum_angle),R[:,3]+[0,-exD['baseline'],0])
            camera_poses[frame_nbr,:]=camera_left_pos+vehicle_pos
            camera_poses_r[frame_nbr, :] = camera_pos_right + vehicle_pos

            # Convert translation into first camera coordinates.
            camera_locations[frame_nbr,:]=np.matmul(M,camera_left_pos+vehicle_pos)+camera_p
            camera_locations_right[frame_nbr, :] =np.matmul(M,camera_pos_right + vehicle_pos)+camera_p
        else:
            print("Does not exist "+timestamp_file+ " "+vehicle_path)

    return camera_locations, camera_locations_right


def get_camera_matrix_IMU(camera):
    # read camera matrix and parameters.
    R, K, Q, exD, inD = readCameraM(camera)
    R_inv = np.transpose(R[0:3, 0:3])
    # Coordinate change between vehicle and camera.
    P = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    P_inv = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    M = np.matmul(P, R_inv)
    camera_p = np.matmul(P, -np.matmul(R_inv, R[:, 3]))  # vechicle center in camera coord.
    return M, R, camera_p, exD


def get_paths(city, frame_nbr, mode_name, seq_nbr):
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
    return path_camera, timestamp_path, vehicle_path


def write_img_list(path, city, seq_nbr,  cam_loc_left, camera_loc_right , nbr_frames):
    plot = False


    img_list = open(path+"/list_images.txt", 'w')

    for frame in range(nbr_frames):
        if os.path.isfile(path+'left/'+city + '_' + seq_nbr +'_%06d_leftImg8bit.png' %frame) and len(cam_loc_left[frame])>0:
            img_list.write('left/'+city + '_' + seq_nbr +'_%06d_leftImg8bit.png %f %f %f '%(frame, cam_loc_left[frame][0], cam_loc_left[frame][1],
                     cam_loc_left[frame][2]))
        if os.path.isfile(path + 'right/' + city + '_' + seq_nbr + '_%06d_rightImg8bit.png' % frame) and len(camera_loc_right[frame])>0:
            img_list.write('right/' + city + '_' + seq_nbr + '_%06d_rightImg8bit.png %f %f %f ' % (
            frame, camera_loc_right[frame][0], camera_loc_right[frame][1],
            camera_loc_right[frame][2]))
    img_list.close()

def remove_moving(city, seq_nbr, path, frame_nbr, nbr_frames):
    plot = False

    cam_loc_left, camera_loc_right = get_camera_matrices(city, seq_nbr, 'train',frame_nbr=frame_nbr, nbr_frames=nbr_frames)
    write_img_list(path, city, seq_nbr, cam_loc_left, camera_loc_right, nbr_frames)

    img_check_change = dict()
    connection = sqlite3.connect(path+'/database.db')
    cursor = connection.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")


    # Get a mapping between image ids and image names
    image_id_to_name = dict()
    cursor.execute('SELECT image_id, name FROM images;')

    # Gather image id to names mapping
    for row in cursor:
        image_id = row[0]
        name = row[1]
        image_id_to_name[image_id] = name

    # Update camera positions.
    for id in list(image_id_to_name.keys()):
        basename = os.path.basename(image_id_to_name[id])
        parts = basename.split('_')

        if 'left' in image_id_to_name[id]:
            cursor.execute(
                "UPDATE images SET  prior_tx = ?, prior_ty = ?, prior_tz = ? , camera_id=1 WHERE image_id = ?;",
                (cam_loc_left[int(parts[2])][0], cam_loc_left[int(parts[2])][1],
                 cam_loc_left[int(parts[2])][2], id))
        else:
            cursor.execute(
                "UPDATE images SET  prior_tx = ?, prior_ty = ?, prior_tz = ?, camera_id=2  WHERE image_id = ?;",
                (camera_loc_right[int(parts[2])][0], camera_loc_right[int(parts[2])][1],
                 camera_loc_right[int(parts[2])][2], id))

    # Gather all feature positions
    cursor.execute("SELECT image_id,rows,cols FROM descriptors;")
    features_dict = dict()
    for row in cursor:
        image_id = row[0]
        x = row[1]
        y = row[2]
        features_dict[image_id] = (x, y)

    # Look up segmentation
    for image_id in list(features_dict.keys()):

        name = image_id_to_name[image_id]

        basename = os.path.basename(name)
        parts = basename.split('_')
        city = parts[0]
        seq_nbr = parts[1]
        frame_nbr = parts[2]

        results_path = "Datasets/cityscapes/results/train/" + city
        results_path2 = "GRFP/results_right"
        pred_file_name = os.path.join(results_path, name[5:])
        pred_file_name_right = os.path.join(results_path2, name[6:])

        if "left" in name:
            # print pred_file_name
            if os.path.exists(pred_file_name):
                segmentation = misc.imread(pred_file_name)
        else:
            if os.path.exists(pred_file_name_right):
                # print pred_file_name_right
                segmentation = misc.imread(pred_file_name_right)
            else:
                disp_file = city + "_" + seq_nbr + "_" + frame_nbr + "_disparity.png"
                disp_path = os.path.join(
                    "Datasets/cityscapes/disparity_sequence_trainvaltest/disparity_sequence/",
                    "train", city, disp_file)
                disparity = np.array(Image.open(disp_path))
                disparity[disparity != 0] = (disparity[disparity != 0].astype(np.float) - 1.0) / 256
                pred_file_name = os.path.join(results_path, city + "_" + seq_nbr + "_" + frame_nbr + "_leftImg8bit.png")
                if os.path.isfile(pred_file_name):
                    segmentation = misc.imread(pred_file_name)
        if os.path.isfile(pred_file_name) or pred_file_name_right:

            segmentation_max = max(segmentation.flatten())
            train_indx = (segmentation_max == 255 or segmentation_max < 20)

            # For this image!
            cursor.execute("SELECT data, rows, cols FROM keypoints WHERE image_id=?;",
                           (image_id,))
            row = next(cursor)
            if row[0] is None:
                keypoints = np.zeros((0, 4), dtype=np.float32)
                descriptors = np.zeros((0, 128), dtype=np.uint8)
            else:
                keypoints = np.fromstring(row[0], dtype=np.float32).reshape(-1, 6)

                delete_rows = []
                deleted = [[], []]
                for i, pos in enumerate(keypoints[:, 0:2]):
                    x = int(pos[0])
                    y = int(pos[1])
                    if (train_indx and segmentation[y, x] > 10) or segmentation[y, x] > 23:
                        delete_rows.append(i)
                        deleted[0].append(x)
                        deleted[1].append(y)

                for i in sorted(delete_rows, reverse=True):
                    keypoints = np.delete(keypoints, i, 0)

                img_check_change[image_id] = len(keypoints)
                blob = keypoints.tobytes()
                rows = keypoints.shape[0]

                if len(delete_rows) > 0:
                    cursor.execute("UPDATE keypoints SET data =?, rows=?  WHERE image_id = ?;",
                                   (buffer(blob), rows, image_id))

                cursor.execute("SELECT data, rows, cols FROM descriptors WHERE image_id=?;",
                               (image_id,))
                row = next(cursor)
                descriptors = np.fromstring(row[0], dtype=np.uint8).reshape(-1, 128)
                for i in sorted(delete_rows, reverse=True):
                    descriptors = np.delete(descriptors, i, 0)

                blob = descriptors.tobytes()
                rows = descriptors.shape[0]

                if len(delete_rows) > 0:
                    cursor.execute("UPDATE descriptors SET data =?, rows=?  WHERE image_id = ?;",
                                   (buffer(blob), rows, image_id))
    connection.commit()
    cursor.close()
    connection.close()

# 'bochum', -/38150 frames
# 'hamburg', -106102 frames
# 'hanover', -058189 frames
# 'krefeld', -0/036299 frames
# 'monchengladbach',- _036139 frames and 002353 frames
# 'strasbourg',-36016 and 065572 frames
#
# Test:
# lindau -/58
# munster 0-172/173
# frankfurt -022797 frames and 050686 frames

if __name__ == "__main__":
    const=Constants()
    mode_name='train'
    number_of_dirs=len(glob.glob(const.gt_path+mode_name+'/*/*.png'))
    number_recons=0
    number_done=0


    for city in ['bochum', 'hamburg', 'hanover', 'krefeld', 'monchengladbach', 'strasbourg']:
        for filename in glob.glob(const.gt_path+mode_name+'/'+city+'*/*.png'):
            # If this is an image.

            if const.label_name in filename:  # 'gtFine_labelIds.png' in file:
                # Split name.
                segmentation_path='GRFP/results'
                segmentation_path_right='GRFP/results_right'
                basename = os.path.basename(filename)
                parts = basename.split('_')
                city = parts[0]
                seq_nbr = parts[1]
                frame_nbr = parts[2]
                path='Datasets/colmap/'+city+'_'+seq_nbr

                if True:#int(frame_nbr) == 19 :
                    if True:#not os.path.exists(path):
                        #if os.path.exists(path + '/dense') and not os.path.exists(path + '/dense/fused_text.ply'):

                        if not os.path.exists(path):
                            os.mkdir(path)
                        if not os.path.exists(path+'/images'):
                            os.mkdir(path+'/images')
                        if not os.path.exists(path + '/images/right'):
                            os.mkdir(path + '/images/right')
                            status = subprocess.call(
                                'cp ' + const.right_path + '/' + mode_name + '/' + city + '/' + city + '_' + seq_nbr + '* ' + path + '/images/right/',
                                shell=True)
                        if not os.path.exists(path+'/images/left'):
                            os.mkdir(path + '/images/left')
                            status = subprocess.call(
                                'cp ' + const.img_path + '/' + mode_name + '/' + city + '/' + city + '_' + seq_nbr + '* ' + path + '/images/left/',
                                shell=True)
                        if not os.path.exists(path + '/segmentation'):
                            os.mkdir(path + '/segmentation')
                        if not os.path.exists(path + '/segmentation/left'):

                            os.mkdir(path + '/segmentation/left')
                            os.mkdir(path + '/segmentation/right')
                            status = subprocess.call(
                                'cp ' + segmentation_path + '/' + city + '_' + seq_nbr + '* ' + path + '/segmentation/left/',
                                shell=True)
                            status = subprocess.call(
                                'cp ' + segmentation_path_right + '/' + city + '_' + seq_nbr + '* ' + path + '/segmentation/right/',
                                shell=True)
                        if not os.path.exists(path + '/sparse'):
                            os.mkdir(path + '/sparse')




                        print('Packages/colmap/colmap/build/src/exe/colmap  database_creator --database_path='+path+'/database.db')
                        status = subprocess.call('Packages/colmap/colmap/build/src/exe/colmap  database_creator --database_path='+path+'/database.db', shell=True)

                        # Determine file paths to: camera matrix, and vehicle accelerometer/GPS data and the frame's timestamp.
                        camera_file = city + "_" + seq_nbr + "_" + frame_nbr + "_camera.json"
                        path_camera = os.path.join("Datasets/cityscapes/camera_trainvaltest/camera", mode_name,
                                                   city, camera_file)

                        nbr_frames=len(glob.glob(path+"/images/left/*"))
                        print(nbr_frames)
                        # Open vehicle and camera files.
                        camera = open(path_camera)

                        # read camera matrix and parameters.
                        R, K, Q, exD, inD = readCameraM(camera)

                        # feature extraction
                        status = subprocess.call('Packages/colmap/colmap/build/src/exe/colmap feature_extractor --ImageReader.camera_model PINHOLE '+
                                                 '--ImageReader.camera_params '+str(inD['fx'])+','+str(inD['fy'])+','+str(inD['u0'])+','+str(inD['v0'])+' '+
                                                 '--ImageReader.default_focal_length_factor '+str(inD['fy']/2048.0)+' '+
                                                 '--database_path '+path+'/database.db --image_path '+path+'/images', shell=True)

                        remove_moving(city, seq_nbr, path, frame_nbr, nbr_frames)

                        # Matching
                        status = subprocess.call('Packages/colmap/colmap/build/src/exe/colmap exhaustive_matcher --database_path '+path+'/database.db'+
                                                 ' --SiftMatching.multiple_models 1 --ExhaustiveMatching.block_size 10'+
                                                 ' --SiftMatching.guided_matching 1 ', shell=True)

                        status = subprocess.call('Packages/colmap/colmap/build/src/exe/colmap sequential_matcher --database_path '+path+'/database.db '+
                                                 '--SequentialMatching.overlap 90  --SiftMatching.max_distance 0.9 '+
                                                 '--SequentialMatching.vocab_tree_path vocab_tree-1048576.bin'+
                                                 ' --SiftMatching.multiple_models 1 --SiftMatching.guided_matching 1 '+
                                                 '--SequentialMatching.loop_detection 1 ', shell=True)

                        status = subprocess.call('Packages/colmap/colmap/build/src/exe/colmap spatial_matcher --database_path '+path+'/database.db'+
                                                 ' --SiftMatching.max_distance 0.9 --SiftMatching.multiple_models 1'+
                                                 ' --SiftMatching.guided_matching 1  ', shell=True)

                        # Sparse reconstruction

                        status=subprocess.call('Packages/colmap/colmap/build/src/exe/colmap mapper --Mapper.min_focal_length_ratio 0.7  '+
                                               '--Mapper.max_focal_length_ratio 1.5 --database_path '+path+'/database.db '+
                                               '--image_path '+path+'/images --export_path '+path+'/sparse  '+
                                               '--Mapper.ba_refine_extra_params 1 --Mapper.init_max_forward_motion 1  '+
                                               '--Mapper.init_min_tri_angle 8  --Mapper.init_max_reg_trials 4 '+
                                               '--Mapper.abs_pose_min_inlier_ratio 0.05 --Mapper.filter_min_tri_angle 0.7 '+
                                               '--Mapper.init_num_trials 500', shell=True)


                        if os.path.isfile(path+'/sparse/0/cameras.bin') or os.path.isfile(path+'/sparse/cameras.bin'):

                            print('Packages/colmap/colmap/build/src/exe/colmap rig_bundle_adjuster --input_path ' + path + '/sparse/0 --output_path ' + path + '/sparse/0 ')
                            status = subprocess.call(
                                'Packages/colmap/colmap/build/src/exe/colmap rig_bundle_adjuster ' +
                                '--input_path ' + path + '/sparse/0 --output_path ' + path + '/sparse/0 --rig_config_path Datasets/colmap/rig_camera.json',
                                shell=True)

                            print('Packages/colmap/colmap/build/src/exe/colmap model_aligner --input_path ' + path + '/sparse/0 --output_path ' + path + '/sparse/0 --ref_images_path ' + path + "/images.txt")
                            status = subprocess.call('Packages/colmap/colmap/build/src/exe/colmap model_aligner '
                                                     '--robust_alignment_max_error 0.01 --input_path ' + path + '/sparse/0 --output_path '
                                                     + path + '/sparse/0 --ref_images_path ' + path + "/list_images.txt",
                                                     shell=True)

                            print('Packages/colmap/colmap/build/src/exe/colmap image_undistorter  --image_path ' + path + '/images ' + '--input_path ' + path + '/sparse/0 --output_path ' + path + '/dense ' + '--output_type COLMAP --max_image_size 2000')
                            status = subprocess.call(
                                'Packages/colmap/colmap/build/src/exe/colmap image_undistorter  --image_path ' + path + '/images ' +
                                '--input_path ' + path + '/sparse/0 --output_path ' + path + '/dense ' +
                                '--output_type COLMAP --max_image_size 2000', shell=True)

                            print('Packages/colmap/colmap/build/src/exe/colmap dense_stereo --workspace_path ' + path + '/dense --workspace_format COLMAP --DenseStereo.geom_consistency true')
                            status = subprocess.call(
                                'Packages/colmap/colmap/build/src/exe/colmap dense_stereo --workspace_path ' + path + '/dense ' +
                                '--workspace_format COLMAP --DenseStereo.geom_consistency true', shell=True)

                            print('Packages/colmap/colmap/build/src/exe/colmap dense_fuser --workspace_path ' + path + '/dense  --workspace_format COLMAP --input_type geometric --output_path ' + path + '/dense/fused.ply --output_path_text ' + path + '/dense/fused_text.ply --output_path_seg ' + path + '/dense/fused_seg.ply  --output_path_seg_col '+  path + '/dense/fused_seg_col.ply')
                            status = subprocess.call(
                                'Packages/colmap/colmap/build/src/exe/colmap dense_fuser --workspace_path ' + path + '/dense  ' +
                                '--workspace_format COLMAP --input_type geometric --output_path ' +
                                '' + path + '/dense/fused.ply --output_path_text ' + path + '/dense/fused_text.ply --output_path_seg_col ' + path
                                + '/dense/fused_seg_col.ply --output_path_seg ' + path + '/dense/fused_seg.ply', shell=True)

                            print('Packages/colmap/colmap/build/src/exe/colmap dense_mesher --input_path ' + path + '/dense/fused.ply --output_path ' + path + '/dense/meshed.ply')
                            status = subprocess.call(
                                'Packages/colmap/colmap/build/src/exe/colmap dense_mesher --input_path ' + path + '/dense/fused.ply' +
                                ' --output_path ' + path + '/dense/meshed.ply', shell=True)

                            print('Packages/colmap/colmap/build/src/exe/colmap dense_mesher --input_path ' + path + '/dense/fused_seg.ply --output_path ' + path + '/dense/meshed_seg.ply')
                            status = subprocess.call(
                                'Packages/colmap/colmap/build/src/exe/colmap dense_mesher --input_path ' + path + '/dense/fused_seg.ply' +
                                ' --output_path ' + path + '/dense/meshed_seg.ply',
                                shell=True)

                            print('Packages/colmap/colmap/build/src/exe/colmap dense_mesher --input_path ' + path + '/dense/fused_seg_col.ply --output_path ' + path + '/dense/meshed_col.ply')
                            status = subprocess.call(
                                'Packages/colmap/colmap/build/src/exe/colmap dense_mesher --input_path ' + path + '/dense/fused_seg_col.ply' +
                                ' --output_path ' + path + '/dense/meshed_col.ply', shell=True)

                            status = subprocess.call(
                                'colmap model_converter --input_path ' + path + '/sparse/0 --output_path '
                                + path + ' --output_type TXT',
                                shell=True)

                            number_done += 1

                        number_recons += 1
                        print("Ronstruction done for " + str(number_done) + "Out of " + str(number_recons) + " " + str(
                            number_done * 1.0 / number_recons))




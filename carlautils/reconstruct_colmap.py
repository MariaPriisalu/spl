from utils.constants import Constants
import glob, os, subprocess, sqlite3, json
from scipy import misc
import numpy as np
from PIL import Image as Image
from triangulate_cityscapes.movement import rotation
from triangulate_cityscapes.camera import readCameraM
import pickle





def get_camera_matrices(path):
    K = pickle.load(open(os.path.join(path, 'camera_intrinsics.p'), "rb"), encoding="latin1", fix_imports=True)
    cameras_dict = pickle.load(open(os.path.join(path, 'cameras.p'), "rb"), encoding="latin1", fix_imports=True)
    return K





# def write_img_list(path):
#     plot = False
#
#     img_list = open(path+"/list_images.txt", 'w')
#
#     for frame in range(30):
#         if os.path.isfile(path+'left/'+city + '_' + seq_nbr +'_%06d_leftImg8bit.png' %frame) and len(cam_loc_left[frame])>0:
#             img_list.write('left/'+city + '_' + seq_nbr +'_%06d_leftImg8bit.png %f %f %f '%(frame, cam_loc_left[frame][0], cam_loc_left[frame][1],
#                      cam_loc_left[frame][2]))
#         if os.path.isfile(path + 'right/' + city + '_' + seq_nbr + '_%06d_rightImg8bit.png' % frame) and len(camera_loc_right[frame])>0:
#             img_list.write('right/' + city + '_' + seq_nbr + '_%06d_rightImg8bit.png %f %f %f ' % (
#             frame, camera_loc_right[frame][0], camera_loc_right[frame][1],
#             camera_loc_right[frame][2]))
#     img_list.close()

def remove_moving( path):
    plot = False

    K, cameras_dict = get_camera_matrices(path)
    #write_img_list(path)

    img_check_change = dict()
    connection = sqlite3.connect(path+'/database.db')
    cursor = connection.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print((cursor.fetchall()))

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
        parts = basename.split('.png')
        frame=int(parts[0])
        print(image_id_to_name[id] + "  " + str(id))
        loc=cameras_dict[frame]['rotation'][0:3,3]
        cursor.execute(
            "UPDATE images SET  prior_tx = ?, prior_ty = ?, prior_tz = ?, camera_id=2  WHERE image_id = ?;",(loc[0], loc[1],loc[2] ,id))

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
        parts = basename.split('.png')
        seg_name=parts[0]+"_seg.png"
        pred_file_name = os.path.join(path, seg_name)


        # print pred_file_name
        if os.path.exists(pred_file_name):
            segmentation = misc.imread(pred_file_name)

        if os.path.isfile(pred_file_name):

            segmentation_max = max(segmentation.flatten())


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
                    if  segmentation[y, x] ==4 or segmentation[y, x]==10:
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



if __name__ == "__main__":
    const=Constants()

    number_done=0
    path='Packages/CARLA_0.8.2/PythonClient/recon3/'

    if  os.path.exists(path):
        #if os.path.exists(path + '/dense') and not os.path.exists(path + '/dense/fused_text.ply'):

        if not os.path.exists(path + '/sparse'):
            os.mkdir(path + '/sparse')

        print('Packages/colmap/colmap/build/src/exe/colmap  database_creator --database_path='+path+'/database.db')
        status = subprocess.call('colmap  database_creator --database_path='+path+'/database.db', shell=True)

        # Determine file paths to: camera matrix, and vehicle accelerometer/GPS data and the frame's timestamp.
        path_camera = os.path.join(path,  "cameras.p")

        # Open vehicle and camera files.
        camera = open(path_camera)

        # read camera matrix and parameters.

        K = pickle.load(open(os.path.join(path, 'camera_intrinsics.p'), "rb"), encoding="latin1", fix_imports=True)
        cameras_dict = pickle.load(open(os.path.join(path, 'cameras.p'), "rb"), encoding="latin1", fix_imports=True)

        # feature extraction
        print('colmap feature_extractor --ImageReader.camera_model PINHOLE '+'--ImageReader.camera_params '+str(K[0,0])+','+str(K[1,1])+','+str(K[0,2])+','+str(K[1,2])+'--ImageReader.default_focal_length_factor '+str(K[0,0]/800)+' --database_path '+path+'/database.db --image_path '+path+'/images')
        status = subprocess.call('colmap feature_extractor --ImageReader.camera_model PINHOLE '+
                                 '--ImageReader.camera_params '+str(K[0,0])+','+str(K[1,1])+','+str(K[0,2])+','+str(K[1,2])+' '+
                                 '--database_path '+path+'/database.db --image_path '+path+'/images', shell=True)
        # '--ImageReader.default_focal_length_factor '+str(K[0,0]/800)+' '+

        #remove_moving(city, seq_nbr, path)

        # Matching
        status = subprocess.call('colmap exhaustive_matcher --database_path '+path+'/database.db', shell=True)
        #                          ' --SiftMatching.multiple_models 1 --ExhaustiveMatching.block_size 10'+
        #                          ' --SiftMatching.guided_matching 1 ', shell=True)

        status = subprocess.call('colmap sequential_matcher --database_path '+path+'/database.db ', shell=True)
                                 # '--SequentialMatching.overlap 90  --SiftMatching.max_distance 0.9 '+
                                 # '--SequentialMatching.vocab_tree_path vocab_tree-1048576.bin'+
                                 # ' --SiftMatching.multiple_models 1 --SiftMatching.guided_matching 1 '+
                                 # '--SequentialMatching.loop_detection 1 '

        status = subprocess.call('colmap spatial_matcher --database_path '+path+'/database.db', shell=True)
                                 # ' --SiftMatching.max_distance 0.9 --SiftMatching.multiple_models 1'+
                                 # ' --SiftMatching.guided_matching 1  ', shell=True)

        # Sparse reconstruction

        status=subprocess.call('colmap mapper --database_path '+path+'/database.db '+
                               '--image_path '+path+'/images --export_path '+path+'/sparse  ', shell=True)


        if os.path.isfile(path+'/sparse/0/cameras.bin') or os.path.isfile(path+'/sparse/cameras.bin'):

            # print 'Packages/colmap/colmap/build/src/exe/colmap rig_bundle_adjuster --input_path ' + path + '/sparse/0 --output_path ' + path + '/sparse/0 '
            # status = subprocess.call(
            #     'Packages/colmap/colmap/build/src/exe/colmap rig_bundle_adjuster ' +
            #     '--input_path ' + path + '/sparse/0 --output_path ' + path + '/sparse/0 --rig_config_path Datasets/colmap/camera.json',
            #     shell=True)

            # print 'Packages/colmap/colmap/build/src/exe/colmap model_aligner --input_path ' + path + '/sparse/0 --output_path ' + path + '/sparse/0 --ref_images_path ' + path + "/images.txt"
            # status = subprocess.call('Packages/colmap/colmap/build/src/exe/colmap model_aligner --input_path ' + path + '/sparse/0 --output_path '
            #                          + path + '/sparse/0 --ref_images_path ' + path + "/list_images.txt",
            #                          shell=True)

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

            number_done += 1

        #number_recons += 1
        #print "Ronstruction done for " + str(number_done) + "Out of " + str(number_recons) + " " + str(
        #    number_done * 1.0 / number_recons)




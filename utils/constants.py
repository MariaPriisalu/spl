
import os

from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from sklearn.neighbors import NearestNeighbors

def proc(v,u):

    _ ,_ , disparity =procrustes(u.reshape(-1,3), v.reshape(-1,3)) # np.norm(u.reshape(-1,3)- v.reshape(-1,3))
    return disparity


class Constants(object):

    def thresholdDist(v,u):

        dist=abs(u-v)
        dist_out=0
        for row in dist:
            if row>400:
                dist_out+=400
            else:
                dist_out+=row
        return dist_out

    def __init__(self):
        ## Some options:
        # Visualize images
        # or evaluate all.
        self.visualize=True
        self.show_imgs=True#False#True
        self.instance_level= True
        self.posenet=False


        self.reconstruct_each_frame=False

        # Which model for segmentation?
        # Model 0 -GT
        # Model 1- Dilational net.
        # Model 2- GRFP
        # Model 3- Fast RCNN
        # Model 4 - Openpose without segmentation.
        self.model=0

        # Which model for segmentation?
        # Model 0 -PoseMachines
        # Model 1- OpenPose
        self.pose_model=0
        self.modes=[0,2]
        self.modes_iter=0
        self.mode=self.modes[self.modes_iter]
        self.all_frames= False

        if  self.model==0: # When only bounding boxes are given this cannot be calulated!
            self.all_frames=False

        self.gt_label_name="_gtFine_labelIds.png"
        if self.instance_level:
            self.gt_label_name="_gtFine_instanceIds.png"

        self.label_name="_gtFine_labelIds.png"
        if self.instance_level:
            self.label_name="_gtFine_instanceIds.png"

        if self.all_frames:
            #self.instance_level=False
            self.label_name="_leftImg8bit.png"


        # Debug mode?
        self.debug=0

        # Display images?
        self.display=0

        # Recalculate the poses even in they already exist?
        self.recalculate=True#False
        self.recalculate_csv=True
        # if self.model==0:
        #     self.recalculate_city="tubingen"
        #     self.recalculate_seq=54
        #     self.recalculate_frame=19
        # elif self.model==1:
        #     self.recalculate_city="krefeld"
        #     self.recalculate_seq=0
        #     self.recalculate_frame=23510
        # elif self.model==3:
        #     self.recalculate_city="weimar"
        #     self.recalculate_seq=40
        #     self.recalculate_frame=19
        self.recalculate_city=""
        # Write to file the predictions, and heatmaps?
        self.write_to_file_pred=True # Predictions of joint positions.
        self.write_to_file_heat=True # Heatmap outputs of pm.

        self.human_bbx_path="Datasets/cityscapes/pannet2/"#"Datasets/cityscapes/results/FCRNN/bboxes_all_train.csv"
        if not os.path.exists(self.human_bbx_path):
            self.human_bbx_path="Datasets/colmap/pannet/"

        if self.visualize and  self.all_frames:
            self.cities=['aachen'] #'darmstadt', 'tubingen', 'weimar',
            self.recalculate_csv=False
            self.write_to_file_pred=False # Predictions of joint positions.
            self.write_to_file_heat=False

        # Maximum allowed distance between pixels in the same clusters.
        self.max_dist=2
        if self.model==1:
            self.max_dist=5

        # Minimum percentage of clusters allowed to overlap
        self.dif_clusters=0.05
        if self.model==1:
            self.dif_clusters=0.1

        # Minimum width of cluster
        self.cluster_min_x=1
        if self.model==1 or self.model==2:
            self.cluster_min_x=1


        # Minimum height of cluster
        self.cluster_min_y=1
        if self.model==1:
            self.cluster_min_y=1#30
        if self.model==2:
            self.cluster_min_y=1# 10


        self.boxsize=368

        self.threshold=0.5

        # Cities in different training sets
        self.train_extra=['nuremberg', 'konigswinter', 'muhlheim-ruhr',
                        'schweinfurt', 'saarbrucken', 'mannheim', 'heidelberg',
                         'bad-honnef', 'dortmund', 'troisdorf', 'bayreuth',
                          'wurzburg', 'oberhausen', 'bamberg', 'dresden',
                          'augsburg', 'erlangen', 'freiburg', 'duisburg',
                          'karlsruhe', 'wuppertal', 'heilbronn', 'konstanz']

        self.train=['darmstadt', 'tubingen', 'weimar', 'bremen', 'krefeld',
                    'ulm', 'jena', 'stuttgart', 'erfurt', 'strasbourg',
                     'cologne', 'zurich', 'hanover', 'hamburg', 'aachen',
                     'monchengladbach', 'bochum', 'dusseldorf']
        self.test=['bielefeld', 'berlin', 'mainz', 'leverkusen', 'munich', 'bonn']
        self.val=['frankfurt', 'munster', 'lindau']

        # Load the deep net model
        if self.posenet:
            import caffe
            self.net = caffe.Net('Code/convolutional-pose-machines-release-master/model/_trained_MPI/pose_deploy_centerMap.prototxt','Code/convolutional-pose-machines-release-master/model/_trained_MPI/pose_iter_985000_addLEEDS.caffemodel',caffe.TEST)
            self.net.blobs['data'].reshape(1,4,368,368)

        # Labels!
        if os.path.exists('GRFP'):
            import pickle
            f = open('GRFP/GRFP_compact/misc/cityscapes_labels.pckl', 'rb')
            self.cs_id2trainid, self.cs_id2name = pickle.load(f, encoding="latin1", fix_imports=True)
            f.close()

        ########################################################################################################################## CODE BEGINS HERE
        self.gt_human_label=24
        self.gt_biker_label=25
        self.gt_car_labels=[26,27,28,29,30,31,32,33,1]

        self.human_label=11
        self.biker_label=12
        self.bike_label=17
        self.motorcycle_label=18
        self.car_labels = [13,14,15,16,17,18]


        if self.model==2 or self.model==0:
            self.human_label=24
            self.biker_label=25
            self.bike_label=32
            self.motorcycle_label=33
            self.car_labels=[26,27,28,29,30,31,32,33,1]

        self.dynamic_labels=self.car_labels+[self.motorcycle_label]+[self.bike_label]+[self.biker_label]+[self.human_label]

        self.img_path="cityscapes_dataset/cityscapes_videos/leftImg8bit_sequence" # images
        self.gt_path="Datasets/cityscapes/gtFine/" # Ground truth
        self.results_path="Datasets/cityscapes/poses" # Where to find joint positions.
        self.vis_path="Datasets/cityscapes/visual"
        self.right_path="Datasets/cityscapes/rightImg8bit_sequence_trainvaltest/rightImg8bit_sequence"

        self.main_path=""
        if self.model==1:
            self.main_path="Datasets/cityscapes/results/" # Labels
            self.model_name="Dilational"
        if self.model==2:
            self.main_path="cityscapes_dataset/cityscapes_prepared/results"
            self.model_name="GRFP"
            self.results_path="Datasets/cityscapes/poses"
        if self.model==3:
            self.model_name="FRCNN"
            self.main_path="Datasets/cityscapes/results/FCRNN/bboxes_"
        if self.model==0:
            self.model_name="GT"
            self.main_path=self.gt_path
        self.results_path=os.path.join(self.results_path, self.model_name)

        self.disparity_read=True # Read disparity from file?
        self.same_coordinates=True # Same coordinates between frames?
        self.opencv=True # Use opencv for 3D reconstruction.
        self.uniform=True # Clip too long limbs in skeleton 3D reconstruction?
        self.scaling=0# 0=hip, 1= back , 2= height
        self.reproject=True # Reproject the 3D skeleton pose onto 2D.
        # initialize paths
        # if os.path.exists("Datasets/cityscapes/poses/poses.mat"):
        #     import scipy.io as sio
        #     import numpy as np
        #     self.poses=sio.loadmat("Datasets/cityscapes/poses/poses.mat")
        #     if self.scaling==0:
        #         self.poses=sio.loadmat("Datasets/cityscapes/poses/poses_hips.mat")
        #     self.poses=np.delete(self.poses["poses"],[24,25,26], 1)
        #     self.nearest_neigbor = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric=proc).fit(self.poses)

        # Deterine dataset.

        if self.mode==0:
            self.mode_name="train"
        elif self.mode==1:
            self.mode_name="test"
        elif self.mode==2:
            self.mode_name="val"
        elif self.mode==3:
            self.mode_name="train_extra"
        if self.mode==0 and self.model==2:
            self.main_path="GRFP/results"


        # Set correct path accoring to the mode.
        if self.model==1:
            self.main_path_m=os.path.join(self.main_path, self.mode_name)
        elif self.model==2:
            self.main_path_m=self.main_path
        elif self.model==3:
            self.main_path_m=self.main_path+self.mode_name+".csv"
        self.img_path_m=os.path.join(self.img_path, self.mode_name)

        # Csv reader for reading bounding boxes from file.
        self.csv_bbox=None
        if self.model==3:
            import csv
            self.csv_bbox=csv.reader(open(self.main_path_m, 'r'), delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        # Calculate error for onlt GT frames?
        self.search_dir=self.img_path_m
        if not self.all_frames:
            self.search_dir=os.path.join(self.gt_path, self.mode_name)


        self.ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''

        self.ply_header_skeleton = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        element edge 13
        property int vertex1
        property int vertex2
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''

        self.ply_header_skeleton2 = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        element edge 26
        property int vertex1
        property int vertex2
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''

        self.skeleton_edges='''
        0 1 0 0 255
        1 2  0 170 255
        2 3 0 255 170
        3 4 0 255 0
        1 5 170 255 0
        5 6 255 170 0
        6 7 255 0 0
        8 11 255 255 255
        8 9 255 255 255
        9 10 255 255 255
        11 12 255 0 170
        12 13 170 0 255
        1 14 255 255 255
        '''

        self.skeleton_edges2='''
        0 1 0 0 255
        1 2  0 170 255
        2 3 0 255 170
        3 4 0 255 0
        1 5 170 255 0
        5 6 255 170 0
        6 7 255 0 0
        8 11 255 255 255
        8 9 255 255 255
        9 10 255 255 255
        11 12 255 0 170
        12 13 170 0 255
        1 14 255 255 255
        15 16 0 0 255
        16 17  0 170 255
        17 18 0 255 170
        18 19 0 255 0
        16 20 170 255 0
        20 21 255 170 0
        21 22 255 0 0
        23 26 255 255 255
        23 24 255 255 255
        24 25 255 255 255
        26 27 255 0 170
        27 28 170 0 255
        16 29 255 255 255
        '''
    def nextMode(self):
        self.modes_iter+=1
        if self.modes.len()>self.modes_iter:
            self.mode=self.modes[self.modes_iter]
            if self.mode==0:
                self.mode_name="train"
            elif self.mode==1:
                self.mode_name="test"
            elif self.mode==2:
                self.mode_name="val"
            elif self.mode==3:
                self.mode_name="train_extra"
            if self.mode==0 and self.model==2:
                self.main_path="GRFP/results"
            # Set correct path accoring to the mode.
            if self.model==1:
                self.main_path_m=os.path.join(self.main_path, self.mode_name)
            elif self.model==2:
                self.main_path_m=self.main_path
            elif self.model==3:
                self.main_path_m=self.main_path+self.mode_name+".csv"
            self.img_path_m=os.path.join(self.img_path, self.mode_name)

            # Csv reader for reading bounding boxes from file.
            self.csv_bbox=None
            if self.model==3:
                import csv
                self.csv_bbox=csv.reader(open(self.main_path_m, 'r'), delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

            # Calculate error for only GT frames?
            self.search_dir=self.img_path_m
            if not self.all_frames:
                self.search_dir=os.path.join(self.gt_path, self.mode_name)

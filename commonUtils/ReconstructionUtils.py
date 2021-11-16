# D######################
# DO NOT FORGET ABOUT FORCE RECOMPUTE AND SLACE STUFF !!!!!!!!! + FRAME break!
# TODO: add some discretization *10 , one decimal etc to improve space

from plyfile import PlyData, PlyElement
import glob
import os.path
import numpy as np
import pickle
import csv
import copy
import scipy as sc
import math
import copy

#from commonUtils.RealTimeEnv import CarlaRealTimeUtils

# Coordinate change between vehicle and camera.
# Note: this is for cityscapes, with forward axis on X, Z up.
# Basically camera space from Carla to Citiscapes (see their calibration document).
# They use a different camera system. Sequence of operations:
#   cityscapes_camera_pos = P_inv * carla_camera_pos * [-1, -1, 1]
# BUT don't forget that carla data in our dataset has all points in camera space (which means that they are like    pos in camera space = (pos in vehicle space ) * P
# To transform them back to vehicle space =>  pos in camera space * P_inv
P = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])  # vehicle coord to camera
P_inv = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # camera to vehicle

# These are the labels mapping from Carla to Citiscapes
# NOTE: we are using by default Carla segmentation label because it is the most simplest mapping possible. We do not map cars and people.
# So every dataset we get should map and save their data in Carla space !!!

# Mapping of CARLA labels to Cityscapes
# Carla label       # Cityscapes label  name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
# 0 	None		Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# 1 	Buildings	Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
# 2 	Fences		Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
# 3 	Other		Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# 4 	Pedestrians	Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
# 5 	Poles		Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
# 6 	RoadLines	Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
# 7 	Roads		Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
# 8 	Sidewalks	Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
# 9 	Vegetation	Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
# 10 	Vehicles	Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
# 11 	Walls		Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
# 12    TrafficSign Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
# 13    Sky         Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
# 14    Ground      Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
# 15    Bridge      Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
# 16    RailTrack   Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
# 17    GuardRail   Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
# 18    TrafficLight    Label(  'traffic light'    , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
# 19    Static      Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# 20    Dynamic     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
# 21    Water       Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# 22    Terrain     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
# 23    Crosswalk   Label(  'crosswalk'              , 34 ,        - , 'flat'          , 1       , False        , False        , (255, 255, 255)),
# 24    Sidewalk extra Label(  'crosswalk_extra'              , 35,        - , 'flat'          , 1       , False        , False        , (255, 255, 0)),

SPECIAL_LABEL_CROSSWALKS = 34 # This doesn't exist in the citiscapes !
SPECIAL_LABEL_SIDEWALK_EXTRA = 35 # Extra sidewalks such as grass
LAST_CITYSCAPES_SEMLABEL = SPECIAL_LABEL_SIDEWALK_EXTRA

carla_label_colours = [(0, 0, 0),  # 0
                       (70, 70, 70),  # 1
                       (190, 153, 153),  # 2
                       (0, 0, 0),  # 3
                       (220, 20, 60),  # 4
                       (153, 153, 153),  # 5
                       (128, 64, 128),  # 6
                       (128, 64, 128),  # 7
                       (244, 35, 232),  # 8,
                       (107, 142, 35),  # 9,
                       (0, 0, 142),  # 10
                       (102, 102, 156),  # 11
                       (220, 220, 0),  # 12
                        (70, 130, 180), # 13
                        (81, 0, 81), # 14
                        (150, 100, 100), # 15
                        (230, 150, 140), # 16
                        (180, 165, 180), # 17
                        (250, 170, 30), # 18
                        (110, 190, 160), # 19
                        (170, 120, 50), # 20
                        (45, 60, 150), # 21
                        (145, 170, 100), # 22
                        (255, 255, 255), # 23
                        (255, 255, 0) # 24
                       ]

carla_labels = ['unlabeled',  # 0
                'building',  # 1
                'fence',  # 2
                'other',  # 3
                'person',  # 4
                'pole',  # 5
                'roadlines',  # 6 -roadlines
                'road',  # 7
                'sidewalk',  # 8
                'vegetation',  # 9
                'car',  # 10
                'wall',  # 11
                'traffic sign',  # 12
                'sky', # 13
                'ground', # 14- new here onwards
                'bridge', # 15
                'rail track', # 16
                'guard rail', # 17
                'traffic light', # 18
                'static', # 19
                'dynamic', # 20
                'water', # 21
                'terrain', # 22
                'crosswalk', # 23 - additional label,  not present in carla
                'sidewalk_extra' # 24 - additional label, not present in carla
                ]

def fillEmptyMappings(mappingDS):
    missingItems = LAST_CITYSCAPES_SEMLABEL - len(mappingDS) + 1
    assert missingItems >= 0
    mappingDS.extend([None]*missingItems)
    assert len(mappingDS) == LAST_CITYSCAPES_SEMLABEL + 1
    return mappingDS

# Fill the things for the extra classes in CARLA mappings
carla_labels = fillEmptyMappings(carla_labels)
carla_label_colours = fillEmptyMappings(carla_label_colours)
carla_labels[SPECIAL_LABEL_SIDEWALK_EXTRA] ="sidewalk_extra"
carla_labels[SPECIAL_LABEL_CROSSWALKS] = "crosswalk"
carla_label_colours[SPECIAL_LABEL_CROSSWALKS] = (255, 255, 255)
carla_label_colours[SPECIAL_LABEL_SIDEWALK_EXTRA] = (255, 255, 0)

carla_labels_dict={}
for i, label in enumerate(carla_labels):
    if label!=None:
        carla_labels_dict[label]=i




cityscapes_colours = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                      (111, 74, 0), (81, 0, 81), (128, 64, 128), (244, 35, 232), (250, 170, 160),
                      (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153), (180, 165, 180),
                      (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
                      (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                      (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90),
                      (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)]

cityscapes_labels = ['unlabeled', # 0
                     'ego vehicle', # 1
                     'rectification border', # 2
                     'out of roi', # 3
                     'static', # 4
                     'dynamic', # 5
                     'ground',# 6
                     'road', # 7
                     'sidewalk',# 8
                     'parking', # 9
                     'rail track', # 10
                     'building', # 11
                     'wall', # 12
                     'fence', # 13
                     'guard rail', # 14
                     'bridge', # 15
                     'tunnel', # 16
                     'pole',# 17
                     'polegroup',# 18
                     'traffic light', # 19
                     'traffic sign', # 20
                     'vegetation', # 21
                     'terrain', # 22
                     'sky', # 23
                     'person', # 24
                     'rider', # 25
                     'car', # 26
                     'truck', # 27
                     'bus', # 28
                     'caravan', # 29
                     'trailer', # 30
                     'train', # 31
                     'motorcycle', # 32
                     'bicycle'] # 32

# Add the extra classes
cityscapes_labels = fillEmptyMappings(cityscapes_labels)
cityscapes_colours = fillEmptyMappings(cityscapes_colours)
cityscapes_labels[SPECIAL_LABEL_SIDEWALK_EXTRA] ="sidewalk_extra"
cityscapes_labels[SPECIAL_LABEL_CROSSWALKS] = "crosswalk"
cityscapes_colours[SPECIAL_LABEL_CROSSWALKS] = (255, 255, 255)
cityscapes_colours[SPECIAL_LABEL_SIDEWALK_EXTRA] = (255, 255, 0)
print (cityscapes_labels)
cityscapes_labels_dict={}
for i, label in enumerate(cityscapes_labels):
    cityscapes_labels_dict[label]=i

def getLabelFromPoint(t):
    return t[6] # 6 is the current index for label inside a cloud point structure

def getRgbFromPoint(t):
    return t[3], t[4], t[5] # where rgb indices are...

# Mapping from CARLA labels to CITYSCAPES labels
def get_label_mapping():
    label_mapping = {}
    label_mapping[carla_labels_dict['unlabeled']] = cityscapes_labels_dict['unlabeled']  # None
    label_mapping[carla_labels_dict['building']] =cityscapes_labels_dict['building'] # Building
    label_mapping[carla_labels_dict['fence']] = cityscapes_labels_dict['fence']  # Fence
    label_mapping[carla_labels_dict['other']] = cityscapes_labels_dict['static']  # Other/Static
    label_mapping[carla_labels_dict['static']] = cityscapes_labels_dict['static']  # Other/Static
    label_mapping[carla_labels_dict['person']] = cityscapes_labels_dict['person'] # Pedestrian
    label_mapping[carla_labels_dict['pole']] = cityscapes_labels_dict['pole']  # Pole
    label_mapping[carla_labels_dict['road']] = cityscapes_labels_dict['road']  # Roadlines
    label_mapping[carla_labels_dict['roadlines']] = cityscapes_labels_dict['road']  # Road
    label_mapping[carla_labels_dict['sidewalk']] = cityscapes_labels_dict['sidewalk']  # Sidewalk
    label_mapping[carla_labels_dict['vegetation']] = cityscapes_labels_dict['vegetation']  # Sidewalk
    label_mapping[carla_labels_dict['car']] = cityscapes_labels_dict['car' ]  # Vehicles
    label_mapping[carla_labels_dict['wall']] = cityscapes_labels_dict['wall']  # Vehicles
    label_mapping[carla_labels_dict['traffic sign']] = cityscapes_labels_dict['traffic sign']  # Vehicles
    label_mapping[carla_labels_dict['sky']] = cityscapes_labels_dict['sky']  # Vehicles
    label_mapping[carla_labels_dict['ground']] = cityscapes_labels_dict['ground']  # Ground
    label_mapping[carla_labels_dict['bridge']] = cityscapes_labels_dict['bridge']  # Bridge
    label_mapping[carla_labels_dict['rail track']] = cityscapes_labels_dict['rail track']  # Bridge
    label_mapping[carla_labels_dict['guard rail']] = cityscapes_labels_dict['guard rail']  # Bridge
    label_mapping[carla_labels_dict['traffic light']] = cityscapes_labels_dict['traffic light']  # Bridge
    label_mapping[carla_labels_dict['dynamic']] = cityscapes_labels_dict['dynamic']  # Bridge
    label_mapping[carla_labels_dict['water']] = cityscapes_labels_dict['static']  # Bridge
    label_mapping[carla_labels_dict['terrain']] = cityscapes_labels_dict['terrain']  # Bridge
    # The special labels added by us or overriden"crossWalk"
    label_mapping[carla_labels_dict['sidewalk_extra']] = cityscapes_labels_dict['sidewalk_extra'] # Ground to sidewalk extra
    label_mapping[carla_labels_dict['crosswalk']] = cityscapes_labels_dict['crosswalk']
    # for label, indx in carla_labels_dict.items():
    #     if label!=None:
    #         print (" Label "+label+" in carla "+str(indx)+" mapped to cityscapes "+str(label_mapping[indx])+" which is "+cityscapes_labels[label_mapping[indx]])

    return label_mapping

NUM_LABELS_CARLA = len(carla_labels)-1 # 13

NO_LABEL_POINT = cityscapes_labels_dict['unlabeled']
OTHER_SEGMENTATION_POINT_CARLA = carla_labels_dict['other']
SEGLABEL_PEDESTRIAN_CARLA =carla_labels_dict['person']
SEGLABEL_CAR_CARLA = carla_labels_dict['car']
ROAD_LABELS=(cityscapes_labels_dict['ground'], cityscapes_labels_dict['road'], cityscapes_labels_dict['parking'], cityscapes_labels_dict['rail track'] )
SIDEWALK_LABELS=(cityscapes_labels_dict['sidewalk'], cityscapes_labels_dict['sidewalk_extra'], cityscapes_labels_dict['crosswalk'])
print (" Sidewalk labels"+str(SIDEWALK_LABELS))
print (" Road labels"+str(ROAD_LABELS))
OBSTACLE_LABELS=(cityscapes_labels_dict['building'],
                 cityscapes_labels_dict['wall'],
                 cityscapes_labels_dict['fence'],
                 cityscapes_labels_dict['guard rail'],
                 cityscapes_labels_dict['bridge'],
                 cityscapes_labels_dict['tunnel'],
                 cityscapes_labels_dict['pole'],
                 cityscapes_labels_dict['polegroup'],
                 cityscapes_labels_dict['traffic light'],
                 cityscapes_labels_dict['traffic sign'])
OBSTACLE_LABELS_NEW=(cityscapes_labels_dict['building'],
                 cityscapes_labels_dict['wall'],
                 cityscapes_labels_dict['fence'],
                 cityscapes_labels_dict['guard rail'],
                 cityscapes_labels_dict['bridge'],
                 cityscapes_labels_dict['tunnel'],
                 cityscapes_labels_dict['pole'],
                 cityscapes_labels_dict['polegroup'],
                 cityscapes_labels_dict['traffic light'],
                 cityscapes_labels_dict['traffic sign'],
                 cityscapes_labels_dict['vegetation'],
                 cityscapes_labels_dict['static'],
                 cityscapes_labels_dict['dynamic'])

MOVING_OBSTACLE_LABELS=(cityscapes_labels_dict['rider'],
                 cityscapes_labels_dict['car'],
                 cityscapes_labels_dict['truck'],
                 cityscapes_labels_dict['bus'],
                 cityscapes_labels_dict['trailer'],
                 cityscapes_labels_dict['train'],
                 cityscapes_labels_dict['motorcycle'],
                 cityscapes_labels_dict['bicycle'])
print(MOVING_OBSTACLE_LABELS)
carlaToCitiscapesLabelMapping = get_label_mapping()
SEGLABEL_PEDESTRIAN_CITYSCAPES = carlaToCitiscapesLabelMapping[SEGLABEL_PEDESTRIAN_CARLA]
SEGLABEL_CAR_CITYSCAPES = carlaToCitiscapesLabelMapping[SEGLABEL_CAR_CARLA]


FILENAME_CARS_TRAJECTORIES = 'cars.p'
FILENAME_PEOPLE_TRAJECTORIES = 'people.p'
FILENAME_CARLA_BBOXES = 'carla_bboxes.csv'
FILENAME_CARLA_SCENEORIGIN = "translation.txt"
FILENAME_CARLA_HEROCARTRANSFORM = "herocar.txt"
FILENAME_COMBINED_CARLA_ENV_POINTCLOUD = 'combined_carla_moving.ply'
FILENAME_COMBINED_CARLA_ENV_POINTCLOUD_SEGCOLOR = 'combined_carla_moving_segColor.ply'
FILENAME_COMBINED_CARLA_ENV_POINTCLOUD_Fi = 'combined_carla_moving_f{0:05d}.ply'
FILENAME_COMBINED_CARLA_ENV_POINTCLOUD_SEGCOLOR_Fi = 'combined_carla_moving_segColor_f{0:05d}.ply'
FILENAME_CENTERING_ENV = 'centering.p'
FILENAME_CAMERA_INTRISICS = 'camera_intrinsics.p'

# Filter  to remove noise with KNN
REFINE_3DRECONSTRUCTION_WITH_KNN = True
DEBUG_NEIGHBOORS_STATISTICS = True
if REFINE_3DRECONSTRUCTION_WITH_KNN:
    from scipy.spatial import KDTree

# useEnvironment = True if you want to consider only environment points, False only if motion points. TODO: should be an enum ?
def isUsefulLabelForReconstruction(label, useEnvironment=True):
    if (label == NO_LABEL_POINT or label == OTHER_SEGMENTATION_POINT_CARLA):
        return False

    return (not isLabelIgnoredInReconstruction(label, useEnvironment))  # and (label not in [1])


def isCloudPointRoadOrSidewalk(pos, label):
    #return (label > 5 and label < 9) or (label == SPECIAL_LABEL_SIDEWALK_EXTRA)
    return label in SIDEWALK_LABELS or label in ROAD_LABELS

def isLabelIgnoredInReconstruction(labels, isCarlaLabel=True, useEnvironment=True):
    if isCarlaLabel == False: # Cityscapes ..
        if labels < cityscapes_labels_dict['static']: # If label is 'unlabeled'-0, 'ego vehicle'- 1, 'rectification border'- 2 or 'out of roi'-3
            return True

    # Is it a vehicle or pedestrian (i.e. motion label ? ignore it if not expliciticely required)
    isAMotionLabel = None
    if isinstance(labels, list):
        isAMotionLabel = (SEGLABEL_PEDESTRIAN_CARLA in labels or SEGLABEL_CAR_CARLA in labels) if isCarlaLabel \
                            else (SEGLABEL_PEDESTRIAN_CITYSCAPES in labels or SEGLABEL_CAR_CITYSCAPES in labels)
    else:
        isAMotionLabel = isLabelForMotionEntity(labels, isCarlaLabel)

    if useEnvironment == True:
        return isAMotionLabel
    else:
        return not isAMotionLabel


def isLabelForMotionEntity(labels, isCarlaLabel):
    if isCarlaLabel:
        return (labels == SEGLABEL_PEDESTRIAN_CARLA or labels == SEGLABEL_CAR_CARLA)
    else:
        return (labels == SEGLABEL_PEDESTRIAN_CITYSCAPES or labels == SEGLABEL_CAR_CITYSCAPES)


# NOTE: scale = 5 represents conversion from camera coordinate system to because before the coordinate system was in meters, 1 * 5

# Given a dictionary of [3d points from the cloud] -> {rgb segmentation label , raw segmentation label}, save it to the specified file
# (Taken from CARLA's code)
def save_3d_pointcloud(points_3d, filename):
    """Save this point-cloud to disk as PLY format."""

    def construct_ply_header():
        """Generates a PLY header given a total number of 3D points and
        coloring property if specified
        """
        points = len(points_3d)  # Total point number
        header = ['ply',
                  'format ascii 1.0',
                  'element vertex {}',
                  'property float32 x',
                  'property float32 y',
                  'property float32 z',
                  'property uchar diffuse_red',
                  'property uchar diffuse_green',
                  'property uchar diffuse_blue',
                  'property uchar label',
                  'end_header']
        return '\n'.join(header).format(points)

    for point in points_3d:
        label = getLabelFromPoint(point)
        for p in point:
            try:
                n = float(p)
            except ValueError:
                print(("Problem " + str(point)))
    # points_3d = np.concatenate(
    #     (point_list._array, self._color_array), axis=1)
    try:
        ply = '\n'.join(
            ['{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(*p) for p in points_3d])  # .replace('.',',')
    except ValueError:
        for point in points_3d:
            print(point)
            print(('{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(*point)))
    # Create folder to save if does not exist.
    # folder = os.path.dirname(filename)
    # if not os.path.isdir(folder):
    #     os.makedirs(folder)

    # Open the file and save with the specific PLY format.
    print (filename)
    with open(filename, 'w+') as ply_file:
        ply_file.write('\n'.join([construct_ply_header(), ply]))


def save_3d_pointcloud_asSegLabel(points_3d, filename):
    points_3d_seg = [None] * len(points_3d)
    """Save this point-cloud to disk as PLY format."""

    def construct_ply_header():
        """Generates a PLY header given a total number of 3D points and
        coloring property if specified
        """
        points = len(points_3d)  # Total point number
        header = ['ply',
                  'format ascii 1.0',
                  'element vertex {}',
                  'property float32 x',
                  'property float32 y',
                  'property float32 z',
                  'property uchar diffuse_red',
                  'property uchar diffuse_green',
                  'property uchar diffuse_blue',
                  'property uchar label',
                  'end_header']
        return '\n'.join(header).format(points)

    for index, origPoint in enumerate(points_3d):
        # Override the RGB to the RGB of the segmentation
        label = getLabelFromPoint(origPoint)
        try:
            colorForLabel = cityscapes_colours[label]
        except:
            assert False, "invalid label or something"
            colorForLabel = (0, 0, 0)
        point_seg = (
        origPoint[0], origPoint[1], origPoint[2], colorForLabel[0], colorForLabel[1], colorForLabel[2], label)
        points_3d_seg[index] = point_seg

        for p in origPoint:
            try:
                n = float(p)
            except ValueError:
                print(("Problem " + str(origPoint)))
    # points_3d = np.concatenate(
    #     (point_list._array, self._color_array), axis=1)
    try:
        ply = '\n'.join(
            ['{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(*p) for p in points_3d_seg])  # .replace('.',',')
    except ValueError:
        for point in points_3d_seg:
            print(point)
            print(('{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(*point)))
    # Create folder to save if does not exist.
    # folder = os.path.dirname(filename)
    # if not os.path.isdir(folder):
    #     os.makedirs(folder)

    # Open the file and save with the specific PLY format.
    with open(filename, 'w+') as ply_file:
        ply_file.write('\n'.join([construct_ply_header(), ply]))


# Read pointcloud from ply file - vertices coord and semnatic labels
#
def read_3D_pointcloud(filename, file_end='/dense/fused_text.ply'):
    plydata_3Dmodel = PlyData.read(filename + file_end)
    nr_points = plydata_3Dmodel.elements[0].count
    pointcloud_3D = np.array([plydata_3Dmodel['vertex'][k] for k in range(nr_points)])
    return pointcloud_3D


# Read a ply file with the correspondences between 3d points and RGB values
def get_rgb_point_cloud(filepath, parts):
    rgb_cloud = os.path.join(filepath, parts[0] + '.ply')
    print(rgb_cloud)
    pointcloud_3D_rgb = read_3D_pointcloud(rgb_cloud, '')
    rgb_dict = {}
    points=[[],[],[]]
    for point in pointcloud_3D_rgb:
        key = (point[0], point[1], point[2])
        rgb_dict[key] = (point[3], point[4], point[5])
        points[0].append(point[0])
        points[1].append(point[1])
        points[2].append(point[2])
    print(" Min dim 1 "+str(min(points[0]))+" max "+str(max(points[0])))
    print(" Min dim 2 " + str(min(points[1])) + " max " + str(max(points[1])))
    print(" Min dim 3 " + str(min(points[2])) + " max " + str(max(points[2])))
    return rgb_dict



# A refining parameter....use statistics to tune
# If below this number, we'll cut the point at all we call it noise. But we need the histograms debug enabled to see the stats !
MIN_NEIGHBOORS_COUNT_TO_VALIDATE_NEW_LABEL = 20

# given a set of (x,y,z, r, g, b, label), eliminate noise by voting what are the most common labels around it
# discretizedDictionaryCloudData is optional...
# returns the new point cloud data and updated discretizedDictionary if provided
# Use Knn method below if you want voting too, otherwise, if you need just quick removal of noisy points use this one instead
def refinePointCloud_voxelization(pointCloudData, scaleAppliedOverOriginalData,
                                  discretizedDictionaryCloudData = None):
    # An unbounded type of octree (without spatial constraints, keys are x,y,z + util info in the next parameters).
    class OctreeUnbounded:
        def __init__(self, cellSize, pointCloudData, fastDiscard=False):
            self.D = {}
            self.cellSize = cellSize
            self.cellSizeSqr = self.cellSize*self.cellSize
            self.invCellSize = 1.0/cellSize
            self.pointCloudData = pointCloudData
            self.fastDiscard = fastDiscard

            for index, pointData in enumerate(pointCloudData):
                self.addData(pointData, index)

        def reset(self):
            self.D = {}

        # Store the indices as given in the octree
        def addData(self, pointData, pointIndex):
            #origX, origY, origZ = data[0], data[1], data[2]

            # Convert point to the octree space
            points_scaled = np.round(np.array(pointData[0:3]) * self.invCellSize).astype(np.int)
            x, y, z = points_scaled[0], points_scaled[1], points_scaled[2]

            key = (x, y, z)
            if self.D.get(key) == None:
                self.D[key] = []
            self.D[key].append(pointIndex)

        # selects the indices as stored corresponding to the original data
        # returns a list of selected indices and a list of discarded indices
        def selectIndicesByFilter(self, minNumNeighboors):
            outSelectedIndices = []
            outDiscardedIndices = []
            # Iterate over each cell.
            for cellPos, cellData in list(D.items()):
                # if the cell has >= minNumNeihboors, all points inside are fine, add them to the final output
                if len(cellData) > minNumNeighboors:
                    outSelectedIndices.extend(cellData)

                # else look at the neighbooring cells.
                elif self.fastDiscard == False:

                    for srcIndex in cellData:
                        srcPointData = self.pointCloudData[srcIndex]
                        origCenterX, origCenterY, origCenterZ = srcPointData[0:3]

                        # if we can find in those a number of points >= minNumNeighboors, which are closer than self.CELL_SIZE add it
                        numNeighbClose = len(cellData)
                        numCellsToInvestigateAround = 1
                        wasSelected = False
                        for cellX in range(centerX - numCellsToInvestigateAround, centerX + numCellsToInvestigateAround + 1):
                            for cellY in range(centerY - numCellsToInvestigateAround, centerY + numCellsToInvestigateAround + 1):
                                for cellZ in range(centerZ - numCellsToInvestigateAround, centerZ + numCellsToInvestigateAround + 1):
                                    key = (cellX, cellY, cellZ)
                                    if key in self.D:
                                        for targetIndex in self.D[key]:
                                            targetPointData = self.pointCloudData[targetIndex]
                                            origX, origY, origZ = targetPointData[0:3]
                                            dist = (origCenterX - origX) ** 2 + (origCenterY - origY) ** 2 + (origCenterZ - origZ) ** 2
                                            if dist < self.cellSizeSqr:
                                                numNeighbClose += 1
                                                if numNeighbClose >= minNumNeighboors:
                                                    outSelectedIndices.append(srcIndex)
                                                    wasSelected = True
                                                    break

                        if wasSelected == False:
                            outDiscardedIndices.append(srcIndex)
                else:
                    outDiscardedIndices.extend(cellData)

            return outSelectedIndices, outDiscardedIndices

    MAX_DISTANCE = scaleAppliedOverOriginalData # voxels here, 1 voxel = 20 cm, so 1 meter has 5 voxels.
    octreeData = OctreeUnbounded(cellSize=MAX_DISTANCE, pointCloudData=pointCloudData, fastDiscard=True)
    selectedIndices, discardedIndices = octreeData.selectIndicesByFilter(MIN_NEIGHBOORS_COUNT_TO_VALIDATE_NEW_LABEL)

    # Compute the filtered point cloud with noise removed
    filteredPointCloud = []
    for index in selectedIndices:
        filteredPointCloud.append(pointCloudData[index])

    # Remove from the discretized dictionary, if given, the discarded keys
    for index in discardedIndices:
        x, y, z = pointCloudData[index][0:3]
        key_asint = (int(x), int(y), int(z))
        discretizedDictionaryCloudData.pop(key_asint, None)

    return filteredPointCloud


# Given parameters, eliminate noise from the pcd and return the indices removed from it
# TODO: we also need a kind of connected components with grounds since many artefacts are in air and everything has to be connected with ground because of gravity
def _noiseRemovalWithStatsAndKnn(pcd, scale, debug=False):
    import open3d as o3d

    # Step 1: remove statistically the points depending on deviation and neighboors mean
    # print("Statistical oulier removal")
    # start_time = time.time()
    nb_neighbors = 500.0 if scale <= 1.1 else (50.0 if scale == 5 else (1000.0 / (4.0 * scale)))  # TODO: this is highely tunable depinding on data...
    cl, includedIndices = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=2.0)
    # print("--- %s seconds ---" % (time.time() - start_time))

    # Step 2: find the terrain Z coordinate and select only the points above the estimated therrain.
    # This is done because the tendency is to cut too many on the ground...

    # Select only the points which are close to the camera sensor..those are most valuable in terms of statistical evaluation
    cloud_points = np.asarray(pcd.points)
    cloud_bbox = np.array(pcd.get_axis_aligned_bounding_box().get_box_points())
    cloud_bbox_min = np.min(cloud_bbox, axis=0)
    cloud_bbox_max = np.max(cloud_bbox, axis=0)
    x_quarter = (cloud_bbox_max[0] - cloud_bbox_min[0]) * 0.125
    y_quarter = (cloud_bbox_max[1] - cloud_bbox_min[1]) * 0.125
    cloud_points_stat = np.asarray(pcd.points).copy()
    cloud_points_stat = cloud_points_stat[np.where((0 <= cloud_points_stat[:, 0]) & (cloud_points_stat[:, 0] <= x_quarter) & \
                                                   (0 <= cloud_points_stat[:, 1]) & (cloud_points_stat[:, 1] <= y_quarter))]

    zValues = cloud_points_stat[:, 2]  # sorted(cloud_points[:,2])

    # Select between 2 and 100 percentiles...
    leftCut = np.percentile(zValues, 2)  # 2% as noise...
    rightCut = np.percentile(zValues, 100)
    zValues = zValues[np.where((leftCut <= zValues) & (zValues <= rightCut))]

    zValues_mean = np.mean(zValues)
    zValues_median = np.median(zValues)
    zValues_min = np.min(zValues)
    zValues_max = np.max(zValues)
    # print(f"mean Z: {zValues_mean}, median {zValues_median}, min {zValues_min} max {zValues_max}")

    # hist, bins = np.histogram(zValues, bins=[6, 7.0, 7.10, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 9, 10, 11, 12])
    # print(f"histogram. bins: {bins}, hist: {hist}")

    zMax = zValues_min + 0.1
    zMin = zValues_min
    DIST = zMin + 0.4 * scale

    includedIndices = np.array(includedIndices)
    excluded_indices = np.setdiff1d(np.arange(1, len(cloud_points)), includedIndices)
    initial_excluded_len = len(excluded_indices)
    sel_excluded_indices = np.where(cloud_points[excluded_indices, 2] > DIST)[0]
    excluded_indices = excluded_indices[sel_excluded_indices]

    # Display results
    if debug == True:
        inlier_cloud = pcd.select_by_index(excluded_indices, invert=True)
        outlier_cloud = pcd.select_by_index(excluded_indices)

        # Draw geometries for ground estimation
        inlier_cloud_bbox = inlier_cloud.get_axis_aligned_bounding_box()
        cloudBBox = np.array(inlier_cloud_bbox.get_box_points())
        xExtend, yExtend, _ = inlier_cloud_bbox.get_extent()
        # cloudBBox[0:4,2] = zMi
        # cloudBBox[4:8,2] = zMax
        # cloudBBox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(cloudBBox))

        estimatedGroundMeshBox = o3d.geometry.TriangleMesh.create_box(width=xExtend, height=yExtend, depth=abs(zMax - zMin))
        estimatedGroundMeshBox.compute_triangle_normals()
        estimatedGroundMeshBox.translate([-xExtend / 2, -yExtend / 2, zMin + abs(zMax - zMin) / 2])
        estimatedGroundMeshBox.paint_uniform_color([0, 0, 1])

        print("Showing outliers (red) and inliers (gray)")
        print(("Dist {0}".format(DIST)))
        print(("Removed percent after terrain filter: {0}".format((len(excluded_indices) / len(cloud_points)))))
        print(("Initial percent {0}".format((initial_excluded_len / len(cloud_points)) * 100.0)))
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, estimatedGroundMeshBox])

    return includedIndices, excluded_indices


def refinePointCloud_knnStatistical(pointCloudData, scaleAppliedOverOriginalData, discretizedDictionaryCloudData=None):
    import open3d as o3d

    # Using c++ version of knn implemented in cpp this time...
    kdTreePoints = []
    for index, origPoint in enumerate(pointCloudData):
        x, y, z = origPoint[0], origPoint[1], origPoint[2]
        kdTreePoints.append((x, y, z))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(kdTreePoints)
    indices_selected, indices_removed = _noiseRemovalWithStatsAndKnn(pcd, scale=scaleAppliedOverOriginalData, debug=False)

    # update the discretized table
    if discretizedDictionaryCloudData != None:
        for index in indices_removed:
            origPoint = pointCloudData[index]
            keyPoint_asint = int(origPoint[0]), int(origPoint[1]), int(origPoint[2])
            discretizedDictionaryCloudData.pop(keyPoint_asint, None)

    filteredPointCloudData = []
    for indexSelected in indices_selected:
        filteredPointCloudData.append(pointCloudData[indexSelected])

    print(("Filtering method discarded {0}%".format((len(filteredPointCloudData) / len(pointCloudData))*100.0)))
    return filteredPointCloudData

# given a set of (x,y,z, r, g, b, label), eliminate noise by voting what are the most common labels around it
# discretizedDictionaryCloudData is optional...
# returns the new point cloud data and updated discretizedDictionary if provided
def refinePointCloud_knn(pointCloudData, scaleAppliedOverOriginalData, discretizedDictionaryCloudData = None):
    stats_num_neighboors = []  # A histogram calculation to understand the number of neighboors in the area around a point

    kdTreePoints = []

    MAX_DISTANCE = scaleAppliedOverOriginalData if scaleAppliedOverOriginalData > 1.0 else scaleAppliedOverOriginalData # 5.0  # voxels here, 1 voxel = 20 cm, so 1 meter has 5 voxels.
    MAX_CLOSEST_POINT_TO_EVALUATE = MIN_NEIGHBOORS_COUNT_TO_VALIDATE_NEW_LABEL*5 # maximum number of points to look around

    # Add all the points in the same order in kd tree, this way we'll get the same indices correspondeces.
    for index, origPoint in enumerate(pointCloudData):
        x, y, z = origPoint[0], origPoint[1], origPoint[2]
        kdTreePoints.append((x, y, z))




    numTotalPoints = len(pointCloudData)
    filteredPointCloudData = [] #[None]*numTotalPoints

    kdTreePoints = np.array(kdTreePoints)
    if len(kdTreePoints) > 0:
        kdTree = KDTree(kdTreePoints)

        for index, origPoint in enumerate(pointCloudData):
            origPoint_x, origPoint_y, origPoint_z, r,g, b, origPoint_Label = origPoint
            if index % 20000 == 0:
                print(("Refining at {0}/{2}".format(index, len(pointCloudData))))

            # Get the mode label around this point
            queryPoint = (origPoint_x, origPoint_y, origPoint_z)
            dist, indices = kdTree.query(queryPoint, k=MAX_CLOSEST_POINT_TO_EVALUATE, distance_upper_bound=MAX_DISTANCE)
            # indices = kdTree.query_ball_point(queryPoint, MAX_DISTANCE)

            pointsAroundData = [pointCloudData[index] for index in indices if index < numTotalPoints]
            if DEBUG_NEIGHBOORS_STATISTICS:
                stats_num_neighboors.append(len(pointsAroundData))

            # Noise removal !
            if (len(pointsAroundData) < MIN_NEIGHBOORS_COUNT_TO_VALIDATE_NEW_LABEL):
                if discretizedDictionaryCloudData != None:
                    keyPoint_asint = int(origPoint_x), int(origPoint_y), int(origPoint_z)
                    discretizedDictionaryCloudData.pop(keyPoint_asint, None)
                continue

            labelsVoted = np.zeros(len(cityscapes_labels) + 1)
            for pAround in pointsAroundData:
                pAround_label = getLabelFromPoint(pAround) # 6 is the index for label in the table. WE REALLY need to do something about this, get some functor functions. What if we change the index at some point ??
                labelsVoted[pAround_label] += 1

            mostVotedLabel = np.argmax(labelsVoted)
            if mostVotedLabel != origPoint_Label and labelsVoted[mostVotedLabel] > MIN_NEIGHBOORS_COUNT_TO_VALIDATE_NEW_LABEL:
                origPoint_Label = mostVotedLabel

            filteredPointCloudData.append((origPoint_x, origPoint_y, origPoint_z, r, g, b, origPoint_Label))

    if DEBUG_NEIGHBOORS_STATISTICS:
        with open("neighStats.pkl", "wb") as f:
            pickle.dump(stats_num_neighboors, f)
        """
        stats_num_neighboors = np.array(stats_num_neighboors)
        values, bins = np.histogram(stats_num_neighboors, bins=30)
        values = np.sort(values)
        print("The first 40 min values are: ", values[:40])
        import sys
        np.set_printoptions(threshold=sys.maxsize)
        print ("Values " , values)
        print ("bins " , values)
        np.set_printoptions(threshold=100)
        """


    print(("Filtering method discarded {0}".format((len(filteredPointCloudData) / len(pointCloudData))*100.0)))
    return filteredPointCloudData


#######################

# Given a dictionary of people location in RGB images, save on each row coordinates corresponding to each frame
def save_2D_poses(filepath, people_2D, frames):
    frame_i = 0
    csv_rows = []
    image_sz = [800, 600]
    for frame in sorted(frames):
        frame_name = "%06d.png" % frame
        csv_rows.append([])
        csv_rows[frame_i].append(os.path.join(filepath, 'CameraRGB', frame_name))

        rectangles = []

        for pers_long in people_2D[frame]:
            # print pers_limited
            if len(pers_long) > 0:

                pers = [[min(pers_long[1]), max(pers_long[1])], [min(pers_long[0]), max(pers_long[0])]]
                pers_limited = [[max(int(pers[0][0]), 0), min(int(pers[0][1]), image_sz[1])],
                                [max(int(pers[1][0]), 0), min(int(pers[1][1]), image_sz[0])]]
                if pers_limited[0][1] - pers_limited[0][0] > 10 and pers_limited[1][1] - pers_limited[1][0] > 10:
                    csv_rows[frame_i].append(pers_limited)

        frame_i = frame_i + 1

    with open(os.path.join(filepath, FILENAME_CARLA_BBOXES), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

def readFloatLines(fileHandle):
    content = fileHandle.readline().split()
    content = [float(x) for x in content]
    return content

# Gets a translation from a filepath
def get_translation(filepath, realtime_data=False):
    translation = [0.0, 0.0, 0.0] # Default if no translation exists
    print(realtime_data)
    if filepath is not None and not realtime_data:
        filepath = os.path.join(filepath, FILENAME_CARLA_SCENEORIGIN)
        if os.path.exists(filepath):
            f = open(filepath, "r")
            translation = readFloatLines(f)
            assert len(translation) == 3
    print (translation)
    return translation

def get_herocartransform(filepath):
    filepath = os.path.join(filepath, FILENAME_CARLA_HEROCARTRANSFORM)
    if not os.path.exists(filepath):
        return None
    else:
        f = open(filepath, "r")
        # Spawn origin - 3 floats,  followed by yaw in degrees - 1 float, then destination pos - 3 floats
        translation = readFloatLines(f)
        assert len(translation) == 3
        yaw = readFloatLines(f)
        assert len(translation) == 1
        destination = readFloatLines(f)
        assert len(destination) == 3


        return (translation, yaw[0], destination)



# Returns the transformation from world-to-camera space (R_inv/'inverse_rotation'):
# First is rotation, second is translation, third is a list of frame indices where we gathered data
def get_camera_matrix(filepath):


    cameras_path = os.path.join(filepath,'cameras.p')  # Transforms from camera coordinate system to world coordinate system

    if os.path.exists(cameras_path):
        cameras_dict = pickle.load(open(cameras_path, "rb"), encoding="latin1", fix_imports=True)

        frames = sorted(cameras_dict.keys())

        # if len(frames) > LIMIT_FRAME_NUMBER:
        #    frames = frames[:LIMIT_FRAME_NUMBER]

        frame = frames[-1]  # np.min(cameras_dict.keys())
        R_inv = cameras_dict[frame]['inverse_rotation']
        middle = R_inv[0:3, 3]
        R = np.transpose(R_inv[0:3, 0:3])
        middle = -np.matmul(R, middle)  # Camera center

        t = R_inv[0:3, 3]

        return R_inv[0:3, 0:3], t, frames  # , car_pos

    else: # Cpp will write to a text file with different format
        cameras_path = os.path.join(filepath,
                                    'cameras.txt')  # Transforms from camera coordinate system to world coordinate system
        assert os.path.exists(cameras_path), "There is no camera definition file !!"
        f = open(cameras_path, "r")

        # Read the inverse camera transform
        numFrames = int(f.readline())
        R_inv = np.zeros(shape=(4,4))
        for row_index in range(4):
            R_inv[row_index][:] = readFloatLines()

        frames = [x for x in range(numFrames)]
        t = R_inv[0:3, 3]

        return R_inv[0:3, 0:3], t, frames

def get_camera_intrinsic_matrix(filepath):
    cameras_path = os.path.join(filepath, FILENAME_CAMERA_INTRISICS)
    K = pickle.load(open(cameras_path, "rb"), encoding="latin1", fix_imports=True)

    return K


def get_cars_list(list, cars_dict, frames, WorldToCameraRotation, WorldToCameraTranslation,
                  heightOffset, scale, K, filepath, people_flag=False, find_poses=False, datasetOptions=None):
    image_size = [800, 600]
    list_2D = []
    poses = []
    print(" Frames in car "+str(max(cars_dict.keys()))+" min frame "+str(min(cars_dict.keys())))
    for i, frame in enumerate(sorted(cars_dict.keys())):
        # If there is some dataset options and there is a number of items limit..
        if datasetOptions is not None and i >= datasetOptions.LIMIT_FRAME_NUMBER:
            break

        frame_cars = cars_dict[frame]
        list_2D.append([])
        poses.append([])
        for car_id in frame_cars:

            x_values = []
            y_values = []
            z_values = []
            if False:  # 'bounding_box_coord' in frame_cars[car_id].keys() and not people_flag:
                for point in frame_cars[car_id]['bounding_box_coord']:
                    point = np.squeeze(
                        np.array(np.matmul(WorldToCameraRotation, point.T) + WorldToCameraTranslation).reshape(
                            -1, )) * scale

                    point = point * [-1, -1, 1]  # normal camera coodrinates

                    point = np.squeeze(np.asarray(np.matmul(P_inv, point)))
                    point[2] = point[2] - heightOffset

                    x_values.append(point[0])
                    y_values.append(point[1])
                    z_values.append(point[2])
                car = np.array([[np.min(x_values), np.max(x_values)], [np.min(y_values), np.max(y_values)],
                                [np.min(z_values), np.min(z_values)]])

                x_values = []
                y_values = []
                if find_poses:
                    if 'bounding_box_2D' in list(frame_cars[car_id].keys()):

                        for point in frame_cars[car_id]['bounding_box_2D']:
                            x_values.append(point[0])
                            y_values.append(point[1])

            else:
                dataAlreadyInGlobalSpace = datasetOptions is not None and datasetOptions.dataAlreadyInWorldSpace is True
                #print(" Data already in global space "+str(dataAlreadyInGlobalSpace))
                if not dataAlreadyInGlobalSpace:
                    car, bboxes_2d = bbox_of_car(K, WorldToCameraRotation, car_id, find_poses, frame_cars, heightOffset,
                                                 image_size,
                                                 WorldToCameraTranslation, scale)
                else:


                    car, bboxes_2d = bbox_of_entity_alreadyInGlobalSpace(K, WorldToCameraRotation, car_id,
                                                                         find_poses, frame_cars, heightOffset,
                                                                         image_size, WorldToCameraTranslation, scale,
                                                                         people_flag)




                list_2D[i] = bboxes_2d

            if not people_flag:
                car = car.flatten()
                car[1] = car[1] + 1
                car[3] = car[3] + 1
                car[5] = car[5] + 1
            if len(list)<=i:
                list.append([])
            list[i].append(car.astype(int))

    return list, list_2D


# Build a 3D rotation matrix giving yaw
def get_car_rotation_matrix(yaw):
    cy = np.cos(np.radians(yaw))
    sy = np.sin(np.radians(yaw))
    cr = 1
    sr = 0
    cp = 1
    sp = 0
    matrix = np.matrix(np.identity(3))
    matrix[0, 0] = cp * cy
    matrix[0, 1] = cy * sp * sr - sy * cr
    matrix[0, 2] = - (cy * sp * cr + sy * sr)
    matrix[1, 0] = sy * cp
    matrix[1, 1] = sy * sp * sr + cy * cr
    matrix[1, 2] = cy * sr - sy * sp * cr
    matrix[2, 0] = sp
    matrix[2, 1] = -(cp * sr)
    matrix[2, 2] = cp * cr

    return matrix


def bbox_of_car(K, WorldToCameraRotation, car_id, find_poses, frame_cars, heightOffset, image_size, middle, scale):
    pos_car = frame_cars[car_id]['transform']  # Transformation this is the position of car world space.
    bbox_car = np.squeeze(
        np.asarray(np.matmul(np.matmul(WorldToCameraRotation, get_car_rotation_matrix(frame_cars[car_id]['yaw'])),
                             frame_cars[car_id]['bounding_box'])))  # Bbox rotated to the camera space
    point_0 = np.squeeze(np.asarray(np.matmul(WorldToCameraRotation, np.array(pos_car).reshape(
        (3, 1))) + middle))  # Que 5: Transformation -  This point is on relative to the camera space
    boxes_2d = []
    if find_poses:
        x_values = []
        y_values = []
        bbx = [np.array([point_0[0] - bbox_car[0], point_0[1] - bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] - bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] - bbox_car[1], point_0[2] + bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] + bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] + bbox_car[1], point_0[2] + bbox_car[2]]),
               np.array([point_0[0] - bbox_car[0], point_0[1] - bbox_car[1], point_0[2] + bbox_car[2]]),
               np.array([point_0[0] - bbox_car[0], point_0[1] + bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] - bbox_car[0], point_0[1] + bbox_car[1], point_0[2] + bbox_car[2]])]
        for point in bbx:
            point_2D = np.matmul(K, point.reshape((3, 1)))
            point_2D = point_2D / point_2D[2]
            x_2d = int(image_size[0] - point_2D[0])
            y_2d = int(image_size[1] - point_2D[1])
            x_values.append(x_2d)
            y_values.append(y_2d)
        boxes_2d.append([y_values, x_values])
    point_0 = np.squeeze(np.asarray(np.matmul(WorldToCameraRotation, np.array(pos_car).reshape((3, 1))) + middle))
    bbox_car = bbox_car * scale
    point_0 = point_0 * scale
    bbox_car = bbox_car * [-1, -1, 1]  # normal camera coodrinates
    point_0 = point_0 * [-1, -1, 1]  # normal camera coodrinates
    point_0 = np.squeeze(np.asarray(np.matmul(P_inv, point_0)))
    point_0[2] = point_0[2] - heightOffset
    bbox_car = np.squeeze(np.asarray(np.matmul(P_inv, bbox_car)))
    car = np.column_stack((point_0 - np.abs(bbox_car), point_0 + np.abs(bbox_car)))
    return car, boxes_2d


def bbox_of_entity_alreadyInGlobalSpace(K, WorldToCameraRotation, car_id, find_poses, frame_cars, heightOffset,
                                        image_size, middle, scale, peopleFlag):
    minMaxWorldPos = copy.copy(frame_cars[car_id]['BBMinMax'])

    if middle is not None:
        minMaxWorldPos -= middle

    assert minMaxWorldPos.shape == (3, 2), "Incorrect format should be xmin,xmax first row, then ymin, ymax etc"
    # scale down from meters to voxels
    minMaxWorldPos *= scale
    # Add height offset
    minMaxWorldPos[2, :] -= heightOffset

    return minMaxWorldPos, []


def get_cars_map(cars_dict, frames, R, middle, height, scale, K, filepath, people_flag=False, find_poses=False,
                 datasetOptions=None):
    image_sz = [800, 600]
    poses_map = {}
    init_frames = {}  # Map from car_id to first frame index when it appears in the scene
    valid_ids = []  # Cars that appear in frame 0
    cars_map = {}  # Map from car_id to a list of ordered (by frame) positions on each frame

    for i, frame in enumerate(frames):
        frame_cars = cars_dict[frame]
        for car_id in frame_cars:

            if datasetOptions is None or datasetOptions.dataAlreadyInWorldSpace is False:
                car, bboxes_2d = bbox_of_car(K, R, car_id, find_poses, frame_cars, height, image_sz, middle, scale)
            else:

                car, bboxes_2d = bbox_of_entity_alreadyInGlobalSpace(K, R, car_id, find_poses, frame_cars, height,
                                                                     image_sz, middle, scale, people_flag)

            if frame == 0:
                valid_ids.append(car_id)
            if car_id not in list(init_frames.keys()):
                init_frames[car_id] = int(frame)
            if not people_flag:
                car = car.flatten()

            if not car_id in cars_map:
                cars_map[car_id] = []
            cars_map[car_id].append(car)  # .astype(int))
    return cars_map, poses_map, valid_ids, init_frames


def get_people_and_cars(WorldToCameraRotation, cars, filepath, frames, WorldToCameraTranslation, heightOffset,
                        people, scale_x, find_poses=False, datasetOptions=None):
    # TODO: make this an utility, do not leave it here
    simplifyDataSet = False  # Used for debugging purposes to cut the load time of cars/people stuff

    # Read the intrinsics matrix if needed
    K = None
    if datasetOptions is None or datasetOptions.dataAlreadyInWorldSpace is False:
        K = get_camera_intrinsic_matrix(filepath)

    # Read the cars and people dictionary in format: {frame_id : { entity_id : {...dict with data... } } }
    cars_path = os.path.join(filepath, FILENAME_CARS_TRAJECTORIES)
    print (cars_path)
    with open(cars_path, 'rb') as handle:
        cars_dict = pickle.load(handle, encoding="latin1", fix_imports=True)
    if simplifyDataSet == True and len(cars_dict) > 1:
        cars_dict_new = {}
        for x in range(len(frames)):
            cars_dict_new[x] = cars_dict[x]
        cars_dict = cars_dict_new
        with open(cars_path, 'wb') as handle:
            pickle.dump(cars_dict, handle, protocol=2)  # Protocol 2 ensures compatibility between python 2 and 3

    people_path = os.path.join(filepath, FILENAME_PEOPLE_TRAJECTORIES)
    with open(people_path, 'rb') as handle:
        people_dict = pickle.load(handle, encoding="latin1", fix_imports=True)

    if simplifyDataSet == True and len(people_dict) > 1:
        people_dict_new = {}
        for x in range(len(frames)):
            if x in people_dict:
                people_dict_new[x] = people_dict[x]
            else:
                people_dict_new[x] = {}

        people_dict = people_dict_new
        with open(people_path, 'wb') as handle:
            pickle.dump(people_dict, handle, protocol=2)  # Protocol 2 ensures compatibility between python 2 and 3

    #see the comments below to understand what these means
    cars, cars_2D = get_cars_list(cars, cars_dict, frames, WorldToCameraRotation, WorldToCameraTranslation,
                                  heightOffset, scale_x, K, filepath, find_poses=find_poses,
                                  datasetOptions=datasetOptions)



    people, people_2D = get_cars_list(people, people_dict, frames, WorldToCameraRotation, WorldToCameraTranslation,
                                      heightOffset, scale_x, K, filepath, people_flag=True, find_poses=find_poses,
                                      datasetOptions=datasetOptions)
    #print (" People "+str(people))
    numValidFrames = min(len(cars_dict), len(people_dict))
    if len(frames) != numValidFrames:
        print("WARNING: the number of frames given initially is not equal to the data gathered !")
        frames = list(range(numValidFrames))

    # init_frames = frame id when each car_id and pedestrian_id appears for the first time in the scene
    # poses, cars 2D, people2D  = 2d stuff - not working now..
    # ped_dict = {pedestrian_id -> list of positions ordered by frame | positions are [x,y,z] space in voxels, world coordinate}
    # cars_dict = {car_id -> list of bbox ordered by frame | bboxes are ooab in voxels, [xmin, xmax+1, ymin, ymax+1, zmin, zmax+1] world coordinate}
    # cars, people have the same coordinates and bboxes as defined above that are organized differently:
    #       they lists of lists such [frame 0 ; frame 1 ; ... frame num frames], where each frame data is a list [ position / ooab for each pedestrian / car index]
    #                           note that info is indexed by the sorted keys of pedestrian / car_id !].

    ped_dict, poses, valid_ids, init_frames = get_cars_map(people_dict, frames, WorldToCameraRotation,
                                                           WorldToCameraTranslation, heightOffset, scale_x, K, filepath,
                                                           people_flag=True, datasetOptions=datasetOptions)

    car_dict, _, _, init_frames_cars = get_cars_map(cars_dict, frames, WorldToCameraRotation, WorldToCameraTranslation,
                                                    heightOffset, scale_x, K, filepath, people_flag=False,
                                                    datasetOptions=datasetOptions)

    return cars, people, ped_dict, cars_2D, people_2D, poses, valid_ids, car_dict, init_frames, init_frames_cars


# Reads the point cloud from a single frame (segmented labelels and rgb ones) and append those to the input/output dictionaries
# plyfile - is the frame with point cloud segmented data path
def combine_frames(WorldToCameraRotation, filepath, WorldToCameraTranslation,
                   middle_height, plyfile, reconstruction_label, reconstruction_rgb, reconstruction_origPoints,
                   scale, datasetOptions=None, params=None):
    filename = os.path.basename(plyfile)
    print(("Processing file ", plyfile))
    pointcloud_3D = read_3D_pointcloud(plyfile, '')  # THis is the semantic labeled point cloud

    # The file without _seg contains the RGB segmentation labels of each 3d point
    parts = filename.split('_seg')
    rgb_dict = get_rgb_point_cloud(filepath,
                                   parts)  # This is the RGB labels, rgb_dict{(x,y,z)}->(R,G,B) segmentation label

    numPoints = len(pointcloud_3D)
    logRate = numPoints / 10
    for index, point in enumerate(pointcloud_3D):
        if index % logRate == 0:
            print(f"Processing {index}/{numPoints}. Completion rate {index/numPoints * 100.0}")

        # Each 3D position in the dictionary contains at most 3 labels voted for this frame.
        # Take the one that is the most voted
        pos = (point[0], point[1], point[2])
        labels = [point[3]] if len(point) == 4 else [point[3], point[4], point[5]]
        counts = np.bincount(labels)
        label = np.argmax(counts)  # voted label

        # If car or pedestrian, remove from reconstruction

        if not isLabelIgnoredInReconstruction(labels, isCarlaLabel=True):
            point_new_coord = np.array(pos)

            # Does it needs to conversion or is it already in global space ?
            if datasetOptions is None or datasetOptions.dataAlreadyInWorldSpace is False:
                point_new_coord = np.squeeze(
                    np.asarray(
                        np.matmul(WorldToCameraRotation, np.array(pos).reshape((3, 1))) + WorldToCameraTranslation))
                point_new_coord = point_new_coord * [-1, -1, 1]  # In camera coordinate system  of cityscapes
                point_new_coord = np.matmul(P_inv, point_new_coord)  # Now the point is in vehicle space, citiyscapes

            # Scale the point as needed to our framework environment
            point_float_coord = point_new_coord * scale
            point_new_coord = tuple(np.round(point_float_coord).astype(int))

            # If road or sidewalks, just add the coordinate to the heights statistics
            # The second condition ensures that only the point data very close to left/right on our sensor are taken into account. There is a lot of noise around...
            if isCloudPointRoadOrSidewalk(pos, label):  # and (point_new_coord[1] < (1 * scale)):
                middle_height.append(point_float_coord[2])

            # If  point not in dict, add it.
            if point_new_coord not in reconstruction_label:
                reconstruction_label[point_new_coord] = []
                reconstruction_rgb[point_new_coord] = []
                reconstruction_origPoints[point_new_coord] = set()
            else:
                # If the previous points had only unsefull points, erase all
                # We add only a single unsefull point that's why we can put this condition easly
                if isUsefulLabelForReconstruction(label) and len(reconstruction_label[point_new_coord]) == 1 and (
                not isUsefulLabelForReconstruction(reconstruction_label[point_new_coord][0])):
                    reconstruction_label[point_new_coord].clear()

            # Add point only if it is other than no label, or it is the only one available
            if isUsefulLabelForReconstruction(label) or len(reconstruction_label[point_new_coord]) == 0:
                reconstruction_label[point_new_coord].append(label)
                reconstruction_rgb[point_new_coord].append(rgb_dict[pos])  # Set the RGB segmentation value

            # Add the original floating point anyway to the discretized list
            if hasattr(params, 'KEEP_ORIGINAL_FLOATING_POINTS') and params.KEEP_ORIGINAL_FLOATING_POINTS:
                reconstruction_origPoints[point_new_coord].add(tuple(point_float_coord) + tuple(rgb_dict[pos]))


''' 
REconstructs or read one already existing in filepath if recalculate is False
local_scale_x - is the scale to move from the source data system to our framework (voxels, which in this moment 1m = 5 voxels, each having 20 cm)
label_mapping - must contain a table that maps from source segmentation labels to citiscapes segmentation labels (which is the reference ones). 
find_poses - if you need 2D labels
datasetOptions - various hacks to inject your own options from the source dataset...
'''


def reconstruct3D_ply(filepath, local_scale_x=5, recalculate=False, find_poses=False, read_3D=True,
                      datasetOptions=None, params=None):
    frameSkip = 1  # How many frames modulo to skip for reconstruction (if needed. default is 50)
    numInitialFramesToIgnore = 0  # How many frames in the beginning of the scene to ignore

    FRAMEINDEX_MIN, FRAMEINDEX_MAX = None, None
    if datasetOptions is None or datasetOptions.dataAlreadyInWorldSpace is False:
        print (" Data is already in world coordinates")
        WorldToCameraRotation, WorldToCameraTranslation, frames = get_camera_matrix(filepath)
        frameSkip = 50
        numInitialFramesToIgnore = 1
    else:
        WorldToCameraRotation = np.eye(3)
        WorldToCameraTranslation = np.array(get_translation(filepath,datasetOptions.realtime_data)).reshape((3,1))
        frames = datasetOptions.framesIndicesCaptured
        frameSkip = datasetOptions.frameSkip
        numInitialFramesToIgnore = datasetOptions.numInitialFramesToIgnore
        #print(" Data is not in world coordinates "+str(frames)+" ignore frames "+str(numInitialFramesToIgnore))

    heroCarDetails = get_herocartransform(filepath)


    FRAMEINDEX_MIN = int(min(frames))
    FRAMEINDEX_MAX = int(max(frames))
    print (" min frames "+str(FRAMEINDEX_MIN)+" max "+str(FRAMEINDEX_MAX))

    people = []
    cars = []
    for frame in range(min(0, FRAMEINDEX_MIN), (FRAMEINDEX_MAX + 1)):
        cars.append([])
        people.append([])

    # Try read point cloud
    reconstruction_path = os.path.join(filepath, FILENAME_COMBINED_CARLA_ENV_POINTCLOUD)
    reconstruction_path_segColored = os.path.join(filepath, FILENAME_COMBINED_CARLA_ENV_POINTCLOUD_SEGCOLOR)
    scale = local_scale_x
    labels_mapping = get_label_mapping()
    print (" Reconstruction path " +str(reconstruction_path))

    centeringFilePath = os.path.join(filepath, FILENAME_CENTERING_ENV)

    if read_3D:
        print(" Read 3D "+reconstruction_path+" centering "+str(centeringFilePath)+" recaclulate "+str(recalculate))
        # If reconstruction files are saved on disk already, and not forcing a recalculation, reload it
        # Centering file is a dumped dictionary contains 3 values: the hight offset, middle - world to camera translation, scale - scale value for converting from dataspace to voxel space
        if os.path.isfile(reconstruction_path) and os.path.isfile(centeringFilePath) and not recalculate:
            print("Loading existing reconstruction")

            with open(centeringFilePath, 'rb') as centeringFileHandle:
                centering = pickle.load(centeringFileHandle, encoding="latin1", fix_imports=True)
                scale = centering['scale']

            plydata = PlyData.read(reconstruction_path)
            nr_points = plydata.elements[0].count
            pointcloud_3D = np.array([plydata['vertex'][k] for k in range(nr_points)])
            final_reconstruction = {}  # final_reconstruction[(x,y,z)] = (rgb label, label id from segmentation)
            for point in pointcloud_3D:
                final_reconstruction[(int(point[0]), int(point[1]), int(point[2]))] = (point[3], point[4], point[5], getLabelFromPoint(point))#(*getRgbFromPoint(point), getLabelFromPoint(point))
        else:
            print("Create reconstruction")

            # For each 3D (x,y,z) point, which is the rgb and label (segmentation) in source domain ?
            reconstruction_rgb = {}
            reconstruction_label = {}
            # For each 3D point, which are the original floating point associated with a voxelized / rounded one ?
            reconstruction_origPoints = {}  # (x,y,z) -> [list of original positions in the point cloud]

            # Collect Middle Heights of the analized segmentation frame
            middle_height = []

            # A small hack to force frame 0 even if not specified...it is usefull because sometimes this represent the startup position of the reference point and we might want to get statstics out of that
            isFrame_0_needed = (datasetOptions is not None and datasetOptions.dataAlreadyInWorldSpace == True)

            # Take all *seg.ply files (basically different frames along the scene) in the filepath and combine the information from them in the hashes created above
            # These files contains the segmentated 3D point cloud to build the maps above.
            for plyfile in sorted(glob.glob(os.path.join(filepath, '0*_seg.ply'))):
                frame_nbr = os.path.basename(plyfile)
                frame_nbr = int(frame_nbr[:-len("_seg.ply")])
                # frames.append(frame_nbr)

                if (FRAMEINDEX_MIN is not None and frame_nbr < FRAMEINDEX_MIN) and (not isFrame_0_needed):
                    continue
                if FRAMEINDEX_MAX is not None and frame_nbr > FRAMEINDEX_MAX:
                    break

                if (frame_nbr >= numInitialFramesToIgnore and frame_nbr % frameSkip == 0) or (
                        frame_nbr == 0 and isFrame_0_needed):
                    combine_frames(WorldToCameraRotation, filepath, WorldToCameraTranslation, middle_height, plyfile,
                                   reconstruction_label, reconstruction_rgb, reconstruction_origPoints, scale,
                                   datasetOptions, params)

            # Offset everything by the minimum of heights in the scene (height normalization)
            height = np.min(middle_height) if len(middle_height) > 0 else 0.0
            height_asint = int(height)

            def offsetZAxisKey(original, offset):
                new_key = int(original) - int(offset)
                return new_key

            # Take all 3D point cloud votes and select the mode for each one.
            combined_reconstruction = []
            final_reconstruction = {}
            for point in reconstruction_rgb:
                assert isinstance(height, float) or isinstance(height, np.float32) or isinstance(height, np.float64), "WE ARE LOOSING PRECISION !! type({0})".format(type(height))
                assert len(reconstruction_rgb[point]) > 0 and (params == None or params.KEEP_ORIGINAL_FLOATING_POINTS == False or len(reconstruction_origPoints) > 0), "No points inside"
                assert params == None or params.KEEP_ORIGINAL_FLOATING_POINTS == False or (len(reconstruction_origPoints) == len(reconstruction_rgb)), "Not all dictionaries have the same size !! WRONG !!"
                assert (len(reconstruction_rgb) == len(reconstruction_label)), "Not all dictionaries have the same size !! WRONG !!"

                # Get the voted rgb and label for this point
                rgb = max(set(reconstruction_rgb[point]), key=reconstruction_rgb[point].count)
                label = max(set(reconstruction_label[point]), key=reconstruction_label[point].count)
                mapped_label = labels_mapping[label]  # Convert to the output labels map

                voxelized_point_pos = (int(point[0]), int(point[1]), offsetZAxisKey(point[2], height_asint))

                if params != None and params.KEEP_ORIGINAL_FLOATING_POINTS:
                    # Note that in the reconstructed space we subtract the height
                    # This contains the output that we are going to write to the output file.
                    # So add all the floating points coresponding to the discretized coordinate with the same RGB and label !
                    for origP in reconstruction_origPoints[point]:
                        assert isinstance(origP[0], np.float32) or isinstance(origP[0], np.float64)
                        combined_reconstruction.append((origP[0], origP[1], (origP[2] - height), origP[3], origP[4], origP[5], mapped_label))
                else:
                    combined_reconstruction.append((voxelized_point_pos[0], voxelized_point_pos[1], voxelized_point_pos[2], rgb[0], rgb[1], rgb[2], mapped_label))

                # The output for the episode stuff doesn't care so just discretize
                final_reconstruction[voxelized_point_pos] = (rgb[0], rgb[1], rgb[2], mapped_label)

            if params:
                if params.NOISE_FILTERING_WITH_KNN:
                    combined_reconstruction = refinePointCloud_knn(combined_reconstruction, scaleAppliedOverOriginalData=scale,
                                                                   discretizedDictionaryCloudData=final_reconstruction)
                elif params.NOISE_FILTERING_WITH_VOXELIZATION:
                    combined_reconstruction = refinePointCloud_voxelization(combined_reconstruction, scaleAppliedOverOriginalData=scale,
                                                                   discretizedDictionaryCloudData=final_reconstruction)
                if params.NOISE_FILTERING_WITH_KNNStatistical:
                    combined_reconstruction = refinePointCloud_knnStatistical(combined_reconstruction, scaleAppliedOverOriginalData=scale,
                                                                   discretizedDictionaryCloudData=final_reconstruction)


            # Build per global environment + each frame motion entities combined
            if params and params.POINT_CLOUD_FOR_MOTION_FRAMES == True:
                # From the output dictionaries constructed above, we are going to reuse the reconstruction_origPoints.
                # In this purpose, what we need is to modify the reconstruction_origPoints keys, by subtracting the height
                reconstruction_rgb, reconstruction_label = None, None
                if params and params.KEEP_ORIGINAL_FLOATING_POINTS:
                    reconstruction_origPoints_new = {}
                    for xyz_int_key, setOfOrigFloatPoints in list(reconstruction_origPoints.items()):
                        new_xyz_int_key = (xyz_int_key[0], xyz_int_key[1], offsetZAxisKey(xyz_int_key[2], height_asint))
                        setOfNewFloatPoints = set()
                        for eachOrigP in setOfOrigFloatPoints:
                            x, y, z, r, g, b = eachOrigP
                            z -= height
                            setOfNewFloatPoints.add((x, y, z, r, g, b))
                        reconstruction_origPoints_new[new_xyz_int_key] = setOfNewFloatPoints
                    reconstruction_origPoints = reconstruction_origPoints_new

                print("Now building the environment + motion combined point cloud for each frame....")
                for frameIdx in range(min(0, FRAMEINDEX_MIN), (FRAMEINDEX_MAX + 1)):
                    if frameIdx % 50 == 0:
                        print(("Reconstructing motion frame at {0}".format(frameIdx)))

                    reconstruction_path_frame = (
                        os.path.join(filepath, FILENAME_COMBINED_CARLA_ENV_POINTCLOUD_Fi)).format(frameIdx)
                    reconstruction_path_segColored_frame = os.path.join(filepath, FILENAME_COMBINED_CARLA_ENV_POINTCLOUD_SEGCOLOR_Fi).format(frameIdx)

                    # Copy the original final reconstruction thing to a new S_frame
                    final_reconstruction_per_frame = copy.copy(final_reconstruction)

                    inputFramePath_rgb = os.path.join(filepath, ("{0:05d}_motion.ply").format(frameIdx))
                    plydataMotion_rgb = PlyData.read(inputFramePath_rgb)
                    inputFramePath_seg = os.path.join(filepath, ("{0:05d}_motionseg.ply").format(frameIdx))
                    plydataMotion_seg = PlyData.read(inputFramePath_seg)

                    nr_points = plydataMotion_rgb.elements[0].count
                    assert plydataMotion_rgb.elements[0].count == nr_points, "They should have the same num points"

                    pointcloud_3D_rgb = np.array([plydataMotion_rgb['vertex'][k] for k in range(nr_points)])
                    pointcloud_3D_seg = np.array([plydataMotion_seg['vertex'][k] for k in range(nr_points)])

                    if params and params.KEEP_ORIGINAL_FLOATING_POINTS:
                        # Copy the original float points from reconstruction
                        reconstruction_frameMotion_origPoints = copy.copy(reconstruction_origPoints)

                    # Add the motion only points to S_frame
                    for index, _ in enumerate(pointcloud_3D_rgb):
                        pointdata_rbg = pointcloud_3D_rgb[index]
                        pointdata_seg = pointcloud_3D_seg[index]
                        xyz = np.array([pointdata_rbg[0], pointdata_rbg[1], pointdata_rbg[2]])
                        xyz_seg = np.array([pointdata_seg[0], pointdata_seg[1], pointdata_seg[2]])
                        assert (np.allclose(xyz, xyz_seg)), "Positions are not similar !!"

                        label = pointdata_seg[3]
                        r, g, b = pointdata_rbg[3], pointdata_rbg[4], pointdata_rbg[5]

                        # Scale coordinate and get both float and rounded int space
                        xyz_float = xyz * scale
                        xyz_int = np.round(xyz_float).astype(int)

                        # Get the height offset in both float and int space
                        xyz_float[2] -= height
                        assert isinstance(height, np.float32) or isinstance(height, np.float64)
                        xyz_int[2] = offsetZAxisKey(xyz_int[2], height_asint)
                        xyz_askey = tuple(xyz_int)

                        final_reconstruction_per_frame[xyz_askey] = (r, g, b, label)

                        if params and params.KEEP_ORIGINAL_FLOATING_POINTS:
                            # Append to the float points from the frame
                            if xyz_askey not in reconstruction_frameMotion_origPoints:
                                reconstruction_frameMotion_origPoints[xyz_askey] = set()
                            reconstruction_frameMotion_origPoints[xyz_askey].add(tuple(xyz_float)+(r,g,b))

                    combined_reconstruction_frame = []
                    for point_key, point_data in list(final_reconstruction_per_frame.items()):
                        if params and params.KEEP_ORIGINAL_FLOATING_POINTS:
                            # Add all original float points
                            for origP in reconstruction_frameMotion_origPoints[point_key]:
                                assert isinstance(origP[0], np.float32) or isinstance(origP[0], np.float64)
                                combined_reconstruction_frame.append((origP[0], origP[1], origP[2], origP[3], origP[4], origP[5], point_data[3]))
                        else:
                            combined_reconstruction_frame.append((point_key[0], point_key[1], point_key[2], point_data[0], point_data[1], point_data[2], point_data[3]))

                    #print("Number of keys in frame {2}, combined reconstruction using floats {0}, using discretization {1}".format(len(combined_reconstruction_frame), len(final_reconstruction_per_frame), frameIdx))

                    #combined_reconstruction_frame = refinePointCloud(combined_reconstruction_frame)
                    save_3d_pointcloud(combined_reconstruction_frame, reconstruction_path_frame)
                    save_3d_pointcloud_asSegLabel(combined_reconstruction_frame, reconstruction_path_segColored_frame)

            # Save the segmentation reconstruction file in the same folder for caching purposes
            #print("DEBUG: number of keys in combined reconstruction using floats {0}, using discretization {1}",len(combined_reconstruction), len(final_reconstruction))
            save_3d_pointcloud(combined_reconstruction, reconstruction_path)
            save_3d_pointcloud_asSegLabel(combined_reconstruction, reconstruction_path_segColored)

            allPointsCoords = np.array(list(final_reconstruction.keys()))
            minBBox = allPointsCoords.min(axis=0)
            maxBBox = allPointsCoords.max(axis=0)

            print(" min bbox "+str(minBBox))
            print(" max bbox " + str(maxBBox))
            # Save the centering file
            centering = {}
            centering['height'] = height
            centering['middle'] = WorldToCameraTranslation
            centering['scale'] = local_scale_x
            centering['frame_min'] = FRAMEINDEX_MIN
            centering['frame_max'] = FRAMEINDEX_MAX
            centering['min_bbox'] = list(minBBox)
            centering['max_bbox'] = list(maxBBox)

            with open(centeringFilePath, 'wb') as centeringFileHandle:
                pickle.dump(centering, centeringFileHandle,
                            protocol=2)  # Protocol 2 ensures compatibility between python 2 and 3
    else:
        centering = {}
        with open(centeringFilePath, 'rb') as centeringFileHandle:
            centering = pickle.load(centeringFileHandle, encoding="latin1", fix_imports=True)

        plydata = PlyData.read(reconstruction_path)
        nr_points = plydata.elements[0].count
        pointcloud_3D = np.array([plydata['vertex'][k] for k in range(nr_points)])

        # Note : the height (Z axis) offset is already offset in the reconstruction
        final_reconstruction = {}
        for point in pointcloud_3D:
            final_reconstruction[(int(point[0]), int(point[1]), int(point[2]))] = (
                point[3], point[4], point[5], point[6])


    # See the comments inside the called function, before return to understand meaning of all these
    cars, people, ped_dict, cars_2D, people_2D, poses, valid_ids, car_dict, init_frames, init_frames_cars = get_people_and_cars(
        WorldToCameraRotation,
        cars,
        filepath,
        frames,
        centering['middle'],
        centering['height'],
        people,
        centering['scale'],
        find_poses=find_poses,
        datasetOptions=datasetOptions)
    if find_poses:
        save_2D_poses(filepath, people_2D, frames)

    return final_reconstruction, people, cars, local_scale_x, ped_dict, cars_2D, people_2D, valid_ids, car_dict, init_frames, init_frames_cars, heroCarDetails


# TODO: create a factory here maybe in the future to handle different datasets...
def CreateDefaultDatasetOptions_Waymo(metadata):
    class DatasetOptions:
        def __init__(self):
            self.dataAlreadyInWorldSpace = True  # Is data already converted to world space ?
            self.framesIndicesCaptured = list(np.arange(min(0, metadata['frame_min']), metadata['frame_max']))
            self.LIMIT_FRAME_NUMBER = metadata['frame_max']  # How much we want to go with processing...max is default
            self.KEEP_ORIGINAL_FLOATING_POINTS = False
            self.frameSkip = 1  # Not needed on inference or training, just for reconstruction process
            self.numInitialFramesToIgnore = (metadata[
                                                 'frame_min'] - 1)  # Somehow redundant, but it is used in CARLA for example to force ignore first frame...

            self.filter2D = True # if used, the filter will try to extract noise from the reconstructed environment using a simple median filter
            self.isColmap = False
            self.envSettings = None # Used only for colmap for now, maybe we should use it uniformly
            self.realtime_data = False
    options = DatasetOptions()
    return options

def CreateDefaultDatasetOptions_CarlaRealTime(metadata):
    class DatasetOptions:
        def __init__(self):
            self.dataAlreadyInWorldSpace = True  # Is data already converted to world space ?
            self.framesIndicesCaptured = list(np.arange(0, metadata.onlineEnvSettings.numFramesPerEpisode))
            self.LIMIT_FRAME_NUMBER = 99999999
            #self.KEEP_ORIGINAL_FLOATING_POINTS = False
            self.frameSkip = 1  # Not needed on inference or training, just for reconstruction process
            self.numInitialFramesToIgnore = -1 #(metadata[
            #                                     'frame_min'] - 1)  # Somehow redundant, but it is used in CARLA for example to force ignore first frame...

            self.filter2D = False # if used, the filter will try to extract noise from the reconstructed environment using a simple median filter
            self.isColmap = False
            self.envSettings = None # Used only for colmap for now, maybe we should use it uniformly
            self.realtime_data=True
            self.filter2D=False
    options = DatasetOptions()
    return options

def CreateDefaultDatasetOptions_Cityscapes(metadata):
    class DatasetOptions:
        def __init__(self):
            self.filter2D = True # if used, the filter will try to extract noise from the reconstructed environment using a simple median filter
            self.LIMIT_FRAME_NUMBER = 99999999
            self.dataAlreadyInWorldSpace = False  # Is data already converted to world space ?
            self.isColmap = True
            self.envSettings = None
            self.realtime_data =False

    options = DatasetOptions()
    return options
 
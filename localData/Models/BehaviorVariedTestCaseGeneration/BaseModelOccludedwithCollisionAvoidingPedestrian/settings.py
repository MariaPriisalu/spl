
from datetime import datetime
import os
import sys

import shlex
import subprocess
import time

from dotmap import DotMap
import numpy as np
from commonUtils.ReconstructionUtils import LAST_CITYSCAPES_SEMLABEL
USE_LOCAL_PATHS = 0

RANDOM_SEED=8018#8018#2610#1234- random seed for training

DEFAULT_SKIP_RATE_ON_EVALUATION_CARLA = 4
DEFAULT_SKIP_RATE_ON_EVALUATION_WAYMO = 1

# Define some paths, names for models and datasets
CARLA_CACHE_PREFIX_EVALUATE = "carla_evaluate"
CARLA_CACHE_PREFIX_EVALUATE_TOY = "carla_evaluate_toy"
CARLA_CACHE_PREFIX_EVALUATE_NEW = "carla_evaluate_new"
CARLA_CACHE_PREFIX_EVALUATE_REALTIME = "carla_evaluate_realtime"
WAYMO_CACHE_PREFIX_EVALUATE = "waymo_evaluate"

CARLA_CACHE_PREFIX_EVALUATE_SUPERVISED = "carla_evaluate_sup"
WAYMO_CACHE_PREFIX_EVALUATE_SUPERVISED = "waymo_evaluate_sup"

CARLA_CACHE_PREFIX_TRAIN = "carla_train"
CARLA_CACHE_PREFIX_TRAIN_TOY = "carla_train_toy"
CARLA_CACHE_PREFIX_TRAIN_NEW = "carla_train_new"
CARLA_CACHE_PREFIX_TRAIN_REALTIME = "carla_train_realtime"
WAYMO_CACHE_PREFIX_TRAIN = "waymo_train"

CARLA_CACHE_PREFIX_TRAIN_SUPERVISED = "carla_train_sup"
WAYMO_CACHE_PREFIX_TRAIN_SUPERVISED = "waymo_train_sup"

CARLA_CACHE_PREFIX_TEST = "carla_test"
CARLA_CACHE_PREFIX_TEST_TOY = "carla_test_toy"
CARLA_CACHE_PREFIX_TEST_NEW = "carla_test_new"
CARLA_CACHE_PREFIX_TEST_REALTIME = "carla_test_realtime"
WAYMO_CACHE_PREFIX_TEST = "waymo_test"

CARLA_CACHE_PREFIX_TEST_SUPERVISED = "carla_test_sup"
WAYMO_CACHE_PREFIX_TEST_SUPERVISED = "waymo_test_sup"

STGCNN_CODEPATHS = ["nothing"] if USE_LOCAL_PATHS == 0 else ["Work/Social-STGCNN"]
STGCNN_SCRIPTNAME = os.path.join(STGCNN_CODEPATHS[0],"test_stgcnn.py")
STGCNN_MODELPATH_CARLA = "nothing" if USE_LOCAL_PATHS == 0 else os.path.join(STGCNN_CODEPATHS[0], "checkpoint/custom_carla")
STGCNN_MODELPATH_WAYMO = "nothing" if USE_LOCAL_PATHS == 0 else os.path.join(STGCNN_CODEPATHS[0], "checkpoint/custom_waymo")

STGAT_CODEPATHS         = ["nothing"] if USE_LOCAL_PATHS == 0 else ["Work/STGAT/STGAT"]
STGAT_SCRIPTNAME        = os.path.join(STGAT_CODEPATHS[0],"evaluate_model.py")
STGAT_MODELPATH_CARLA   = "nothing" if USE_LOCAL_PATHS == 0 else os.path.join(STGAT_CODEPATHS[0], "models/carla_len8_pred8_model.pth.tar")
STGAT_MODELPATH_WAYMO   = "nothing" if USE_LOCAL_PATHS == 0 else os.path.join(STGAT_CODEPATHS[0], "models/waymo_len8_pred8_model.pth.tar")

SGAN_CODEPATHS         = ["nothing"] if USE_LOCAL_PATHS == 0 else ["Work/sgan"]
sys.path.append(SGAN_CODEPATHS) # Pretty hacky i know...
SGAN_SCRIPTNAME        = os.path.join(SGAN_CODEPATHS[0],"scripts/evaluate_model.py")
SGAN_MODELPATH_CARLA   = "nothing" if USE_LOCAL_PATHS == 0 else os.path.join(SGAN_CODEPATHS[0], "models/sgan-p-models/carla_8_model.pt" )
SGAN_MODELPATH_WAYMO   = "nothing" if USE_LOCAL_PATHS == 0 else os.path.join(SGAN_CODEPATHS[0], "models/sgan-p-models/waymo_8_model.pt" )

METER_TO_VOXEL = 5.0
VOXEL_TO_METER = 0.2
AGENT_MAX_HEIGHT_VOXELS =  10 # 2meters
NUM_SEM_CLASSES = float(LAST_CITYSCAPES_SEMLABEL)
NUM_SEM_CLASSES_ASINT = LAST_CITYSCAPES_SEMLABEL
POSE_DIM=512

class RLCarlaOnlineEnv():
    width = 128
    height = 256
    numFramesPerEpisode = 500 # TODO: this is from gatherdataset param. Instead provide metadataoutput or just put this at least in the json config !!!
    no_renderig = True
    no_client_rendering = True
    numCarlaVehicles = 3
    numCarlaPedestrians = 5
    scenesConfigFile="Datasets/Data1/scenesConfig.json"
    outputDataBasePath="Datasets/Data1"
    scenesToUse= ["Datasets/carla-realtime/train/test_0",
                  "Datasets/carla-realtime/train/test_1",
                  ]#["DatasetCustom/Data1/Town03/0/157"] #["scene1_0"]#, "scene2"] # Could be just all from sceneConfig file. What do we want ?!
    scenesPathToUse=scenesToUse#[]
    # for index, sceneName in enumerate(scenesToUse):
    #     scenesPathToUse.append(os.path.join(outputDataBasePath, scenesToUse[index]))
    #     assert os.path.exists(scenesPathToUse[-1])

def get_flat_indx_list(stat):
    values_list=[]
    for value in  stat:
        if type(value) == list:
            for element in value:
                values_list.append(element)
        else:
            values_list.append(value)
    return values_list

# Add external code to the python path such that you can run the scripts from external models

# --------------------------------------------------- Initialization indexes
allExternalCodePaths = []
allExternalCodePaths.extend(STGCNN_CODEPATHS)

PEDESTRIAN_INITIALIZATION_CODE=DotMap()
PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian=1
PEDESTRIAN_INITIALIZATION_CODE.by_car=2
PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian=3
PEDESTRIAN_INITIALIZATION_CODE.randomly=4
PEDESTRIAN_INITIALIZATION_CODE.in_front_of_car=5
PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian=6
PEDESTRIAN_INITIALIZATION_CODE.learn_initialization=7 # i.e. learnt by init_net
PEDESTRIAN_INITIALIZATION_CODE.on_pavement=8
PEDESTRIAN_INITIALIZATION_CODE.near_obstacle=9
PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian_trajectory=10

# ---------------------------------------------------  Measures indexes
PEDESTRIAN_MEASURES_INDX=DotMap()
PEDESTRIAN_MEASURES_INDX.hit_by_car=0
PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory=1
PEDESTRIAN_MEASURES_INDX.iou_pavement=2
PEDESTRIAN_MEASURES_INDX.hit_obstacles=3
PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init=4
PEDESTRIAN_MEASURES_INDX.out_of_axis=5
PEDESTRIAN_MEASURES_INDX.dist_to_final_pos=6
PEDESTRIAN_MEASURES_INDX.dist_to_goal=7
PEDESTRIAN_MEASURES_INDX.hit_pedestrians=8
PEDESTRIAN_MEASURES_INDX.dist_to_goal_from_current=9
PEDESTRIAN_MEASURES_INDX.one_step_prediction_error=9
PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap=10
PEDESTRIAN_MEASURES_INDX.change_in_direction=11
PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init=12
PEDESTRIAN_MEASURES_INDX.goal_reached=13
PEDESTRIAN_MEASURES_INDX.agent_dead=14
PEDESTRIAN_MEASURES_INDX.difference_to_goal_time=15
PEDESTRIAN_MEASURES_INDX.change_in_pose=16
PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car=17
PEDESTRIAN_MEASURES_INDX.distracted=18
PEDESTRIAN_MEASURES_INDX.hit_by_hero_car=19

# ---- Car ------
CAR_MEASURES_INDX = PEDESTRIAN_MEASURES_INDX # Change if necessary!
CAR_MEASURES_INDX.dist_to_closest_pedestrian=1
CAR_MEASURES_INDX.dist_to_closest_car=10
CAR_MEASURES_INDX.dist_to_agent=17
CAR_MEASURES_INDX.hit_by_agent=19


# ---- Number of measures ------
NBR_MEASURES=max(PEDESTRIAN_MEASURES_INDX.values())+1
NBR_MEASURES_CAR = max(CAR_MEASURES_INDX.values())+1

# ---------------------------------------------------  Rewards indexes
PEDESTRIAN_REWARD_INDX=DotMap()
PEDESTRIAN_REWARD_INDX.collision_with_car=0
PEDESTRIAN_REWARD_INDX.pedestrian_heatmap=1
PEDESTRIAN_REWARD_INDX.on_pavement=2
PEDESTRIAN_REWARD_INDX.collision_with_objects=3
PEDESTRIAN_REWARD_INDX.distance_travelled=4
PEDESTRIAN_REWARD_INDX.out_of_axis=5
PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal=6
PEDESTRIAN_REWARD_INDX.collision_with_pedestrian=7
PEDESTRIAN_REWARD_INDX.reached_goal=8
PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory=9
PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently=10
PEDESTRIAN_REWARD_INDX.one_step_prediction_error=11
PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance=12
PEDESTRIAN_REWARD_INDX.not_on_time_at_goal=13
PEDESTRIAN_REWARD_INDX.large_change_in_pose=14
PEDESTRIAN_REWARD_INDX.inverse_dist_to_car=15
PEDESTRIAN_REWARD_INDX.collision_with_car_agent=16

# ---- Car ------
CAR_REWARD_INDX=DotMap()
CAR_REWARD_INDX.distance_travelled=0
CAR_REWARD_INDX.collision_pedestrian_with_car=1
CAR_REWARD_INDX.distance_travelled_towards_goal=2
CAR_REWARD_INDX.reached_goal=3
CAR_REWARD_INDX.collision_car_with_car=4
CAR_REWARD_INDX.penalty_for_intersection_with_sidewalk=5
CAR_REWARD_INDX.collision_car_with_objects=6
CAR_REWARD_INDX.penalty_for_speeding=7

# ---- Number of reward weights ------
NBR_REWARD_WEIGHTS=max(PEDESTRIAN_REWARD_INDX.values())+1
NBR_REWARD_CAR = max(CAR_REWARD_INDX.values())+1

# ---------------------------------------------------  Statistics indexes pedestrian agent
STATISTICS_INDX=DotMap()
STATISTICS_INDX.agent_pos=[0, 3]
STATISTICS_INDX.velocity=[3, 6]
STATISTICS_INDX.action=6
STATISTICS_INDX.probabilities=[7,34]
STATISTICS_INDX.angle=33
STATISTICS_INDX.reward=34
STATISTICS_INDX.reward_d=35
STATISTICS_INDX.loss=36
STATISTICS_INDX.speed=37
STATISTICS_INDX.measures=[38, 38+NBR_MEASURES]
STATISTICS_INDX.init_method=38+NBR_MEASURES
STATISTICS_INDX.goal=38+NBR_MEASURES
NBR_STATS=max(get_flat_indx_list(STATISTICS_INDX.values()))+1

# ---- Pose ------
STATISTICS_INDX_POSE=DotMap()
STATISTICS_INDX_POSE.pose=[0, 93]
STATISTICS_INDX_POSE.agent_high_frq_pos=[93, 95]
STATISTICS_INDX_POSE.agent_pose_frames=95
STATISTICS_INDX_POSE.avg_speed=96
STATISTICS_INDX_POSE.agent_pose_hidden=[97, 97+512]
NBR_POSES=max(get_flat_indx_list(STATISTICS_INDX_POSE.values()))+1

# ---------------------------------------------------  Statistics indexes pedestrian initializer
# ---- Initializer Heatmap ------
STATISTICS_INDX_MAP=DotMap()
STATISTICS_INDX_MAP.prior=0
STATISTICS_INDX_MAP.init_distribution=1
STATISTICS_INDX_MAP.goal_prior=2
STATISTICS_INDX_MAP.goal_distribution=3
NBR_MAPS=max(get_flat_indx_list(STATISTICS_INDX_MAP.values()))+1

# ---- Initializer Heatmap statistics (takes less space than saving initializer output) ------
STATISTICS_INDX_MAP_STAT=DotMap()
STATISTICS_INDX_MAP_STAT.init_position_mode=[0,2]
STATISTICS_INDX_MAP_STAT.goal_position_mode=[2,4]
STATISTICS_INDX_MAP_STAT.init_prior_mode=[4,6]
STATISTICS_INDX_MAP_STAT.goal_prior_mode=[6,8]
STATISTICS_INDX_MAP_STAT.entropy=8
STATISTICS_INDX_MAP_STAT.entropy_goal=9
STATISTICS_INDX_MAP_STAT.entropy_prior=10
STATISTICS_INDX_MAP_STAT.entropy_prior_goal=11
STATISTICS_INDX_MAP_STAT.kullback_leibler_divergence_init_and_prior=12
STATISTICS_INDX_MAP_STAT.kullback_leibler_divergence_init_and_prior_goal=13
STATISTICS_INDX_MAP_STAT.prior_init_difference=14
STATISTICS_INDX_MAP_STAT.prior_init_difference_goal=15
NBR_MAP_STATS=max(get_flat_indx_list(STATISTICS_INDX_MAP_STAT.values()))+1

# ---- Initializer statistics on car that is being targeted by initializer ------
STATISTICS_INDX_CAR_INIT=DotMap()
STATISTICS_INDX_CAR_INIT.car_id=0
STATISTICS_INDX_CAR_INIT.car_pos=[1,3]
STATISTICS_INDX_CAR_INIT.car_vel=[3,5]
STATISTICS_INDX_CAR_INIT.manual_goal=[5,7]
NBR_CAR_MAP=max(get_flat_indx_list(STATISTICS_INDX_CAR_INIT.values()))+1

# ---------------------------------------------------  Statistics indexes car model
STATISTICS_INDX_CAR=DotMap()
STATISTICS_INDX_CAR.agent_pos=[0, 3]
STATISTICS_INDX_CAR.velocity=[3, 6]
STATISTICS_INDX_CAR.action=6
STATISTICS_INDX_CAR.probabilities=[7,9]
STATISTICS_INDX_CAR.goal=9
STATISTICS_INDX_CAR.reward=10
STATISTICS_INDX_CAR.reward_d=11
STATISTICS_INDX_CAR.loss=12
STATISTICS_INDX_CAR.speed=13
STATISTICS_INDX_CAR.bbox=[14, 20]
STATISTICS_INDX_CAR.measures=[20, 20+NBR_MEASURES_CAR]
STATISTICS_INDX_CAR.dist_to_agent=20+CAR_MEASURES_INDX.dist_to_agent
STATISTICS_INDX_CAR.angle=20+CAR_MEASURES_INDX.dist_to_agent+1
NBR_STATS_CAR=max(get_flat_indx_list(STATISTICS_INDX_CAR.values()))+1

# ---------------------------------------------------  Channels used in Episode class
CHANNELS=DotMap()
CHANNELS.rgb=[0,3]
CHANNELS.semantic=3
CHANNELS.pedestrian_trajectory=4
CHANNELS.cars_trajectory=5



for path in allExternalCodePaths:
    sys.path.append(path)

print((sys.path))


class run_settings(object):

    def __init__(self,evaluate=False, name_movie="", env_width=128, env_depth=256, env_height=32, likelihood=False,
                 evaluatedModel="", datasetToUse="carla",pfnn=True):

        # Main settings, defining type of run.
        self.goal_dir=True # Does pedestrian agent have goal?
        self.learn_init =True # Do we need to learn initialization
        self.learn_goal=False # Do we need to learn where to place the goal of the pedestrian at initialization
        self.learn_time=False # Do we need to learn temporal constraint given to pedestrian

        # Debugging, profiling, caching
        self.fastTrainEvalDebug = False
        self.profile=False
        self.useCaching = True

        self.VAL_JUMP_STEP_CARLA = DEFAULT_SKIP_RATE_ON_EVALUATION_CARLA # How many to skip to do faster evaluation
        self.VAL_JUMP_STEP_WAYMO = DEFAULT_SKIP_RATE_ON_EVALUATION_WAYMO

        # ---------------------------------------------------  Pedestrian reward weights
        # Reward weights.- weighst for different reward terms.
        # cars, people, pavement, objs, dist, out_of_axis
        self.reward_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0]
        # cars
        self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_car] =-2

        # Pedestrian heatmap reward
        self.reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap]=0.01
        # Pavement
        self.reward_weights[PEDESTRIAN_REWARD_INDX.on_pavement] =0#0.1# 0
        # Objs
        self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_objects] =-0.02
        # Reward for dist travelled
        self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled] =0#1#0#0.0001#-0.001#0.001#0.001#-
        # Out of axis
        self.reward_weights[PEDESTRIAN_REWARD_INDX.out_of_axis] = 0
        # linear reward for dist, gradual improvement towards goal

        self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] = 0.0 if self.goal_dir  == False else 0.1#0.1# 0.1#0.1
        # Neg reward bump into people
        self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] =-0.1#15# -0.15
        # Positive reward for reaching goal
        self.reward_weights[PEDESTRIAN_REWARD_INDX.reached_goal] = 0.0 if self.goal_dir == False or self.learn_init else 2#-2#1

        # Reward for being on top of pedestrian trajectory
        self.reward_weights[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] =0.01 #
        # Reward for not changing direction
        self.reward_weights[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] = 0#-0.01  #
        # Reward for being close to following agent
        self.reward_weights[PEDESTRIAN_REWARD_INDX.one_step_prediction_error] = 0#0.1
        # Penalty for flight of distance
        self.reward_weights[PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] =0#-0.001 #-0.1#-0.1# -0.0001 # 0.1

        # Penalty for not reaching goal on time
        self.reward_weights[PEDESTRIAN_REWARD_INDX.not_on_time_at_goal] =0#-0.0001  # 0#-0.1#-0.1# -0.0001 # 0.1

        # Penalty for large change in poses
        self.reward_weights[PEDESTRIAN_REWARD_INDX.large_change_in_pose] =0#-0.0001  # 0#-0.1#-0.1# -0.0001 # 0.1
        # distance to car reward
        self.reward_weights[PEDESTRIAN_REWARD_INDX.inverse_dist_to_car] =0#0.001#0#1  # -0.0001  # 0#-0.1#-0.1# -0.0001 # 0.1
        # collision with car agent
        self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_car_agent] = -2

        # ---------------------------------------------------  Car reward weights
        self.reward_weights_car=np.zeros(NBR_REWARD_CAR)
        # Reward for distnace travelled by car
        self.reward_weights_car[CAR_REWARD_INDX.distance_travelled] = 0.01#7/4#0.0001
        self.reward_weights_car[CAR_REWARD_INDX.collision_pedestrian_with_car] = -2
        self.reward_weights_car[CAR_REWARD_INDX.distance_travelled_towards_goal] =0#0.1
        self.reward_weights_car[CAR_REWARD_INDX.reached_goal] = 0#2
        self.reward_weights_car[CAR_REWARD_INDX.collision_car_with_car]=-2
        self.reward_weights_car[CAR_REWARD_INDX.penalty_for_intersection_with_sidewalk]=-.1
        self.reward_weights_car[CAR_REWARD_INDX.collision_car_with_objects] = -2
        self.reward_weights_car[CAR_REWARD_INDX.penalty_for_speeding]=0#-0.1

        # ---------------------------------------------------  Load weights

        experiment_name = "base_model_occluded"  # Whatever is being tested right now. It is not possible to keep track of runs without giving them distinct names!

        # When running on trained or pretrained weights.
        model ="run_agent__carla_pfnntrain_model_with_car_and_corrected_reward_occluded_2021-09-24-10-24-05.839933/model.ckpt-8"

        car_model = ""
        # ---------------------------------------------------  Pedestrian agent parameters
        # Pedestrian agent parameters
        self.curriculum_goal=False
        self.curriculum_reward = False
        self.curriculum_seq_len = False

        # Model input and architecture
        self.predict_future=True
        self.temporal =True
        self.lstm =True
        self.extend_lstm_net=True
        self.extend_lstm_net_further = True
        self.old_lstm=False
        self.old=False
        self.old_fc=False
        self.old_mem=False
        self.conv_bias=True
        self.past_filler=False
        self.N = 10

        # Alternative larger scale model choices
        self.resnet = False
        self.mem_2d = False
        self.mininet = False
        self.estimate_advantages = False

        # PFNN related paremeters
        self.pfnn=True#pfnn  # Is possible use the init parameter
        self.max_frames_to_init_pfnnvelocity = 300 # How many frames to allow PFNN init
        self.sigma_vel=0.5#2.0#2.0#1.8#0.1#0.5#2.0_ca
        self.action_freq=1
        self.pose =False

        # Action space choices
        self.velocity=True
        self.velocity_sigmoid=True
        self.acceleration = False
        self.detect_turning_points=False
        self.continous=False
        self.speed_input=False
        if self.learn_time:
            self.speed_input = True
        self.angular=False
        self.actions_3d = False
        self.constrain_move = False
        self.attention = False
        self.confined_actions = False
        self.reorder_actions = False

        # When to end episode
        self.end_on_bit_by_pedestrians = False  # End episode on collision with pedestrians.
        self.stop_on_goal = True

        # Training type
        self.refine_follow = False
        self.supervised = True #applies only when initialised on pedestrians
        self.heatmap = False

        # ---------------------------------------------------  Initializer parameters
        # Initializer parameters

        # Architecture parameters
        self.inilitializer_interpolate=True # interpolate convolutional layers?
        self.inilitializer_interpolated_add=True # add interpolates convolutional layers?
        self.sem_arch_init=False # alternative architecture
        self.gaussian_init_net=False
        self.separate_goal_net=False
        self.goal_gaussian=False
        self.init_std=2.0
        self.goal_std = 10.0#2.0#0.01#.5#2.0
        # Occlusion?
        self.use_occlusion=True
        self.lidar_occlusion=False
        self.evaluate_prior=False

        # ---------------------------------------------------  Car parameters
        # Car model
        self.car_input = "distance_to_car_and_pavement_intersection"  # "distance"# time_to_collision# scalar_product "difference" # "distance_to_car_and_pavement_intersection"
        self.supervised_car = False
        self.supervised_and_rl_car = False
        self.linear_car = True
        self.car_constant_speed = False
        self.angular_car=False
        self.angular_car_add_previous_angle=False
        self.sigma_car = 0.1  # variance for linear car
        self.sigma_car_angular = 0.1 # variance for linear car
        self.car_reference_speed = 40 / 3.6 * 5 / 17
        self.car_max_speed = 70 / 3.6 * 5 / 17
        self.allow_car_to_live_through_collisions = False
        self.lr_car = 0.1/6.0  # 0.1#0.005#0.005#0.0001#.5 * 1e-3  # learning rate # 5*1e-3

        # ---------------------------------------------------  Real time/ Hero car run settings
        # Real time env settings
        self.useRealTimeEnv = False # Train car?
        self.useHeroCar = True
        self.useRLToyCar = True
        self.train_init_and_pedestrian=False

        # Alternative training of initializer and car
        self.train_car_and_initialize = "according_to_stats"  # "simultaneously", "alternatively" ,"according_to_stats" or ""
        self.car_success_rate_threshold = 1.0#0.7
        self.initializer_collision_rate_threshold = 1.0
        self.num_init_epochs = 1
        self.num_car_epochs = 2
        if len(self.train_car_and_initialize)==0:
            self.keep_init_net_constant = True
            self.save_init_stats = False
        else:
            self.keep_init_net_constant = False
            self.save_init_stats = True  # instead of saving init maps, save only statistics. takes much less space.

        # Noise
        self.add_noise_to_car_input = False
        self.car_noise_probability = 0.3
        self.pedestrian_noise_probability = 0.3
        self.distracted_pedestrian = False
        self.avg_distraction_length = 2
        self.pedestrian_view_occluded = False

        # ---------------------------------------------------  General learning parameters (common to car, agent, initializer)
        # Run types
        self.stop_for_errors = False
        self.overfit = False
        self.save_frequency = 100  # How frequently to save stats when overfitting?
        self.likelihood = likelihood
        self.run_evaluation_on_validation_set = False

        # Training  model hyper parameters
        if self.learn_init:
            self.multiplicative_reward = True
        else:
            self.multiplicative_reward = False
        self.entr_par=0#.1#.2 # Entropy of pedestrian
        self.entr_par_init = 0.001  # .1#.2 # Entropy of initializer
        self.entr_par_goal = 0#0.001  # .1#.2 # Entropy of goal-initializer
        self.replay_buffer = 0  # 10
        self.polynomial_lr = False

        # Learning rate and batch size
        self.lr = 0.005  # 0.1#0.005#0.005#0.0001#.5 * 1e-3  # learning rate # 5*1e-3
        self.batch_size = 1  # 50
        #self.epsilon = 5  # 0.2

        # Discount rate
        self.gamma = .99  # 1
        self.people_traj_gamma = .95
        self.people_traj_tresh = 0.1

        # Test other types of agents.
        self.random_agent=False
        self.goal_agent=False
        self.pedestrian_agent=False
        self.carla_agent=False
        self.cv_agent =False
        self.toy_case=False

        # Sample actions from a distribution or sample the most likely action
        self.deterministic = True
        self.deterministic_test = True  # evaluate deterministrically
        if not self.deterministic:
            self.prob = 1

        # Dataset
        self.carla=True # Run on CARLA?
        self.new_carla=False
        self.new_carla_net=False
        self.realtime_carla=True
        self.realtime_carla_only = False
        self.waymo=False # Run on waymo?

        if self.realtime_carla or self.realtime_carla_only:
            self.onlineEnvSettings=RLCarlaOnlineEnv()

        # Other environemnt settings
        self.temp_case=False
        self.paralell=False
        self.pop=False
        self.social_lstm=False

        self.random_seed_np = RANDOM_SEED
        self.random_seed_tf = self.random_seed_np
        self.timing = False

        # ---------------------------------------------------  Pedestrian model architecture specific paramteres

        # Channels of pedestrian agent
        self.run_2D = True
        self.min_seg = True
        self.no_seg = False
        self.mem = False
        self.action_mem = True
        if self.mem or self.action_mem:
            self.nbr_timesteps = 10  # 32
        self.sem = 1  # [1-semantic channels ,0- one semantic channel,-1- no semantic channel]# 1
        self.cars = True  # Car channel?
        self.people = True  # People channel
        self.avoiding_people_step = 10000000
        self.car_var = True

        # Network layers -pedestrian agent and initializer
        self.pooling = [1, 2]
        self.num_layers = 2
        self.num_layers_init = 2
        # self.fully_connected_layer=0
        self.outfilters = [1, 1]  # 64,128]
        self.controller_len = -1  # -1
        self.sigma = 0.1
        self.max_speed = 2
        # Fully connected layers
        self.fc = [0, 0]  # 1052, 512
        self.fully_connected_layer = -1  # 128
        self.pool = False
        self.batch_normalize = False
        self.clip_gradients = False
        self.regularize = False
        self.normalize = False

        # ------------------------------------------------------- Set Run name


        if evaluate:
            self.carla = self.waymo = False
            if datasetToUse == "carla":  # default
                self.carla = True  # Run on CARLA?
            elif datasetToUse == "waymo":
                self.waymo = True
            else:
                assert datasetToUse == "cityscapes"
            # Note: if not, we run citiscapes...

        runMode = "evaluate_" if evaluate else "run_"
        self.name_movie = runMode

        # Put the name of the external model to run comparisons against, or None if working as standalone on our model only
        self.evaluateExternalModel = None if evaluatedModel == "" else evaluatedModel
        self.lastAgentExternalProcess = None
        # Basename
        if self.likelihood:
            self.name_movie += "likelihood_"
        else:
            self.name_movie += "agent_"

        # Which dataset ?
        if self.carla == True:
            self.name_movie += "_carla"
        elif self.waymo == True:
            self.name_movie += "_waymo"
        else:
            self.name_movie += "_cityscapes"

        # Agent type:
        if self.evaluateExternalModel:
            self.name_movie += "_" + self.evaluateExternalModel

        # Using pfnn ?
        if self.pfnn:
            self.name_movie += "_pfnn"

        if not evaluate:
            self.name_movie += experiment_name
        print ("Movie name " + self.name_movie)

        # ------------------------------------------------------- Evaluation parameters
        if evaluate:
            # Correct settings for evaluation!
            self.temporal = True
            self.predict_future = True
            self.continous = False
            self.overfit = False
            self.pfnn = True
            self.speed_input = False
            self.end_on_bit_by_pedestrians = False
            self.angular = False
            self.stop_on_goal = True
            self.replay_buffer = 0  # 10
            self.entr_par = 0  # .1#.2
            self.stop_for_errors = False
            self.overfit = False
            self.mininet = False
            self.refine_follow = False
            self.heatmap = False
            self.confined_actions = False
            self.reorder_actions = False
            self.run_evaluation_on_validation_set = False
            self.attention = False
            self.settings = False
            self.supervised = True
            self.random_agent = False
            self.goal_agent = False
            self.pedestrian_agent = False
            self.carla_agent = False
            self.cv_agent = False
            self.toy_case = False
            self.temp_case = False
            self.paralell = False
            self.pop = False
        # ------------------------------------------------------- PFNN parameters
        # PFNN and continous agent special paramters
        if self.pfnn and not self.continous:  # and self.lr:
            self.pose = True
            self.velocity = True
            self.velocity_sigmoid = True
            self.action_freq = 4
            if self.goal_dir:
                self.action_freq = 1
                self.reward_weights[6] = 0.1
                self.reward_weights[8] = 2
            else:
                self.action_freq = 4
                self.reward_weights[6] = 0
                self.reward_weights[8] = 0
            self.sigma_vel = 0.1#2.0
            self.lr = 0.001
            self.reward_weights[14] = -0.0001
        # ------------------------------------------------------- PATHS
        # Where to save model
        self.model_path = ""

        # Where to save statistics file
        # Statistics dir and img_dir are computed below. Please don't hardcode let it be data driven by settings..
        #if not os.path.exists(self.statistics_dir):
        if USE_LOCAL_PATHS == 0:
            self.profiler_file_path ="/profiler_files/"
            #  Save to /profiler_files/run_agent__carlatrain_distance_car.psats
            if not os.path.exists(""):
                self.statistics_dir = "Results/statistics"
                self.img_dir = "Results/agent/"
            else:
                self.statistics_dir = "Results/statistics"
                self.img_dir = "Results/agent/"

        else:
            self.profiler_file_path = "./"
            self.statistics_dir = "RL/localUserData/Results/statistics"
            self.img_dir = "RL/localUserData/Results/agent/"

        # Episode cache settings
        if not os.path.exists(""):
            self.target_episodesCache_Path = "Datasets/CachedEpisodes" if USE_LOCAL_PATHS == 0 else "RL/localUserData/CachedEpisodes"
        else:
            self.target_episodesCache_Path = "Datasets/CachedEpisodes"
        self.deleteCacheAtEachRun = False

        self.timestamp=datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')

        ### Change paths from here onwards
        self.pfnn_path="commonUtils/PFNNBaseCode/" if USE_LOCAL_PATHS == 0 else "commonUtils/PFNNBaseCode/"
        self.cadrl_path='/cadrl/cadrl_ros/' if USE_LOCAL_PATHS == 0 else 'RL/localUserData/cadrl/cadrl_ros/'

        # Where to save current experiment's settings file.
        self.path_settings_file = "/some_results/" if USE_LOCAL_PATHS == 0 else  "RL/localUserData/some_results/"  # "~/Documents/Results/obstacles/sem_working/"

        # Where to save statistics files from runs?
        self.evaluation_path=("Models/eval/" if USE_LOCAL_PATHS == 0 else "RL/localUserData/Models/eval/") + self.name_movie+"_"+self.timestamp

        # Where to save simple visualizations of agent behaviour?
        self.name_movie_eval = "Datasets/cityscapes/pointClouds/val/" if USE_LOCAL_PATHS == 0 else "RL/localUserData/Datasets/cityscapes/pointClouds/val/"

        self.LocalResultsPath = '/visualization-results/manual/' if USE_LOCAL_PATHS == 0 else "RL/localUserData/visualization-results/manual/"

        # Cityscapes reconstructions path
        self.colmap_path = "Datasets/colmap2/"

        # Path to other cityscapes files
        self.cityscapes_path="Datasets/cityscapes/"

        # Path to PANNET masks to find people in cityscapes.
        self.path_masks = os.path.join(self.cityscapes_path, "pannet2/")

        # Path to Camera matrices.
        self.camera_path=os.path.join(self.cityscapes_path,"camera_trainvaltest/camera")
        # Path to cityscapes disparity sequence -- Not needed.
        self.disparity_dir=os.path.join(self.cityscapes_path,"disparity_sequence_trainvaltest/disparity_sequence/")


        # Path to CARLA dataset
        self.carla_main="Datasets/carla-sync"
        self.carla_path = os.path.join(self.carla_main, "train/")
        self.carla_path_test = os.path.join(self.carla_main, "val/")

        # self.carla_path = os.path.join(self.carla_main, "new_data/")  # "Packages/CARLA_0.8.2/unrealistic_dataset/"
        # # Path to CARLA test set
        # self.carla_path_test =os.path.join(self.carla_main,"new_data2/")
        # # Path to CARLA visualization set
        self.carla_path_viz = os.path.join(self.carla_main,"new_data-viz/")

        # Path to Waymo dataset

        self.waymo_path="Datasets/WAYMOOUTPUTMIN/" if USE_LOCAL_PATHS == 0 else "RL/localUserData/Datasets/waymo/train"
        self.waymo_path_test="RL/localUserData/Datasets/waymo/val"
        self.waymo_path_viz="RL/localUserData/Datasets/waymo-viz/"

        if not os.path.exists(self.path_masks):
            self.target_episodesCache_Path = "Datasets/CachedEpisodes" if USE_LOCAL_PATHS == 0 else "RL/localUserData/CachedEpisodes"


            ## This is the structure of the files that you are receiving. Change here.
            self.evaluation_path = ("Results/statistics/val/" if USE_LOCAL_PATHS == 0 else "RL/localUserData/Results/statistics/val/"+self.name_movie+"_"+self.timestamp)
            self.cityscapes_path ="Datasets/colmap/" if USE_LOCAL_PATHS == 0 else "RL/localUserData/Datasets/colmap/"

            self.camera_path=os.path.join(self.cityscapes_path,"cameras/")
            self.path_masks = os.path.join(self.cityscapes_path,"pannet/")
            self.colmap_path = os.path.join(self.cityscapes_path,"colmap/")

            self.carla_main ="Datasets/carla-sync/" if USE_LOCAL_PATHS == 0 else "RL/localUserData/Datasets/carla-sync/"
            self.name_movie_eval = os.path.join(self.carla_main,"eval/")
            self.carla_path = os.path.join(self.carla_main,"train/")
            self.carla_path_test = os.path.join(self.carla_main,"val/")

            self.carla_path_viz = "Datasets/carla-viz/" if USE_LOCAL_PATHS == 0 else "RL/localUserData/Datasets/carla-viz/" # Not needed to begin with.

            self.waymo_path = "Datasets/WAYMOOUTPUTMIN/"
        if self.new_carla:
            # Path to CARLA dataset
            self.carla_main = "Datasets/carla-testing" if USE_LOCAL_PATHS == 0 else "RL/localUserData/Datasets/carla-new"

            self.carla_path = os.path.join(self.carla_main,
                                           "Town03/")  # "Packages/CARLA_0.8.2/unrealistic_dataset/"
            print (" Carla path in settings "+str(self.carla_path))
            # Path to CARLA test set
            self.carla_path_test = os.path.join(self.carla_main, "Town03/")
            # Path to CARLA visualization set
            self.carla_path_viz = os.path.join(self.carla_main, "Town03/")
        if self.realtime_carla:

            self.carla_main_realtime = "Datasets/carla-realtime/"  if USE_LOCAL_PATHS == 0 else "RL/localUserData/Datasets/carla-realtime/"

            self.name_movie_eval_realtime = os.path.join(self.carla_main_realtime, "eval/")
            self.carla_path_realtime = os.path.join(self.carla_main_realtime, "train/")
            self.carla_path_test_realtime = os.path.join(self.carla_main_realtime, "val/")
            self.realtime_freq=90
        if self.realtime_carla_only:
            self.realtime_carla=False

            self.carla_main = "Datasets/carla-realtime_2/carla-realtime/" if USE_LOCAL_PATHS == 0 else "RL/localUserData/Datasets/carla-realtime/"

            self.carla_path_viz = os.path.join(self.carla_main, "viz/")
            self.name_movie_eval = os.path.join(self.carla_main, "eval/")
            self.carla_path= os.path.join(self.carla_main, "train/")
            self.carla_path_test= os.path.join(self.carla_main, "val/")
            self.realtime_freq = 0
            self.VAL_JUMP_STEP_CARLA = 1#4  # How many to skip to do faster evaluation
            self.VAL_JUMP_STEP_WAYMO = 1

        self.mode_name = "train"  # mode in cityscapes data

        if len(model)>0:
            self.load_weights = os.path.join(self.img_dir,model)  # "Results/agent/run_agent__carlalearn_initializer_multiplicative_2020-12-29-20-58-38.182501/"
            print(" Load "+str(self.load_weights ))
        else:
            self.load_weights =""

        if len(car_model)>0:
            self.load_weights_car = os.path.join(self.img_dir,car_model)  # "Results/agent/run_agent__carlalearn_initializer_multiplicative_2020-12-29-20-58-38.182501/"
            print(" Load " + str(self.load_weights))
        else:
            self.load_weights_car = ""

        # ---------------------------------------------------  Simulation specific parameters------------------------
        # --- Agent, car and environment size----
        # Size of agent-not used currently.
        self.height_agent = 4
        self.width_agent = 2
        self.depth_agent = 2
        self.agent_shape=[self.height_agent, self.width_agent,self.depth_agent]

        # How many voxels beyond oneself can the agent see.
        self.height_agent_s = 1
        self.width_agent_s = 10
        self.depth_agent_s = 10
        if self.fastTrainEvalDebug:
            self.width_agent_s = 10
            self.depth_agent_s = 10
        self.agent_shape_s = [self.height_agent_s,  self.width_agent_s,self.depth_agent_s]
        self.net_size=[1+2*(self.height_agent_s+self.height_agent), 1+2*(self.depth_agent_s+self.depth_agent), 1+2*(self.width_agent_s+self.width_agent)]
        self.agent_s = [self.height_agent_s + self.height_agent, self.depth_agent_s + self.depth_agent,
                        self.width_agent_s + self.width_agent]

        # Size of car agent in voxels
        self.car_dim = [self.height_agent_s, 4, 7]

        # Scaling of environment:
        self.scale_y = 5
        self.scale_x = 5
        self.scale_z = 5

        # Size of environment:
        self.height = env_height
        self.width = env_width
        self.depth = env_depth
        self.env_shape = [self.height, self.width, self.depth]

        # --- Sequence length, testing, network update and visualization frequencies----

        # Sequence length
        self.seq_len_train =30*self.action_freq#10*self.action_freq
        self.seq_len_train_final =30*self.action_freq#100
        self.seq_len_evaluate = 100#450#0#*self.action_freq#600#300#300
        self.seq_len_test =30*self.action_freq
        self.seq_len_test_final = 30*self.action_freq#450#100#*self.action_freq#200
        if (self.pfnn or self.goal_dir) and not self.useRLToyCar and not self.learn_init:
            self.seq_len_train = 10*self.action_freq  # *self.action_freq#10*self.action_freq
            self.seq_len_train_final = 30*self.action_freq  # *self.action_freq#100
            self.seq_len_test = 10*self.action_freq  # 30#10#*self.action_freq
            self.seq_len_test_final = 30*self.action_freq  # 30#400#100*self.action_freq#200
        self.threshold_dist =100#231#100#*self.action_freq # Longest distance to goal
        if  (not self.pfnn and self.goal_dir) and not self.useRLToyCar and not self.learn_init:
            self.threshold_dist =self.seq_len_train
        train_only_initializer=self.train_car_and_initialize == "according_to_stats" and self.initializer_collision_rate_threshold==1.0
        if self.learn_init and (not self.useRLToyCar or train_only_initializer):
            self.seq_len_train = 100#450
            self.seq_len_train_final = 100#450
            self.seq_len_evaluate = 100#450
            self.seq_len_test = 100#450
            self.seq_len_test_final = 100#450

        # Update network gradients with this frequency.
        self.update_frequency =30
        self.update_frequency_test=10#10

        # visualization frequency
        self.vizualization_freq=50
        self.vizualization_freq_test =15#50
        self.vizualization_freq_car=5#1

        # Test with this frequency.1
        self.test_freq =100#100
        if self.waymo:
            self.test_freq = 20
        if self.realtime_carla_only:
            self.test_freq = 15
        self.car_test_frq=50 # test on toy car environemnt with this frequency

        # Save with this frequency.
        self.save_freq = 100  # 200  # 50

        # Evaluation specific frequencies
        if evaluate:
            if self.likelihood:
                if self.waymo:
                    self.seq_len_evaluate = 200
                else:
                    self.seq_len_evaluate =450
            else:
                self.seq_len_evaluate = 300
                self.threshold_dist = 155# 31 m or average speed 1.75m/s

        # Override some values to get fast feedback
        if self.fastTrainEvalDebug:
            self.update_frequency = 2
            self.update_frequency_test=2#10

            self.vizualization_freq=2
            self.vizualization_freq_test =1#50
            self.vizualization_freq_car=1#1

            self.test_freq = 2#100#1
            if self.waymo:
                self.test_freq = 1#20
            #self.test_frequency=2#3
            self.car_test_frq=1
            self.seq_len_train = 10 * self.action_freq  # *self.action_freq#10*self.action_freq
            self.seq_len_train_final = 30 * self.action_freq  # *self.action_freq#100
            self.seq_len_test = 10 * self.action_freq  # 30#10#*self.action_freq
            self.seq_len_test_final = 30 * self.action_freq  # 30#400#100*self.action_freq#200

        # --- Initializations ----
        # Which initializations to use. Note overriden in environment class.
        self.train_nbrs = [6, 7, 8, 9]  # , 0,2,4,5] #[1,0,4,2,5,3,6,8,9,7]#
        self.test_nbrs =[6,8,9,7] #[3, 0, 2, 4, 5]
        self.widths = [7]#[2, 4, 6, 7, 10] # road width in toy class
        self.repeat = len(self.widths) * len(self.train_nbrs) * 50

        # POP optimization specififc parameters
        if self.pop:
            self.normalize = True
            self.fully_connected_layer = 128
            self.estimate_advantages=True
            self.vf_par=0.5
            self.entr_par=0
        # Initialize on pavement?
        self.init_on_pavement = 0

        self.sem_class = []

        # mapping between cityscapes and carla semantic labels.
        self.sem_dict = {11: 0, 13: 1, 14: 2, 4: 2, 5: 2, 23: 2, 15: 2, 16: 2, 17: 3, 18: 3, 7: 4, 9: 4, 6: 4,
                         10: 4, 8: 5,
                         21: 6, 22: 6, 12: 7, 20: 8, 19: 8}

    def getFrameRateAndTime(self):
        frame_rate = None
        if self.waymo:
            frame_rate = 10.0
        else:
            frame_rate = 17.0
        frame_time = 1 / frame_rate
        return int(frame_rate), frame_time

    # self.speed_input = False
    # self.end_on_bit_by_pedestrians = False

    # returns: the address of the external model, the script name it needs to invoke and the path to the model used
    def getExternalModelDeploymentAddressAndPaths(self):
        # Hardcoded values with correct things from the other side
        if self.evaluateExternalModel == None:
            return ""
        elif self.evaluateExternalModel == "STGCNN":
            if self.carla:
                return "http://localhost:5000/predict", STGCNN_SCRIPTNAME, STGCNN_MODELPATH_CARLA
            elif self.waymo:
                return "http://localhost:5001/predict", STGCNN_SCRIPTNAME, STGCNN_MODELPATH_WAYMO
            else:
                raise NotImplementedError
        elif self.evaluateExternalModel == "SGAN":
            if self.carla:
                return "http://localhost:5100/predict", SGAN_SCRIPTNAME, SGAN_MODELPATH_CARLA
            elif self.waymo:
                return "http://localhost:5101/predict", SGAN_SCRIPTNAME, SGAN_MODELPATH_WAYMO
            else:
                raise NotImplementedError
        elif self.evaluateExternalModel == "STGAT":
            if self.carla:
                return "http://localhost:5200/predict", STGAT_SCRIPTNAME, STGAT_MODELPATH_CARLA
            elif self.waymo:
                return "http://localhost:5201/predict", STGAT_SCRIPTNAME, STGAT_MODELPATH_WAYMO
            else:
                raise NotImplementedError
        else:
            raise  NotImplementedError

    # Starts an external agent (flask deployed) according to the settings set inside
    def startExternalAgent(self):
        assert self.lastAgentExternalProcess == None, "You didn't cleanup the last external agent used !!"

        def getPortFromAddress(addr):
            startIdx = addr.rfind(":")
            endIdx = addr.rfind("/")
            return addr[startIdx+1 : endIdx]

        address, scriptName, pathToModel = self.getExternalModelDeploymentAddressAndPaths()

        # Do we must start something externally really ?
        if address != "":
            # Kill the port at the address first
            portUsed = getPortFromAddress(address)
            #cmd_line = "lsof -ti:" + str(getPortFromAddress(address)) + " | xargs kill -9"
            #args = shlex.split(cmd_line)
            #subprocess.Popen(args)
            p1 = subprocess.Popen(["lsof", "-ti:"+str(portUsed)], stdout=subprocess.PIPE)
            p2 = subprocess.Popen(["xargs", "kill", "-9"], stdin=p1.stdout, stdout=subprocess.PIPE)
            p1.stdout.close()
            p2.communicate()

            # Run the corresponding process
            cmd_line = "python " + scriptName + " --model_path " + pathToModel + " --external 1"
            args = shlex.split(cmd_line)

            #cmds = ['python', 'Work/Social-STGCNN/test_stgcnn.py', '--model_path', 'Work/Social-STGCNN/checkpoint/custom_carla', '--external', '1']
            #subprocess.Popen(cmds)

            #print(args)
            self.lastAgentExternalProcess = subprocess.Popen(args)
            time.sleep(7) # Wait for server to start
            print(("Process opened: ", self.lastAgentExternalProcess))

    # Clean previous external agent process
    def cleanExternalAgent(self):
        if self.lastAgentExternalProcess != None:
            pass

            self.lastAgentExternalProcess = None


    def save_settings(self, filepath):
        self.path_settings_file = os.path.join(self.path_settings_file, self.name_movie)
        if not os.path.exists(self.path_settings_file):
            os.makedirs(self.path_settings_file)
        with  open(os.path.join(self.path_settings_file,self.timestamp+"_"+self.name_movie+"_settings.txt"), 'w+') as file:
            file.write("Name of frames: " + str(self.name_movie) + "\n")
            file.write("Timestamp: "+str(self.timestamp)+ "\n")
            file.write("Model path: " + str(self.model_path) + "\n")
            file.write("Overfit: " + str(self.overfit) + "\n")
            #file.write("Only people histogram reward: " + str(self.heatmap) + "\n")
            file.write("Curriculum goal: " + str(self.curriculum_goal) + "\n")
            file.write("Controller length: " + str(self.controller_len) + "\n")
            file.write("Rewards: " + str(self.reward_weights) + "\n")
            file.write("Pop: " + str(self.pop) + "\n")
            file.write("Confined actions : " + str(self.confined_actions) + "\n")

            file.write("Velocity : " + str(self.velocity) + "\n")
            file.write("Velocity sigmoid : " + str(self.velocity_sigmoid) + "\n")
            file.write("Velocity std: " + str(self.sigma_vel) + "\n")
            file.write("LSTM: " + str(self.lstm) + "\n")
            file.write("PFNN: " + str(self.pfnn) + "\n")
            file.write("Pose: " + str(self.pose) + "\n")
            file.write("Continous: " + str(self.continous) + "\n")
            file.write("Acceleration : " + str(self.acceleration) + "\n")


            file.write("reachin goal : " + str(self.goal_dir) + "\n")
            file.write("Action frequency : " + str(self.action_freq) + "\n")
            file.write("Semantic goal : " + str(self.sem_class) + "\n")
            file.write("Number of measures: " + str(NBR_MEASURES) + "\n")
            file.write("Number of measures car : " + str(NBR_MEASURES_CAR) + "\n")
            file.write("Toy case : " + str(self.toy_case) + "\n")
            file.write("CARLA : " + str(self.carla) + "\n")
            file.write("Temporal case : " + str(self.temp_case) + "\n")
            file.write("Temporal model : " + str(self.temporal) + "\n")
            file.write("Temporal prediction: " + str(self.predict_future) + "\n")
            file.write("Old conv modelling: " + str(self.old) + "\n")
            file.write("Old lstm modelling: " + str(self.old_lstm) + "\n")
            file.write("Fully connected size : " + str(self.fc) + "\n")
            file.write("Switch goal after #test steps : " + str(self.avoiding_people_step) + "\n")
            file.write("Penalty cars: "+str(self.reward_weights[0]) + "\n")
            file.write("Reward people: " + str(self.reward_weights[1]) + "\n")
            file.write("Reward pavement: " + str(self.reward_weights[2]) + "\n")
            file.write("Penalty objects: " + str(self.reward_weights[3]) + "\n")
            file.write("Reward distance travelled: " + str(self.reward_weights[4]) + "\n")
            file.write("Penalty out of axis: " + str(self.reward_weights[5]) + "\n")
            file.write("Reward distance left: " + str(self.reward_weights[6]) + "\n")
            file.write("Initialize on pavement: " + str(self.init_on_pavement) + "\n")
            file.write("Random seed of numpy: " + str(self.random_seed_np) + "\n")
            file.write("Random seed of tensorflow: " + str(self.random_seed_tf) + "\n")
            file.write("Timing: " + str(self.timing) + "\n")
            file.write("2D input to network: " + str(self.run_2D) + "\n")
            file.write("Semantic channels separated: " + str(self.sem) + "\n")
            file.write("Minimal semantic channels : " + str(self.min_seg) + "\n")
            file.write("People channel: " + str(self.people) + "\n")
            file.write("Cars channel: " + str(self.cars) + "\n")
            file.write("Car variable: " + str(self.car_var) + "\n")
            file.write("Network memory: " + str(self.mem) + "\n")
            file.write("Network action memory: " + str(self.action_mem) + "\n")
            if self.mem:
                file.write("Number of timesteps: " + str(self.nbr_timesteps) + "\n")
            file.write("Load weights from file: " + self.load_weights + "\n")
            file.write("Load car weights from file: " + self.load_weights_car + "\n")
            file.write("Learning rate: " + str(self.lr) + "\n")
            file.write("Batch size: " + str(self.batch_size) + "\n")
            file.write("Agents shape: " + str(self.agent_shape) + "\n")
            file.write("Agents sight: " + str(self.agent_shape_s) + "\n")
            file.write("Environment shape: " + str(self.env_shape) + "\n")
            file.write("Controller length: " + str(self.controller_len) + "\n")
            file.write("Discount rate: " + str(self.gamma) + "\n")
            file.write("Goal distance: " + str(self.threshold_dist) + "\n")
            file.write("Sequence length train: " + str(self.seq_len_train) + "\n")
            file.write("Sequence length test: " + str(self.seq_len_test) + "\n")
            file.write("Sequence length train final: " + str(self.seq_len_train_final) + "\n")
            file.write("Sequence length test final: " + str(self.seq_len_test_final) + "\n")
            file.write("Weights update frequency: " + str(self.update_frequency) + "\n")
            file.write("Network saved with frequency: " + str(self.save_freq) + "\n")
            file.write("Test frequency: " + str(self.test_freq) + "\n")
            file.write("Deterministic: " + str(self.deterministic) + "\n")
            file.write("Deterministic testing: " + str(self.deterministic_test) + "\n")
            # if not self.deterministic:
            #     file.write("Probability: " + str(self.prob) + "\n")
            file.write("Curriculum learning reward: " + str(self.curriculum_reward) + "\n")
            file.write("Curriculum seq_len reward: " + str(self.curriculum_seq_len) + "\n")
            file.write("Training environments: " + str(self.train_nbrs) + "\n")
            file.write("Testing environments: " + str(self.test_nbrs) + "\n")
            file.write("Repeat for number of iterations: " + str(self.repeat) + "\n")
            file.write("Network pooling: " + str(self.pooling) + "\n")
            file.write("Normalize batches: " + str(self.batch_normalize) + "\n")
            file.write("Clip gradients: " + str(self.clip_gradients) + "\n")
            file.write("Regularize loss: " + str(self.regularize) + "\n")
            file.write("Allow 3D actions: " + str(self.actions_3d) + "\n")
            file.write("Normalize reward: " + str(self.normalize) + "\n")
            file.write("Number of layers in network: " + str(self.num_layers) + "\n")
            file.write("Number of out filters: " + str(self.outfilters) + "\n")
            file.write("Learn initialization:  " + str(self.learn_init) + "\n")
            file.write("Gaussian init net: " + str(self.gaussian_init_net) + "\n")
            file.write("Std of init net : " + str(self.init_std) + "\n")

            file.write("Train RL toy car : " + str(self.useRLToyCar) + "\n")
            file.write("Train CARLA env live : " + str(self.useRealTimeEnv) + "\n")
            file.write("Car input : " + str(self.car_input) + "\n")

            file.write("Learn initialization : "+str(self.learn_init)+ "\n")
            file.write("Learn goal : " + str(self.learn_goal) + "\n")
            file.write("Learn time : " + str(self.learn_time) + "\n")
            file.write("Fast debugging training : " + str(self.fastTrainEvalDebug) + "\n")
            file.write("Use Cahcing : " + str(self.useCaching) + "\n")


            #file.write("Initialization accv architecture : " + str(self.accv_arch_init) + "\n")
            file.write("Initialization semantic architecture : " + str(self.sem_arch_init) + "\n")
            file.write("Separate Goal Net : " + str(self.separate_goal_net) + "\n")
            file.write("Gaussian Initializer : " + str(self.gaussian_init_net) + "\n")
            file.write("Gaussian Goal Net : " + str(self.goal_gaussian) + "\n")
            file.write("Initializer std : " + str(self.init_std) + "\n")
            file.write("Goal std : " + str(self.goal_std) + "\n")
            file.write("Multiplicative reward : " + str(self.multiplicative_reward) + "\n")
            file.write("Use occlusion : " + str(self.use_occlusion) + "\n")
            file.write("Entropy parameter agent : " + str(self.entr_par) + "\n")
            file.write("Entropy parameter initializer : " + str(self.entr_par_init) + "\n")
            file.write("Entropy parameter goal : " + str(self.entr_par_goal) + "\n")

            file.write("Add noise to car input : " + str(self.add_noise_to_car_input) + "\n")
            file.write("Car noise probability : " + str(self.car_noise_probability) + "\n")
            file.write("Pedestrian view occluded : " + str(self.pedestrian_view_occluded) + "\n")
            file.write("Toy Car input : " + str(self.car_input) + "\n")
            file.write("Toy Car linear : " + str(self.linear_car) + "\n")
            file.write("Toy Car linear std : " + str(self.sigma_car) + "\n")
            file.write("Car reference speed : " + str(self.car_reference_speed) + "\n")

            file.write("New carla training set  : " + str(self.new_carla) + "\n")
            file.write("Train car and initializer : " + self.train_car_and_initialize + "\n")
            file.write("Car success rate threshold : " + str(self.car_success_rate_threshold) + "\n")
            file.write("Initializer collision rate theshold : " + str(self.initializer_collision_rate_threshold) + "\n")
            file.write("Number of iteration to train initializer : " + str(self.num_init_epochs) + "\n")
            file.write("Number of iterations to train car : " + str(self.num_car_epochs) + "\n")
            file.write("Car reward  : "+str(self.reward_weights_car)  + "\n")

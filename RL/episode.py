
import numpy as np
from extract_tensor import frame_reconstruction,reconstruct_static_objects
import os
import random, math

from utils.utils_functions import overlap, find_border

from settings import DEFAULT_SKIP_RATE_ON_EVALUATION_CARLA, DEFAULT_SKIP_RATE_ON_EVALUATION_WAYMO, NUM_SEM_CLASSES, PEDESTRIAN_MEASURES_INDX, CAR_MEASURES_INDX,PEDESTRIAN_INITIALIZATION_CODE,PEDESTRIAN_REWARD_INDX, NBR_MEASURES, NBR_MEASURES_CAR,STATISTICS_INDX, STATISTICS_INDX_POSE, STATISTICS_INDX_CAR,STATISTICS_INDX_CAR_INIT, STATISTICS_INDX_MAP,CHANNELS
import copy
import time
import pickle
from commonUtils.systemUtils import deep_getsizeof
from commonUtils.ReconstructionUtils import ROAD_LABELS,SIDEWALK_LABELS, OBSTACLE_LABELS_NEW, OBSTACLE_LABELS, MOVING_OBSTACLE_LABELS,cityscapes_labels_dict
from scipy import ndimage

from settings import RANDOM_SEED
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def collide_with_people( frame_input, people_overlapped, x_range, y_range, z_range, people, agent_bbox=[]):
    # print ("Car measure:  People hit by car frame" + str(frame_input)+" bbox "+str([y_range[0], y_range[1], z_range[0], z_range[1]]))
    # print ("People in frame "+str(people[frame_input]))

    for person in people[frame_input]:
        x_pers = [min(person[1, :]), max(person[1, :]), min(person[1, :]), max(person[1, :]), min(person[2, :]),
                  max(person[2, :])]
        if overlap(x_pers[2:], [y_range[0], y_range[1], z_range[0], z_range[1]], 1):
            people_overlapped.append(person)
            return people_overlapped
    if len(agent_bbox)>0:
        # print("Agent "+str(np.mean(agent_bbox[:2]))+"  "+str(np.mean(agent_bbox[2:4])))
        # print ("Overlap agent ? " + str(agent_bbox) + " and car " + str([y_range[0], y_range[1], z_range[0], z_range[1]]))
        if overlap(agent_bbox[2:], [y_range[0], y_range[1], z_range[0], z_range[1]], 1):
            # print (" overlapped")
            people_overlapped.append([y_range[0], y_range[1], z_range[0], z_range[1]])
            return people_overlapped
        # print(" did not overlap")
    return people_overlapped



def intercept_hero_car( frame_in, cars, bbox, all_frames=False, agent_frame=-1):
    # can be replaced by reurning the content of hero_car_measures
    # Return number of cars overlapped

    overlapped = 0
    frames = []
    if all_frames or frame_in >= len(cars):
        frames = list(range(len(cars)))
    else:
        frames.append(frame_in)


    for frame in frames:
        # print (" Cars "+str(cars[frame]))
        for car in cars[frame]:

            if overlap(car[2:6], bbox[2:], 1) or overlap(bbox[2:], car[2:6], 1):

                overlapped += 1

    return overlapped

def end_of_episode_measures_car(frame, measures, cars, people, bbox, reconstruction,agent_bbox, agent_pos, end_on_hit_by_pedestrians, goal, allow_car_to_live_through_collisions,carla_new=False):
    people_overlapped=[]
    per_id = -1
    car=[]
    x_range = bbox[0:2]
    y_range = bbox[2:4]
    z_range = bbox[4:]
    car_pos=np.array([np.mean(x_range),np.mean(y_range), np.mean(z_range)])


    # print (" End of episode measures in car: frame "+str(frame )+" car pos "+str(bbox)+ " evaluate "+str(frame+1))

    # Hit by car

    measures[frame, CAR_MEASURES_INDX.hit_by_car] = max(measures[max(frame - 1, 0), CAR_MEASURES_INDX.hit_by_car], intercept_hero_car(frame + 1, all_frames=False, cars=cars,  bbox=bbox))

    if not allow_car_to_live_through_collisions:
        measures[frame, CAR_MEASURES_INDX.agent_dead] = max(measures[frame, CAR_MEASURES_INDX.hit_by_car],
                                                            measures[frame, CAR_MEASURES_INDX.agent_dead])

    # print (" Car hit by cars "+str(measures[frame, CAR_MEASURES_INDX.hit_by_car]))
    # hit by pedestrian
    # next frame not aviable yet. Should be evaluated externally.
    collide_pedestrians = len(collide_with_people(frame+1 ,people_overlapped, x_range, y_range, z_range, people,agent_bbox )) > 0

    if overlap(agent_bbox[2:], [y_range[0], y_range[1], z_range[0], z_range[1]], 1):
        measures[frame, CAR_MEASURES_INDX.hit_by_agent] =1
        if allow_car_to_live_through_collisions:
            measures[frame, CAR_MEASURES_INDX.agent_dead] =1
    if collide_pedestrians:
        measures[frame, CAR_MEASURES_INDX.hit_pedestrians] = 1

    # print (" Car hit by pedestrians " + str(measures[frame, CAR_MEASURES_INDX.hit_pedestrians]))
    if end_on_hit_by_pedestrians:
        measures[frame, CAR_MEASURES_INDX.hit_pedestrians]=max(measures[max(frame - 1, 0), CAR_MEASURES_INDX.hit_pedestrians],measures[frame, CAR_MEASURES_INDX.hit_pedestrians])
        if not allow_car_to_live_through_collisions:
            measures[frame, CAR_MEASURES_INDX.agent_dead] = max(measures[frame, CAR_MEASURES_INDX.hit_pedestrians],measures[frame, CAR_MEASURES_INDX.agent_dead] )


    measures[frame, CAR_MEASURES_INDX.hit_obstacles] = max(intercept_objects_cars(bbox,reconstruction, no_height=True,carla_new=carla_new),measures[frame, CAR_MEASURES_INDX.hit_obstacles])
    if not allow_car_to_live_through_collisions:
        if measures[frame, CAR_MEASURES_INDX.hit_obstacles]>0:
            measures[frame, CAR_MEASURES_INDX.agent_dead] = max(1,measures[frame, CAR_MEASURES_INDX.agent_dead])



    if len(goal)>0:  # If using a general goal

        # Distance to goal.
        measures[frame, CAR_MEASURES_INDX.dist_to_goal] = np.linalg.norm(np.array(goal[1:]) - car_pos[1:])

        # print ("Distance to goal car: " + str(measures[frame, CAR_MEASURES_INDX.dist_to_goal]) + " goal pos: " + str(goal[1:]) + "current pos: " + str(car_pos[1:]))

        # Reached goal?
        if y_range[0]<=goal[1] and y_range[1]>=goal[1] and z_range[0]<=goal[2] and z_range[1]>=goal[2]:
            measures[frame,CAR_MEASURES_INDX.goal_reached]=1
        # print("Goal reached ? goal "+str(goal)+" car "+str(bbox))
    measures[frame, CAR_MEASURES_INDX.dist_to_agent]= np.linalg.norm(car_pos[1:]-agent_pos[1:])
    #print(" Agent distance to car "+str(measures[frame, CAR_MEASURES_INDX.dist_to_agent])+" agent "+str(agent_pos[1:])+" car "+str(car_pos[1:])+" diff "+str(car_pos[1:]-agent_pos[1:])+" frame "+str(frame))
    return measures



def car_bounding_box(bbox, reconstruction, no_height=False, channel=3):

    if no_height:
        x_range = [0, reconstruction.shape[0]]
    else:
        x_range = [bbox[0], bbox[1]]
    # print (" Car bbox " + str(bbox)+" x range "+str(x_range))
    segmentation = (
        reconstruction[int(x_range[0]):int(x_range[1]) + 1, int(bbox[2]):int(bbox[3]) + 1,
        int(bbox[4]):int(bbox[5]) + 1, channel] * int(NUM_SEM_CLASSES)).astype(
        np.int32)
    # print (" Car bbox " + str(segmentation.shape))
    return segmentation

def intercept_objects_cars( bbox,reconstruction, no_height=False, cars=False,carla_new=False):  # Normalisera/ binar!
    # print (" Intercept objects ----------------------------------------------------------"+str(bbox))
    segmentation = car_bounding_box(bbox, reconstruction,no_height)

    # print (" Intercept objects bbox shape " + str(segmentation.shape)+" sum "+str(np.unique(segmentation)))
    if carla_new:
        obstacles=OBSTACLE_LABELS_NEW
    else:
        obstacles = OBSTACLE_LABELS
    #print (" Obstacle class "+str(obstacles))
    count=0
    for label in obstacles:
        count+=(segmentation==label).sum()
    # print(" Hit objects " + str(count) + " return " + str(
    #     count * 1.0 / ((bbox[1] + 2 - bbox[0]) * (bbox[2] + 2 - bbox[3]))))
    if cars:
        for label in MOVING_OBSTACLE_LABELS:
            count += (segmentation == label).sum()
        count += (segmentation == cityscapes_labels_dict['person'] ).sum()
    #print(" Moving obstacle class " + str(MOVING_OBSTACLE_LABELS))
        #count += (segmentation > 25).sum()
    # print(" Hit objects add cars " + str(count) + " return " + str(
    #     count * 1.0 / ((bbox[1] + 2 - bbox[0]) * (bbox[2] + 2 - bbox[3]))))
    if count > 0:
        # print(" Hit objects final" + str(count) + " return " + str(
        #     count * 1.0 / ((bbox[1] + 2 - bbox[0]) * (bbox[2] + 2 - bbox[3]))))
        if (bbox[1]+2-bbox[0])*(bbox[3]+2-bbox[2])>0:

            return count *1.0/((bbox[1]+2-bbox[0])*(bbox[3]+2-bbox[2]))#*(z_range[1]+2-z_range[0]))
        else:
            return 0
    return 0

def find_closest_car_in_list( frame, pos, cars):
    # print("Find  closest car in list "+str(pos)+" frame "+str(min(frame, len(cars) - 1))+" in  "+str(cars[min(frame, len(cars) - 1)]))
    # print (" Cars "+str(cars))
    closest_car = []
    min_dist = -1
    for car in cars[min(frame, len(cars) - 1)]:
        car_pos = np.array([np.mean(car[0:2]), np.mean(car[2:4]), np.mean(car[4:])])

        if min_dist < 0 or min_dist > np.linalg.norm(np.array(pos[1:]) - car_pos[1:]):
            closest_car = car_pos
            min_dist = np.linalg.norm(np.array(pos[1:]) - car_pos[1:])
    # print (" Closest car "+str(closest_car )+" min dist "+str( min_dist))
    return closest_car, min_dist

def find_closest_pedestrian_in_list( frame, pos, people):
    closest_pedestrian = []
    min_dist = -1
    for person in people[min(frame, len(people) - 1)]:
        person_pos = np.mean(person, axis=1)

        if min_dist < 0 or min_dist > np.linalg.norm(np.array(pos[1:]) - person_pos[1:]):
            closest_pedestrian = person_pos
            min_dist = np.linalg.norm(np.array(pos[1:]) - person_pos[1:])

    return closest_pedestrian, min_dist

def iou_sidewalk_car( bbox, reconstruction, no_height=True):
    segmentation = car_bounding_box(bbox, reconstruction,no_height)
    area =0
    for label in SIDEWALK_LABELS:
        area+=(segmentation == label).sum()
    divisor=(bbox[1]-bbox[0])*(bbox[3]-bbox[2])*(bbox[5]-bbox[4])

    return area * 1.0 / divisor

class SimpleEpisode(object):
    def __init__(self, tensor, people_e, cars_e, pos_x, pos_y, gamma, seq_len, reward_weights, agent_size=(2, 2, 2),
                 people_dict={}, cars_dict={}, init_frames={}, adjust_first_frame=False,
                 follow_goal=False, action_reorder=False, threshold_dist=-1, init_frames_cars={}, temporal=False,
                 predict_future=False, run_2D=False, agent_init_velocity=False, velocity_actions=False, seq_len_pfnn=0,
                 end_collide_ped=False, stop_on_goal=True, waymo=False, defaultSettings=None, agent_height=4,
                 multiplicative_reward=False, learn_goal=False, use_occlusion=False,
                 heroCarDetails=None, useRealTimeEnv=None,  car_vel_dict=None, people_vel_dict=None, car_dim=None, new_carla=False, lidar_occlusion=False):


        # Environment variables
        self.DTYPE = np.float64
        self.pos = [pos_x, pos_y]
        self.cars = cars_e
        self.people = people_e
        self.reconstruction = tensor
        self.init_frames = init_frames
        self.init_frames_car = init_frames_cars
        self.pavement = []#[6, 7, 8, 9, 10]
        for label in SIDEWALK_LABELS:
            self.pavement.append(label)
        for label in ROAD_LABELS:
            self.pavement.append(label)
        print("Pavement class "+str(self.pavement))

        self.valid_people = []  # valid people positions, without keys
        if len(tensor) > 0:
            self.valid_positions = np.zeros(self.reconstruction.shape[1:3])
        else:
            self.valid_positions = []
        self.people_dict = people_dict
        self.key_map = []
        if self.people_dict:
            for counter, val in enumerate(self.people_dict.keys()):
                self.key_map.append(val)
                # print self.key_map
        self.cars_dict = cars_dict
        self.init_cars = []
        self.valid_keys = []  # people the central 75% of the scene, present in the first frame
        self.valid_people_tracks = {}  # people in the central 75% of the scene
        self.reconstruction_2D = []
        self.test_positions = []
        self.cars_predicted = []
        self.people_predicted = []
        self.heights = []
        self.heroCarDetails = heroCarDetails
        self.useRealTimeEnv = useRealTimeEnv
        if useRealTimeEnv:
            self.heroCarDetails={}

        # Variables for agent initialization -- constants
        self.speed_init = 0
        self.vel_init = np.zeros((1, 3), dtype=self.DTYPE)
        self.goal_time = -1
        self.actions = []
        v = [-1, 0, 1]
        j = 0
        for y in range(3):
            for x in range(3):
                self.actions.append([0, v[y], v[x]])
                j += 1
        self.actions_ordered = [4, 1, 0, 3, 6, 7, 8, 5, 2]  # [5, 6, 7, 8, 3, 2, 1, 0]

        # Run specific variables
        self.temporal = temporal
        self.predict_future = predict_future
        self.run_2D = run_2D
        self.seq_len = seq_len

        self.stop_on_goal = stop_on_goal
        self.follow_goal = follow_goal
        self.threshold_dist = threshold_dist # max distance to goal
        self.gamma = gamma
        self.agent_size = agent_size
        self.reward_weights = reward_weights  # cars, people, pavement, objs, dist, out_of_axis
        self.end_on_hit_by_pedestrians = end_collide_ped
        self.velocity_agent = False#agent_init_velocity
        self.past_move_len=2
        self.evaluation = False  # Set goal according to evaluation?
        self.first_frame = 0
        self.initOtherParameters(velocity_actions, defaultSettings,agent_height,multiplicative_reward, learn_goal,use_occlusion, car_vel_dict, people_vel_dict, car_dim,new_carla, lidar_occlusion)


        if action_reorder:
            self.actions =[self.actions[k] for k in [4,1,0,3,6,7,8,5,2]]#[8, 7, 6, 5, 0, 1, 2, 3, 4]]
            self.actions_ordered = [0,1,2,3,4,5,6,7,8]


        # Depends on run specific variables
        vector_len = max(seq_len - 1, 1)
        self.measures = np.zeros((vector_len, NBR_MEASURES), dtype=self.DTYPE)


        # Variables for gathering statistics/ agent movement
        self.agent = [[] for _ in range(seq_len)]
        self.agent[0] = np.zeros(1 * 3, dtype=self.DTYPE)
        self.turning_point = np.ones(seq_len)
        self.velocity = [[]] * vector_len
        self.velocity[0] = np.zeros(3, dtype=self.DTYPE)
        self.action = [None] * vector_len  # For softmax
        self.action[0] = [4]
        self.speed = [None] * vector_len
        self.angle = np.zeros(vector_len, dtype=self.DTYPE)
        self.reward = np.zeros(vector_len, dtype=self.DTYPE)
        self.reward_d = np.zeros(vector_len, dtype=self.DTYPE)
        self.init_method = 0
        self.accummulated_r = 0
        self.loss = np.zeros(vector_len, dtype=self.DTYPE)
        self.probabilities = np.zeros((vector_len, 27), dtype=self.DTYPE)
        self.goal_person_id = -1
        self.goal_person_id_val = ""
        self.goal = [0, 0, 0]

        # Pose variables- depend on if pfnn is used
        self.avg_speed = np.zeros((seq_len_pfnn + seq_len))
        self.agent_high_frq_pos = np.zeros((seq_len_pfnn+seq_len, 2))
        self.agent_pose_frames=np.zeros((seq_len_pfnn+seq_len),dtype=np.int)
        self.agent_pose = np.zeros((seq_len_pfnn + seq_len, 93))
        self.agent_pose_hidden = np.zeros((seq_len_pfnn + seq_len, 512))


        # INITIALIZATION
        if len(tensor)>0:
            self.initialization_set_up(seq_len, adjust_first_frame)

    def initOtherParameters(self, velocity_actions, defaultSettings, agent_height,multiplicative_reward, learn_goal,use_occlusion,  car_vel_dict, people_vel_dict, car_dim, new_carla, lidar_occlusion):
        self.frame_rate, self.frame_time = defaultSettings.getFrameRateAndTime()

        if velocity_actions:
            self.max_step = 3 / self.frame_rate * 5
        else:
            self.max_step = np.sqrt(2)

        print(("Maximum step " + str(self.max_step)))
        self.lidar_occlusion=lidar_occlusion
        self.prior=self.valid_positions.copy()
        self.goal_prior = self.valid_positions.copy()#*self.heatmap
        self.occluded = self.valid_positions.copy()
        self.agent_height = agent_height
        self.init_distribution=None
        self.goal_distribution=None
        self.init_car_id=-1
        self.init_car_vel=[]
        self.init_car_pos=[]
        self.multiplicative_reward = multiplicative_reward
        self.manual_goal=[]
        self.learn_goal=learn_goal
        self.use_occlusions=use_occlusion
        self.agent_prediction_people=[]
        self.new_carla=new_carla
        self.car_goal = []


        # Variables needed to keep track of car statistics. So we can have a joint overview.
        if  self.useRealTimeEnv:
            vector_len = max(self.seq_len , 1)
            self.measures_car = np.zeros((vector_len, NBR_MEASURES_CAR), dtype=self.DTYPE)
            self.car = [[] for _ in range(self.seq_len)]
            self.car[0]=np.zeros(1 * 3, dtype=self.DTYPE)
            self.velocity_car = [[]] * (vector_len)
            self.velocity_car[0] = np.zeros(3, dtype=self.DTYPE)
            self.speed_car=np.zeros(vector_len, dtype=self.DTYPE)
            self.action_car=np.zeros(vector_len, dtype=self.DTYPE)
            self.reward_car = np.zeros(vector_len, dtype=self.DTYPE)
            self.reward_car_d = np.zeros(vector_len, dtype=self.DTYPE)
            self.accummulated_r_car = 0
            self.loss_car = np.zeros(vector_len, dtype=self.DTYPE)
            self.probabilities_car = np.zeros((vector_len, 2), dtype=self.DTYPE)
            self.car_dir=np.zeros(3, dtype=self.DTYPE)
            self.people_vel_dict=[None]*self.seq_len
            self.car_vel_dict=[None]*self.seq_len
            self.car_vel_dict[0]=car_vel_dict
            self.people_vel_dict[0] = people_vel_dict
            self.car_dims=car_dim

            self.car_bbox=[None]*self.seq_len
            self.people_predicted_init=np.zeros_like(self.reconstruction_2D)
            self.cars_predicted_init = np.zeros_like(self.reconstruction_2D)
            self.car_angle=np.zeros(vector_len, dtype=self.DTYPE)
        else:
            try:
                self.cars_predicted=self.car_predicted
            except AttributeError:
                print ("car_predicted does not exist in episode")



    ####################################### Post initialization set up
    def set_correct_run_settings(self, run_2D, seq_len,  stop_on_goal,follow_goal,threshold_dist, gamma,
                                 reward_weights, end_collide_ped, agent_init_velocity, waymo,action_reorder,seq_len_pfnn,
                                 velocity_actions,agent_height, evaluation=False, defaultSettings = None,
                                 multiplicative_reward=False, learn_goal=False, use_occlusion=False, heroCarDetails=None,
                                 useRealTimeEnv=False,  car_vel_dict=None, people_vel_dict=None, car_dim=None, new_carla=False, lidar_occlusion=False):
        # print("Set values")
        # print(("Seq len in episode: " + str(self.seq_len) + " seq len " + str(seq_len)))
        # Run specific variables
        self.run_2D = run_2D
        # print(("RUn on 2D "+str(self.run_2D)))
        self.stop_on_goal = stop_on_goal
        # print(("Stop on goal " + str(self.stop_on_goal)))
        self.follow_goal = follow_goal
        # print(("Follow goal  " + str(self.follow_goal)))
        self.threshold_dist = threshold_dist  # max distance to goal
        # print(("Threshold dist   " + str(self.threshold_dist)))
        self.gamma = gamma
        # print(("Gamma   " + str(self.gamma)))
        self.reward_weights = reward_weights  # cars, people, pavement, objs, dist, out_of_axis
        # print(("Reward weights    " + str(self.reward_weights)))
        self.end_on_hit_by_pedestrians = end_collide_ped

        # print(("end_on_hit_by_pedestrians    " + str(self.end_on_hit_by_pedestrians)))
        self.velocity_agent = agent_init_velocity
        # print(("velocity_agent    " + str(self.velocity_agent)))

        self.past_move_len = 2
        self.evaluation = evaluation  # Set goal according to evaluation?
        # print(("evaluation    " + str(self.evaluation)))
        self.first_frame = 0

        self.VAL_JUMP_STEP = DEFAULT_SKIP_RATE_ON_EVALUATION_CARLA # How many to skip to do faster evaluation
        self.VAL_JUMP_STEP_WAYMO = DEFAULT_SKIP_RATE_ON_EVALUATION_WAYMO
        self.heroCarDetails = heroCarDetails
        self.useRealTimeEnv = useRealTimeEnv

        if useRealTimeEnv:
            self.heroCarDetails = {}
        self.initOtherParameters(velocity_actions, defaultSettings, agent_height, multiplicative_reward, learn_goal, use_occlusion,car_vel_dict, people_vel_dict,car_dim, new_carla,lidar_occlusion)

        # print(("Maximum step " + str(self.max_step)))
        if action_reorder:
            self.actions = [self.actions[k] for k in [4, 1, 0, 3, 6, 7, 8, 5, 2]]  # [8, 7, 6, 5, 0, 1, 2, 3, 4]]
            self.actions_ordered = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            # print(("Actions "+str(self.actions)))

        vector_len = max(seq_len - 1, 1)
        # print(("Seq len in episode: " + str(self.seq_len) + " seq len " + str(seq_len)))
        if self.seq_len!=seq_len:
            # print(("Seq len in episode: "+str(self.seq_len)+" seq len "+str(seq_len)))
            self.seq_len = seq_len

            for valid_key in self.valid_keys[:]:
                if self.init_frames[valid_key]> seq_len:
                    self.valid_keys.remove(valid_key)
                    # print(("Remove "+str(valid_key)+" frame init "+str(self.init_frames[valid_key])))

            for valid_key in self.init_cars[:]:
                if self.init_frames_car[valid_key] > seq_len:
                    self.init_cars.remove(valid_key)
                    # print(("Remove " + str(valid_key) + " frame init " + str(self.init_frames_car[valid_key])))

            for valid_key in list(self.valid_people_tracks.keys()):
                if self.init_frames[valid_key]> seq_len:
                    self.valid_people_tracks.pop(valid_key)
                    # print(("Remove " + str(valid_key) + " frame init " + str(self.init_frames[valid_key])))
            self.key_map = self.valid_keys
            # print(("Key map " + str(self.key_map)))
            # Variables for gathering statistics/ agent movement
            self.agent=self.agent[:seq_len]
            # print(("Agent len " + str(len(self.agent))))
            self.turning_point = self.turning_point[:seq_len]
            # print(("Turning point " + str(len(self.turning_point))))
            self.velocity = self.velocity[:vector_len]
            # print(("Velocity " + str(len(self.velocity))))
            self.action = self.action[:vector_len]
            # print(("Action " + str(len(self.action))))
            self.speed = self.speed[:vector_len]
            # print(("Speed " + str(len(self.speed))))
            self.angle = self.angle[:vector_len]
            # print(("Angle " + str(len(self.angle))))
            self.reward = self.reward[:vector_len]
            # print(("Reward " + str(len(self.reward))))
            self.reward_d = self.reward_d[:vector_len]
            # print(("Reward discounted " + str(len(self.reward_d))))
            self.loss = self.loss[:vector_len]
            # print(("Loss " + str(len(self.loss))))
            self.probabilities =  self.probabilities[:vector_len,:]
            # print(("Probabilities " + str(len(self.probabilities))))

        if seq_len_pfnn>0:
            # Pose variables- depend on if pfnn is used
            self.avg_speed = np.zeros((seq_len_pfnn + seq_len))
            # print(("Avg Speed " + str(len(self.avg_speed))))
            self.agent_high_frq_pos = np.zeros((seq_len_pfnn + seq_len, 2))
            # print(("High freq pos " + str(len(self.agent_high_frq_pos))))
            self.agent_pose_frames = np.zeros((seq_len_pfnn + seq_len), dtype=np.int)
            # print(("Agent pose frames " + str(len(self.agent_pose_frames))))
            self.agent_pose = np.zeros((seq_len_pfnn + seq_len, 93))
            # print(("Agent pose" + str(len(self.agent_pose))))
            self.agent_pose_hidden = np.zeros((seq_len_pfnn + seq_len, 512))
            # print(("Agent pose hidden" + str(len(self.agent_pose_hidden))))


        vector_len = max(seq_len - 1, 1)

        # print(("Nbr measures" + str(NBR_MEASURES)+" "+str(vector_len)))
        self.measures = np.zeros((vector_len, NBR_MEASURES), dtype=self.DTYPE)


    ######################################## Initializations


    # Set up for initialization
    # Q: save the postprocessed reconstruction, car_predicted, people_predicted either on disk or RAM as an option / memory budget
    # for each episode and have a fast reload option that takes data from cache and avoid calls to frame_reconstruction, predict_cars_and_people, get_heatmap
    def initialization_set_up(self,  seq_len, adjust_first_frame):

        # Correct frame when the episode is inirtialized
        if not self.useRealTimeEnv:
            self.correct_init_frame(adjust_first_frame, seq_len) # Only needed for saved data!

        # Get reconstruction and people and car prediction maps
        if self.useRealTimeEnv:
            self.reconstruction, self.reconstruction_2D=reconstruct_static_objects(self.reconstruction,  run_2D=self.run_2D)
            self.people_predicted=[]
            self.cars_predicted=[]
            for frame in range(self.seq_len):
                self.people_predicted.append([])
                self.cars_predicted.append([])
            self.people_predicted[0]=np.zeros(self.reconstruction_2D.shape)
            self.cars_predicted[0] = np.zeros(self.reconstruction_2D.shape)
        else:

            self.reconstruction, self.cars_predicted, self.people_predicted, self.reconstruction_2D = frame_reconstruction(
                self.reconstruction, self.cars, self.people, False, temporal=self.temporal,
                predict_future=self.predict_future, run_2D=self.run_2D)



        # Get valid pedestrian trajectory positions
        if not self.useRealTimeEnv:
            # Find valid people if no trajectories are provided.
            self.get_valid_positions(mark_out_people=False)
            self.get_valid_traj()
            self.mark_out_people_in_valid_pos()
        else:
            self.valid_people_tracks=self.people_dict
            self.valid_keys=list(self.people_dict.keys())
            self.init_cars=list(self.cars_dict.keys())

            # Find valid people if no trajectories are provided.
            self.get_valid_positions()

        # Predict car positions
        if not self.useRealTimeEnv:
            print (" Predict cars and people")
            self.predict_cars_and_people()

        # Find sidewalk
        self.valid_pavement = self.find_sidewalk(True)


        # Get pedestrian heatmap for reward calculation
        if not self.useRealTimeEnv:
            self.get_heatmap()


        # Save valid keys
        self.key_map=self.valid_keys

        # Find borders of valid areas to initialize agent there.
        self.border =[]
        self.border=find_border(self.valid_positions)
        self.prior = self.valid_positions.copy()
        self.goal_prior = self.valid_positions.copy()#*self.heatmap
        self.occluded=self.valid_positions.copy()



        # DEBUG CODE BEGIN
        """
        objectss= [self.reconstruction, self.cars_predicted, self.people_predicted, self.reconstruction_2D,
                   self.valid_pavement, self.heights, self.valid_positions, self.init_frames, self.init_frames_car,
                   self.people_dict, self.people, self.cars, self.cars_dict, self.valid_people, self.valid_people_tracks,
                   self.valid_pavement,
                   self.valid_keys,
                   self.heatmap]

        
        reconstructionObjects = {'reconstruction' : self.reconstruction,
                                 'reconstruction_2D' : self.reconstruction_2D,
                                 'people' : self.people,
                                 'cars' : self.cars,
                                 'people_dict' : self.people_dict,
                                 'cars_dict' : self.cars_dict}
        
        sz = deep_getsizeof(objectss, set())

        szOfReconstructionObjects = deep_getsizeof(reconstructionObjects, set())
        print("Size of reconstruction objects", szOfReconstructionObjects)

        with open("test.pkl", "wb") as testFile:
            serial_start = time.time()
            episode_serialized = pickle.dump(reconstructionObjects, testFile, protocol=pickle.HIGHEST_PROTOCOL)
            serial_time = time.time() - serial_start
            print("Serialization time (initialization_set_up) in s {:0.2f}s".format(serial_time))

        with open("test.pkl", "rb") as testFile:
            serial_start = time.time()
            episode_serialized = pickle.load(testFile)
            serial_time = time.time() - serial_start
            print("Loading  time (initialization_set_up) in s {:0.2f}s".format(serial_time))
        # DEBUG CODE END

        print("size of all objects here is ..{0}".format(sz))
        """

        # Set up for initialization
    def initialization_set_up_fast(self, seq_len, adjust_first_frame):
        print("Fast set up")
        if not self.useRealTimeEnv:
            self.correct_init_frame(adjust_first_frame, seq_len)
        self.get_valid_traj()

    # To update cars and pedestrians in episode from the real time environment.
    def update_pedestrians_and_cars(self,frame, car_pos,car_vel,car_angle, car_goal,car_bbox, people_dict, cars_dict, people_vel_dict, car_vel_dict, car_measurements, car_reward, probabilities, car_init_dir, car_action):

        # Remove any past
        if frame==0:
            #print("Update pedestrians and car: people predicted init at frame 0 "+str(self.reconstruction_2D[:, :, CHANNELS.pedestrian_trajectory]))
            self.people_predicted_init = np.zeros_like(self.reconstruction_2D[:, :, CHANNELS.pedestrian_trajectory])
            self.cars_predicted_init = np.zeros_like(self.reconstruction_2D[:,:,CHANNELS.cars_trajectory])

            self.reconstruction[:, :, :, CHANNELS.pedestrian_trajectory] = np.zeros_like(self.reconstruction[:, :, :, CHANNELS.pedestrian_trajectory])
            self.reconstruction[:,:,:,CHANNELS.cars_trajectory]=np.zeros_like(self.reconstruction[:,:,:,CHANNELS.cars_trajectory])

            if self.run_2D:
                self.reconstruction_2D[:, :, CHANNELS.pedestrian_trajectory] = np.zeros_like(self.reconstruction_2D[:, :, CHANNELS.pedestrian_trajectory])
                self.reconstruction_2D[:,:, CHANNELS.cars_trajectory] =np.zeros_like(self.reconstruction_2D[:,:, CHANNELS.cars_trajectory] )
            self.people=[]
            self.cars=[]
            self.car=[None]*self.seq_len
            self.car_angle = [None] * self.seq_len
            self.people_dict={}
            self.cars_dict={}
            self.car_bbox=[None]*self.seq_len
            self.people_vel_dict = [None] * self.seq_len
            self.car_vel_dict = [None] * self.seq_len

            self.car_vel_dict[0] = car_vel_dict
            self.people_vel_dict[0] = people_vel_dict

            self.cars_predicted=[]
            self.people_predicted=[]
            self.measures_car=np.zeros_like(self.measures_car)
            self.reward_car = np.zeros_like(self.reward_car )

            self.probabilities_car=np.zeros_like(self.probabilities_car)



        if len(car_init_dir)>0:
            self.car_dir = car_init_dir
        else:
            # print (" Update car velocity frame "+str(frame)+" vel "+str(car_vel))
            self.velocity_car[frame-1] = car_vel
            self.speed_car[frame-1]=np.linalg.norm(car_vel[1:])
            self.action_car[frame - 1]=car_action
            self.car_angle[frame-1] = car_angle
            # print(" Update episode action "+str(self.action_car[frame - 1])+" frame "+str(frame-1))

        # print(" Update frame "+str(frame)+" -----------------------------------------------------------_-----------------")
        # print ("People " + str(self.people))
        # print ("Update to " + str(people_dict))
        if len(self.people)<=frame:
            self.people.append(list(people_dict.values()))
        else:
            self.people[frame]=list(people_dict.values())
        # print ("After update  " + str(self.people[frame]))
        # print ("Update pedestrians and cars ")
        # print ("Cars " + str(self.cars))
        # print ("Update to " + str(cars_dict))
        if len(self.cars) <= frame:
            self.cars.append(list(cars_dict.values()))
        else:
            self.cars[frame]=list(cars_dict.values())

        # for car_traj in cars_dict.values():
        #     if self.init_cars[]
        #self.cars[frame]=#list(cars_dict.values())
        # print ("After update  " + str(self.cars[frame]))
        self.car[frame]=car_pos
        # print ("Car pos after update  " + str(self.car[frame]))
        self.car_bbox[frame]=car_bbox
        #print ("Car bbox pos after update  " + str( self.car_bbox[frame]))


        car_limits = np.zeros(6, dtype=np.int)
        for dim in range(len(self.car_bbox[frame])):
            car_limits[dim] = int(round(min(max(self.car_bbox[frame][dim], 0), self.reconstruction.shape[dim // 2] - 1)))
        # print (" Car bounding box " + str(
        #     [car_limits[0], car_limits[1], car_limits[2], car_limits[3], car_limits[4], car_limits[5]]))
        self.reconstruction[car_limits[0]:car_limits[1], car_limits[2]:car_limits[3], car_limits[4]:car_limits[5],
        CHANNELS.cars_trajectory] = (frame + 1) * np.ones_like(
            self.reconstruction[car_limits[0]:car_limits[1], car_limits[2]:car_limits[3], car_limits[4]:car_limits[5],
            CHANNELS.cars_trajectory])
        # print (" Car bounding box " + str(np.sum(
        #     self.reconstruction[car_limits[0]:car_limits[1], car_limits[2]:car_limits[3], car_limits[4]:car_limits[5],
        #     5])))
        if self.run_2D:
            self.reconstruction_2D[car_limits[2]:car_limits[3], car_limits[4]:car_limits[5], CHANNELS.cars_trajectory] = (
                                                                                                  frame + 1) * np.ones_like(
                self.reconstruction_2D[car_limits[2]:car_limits[3], car_limits[4]:car_limits[5], CHANNELS.cars_trajectory])



        # print ("Car bbox pos after update  " + str(self.car_bbox[frame]))
        self.car_goal=np.array(car_goal)
        # print ("Car goal after update  " + str(self.car_goal))

        self.people_vel_dict[frame]=people_vel_dict
        # print ("People velocities update  " + str(self.people_vel_dict[frame]))
        self.car_vel_dict[frame]=car_vel_dict
        # print ("Cars velocities update  " + str(self.car_vel_dict[frame]))

        # print ("Cars dict before update  " + str(self.cars_dict))
        # print ("Cars Reconstruction before update  " + str(np.sum(self.reconstruction[:,:,:, 5])))
        # print ("Cars Reconstruction 2D before update  " + str(np.sum(self.reconstruction_2D[ :, :, 5])))
        #print ("Cars Init frames car before update " + str(self.init_frames_car))
        for car_key in cars_dict.keys():

            if car_key not in self.cars_dict or frame==0:
                self.cars_dict[car_key]=[]
                self.init_frames_car[car_key]=frame

            car_a=cars_dict[car_key]

            if len(self.cars_dict[car_key])<=frame:
                self.cars_dict[car_key].append(car_a)
                # print("Append car "+str(car_a)+" key "+str(car_key))

            car_limits=np.zeros(6, dtype=np.int)
            for dim in range(len(car_a)):
                car_limits[dim] = int(round(min(max(car_a[dim], 0), self.reconstruction.shape[dim // 2] - 1)))

            # print (" Car bounding box "+str([car_limits[0],car_limits[1], car_limits[2],car_limits[3], car_limits[4],car_limits[5]]))
            self.reconstruction[car_limits[0]:car_limits[1], car_limits[2]:car_limits[3], car_limits[4]:car_limits[5], CHANNELS.cars_trajectory] = (frame+1) * np.ones_like(
                self.reconstruction[car_limits[0]:car_limits[1], car_limits[2]:car_limits[3], car_limits[4]:car_limits[5], CHANNELS.cars_trajectory])
            # print (" Car bounding box " + str(np.sum(self.reconstruction[car_limits[0]:car_limits[1], car_limits[2]:car_limits[3], car_limits[4]:car_limits[5], 5])))
            if self.run_2D:
                self.reconstruction_2D[car_limits[2]:car_limits[3], car_limits[4]:car_limits[5], CHANNELS.cars_trajectory] = (frame+1) * np.ones_like(
                    self.reconstruction_2D[car_limits[2]:car_limits[3], car_limits[4]:car_limits[5], CHANNELS.cars_trajectory])

        # print ("Cars Reconstruction after update  " + str(np.sum(self.reconstruction[:, :, :, 5])))
        # print ("Cars Reconstruction 2D after update  " + str(np.sum(self.reconstruction_2D[:, :, 5])))
        # #print ("Cars Init frames car after update "+str(self.init_frames_car))
        #
        # print ("People dict before update  " + str(self.people_dict))
        # print ("People Reconstruction before update  " + str(np.sum(self.reconstruction[:, :, :, 4])))
        # print ("People Reconstruction 2D before update  " + str(np.sum(self.reconstruction_2D[:, :, 4])))
        #print ("People Init frames before update " + str(self.init_frames))
        for ped_key in people_dict.keys():
            if ped_key not in self.people_dict or frame==0:
                self.people_dict[ped_key] = []
                self.init_frames[ped_key] = frame
            x_pers=people_dict[ped_key]
            if len(self.people_dict[ped_key])<=frame:
                self.people_dict[ped_key].append(x_pers)

            person = np.zeros(6, dtype=np.int)
            for i in range(x_pers.shape[0]):
                for j in range(x_pers.shape[1]):
                    # print("i "+str(i)+" j "+str(j)+" shape 0 "+str(x_pers.shape[1])+" dim "+str(i*x_pers.shape[1]+j))
                    person[i*x_pers.shape[1]+j] = int(round(min(max(x_pers[i][j], 0), self.reconstruction.shape[i] - 1)))
            # print (" person "+str(person))
            if self.run_2D:
                self.reconstruction_2D[person[2]:person[3],person[4]:person[5], CHANNELS.pedestrian_trajectory] = (frame+1) * np.ones_like(
                    self.reconstruction_2D[person[2]:person[3],person[4]:person[5], CHANNELS.pedestrian_trajectory])

            self.reconstruction[person[0]:person[1], person[2]:person[3],person[4]:person[5], CHANNELS.pedestrian_trajectory] = (frame+1) * np.ones_like(
                self.reconstruction[person[0]:person[1], person[2]:person[3],person[4]:person[5], CHANNELS.pedestrian_trajectory])
        # print ("People dict after update  " + str(self.people_dict))
        # print ("People Reconstruction after update  " + str(np.sum(self.reconstruction[:, :, :, 4])))
        # print ("People Reconstruction 2D after update  " + str(np.sum(self.reconstruction_2D[:, :, 4])))
        #print ("People Init frames after update " + str(self.init_frames))

        if frame >= len(self.cars_predicted):
            self.cars_predicted.append(copy.copy(self.reconstruction_2D[:, :, CHANNELS.cars_trajectory]))
        else:
            self.cars_predicted[frame]= self.reconstruction_2D[:, :, CHANNELS.cars_trajectory]
        # print ("Car predicted len " + str(len(self.cars_predicted)) + " sum " + str(np.sum(self.cars_predicted[frame])))

        if frame >= len(self.people_predicted):
            self.people_predicted.append(copy.copy(self.reconstruction_2D[:, :, CHANNELS.pedestrian_trajectory]))
        else:
            self.people_predicted[frame] = self.reconstruction_2D[:, :, CHANNELS.pedestrian_trajectory]
        # print ("People predicted len " + str(len(self.people_predicted)) + " sum " + str(np.sum(self.people_predicted[frame])))

        #print ("People predicted shape in episode update " + str(self.people_predicted[-1].shape)+" pos "+str(len(self.people_predicted)-1))
        self.measures_car[max(frame-1,0),:]=car_measurements
        #print ("Car measures after update  " + str(self.measures_car[max(frame-1,0),:]))
        # print ("Episode Saved car is distracted "+str(self.measures_car[max(frame-1,0),CAR_MEASURES_INDX.distracted]))
        self.reward_car[max(frame-1,0)] = car_reward
        # print ("Car reward after update  " + str(np.sum(self.reward_car[max(frame-1,0)]))+" reward "+str(car_reward)+" frame "+str(max(frame-1,0)))
        self.probabilities_car[max(frame-1,0),:]=probabilities
        # print ("Car probabilities after update  " + str(np.sum(self.probabilities_car[max(frame-1,0), :])))

        if frame==0:
            self.predict_cars_and_people()

            self.cars_predicted_init=self.mark_out_car_cv_trajectory(self.cars_predicted_init, self.car_bbox[0], self.car_goal, self.car_dir, time=1)
        # print (" People predicted init size "+str(self.people_predicted_init.shape)+" and sum "+str(np.sum(self.people_predicted_init)))
        # print (" Cars predicted init size " + str(self.cars_predicted_init.shape)+ " and sum "+str(np.sum(self.cars_predicted_init)))

    # Find cars and pedestrians that are at valid locations. i.e. can be used ofr initialization.
    def get_valid_traj(self):
        heights=[]
        print ("Valid trajectories")
        if len(self.people_dict) == 0 and len(self.cars_dict) == 0:
            print ("No dictionary! " )

            for sublist in self.people:
                for item in sublist:
                    person_middle = np.mean(item, axis=1).astype(int)

                    if person_middle[1] >= 0 and person_middle[0] >= 0 and person_middle[1] < \
                            self.valid_positions.shape[0] and person_middle[2] < self.valid_positions.shape[
                        1] and self.valid_positions[person_middle[1], person_middle[2]] == 1:
                        self.valid_people.append(item)
                        heights.append(person_middle[0])

        else:
            # Find valid people tracks.- People who are present during frame 0.

            for key in list(self.people_dict.keys()):
                fine_people = []
                valid = False

                # Pedestrian present before first frame.

                if self.init_frames[key] + len(self.people_dict[key]) < self.first_frame:
                    self.people_dict[key] = []

                else:

                    dif = self.first_frame - self.init_frames[key]
                    #  If agent present before first frame. Then update agent's initialization ro first frame.
                    if self.init_frames[key] < self.first_frame:
                        self.people_dict[key] = self.people_dict[key][dif:]
                        self.init_frames[key] = 0
                    else:  # Otherwise update when agent appears.
                        self.init_frames[key] = self.init_frames[key] - self.first_frame



                    for j, person in enumerate(self.people_dict[key], 0):

                        person_middle = np.mean(person, axis=1).astype(int)

                        if person_middle[1] >= 0 and person_middle[1] < self.reconstruction.shape[1] and person_middle[
                            2] >= 0 and person_middle[2] < self.reconstruction.shape[2] and self.valid_positions[
                            person_middle[1], person_middle[2]] == 1:
                            fine_people.append(person)

                            if j == 0 and self.init_frames[key] == 0:
                                valid = True


                            heights.append(person_middle[0])

                    if len(fine_people) > 1 or self.useRealTimeEnv:
                        self.valid_people_tracks[key] = list(fine_people)

                        if valid:
                            self.valid_keys.append(key)
            # print ("Cars dictionary "+str(len(self.cars_dict)))
            for key in list(self.cars_dict.keys()):
                # Pedestrian present before first frame.
                # print ("First frame  "+str(self.first_frame))
                if self.init_frames_car[key] + len(self.cars_dict[key]) <= self.first_frame:
                    self.cars_dict[key] = []

                else:
                    dif = self.first_frame - self.init_frames_car[key]
                    # print (" dif "+str(dif)+" key "+str(key)+" init frame "+str(self.init_frames_car[key]))
                    #  If agent present before first frame. Then update agent's initialization ro first frame.
                    if self.init_frames_car[key] <= self.first_frame:

                        self.cars_dict[key] = self.cars_dict[key][dif:]
                        self.init_frames_car[key] = 0
                        car=self.cars_dict[key][0]
                        if self.useRealTimeEnv:
                            self.init_cars.append(key)
                        elif len(self.cars_dict[key]) > 10:
                            car = self.cars_dict[key][10]
                            if car[3] >= 0 and car[2] < self.reconstruction.shape[1] and car[5] >= 0 and car[4] < self.reconstruction.shape[2]:
                                self.init_cars.append(key)
                    else:  # Otherwise update when agent appears.
                        self.init_frames_car[key] = self.init_frames_car[key] - self.first_frame
        self.heights=heights

                            # Find valid people tracks.- People who are present during frame 0.



    def get_heatmap(self):
        from sklearn.neighbors import KernelDensity
        people_centers = []
        for people_list in self.people:
            for pers in people_list:
                people_centers.append(np.mean(pers, axis=1)[1:])
        self.heatmap = np.zeros((self.reconstruction.shape[1], self.reconstruction.shape[2]))
        if len(people_centers) > 1:
            # instantiate and fit the KDE model
            kde = KernelDensity(bandwidth=0.0001, kernel='exponential')
            kde.fit(people_centers)

            for x in range(self.heatmap.shape[0]):
                for y in range(self.heatmap.shape[1]):
                    self.heatmap[x, y] = kde.score_samples(np.array([x, y]).reshape(1, -1))
                    if np.isnan(self.heatmap[x, y]):
                        self.heatmap[x, y] = 0
        # convolutional solution
        #import scipy.ndimage
        #self.heatmap = scipy.ndimage.gaussian_filter(np.sum(self.reconstruction[:, :, :, CHANNELS.pedestrian_trajectory], axis=0), 15)

        # Normalize heatmap
        max_value = np.max(self.heatmap[np.isfinite(self.heatmap)])
        min_value = np.min(self.heatmap[np.isfinite(self.heatmap)])
        if max_value - min_value > 0:
            self.heatmap = (self.heatmap - min_value) / (max_value - min_value)
        for x in range(self.heatmap.shape[0]):
            for y in range(self.heatmap.shape[1]):

                if np.isinf(self.heatmap[x, y]):
                    if self.heatmap[x, y] > 0:
                        self.heatmap[x, y] = 1
                    else:
                        self.heatmap[x, y] = 0
                        # print "Done renormalizing"
        self.heatmap

    def find_time_in_boundingbox(self, pos_init_average, vel, bounding_box):
        t_crossing=[]
        for i in range(2):
            if abs(vel[i]) > 0:
                for j in range(2):
                    #print ("Calculate t "+str(bounding_box[i][j]-pos_init_average[i])+" divided by "+str( vel[i]))
                    t=(bounding_box[i][j]-pos_init_average[i])/ vel[i]
                    second_coordinate=(t*vel[i-1])+pos_init_average[i-1]
                    #print (" Time "+str(t)+" i "+str(i)+" j "+str(j)+" second_coordinate "+str(second_coordinate)+" compare to "+str(bounding_box[i-1][0])+" and "+str(bounding_box[i-1][1]))
                    if t>=0 and bounding_box[i-1][0]<=second_coordinate and second_coordinate<=bounding_box[i-1][1]:
                        t_crossing.append(copy.copy(t))
        return t_crossing

    def person_bounding_box(self, frame_in,no_height=False, channel=3, pos=()):
        x_range, y_range, z_range = self.pose_range(frame_in, pos=pos, as_int=True)
        # print "Position "+str(pos)+ " "+str(x_range)+" "+str(y_range)+" "+str(z_range)
        if no_height:
            x_range = [0, self.reconstruction.shape[0]]
        segmentation = (
            self.reconstruction[int(x_range[0]):int(x_range[1]) + 1, int(y_range[0]):int(y_range[1]) + 1,
            int(z_range[0]):int(z_range[1]) + 1, channel] * int(NUM_SEM_CLASSES)).astype(
            np.int32)

        return segmentation


    def predict_cars_and_people_in_a_bounding_box(self, prediction_people,prediction_car,bounding_box_of_prediction, start_pos,frame):

        if len(bounding_box_of_prediction) == 0:
            bounding_box_of_prediction = [[0, self.reconstruction.shape[1]], [0, self.reconstruction.shape[2]]]
            frame = self.seq_len
        bounding_box_reshape=[[bounding_box_of_prediction[0],bounding_box_of_prediction[1]],[bounding_box_of_prediction[2],bounding_box_of_prediction[3]]]
        # print ("Predict cars and people frame "+str(frame)+" bbox "+str(bounding_box_of_prediction))
        # print ("Number of People "+str(len(self.people_vel_dict[frame].keys()))+" cars "+str(len(self.car_vel_dict[frame].keys())))
        for key, track in list(self.people_dict.items()):  # Go through all pedestrians
            frame_init = self.init_frames[key]
            previous_pos = np.array([0, 0, 0])
            # print(" Person "+str(key)+" init frame "+str(frame_init)+" frame "+str(frame))
            if frame_init<=frame: # Find position of pedestrian in current frame
                frame_diff=frame-frame_init
                #print(" len " + str(len(self.people_dict[key])))
                if len(self.people_dict[key])>frame_diff:
                    pos_init=self.people_dict[key][frame_diff]
                    pos_init_average = np.array([np.mean(pos_init[0]), np.mean(pos_init[1]), np.mean(pos_init[2])])
                    if key in self.people_vel_dict[frame].keys():
                        vel= self.people_vel_dict[frame][key]
                        # print (" person "+str(key)+" frame "+str(frame))
                        # print(" Average pos "+str(pos_init_average)+" vel "+str(vel))
                        # Find out if pedestrian's tracks come into the bounding box

                        self.add_predicted_car_to_matrix(bounding_box_reshape, pos_init, pos_init_average,
                                                         prediction_people,
                                                         vel,start_pos)
        # print (" After prediction people " + str(prediction_people))
        for key, track in list(self.cars_dict.items()):  # Go through all pedestrians
            frame_init = self.init_frames_car[key]
            #print(" Car " + str(key) + " init frame " + str(frame_init) + " frame " + str(frame))
            if frame_init <= frame:  # Find position of pedestrian in current frame
                frame_diff = frame - frame_init
                if len(self.cars_dict[key]) > frame_diff:
                    pos_init = self.cars_dict[key][frame_diff]
                    pos_init_average = np.array([np.mean(pos_init[:2]), np.mean(pos_init[2:4]), np.mean(pos_init[4:])])
                    if key in self.car_vel_dict[frame].keys():
                        vel = self.car_vel_dict[frame][key]
                        # print (" car " + str(key)+" frame "+str(frame))
                        # print(" Average pos " + str(pos_init_average) + " vel " + str(vel))
                        # Find out if pedestrian's tracks come into the bounding box
                        self.add_predicted_car_to_matrix(bounding_box_reshape, np.reshape(pos_init,(3,2)), pos_init_average, prediction_car,
                                                         vel,start_pos)
        pos_init = self.car_bbox[frame]

        pos_init_average = np.array([np.mean(pos_init[:2]), np.mean(pos_init[2:4]),np.mean(pos_init[4:])])
        if frame==0:
            vel=self.car_dir
        else:
            vel = self.velocity_car[frame-1]
        # print (" Add car to prediction " + str(pos_init)+" velocity "+str(vel))
        # Find out if pedestrian's tracks come into the bounding box
        self.add_predicted_car_to_matrix(bounding_box_reshape, np.reshape(pos_init,(3,2)), pos_init_average, prediction_car,vel,start_pos)

        # print ("After prediction cars " + str(prediction_car))
        self.add_predicted_people(frame, prediction_people)
        #self.agent_prediction_car.append(prediction_car)
        #print ("Added to predicted people list " + str(len(self.agent_prediction_people))+" frame "+str(frame))
        return prediction_people, prediction_car

    def add_predicted_people(self, frame, prediction_people):
        if len(self.agent_prediction_people) <= frame:
            self.agent_prediction_people.append(prediction_people)
        else:
            self.agent_prediction_people[frame] = prediction_people

    def add_predicted_car_to_matrix(self, bounding_box_of_prediction, pos_init, pos_init_average, prediction_car, vel, start_pos):
        t_crossing = self.find_time_in_boundingbox(pos_init_average[1:], vel[1:], bounding_box_of_prediction)

        if len(t_crossing)==1:
            t_crossing.append(0)
        # print (" Time in bounding box " + str(t_crossing))
        if len(t_crossing) == 2:  # if yes then we can predict
            t_min = int(min(t_crossing))

            t_max = int(max(t_crossing))#, self.seq_len+10])
            if np.linalg.norm(vel[1:])< 0.01:
                t_max=t_min+1
                # print("Velocity is too small "+str(np.linalg.norm(vel[1:]))+" set tmax to "+str(t_max))

            t_diff=1
            if max(abs(vel[1:]))<1:
                t_diff=int(1/max(abs(vel[1:])))
                # print (" Adapt t dif "+str(t_diff))

            # print (" t_min "+str(t_min)+" t max "+str(t_max)+" location "+str(np.mean(pos_init, axis=1))+" final "+str(np.mean(pos_init + np.tile(t_max * vel, [2, 1]).T, axis=1))+" tdiff "+str(t_diff)+" vel "+str(vel))
            for t in range(t_max, t_min, -t_diff):
                location = pos_init + np.tile(t * vel, [2, 1]).T
                average_location = pos_init_average + t * vel
                # print (" Average location "+str(average_location)+" location "+str(location[1:])+" bounding box "+str(bounding_box_of_prediction))
                dir = [start_pos[0]+max(location[0][0].astype(int), 0),
                       start_pos[0] +min(location[0][1].astype(int) + 1, self.reconstruction.shape[0]),
                       start_pos[1] +max(location[1][0].astype(int)-bounding_box_of_prediction[0][0], 0),
                       start_pos[1] +min(location[1][1].astype(int)-bounding_box_of_prediction[0][0] + 1, prediction_car.shape[0]),
                       start_pos[2] +max(location[2][0].astype(int)-bounding_box_of_prediction[1][0], 0),
                       start_pos[2] +min(location[2][1].astype(int)-bounding_box_of_prediction[1][0] + 1, prediction_car.shape[1])]
                # print (" Before change " + str(np.sum(prediction_car[dir[2]:dir[3],
                #                                      dir[4]:dir[5]])))
                prediction_car[dir[2]:dir[3],
                dir[4]:dir[5]] = t * np.ones_like(prediction_car[dir[2]:dir[3], dir[4]:dir[5]])
                # print (" After change " + str(np.sum(prediction_car[dir[2]:dir[3],
                # dir[4]:dir[5]])) +" shape of input "+str(prediction_car.shape)+"  dims "+str([dir[2],dir[3], dir[4],dir[5]])+" "+str(t))


    def predict_cars_and_people(self, bounding_box_of_prediction=[], frame=-1):

        # print ("Episode function predict cars and people ")
        # Each people_predicted[frame] contains constant velocity predictions of all pedestrians
        if self.predict_future:

            for key, track in list(self.people_dict.items()): # Go through all pedestrians

                frame_init = self.init_frames[key]
                previous_pos = np.array([0, 0, 0])
                for index, pos in enumerate(track): # Go through all frames of pedestrian
                    frame = frame_init + index

                    location = pos
                    average_location = np.array([np.mean(pos[0]), np.mean(pos[1]), np.mean(pos[2])])
                    #print (" Init frame " + str(frame)+ " average location "+str(average_location))
                    if self.useRealTimeEnv or index > 0:
                        if self.useRealTimeEnv:

                            #if self.people_vel_dict[frame]!=None:
                            vel= self.people_vel_dict[frame][key]
                            #print ("person vel "+str(vel))
                        else:
                            vel = average_location - previous_pos
                            vel[0] = 0
                        if np.linalg.norm(vel) > 0.1:
                            loc_frame = 1
                            if self.useRealTimeEnv:
                                loc_frame=2
                            #Do linear predictions
                            while np.all(average_location[1:] > 1) and np.all(
                                                    self.reconstruction.shape[1:3] - average_location[1:] > 1):
                                location = location + np.tile(vel, [2, 1]).T
                                average_location = average_location + vel
                                #print (" Init frame " + str(frame) + " average location " + str(average_location))
                                dir = [max(location[0][0].astype(int), 0),
                                       min(location[0][1].astype(int) + 1, self.reconstruction.shape[0]),
                                       max(location[1][0].astype(int), 0),
                                       min(location[1][1].astype(int) + 1, self.reconstruction.shape[1]),
                                       max(location[2][0].astype(int), 0),
                                       min(location[2][1].astype(int) + 1, self.reconstruction.shape[2])]
                                #print (" Bounding box "+str(dir)+ " size "+str(dir[3]-dir[2])+" "+str(dir[5]-dir[4]))
                                if self.temporal:# Fill with frame number
                                    if self.run_2D:
                                        if self.useRealTimeEnv:
                                            # print(" Shape "+str(self.people_predicted_init.shape)+" bbox "+str([dir[2],dir[3],
                                            # dir[4],dir[5]]))
                                            self.people_predicted_init[dir[2]:dir[3],
                                            dir[4]:dir[5]] = loc_frame * np.ones_like(
                                                self.people_predicted_init[dir[2]:dir[3], dir[4]:dir[5]])
                                        else:
                                            self.people_predicted[frame][dir[2]:dir[3],
                                            dir[4]:dir[5]] = loc_frame * np.ones_like(
                                                self.people_predicted[frame][dir[2]:dir[3], dir[4]:dir[5]])
                                    else:
                                        self.people_predicted[frame][dir[0]:dir[1], dir[2]:dir[3],
                                        dir[4]:dir[5]] = loc_frame * np.ones_like(
                                            self.people_predicted[frame][dir[0]:dir[1], dir[2]:dir[3],
                                            dir[4]:dir[5]])
                                else:
                                    if self.run_2D:  # Fill with 0.1s
                                        self.people_predicted[frame][dir[2]:dir[3], dir[4]:dir[5]] = \
                                            self.people_predicted[frame][dir[2]:dir[3], dir[4]:dir[5]] + 0.1
                                    else:
                                        self.people_predicted[frame][dir[0]:dir[1], dir[2]:dir[3], dir[4]:dir[5]] = \
                                            self.people_predicted[frame][dir[0]:dir[1], dir[2]:dir[3],
                                            dir[4]:dir[5]] + 0.1
                                # print (" Bounding box " + str(dir)+" "+str(np.mean(self.people_predicted[frame][dir[2]:dir[3],
                                #         dir[4]:dir[5]]))+" init sum "+str(np.mean(self.people_predicted_init[dir[2]:dir[3],
                                #         dir[4]:dir[5]])))
                                loc_frame = loc_frame + 1
                    previous_pos = np.array([np.mean(pos[0]), np.mean(pos[1]), np.mean(pos[2])])
            # print "After  --------------------------------------------"
            #  for frame, pred in enumerate(self.people_predicted):
            #      print str(frame)
            #      print np.sum(pred)
            #print (" Cars dict "+str(self.cars_dict))
            for key, track in list(self.cars_dict.items()):
                frame_init = self.init_frames_car[key]
                previous_pos = np.array([0, 0, 0])
                for index, pos in enumerate(track):
                    frame = frame_init + index
                    location = pos
                    average_location = np.array(
                        [np.mean([pos[0], pos[1]]), np.mean([pos[2], pos[3]]), np.mean([pos[4], pos[5]])])
                    #print (" Init frame " + str(frame) + " average location " + str(average_location))
                    if  self.useRealTimeEnv or index > 0:
                        if self.useRealTimeEnv:
                            vel= self.car_vel_dict[frame][key]
                            #print ("car vel " + str(vel))
                        else:
                            vel = average_location - previous_pos
                            vel[0] = 0
                        if np.linalg.norm(vel) > 0.1:
                            loc_frame = 1
                            if self.useRealTimeEnv:
                                loc_frame = 2
                            while np.all(average_location[1:] > 1) and np.all(
                                                    self.reconstruction.shape[1:3] - average_location[1:] > 1):
                                location = location + np.array([0, 0, vel[1], vel[1], vel[2], vel[2]])
                                average_location = average_location + vel
                                #print (" Init frame " + str(frame) + " average location " + str(average_location))

                                dir = [max(location[0].astype(int), 0),
                                       min(location[1].astype(int) + 1, self.reconstruction.shape[0]),
                                       max(location[2].astype(int), 0),
                                       min(location[3].astype(int) + 1, self.reconstruction.shape[1]),
                                       max(location[4].astype(int), 0),
                                       min(location[5].astype(int) + 1, self.reconstruction.shape[2])]
                                #print (" Bounding box " + str(dir) + " size " + str(dir[3] - dir[2]) + " " + str(
                                #    dir[5] - dir[4]))
                                if self.temporal:
                                    if self.run_2D:
                                        if self.useRealTimeEnv:
                                            self.cars_predicted_init[dir[2]:dir[3],
                                            dir[4]:dir[5]] = loc_frame * np.ones_like(
                                                self.cars_predicted_init[dir[2]:dir[3], dir[4]:dir[5]])
                                            loc_frame = loc_frame + 1
                                        else:
                                            self.cars_predicted[frame][dir[2]:dir[3],
                                            dir[4]:dir[5]] = loc_frame * np.ones_like(
                                                self.cars_predicted[frame][dir[2]:dir[3], dir[4]:dir[5]])
                                            loc_frame = loc_frame + 1
                                    else:
                                        self.cars_predicted[frame][dir[0]:dir[1], dir[2]:dir[3],
                                        dir[4]:dir[5]] = loc_frame * np.ones_like(
                                            self.cars_predicted[frame][dir[0]:dir[1], dir[2]:dir[3], dir[4]:dir[5]])
                                else:
                                    if self.run_2D:
                                        self.cars_predicted[frame][dir[2]:dir[3], dir[4]:dir[5]] = \
                                            self.cars_predicted[frame][dir[2]:dir[3], dir[4]:dir[5]] + 0.1

                                    else:
                                        self.cars_predicted[frame][dir[0]:dir[1], dir[2]:dir[3], dir[4]:dir[5]] = \
                                            self.cars_predicted[frame][dir[0]:dir[1], dir[2]:dir[3], dir[4]:dir[5]] + 0.1
                                # print (" Bounding box " + str(dir) + " " + str(
                                #     np.mean(self.cars_predicted[frame][dir[2]:dir[3],
                                #             dir[4]:dir[5]])) + " init sum " + str(
                                #     np.mean(self.cars_predicted_init[dir[2]:dir[3],
                                #             dir[4]:dir[5]])))
                                loc_frame = loc_frame + 1
                    previous_pos = np.array(
                        [np.mean([pos[0], pos[1]]), np.mean([pos[2], pos[3]]), np.mean([pos[4], pos[5]])])

    def get_valid_positions(self, mark_out_people=True):
        # Find valid positions in tensor.
        for x in range(self.valid_positions.shape[1]):
            for y in range(self.valid_positions.shape[0]):
                if self.valid_position([0, y, x], no_height=True):  # and np.sum(self.reconstruction[:, y, x, 5]) == 0:
                    self.valid_positions[y, x] = 1
                    #print(" Valid pos x: "+str(x)+" y: "+str(y))

        self.mark_out_cars_in_valid_pos()
        #print(" After mark out cars "+str(np.where(self.valid_positions)))
        if mark_out_people:
            self.mark_out_people_in_valid_pos()
            #print(" After mark out people " + str(np.where(self.valid_positions)))

    def mark_out_cars_in_valid_pos(self):
        if self.useRealTimeEnv:

            for car_id in self.init_cars:
                t = 0
                dif = max(t - 1, 0) - 1
                car, car_vel = self.get_car_pos_and_vel(car_id)
                limits = self.get_car_limits(car, dif)

                while limits[0] < limits[1] and limits[2] < limits[3]:
                    self.valid_positions[limits[0]:limits[1], limits[2]:limits[3]] = 0 * self.valid_positions[
                                                                                         limits[0]:limits[1],
                                                                                         limits[2]:limits[3]]
                    t = t + 1
                    dif = max(t - 1, 0) - 1
                    car = self.extract_car_pos(car, car_vel)
                    limits = self.get_car_limits(car, dif)

        else:
            # for t in range(0, min(max(abs(self.cars[0][0][2] - self.cars[0][0][3]), abs(self.cars[0][0][4] - self.cars[0][0][5])) + 2,len(self.cars))):
            for t in range(len(self.cars)):
                for car in self.cars[t]:
                    dif = max(t - 1, 0) - 1
                    limits = self.get_car_limits(car, dif)
                    if limits[0] < limits[1] and limits[2] < limits[3]:
                        self.valid_positions[limits[0]:limits[1], limits[2]:limits[3]] = 0 * self.valid_positions[
                                                                                             limits[0]:limits[1],
                                                                                             limits[2]:limits[3]]

    def mark_out_people_in_valid_pos(self):
        if self.useRealTimeEnv:
            for pedestrian_id in self.valid_keys:
                t = 0
                dif = max(t - 1, 0) #- 1
                person, pedestrian_vel = self.get_pedestrian_pos_and_vel(pedestrian_id)
                limits = self.get_pedestrian_limits(person, dif)
                while limits[0] < limits[1] and limits[2] < limits[3]:
                    self.valid_positions[limits[0]:limits[1], limits[2]:limits[3]] = 0 * self.valid_positions[
                                                                                         limits[0]:limits[1],
                                                                                         limits[2]:limits[3]]
                    t = t + 1
                    dif = max(t - 1, 0)# - 1
                    person_new = self.extract_pedestrian_pos(person, pedestrian_vel)
                    limits = self.get_car_limits(person_new, dif)

        else:
            # for t in range(0, min(max(abs(self.cars[0][0][2] - self.cars[0][0][3]), abs(self.cars[0][0][4] - self.cars[0][0][5])) + 2,len(self.cars))):
            for t in range(len(self.people)):
                for person in self.people[t]:
                    dif = max(t - 1, 0) #- 1
                    limits = self.get_pedestrian_limits(person, dif)
                    if limits[0] < limits[1] and limits[2] < limits[3]:
                        self.valid_positions[limits[0]:limits[1], limits[2]:limits[3]] = 0 * self.valid_positions[
                                                                                             limits[0]:limits[1],
                                                                                             limits[2]:limits[3]]

    def get_car_pos_and_vel(self, car_id, frame=0):
        car = self.cars_dict[car_id][frame]
        car_vel = self.car_vel_dict[frame][car_id]

        return car, car_vel

    def get_pedestrian_pos_and_vel(self, pedestrian_id,frame=0):
        pedestrian = self.people_dict[pedestrian_id][frame]
        pedestrian_vel = self.people_vel_dict[frame][pedestrian_id]

        return pedestrian, pedestrian_vel

    def extract_car_pos(self, car, car_vel):
        return [car[0] + car_vel[0], car[1] + car_vel[0], car[2] + car_vel[1], car[3] + car_vel[2],
                car[4] + car_vel[2], car[5] + car_vel[2]]

    def extract_pedestrian_pos(self, person, ped_vel):
        return [person[0][0] + ped_vel[0], person[0][1] + ped_vel[0], person[1][0] + ped_vel[1], person[1][1] + ped_vel[1],
                person[2][0] + ped_vel[2], person[2][1] + ped_vel[2]]

    def get_car_limits(self, car, dif):
        limits = [max(int(car[2]) + dif - self.agent_size[1], 0),
                  min(int(car[3]) - dif + self.agent_size[1] + 1, self.valid_positions.shape[0]),
                  max(int(car[4]) - self.agent_size[2] + dif, 0),
                  min(int(car[5]) - dif + self.agent_size[2] + 1, self.valid_positions.shape[1])]
        return limits

    def get_pedestrian_limits(self, person, dif):
        # print(" Agent size "+str(self.agent_size)+" person "+str(person))
        limits = [max(int(person[1][0]) + dif - self.agent_size[1], 0),
                          min(int(person[1][1]) - dif + self.agent_size[1]+1 , self.valid_positions.shape[0]),
                          max(int(person[2][0]) - self.agent_size[2] + dif, 0),
                          min(int(person[2][1]) - dif + self.agent_size[2]+1, self.valid_positions.shape[1])]
        # print ("Limits "+str(limits )+" dif "+str(dif))

        return limits

    def correct_init_frame(self, adjust_first_frame, seq_len):
        self.first_frame = 50
        if len(self.people) < 500 or len(self.cars) < 500: # If a carla run adapt the first frame
            self.first_frame = 0
        if adjust_first_frame:
            if len(self.people[self.first_frame]) == 0:
                while (self.first_frame < len(self.people) and len(self.people[self.first_frame]) == 0):
                    self.first_frame += 1
                if self.first_frame == len(self.people):
                    self.first_frame = 0
                if self.first_frame >= len(self.people) - seq_len:
                    self.first_frame = len(self.people) - seq_len
        self.people = self.people[self.first_frame:]
        self.cars = self.cars[self.first_frame:]
        for frame in range(self.first_frame):
            self.people.append([])
            self.cars.append([])




    def initial_position(self, poses_db, initialization=-1, training=True, init_key=-1, set_vel_init=True):
        if set_vel_init:
            maxAgentSpeed_voxels = 300 / 15 * self.frame_time
            random_speed = np.random.rand() * maxAgentSpeed_voxels
            dir = np.array(self.actions[random.randint(0, len(self.actions) - 1)])
            if np.linalg.norm(dir) > 1:
                dir_norm = dir / np.linalg.norm(dir)
            else:
                dir_norm = dir
            self.vel_init = dir_norm * random_speed
            print(("Initial vel init "+str(self.vel_init)+" follow goal "+str(self.follow_goal)))
        # Reset measures.
        i = 0
        self.goal_person_id=-1
        self.goal=[]
        self.measures = np.zeros(self.measures.shape)
        # Check if goal can be found

        # If given specification on initialization type.
        if initialization==PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian:

            if self.people_dict and  len(self.valid_keys)>0:
                self.init_method = PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian
                print("initialize on pedestrian ")
                if init_key < 0:
                    init_key=random.randint(len(self.valid_keys))
                self.init_method = PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian
                return self.initial_position_key(init_key)
            else:
                print("No valid people in frame")
                return [], -1, self.vel_init

        elif initialization==PEDESTRIAN_INITIALIZATION_CODE.by_car:
            self.init_method = PEDESTRIAN_INITIALIZATION_CODE.by_car
            print(("initialize by car " + str(self.init_method)))
            return self.initialize_by_car()

        elif initialization==PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian:
            self.init_method = PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian
            print(("initialize by pedestrian " + str(self.init_method)))
            if self.people_dict and  len(list(self.valid_people_tracks.keys()))>0:
                return self.initialize_by_ped_dict()
            else:
                print("No valid pedestrians")
                return [], -1, self.vel_init

        elif initialization==PEDESTRIAN_INITIALIZATION_CODE.randomly:
            self.init_method = PEDESTRIAN_INITIALIZATION_CODE.randomly
            print(("initialize randomly " + str(self.init_method)))
            return self.initialize_randomly()

        elif initialization == PEDESTRIAN_INITIALIZATION_CODE.in_front_of_car:
            self.init_method = PEDESTRIAN_INITIALIZATION_CODE.in_front_of_car
            print(("initialize in front of car " + str(self.init_method)))
            if self.init_cars:
                return self.initialize_by_car_dict()
            else:
                return self.initialize_by_car()

        elif initialization==PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian :
            self.init_method = PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian
            print(("initialize by pedestrian "+str(initialization)+" " + str(self.init_method)))
            if self.people_dict and len(list(self.valid_people_tracks.keys())) > 0:
                return self.initialize_on_pedestrian_with_goal(on_pedestrian=False)
            else:
                print("No valid pedestrians")
            return [], -1, self.vel_init

        elif initialization==PEDESTRIAN_INITIALIZATION_CODE.on_pavement:
            self.init_method =PEDESTRIAN_INITIALIZATION_CODE.on_pavement
            print("Initialize on pavement")
            return self.initialize_on_pavement()

        elif initialization==PEDESTRIAN_INITIALIZATION_CODE.near_obstacle:
            self.init_method=PEDESTRIAN_INITIALIZATION_CODE.near_obstacle
            print(("initialize near obstacles " + str(self.init_method)+" border len "+str(len(self.border))))
            if len(self.border)>0:
                return self.initialize_near_obstacles()
            return self.initialize_randomly()

        elif initialization == PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian_trajectory and not self.useRealTimeEnv:
            self.init_method = PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian_trajectory
            print(("initialize on pedestrian trajectory" + str(self.init_method)))
            keys=self.valid_keys
            if len(keys)==0:
                return [], -1, self.vel_init
            init_id=np.random.randint(len(keys))
            init_key=keys[init_id]
            init_frame=0
            if len(self.valid_people_tracks[init_key])>5:
                init_frame=5+np.random.randint(len(self.valid_people_tracks[init_key])-5)
            else:
                return [], -1, self.vel_init
            return self.initial_position_key(init_id, init_frame)



        # Otherwise if training: initialize randomly, when testing initialize near people
        if training:
            u=random.uniform(0,1) # Random number.
            if len(self.valid_people) > 0:
                if u > 0.5:
                    if u<0.25:
                        self.init_method=PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian
                        return self.initialize_agent_pedestrian()
                    else:
                        self.init_method = PEDESTRIAN_INITIALIZATION_CODE.by_pedestria
                        return self.initialize_agent_pedestrian(on_pedestrian=False)
            if u> 0.75:
                self.init_method = PEDESTRIAN_INITIALIZATION_CODE.by_car
                return self.initialize_by_car()
            self.init_method = PEDESTRIAN_INITIALIZATION_CODE.randomly
            return self.initialize_randomly()
        else:
            print(("Init key "+str(init_key)))
            if init_key >=0 and not self.useRealTimeEnv:
                self.init_method = PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian
                return self.initial_position_key(init_key)
            else:
                print("No People")
                return [], -1, self.vel_init


    def initial_position_key(self,initialization_key, frame_init=0):
        initialization_key_val=initialization_key
        if self.key_map:
            initialization_key_val=self.key_map[initialization_key]
        person=self.valid_people_tracks[initialization_key_val][frame_init]
        self.agent[0] = np.mean(person, axis=1)
        self.vel_init=np.mean(self.valid_people_tracks[initialization_key_val][frame_init+1], axis=1)-np.mean(person, axis=1)

        print(("Vel init "+str(self.vel_init)))

        if self.follow_goal:
            self.goal = np.mean(self.valid_people_tracks[initialization_key_val][min(frame_init+self.seq_len,len(self.valid_people_tracks[initialization_key_val])-1)], axis=1).astype(int)

        if frame_init==0:
            print(" Set initialization key to "+str(initialization_key_val))
            self.goal_person_id=initialization_key
            self.goal_person_id_val=initialization_key_val
        print((self.goal_person_id))
        print(("Starting at "+str(self.agent[0])+" "+str(self.goal)))
        return self.agent[0], self.goal_person_id,self.vel_init


    def initialize_agent_pedestrian(self, on_pedestrian=True):
        if self.people_dict and len(list(self.valid_people_tracks.keys())) > 0:  # Initialize on pedestrian track
            self.initialize_on_pedestrian_with_goal(on_pedestrian=on_pedestrian)
            return self.agent[0], self.goal_person_id, self.vel_init
        else:
            self.initialize_on_pedestrian(on_pedestrian=on_pedestrian)
        return self.agent[0], -1, self.vel_init

    def initialize_car_environment(self, training):
        height = np.mean(self.cars[0][0][0:1])
        if training == False:
            pos = random.randint(0, len(self.test_positions)-1)

            self.agent[0] = np.array([height, self.test_positions[pos][0], self.test_positions[pos][1]])
            self.agent[0] = self.agent[0].astype(int)

            return self.agent[0], -1, self.vel_init
        valid_places = np.where(self.valid_positions)
        pos = random.randint(0, len(valid_places[0])-1)
        self.agent[0] = np.array([height, valid_places[0][pos], valid_places[1][pos]])
        if self.direction == 0:
            self.goal = self.agent[0]
            self.goal[1] = abs(self.reconstruction.shape[1] - self.goal[1])
        elif self.direction == 1:
            self.goal = self.agent[0]
            self.goal[2] = abs(self.reconstruction.shape[2] - self.goal[2])
        else:
            self.goal = self.agent[0]
            self.goal[1] = abs(self.reconstruction.shape[1] - self.goal[1])
            self.goal[2] = abs(self.reconstruction.shape[2] - self.goal[2])
        self.agent[0]=self.agent[0].astype(int)
        self.goal = self.goal.astype(int)
        return self.agent[0], -1, self.vel_init

    def initialize_on_pedestrian_with_goal(self, on_pedestrian=True):

        if len(self.valid_keys)>0:
            # print ("Person ids :"+str(self.valid_keys))
            # for key in range(len(self.valid_keys)):
            #     print (str(key)+" true "+str(self.key_map[key]))
            self.goal_person_id = np.random.randint(len(self.valid_keys))
            print(("goal person id "+str(self.goal_person_id)))
            self.goal_person_id_val=self.key_map[self.goal_person_id]
            person =  self.valid_people_tracks[self.goal_person_id_val][0].astype(int)

            self.agent[0] = np.mean(person, axis=1).astype(int)
            if self.follow_goal:
                if not self.useRealTimeEnv:
                    self.goal=np.mean( self.valid_people_tracks[self.goal_person_id_val][-1], axis=1).astype(int)
                else:
                    self.goal =self.agent[0]+ self.seq_len*self.people_vel_dict[0][self.goal_person_id_val]
        elif len(list(self.valid_people_tracks.keys()))>0:
            keys=list(self.valid_people_tracks.keys())
            self.goal_person_id = np.random.randint(len(keys))
            self.goal_person_id_val =keys[self.goal_person_id]
            self.init_method = PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian
            start_frame=np.random.randint(len(self.valid_people_tracks[self.goal_person_id_val])-1)
            person =  self.valid_people_tracks[self.goal_person_id_val][start_frame].astype(int)
            self.agent[0] = np.mean(person, axis=1).astype(int)
            if self.follow_goal:
                if not self.useRealTimeEnv:
                    self.goal = np.mean( self.valid_people_tracks[self.goal_person_id_val][np.int(len( self.valid_people_tracks[self.goal_person_id_val])-1 )], axis=1).astype(int)
                else:
                    self.goal = self.agent[0] + self.seq_len * self.people_vel_dict[0][self.goal_person_id_val]
        #print("Initialize on pedestrian with goal before "+str(self.goal_person_id)+" "+str(self.agent[0]))
        if not on_pedestrian:
            self.init_method = PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian
            self.goal_person_id=-1

            width = max((person[1][1] - person[1][0])*2,5)

            depth = max((person[2][1] - person[2][0])*2,5)

            lims = [max(person[1][0] - width, 0), max(person[2][0] - depth, 0)]

            cut_out = self.valid_positions[max(person[1][0] - width, 0):min(person[1][1] + width, self.valid_positions.shape[0]),
                      max(person[2][0] - depth, 0):min(person[2][1] + depth+1, self.valid_positions.shape[1])]
            #print("Cut out  " + str(cut_out))
            while(np.sum(cut_out)==0 and width<self.valid_positions.shape[0]/2 and depth<self.valid_positions.shape[1]/2):
                width = width* 2
                depth =depth * 2
                lims = [max(person[1][0] - width, 0), max(person[2][0] - depth, 0)]
                cut_out = self.valid_positions[
                          max(person[1][0] - width, 0):min(person[1][1] + width, self.valid_positions.shape[0]),
                          max(person[2][0] - depth, 0):min(person[2][1] + depth+1, self.valid_positions.shape[1])]

            test_pos = find_border(cut_out)

            #print("Test pos  " + str(test_pos))
            if len(test_pos)>0:
                i = random.choice(list(range(len(test_pos))))
                #print ("Random test pos "+str(i)+" "+str(test_pos[i][0])+" "+str(test_pos[i][1]))
                self.agent[0] = np.array([self.agent[0][0], lims[0] + test_pos[i][0], lims[1] + test_pos[i][1]]).astype(int)
                #print ("Agent " + str(self.agent[0]))



            else:
                self.add_rand_int_to_inital_position(width+depth)
                if self.valid_positions[self.agent[0][1], self.agent[0][2]]==False:
                    height = self.get_height_init()
                    matrix = np.logical_and(self.people_predicted[0] > 0, self.valid_positions)
                    limits = [self.reconstruction.shape[1] // 4, self.reconstruction.shape[1] * 3 // 4,
                              self.reconstruction.shape[2] // 4, self.reconstruction.shape[2] * 3 // 4]
                    car_traj = np.where(matrix[limits[0]:limits[1], limits[2]:limits[3]])
                    if len(car_traj)>0 and len(car_traj[0])>1:
                        pos = random.randint(0, len(car_traj[0])-1)
                        self.agent[0] = np.array([height, car_traj[0][pos], car_traj[1][pos]]).astype(int)
                    else:
                        print("No people")
                        return [], -1, self.vel_init

        self.agent[0]=self.agent[0].astype(int)
        if self.follow_goal:
            self.set_goal()

        #print(("Starting at " + str(self.agent[0]) + " goal is : " + str(self.goal)+" "+str(self.goal_person_id)+" "+str(self.goal_person_id_val)))
        return self.agent[0], self.goal_person_id, self.vel_init

    def initialize_randomly(self):
        height = self.get_height_init()# Set limit so that random spot is chosen in the middle of the scene.
        limits=[self.reconstruction.shape[1] // 4, self.reconstruction.shape[1]*3 // 4, self.reconstruction.shape[2] // 4, self.reconstruction.shape[2]*3 // 4]

        # Find valid places
        valid_places = np.where(self.valid_positions[limits[0]:limits[1], limits[2]:limits[3]])
        #print ("Valid places "+str(valid_places))
        if len(valid_places)>0:
            # Select spot to set agent.
            pos = random.randint(0, len(valid_places[0])-1)
            #print ("Random pos "+str(pos))
            self.agent[0] = np.array([height, limits[0]+valid_places[0][pos], limits[2]+valid_places[1][pos]])

            # Select goal position.
            pos_goal = random.randint(0, len(valid_places[0])-1)

            self.agent[0] = self.agent[0].astype(int)
            if self.follow_goal:
                self.goal = np.array(
                    [height, limits[0] + valid_places[0][pos_goal], limits[2] + valid_places[1][pos_goal]])
                self.set_goal()

            #print ("Valid position " + str(self.valid_positions[self.agent[0][1],self.agent[0][2]]))

            return self.agent[0], -1, self.vel_init
        print("No valid places")
        return [], -1, self.vel_init

    def get_height_init(self):

        height = 0
        if len(self.valid_pavement[0]) > 1 and len(self.heights)==0:
            #indx = np.random.randint(len(self.valid_pavement[0]) - 1)
            height = np.mean(self.valid_pavement[0])+self.agent_height#self.valid_pavement[0][indx]
        else:
            height=np.mean(self.heights)
        if np.isnan(height) or height<0:
            height=0
        elif height>32:
            height=31

        return height

    def mark_out_car_cv_trajectory(self, prior, car_bbox, car_goal, car_vel, time=-1):
        #print (" Before marking out cv trajectory "+ str(np.sum(prior==0))+ "  time "+str(time)+" vel "+str(car_vel[1:]))
        if np.linalg.norm(car_vel[1:])>1e-4:
            car_vel_unit=car_vel[1:]*(1/np.linalg.norm(car_vel[1:]))

            limits = np.array([car_bbox[2]-self.agent_size[1]+car_vel_unit[0]-1, car_bbox[3]+self.agent_size[1]+car_vel_unit[0]+2, car_bbox[4]-self.agent_size[2]+car_vel_unit[1]-1,
                      car_bbox[5]+ self.agent_size[2]+car_vel_unit[1]+2])


            car_pos=np.array([np.mean(limits[0:1]), np.mean(limits[2:])])


            for dim in range(len(limits)):
                limits[dim] = int(min(max(limits[dim], 0), prior.shape[dim // 2] - 1))


            while np.linalg.norm(car_pos-car_goal[1:])> np.sqrt(2) and limits[0]<limits[1] and limits[2]< limits[3]:
                #print (" In whilw loop ")
                limits_int=limits.astype(int)
                if time< 0:

                    prior[limits_int[0]: limits_int[1], limits_int[2]: limits_int[3]]=np.zeros_like(prior[limits_int[0]: limits_int[1], limits_int[2]: limits_int[3]])
                    #print (" Prior limits " + str(limits_int)+ (np.sum(np.abs(prior[limits_int[0]: limits_int[1], limits_int[2]: limits_int[3]]))))
                else:
                    prior[limits_int[0]: limits_int[1], limits_int[2]: limits_int[3]] = time*np.ones_like(
                        prior[limits_int[0]: limits_int[1], limits_int[2]: limits_int[3]])
                    time=time+1

                limits = np.array([limits[0]+car_vel_unit[0], limits[1]+car_vel_unit[0], limits[2]+car_vel_unit[1],limits[3] +car_vel_unit[1]])
                car_pos = [np.mean(limits[0:1]), np.mean(limits[2:])]

                for dim in range(len(limits)):
                    limits[dim] = min(max(limits[dim], 0), prior.shape[dim // 2] - 1)
        #print (" After marking out cv trajectory " + str(np.sum(prior == 0)))

        return prior
    # Car vel is voxels per second
    def calculate_prior(self, car_pos, car_vel, car_max_dim=0, car_min_dim=0):
        if self.use_occlusions:
            self.calculate_occlusions(car_pos, car_vel, car_max_dim, car_min_dim)
            self.prior = self.occlusions.copy() * self.valid_positions.copy()  # *self.heatmap
        else:
            self.prior =  self.valid_positions.copy()
        if not self.useRealTimeEnv:
            self.prior[self.cars_predicted[0] > 0]=0
        else:
            self.prior=self.mark_out_car_cv_trajectory(self.prior, self.car_bbox[0], self.car_goal,  self.car_dir)


        car_speed=np.linalg.norm(car_vel)*.2
        agent_dim = np.sqrt((self.agent_size[1] + 1) ** 2 + ((self.agent_size[2] + 1) ** 2)) * .2

        # print ("Car speed "+str(car_speed))
        # check if car is moving faster than 1km/h
        if car_speed>0.081:
            alpha=np.arctan2(car_vel[1], car_vel[0]) # Car's direction of movement

            max_ped_speed=3*5 # Pedestrian's maximal speep 3m/s in voxels
            beta=np.arctan(max_ped_speed/car_speed)# The maximal difference in angle between the pedestrian and the car's direction of movement leading to a crash.

            car_vel_unit=car_vel*(1/np.linalg.norm(car_vel))

            # Distance to the front and back ends of the car.
            car_pos_back=car_pos-(car_vel_unit*car_max_dim*.5)
            car_pos_front = car_pos + ( car_vel_unit * car_max_dim * .5)

            # Find among the angles which results in an upper and which in a lower constraint.
            constraints=[np.tan(alpha+beta),np.tan(alpha-beta) ]
            multiplier_of_upper_constraint=max(constraints)
            multiplier_of_lower_constraint=min(constraints)

            for x in range(self.reconstruction_2D.shape[0]):
                for y in range(self.reconstruction_2D.shape[1]):
                    # Displacement from current position [x,y] to car's back end
                    displacement_from_car=np.array([x,y])-car_pos_back
                    # Displacement from current position [x,y] to car's front
                    displacement_to_car_front = np.array([x, y]) - car_pos_front
                    # Distance to car's front
                    distance_to_car = np.linalg.norm(displacement_to_car_front) * .2 - agent_dim
                    # Check if the point [x, y] is in front of the car.
                    point_in_front_of_car=np.dot(displacement_from_car, car_vel_unit)

                    # If the point is on front of the car and lies between the upper and lower constraint from above then it shoudl receive a higher weight.
                    if point_in_front_of_car>0 and (displacement_from_car[1]<multiplier_of_upper_constraint*displacement_from_car[0]+(car_min_dim*.5) or displacement_from_car[1]>multiplier_of_lower_constraint*displacement_from_car[0]-(car_min_dim*.5)): # Inside cone of possible initializations
                        # If [x,y] is within braking distance on dry road, then don't initialize here.
                        #print(" distance to car "+ str(distance_to_car)+" car_speed "+str(car_speed)+" test "+str(distance_to_car<(car_speed**2/(250*0.8))))
                        if distance_to_car<(car_speed**2/(250*0.8)):
                            self.prior[x,y]=0
                        else:
                            self.prior[x, y] =self.prior[x, y]/distance_to_car
                    else:
                        if distance_to_car<agent_dim+(car_max_dim*.5):
                            self.prior[x,y]=0
                        else:
                            self.prior[x, y] =self.prior[x, y] / (distance_to_car**2)

        else:
            #print (" otherwise ")
            for x in range(self.reconstruction_2D.shape[0]):
                for y in range(self.reconstruction_2D.shape[1]):
                    displacement_to_car_front = np.array([x, y]) - car_pos
                    distance_to_car=np.linalg.norm(displacement_to_car_front)*.2- agent_dim
                    if distance_to_car < 1:
                        self.prior[x, y] = 0
                    else:
                        self.prior[x, y] =self.prior[x, y] / (distance_to_car)

        self.prior=self.prior*(1/np.sum(self.prior[:]))

        return self.prior

    # Car vel is voxels per second
    # time is in seconds, distances in voxels

    def calculate_goal_prior(self, car_pos, car_vel_extr,agent_init_pos, car_dim_x=0, car_dim_y=0):
        print("Calculate prior input "+str(car_pos)+" car vel "+str(car_vel_extr)+" agent init pos "+str(agent_init_pos) )
        car_vel=car_vel_extr#/self.frame_rate
        self.goal_prior = self.valid_positions.copy()  # *self.heatmap
        print ("Valid pos "+str(np.sum(self.goal_prior)) )

        car_speed = np.linalg.norm(car_vel) * .2
        vel_to_car=car_pos-agent_init_pos # pointing from pedestrian to car
        margin =[self.agent_size[1]+car_dim_x,self.agent_size[2]+car_dim_y ]
        print("margin " + str(margin))
        max_speed_ped = 15.0  # voxels per second
        max_time=self.seq_len*self.frame_time
        if car_speed > 0.081**2:
            print(" Calulate goal "+str(self.reconstruction_2D.shape))
            for x in range(self.reconstruction_2D.shape[0]):
                for y in range(self.reconstruction_2D.shape[1]):
                    pos=np.array([x, y])
                    vel_to_goal=pos-agent_init_pos
                    dist_to_goal = np.linalg.norm(vel_to_goal)
                    vel_to_goal=vel_to_goal*(1/dist_to_goal)


                    min_s=(car_vel[0]*vel_to_car[1]-car_vel[1]*vel_to_car[0])/(vel_to_goal[0]*vel_to_car[1]-vel_to_goal[1]*vel_to_car[0])# scaling of pedestrian velocity

                    max_s=min_s
                    max_candidates_variables, min_candidates_variables = self.get_lower_and_upper_bounds(car_vel,
                                                                                                         margin,
                                                                                                         max_s,
                                                                                                         vel_to_car,
                                                                                                         vel_to_goal)

                    range_in_s = [[], []]
                    range_in_t = [[], []]


                    for coordinate_index in range(2):

                        horizontal_translation = min_candidates_variables[coordinate_index][1]

                        lower_function_value_at_max_s, lower_function_value_at_min_s, upper_function_value_at_max_s, upper_function_value_at_min_s = self.evaluate_end_points_of_upper_and_lower_bound(
                            coordinate_index, max_candidates_variables,  min_candidates_variables, max_s, max_s)
                        range_in_t[0].append(copy.copy(lower_function_value_at_max_s))
                        range_in_t[1].append(copy.copy(upper_function_value_at_max_s))

                    max_t,  min_t = self.get_extreme_s_t(range_in_s, range_in_t)

                    if min_t<max_t and min_s >0 and min_s < max_speed_ped: #and min_s<max_s:


                        time=max(min_t, 1e-3)*max_s
                        #print(" Time "+str(time)+" max s "+str(max_s))

                        self.goal_prior[x,y]=self.goal_prior[x,y]/(time) # dist to collision

                    else:
                        #print ("alpha: "+str(alpha )+" max alpha "+str(15*inverse_dist_to_goal)+" t "+str(t)+" max time "+str( self.seq_len*self.frame_time))
                        #print ("Goal prior is 0: min t "+str(min_t)+" max "+str(max_t)+" min s "+str(min_s)+" max speed  "+str(max_speed_ped))

                        self.goal_prior[x,y] = 0
        else:
            print(" Goal same as prior ")
            self.goal_prior = copy.copy(self.prior )

        if np.sum(self.goal_prior[:])==0:
            self.goal_prior = copy.copy(self.prior)
        self.goal_prior = self.goal_prior * (1 / np.sum(self.goal_prior[:]))

        return self.goal_prior

    def get_extreme_s_t(self, range_in_s, range_in_t):

        min_t = max(range_in_t[0])
        max_t = min(range_in_t[1])
        return max_t, min_t

    def get_range_for_each_dimension(self, break_point, lower_function_value_at_max_s,
                                     lower_function_value_at_min_s, range_in_s, range_in_t,
                                     upper_function_value_at_max_s, upper_function_value_at_min_s,upper_function_value_at_break_point,
                                     x, y, condition, min_t, max_t, min_s, max_s):
        # if lower_function_value_at_max_s>0 or lower_function_value_at_min_s>0:
        #     print ("Lower function values at min "+str(lower_function_value_at_max_s)+" at max: "+str(lower_function_value_at_min_s))
        if condition:
            # Check on which side of the translation point is the upper constraint above the lower?
            if upper_function_value_at_max_s > lower_function_value_at_max_s:
                range_in_s[0].append(copy.copy(break_point))
                range_in_s[1].append(copy.copy(max_s))
                range_in_t[0].append(copy.copy(max(lower_function_value_at_max_s, min_t)))
                range_in_t[1].append(copy.copy(min(upper_function_value_at_break_point, max_t)))
            elif upper_function_value_at_min_s > lower_function_value_at_min_s:
                range_in_s[0].append(min_s)
                range_in_s[1].append(copy.copy(break_point))
                range_in_t[0].append(copy.copy(max(lower_function_value_at_min_s, min_t)))
                range_in_t[1].append(copy.copy(min(upper_function_value_at_break_point, max_t)))
            else:
                self.goal_prior[x, y] = 0
                #print ("No solution!")
                range_in_s[0].append(1)
                range_in_s[1].append(-1)
                range_in_t[0].append(1)
                range_in_t[1].append(-1)
        else:
            if upper_function_value_at_max_s < lower_function_value_at_max_s:
                self.goal_prior[x, y] = 0
                #print ("No solution!")
                range_in_s[0].append(1)
                range_in_s[1].append(-1)
                range_in_t[0].append(1)
                range_in_t[1].append(-1)
            else:
                range_in_s[0].append(copy.copy(min_s))
                range_in_s[1].append(copy.copy(max_s))
                range_in_t[0].append(max(lower_function_value_at_min_s, min_t))
                range_in_t[1].append(min(upper_function_value_at_max_s, max_t))

    def evaluate_end_points_of_upper_and_lower_bound(self, coordinate_index, max_candidates_variables,
                                                     min_candidates_variables, min_s, max_s):
        if coordinate_index>=0:
            max_variables= max_candidates_variables[coordinate_index]
            min_variables = min_candidates_variables[coordinate_index]
        else:
            max_variables = max_candidates_variables
            min_variables = min_candidates_variables
        upper_function_value_at_max_s = max_variables[0] / (
            max_s - max_variables[1])
        lower_function_value_at_max_s = min_variables[0] / (
            max_s - min_variables[1])
        upper_function_value_at_min_s = max_variables[0] / (
            min_s - max_variables[1])
        lower_function_value_at_min_s = min_variables[0] / (
            min_s - min_variables[1])
        # if min_s >0 and min_s<15:
        #     print ("Upper bounds max: "+str(upper_function_value_at_max_s)+" lower bounds: "+str(lower_function_value_at_max_s) )
        #     print ("Upper bounds min: " + str(upper_function_value_at_min_s) + " lower bounds: " + str(
        #         lower_function_value_at_min_s))
        return lower_function_value_at_max_s, lower_function_value_at_min_s, upper_function_value_at_max_s, upper_function_value_at_min_s

    def get_lower_and_upper_bounds(self, car_vel, margin, max_speed_ped, vel_to_car, vel_to_goal):
        min_candidates_variables = []
        max_candidates_variables = []
        # decide on upper and lower bounding functions.
        for i in range(2):
            condition_for_reverting_inequality = (vel_to_goal[i] * max_speed_ped) - car_vel[
                i]  # revert inequality this is negative. We divide by this!
            if condition_for_reverting_inequality > 0:
                min_candidates_variables.append([vel_to_car[i] - margin[i] / vel_to_goal[i], car_vel[i] / vel_to_goal[
                    i]])  # vertical scaling, horizontal scaling, translation
                max_candidates_variables.append(
                    [vel_to_car[i] + margin[i] / vel_to_goal[i], car_vel[i] / vel_to_goal[i]])
            else:
                min_candidates_variables.append(
                    [vel_to_car[i] + margin[i] / vel_to_goal[i], car_vel[i] / vel_to_goal[i]])
                max_candidates_variables.append(
                    [vel_to_car[i] - margin[i] / vel_to_goal[i], car_vel[i] / vel_to_goal[i]])
        return max_candidates_variables, min_candidates_variables

    def getBordersImg(self,img):
        # Discover the borders of the objects in the image by XOR between an erosion with 2x2 structure AND original image
        binstruct = ndimage.generate_binary_structure(2, 2)
        eroded_img = ndimage.binary_erosion(img, binstruct)
        edges_img = (img > 0) ^ eroded_img
        return edges_img




    # Given the pos of the car, unit velocity => occlusion img
    # BorderImg shows some of the borders of validImg
    # validImg- valid locations
    def getOcclusionImg(self, occlusion_map, bordersImg, car_pos, max_dim, validImg):
        edge_points = np.nonzero(bordersImg)

        for pos in range(len(edge_points[0])):
            edge_point = np.array([edge_points[0][pos], edge_points[1][pos]], dtype=np.float)

            # Find the direction from the observer (i.e. car) to the current location. This is the direction of the light ray.
            dir=edge_point- car_pos
            dir_length=np.linalg.norm(dir)

            # If within the observers collision diameter, then this edge is an edge indicating the valid locations
            # on the border to the observer (i.e. car) and this is not an obstacle occluding anything.

            if dir_length>max_dim:
                dir = dir / dir_length  # unit length
                # Find if we move towards the observer (-dir) we are at an invalid
                # position
                pos_towards_observer =edge_point-dir
                roundedPoint_row = int(np.round(pos_towards_observer[0]))
                roundedPoint_col = int(np.round(pos_towards_observer[1]))
                occupied_before_edge=not validImg[int(edge_point[0]), int(edge_point[0])]
                if roundedPoint_row > 0 and roundedPoint_col > 0 and \
                                roundedPoint_row < occlusion_map.shape[0] and roundedPoint_col < occlusion_map.shape[1]:
                    occupied_before_edge = occupied_before_edge or not validImg[roundedPoint_row, roundedPoint_col]

                while True:
                    edge_point += dir
                    roundedPoint_row = int(np.round(edge_point[0]))
                    roundedPoint_col = int(np.round(edge_point[1]))

                    if roundedPoint_row > 0 and roundedPoint_col > 0 and \
                                    roundedPoint_row < occlusion_map.shape[0] and roundedPoint_col < occlusion_map.shape[1]:
                        # Occluded areas are behind edges in which if we move towards the observer (-dir) we are at an invalid
                        # position, but if we move away from the observer (+dir) we are at a valid position.
                        if validImg[roundedPoint_row, roundedPoint_col] == 1 and occupied_before_edge:
                            occlusion_map[roundedPoint_row, roundedPoint_col] = 1
                    else:
                        break  # Outside of img

        return occlusion_map

    def calculate_occlusions(self, car_pos, car_vel, car_max_dim=0, car_min_dim=0):

        self.occlusions=self.valid_positions.copy()
        if not self.useRealTimeEnv:
            self.occlusions[self.cars_predicted[0] > 0]=0
        else:
            self.occlusions=self.mark_out_car_cv_trajectory(self.occlusions, self.car_bbox[0], self.car_goal,self.car_dir)
        # print ("Car pos "+str(car_pos)+" car vel "+str(car_vel))
        # print ("OCCLUSION SUM BEFORE CHANGES "+str(np.sum(self.occlusions)))
        car_speed=np.linalg.norm(car_vel)*.2
        agent_dim = np.sqrt((self.agent_size[1] + 1) ** 2 + ((self.agent_size[2] + 1) ** 2))
        max_dim=car_max_dim+agent_dim

        # Valid positions includes only positions that the agent cannot be initialized in.
        # To get non-occupied locations I add points where there is nothing but the cannot fit due to its size.
        # This is used in border calculation.
        er_mask = np.ones((self.agent_size[1] * 2 + 1, self.agent_size[2] * 2 + 1))
        occupied_positions = ndimage.binary_dilation(self.valid_positions, er_mask)

        self.occlusions= self.find_occlusions(self.occlusions, occupied_positions, car_pos, car_speed, car_vel, max_dim, self.reconstruction_2D.shape)
        self.occlusions=np.logical_and(self.occlusions ,self.valid_positions)



        return self.occlusions

    def find_occlusions(self, occlusion_map,occupied_positions,  car_pos, car_speed, car_vel, max_dim,dimensions, max_angle=np.pi / 4.0):
        # print (" Car speed "+str(car_speed))
        if car_speed > 0.081 ** 2:  # Less than 1km/h
            alpha = np.arctan2(car_vel[1], car_vel[0])

            # print (" Car vel unit "+str(car_vel_unit))
            if self.lidar_occlusion:
                borders_img = self.getBordersImg(occupied_positions)
                occlusion_map = np.zeros_like(occlusion_map)
            else:
                for x in range(dimensions[0]):
                    for y in range(dimensions[1]):
                        dir = np.array([x, y]) - car_pos  # car_pos_front
                        alpha_x = np.arctan2(dir[1], dir[0])
                        if alpha - max_angle < alpha_x and alpha_x < alpha + max_angle:  # (dir[1]<m1*dir[0]+(car_min_dim*.5) or dir[1]>m2*dir[0]-(car_min_dim*.5)): # Inside cone of possible initializations
                            occlusion_map[x, y] = 0
                            # print (str(x)+" "+str(y)+" "+str(self.occlusions[x,y])+" alpha_x "+str(alpha_x)+" alpha "+str(alpha)+" min "+str(alpha-np.pi/4.0)+" max "+str(alpha+np.pi/4.0))



                #Find borders in the area seen by the car.
                borders_img = np.logical_and(self.getBordersImg(occupied_positions), np.logical_not(occlusion_map))


            occlusion_map = self.getOcclusionImg(occlusion_map, borders_img, car_pos, max_dim, occupied_positions)

            # Remove holes in occlusion mask
            binstruct = ndimage.generate_binary_structure(2, 2)
            occlusion_map = ndimage.binary_closing(occlusion_map, binstruct)
        return occlusion_map

    def initialize_by_car(self):

        if len(self.cars[0]) > 0:
            previous_pos=list(self.agent[0])
            if len(self.cars[0]) > 1:

                indx=list(range(len(self.cars[0])))
                random.shuffle(indx)
            else:
                indx=[0]

            for car_indx in indx:
                car= self.cars[0][car_indx]

                width=(car[3]-car[2])*2
                depth=(car[5]-car[4])*2
                lims=[max(car[2]- width,0), max(car[4]- depth,0)]
                cut_out=self.valid_positions[max(car[2]- width,0):min(car[3]+ width, self.valid_positions.shape[0]),
                        max(car[4]- depth,0):min(car[5]+ depth, self.valid_positions.shape[1]) ]
                test_pos=find_border(cut_out)


                if len(test_pos)>0:
                    i=random.choice(list(range(len(test_pos))))
                    self.agent[0] =[(car[0]+car[1])/2, lims[0]+test_pos[i][0],lims[1]+ test_pos[i][1] ]
                    if self.follow_goal:
                        self.set_goal()

                    return self.agent[0], -1,self.vel_init

                #take out bounding box around the car
                if car[2]<self.reconstruction.shape[1] and car[4]<self.reconstruction.shape[2] and car[0]>=0 and car[1]>=0:

                    choices=[1,2,3,4]
                    if car[4]<0:
                        choices.remove(1)
                    if car[2]<0:
                        choices.remove(2)
                    if car[5]>=self.reconstruction.shape[2]:
                        choices.remove(3)
                    if car[3]>=self.reconstruction.shape[1]:
                        choices.remove(4)
                    if len(choices)>0:
                        choice=random.choice(choices)
                        # if self.init_on_pavement>0:
                        #     x=np.random.randint(self.init_on_pavement)+self.agent_size[1]+1
                        #     y=np.random.randint(self.init_on_pavement) + self.agent_size[2]+1
                        # else:
                        x = self.agent_size[1] + 1
                        y = self.agent_size[2] + 1
                        if choice==1:
                            self.agent[0]=[self.agent[0][0], car[2] +x, car[4] -y ]
                        if choice==2:
                            self.agent[0]=[self.agent[0][0], car[2] -x, car[4] +y ]
                        if choice == 3:
                            self.agent[0] = [self.agent[0][0], car[2]+ x, car[5] + y]
                        if choice == 4:
                            self.agent[0] = [self.agent[0][0], car[3] + x, car[4] + y]
                    # print("take out bounding box around the car")
                if self.intercept_car(0, all_frames=False):
                    # print("Agent intercepts car ")
                    height = self.get_height_init()
                    matrix = np.logical_and(self.cars_predicted[0], self.valid_positions)
                    limits = [self.reconstruction.shape[1] // 4, self.reconstruction.shape[1] * 3 // 4,
                              self.reconstruction.shape[2] // 4, self.reconstruction.shape[2] * 3 // 4]
                    car_traj = np.where(matrix[limits[0]:limits[1], limits[2]:limits[3]])
                    if len(car_traj) > 0 and len(car_traj[0]) > 1:
                        pos = random.randint(0, len(car_traj[0]) - 1)
                        self.agent[0] = np.array([height, car_traj[0][pos], car_traj[1][pos]]).astype(int)
                    else:
                        print("No cars")

                        return [], -1, self.vel_init
            self.agent[0]=np.array(self.agent[0]).astype(int)
            if self.follow_goal:
                self.set_goal()

        else:
            # print("Final initialization")
            height = self.get_height_init()
            matrix=[]
            if self.useRealTimeEnv:
                cars_predicted_init = self.mark_out_car_cv_trajectory(self.cars_predicted_init.copy(), self.car_bbox[0],
                                                                           self.car_goal, self.car_dir, time=1)

                matrix = np.logical_and(cars_predicted_init> 0, self.valid_positions)
            else:
                matrix=np.logical_and(self.cars_predicted[0] > 0, self.valid_positions)
            limits = [self.reconstruction.shape[1] // 4, self.reconstruction.shape[1] * 3 // 4,
                      self.reconstruction.shape[2] // 4, self.reconstruction.shape[2] * 3 // 4]
            car_traj=np.where(matrix[limits[0]:limits[1], limits[2]:limits[3]])
            if len(car_traj) > 0 and len(car_traj[0]) > 1:
                pos = random.randint(0, len(car_traj[0])-1)
                self.agent[0] = np.array([height, car_traj[0][pos], car_traj[1][pos]]).astype(int)
            else:
                print("No cars")
                return [], -1, self.vel_init
        if self.follow_goal:
            self.set_goal()

        return self.agent[0], -1, self.vel_init

    def initialize_by_car_dict(self):
        print("Initialize by car dict ")
        if len(self.init_cars) > 1:
            indx=self.init_cars
            random.shuffle(indx)
        else:
            indx=self.init_cars
        agent_width = max(self.agent_size[1:]) + 1
        for car_indx in indx:
            if len(self.cars_dict[car_indx])>1:
                car= self.cars_dict[car_indx][0]
                width = min([car[3] - car[2], car[5] - car[4]])
                height = max([car[3] - car[2], car[5] - car[4]])
                max_step=1
                if self.max_step!=np.sqrt(2):
                    max_step=self.max_step
                timestep = int(np.ceil((width/2.0 + agent_width) / (max_step )))

                car_pos=np.array([np.mean(car[0:2]),np.mean(car[2:4]), np.mean(car[4:])])
                if not self.useRealTimeEnv:
                    car_next= self.cars_dict[car_indx][min(len(self.cars_dict[car_indx])-1, timestep)]
                    car_pos_next = np.array([np.mean(car[0:2]),np.mean(car_next[2:4]), np.mean(car_next[4:])])
                    vel = car_pos_next - car_pos
                else:
                    vel = self.car_vel_dict[0][car_indx]*timestep
                    car_pos_next=car_pos+vel

                vel[0] = 0
                vel = vel * (1 / np.linalg.norm(vel))


                self.agent[0] = vel * np.ceil(height / 2.0 + max(self.agent_size[1:]) + 1) + car_pos_next
                self.agent[0] = np.array(self.agent[0]).astype(int)
                if self.follow_goal:
                    self.set_goal()
                return self.agent[0], -1, self.vel_init

        self.agent[0]=np.array(self.agent[0]).astype(int)
        if self.follow_goal:
            print( "Set goal")
            self.set_goal()
        return self.agent[0], -1, self.vel_init


    def   initialize_by_ped_dict(self):
        print(("Initialize by pedestrian "+ str(self.valid_keys)))
        if len(self.valid_keys) > 1:
            valid_keys=self.valid_keys
            random.shuffle(valid_keys)
        else:
            valid_keys=self.valid_keys
        agent_width = max(self.agent_size[1:]) + 1
        for pedestrian_indx in valid_keys:
            if len(self.people_dict[pedestrian_indx])>1:

                person= self.people_dict[pedestrian_indx][0]
                car_pos=np.mean(person, axis=1)
                #width = min([car[1][1] - car[1][0], car[2][1] - car[2][0]])
                height=max([person[1][1] - person[1][0], person[2][1] - person[2][0]])
                max_step = 1
                if self.max_step != np.sqrt(2):
                    max_step = self.max_step
                timestep=int(np.ceil((height/2.0+agent_width)/max_step))
                if not self.useRealTimeEnv:
                    person_next_pos = self.people_dict[pedestrian_indx][min(len(self.people_dict[pedestrian_indx])-1, timestep)]
                    car_pos_next = np.mean(person_next_pos, axis=1)
                    vel = car_pos_next - car_pos
                else:
                    vel=self.people_vel_dict[0][pedestrian_indx]*timestep
                    car_pos_next=  car_pos+ vel
                vel[0]=0
                vel=vel*(1/np.linalg.norm(vel))

                self.agent[0]=vel*np.ceil(height/2.0+max(self.agent_size[1:])+1)+car_pos_next
                self.agent[0] = np.array(self.agent[0]).astype(int)
                if self.follow_goal:
                    self.set_goal()
                return self.agent[0], -1, self.vel_init

        self.agent[0]=np.array(self.agent[0]).astype(int)
        if self.follow_goal:
            self.set_goal()
        return self.agent[0], -1, self.vel_init

    def initialize_near_obstacles(self):
        print("Initialize by obstacle")
        height = self.get_height_init()
        if len(self.border) > 0:
            i = random.choice(list(range(len(self.border))))
            self.agent[0]=np.array([height,  self.border[i][0],self.border[i][1]])
        if self.follow_goal:
            self.set_goal()
        return self.agent[0], -1, self.vel_init

    def initialize_on_pedestrian(self, on_pedestrian=True):
        #self.add_rand_int_to_inital_position()
        if len(self.valid_people)>0:
            if len(self.valid_people)>1:
                i = np.random.randint(len(self.valid_people) - 1)
            else:
                i=0
            #self.goal_person_id =i
            person = self.valid_people[i]
            self.agent[0] = np.mean(person, axis=1).astype(int)
        else:
            return [],-1, self.vel_init
            # j = np.random.randint(len(self.valid_people)-1)
            # while j==i:
            #     j = np.random.randint(len(self.valid_people)-1)
            #self.goal = np.mean(self.valid_people[j], axis=1).astype(int)
        if on_pedestrian==False:

            width = person[1][1] - person[1][0]
            depth = person[2][1] - person[2][0]
            lims = [max(person[1][0] - width, 0), max(person[2][0] - depth, 0)]
            cut_out = self.valid_positions[
                      max(person[1][0] - width, 0):min(person[1][1] + width, self.valid_positions.shape[0]),
                      max(person[2][0] - depth, 0):min(person[2][1] + depth, self.valid_positions.shape[1])]
            test_pos = find_border(cut_out)


            if len(test_pos)>0:
                if len(test_pos) > 1:
                    i = np.random.randint(len(test_pos)-1)
                else:
                    i=0
                self.agent[0] =np.array( [self.agent[0][0], lims[0] + test_pos[i][0], lims[1] + test_pos[i][1]])
            else:

                self.add_rand_int_to_inital_position(5)
                if self.valid_positions[self.agent[0][1], self.agent[0][2]] == False:
                    height = self.get_height_init()
                    matrix = np.logical_and(np.sum(self.reconstruction[:, :, :, 5], axis=0) > 0, self.valid_positions)
                    limits = [self.reconstruction.shape[1] // 4, self.reconstruction.shape[1] * 3 // 4,
                              self.reconstruction.shape[2] // 4, self.reconstruction.shape[2] * 3 // 4]
                    car_traj = np.where(matrix[limits[0]:limits[1], limits[2]:limits[3]])
                    if len(car_traj)>0 and len(car_traj[0])>1:
                        pos = random.randint(0, len(car_traj[0])-1)
                        self.agent[0] = np.array([height, car_traj[0][pos], car_traj[1][pos]]).astype(int)
                    else:
                        print("No people")
                        return [], -1, self.vel_init
        self.agent[0] = self.agent[0].astype(int)
        if self.follow_goal:
            self.set_goal()

        return self.agent[0],i, self.vel_init



    def initialize_on_pavement(self):
        if len(self.valid_pavement[0])==0:
            return [], -1, self.vel_init
        if len(self.valid_pavement[0])==1:
            indx=0
        else:
            indx = np.random.randint(len(self.valid_pavement[0])-1)

        pos_ground = [self.valid_pavement[0][indx], self.valid_pavement[1][indx],
                      self.valid_pavement[2][indx]]  # Check this!
        self.agent[0]=np.array(pos_ground).astype(int)

        if self.follow_goal:
            if self.useRealTimeEnv:
                goal_loc=[]
                for i in range(len(self.valid_pavement[0])):
                    if np.linalg.norm(np.array([self.valid_pavement[0][i],self.valid_pavement[1][i], self.valid_pavement[2][i]])-pos_ground)>200:
                        goal_loc.append(np.array([self.valid_pavement[0][i],self.valid_pavement[1][i], self.valid_pavement[2][i]]).astype(int))

                if len(goal_loc)==0:
                    self.set_goal()
                elif len(goal_loc)==1:
                    self.goal=goal_loc[0]
                else:
                    j=np.random.randint(len(goal_loc))
                    self.goal = goal_loc[j]
                print(" Goal "+str(self.goal))
            else:
                self.set_goal()
        self.agent[0] = self.agent[0].astype(int)
        return self.agent[0],-1, self.vel_init

    def set_goal(self):
        dist =int(min(self.seq_len *0.44, self.threshold_dist))
        limits = [int(max(self.agent[0][1] - dist, 0)), int(min(self.agent[0][1] + dist, self.reconstruction.shape[1])),
                  int(max(self.agent[0][2] - dist, 0)), int(min(self.agent[0][2] + dist, self.reconstruction.shape[2]))]

        valid_places = np.where(self.valid_positions[limits[0]:limits[1], limits[2]:limits[3]])  # ,
        if self.evaluation:
            correct_places=[[],[],[]]
            if len(valid_places[0])>0:
                for i in range(0,len(valid_places[0])):
                    #print str(i)+" "+str(valid_places[0][i])
                    if np.linalg.norm(np.array([valid_places[0][i], valid_places[1][i]]))>dist/2.0:
                        correct_places[0].append(valid_places[0][i])
                        correct_places[1].append(valid_places[1][i])

                if len(correct_places[0])>0:
                    valid_places=correct_places

        if len(valid_places[0]) > 0:
            # Select spot to set goal.
            if len(valid_places[0]) ==1:
                pos = 0
            else:
                #print "Valid places len "+str(len(valid_places[0]) )
                pos = random.randint(0, len(valid_places[0]) - 1)

            self.goal = np.array([self.agent[0][0], limits[0] + valid_places[0][pos], limits[2] + valid_places[1][pos]])


            if self.velocity_agent:
                self.speed_init=1
                if np.linalg.norm(self.vel_init) > 1e-5:
                    self.speed_init=np.linalg.norm(self.vel_init)
                    print ("Initial speed "+str(self.speed_init))
                self.goal_time=min(self.seq_len-1, np.linalg.norm(self.goal-self.agent[0])/self.speed_init)
                print ("Goal time " + str(self.goal_time))

            #     #mus=[1.4,2.1,2.5]
            #     i=np.random.rand(1)*2+0.7
            #     self.speed_init=i[0]*self.max_step/3.0
            #     if np.linalg.norm(self.vel_init)>1e-5:
            #         self.vel_init=self.vel_init/np.linalg.norm(self.vel_init)*self.speed_init
            #
            #     # print self.seq_len-1
            #     # print "Dist "+str(np.linalg.norm(self.goal-self.agent[0]))+" goal"+str(self.goal)+" start "+str(self.agent[0])
            #     self.goal_time=min(self.seq_len-1, np.linalg.norm(self.goal-self.agent[0])/self.speed_init)
            # print(("Set goal:Starting at "+str(self.agent[0])+" goal is : "+str(self.goal)+" dist "+str(np.linalg.norm(self.agent[0]-self.goal))+" goal time: "+str(self.goal_time)))


        else:
            print("No goal")

    def add_rand_int_to_inital_position(self, steps=-1):
        sign = 1
        if np.random.rand(1) > 0.5:
            sign = -1
        if steps<0:
            steps=5#self.init_on_pavement
        sz = self.agent[0].shape
        if steps > 0:
            changed=False
            tries=0
            while not changed:
                rand_init = np.random.randint(max(1, steps), size=sz[0])
                tmp=self.agent[0] + rand_init
                if tmp.all()>0 and tmp[0]<self.reconstruction.shape[0]and tmp[1]<self.reconstruction.shape[1]and tmp[2]<self.reconstruction.shape[2] and self.valid_positions[tmp[1], tmp[2]]==True:
                    self.agent[0][1:2] = self.agent[0][1:2] + rand_init[1:2]
                    changed=True
                if tries>5:
                    changed = True
                    print(("Could not add random init "+str(self.agent[0])))
                tries=tries+1

    def find_sidewalk(self, init):
        segmentation = (self.reconstruction[:, :, :, CHANNELS.semantic] * NUM_SEM_CLASSES).astype(np.int)
        pavement_map = np.zeros_like(segmentation)
        for label in SIDEWALK_LABELS:
            pavement_map =np.logical_or( pavement_map,segmentation == label)
        pavement=np.where(pavement_map)
        if init:
            pos_init=[[],[],[]]
            for indx, _ in enumerate(pavement[0]):
                if self.valid_positions[pavement[1][indx], pavement[2][indx]]:#, no_height=False):
                    pos_init[0].append(pavement[0][indx])
                    pos_init[1].append(pavement[1][indx])
                    pos_init[2].append(pavement[2][indx])
            return pos_init

        return pavement



    def iou_sidewalk(self, frame_in, no_height=True):
        segmentation = self.person_bounding_box(frame_in, no_height=True)

        # 6-ground, 7-road, 8- sidewalk, 9-paring, 10- rail track
        # area = 0
        # for val in [8]:#self.pavement:
        #    area = max(np.sum(segmentation == val), area)
        area=0#(segmentation == 8).sum()
        for label in SIDEWALK_LABELS:
            area += np.sum(segmentation == label)
        return area * 1.0 / ((self.agent_size[0] * 2.0 + 1)*(self.agent_size[1] * 2.0 + 1)*(self.agent_size[2] * 2.0 + 1) )

    # def get_goal_dist_reached(self):
    #     if self.goal_person_id>0:
    #         for pos in self.agent:
    #             for i in range(self.goal_dist_reached, len(self.valid_people_tracks[self.goal_person_id])-1):
    #                 if (pos == np.mean(self.valid_people_tracks[self.goal_person_id][i], axis=1).astype(int)).all():
    #                     self.goal_dist_reached=int(i)
    #
    #
    #     return self.goal_dist_reached


    def calculate_reward(self, frame, person_id=-1, episode_done=False, supervised=False, print_reward_components=False):

        if not self.people_dict or self.init_method!=PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian:
            person_id=-1
        prev_mul_reward=False
        self.time_dependent_evaluation(frame, person_id)  # Agent on Pavement?
        self.evaluate_measures(frame, episode_done, person_id)
        self.measures_into_future(episode_done, frame)
        if self.multiplicative_reward and not prev_mul_reward:
            self.reward[frame] = 1
        else:
            self.reward[frame]=0
        reward_activated=False
        # if supervised and  self.goal_person_id>0 and frame+1<len(self.valid_people_tracks[self.goal_person_id_val]):
        #     self.measures[frame, 9]=np.linalg.norm(np.mean(self.valid_people_tracks[self.goal_person_id_val][frame+1]-self.valid_people_tracks[self.goal_person_id_val][frame], axis=1)-self.velocity[frame])
        #     self.reward[frame]=-self.measures[frame, 9]
        #     self.agent[frame+1]=np.mean(self.valid_people_tracks[self.goal_person_id_val][frame+1], axis=1)
        #     return self.reward[frame]

        # Penalty for large change in poses
        # itr = self.agent_pose_frames[frame]
        # itr_prev = self.agent_pose_frames[max(frame - 1, 0)]
        # previous_pose = self.agent_pose[itr_prev, :]
        # current_pose = self.agent_pose[itr, :]
        # diff = current_pose - previous_pose
        #self.reward[frame] += self.reward_weights[14]*max(0, np.max(diff)-25.0)
        if self.reward_weights[14]!=0:
            evaluated_term=min(max(0, abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_pose])-1.2), 2.0)
            if self.multiplicative_reward:
                if abs(evaluated_term)>1e-8:
                    multiplicative_term=0
                    if self.reward_weights[PEDESTRIAN_REWARD_INDX.large_change_in_pose]>0:
                        multiplicative_term= self.reward_weights[PEDESTRIAN_REWARD_INDX.large_change_in_pose]*evaluated_term
                        reward_activated=True
                    elif self.reward_weights[PEDESTRIAN_REWARD_INDX.large_change_in_pose]<0:
                        multiplicative_term= -self.reward_weights[PEDESTRIAN_REWARD_INDX.large_change_in_pose]/evaluated_term
                        reward_activated=True
                    if abs(multiplicative_term)>0:
                        if print_reward_components:
                            print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                (multiplicative_term + 1)) + " times R " + str(
                                self.reward[frame] * (multiplicative_term + 1)))
                        if prev_mul_reward:
                            self.reward[frame] *= multiplicative_term
                        else:
                            self.reward[frame]*=(multiplicative_term+1)
            else:

                self.reward[frame] +=self.reward_weights[PEDESTRIAN_REWARD_INDX.large_change_in_pose]*min(max(0, abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_pose])-1.2), 2.0)
            if print_reward_components:
                print ("Pose reward "+str(self.reward_weights[PEDESTRIAN_REWARD_INDX.large_change_in_pose]*min(max(0, abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_pose])-1.2), 2.0))+" reward "+str(self.reward[frame]))



        # Penalty for hitting pedestrians
        if self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian]!=0 and abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians])>1e-8:
            if self.multiplicative_reward:
                reward_activated=True
                multiplicative_term = 0
                if self.reward_weights[7]>0:
                    multiplicative_term= self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]
                else:
                    multiplicative_term= -self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian]/ self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]
                if abs(multiplicative_term) > 0:
                    if print_reward_components:
                        print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                            (multiplicative_term + 1)) + " times R " + str(
                            self.reward[frame] * (multiplicative_term + 1)))

                    if prev_mul_reward:
                        self.reward[frame] *= multiplicative_term
                    else:
                        self.reward[frame] *= (multiplicative_term + 1)

            else:
                self.reward[frame] += self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]
            if print_reward_components:
                print ("Penalty for hitting pedestrians " + str(self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians])+" reward "+str(self.reward[frame]))

        if self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]==0:
            if print_reward_components:
                print(" Reward has not hit pedestrians")
            # Reward for on pedestrian trajectory
            if frame==0 or self.measures[frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init]>self.measures[frame-1, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init]:
                if self.reward_weights[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] != 0 and abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory]) > 1e-8:

                    if self.multiplicative_reward:
                        reward_activated=True
                        multiplicative_term = 0
                        if self.reward_weights[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] > 0:
                            multiplicative_term= self.reward_weights[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory]
                        else:
                            multiplicative_term= -self.reward_weights[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] / self.measures[frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory]
                        if abs(multiplicative_term) > 0:
                            if print_reward_components:
                                print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                    (multiplicative_term + 1)) + " times R " + str(
                                    self.reward[frame] * (multiplicative_term + 1)))

                            if prev_mul_reward:
                                self.reward[frame] *= multiplicative_term
                            else:
                                self.reward[frame] *= (multiplicative_term + 1)
                    else:
                        self.reward[frame] += self.reward_weights[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory]
                    if print_reward_components:
                        print ("Add reward for on traj "+str(self.reward_weights[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory])+" reward "+str(self.reward[frame]))
                if print_reward_components:
                    print("Weight "+str(self.reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] )+" size "+str(abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]) > 1e-8)+ " value "+str(self.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]) )
                if self.reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] > 0:
                    if self.reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] != 0 and abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]) > 1e-8:
                        if self.multiplicative_reward:
                            reward_activated=True
                            multiplicative_term = 0
                            if self.reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] > 0:
                                multiplicative_term= self.reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]
                            else:
                                multiplicative_term= -self.reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] / self.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]
                            if abs(multiplicative_term) > 0:
                                if print_reward_components:
                                    print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                        (multiplicative_term + 1)) + " times R " + str(
                                        self.reward[frame] * (multiplicative_term + 1)))

                                if prev_mul_reward:
                                    self.reward[frame] *= multiplicative_term
                                else:
                                    self.reward[frame] *= (multiplicative_term + 1)
                        else:
                            self.reward[frame] += self.reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]
                    if print_reward_components:
                        print("Add reward for ped heatmap " + str(
                            self.reward_weights[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] * self.measures[
                                frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap])+" reward "+str(self.reward[frame]))

            # Reward for on sidewalk
            if self.reward_weights[PEDESTRIAN_REWARD_INDX.on_pavement] > 0:
                if self.multiplicative_reward:
                    if self.reward_weights[PEDESTRIAN_REWARD_INDX.on_pavement] != 0 and abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.iou_pavement]) > 1e-8:
                        reward_activated=True
                        multiplicative_term = 0
                        if self.reward_weights[PEDESTRIAN_REWARD_INDX.on_pavement] > 0:
                            multiplicative_term= self.reward_weights[PEDESTRIAN_REWARD_INDX.on_pavement] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.iou_pavement]
                        else:
                            multiplicative_term= -self.reward_weights[PEDESTRIAN_REWARD_INDX.on_pavement] /self.measures[frame, PEDESTRIAN_MEASURES_INDX.iou_pavement]
                        if abs(multiplicative_term) > 0:
                            if print_reward_components:
                                print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                    (multiplicative_term + 1)) + " times R " + str(
                                    self.reward[frame] * (multiplicative_term + 1)))

                            if prev_mul_reward:
                                self.reward[frame] *= multiplicative_term
                            else:
                                self.reward[frame] *= (multiplicative_term + 1)
                else:
                    self.reward[frame] += self.reward_weights[PEDESTRIAN_REWARD_INDX.on_pavement] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.iou_pavement]
                if print_reward_components:
                    print ("Add reward for sidewalk " + str(
                self.reward_weights[PEDESTRIAN_REWARD_INDX.on_pavement] * self.measures[
                        frame, PEDESTRIAN_MEASURES_INDX.iou_pavement])+" reward "+str(self.reward[frame]))
        # Penalty for hitting objects
        if self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles] > 0:
            if self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_objects] != 0 and abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles]) > 1e-8:
                if self.multiplicative_reward:
                    reward_activated=True
                    multiplicative_term = 0
                    if self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_objects] > 0:
                        multiplicative_term= self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_objects] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles]
                    else:
                        multiplicative_term= -self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_objects] / self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles]
                        # print ("Add reward for hitting objs " + str(self.reward_weights[3] * self.measures[frame, 3]))
                    if abs(multiplicative_term) > 0:
                        if print_reward_components:
                            print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                (multiplicative_term + 1)) + " times R " + str(
                                self.reward[frame] * (multiplicative_term + 1)))

                        if prev_mul_reward:
                            self.reward[frame] *= multiplicative_term
                        else:
                            self.reward[frame] *= (multiplicative_term + 1)
                else:
                    self.reward[frame] += self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_objects] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles]
                if print_reward_components:
                    print ("Add reward for hitting objs " + str(self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_objects] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles])+" reward "+str(self.reward[frame]))

        # Distance travelled:

        if np.linalg.norm(self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled])>0:
            # Regular positive reward for distance travelled
            if print_reward_components:
                print("Frame "+str(frame )+" equal "+str(self.seq_len - 2))
            if frame == (self.seq_len - 2):
                if print_reward_components:
                    print("Pavement  " + str(self.measures[frame, PEDESTRIAN_MEASURES_INDX.iou_pavement]) + " out of axis " + str(self.out_of_axis(frame + 1) )+" hit by car "+str(self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]))
                if self.measures[frame, PEDESTRIAN_MEASURES_INDX.iou_pavement] > 0 and not self.out_of_axis(frame + 1) and self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car] == 0:
                    if self.multiplicative_reward:

                        if self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled] != 0 and abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init]) > 1e-8:
                            reward_activated = True
                            multiplicative_term = 0
                            if self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled] > 0:
                                multiplicative_term= self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init]  # /(self.seq_len*np.sqrt(2))
                            else:
                                multiplicative_term= -self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled] / self.measures[
                                    frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init]  # /(self.seq_len*np.sqrt(2))
                            if abs(multiplicative_term) > 0:
                                if print_reward_components:
                                    print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                        (multiplicative_term + 1)) + " times R " + str(
                                        self.reward[frame] * (multiplicative_term + 1)))

                                if prev_mul_reward:
                                    self.reward[frame] *= multiplicative_term
                                else:
                                    self.reward[frame] *= (multiplicative_term + 1)
                    else:
                        self.reward[frame] += self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init]
                    if print_reward_components:
                        print("Reward for dist travelled" + str(
                            self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled] * self.measures[
                                frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init]) + " reward " + str(
                            self.reward[frame]))

        # Negative reward for relative distance not travelled in flight-distance.

        if np.linalg.norm(self.reward_weights[PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] )> 0:
             local_measure=0
             if self.measures[frame,PEDESTRIAN_MEASURES_INDX.dist_to_final_pos]>0.5:
                local_measure=(self.measures[frame, PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init]/self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_final_pos]-1)
                if print_reward_components:
                    print ("Add dist travelled total " + str(self.measures[frame,  PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init]) + " distance on the fly " + str(self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_final_pos]) + " ratio " + str(1-local_measure) + " 1-ratio " + str(local_measure)+" reward "+str(self.reward_weights[PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] * local_measure)+" reward "+str(self.reward[frame]))

             else:
               local_measure=2 # penalty for standing still
               if print_reward_components:
                   print ("Add penalty for standing still " + str(self.reward_weights[ PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] * local_measure)+" reward "+str(self.reward[frame]))

             if self.multiplicative_reward:

                 if self.reward_weights[PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] != 0 and abs(local_measure) > 1e-8:
                     reward_activated = True
                     multiplicative_term = 0
                     if self.reward_weights[PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] < 0:
                         multiplicative_term= self.reward_weights[PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] *local_measure
                     else:
                         multiplicative_term= -self.reward_weights[PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] / local_measure
                     if abs(multiplicative_term) > 0:
                         if print_reward_components:
                             print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                 (multiplicative_term + 1)) + " times R " + str(
                                 self.reward[frame] * (multiplicative_term + 1)))

                         if prev_mul_reward:
                             self.reward[frame] *= multiplicative_term
                         else:
                             self.reward[frame] *= (multiplicative_term + 1)
             else:
                 self.reward[frame] += self.reward_weights[PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] * local_measure
             if print_reward_components:
                 print(" reward "+str(self.reward[frame]))
        # Agent out of axis?
        if self.reward_weights[PEDESTRIAN_REWARD_INDX.out_of_axis] != 0 and abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.out_of_axis]) > 1e-8:
            if self.multiplicative_reward:
                reward_activated = True
                multiplicative_term = 0
                if self.reward_weights[PEDESTRIAN_REWARD_INDX.out_of_axis] > 0:
                    multiplicative_term= self.reward_weights[PEDESTRIAN_REWARD_INDX.out_of_axis]*self.measures[frame, PEDESTRIAN_MEASURES_INDX.out_of_axis]
                else:
                    multiplicative_term= -self.reward_weights[PEDESTRIAN_REWARD_INDX.out_of_axis] /self.measures[frame, PEDESTRIAN_MEASURES_INDX.out_of_axis]
                if abs(multiplicative_term) > 0:
                    if print_reward_components:
                        print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                            (multiplicative_term + 1)) + " times R " + str(
                            self.reward[frame] * (multiplicative_term + 1)))

                    if prev_mul_reward:
                        self.reward[frame] *= multiplicative_term
                    else:
                        self.reward[frame] *= (multiplicative_term + 1)
            else:
                self.reward[frame] += self.reward_weights[PEDESTRIAN_REWARD_INDX.out_of_axis] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.out_of_axis]
            if print_reward_components:
                print ("Add Out of axis " + str(self.reward_weights[PEDESTRIAN_REWARD_INDX.out_of_axis] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.out_of_axis])+" reward "+str(self.reward[frame]))


        # % Distance travelled towards goal
        # If pos reward and initial distance to goal is greater than 0.
        #print "Reward before goal terms : "+str( self.reward[frame])+" initial distance to goal: "+str(np.linalg.norm(self.measures[frame, 6]))
        if (self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal]>0 or self.reward_weights[PEDESTRIAN_REWARD_INDX.reached_goal]>0)>0 :
            if self.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached ]==0:
                # print ("Frame in reward " + str(frame))
                local_measure=0
                if frame==0:
                    orig_dist=np.linalg.norm(np.array(self.goal[1:]) - self.agent[0][1:])
                    # print " Reward for one step: " + str(self.reward_weights[6] * (
                    # 1 - (self.measures[frame, 7] /orig_dist))) + " quotient: " + str(
                    #     (self.measures[frame, 7] / orig_dist)) + " dist cur" + str(
                    #     self.measures[frame, 7]) + " dist prev: " + str(orig_dist)
                    local_measure = ((orig_dist-self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal])/self.max_step)
                    if print_reward_components:
                        print (" Reward for one step to goal frame 0: " + str( self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] * local_measure) + " dist prev: " + str(orig_dist)+" cur dist  "+str( self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal])+" reward "+str(self.reward[frame]))
                    # print " Difference "+str((orig_dist-self.measures[frame, 7]))+" max_step "+str(self.max_step)+" ration "+str((orig_dist-self.measures[frame, 7])/self.max_step)
                if frame>0 and self.measures[frame-1, PEDESTRIAN_MEASURES_INDX.dist_to_goal]>0:
                    local_measure=((self.measures[frame-1, PEDESTRIAN_MEASURES_INDX.dist_to_goal] -self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal] )/self.max_step)
                    if print_reward_components:
                        print (" Reward for one step to goal : " + str(self.reward_weights[
                                                                       PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] * local_measure) + " diff: " + str((self.measures[frame-1, PEDESTRIAN_MEASURES_INDX.dist_to_goal]-self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]  )) + " dist cur" + str(self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]) + " dist prev: " + str(self.measures[frame-1, PEDESTRIAN_MEASURES_INDX.dist_to_goal])+" reward "+str(self.reward[frame]))

                    # print " Difference " + str((self.measures[frame-1, 7] - self.measures[frame, 7])) + " max_step " + str(
                    #     self.max_step) + " ration " + str((self.measures[frame-1, 7] - self.measures[frame, 7]) / self.max_step)
                if self.multiplicative_reward:
                    if print_reward_components:
                        print (" Reward for one step to goal  enter if? weigt "+str(self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal])+" local measure "+str(local_measure))
                    if self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] != 0 and abs(local_measure) > 1e-8:
                        reward_activated = True
                        multiplicative_term = 0
                        if self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] > 0:
                            multiplicative_term = self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] * local_measure
                        else:
                            multiplicative_term = -self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] / local_measure
                        if abs(multiplicative_term) > 0:

                            if print_reward_components:
                                print ("Multiplicative component " +str(multiplicative_term)+" +1 "+str((multiplicative_term + 1))+" times R "+str(self.reward[frame] *(multiplicative_term + 1)))
                            if prev_mul_reward:
                                self.reward[frame] *= multiplicative_term
                            else:
                                self.reward[frame] *= (multiplicative_term + 1)

                else:
                    self.reward[frame] += self.reward_weights[PEDESTRIAN_REWARD_INDX.distance_travelled_towards_goal] * local_measure
                if print_reward_components:
                    print(" reward " + str(self.reward[frame]))
            else:

                if self.multiplicative_reward:
                    reward_activated = True

                    self.reward[frame] *=self.reward_weights[PEDESTRIAN_REWARD_INDX.reached_goal]
                    if self.goal_time>0:
                        temp_diff= min(abs(self.goal_time-frame)/self.goal_time,1)
                        if self.reward_weights[PEDESTRIAN_REWARD_INDX.not_on_time_at_goal] != 0 and abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.difference_to_goal_time]) > 1e-8:
                            multiplicative_term = 0
                            if self.reward_weights[PEDESTRIAN_REWARD_INDX.not_on_time_at_goal] > 0:
                                multiplicative_term=self.reward_weights[PEDESTRIAN_REWARD_INDX.not_on_time_at_goal]*self.measures[frame, PEDESTRIAN_MEASURES_INDX.difference_to_goal_time]
                            else:
                                multiplicative_term= -self.reward_weights[PEDESTRIAN_REWARD_INDX.not_on_time_at_goal] / self.measures[frame, PEDESTRIAN_MEASURES_INDX.difference_to_goal_time]
                            if abs(multiplicative_term) > 0:
                                if print_reward_components:
                                    print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                        (multiplicative_term + 1)) + " times R " + str(
                                        self.reward[frame] * (multiplicative_term + 1)))

                                if prev_mul_reward:
                                    self.reward[frame] *= multiplicative_term
                                else:
                                    self.reward[frame] *= (multiplicative_term + 1)
                        # print (" Reward for reaching goal "+str(self.reward_weights[8])+" minus "+str(self.reward_weights[13]*temp_diff)+" diff "+str(temp_diff))
                else:
                    self.reward[frame] += self.reward_weights[PEDESTRIAN_REWARD_INDX.reached_goal]
                    if self.goal_time > 0:
                        temp_diff = min(abs(self.goal_time - frame) / self.goal_time, 1)

                        self.reward[frame] += self.reward_weights[PEDESTRIAN_REWARD_INDX.not_on_time_at_goal] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.difference_to_goal_time]
                if print_reward_components:
                    print("Reward for reaching goal "+str(self.reward_weights[PEDESTRIAN_REWARD_INDX.reached_goal])+" reward "+str(self.reward[frame]))
        # Following correct pedestrian -if initialized on a pedestrian. Negative reward when not follwoing.
        if self.goal_person_id>=0 and self.reward_weights[PEDESTRIAN_REWARD_INDX.one_step_prediction_error]>0:
            denominator=(frame+1)*2*self.max_step
            # print (" Following pedestrian: Denominator " + str(denominator)+" measure "+str(self.measures[frame, 9])+" frame "+str(frame))
            fraction=1-(self.measures[frame, PEDESTRIAN_MEASURES_INDX.one_step_prediction_error]/denominator)
            if self.multiplicative_reward:
                if self.reward_weights[PEDESTRIAN_REWARD_INDX.one_step_prediction_error] != 0 and abs(fraction) > 1e-8:
                    reward_activated = True
                    multiplicative_term = 0
                    if self.reward_weights[PEDESTRIAN_REWARD_INDX.one_step_prediction_error] > 0:
                        multiplicative_term= self.reward_weights[PEDESTRIAN_REWARD_INDX.one_step_prediction_error] * fraction
                    else:
                        multiplicative_term= -self.reward_weights[PEDESTRIAN_REWARD_INDX.one_step_prediction_error] / fraction
                    if abs(multiplicative_term) > 0:
                        if print_reward_components:
                            print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                (multiplicative_term + 1)) + " times R " + str(
                                self.reward[frame] * (multiplicative_term + 1)))

                        if prev_mul_reward:
                            self.reward[frame] *= multiplicative_term
                        else:
                            self.reward[frame] *= (multiplicative_term + 1)
            else:
                self.reward[frame] += self.reward_weights[PEDESTRIAN_REWARD_INDX.one_step_prediction_error] * fraction
            if print_reward_components:
                print ("Reward "+str(self.reward_weights[11] * (1 - fraction)) +" "+str( 1 - fraction)+" "+str(fraction)+" "+str(self.measures[frame, 9])+" "+str(denominator)+" reward "+str(self.reward[frame]))

        # Penalty for changing moving directions
        if self.reward_weights[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] < 0:
            if self.multiplicative_reward:
                if self.reward_weights[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] != 0 and abs(self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_direction]) > 1e-8:
                    reward_activated = True
                    multiplicative_term = 0
                    if self.reward_weights[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] > 0:
                        multiplicative_term= self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_direction] * self.reward_weights[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently]
                    else:
                        multiplicative_term=  self.reward_weights[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently]/ self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_direction]
                    if abs(multiplicative_term) > 0:
                        if print_reward_components:
                            print("Multiplicative component " + str(multiplicative_term) + " +1 " + str(
                                (multiplicative_term + 1)) + " times R " + str(
                                self.reward[frame] * (multiplicative_term + 1)))

                        if prev_mul_reward:
                            self.reward[frame] *= multiplicative_term
                        else:
                            self.reward[frame] *= (multiplicative_term + 1)
            else:
                self.reward[frame] += self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_direction] * self.reward_weights[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently]
            if print_reward_components:
                print ("Reward Changing direction "+str(self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_direction] * self.reward_weights[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently])+" reward "+str(self.reward[frame]))

        # Hitting cars?
        if self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]>0:
            if abs(self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_car_agent])>0 and  self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_hero_car]>0:
                reward_activated = True
                self.reward[frame] = self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_car_agent] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_hero_car]
                if self.multiplicative_reward and not prev_mul_reward:
                    self.reward[frame] =+1
            else:
                reward_activated = True
                self.reward[frame] = self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_car] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]
                if self.multiplicative_reward and not prev_mul_reward:
                    self.reward[frame] = +1
            if print_reward_components:

                print ("Penalty collision with car " + str( self.reward_weights[PEDESTRIAN_REWARD_INDX.collision_with_car] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car])+" reward "+str(self.reward[frame]))

        # reward reciprocal distance to car
        if self.reward_weights[PEDESTRIAN_REWARD_INDX.inverse_dist_to_car]>0:
            reward_activated = True
            self.reward[frame] = self.reward_weights[PEDESTRIAN_REWARD_INDX.inverse_dist_to_car] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car]
            if self.multiplicative_reward and not prev_mul_reward:
                self.reward[frame] = +1
            if print_reward_components:
                print ("Reward reciprocal distance to car " + str(self.reward_weights[PEDESTRIAN_REWARD_INDX.inverse_dist_to_car] * self.measures[frame, PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car])+" reward "+str(self.reward[frame]))

        # Correct reward! when hitting a car!

        #print "Hit by car? "+str(self.measures[frame, 0])
        hit_by_car=self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]>0 and frame >0 and self.measures[frame-1, PEDESTRIAN_MEASURES_INDX.hit_by_car]>0
        reached_goal=self.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached] > 0 and frame > 0 and self.measures[frame - 1, PEDESTRIAN_MEASURES_INDX.goal_reached] > 0 and self.stop_on_goal

        if hit_by_car or reached_goal:
            if print_reward_components:
                print ("Agent dead reward 0")
                print ("Reached goal " +str(self.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached])+" hit by car "+str(self.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached] ))
                if frame>0:
                    print("Reached goal prev " + str(
                        self.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached]-1) + " hit by car prev " + str(
                        self.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached]-1))

            self.reward[frame]=0

        hit_pedestrian=(self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]>0 and frame >0 and self.measures[frame-1, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]>0)

        if self.end_on_hit_by_pedestrians and hit_pedestrian:
            if print_reward_components:
                print ("Agent dead hit by pedestrians")
            self.reward[frame]=0

        if self.multiplicative_reward and not prev_mul_reward:
            if not reward_activated or self.reward[frame] ==0 :
                self.reward[frame] = 0
            elif not self.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached] > 0:
                self.reward[frame] = self.reward[frame] -1
        # else:
        #     self.reward[frame] = self.reward[frame]-0.01
        #print "Intercept car "+str(self.intercept_car(frame , all_frames=False))

        # if frame % 10==0:
        #     print "Frame "+str(frame)+" Reward: "+str(self.reward[frame])+" hit by car: "+str(self.intercept_car(frame + 1, all_frames=False))+"  Reached goal "+str(self.measures[frame, 13])+" agent "+str(self.agent[frame+1])
        return self.reward[frame]

    def evaluate_measures(self, frame, episode_done=False, person_in=-1):

        self.measures[frame, PEDESTRIAN_MEASURES_INDX.iou_pavement ] = 0
        self.measures[frame, PEDESTRIAN_MEASURES_INDX.iou_pavement ] = self.iou_sidewalk(frame + 1)

        # Intercept with objects.
        if self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles] ==0:
            self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles]=self.intercept_objects(frame + 1)

        # Distance travelled
        #print "Frame "+str(frame+1)+" init pos "+str(self.agent[0])+" Pos "+str(self.agent[frame + 1])+" Distance travelled: "+str(np.linalg.norm(self.agent[frame + 1] - self.agent[0]))
        self.measures[frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init] = np.linalg.norm(self.agent[frame + 1] - self.agent[0])


        # Out of axis & Heatmap
        current_pos = self.agent[frame + 1][1:].astype(int)
        if self.out_of_axis(frame + 1): #or self.heatmap.shape[0]<=current_pos[0] or self.heatmap.shape[0]<=current_pos[1]:
            self.measures[frame, PEDESTRIAN_MEASURES_INDX.out_of_axis] = 1  # out of axis
        elif not self.useRealTimeEnv:
            self.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]=self.heatmap[current_pos[0],current_pos[1] ]

        # Changing direction
        self.measures[frame, PEDESTRIAN_MEASURES_INDX.change_in_direction] =self.get_measure_for_zig_zagging(frame)

        if self.follow_goal: # If using a general goal

            # Distance to goal.
            self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal] = np.linalg.norm(np.array(self.goal[1:]) - self.agent[frame + 1][1:])

            #print "Distance to goal: " + str(self.measures[frame, 7]) + " goal pos: " + str(self.goal[1:]) + "current pos: " + str(self.agent[frame+1][1:])+" sqt(2) "+str(np.sqrt(2))+" "+str(self.measures[frame, 7]<=np.sqrt(2))

            # Reached goal?
            if self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal]<=np.sqrt(2):
                self.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached]=1
                #print "Reached goal"

            # if self.measures[frame, 7]==0:
            #     self.goal_dist_reached=int(self.measures[frame, 6])
            #print "Person id "+str(self.goal_person_id)

        # One- time step prediction error in supervised case. otherwise error so far.
        if self.goal_person_id>=0:
            goal_next_frame = min(frame + 1, len(self.people_dict[self.goal_person_id_val]) - 1)
            goal = np.mean(self.people_dict[self.goal_person_id_val][goal_next_frame], axis=1).astype(int)

            self.measures[frame, PEDESTRIAN_MEASURES_INDX.one_step_prediction_error] = np.linalg.norm(goal[1:] - self.agent[frame + 1][1:])
            #previous_pos=np.mean(self.people_dict[self.goal_person_id_val][goal_next_frame-1], axis=1).astype(int)
            # print "Pedestrian frame "+str(goal_next_frame)+" pos "+str(goal[1:])+" previous pos: "+str( previous_pos)
            # print "Agent "+str(frame + 1)+" pos "+str( self.agent[frame + 1][1:])+" velocity "+str(self.velocity[frame])
            # print " error: "+str(goal[1:] - self.agent[frame + 1][1:])+" Measure "+str(self.measures[frame, 9])
        if self.goal_time > 0:
            self.measures[frame, PEDESTRIAN_MEASURES_INDX.difference_to_goal_time] = min(abs(self.goal_time - frame) / self.goal_time, 1)

        #self.measures[frame, 15] = min(abs(self.goal_time - frame) / self.goal_time, 1)

    def get_measure_for_zig_zagging(self, frame):
        start = self.agent[0]  # Start position of agent
        max_penalty = 0
        cur_vel = np.array(self.velocity[max(frame, 0)])  # np.array(self.actions[int(self.action[max(frame, 0)])])
        for t in range(min(self.past_move_len, frame)):

            old_vel = np.array(self.velocity[max(frame - t - 1,
                                                 0)])  # ) #np.array(self.agent[frame])-np.array(self.agent[max(frame-3,0)])
            older_vel = np.array(self.velocity[max(frame - t - 2, 0)])  #
            if len(old_vel) > 0 and len(older_vel) > 0 and len(cur_vel) > 0:

                dif_1 = cur_vel - old_vel
                dif_2 = old_vel - older_vel
                #
                # print "Previous  " +str(t)+"  "+ str(prev_action)+" frame: "+str(frame)
                # print str(cur_vel)+" "+str(old_vel)+" "+str(older_vel)+" dif1: "+str(dif_1)+" dif2: "+str(dif_2)
                #
                # print "Change direction  "+str(np.dot(dif_1, dif_2))+" "+str(dif_1)+" "+str(dif_2)+" "
                if np.linalg.norm(old_vel) > 10 ** -6:

                    old_vel_norm = old_vel * min((1.0 / np.linalg.norm(old_vel)), 1)
                else:
                    old_vel_norm = np.zeros_like(old_vel)
                if np.linalg.norm(cur_vel) > 10 ** -6:
                    cur_vel_norm = cur_vel * min((1.0 / np.linalg.norm(cur_vel)), 1)
                else:
                    cur_vel_norm = np.zeros_like(cur_vel)

                if np.linalg.norm(cur_vel_norm - old_vel_norm) == 0 and np.linalg.norm(old_vel_norm) > 0:
                    max_penalty = max_penalty
                    # self.measures[frame, 11] =0
                elif np.linalg.norm(cur_vel_norm) > 0 and np.linalg.norm(old_vel_norm) > 0:

                    if np.dot(dif_1, dif_2) < 0:
                        cos_theta = np.dot(cur_vel_norm, old_vel_norm) / (
                        np.linalg.norm(old_vel_norm) * np.linalg.norm(cur_vel_norm))
                        # print "Cos  theta "+str(cos_theta)+" "+str(np.dot(cur_vel_norm, old_vel_norm))+" "+str((np.linalg.norm(old_vel_norm) * np.linalg.norm(cur_vel_norm)))
                        if cos_theta < 1.0:
                            max_penalty = np.sqrt((1 - cos_theta) / 2)
                            # else:
                            #     max_penalty =np.sin(np.pi/4)
        return max_penalty

    def measures_into_future(self, episode_done, frame):
        # Total travelled distance

        if episode_done:
            if frame < self.seq_len + 1:
                for f in range(len(self.agent) - 1, frame, -1):
                    self.measures[frame, PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init] += np.linalg.norm(self.agent[f] - self.agent[f - 1])

            self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_final_pos] = np.linalg.norm(self.agent[-1][1:] - self.agent[frame][1:])


    def end_of_episode_measures(self,frame):
        # Intercepting cars
        intercept_cars_in_frame=self.intercept_car(frame + 1, all_frames=False)

        self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car] = max(self.measures[max(frame - 1, 0),PEDESTRIAN_MEASURES_INDX.hit_by_car],
                                      intercept_cars_in_frame)
        #print(" End of episode measures: "+str(frame)+" hit by car in current frame? " + str(intercept_cars_in_frame) + " hit by car measure: "+str(self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]) )
        #print (" Agent evaluate end of episode measures hit by car "+str(self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]))
        if self.follow_goal:  # If using a general goal

            # Distance to goal.
            self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal] = np.linalg.norm(np.array(self.goal[1:]) - self.agent[frame + 1][1:])

            #print "Distance to goal: " + str(self.measures[frame, 7]) + " goal pos: " + str(self.goal[1:]) + "current pos: " + str(self.agent[frame][1:])

            # Reached goal?
            if self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_goal] <= np.sqrt(2):
                self.measures[frame, PEDESTRIAN_MEASURES_INDX.goal_reached] = 1
                # print "Reached goal"

                # if self.measures[frame, 7]==0:
                #     self.goal_dist_reached=int(self.measures[frame, 6])
                # print "Person id "+str(self.goal_person_id)
        per_id=-1
        if self.init_method == PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian or self.init_method == PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian:
            per_id = self.goal_person_id


        collide_pedestrians = len(self.collide_with_pedestrians(frame + 1, no_height=True, per_id=per_id)) > 0
        if collide_pedestrians:
            self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians] = 1
        if self.end_on_hit_by_pedestrians:
            self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]=max(self.measures[max(frame - 1, 0), PEDESTRIAN_MEASURES_INDX.hit_pedestrians],self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians])
        if self.useRealTimeEnv:
            self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_hero_car] = max(
                self.measures[max(frame - 1, 0), PEDESTRIAN_MEASURES_INDX.hit_by_hero_car],
                self.intercept_agent_car(frame + 1, all_frames=False))
        # print(" End of episode measures-2 : " + str(frame) + " hit by car in current frame? " + str(
        #     intercept_cars_in_frame) + " hit by car measure: " + str(
        #     self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]))

    def time_dependent_evaluation(self, frame, person_id=-1):
        # Intercepting cars - nothing needs to change
        intercept_cars_in_frame=self.intercept_car(frame + 1, all_frames=False)
        self.measures[frame,PEDESTRIAN_MEASURES_INDX.hit_by_car] = max(self.measures[max(frame - 1, 0),PEDESTRIAN_MEASURES_INDX.hit_by_car],intercept_cars_in_frame)
        #print(" Time dependent  measures: hit by car in current frame? " + str(intercept_cars_in_frame) + " hit by car measure: " + str(self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]))

        # print ("Frame "+str(frame)+" hit car: "+str(self.measures[frame, measures_indx["hit_by_car"]]))
        per_id = -1
        if len(self.people_dict) > 0:
            per_id = self.goal_person_id

        # Coincide with human trajectory
        self.measures[frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory] = 0

        # Intercepting pedestrian_trajectory_fixed!
        if self.intercept_pedestrian_trajectory(frame + 1,no_height=True):  # self.intercept_person(frame + 1, no_height=True, person_id=per_id):
            self.measures[frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory] = 1

        # Hit pedestrians
        if self.init_method == PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian or self.init_method == PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian :
            # ok assuming one frame
            if len(self.collide_with_pedestrians(frame + 1, no_height=True,per_id=per_id)) > 0:  # self.intercept_person(frame + 1, no_height=True, person_id=per_id, frame_input=frame):

                self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians] = 1
        else:
            # ok assuming one frame
            if len(self.collide_with_pedestrians(frame + 1, no_height=True)) > 0:
                self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians] = 1

        # If agent will be dead after hitting a pedestrian, then the measure i set to 1 after it has occured once.
        if self.end_on_hit_by_pedestrians:
            self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians] = max(self.measures[max(frame - 1, 0), PEDESTRIAN_MEASURES_INDX.hit_pedestrians], self.measures[frame, CAR_MEASURES_INDX.hit_pedestrians])

        # Measure distance to closest car (if using hero car distance to hero car!)
        self.measures[frame, PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car]=self.evaluate_inverse_dist_to_car(frame)#print ("Measure   " + str(self.measures[frame, 17]))

        # Evaluate heatmap!
        if  self.useRealTimeEnv:
            # convolutional solution
            # import scipy.ndimage
            # self.heatmap = scipy.ndimage.gaussian_filter(np.sum(self.reconstruction[:, :, :, CHANNELS.pedestrian_trajectory], axis=0), 15)
            # Use estimate in place of heatmap!
            #print(" Frame "+str(frame)+" len of predictions "+str(len(self.agent_prediction_people)))
            current_prediction=self.agent_prediction_people[frame]
            current_predcition_size=current_prediction.shape[0]*current_prediction.shape[1]

            self.measures[frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap] = np.sum(current_prediction!=0)/current_predcition_size

            self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_hero_car] = max(
                self.measures[max(frame - 1, 0), PEDESTRIAN_MEASURES_INDX.hit_by_hero_car],
                self.intercept_agent_car(frame + 1, all_frames=False))


    def evaluate_inverse_dist_to_car(self, frame):
        closest_car, min_dist = self.find_closest_car(frame+1, self.agent[frame+1])
        if len(closest_car) == 0:
            return 0
        elif min_dist <= self.agent_size[1]:
            return 1
        else:
            return 1.0 / copy.copy(min_dist)

    def on_pavement(self, frame_in):
        segmentation = self.person_bounding_box(frame_in, no_height=True)

        # 6-ground, 7-road, 8- sidewalk, 9-paring, 10- rail track
        for val in self.pavement:
            if np.any(segmentation == val):
                return True
        return False

    def intercept_agent_car(self, frame_in, all_frames=False, agent_frame=-1, bbox=None,car=None):
        if bbox == None:
            if agent_frame < 0:
                x_range, y_range, z_range = self.pose_range(frame_in)
            else:
                x_range, y_range, z_range = self.pose_range(agent_frame)
            overlapped = 0
            frames = []
            if all_frames or frame_in >= len(self.cars):
                frames = list(range(len(self.cars)))
            else:
                frames.append(frame_in)
            person_bbox = [x_range[0], x_range[1], y_range[0], y_range[1], z_range[0], z_range[1]]
        else:
            person_bbox = bbox


        if car == None and self.useRealTimeEnv:
            car = self.car_bbox
            # print (" Car bbox "+str(self.car_bbox))

        # print(" Cars "+str(cars))
        for frame in frames:
                # if evaluate_hero_car and len(self.car)> frame:
            # print (" Hero  car " + str(car[frame][2:6])+" person bbox "+str(person_bbox[2:]))
            if overlap(car[frame][2:6], person_bbox[2:], 1) or overlap( person_bbox[2:],car[frame][2:6], 1):
                overlapped += 1

                # overlap_hero_car=True
        # print ("Overlapped "+str(overlapped))
        return overlapped

    def intercept_car(self, frame_in, all_frames=False, agent_frame=-1, bbox=None,cars=None, car=None ):
        if bbox == None:
            if agent_frame<0:
                x_range, y_range, z_range = self.pose_range(frame_in)
            else:
                x_range, y_range, z_range = self.pose_range(agent_frame)
            overlapped = 0
            frames=[]
            if all_frames or frame_in>=len(self.cars):
                frames=list(range(len(self.cars)))
            else:
                frames.append(frame_in)
            person_bbox=[x_range[0],x_range[1], y_range[0],y_range[1],z_range[0], z_range[1]]
        else:
            person_bbox=bbox

        if cars==None:
            cars=self.cars

        if car ==None and self.useRealTimeEnv:
            car=self.car_bbox
            #print (" Car bbox "+str(self.car_bbox))



        for frame in frames:
            for car_local in cars[frame]:
                # print ("  car " + str(car_local[2:6])+" person bbox "+str(person_bbox[2:]))  #print ("  car " + str(car[frame][2:6])+" person bbox "+str(person_bbox[2:]))
                if overlap(car_local[2:6],person_bbox[2:],1) or overlap(person_bbox[2:],car_local[2:6],1):
                    overlapped += 1
                    # print ("Overlapped")
                # if evaluate_hero_car and len(self.car)> frame:

            if self.useRealTimeEnv:
                # print(" Intercept hero car ? " + str(car[frame][2:6]))
                if overlap(car[frame][2:6],person_bbox[2:],1) or overlap(person_bbox[2:],car[frame][2:6],1) :
                    overlapped+=1
                    # print("Overlapped")

                #overlap_hero_car=True
        # print ("Final overlapped "+str(overlapped))
        return overlapped



    def pose_range(self, frame_in, pos=(), as_int=True):
        if len(pos)==0:
            pos = self.agent[frame_in]
            #print ("Agent pos "+str(pos))
        if not len(pos) == 3:
            print(("Position "+str(pos)+" "+str(len(pos))))
        #print "Position " + str(pos) + " "
        x_range = [max(pos[0] - self.agent_size[0],0), min(pos[0] + self.agent_size[0], self.reconstruction.shape[0])]
        y_range = [max(pos[1] - self.agent_size[1], 0),min(pos[1] + self.agent_size[1], self.reconstruction.shape[1])]
        z_range = [max(pos[2] - self.agent_size[2],0),min(pos[2] + self.agent_size[2], self.reconstruction.shape[2])]
        if as_int:
            for i in range(2):
                x_range[i]=int(round(x_range[i]))
                y_range[i] = int(round(y_range[i]))
                z_range[i] = int(round(z_range[i]))

        return x_range, y_range, z_range

    def pose_range_car(self, frame_in, pos=(), as_int=True):
        if len(pos)==0:
            pos = self.agent[frame_in]
            #print ("Agent pos "+str(pos))
        if not len(pos) == 3:
            print(("Position "+str(pos)+" "+str(len(pos))))
        #print "Position " + str(pos) + " "
        x_range = [max(pos[0] - self.agent_size[0],0), min(pos[0] + self.agent_size[0], self.reconstruction.shape[0])]
        y_range = [max(pos[1] - self.agent_size[1], 0),min(pos[1] + self.agent_size[1], self.reconstruction.shape[1])]
        z_range = [max(pos[2] - self.agent_size[2],0),min(pos[2] + self.agent_size[2], self.reconstruction.shape[2])]
        if as_int:
            for i in range(2):
                x_range[i]=int(round(x_range[i]))
                y_range[i] = int(round(y_range[i]))
                z_range[i] = int(round(z_range[i]))

        return x_range, y_range, z_range

    def intercept_person(self, frame_in, no_height=False, person_id=-1, frame_input=-1):
        x_range, y_range, z_range = self.pose_range(frame_in)
        if no_height:
            x_range=[0, self.reconstruction.shape[0]]
        people_overlapped = []
        if frame_input>=0:
            return  self.collide_with_pedestrians( frame_input, no_height=False, per_id=-1)
        if person_id<0:
            return self.intercept_pedestrian_trajectory(frame_in, no_height=False)
        else:
            for person in self.people_dict[person_id]:
                x_pers = [ min(person[0, :]), max(person[0, :]), min(person[1, :]), max(person[1, :]),min(person[2, :]), max(person[2, :])]
                if overlap(x_pers[2:], [ y_range[0],y_range[1], z_range[0], z_range[1]], 1):
                    people_overlapped.append(person)
                    return people_overlapped
            return people_overlapped

    def collide_with_people_dict(self,frame_in, people_overlapped, x_range, y_range, z_range, per_id):

        for person_key in list(self.people_dict.keys()):

            if self.key_map[per_id] != person_key:
                dif=frame_in-self.init_frames[person_key]

                if dif>=0 and len(self.people_dict[person_key])>dif:
                    person=self.people_dict[person_key][dif]

                    x_pers = [ min(person[1, :]), max(person[1, :]),
                              min(person[2, :]),
                              max(person[2, :])]

                    if overlap(x_pers, [ y_range[0], y_range[1], z_range[0], z_range[1]], 1):

                        people_overlapped.append(person)

                        return people_overlapped
        return people_overlapped

    def intercept_pedestrian_trajectory(self, frame_in, no_height=False):
        x_range, y_range, z_range = self.pose_range(frame_in)
        if no_height:
            x_range = [0, self.reconstruction.shape[0]]
        if self.useRealTimeEnv: # how to make it per person? Avreage by agent size?
            if frame_in<len(self.agent_prediction_people):
                trajectory_current_frame=self.agent_prediction_people[frame_in]
                center=[int(trajectory_current_frame.shape[0]*0.5),int(trajectory_current_frame.shape[1]*0.5 )]
                return np.sum(trajectory_current_frame[center[0]-self.agent_size[1]:center[0]+self.agent_size[1], center[1]-self.agent_size[2]:center[1]-self.agent_size[2]]!=0)
            return 0
        people_overlapped = []
        for frame in range(len(self.people)):
            collide_with_people(frame, people_overlapped, x_range, y_range, z_range, self.people)
        return people_overlapped

    def intercept_pedestrian_trajectory_tensor(self, frame_in, no_height=False):
        x_range, y_range, z_range = self.pose_range(frame_in, as_int=True)
        if no_height:
            x_range = [0, self.reconstruction.shape[0]]

        return np.sum(self.reconstruction[ x_range[0]: x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1],CHANNELS.pedestrian_trajectory ])>0

    def collide_with_pedestrians(self, frame_in, no_height=False, per_id=-1, agent_frame=-1, pos=()):

        if agent_frame<0:
            agent_frame=frame_in

        x_range, y_range, z_range = self.pose_range(agent_frame, pos=pos)

        #print ("Collide with pedestrians " + str(frame_in)+" "+str(x_range)+" "+str(y_range)+" "+str(z_range))
        if no_height:
            x_range = [0, self.reconstruction.shape[0]]
        people_overlapped = []
        if per_id<0:

            return collide_with_people(frame_in, people_overlapped, x_range, y_range, z_range,self.people)

        return self.collide_with_people_dict(frame_in, people_overlapped, x_range, y_range, z_range, per_id)


    def pose_dist(self, pose1, pose2):
        dif = pose1 - np.mean(pose2)
        return np.linalg.norm(dif) / 14

    def intercept_objects(self, frame_in, pos=(), no_height=False, cars=False, reconstruction=None):  # Normalisera/ binar!
        segmentation = self.person_bounding_box(frame_in, pos=pos, no_height=no_height)

        #count=((10 < segmentation) & (segmentation < 21) ).sum()
        if self.new_carla:
            obstacles = OBSTACLE_LABELS_NEW
        else:
            obstacles = OBSTACLE_LABELS
        #print(" Obstacle class " + str(obstacles))
        count = 0
        for label in obstacles:
            count += (segmentation == label).sum()

        if cars:
            for label in MOVING_OBSTACLE_LABELS:
                count += (segmentation == label).sum()
            print(" Moving class " + str(MOVING_OBSTACLE_LABELS))
            #count += (segmentation > 25).sum()
        if count > 0:
            x_range, y_range, z_range = self.pose_range(frame_in)
            if (x_range[1]+2-x_range[0])*(y_range[1]+2-y_range[0])>0:
                #print np.histogram(segmentation.flatten())
                return count *1.0/((x_range[1]+2-x_range[0])*(y_range[1]+2-y_range[0]))#*(z_range[1]+2-z_range[0]))
            else:
                return 0
        return 0

        # To do: add cars!

    def valid_position(self, pos, no_height=True):

        objs = self.intercept_objects(0, pos=pos, no_height=no_height)
        #print "Intercept objects " + str(objs)
        return objs <=0#0.1

    def out_of_axis(self, frame_in):
        directions=[False, False]
        for indx in range(1,3):
            #print "Dir "+str(indx)+" Max agent: "+str(self.agent[frame_in][indx]+self.agent_size[indx])+" Min: "+str(self.agent[frame_in][indx]-self.agent_size[indx])
            if (round(self.agent[frame_in][indx]+self.agent_size[indx]))>=self.reconstruction.shape[indx] or round(self.agent[frame_in][indx]-self.agent_size[indx])<0:
                directions=True

        return np.array(directions).any()


    def discounted_reward(self):
        self.accummulated_r = 0
        for frame_n in range(self.seq_len-2, -1, -1):
            tmp=self.accummulated_r *self.gamma
            self.accummulated_r=tmp
            self.accummulated_r+= self.reward[frame_n]
            self.reward_d[frame_n] =self.accummulated_r
        if self.useRealTimeEnv:
            self.accummulated_r_car = 0
            for frame_n in range(self.seq_len - 2, -1, -1):
                tmp = self.accummulated_r_car * self.gamma
                self.accummulated_r_car = tmp

                self.accummulated_r_car += self.reward_car[frame_n]
                self.reward_car_d[frame_n] = self.accummulated_r_car
                # print (
                # " frame " + str(frame_n) + " car reward " + str(self.reward_car[frame_n]) + " discounted reward " + str(
                #     tmp) + " sum "+str(self.reward_car_d[frame_n]))

        # print ("Reward " + str(self.reward))
        # print ("Discounted reward "+str(self.reward_d))
        # print ("Car Reward " + str(self.reward_car))
        # print ("Discounted reward acr  " + str(self.reward_car_d))

    def get_people_in_frame(self, frame_in, needPeopleNames = False):
        people=[]
        people_names=[]
        for person_key in list(self.people_dict.keys()):

            if self.goal_person_id_val != person_key:

                dif = frame_in - self.init_frames[person_key]

                if dif >= 0 and len(self.people_dict[person_key]) > dif:
                    person = self.people_dict[person_key][dif]
                    person_prev= self.people_dict[person_key][max(dif-1, 0)]

                    x_pers = [np.mean(person[2, :]),np.mean(person[1, :]),np.mean(person[2, :]-person_prev[2,:]),
                              np.mean(person[1, :]-person_prev[1,:]),np.mean([person[2, 1]-person[2, 0], person[1, 1]-person[1, 0]]) ]
                    people.append(np.array(x_pers).astype(int))
                    people_names.append(person_key)
        if needPeopleNames: # To keep compatibility...
            return people, people_names
        else:
            return people




    def get_agent_neighbourhood(self,pos, breadth,frame_in,vel=[], training=True, eval=True, pedestrian_view_occluded=False):
        temporal_scaling=0.1*0.3 # what scaling to have on temporal input to stay in range [-1,1]

        start_pos = np.zeros(3, np.int)
        min_pos=np.zeros(3, np.int)
        max_pos = np.zeros(3, np.int)
        if len(breadth)==2:
            breadth=[self.reconstruction.shape[0]/2+1, breadth[0], breadth[1]]

        # Get 2D or 3D input? Create holders
        if self.run_2D:
            mini_ten = np.zeros(( breadth[1] * 2 + 1, breadth[2] * 2 + 1, 6), dtype=np.float)
            tmp_people = np.zeros(( mini_ten.shape[0], mini_ten.shape[1], 1))
            tmp_cars = np.zeros(( mini_ten.shape[0], mini_ten.shape[1], 1))
            tmp_people_cv = np.zeros(( mini_ten.shape[0], mini_ten.shape[1], 1))
            tmp_cars_cv =np.zeros(( mini_ten.shape[0], mini_ten.shape[1], 1))
        else:
            mini_ten = np.zeros((breadth[0] * 2 + 1, breadth[1] * 2 + 1, breadth[2] * 2 + 1, 6), dtype=np.float)
            tmp_people = np.zeros(( mini_ten.shape[0],  mini_ten.shape[1],  mini_ten.shape[2], 1))
            tmp_cars = np.zeros(( mini_ten.shape[0],  mini_ten.shape[1],  mini_ten.shape[2], 1))
            tmp_people_cv = np.zeros((mini_ten.shape[0], mini_ten.shape[1], mini_ten.shape[2], 1))
            tmp_cars_cv = np.zeros((mini_ten.shape[0], mini_ten.shape[1], mini_ten.shape[2], 1))

        # Range of state

        for i in range(3):
            min_pos[i]=round(pos[i])-breadth[i]
            max_pos[i] = round(pos[i]) + breadth[i] +1

            if min_pos[i]<0: # agent below rectangle of visible.
                start_pos[i] = -min_pos[i]
                min_pos[i]=0
                if max_pos[i]<0:
                    if self.useRealTimeEnv:
                        self.add_predicted_people(frame_in, tmp_people)
                    return mini_ten, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv
            if max_pos[i]>self.reconstruction.shape[i]:
                max_pos[i]=self.reconstruction.shape[i]
                if min_pos[i]>self.reconstruction.shape[i]:
                    if self.useRealTimeEnv:
                        self.add_predicted_people(frame_in, tmp_people)
                    return mini_ten, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv
        # print (" Agent dimensions: "+str([ min_pos[1],max_pos[1], min_pos[2],max_pos[2]]))
        # if self.useRealTimeEnv:
        #     print (" cars "+str(self.car_bbox[frame_in])+" velocity "+str(self.velocity_car[frame_in-1 ]))
        # print (" cars "+str(self.cars[frame_in]))
        # print (" pedestrians " + " cars " + str(self.people[frame_in]))
        # Get reconstruction cut out
        if self.run_2D:
            tmp = self.reconstruction_2D[ min_pos[1]:max_pos[1], min_pos[2]:max_pos[2], :].copy()
        else:
            tmp=self.reconstruction[min_pos[0]:max_pos[0], min_pos[1]:max_pos[1], min_pos[2]:max_pos[2], :].copy()

        max_len_people=len(self.people_predicted)-1
        # constant velocity prediction
        if self.run_2D:
            # Get constant velocity prediction
            tmp_people_cv[ start_pos[1]:start_pos[1] + tmp.shape[0],start_pos[2]:start_pos[2] + tmp.shape[1], :] = self.people_predicted[min(frame_in, max_len_people)][
                                                           min_pos[1]:max_pos[1], min_pos[2]:max_pos[2],np.newaxis].copy()
            tmp_cars_cv[ start_pos[1]:start_pos[1] + tmp.shape[0],start_pos[2]:start_pos[2] + tmp.shape[1], :] = self.cars_predicted[min(frame_in, max_len_people)][
                                                           min_pos[1]:max_pos[1], min_pos[2]:max_pos[2], np.newaxis].copy()
            if self.useRealTimeEnv:
                bounding_box=[min_pos[1],max_pos[1], min_pos[2],max_pos[2]]
                # print (" Bounding box "+str(bounding_box))
                tmp_people_cv, tmp_cars_cv= self.predict_cars_and_people_in_a_bounding_box(tmp_people_cv,tmp_cars_cv,bounding_box, start_pos,frame_in)
                tmp_people_cv = np.expand_dims(tmp_people_cv, axis=2)
                tmp_cars_cv = np.expand_dims(tmp_cars_cv, axis=2)
                #print ("Sum of people traj "+str(np.sum(abs(tmp_people_cv)))+" Sum of car traj "+str(np.sum(abs(tmp_cars_cv))))
            # Insert reconstruction into holder
            mini_ten[ start_pos[1]:start_pos[1] + tmp.shape[0],start_pos[2]:start_pos[2] + tmp.shape[1], :] = tmp

        else:
            # Get constant velocity prediction
            tmp_people_cv[start_pos[0]:start_pos[0]+tmp.shape[0],start_pos[1]:start_pos[1]+tmp.shape[1],start_pos[2]:start_pos[2]+tmp.shape[2], :]=self.people_predicted[min(frame_in, max_len_people)][min_pos[0]:max_pos[0], min_pos[1]:max_pos[1], min_pos[2]:max_pos[2],np.newaxis].copy()
            tmp_cars_cv[start_pos[0]:start_pos[0] + tmp.shape[0], start_pos[1]:start_pos[1] + tmp.shape[1],
            start_pos[2]:start_pos[2] + tmp.shape[2], :] = self.cars_predicted[min(frame_in, max_len_people)][min_pos[0]:max_pos[0],
                                                           min_pos[1]:max_pos[1], min_pos[2]:max_pos[2] , np.newaxis].copy()
            # Insert reconstruction into holder
            mini_ten[start_pos[0]:start_pos[0]+tmp.shape[0],start_pos[1]:start_pos[1]+tmp.shape[1],start_pos[2]:start_pos[2]+tmp.shape[2], :]=tmp

        # Bounds of the bounding box
        bbox=[min_pos[0],max_pos[0], min_pos[1],max_pos[1], min_pos[2],max_pos[2]]

        # Add all pedestrians in loc_frame into pedestrian layer
        loc_frame=frame_in

        if self.goal_person_id>0:
            for person_key in list(self.people_dict.keys()):
                if self.goal_person_id_val != person_key:
                    dif = frame_in - self.init_frames[person_key]
                    if dif >= 0 and len(self.people_dict[person_key]) > dif: # If pedestrian is in frame
                        person = self.people_dict[person_key][dif] # Get pedestrian position in current frame
                        x_pers = [min(person[0, :]), max(person[0, :]), min(person[1, :]), max(person[1, :]),
                                  min(person[2, :]),
                                  max(person[2, :])]
                        if overlap(x_pers[2:], bbox[2:], 1) or self.temporal: # Check if overlapping
                            #print ("Person goal id " + str(x_pers)+"  second "+str(bbox))
                            intersection = [max(x_pers[0], bbox[0]).astype(int), min(x_pers[1], bbox[1]).astype(int), max(x_pers[2], bbox[2]).astype(int),
                                            min(x_pers[3], bbox[3]).astype(int),
                                            max(x_pers[4], bbox[4]).astype(int), min(x_pers[5], bbox[5]).astype(int)]

                            #print (" Intercsetion before goal id " + str(intersection))
                            intersection[0:2] = intersection[0:2] - bbox[0]
                            intersection[2:4] = intersection[2:4] - bbox[2]
                            intersection[4:] = intersection[4:] - bbox[4]
                            #print (" Intercsetion goal id " + str(intersection))
                            if self.run_2D:

                                tmp_people[ intersection[2]:intersection[3],
                                intersection[4]:intersection[5], 0] = 1
                            else:
                                tmp_people[intersection[0]:intersection[1], intersection[2]:intersection[3],
                                intersection[4]:intersection[5], 0] = 1
        else:
            for person in self.people[loc_frame]:
                x_pers = [min(person[0, :]), max(person[0, :])+1, min(person[1, :]), max(person[1, :])+1, min(person[2, :]),
                          max(person[2, :])+1]

                if overlap(x_pers[2:],bbox[2:] , 1) or self.temporal:
                    #print ("Person " + str(x_pers)+"  second "+str(bbox)+" temporal: "+str(self.temporal))
                    intersection = [max(x_pers[0], bbox[0]).astype(int), min(x_pers[1], bbox[1]).astype(int), max(x_pers[2], bbox[2]).astype(int), min(x_pers[3], bbox[3]).astype(int),
                                    max(x_pers[4], bbox[4]).astype(int), min(x_pers[5], bbox[5]).astype(int)]

                    #print (" Intercsetion before " + str(intersection))
                    intersection[0:2] = intersection[0:2] - bbox[0]
                    intersection[2:4] = intersection[2:4] - bbox[2]
                    intersection[4:] = intersection[4:] - bbox[4]
                    #print (" Intercsetion " + str(intersection))
                    if self.run_2D:
                        tmp_people[ intersection[2]:intersection[3],
                        intersection[4]:intersection[5], 0] = 1
                    else:
                        tmp_people[intersection[0]:intersection[1], intersection[2]:intersection[3],
                        intersection[4]:intersection[5], 0] = 1

        # Add colliding cars in loc_frame into colliding cars layer.
        cars_in_scene=copy.copy(self.cars[min(loc_frame, max_len_people)])
        if self.useRealTimeEnv:
            cars_in_scene.append(self.car_bbox[min(loc_frame, max_len_people)])
        for obj in cars_in_scene:
            if overlap(obj[2:], bbox[2:], 1)or self.temporal:
                intersection= [max(obj[0], bbox[0]).astype(int),min(obj[1], bbox[1]).astype(int),max(obj[2], bbox[2]).astype(int),min(obj[3], bbox[3]).astype(int),max(obj[4], bbox[4]).astype(int),min(obj[5], bbox[5]).astype(int)]
                intersection[0:2]=intersection[0:2]-bbox[0]
                intersection[2:4] = intersection[2:4] - bbox[2]
                intersection[4:] = intersection[4:] - bbox[4]
                if self.run_2D:
                    tmp_cars[ intersection[2]:intersection[3],intersection[4]:intersection[5], 0] = 1
                else:
                    tmp_cars[intersection[0]:intersection[1], intersection[2]:intersection[3],
                    intersection[4]:intersection[5], 0] = 1

        if self.temporal:
            if self.run_2D: # GT- pedestrian locations
                temp=mini_ten[:,:,CHANNELS.cars_trajectory]>0 # All places where pedestrians have ever been
                mini_ten[:,:,CHANNELS.cars_trajectory] = mini_ten[:,:,CHANNELS.cars_trajectory]-(frame_in*temp) # Remove current frame
                mini_ten[:,:,CHANNELS.cars_trajectory]=temporal_scaling*mini_ten[:,:,CHANNELS.cars_trajectory] # Add scaling
                temp = mini_ten[:, :, CHANNELS.pedestrian_trajectory] > 0
                mini_ten[ :, :, CHANNELS.pedestrian_trajectory] = mini_ten[ :, :, CHANNELS.pedestrian_trajectory] - (frame_in * temp)
                mini_ten[ :, :, CHANNELS.pedestrian_trajectory] = temporal_scaling * mini_ten[ :, :, CHANNELS.pedestrian_trajectory]
            else:
                temp = mini_ten[:, :, :, CHANNELS.cars_trajectory] > 0
                mini_ten[:, :, :, CHANNELS.cars_trajectory] = mini_ten[:, :, :, CHANNELS.cars_trajectory] - (frame_in * temp)
                mini_ten[:, :, :, CHANNELS.cars_trajectory] = temporal_scaling * mini_ten[:, :, :, CHANNELS.cars_trajectory]
                temp = mini_ten[:, :, :, CHANNELS.pedestrian_trajectory] > 0
                mini_ten[:, :, :, CHANNELS.pedestrian_trajectory] = mini_ten[:, :, :, CHANNELS.pedestrian_trajectory] - (frame_in * temp)
                mini_ten[:, :, :, CHANNELS.pedestrian_trajectory] = temporal_scaling * mini_ten[:, :, :, CHANNELS.pedestrian_trajectory]
            tmp_people_cv=temporal_scaling*tmp_people_cv
            tmp_cars_cv=temporal_scaling*tmp_cars_cv

        # elif self.predict_future:
        #     tmp_people_cv = 10*temporal_scaling * tmp_people_cv
        #     tmp_cars_cv = 10*temporal_scaling * tmp_cars_cv
        if pedestrian_view_occluded:
            valid_positions=np.logical_and(np.logical_and(mini_ten[:,:,CHANNELS.semantic]==0,tmp_cars==0 ), tmp_people)

            position=pos-breadth+1
            occlusion_map = self.find_occlusions(valid_positions, valid_positions, position, np.linalg.norm(vel[1:]), vel[1:], 0, mini_ten.shape)
            occlusion_map_tiled=np.tile(occlusion_map, 6)
            mini_ten[occlusion_map_tiled==0]=0
            tmp_people[occlusion_map == 0] = 0
            tmp_cars[occlusion_map == 0] = 0
            tmp_people_cv[occlusion_map == 0] = 0
            tmp_cars_cv[occlusion_map == 0] = 0

        #print(" temporal cars after scaling "+str(np.sum(np.abs(tmp_cars_cv)))+" people "+str(np.sum(np.abs(tmp_people_cv)))+" scaling "+str(temporal_scaling))

        return mini_ten, tmp_people, tmp_cars, tmp_people_cv, tmp_cars_cv

    def get_intercepting_cars(self, pos, breadth, frame_in):
        overlapped = []

        box_compare=[pos[0]-breadth[0], pos[0]+breadth[0], pos[1]-breadth[1], pos[1]+breadth[1], pos[2]-breadth[2], pos[2]+breadth[2]]
        for obj in self.cars[frame_in]:
            if overlap(obj[2:], box_compare[2:], 1):
                o_car=[]
                for indx in range(6):
                    o_car.append(int(round(min(max(obj[indx]-pos[indx/2]-breadth[indx/2],0),pos[indx/2]+breadth[indx/2]))))
                overlapped.append(np.array(o_car))
        return overlapped

    def save(self, statistics, num_episode, poses,initialization_map,initialization_car,statistics_car, ped_seq_len=-1):
        ped_seq_len=self.seq_len


        statistics[num_episode, :ped_seq_len, STATISTICS_INDX.agent_pos[0]:STATISTICS_INDX.agent_pos[1]] = np.vstack(self.agent[:ped_seq_len-1]).copy()

        statistics[num_episode, :ped_seq_len, STATISTICS_INDX.velocity[0]:STATISTICS_INDX.velocity[1]] = np.vstack(self.velocity[:ped_seq_len]).copy()
        statistics[num_episode, -1, STATISTICS_INDX.velocity[0]:STATISTICS_INDX.velocity[1]] = copy.copy(self.vel_init)
        statistics[num_episode, :ped_seq_len, STATISTICS_INDX.action] = np.vstack(self.action[:ped_seq_len]).reshape(ped_seq_len - 1)
        statistics[num_episode, :ped_seq_len, STATISTICS_INDX.probabilities[0]:STATISTICS_INDX.probabilities[1]] = np.vstack(self.probabilities[:ped_seq_len]).copy()
        if np.sum(self.angle)>0:
            statistics[num_episode,:ped_seq_len, STATISTICS_INDX.angle] = self.angle[:ped_seq_len].copy()
        statistics[num_episode, :ped_seq_len, STATISTICS_INDX.reward] = self.reward[:ped_seq_len].copy()
        statistics[num_episode, :ped_seq_len, STATISTICS_INDX.reward_d] = self.reward_d[:ped_seq_len].copy()
        statistics[num_episode, :ped_seq_len, STATISTICS_INDX.loss] = self.loss[:ped_seq_len]
        statistics[num_episode, :ped_seq_len, STATISTICS_INDX.speed] = np.vstack(self.speed[:ped_seq_len]).reshape(ped_seq_len - 1)
        statistics[num_episode, :ped_seq_len, STATISTICS_INDX.measures[0]:STATISTICS_INDX.measures[1]] = self.measures[:ped_seq_len].copy()
        #print (" Agent distracted: "+str(statistics[num_episode, :ped_seq_len, STATISTICS_INDX.measures[0]+PEDESTRIAN_MEASURES_INDX.distracted])+" save to file ")
        # print "Save hit by car: "+str(statistics[num_episode, :, 38])
        #  if self.goal_person_id>0:
        #      statistics[num_episode, 0:3, 38 + NBR_MEASURES] = np.mean(self.people_dict[self.goal_person_id][-1], axis=1)
        #print "Save init method "+str(self.init_method)+" "+str( 38+self.nbr_measures)
        statistics[num_episode, 0, STATISTICS_INDX.init_method] = self.init_method


        if self.follow_goal and len(self.goal)>0:
            statistics[num_episode, 3, STATISTICS_INDX.goal] = self.goal[1]
            statistics[num_episode, 4, STATISTICS_INDX.goal] = self.goal[2]
            if self.goal_time>0:
                statistics[num_episode, 5, STATISTICS_INDX.goal] = self.goal_time
            if self.goal_person_id >=0:
                statistics[num_episode, 6, STATISTICS_INDX.goal] = self.goal_person_id

        if np.sum(self.agent_pose[0,:])>0:
            poses[num_episode, :, STATISTICS_INDX_POSE.pose[0]:STATISTICS_INDX_POSE.pose[1]]=copy.deepcopy(self.agent_pose)
            poses[num_episode, :, STATISTICS_INDX_POSE.agent_high_frq_pos[0]:STATISTICS_INDX_POSE.agent_high_frq_pos[1]]=copy.deepcopy(self.agent_high_frq_pos)
            poses[num_episode, :, STATISTICS_INDX_POSE.agent_pose_frames] = copy.deepcopy(self.agent_pose_frames)
            poses[num_episode, :, STATISTICS_INDX_POSE.avg_speed] = copy.deepcopy(self.avg_speed)
            poses[num_episode, :, STATISTICS_INDX_POSE.agent_pose_hidden[0]:STATISTICS_INDX_POSE.agent_pose_hidden[1]] = copy.deepcopy(self.agent_pose_hidden)
        # if self.env_nbr > 5 and self.env_nbr<8:
        #     statistics[num_episode, 2, 38+NBR_MEASURES] =self.num_crossings
        #     statistics[num_episode, 3:5, 38+NBR_MEASURES]=self.point
        #     statistics[num_episode, 5:5+len(self.thetas), 38+NBR_MEASURES] = self.thetas
        if len(initialization_map)>0 and self.init_method == 7:

            initialization_map[num_episode, :len(self.prior.flatten()), STATISTICS_INDX_MAP.prior] = self.prior.flatten()
            initialization_map[num_episode, :len(self.init_distribution), STATISTICS_INDX_MAP.init_distribution] = self.init_distribution
            if self.learn_goal:
                initialization_map[num_episode, :len(self.goal_prior.flatten()), STATISTICS_INDX_MAP.goal_prior] = self.goal_prior.flatten()
                initialization_map[num_episode, :len(self.goal_distribution.flatten()), STATISTICS_INDX_MAP.goal_distribution] = self.goal_distribution.flatten()
            initialization_car[num_episode, STATISTICS_INDX_CAR_INIT.car_id]=self.init_car_id
            initialization_car[num_episode, STATISTICS_INDX_CAR_INIT.car_pos[0]:STATISTICS_INDX_CAR_INIT.car_pos[1]] = self.init_car_pos
            initialization_car[num_episode, STATISTICS_INDX_CAR_INIT.car_vel[0]:STATISTICS_INDX_CAR_INIT.car_vel[1]] = self.init_car_vel
            if self.learn_goal:
                initialization_car[num_episode, 5:5+len(self.manual_goal)] = self.manual_goal

        if self.useRealTimeEnv:
            statistics_car[num_episode, :ped_seq_len - 1, STATISTICS_INDX_CAR.agent_pos[0]:STATISTICS_INDX_CAR.agent_pos[1]] = np.vstack(self.car[:ped_seq_len - 1]).copy()

            statistics_car[num_episode, :ped_seq_len - 1, STATISTICS_INDX_CAR.velocity[0]:STATISTICS_INDX_CAR.velocity[1]] = np.vstack(self.velocity_car[:ped_seq_len - 1]).copy()
            statistics_car[num_episode, :ped_seq_len-1, STATISTICS_INDX_CAR.action] = self.action_car[:ped_seq_len-1].copy()
            #print (" Size before addition "+str(np.squeeze(self.probabilities_car[:ped_seq_len - 1]).shape)+" after "+str(statistics_car[num_episode, :ped_seq_len - 1, STATISTICS_INDX_CAR.probabilities].shape))
            statistics_car[num_episode, :ped_seq_len - 1, STATISTICS_INDX_CAR.probabilities[0]:STATISTICS_INDX_CAR.probabilities[1]] = np.squeeze(self.probabilities_car[:ped_seq_len - 1,:])

            statistics_car[num_episode, :len(self.car_goal), STATISTICS_INDX_CAR.goal] = self.car_goal.copy()
            statistics_car[num_episode, len(self.car_goal):len(self.car_goal)+len(self.car_dir), STATISTICS_INDX_CAR.goal] = self.car_dir.copy()

            statistics_car[num_episode, :ped_seq_len - 1, STATISTICS_INDX_CAR.reward] = self.reward_car[:ped_seq_len - 1].copy()
            statistics_car[num_episode, :ped_seq_len - 1, STATISTICS_INDX_CAR.reward_d] = self.reward_car_d[:ped_seq_len - 1].copy()
            #statistics_car[num_episode, :ped_seq_len - 1, STATISTICS_INDX_CAR.loss] = self.loss_car[:ped_seq_len - 1]
            #print(" Save car loss "+str(statistics_car[num_episode, :ped_seq_len - 1, STATISTICS_INDX_CAR.loss])+" pos "+str(STATISTICS_INDX_CAR.loss)+" "+str(num_episode))
            statistics_car[num_episode, :ped_seq_len - 1, STATISTICS_INDX_CAR.speed] = np.vstack(self.speed_car[:ped_seq_len - 1]).reshape(
                ped_seq_len - 1)

            statistics_car[num_episode, :ped_seq_len - 1, STATISTICS_INDX_CAR.bbox[0]:STATISTICS_INDX_CAR.bbox[1]] = np.vstack(self.car_bbox[:ped_seq_len - 1]).copy()
            statistics_car[num_episode, :ped_seq_len - 1, STATISTICS_INDX_CAR.measures[0]:STATISTICS_INDX_CAR.measures[1]] = self.measures_car[:ped_seq_len - 1].copy()
            statistics_car[num_episode, :ped_seq_len - 1, STATISTICS_INDX_CAR.angle] = self.car_angle[:ped_seq_len - 1].copy()

        return statistics, poses, initialization_map, initialization_car, statistics_car

    def save_loss(self, statistics):

        statistics[:, :, 36] =np.copy( self.loss)
        return statistics




    def get_input_cars(self, pos, frame, distracted=False):
        feature=np.zeros([1,len(self.actions)-1], dtype=np.float32)

        closest_car, min_dist = self.find_closest_car(frame, pos)
        #print(" Get input cars distracted "+str(distracted))
        if len(closest_car)==0 or distracted:
            return feature
        car_dir=closest_car-np.array(pos)
        car_dir[0] = 0
        dir = self.find_action_to_direction(car_dir, min_dist)
        if dir>4:
            dir=dir-1
        feature[0,dir]=min_dist/np.linalg.norm(self.reconstruction.shape[0:3])


        return feature

    def find_closest_car(self, frame, pos, ignore_car_agent=False):
        if not self.useRealTimeEnv or ignore_car_agent:
            return find_closest_car_in_list(frame, pos, self.cars)
            # closest_car = []
            # min_dist = -1
            # for car in self.cars[min(frame, len(self.cars) - 1)]:
            #     car_pos = np.array([np.mean(car[0:2]), np.mean(car[2:4]), np.mean(car[4:])])
            #     #print ("regular car " + str(car_pos)+" dist "+str(np.linalg.norm(np.array(pos[1:]) - car_pos[1:])))
            #     if min_dist < 0 or min_dist > np.linalg.norm(np.array(pos[1:]) - car_pos[1:]):
            #         closest_car = car_pos
            #         min_dist = np.linalg.norm(np.array(pos[1:]) - car_pos[1:])
        else:
            closest_car = np.array(self.car[frame])
            #print ("Agent car "+str(self.car[frame ]))
            min_dist = np.linalg.norm(np.array(pos[1:]) - closest_car[1:])

        #print (" Closest car " + str(closest_car)+" to "+str(pos)+" min distance "+str(min_dist))
        return closest_car, min_dist

    def get_input_cars_smooth(self, pos, frame, distracted=False):
        feature=np.zeros([1,len(self.actions)-1], dtype=np.float32)

        closest_car, min_dist = self.find_closest_car(frame, pos)

        #print(" Get input cars distracted " + str(distracted) +" frame "+str(frame))
        if len(closest_car)==0 or distracted:
            return feature
        car_dir=closest_car-np.array(pos)
        car_dir[0] = 0
        feature[0,:] = self.find_action_to_direction_smooth(car_dir, min_dist)*(1.0/np.linalg.norm(self.reconstruction.shape[0:3]))
        #print(" Scale car proximty feature dividing by"+str(np.linalg.norm(self.reconstruction.shape[0:3])))
        return feature

    def get_input_cars_cont_linear(self, pos, frame, distracted=False):
        feature = np.zeros([1, 2], dtype=np.float32)

        closest_car, min_dist = self.find_closest_car(frame, pos)
        #print(" Get input cars distracted " + str(distracted) + " return " + str(feature))
        if len(closest_car) == 0 or distracted:
            return feature
        feature=(closest_car - np.array(pos))*1.0/np.sqrt(self.reconstruction.shape[1] ** 2 + self.reconstruction.shape[2] ** 2)
        #print "Distance to car "+str(np.linalg.norm(closest_car - np.array(pos)))+ " vector "+str(closest_car - np.array(pos))+" car "+str(closest_car)
        # if abs(feature[1])<0.5:
        #     feature[1]=2
        # else:
        #     feature[1] = 1 / feature[1]
        # if abs(feature[2])<.5:
        #     feature[2] = 2
        # else:
        #     feature[2] = 1 / feature[2]

        return np.expand_dims(feature[1:], axis=0)

    def get_input_cars_cont_angular(self, pos, frame, distracted=False):
        feature = np.zeros([1, 2], dtype=np.float32)

        closest_car, min_dist = self.find_closest_car(frame, pos)
        #print(" Get input cars distracted " + str(distracted) )
        #print "Closest car " + str(closest_car) + " dist " + str(min_dist) + " " + str(closest_car - np.array(pos))
        if len(closest_car) == 0 or distracted:
            return feature
        dir=closest_car[1:] - np.array(pos)[1:]

        feature[0,0]=np.linalg.norm(dir)*1.0/np.sqrt(self.reconstruction.shape[1] ** 2 + self.reconstruction.shape[2] ** 2)

        feature[0,1]=np.arctan2(dir[0],dir[1])
        if feature[0,1] <= -np.pi:
            feature[0,1]=feature[0,1]+2*np.pi
        #print "Car direction "+str(np.arctan2(dir[0],dir[1])/np.pi)+" after minus "+str(feature[0,1]/np.pi)
        #
        feature[0,1]=feature[0,1]/np.pi
        # if abs(feature[1])<0.5:
        #     feature[1]=2
        # else:
        #     feature[1] = 1 / feature[1]
        # if abs(feature[2])<.5:
        #     feature[2] = 2
        # else:
        #     feature[2] = 1 / feature[2]

        return feature

    def get_input_cars_cont(self, pos, frame, distracted=False):
        feature = np.zeros([1, 2], dtype=np.float32)

        closest_car = []
        closest_car, min_dist = self.find_closest_car(frame, pos)
        #print(" Get input cars distracted " + str(distracted) )
        if len(closest_car) == 0 or distracted :
            #print feature
            return feature
        feature=(closest_car - np.array(pos))#*1.0/(self.reconstruction.shape[1]*self.reconstruction.shape[2])
        if abs(feature[1])<0.5:
            feature[1]=2
        else:
            feature[1] = 1 / feature[1]
        if abs(feature[2])<.5:
            feature[2] = 2
        else:
            feature[2] = 1 / feature[2]
        #print feature
        return np.expand_dims(feature[1:], axis=0)


    def get_goal_dir(self, pos, goal, training=True, distracted=False):
        feature=np.zeros([1,len(self.actions)+1], dtype=np.float32)
        car_dir=goal-np.array(pos)
        car_dir[0]=0
        min_dist= np.linalg.norm(car_dir)
        #print "Distance to goal: "+str(min_dist)+" seq len: "+str(self.seq_len)
        dir = self.find_action_to_direction(car_dir,min_dist)
        feature[0,dir]=1
        feature[0,-1]=min_dist*2.0/(self.seq_len*np.sqrt(2))
        return feature

    def get_goal_dir_smooth(self, pos, goal, training=True):
        feature=np.zeros([1,len(self.actions)+1], dtype=np.float32)
        car_dir=goal-np.array(pos)
        car_dir[0]=0
        min_dist= np.linalg.norm(car_dir)
        #print "Distance to goal: "+str(min_dist)+" seq len: "+str(car_dir)
        feature[0,:-2] = self.find_action_to_direction_smooth(car_dir,min_dist)
        feature[0,-1]=min_dist*2.0/(self.seq_len*np.sqrt(2))
        #print feature
        return feature

    def get_goal_dir_cont(self, pos, goal):
        feature=np.zeros([1,2], dtype=np.float32)

        car_dir=goal-np.array(pos)
        #print "Goal "+str(goal)+" pos "+str(pos)+" "+str(car_dir)+" scaling: "+str(1.0/np.sqrt(self.reconstruction.shape[1]**2+self.reconstruction.shape[2]**2))
        feature[0,0]=car_dir[1]*1.0/np.sqrt(self.reconstruction.shape[1]**2+self.reconstruction.shape[2]**2)
        feature[0,1] = car_dir[2] * 1.0 / np.sqrt(self.reconstruction.shape[1] ** 2 + self.reconstruction.shape[2] ** 2)
        #print "Feature "+str(feature)

        return feature

    def get_goal_dir_angular(self, pos, goal):
        feature=np.zeros([1,2], dtype=np.float32)

        dir=goal[1:]-np.array(pos)[1:]
        #print "Goal "+str(goal)+" pos "+str(pos)+" "+str(car_dir)+" scaling: "+str(1.0/np.sqrt(self.reconstruction.shape[1]**2+self.reconstruction.shape[2]**2))
        feature[0,0]=np.linalg.norm(dir)*1.0/np.sqrt(self.reconstruction.shape[1]**2+self.reconstruction.shape[2]**2)
        feature[0,1] = np.arctan2(dir[0],dir[1])-(np.pi/2)
        if feature[0,1]<=-np.pi:
            feature[0,1]=feature[0,1]+2*np.pi
        feature[0, 1]=feature[0,1]/np.pi
        #print "Goal direction " + str(np.arctan2(dir[0], dir[1]) / np.pi) + " after minus " + str(feature[0, 1] )
        #print "Feature "+str(feature)

        return feature

    def get_time(self, pos, goal,frame, goal_time=-1):

        goal_displacement = goal - np.array(pos)

        goal_dist = np.linalg.norm(goal_displacement)

        if goal_time<0:
            goal_time=self.goal_time


        if frame< goal_time:
            speed=goal_dist / (goal_time - frame)
            return min(speed, 1)
        else:

            return 1

    def find_action_to_direction(self, car_dir, min_dist):
        max_cos = -1
        dir = -1
        if min_dist==0:
            return 4
        for j, action in enumerate(self.actions, 0):
            if np.linalg.norm(action)>0.1:#j != 4:

                if np.dot(action, car_dir) / (np.linalg.norm(action) * min_dist) > max_cos:
                    dir = j
                    max_cos = np.dot(action, car_dir) / (np.linalg.norm(action) * min_dist)
        return dir

    def find_action_to_direction_smooth(self, car_dir, min_dist):
        directions=np.zeros(len(self.actions)-1)

        dir = -1
        if min_dist==0:
            directions[4] =1
            return directions
        for j, action in enumerate(self.actions, 0):
            if np.linalg.norm(action)>0.1:#j != 4:
                if j>4:
                    directions[j-1] = np.dot(action, car_dir) / (np.linalg.norm(action) * min_dist)
                else:
                    directions[j ] = np.dot(action, car_dir) / (np.linalg.norm(action) * min_dist)
        #print (" Car variable directions dot product "+str(directions)+" actions "+str(self.actions) )
        return directions


class SupervisedEpisode(SimpleEpisode):

    def __init__(self, tensor, people_e, cars_e, pos_x, pos_y, gamma, seq_len, reward_weights, agent_size=(2, 2, 2),
                 people_dict={}, cars_dict={}, init_frames={}, adjust_first_frame=False,
                 follow_goal=False, action_reorder=False, threshold_dist=-1, init_frames_cars={}, temporal=False,
                 predict_future=False, run_2D=False, agent_init_velocity=False, velocity_actions=False, seq_len_pfnn=0,
                 end_collide_ped=False, stop_on_goal=True, defaultSettings = None, heroCarDetails=None):


        super(SupervisedEpisode, self).__init__(tensor, people_e, cars_e, pos_x, pos_y, gamma, seq_len, reward_weights,
                                                agent_size=agent_size, people_dict=people_dict, cars_dict=cars_dict,
                                                init_frames= init_frames, adjust_first_frame=adjust_first_frame,
                                                follow_goal=follow_goal, action_reorder=action_reorder,
                                                threshold_dist = threshold_dist, init_frames_cars = init_frames_cars,
                                                temporal = temporal, predict_future = predict_future, run_2D=run_2D,
                                                agent_init_velocity=agent_init_velocity, velocity_actions=velocity_actions,
                                                seq_len_pfnn=seq_len_pfnn, end_collide_ped=end_collide_ped, stop_on_goal=stop_on_goal, defaultSettings=defaultSettings, heroCarDetails = heroCarDetails)
        self.correct_velocities={}
        self.current_frame=0
        self.start_frame=0
        self.ped_seq_len=0

        self.allow_different_inits=False
        if self.allow_different_inits:
            self.valid_keys = []
            for person_id, trace in list(self.valid_people_tracks.items()):
                if len(trace)>1:
                    self.correct_velocities[person_id] = []
                    for indx in range(1,len(trace)):
                        vel=np.mean(trace[indx], axis=1)-np.mean(trace[indx-1], axis=1)
                        self.correct_velocities[person_id].append(np.copy(vel.astype(int)))
                self.valid_keys.append(person_id)
        else:
            for person_id in self.valid_keys:
                trace=self.valid_people_tracks[person_id]
                if len(trace)>1:
                    self.correct_velocities[person_id] = []
                    for indx in range(1,len(trace)):
                        vel=np.mean(trace[indx], axis=1)-np.mean(trace[indx-1], axis=1)
                        self.correct_velocities[person_id].append(np.copy(vel.astype(int)))
        self.key_map=self.valid_keys



    def get_valid_move(self, frame, person_id):
        return self.correct_velocities[person_id][frame]

    def get_valid_pos(self, frame, person_id):
        if frame< len(self.valid_people_tracks[person_id]):
            return np.mean(self.valid_people_tracks[person_id][frame], axis=1)
        else:
            return np.mean(self.valid_people_tracks[person_id][-1], axis=1)

    def get_valid_vel(self, frame, person_id):
        #print "Get valid vel :Difference: "+str(frame)+" "+str(np.mean(self.valid_people_tracks[person_id][frame], axis=1))+" "+str(frame+1)+" "+str(np.mean(self.valid_people_tracks[person_id][frame+1], axis=1))
        if frame+1< len(self.valid_people_tracks[person_id]):
            return np.mean(self.valid_people_tracks[person_id][frame+1]-self.valid_people_tracks[person_id][frame], axis=1)#.astype(int)
        else:
            return np.mean(self.valid_people_tracks[person_id][-1]-self.valid_people_tracks[person_id][-2], axis=1)#.astype(int)

    def get_valid_action(self, frame, person_id):
        if frame>=0 and frame<len(self.correct_velocities[person_id]):
            vel= self.correct_velocities[person_id][frame]
            #print "Vel "+str(vel)+" "+str(frame)+" "+str(np.mean(self.valid_people_tracks[person_id][frame], axis=1).astype(int))+" "+str(np.mean(self.valid_people_tracks[person_id][frame+1].astype(int), axis=1).astype(int))
            return self.find_action_to_direction( vel, np.linalg.norm(vel))
        else:
            return -1

    def get_valid_action_n(self, frame, person_id):
        if frame>=0 and frame<len(self.correct_velocities[person_id]):
            vel= np.mean(self.valid_people_tracks[person_id][frame], axis=1)-self.agent[max(frame-1,0)]
            #print "Vel "+str(vel)+" "+str(frame)+" "+str(np.mean(self.valid_people_tracks[person_id][frame], axis=1).astype(int))+" "+str(np.mean(self.valid_people_tracks[person_id][frame+1].astype(int), axis=1).astype(int))
            return self.find_action_to_direction( vel, np.linalg.norm(vel))
        else:
            return -1

    def get_valid_vel_n(self, frame, person_id):

        if frame>=0 and frame<len(self.correct_velocities[person_id]):

            vel= np.mean(self.valid_people_tracks[person_id][frame], axis=1)-self.agent[max(frame-1,0)]
            return vel[1:]
        else:
            return -1

    def save(self, statistics, num_episode, poses, ped_seq_len=-1):
        super(SupervisedEpisode, self).save(statistics, num_episode, poses, self.ped_seq_len )
        statistics[num_episode, 5, 38 + NBR_MEASURES] = self.start_frame
        statistics[num_episode, 6, 38 + NBR_MEASURES] = self.goal_person_id



    def get_valid_speed(self, frame, person_id):
        if frame>=len(self.correct_velocities[person_id]):
            #print "Frame "+str(frame)+"    "+str(len(self.correct_velocities[person_id]))
            vel = self.correct_velocities[person_id][min(frame, len(self.correct_velocities[person_id])-1)]
            return np.linalg.norm(vel)
        return 0

    def get_start_frame(self,  person_id):
        person_key = self.key_map[person_id]
        return self.init_frames[person_key]

    def initial_position(self, poses_db, initialization=-1, training=True, init_key=-1):
        print("Supervised")
        if training or initialization<0:
            self.init_method =PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian
            if len(self.valid_keys)<1:
                return [], -1, []
            if init_key not in self.valid_keys:
                init_key_val=self.valid_keys[init_key]
            else:
                init_key_val=init_key

            self.current_frame=self.init_frames[init_key_val]
            self.start_frame =self.init_frames[init_key_val]
            # print "Initialize on frame " + str(self.current_frame)+" "+str(init_key)
            # print self.valid_people_tracks[init_key]
            return self.initial_position_key(init_key)
        else:
            return super(SupervisedEpisode, self).initial_position(poses_db, initialization=initialization, training=training, init_key=init_key)

    def initial_position_n_frames(self,frame,itr):

        self.init_method =PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian
        if len(self.valid_keys)<1:
            return [], -1

        init_key_val=self.valid_keys[itr]

        self.current_frame=self.init_frames[init_key_val]
        self.start_frame =self.init_frames[init_key_val]

        if frame +1> len(self.valid_people_tracks[init_key_val]):
            return [], -1

        self.initial_position_key(init_key)
        for t in range(frame+1):
            self.agent[t]=np.array( np.mean( self.valid_people_tracks[init_key_val][t],1 ) )
            self.action[t] = self.get_valid_action(t, init_key_val)
            self.speed[t] = self.get_valid_speed(t,init_key_val)
            self.velocity[t]=self.correct_velocities[init_key_val][min(frame, len(self.correct_velocities[init_key_val])-1)]
        self.ped_seq_len=len(self.valid_people_tracks[init_key_val])
        return self.agent[frame], init_key






    def time_dependent_evaluation(self, frame, person_id=-1):
        # Intercepting cars
        if person_id<0:
            super(SupervisedEpisode, self).time_dependent_evaluation(frame)
        else:
            person_key=self.key_map[person_id]
            current_frame = self.init_frames[person_key] + frame + 1
            self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_by_car] = max(self.measures[max(frame - 1, 0), PEDESTRIAN_MEASURES_INDX.hit_by_car],
                                          self.intercept_car(current_frame, all_frames=False, agent_frame=frame+1))
            per_id = person_id


                # Coincide with human trajectory
            self.measures[frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory] = 0
            if self.intercept_pedestrian_trajectory_tensor(frame + 1,no_height=True):
                self.measures[frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory] = 1

            # Hit pedestrians
            if self.init_method <= PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian :
                if len(self.collide_with_pedestrians(current_frame, no_height=True,per_id=per_id, agent_frame=frame+1)) > 0:  # self.intercept_person(frame + 1, no_height=True, person_id=per_id, frame_input=frame):
                    self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians] = 1
            else:
                if len(self.collide_with_pedestrians(frame + 1, no_height=True)) > 0:
                    self.measures[frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians] = 1

    def measures_into_future(self, episode_done, frame):
        # Total travelled distance
        if episode_done:
            if frame < self.ped_seq_len + 1:
                for f in range(self.ped_seq_len - 1, frame, -1):
                    # print "Frame "+str(frame)+"Difference: "+str(f)+" "+str(f-1)+" "+str(self.agent[f-1])+" "+str(self.agent[f])+" "+str(np.linalg.norm(self.agent[f]-self.agent[f-1]))
                    self.measures[frame, PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init] += np.linalg.norm(self.agent[f] - self.agent[f - 1])

        # Distance to final position.
        if len(self.agent[self.ped_seq_len - 1]) > 0:
            self.measures[frame, PEDESTRIAN_MEASURES_INDX.dist_to_final_pos] = np.linalg.norm(self.agent[self.ped_seq_len - 1][1:] - self.agent[frame][1:])

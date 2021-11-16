
import numpy as np
from environment import Environment
from episode import SimpleEpisode
from visualization import make_movie_cars, make_image_cars, make_movie_paralell, plot
from random_path_generator import random_path, random_walk
from utils.utils_functions import overlap, find_border
import time
import os
from extract_tensor import frame_reconstruction
from supervised_environment import SupervisedEnvironment
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Queue as PQueue
import queue
from episode import SupervisedEpisode
from settings import run_settings

from scipy import ndimage




class CarEnvironment(Environment):
    # (self, path, sess, writer, gradBuffer, log,settings, net=None):

    def __init__(self, path,  sess, writer, gradBuffer, log,settings, net=None):
        super(CarEnvironment,self).__init__( path, sess, writer, gradBuffer, log,settings, net=net)

        self.counter=0
        self.settings=settings
        print("Settings ")
        print(settings)

        self.frame_counter=0
        self.train_frame_counter=0


        self.scene_count=0
        self.counter_train=0
        self.viz_counter=0
        self.viz_counter_test=0
        self.variations=[[0,0,0], # horizontal
                         [0,1,0], # horizontal opposite direction
                         [1, 0, 1],# vertical
                         [1, 1, 1], # certical opposite direction
                         [2, 0, 5], # diagonal
                         [2, 2, 5],  # diagonal- oposite direcion
                         [2, 1, 2], # diagonal 90deg
                         [2, 2, 2], # diagonal 90deg- opposite direction
                         [2, 1, 3],  # diagonal 180deg
                         [2, 2, 3],  # diagonal 180deg- opposite direction
                         [2, 1, 4],  # diagonal 270deg
                         [2, 2, 4],  # diagonal 270deg- opposite direction
                         ]
        self.init_methods = list(range(len(self.variations)))
        self.init_methods_train = list(range(len(self.variations)))

    # def get_repeat_reps_and_init(self, training):
    #     if training:
    #         repeat_rep = 1
    #         init_methods = self.init_methods  # self.init_methods_train
    #     else:
    #         repeat_rep = 1
    #         init_methods = self.init_methods  # self.init_methods_train
    #     return repeat_rep

    def save_people_cars(self,ep_nbr, episode, people_list, car_list):
        for frame, cars in enumerate(episode.cars):
            if len(cars)>0:
                car_list[ep_nbr, frame,:]=np.copy(cars[0])

    # Take steps, calculate reward and visualize actions of agent.


    def work(self,cachePrefix, file_name, agent, poses_db, epoch, saved_files_counter, car=None, training=True, road_width=0, conv_once=False,time_file=None, act=True,save_stats=True):


        file_agent = self.get_file_agent_path(file_name)

        if training:
            seq_len = self.settings.seq_len_train
        else:
            seq_len = self.settings.seq_len_test
        #for road_width in range(len(self.variations)):
        seq_len_pfnn = -1

        frameRate, frameTime = self.settings.getFrameRateAndTime()

        if self.settings.pfnn:

            seq_len_pfnn = int(seq_len *60//frameRate)


        episode = self.set_up_episode(None, None, 0, 0, None, training, 0, seq_len_pfnn=seq_len_pfnn)

        # pos, _ = episode.initial_position(poses_db)
        # statistics, saved_files_counter, people_list, car_list, poses, initialization_map, initialization_car,statistics_car
        statistics,saved_files_counter,people_list,car_list, poses,_,_,_ = self.act_and_learn(agent, file_agent, episode, poses_db, training, saved_files_counter,save_stats=save_stats)

        if len(statistics) > 0:
            if training:
                self.scene_count = self.scene_count + 1

            print(("Cars test statistics size" + str(len(statistics))))
            # if density >0.001* self.settings.height * self.settings.width * self.settings.depth:
            print ("Make movie")
            if (self.scene_count / self.settings.vizualization_freq_car > self.viz_counter or not training) :
                print(("Make movie " + str(training)+" "+str(self.scene_count) +" "+str(self.scene_count / self.settings.vizualization_freq_car)+" "+str(self.viz_counter))) # Debug
                self.visualize(episode, file_name, statistics, training, people_list=people_list, car_list=car_list, poses=poses)

                self.viz_counter = self.scene_count / self.settings.vizualization_freq_car

        return [], saved_files_counter

    def visualize(self, episode, file_name, statistics, training, people_list=[], car_list=[], poses=[], initialization_map=[],initialization_car=[]):
        seq_len = self.settings.seq_len_train

        if not training:
            seq_len = self.settings.seq_len_test
        if training:
            self.train_frame_counter = make_movie_cars(episode.reconstruction,
                                           [statistics],
                                           seq_len,
                                           self.train_frame_counter,
                                           self.settings.agent_shape,
                                           self.settings.agent_shape_s,
                                           episode,
                                           people_list,
                                           car_list,
                                           nbr_episodes_to_vis=5,
                                           name_movie=self.settings.name_movie+"_train_",
                                           path_results=self.settings.statistics_dir + "/agent/")
            make_image_cars(episode.reconstruction,
                            [statistics],
                            seq_len,
                            self.train_frame_counter,
                            self.settings.agent_shape,
                            self.settings.agent_shape_s,
                            episode,
                            people_list=people_list,
                            car_list=car_list,
                            nbr_episodes_to_vis=len(self.variations),
                            name_movie=self.settings.name_movie+"_train_",
                            path_results=self.settings.statistics_dir + "/agent/")
        else:
            self.frame_counter = make_movie_cars(episode.reconstruction,
                                                 [statistics],
                                                 seq_len,
                                                 self.frame_counter,
                                                 self.settings.agent_shape,
                                                 self.settings.agent_shape_s,
                                                 episode,
                                                 people_list=people_list,
                                                 car_list=car_list,
                                                 nbr_episodes_to_vis=5,
                                                 name_movie=self.settings.name_movie,
                                                 path_results=self.settings.statistics_dir + "/agent/")
            make_image_cars(episode.reconstruction,
                            [statistics],
                            seq_len,
                            self.frame_counter,
                            self.settings.agent_shape,
                            self.settings.agent_shape_s,
                            episode,
                            people_list=people_list,
                            car_list=car_list,
                            nbr_episodes_to_vis=len(self.variations),
                            name_movie=self.settings.name_movie,
                            path_results=self.settings.statistics_dir + "/agent/")

    # Return file name as to where to save statistics.
    def statistics_file_name(self, file_agent, pos_x, pos_y, training,saved_files_counter, init_meth=-1):
        if not training:
            if init_meth>0:
                return file_agent + "_cartest"+str(init_meth)+"_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)
            return file_agent + "_cartest_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)
        return  file_agent + "_car_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)

    # agent_initialization(self, agent, episode, itr, pos_init, poses_db, training, init_m=-1):
    def agent_initialization(self, agent, episode, itr,  poses_db, training, init_m=-1, set_goal="", on_car=False):
        cars, direction, people, seq_len, cars_dict, people_dict = self.get_variation(init_m, training, augment_speed=True)

        episode.cars = cars
        episode.cars_dict=cars_dict
        episode.init_frames_car={0:0}
        episode.reconstruction, episode.cars_predicted, episode.people_predicted, episode.reconstruction_2D= frame_reconstruction(np.zeros_like(episode.reconstruction), cars, people,temporal=self.settings.temporal , predict_future=self.settings.predict_future , run_2D=self.settings.run_2D)
        episode.initialization_set_up(seq_len, adjust_first_frame=False)
        episode.test_positions = find_border(episode.valid_positions)
        episode.env_nbr=init_m
        if  training :
            print("Training car env")

        else:
            print("Test car env")
        pos, indx, vel = episode.initial_position(poses_db, training=training, initialization=5)


        return pos, indx, vel


    # Set up toy episodes.
    #(self, cars, people, pos_x, pos_y, tensor, training, road_width=0, time_file=None, people_dict={}, car_dict={}, init_frames={}, init_frames_cars={})
    def set_up_episode(self, cars, people, pos_x, pos_y, tensor, training, road_width=0, time_file=None, people_dict={},
                       car_dict={}, init_frames={}, init_frames_cars={}, seq_len_pfnn=-1):
        print (" Set up car episode")
        cars, direction, people, seq_len, car_dict, people_dict = self.get_variation(road_width, training)
        ep= self.init_episode(car_dict, cars, {}, {0:0}, people_dict,
                                    people, pos_x, pos_y, seq_len_pfnn, np.zeros_like(self.reconstruction), training,None, False)
        # ep = SimpleEpisode(self.reconstruction, people, cars, pos_x, pos_y, self.settings.gamma, seq_len,
        #                    self.settings.reward_weights,
        #                    env_nbr=pos_x, width=road_width, init_frames=init_frames,
        #                    direction=direction, direction_aug=0,
        #                    init_frames_cars=init_frames_cars, temporal=self.settings.temporal ,
        #                    predict_future=self.settings.predict_future, run_2D=self.settings.run_2D,
        #                    seq_len_pfnn=seq_len_pfnn, follow_goal=self.settings.goal_dir)
        ep.test_positions=find_border(ep.valid_positions)
        #self.check_valid_positions(ep)

        return ep

    def get_variation(self, road_width, training, augment_speed=False):
        tmp = self.variations[road_width]
        direction = tmp[0]
        augment = tmp[1]
        direction_aug = tmp[2]
        # 0 - vertical
        # 1 - horizontal
        # 2- diagonal
        if training:
            seq_len = self.settings.seq_len_train
        else:
            seq_len = self.settings.seq_len_test
        # Local constants
        car_size = [8, 20]
        people = []
        cars = []
        cars_dict={}
        people_dict={}
        self.reconstruction = np.zeros((self.settings.height, max(car_size) * 3, max(car_size) * 3, 6))
        print((os.path.join(self.img_path, "car_horizontal.npy")))
        if direction == 0:
            cars_loaded = np.load(os.path.join(self.img_path, "car_horizontal.npy"))
        elif direction == 1:
            cars_loaded = np.load(os.path.join(self.img_path, "car_vertical.npy"))
        else:
            cars_loaded = np.load(os.path.join(self.img_path, "car_diagonal.npy"))
        for pos in range(seq_len):
            people.append([])
            if len(cars) < len(cars_loaded):
                cars.append(cars_loaded[pos])
            else:
                cars.append([])
        if augment > 0:
            new_list = []
            for car_list in cars:
                new_list.append([])
                for car in car_list:
                    new_list[-1].append(car)
                    temp = car.copy()
                    if direction_aug == 0:
                        new_list[-1][-1][5] = self.reconstruction.shape[2] - temp[4]
                        new_list[-1][-1][4] = self.reconstruction.shape[2] - temp[5]
                    elif direction_aug == 1:
                        new_list[-1][-1][2] = self.reconstruction.shape[1] - temp[3]
                        new_list[-1][-1][3] = self.reconstruction.shape[1] - temp[2]
                    else:
                        for i in range(direction_aug - 1):
                            car_tmp = np.copy(new_list[-1][-1])
                            temp = np.copy(new_list[-1][-1])

                            car_tmp[4] = self.reconstruction.shape[1] - temp[3]
                            car_tmp[5] = self.reconstruction.shape[1] - temp[2]
                            car_tmp[2] = temp[4]
                            car_tmp[3] = temp[5]

                            new_list[-1][-1] = np.copy(car_tmp)
                            if augment == 2:
                                car_tmp = np.copy(new_list[-1][-1])

                                new_list[-1][-1][5] = self.reconstruction.shape[1] - car_tmp[4]
                                new_list[-1][-1][4] = self.reconstruction.shape[1] - car_tmp[5]
            cars = new_list
        car_flat=[]
        for frame_list in cars:
            for car in frame_list:
                car_flat.append(car)

        # print (" Car flat!!")
        # print (car_flat)
        cars_dict = {}
        cars_dict[0] = car_flat
        if augment_speed:
            distances_list=[]
            distances_list.append(0)
            initial_pos=np.array([np.mean(car_flat[0][2:4]), np.mean(car_flat[0][4:])])
            # print (" initial car pos "+str(car_flat[0])+" mean pos "+str(initial_pos))
            for i in range(1,len(car_flat)):
                cur_pos=np.array([np.mean(car_flat[i][2:4]), np.mean(car_flat[i][4:])])
                distances_list.append(np.linalg.norm(cur_pos-initial_pos))
                # print(" initial car pos " + str(car_flat[i]) + " mean pos " + str(cur_pos))
                # print(" Distance from "+str(initial_pos)+" to "+str(cur_pos)+" : "+str(distances_list[-1]))

            random_speed=np.random.rand()#*self.settings.car_max_speed
            # print(" Random speed " + str(random_speed) +" scaled "+str(random_speed*self.settings.car_max_speed))
            random_speed=random_speed*self.settings.car_max_speed

            nex_car_pos=[]
            nex_car_pos.append(car_flat[0])
            cars=[]
            for pos in range(seq_len):
                    cars.append([])
            cars[0].append(car_flat[0])
            previous_pos=0
            next_pos=0
            for i in range(1, len(car_flat)):
                dist_travelled=i*random_speed
                #print (" In frame "+str(i)+" dist travelled "+str(dist_travelled))
                while next_pos < len(distances_list) and distances_list[next_pos]<dist_travelled:
                    next_pos=next_pos+1
                previous_pos=next_pos-1
                #print(" Done with for loop next pos is " + str(next_pos)+" and previous pos is "+str(previous_pos))
                if next_pos < len(distances_list):
                    #print(" Done with for loop next pos dist is " + str(distances_list[next_pos]) + " and previous pos dist is " + str(distances_list[previous_pos]))
                    fraction=(dist_travelled-distances_list[previous_pos])/(distances_list[next_pos]-distances_list[previous_pos])
                    #print (" Fraction is "+str(fraction )+"  nimetaja:  "+str(dist_travelled-distances_list[previous_pos])+"  jagaja:  "+str(distances_list[next_pos]-distances_list[previous_pos]))
                    new_car_pos=(1-fraction)*np.array(car_flat[previous_pos])+fraction*np.array(car_flat[next_pos])
                    #print (" New car pos "+str(new_car_pos))
                else:
                    #print (" out of list, last poses "+str(car_flat[-2])+" "+str(car_flat[-1]))
                    vel=np.array(car_flat[-1])-np.array(car_flat[-2])
                    speed=np.linalg.norm(np.array([vel[2],vel[4]]))
                    #print (" last vel "+str(vel)+" speed "+str(speed))
                    vel_to_take=vel*(random_speed/speed)
                    #print (" Adapted vel to take "+str(vel_to_take)+" adjusted by "+str(random_speed/speed))
                    new_car_pos=nex_car_pos[-1]+vel_to_take
                    #print (" New car pos "+str(new_car_pos))

                nex_car_pos.append(new_car_pos)
                cars[i].append(np.round(new_car_pos).astype(int))
                next_pos=previous_pos

            cars_dict[0] = nex_car_pos
            # print (" adjusted cars ")
            # print (nex_car_pos)
            # print ("Cars list ")
            # print (cars)
            # print (" Cars dict ")
            # print (cars_dict)



        return cars, direction, people, seq_len, cars_dict, people_dict

    def check_valid_positions(self, ep):
        v_pos = ep.valid_positions.copy()
        # define breadth
        test_positions = []
        breadth=self.settings.agent_s[1:2]
        for x in range(v_pos.shape[0]):
            for y in range(v_pos.shape[1]):
                pos = [x, y]

                min_pos = np.zeros(2, np.int)
                max_pos = np.zeros(2, np.int)
                breadth = [breadth[0], breadth[1]]

                for i in range(2):
                    min_pos[i] = pos[i] - breadth[i]
                    max_pos[i] = pos[i] + breadth[i] + 1
                    if min_pos[i] < 0:  # agent below rectangle of visible.
                        min_pos[i] = 0

                    if max_pos[i] > self.reconstruction.shape[i]:
                        max_pos[i] = self.reconstruction.shape[i]
                if np.sum(v_pos[min_pos[0]:max_pos[0], min_pos[1]:max_pos[1]]) > 0:
                    ep.valid_positions[x, y] = 0
                if x > 0 and y > 0:
                    if ep.valid_positions[x - 1, y] == 1 and ep.valid_positions[x, y] == 0:
                        test_positions.append([x - 1, y])
                    elif x > 0 and y > 0 and ep.valid_positions[x - 1, y] == 0 and ep.valid_positions[x, y] == 1:
                        test_positions.append([x, y])
                    elif x > 0 and y > 0 and ep.valid_positions[x, y - 1] == 1 and ep.valid_positions[x, y] == 0:
                        test_positions.append([x, y - 1])
                    elif x > 0 and y > 0 and ep.valid_positions[x, y - 1] == 0 and ep.valid_positions[x, y] == 1:
                        test_positions.append([x, y])
            ep.test_positions = test_positions

    def create_person(self,y_p,x_p,w_p):
        #[self.settings.width / 2 - 2, self.settings.width / 2 + 2], [frame, frame + 2]]
        return np.array([[0, 8],[y_p-w_p,y_p+w_p], [x_p-w_p, x_p+w_p]] ).reshape(3, 2)

    def set_values(self, indx, val):
        tmp=self.reconstruction[indx[0]:indx[1], indx[2]:indx[3], indx[4]:indx[5],3].shape
        self.reconstruction[indx[0]:indx[1], indx[2]:indx[3], indx[4]:indx[5],3]=val*np.ones(tmp)

class PeopleEnvironment(CarEnvironment):

    def __init__(self,  path,  sess, writer, gradBuffer, log, settings, net=None):
        super(PeopleEnvironment, self).__init__( path,  sess, writer, gradBuffer, log, settings, net=net)
        self.counter = 0
        self.settings = settings
        # self.init_methods = range(len(variations))
        # self.init_methods_train = range(len(variations))

    def statistics_file_name(self, file_agent, pos_x, pos_y, training, saved_files_counter,init_meth=-1):
        if not training:

            return file_agent + "_peopletest_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)
        return file_agent + "_people_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)

    #(agent, episode, itr, poses_db, training, init_m = initialization)
    def agent_initialization(self, agent, episode, itr,  poses_db, training, init_m=-1, set_goal="", on_car=False):
        cars, direction, init_frames, people, people_dict, seq_len = self.get_variation(init_m, training)
        episode.people = people
        episode.people_dict=people_dict
        episode.reconstruction, episode.cars_predicted, episode.people_predicted, episode.reconstruction_2D= frame_reconstruction(np.zeros_like(episode.reconstruction), cars, people,temporal=self.settings.temporal , predict_future=self.settings.predict_future, run_2D=self.settings.run_2D)
        episode.initialization_set_up( seq_len, adjust_first_frame=False)
        episode.test_positions = find_border(episode.valid_positions)
        episode.env_nbr = direction
        if not training and len(episode.valid_keys) > 0:
            print("Training people env")

        else:

            print("Testing people env")
        pos, indx, vel = episode.initial_position(poses_db, training=training, initialization=3)



        return pos, indx, vel

    def save_people_cars(self, ep_nbr, episode, people_list, car_list):
        for frame, cars in enumerate(episode.people):
            if len(cars) > 0:
                people_list[ep_nbr, frame, 0:2] = np.copy(cars[0][0])
                people_list[ep_nbr, frame, 2:4] = np.copy(cars[0][1])
                people_list[ep_nbr, frame, 4:] = np.copy(cars[0][2])
    # Set up toy episodes.
    def set_up_episode(self, cars, people, pos_x, pos_y, tensor, training, road_width=0, time_file=None,
                       people_dict={},
                       car_dict={}, valid_ids=None, direction=0, augment=0, seq_len_pfnn=-1, init_frames_cars={}):
        cars, direction, init_frames, people, people_dict, seq_len = self.get_variation( road_width,
                                                                                        training)

        ep= self.init_episode(car_dict, cars, init_frames, {}, people_dict,
                                    people, pos_x, pos_y, seq_len_pfnn,np.zeros_like(self.reconstruction), training, None, False)
        ep.test_positions = find_border(ep.valid_positions)



        
        return ep


    def get_variation(self ,road_width, training):
        tmp = self.variations[road_width]
        direction = tmp[0]
        augment = tmp[1]
        direction_aug = tmp[2]
        print(("Direction " + str(direction) + " augnment " + str(augment) + " direction aug " + str(direction_aug)))
        # 0 - vertical
        # 1 - horizontal
        # 2- diagonal
        if training:
            seq_len = self.settings.seq_len_train
        else:
            seq_len = self.settings.seq_len_test

        # Local constants
        car_size = [8, 10]
        people = []
        cars = []
        self.reconstruction = np.zeros((self.settings.height, max(car_size) * 3, max(car_size) * 3, 6))

        if direction == 0:
            people_loaded = np.load(os.path.join(self.img_path, "people_horizontal.npy"))
        elif direction == 1:
            people_loaded = np.load(os.path.join(self.img_path, "people_vertical.npy"))
        else:
            people_loaded = np.load(os.path.join(self.img_path, "people_diagonal.npy"))
        for pos in range(seq_len):
            cars.append([])
            if len(people) < len(people_loaded):
                people.append(people_loaded[pos])
            else:
                people.append([])
        if augment > 0:
            new_list = []
            for car_list in people:
                new_list.append([])
                for car in car_list:
                    new_list[-1].append(car)
                    temp = car.copy()
                    if direction_aug == 0:
                        new_list[-1][-1][2][1] = self.reconstruction.shape[2] - temp[2][0]
                        new_list[-1][-1][2][0] = self.reconstruction.shape[2] - temp[2][1]
                    elif direction_aug == 1:
                        new_list[-1][-1][1][0] = self.reconstruction.shape[1] - temp[1][1]
                        new_list[-1][-1][1][1] = self.reconstruction.shape[1] - temp[1][0]
                    else:
                        for i in range(direction_aug - 1):
                            car_tmp = np.copy(new_list[-1][-1])
                            temp = np.copy(new_list[-1][-1])

                            car_tmp[2][0] = self.reconstruction.shape[1] - temp[1][1]
                            car_tmp[2][1] = self.reconstruction.shape[1] - temp[1][0]
                            car_tmp[1][0] = temp[2][0]
                            car_tmp[1][1] = temp[2][1]

                            new_list[-1][-1] = np.copy(car_tmp)
                            if augment == 2:
                                car_tmp = np.copy(new_list[-1][-1])

                                new_list[-1][-1][2][1] = self.reconstruction.shape[1] - car_tmp[2][0]
                                new_list[-1][-1][2][0] = self.reconstruction.shape[1] - car_tmp[2][1]
            people = new_list
        people_flat = []
        for frame_list in people:
            for person in frame_list:
                people_flat.append(person)
        people_dict = {}
        people_dict[0] = people_flat
        init_frames = {0: 0}
        return cars, direction, init_frames, people, people_dict, seq_len

    def create_person(self, y_p, x_p, w_p):
        # [self.settings.width / 2 - 2, self.settings.width / 2 + 2], [frame, frame + 2]]
        return np.array([[0, 8], [y_p - w_p, y_p + w_p], [x_p - w_p, x_p + w_p]]).reshape(3, 2)

    def set_values(self, indx, val):
        tmp = self.reconstruction[indx[0]:indx[1], indx[2]:indx[3], indx[4]:indx[5], 3].shape
        self.reconstruction[indx[0]:indx[1], indx[2]:indx[3], indx[4]:indx[5], 3] = val * np.ones(tmp)


class CarEnvironmentSupervised(SupervisedEnvironment):
    def __init__(self, path,  sess, writer, gradBuffer, log, settings, net=None):
        super(CarEnvironmentSupervised, self).__init__(path, sess, writer, gradBuffer, log, settings, net=net)
        print((super(CarEnvironmentSupervised, self)))
        self.counter = 0
        self.settings = settings

        self.frame_counter = 0
        self.train_frame_counter = 0

        self.scene_count = 0
        self.counter_train = 0
        self.viz_counter = 0
        self.viz_counter_test = 0
        self.variations = [[0, 0, 0],  # horizontal
                           [0, 1, 0],  # horizontal opposite direction
                           [1, 0, 1],  # vertical
                           [1, 1, 1],  # certical opposite direction
                           [2, 0, 5],  # diagonal
                           [2, 2, 5],  # diagonal- oposite direcion
                           [2, 1, 2],  # diagonal 90deg
                           [2, 2, 2],  # diagonal 90deg- opposite direction
                           [2, 1, 3],  # diagonal 180deg
                           [2, 2, 3],  # diagonal 180deg- opposite direction
                           [2, 1, 4],  # diagonal 270deg
                           [2, 2, 4],  # diagonal 270deg- opposite direction
                           ]
        self.init_methods = list(range(len(self.variations)))
        self.init_methods_train = list(range(len(self.variations)))

    # def get_repeat_reps_and_init(self, training):
    #     if training:
    #         repeat_rep = 1
    #         init_methods = self.init_methods  # self.init_methods_train
    #     else:
    #         repeat_rep = 1
    #         init_methods = self.init_methods  # self.init_methods_train
    #     return repeat_rep

    def save_people_cars(self, ep_nbr, episode, people_list, car_list):
        for frame, cars in enumerate(episode.cars):
            if len(cars) > 0:
                car_list[ep_nbr, frame, :] = np.copy(cars[0])

    # Take steps, calculate reward and visualize actions of agent.
    def work(self, file_name, agent, poses_db, epoch, saved_files_counter, car=None,training=True, road_width=0, conv_once=False,
             time_file=None, act=True, save_stats=True):

        file_agent = self.get_file_agent_path(file_name)

        if training:
            seq_len = self.settings.seq_len_train
        else:
            seq_len = self.settings.seq_len_test
        # for road_width in range(len(self.variations)):

        frameRate, frameTime = self.settings.getFrameRateAndTime()
        seq_len_pfnn = -1
        if self.settings.pfnn:
            seq_len_pfnn = seq_len  * 60//frameRate

        episode = self.set_up_episode(None, None, 0, 0, None, training, 0, seq_len_pfnn=seq_len_pfnn)

        # pos, _ = episode.initial_position(poses_db)

        statistics, saved_files_counter, people_list, car_list, poses,_,_,_ = super(CarEnvironmentSupervised, self).act_and_learn(agent, file_agent, episode,
                                                                                    poses_db, training,
                                                                                    saved_files_counter, save_stats=save_stats)
        print(("Cars test statistics size" + str(len(statistics))))
        if len(statistics) > 0:
            if training:
                self.scene_count = self.scene_count + 1

            print(("Cars test statistics size" + str(len(statistics))))
            # if density >0.001* self.settings.height * self.settings.width * self.settings.depth:
            if (self.scene_count / self.settings.vizualization_freq_car > self.viz_counter or not training):
                print(("Make movie " + str(training) + " " + str(self.scene_count) + " " + str(
                    self.scene_count / self.settings.vizualization_freq_car) + " " + str(self.viz_counter)))  # Debug
                self.visualize(episode, file_name, statistics, training, people_list=people_list, car_list=car_list, poses=poses)# people_list=people_list, car_list=car_list)

                self.viz_counter = self.scene_count / self.settings.vizualization_freq_car

        return [], saved_files_counter

    def visualize(self, episode, file_name, statistics, training, people_list=[], car_list=[], poses=[], initialization_map=[], initialization_car=[]):
        seq_len = self.settings.seq_len_train
        if not training:
            seq_len = self.settings.seq_len_test
        if training:
            self.train_frame_counter = make_movie_cars(episode.reconstruction,
                                                       [statistics],
                                                       seq_len,
                                                       self.train_frame_counter,
                                                       self.settings.agent_shape,
                                                       self.settings.agent_shape_s,
                                                       episode,
                                                       people_list,
                                                       car_list,
                                                       nbr_episodes_to_vis=5,
                                                       name_movie=self.settings.name_movie + "_train_",
                                                       path_results=self.settings.statistics_dir + "/agent/")
            make_image_cars(episode.reconstruction,
                            [statistics],
                            seq_len,
                            self.train_frame_counter,
                            self.settings.agent_shape,
                            self.settings.agent_shape_s,
                            episode,
                            people_list=people_list,
                            car_list=car_list,
                            nbr_episodes_to_vis=len(self.variations),
                            name_movie=self.settings.name_movie + "_train_",
                            path_results=self.settings.statistics_dir + "/agent/")
        else:
            self.frame_counter = make_movie_cars(episode.reconstruction,
                                                 [statistics],
                                                 seq_len,
                                                 self.frame_counter,
                                                 self.settings.agent_shape,
                                                 self.settings.agent_shape_s,
                                                 episode,
                                                 people_list=people_list,
                                                 car_list=car_list,
                                                 nbr_episodes_to_vis=5,
                                                 name_movie=self.settings.name_movie,
                                                 path_results=self.settings.statistics_dir + "/agent/")
            make_image_cars(episode.reconstruction,
                            [statistics],
                            seq_len,
                            self.frame_counter,
                            self.settings.agent_shape,
                            self.settings.agent_shape_s,
                            episode,
                            people_list=people_list,
                            car_list=car_list,
                            nbr_episodes_to_vis=len(self.variations),
                            name_movie=self.settings.name_movie,
                            path_results=self.statistics_dir + "/agent/")

    # Return file name as to where to save statistics.
    def statistics_file_name(self, file_agent, pos_x, pos_y, training, saved_files_counter, init_meth=-1):
        if not training:
            if init_meth > 0:
                return file_agent + "_cartest" + str(init_meth) + "_" + str(pos_x) + "_" + str(pos_y) + "_" + str(
                    saved_files_counter)
            return file_agent + "_cartest_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)
        return file_agent + "_car_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)

    # agent_initialization(self, agent, episode, itr, pos_init, poses_db, training, init_m=-1):
    def agent_initialization(self, agent, episode, itr, poses_db, training, init_m=-1, set_goal="", on_car=False):
        cars, direction, people, seq_len = self.get_variation(init_m, training)
        episode.cars = cars
        episode.reconstruction,episode.cars_predicted, episode.people_predicted, episode.reconstruction_2D = frame_reconstruction(np.zeros_like(episode.reconstruction), cars, people,temporal=self.settings.temporal , predict_future=self.settings.predict_future, run_2D=self.settings.run_2D)
        # semantic_class, seq_len, adjust_first_frame
        episode.initialization_set_up(  seq_len, adjust_first_frame=False)
        episode.test_positions = find_border(episode.valid_positions)
        episode.env_nbr = init_m
        if training:
            print("Training car env")

        else:
            print("Test car env")
        pos, indx, vel = episode.initial_position(poses_db, training=training, initialization=5)

        return pos, indx, vel

    # Set up toy episodes.
    # self, cars, people, pos_x, pos_y, tensor, training, road_width=0, time_file=None, people_dict={}, car_dict={}, init_frames={}, init_frames_cars={}):
    def set_up_episode(self, cars, people, pos_x, pos_y, tensor, training, road_width=0, time_file=None, people_dict={},
                       car_dict={},init_frames={},  init_frames_cars={}, seq_len_pfnn=-1):

        cars, direction, people, seq_len = self.get_variation(road_width, training)
        ep = SupervisedEpisode(self.reconstruction, people, cars, pos_x, pos_y, self.settings.gamma, seq_len,
                               self.settings.reward_weights,
                               temporal=self.settings.temporal, predict_future=self.settings.predict_future,
                               run_2D=self.settings.run_2D, seq_len_pfnn=seq_len_pfnn)
        ep.test_positions = find_border(ep.valid_positions)
        # self.check_valid_positions(ep)

        return ep

    def get_variation(self, road_width, training):
        tmp = self.variations[road_width]
        direction = tmp[0]
        augment = tmp[1]
        direction_aug = tmp[2]
        # 0 - vertical
        # 1 - horizontal
        # 2- diagonal
        if training:
            seq_len = self.settings.seq_len_train
        else:
            seq_len = self.settings.seq_len_test
        # Local constants
        car_size = [8, 20]
        people = []
        cars = []
        self.reconstruction = np.zeros((self.settings.height, max(car_size) * 3, max(car_size) * 3, 6))
        if direction == 0:
            cars_loaded = np.load(os.path.join(self.img_path, "car_horizontal.npy"))
        elif direction == 1:
            cars_loaded = np.load(os.path.join(self.img_path, "car_vertical.npy"))
        else:
            cars_loaded = np.load(os.path.join(self.img_path, "car_diagonal.npy"))
        for pos in range(seq_len):
            people.append([])
            if len(cars) < len(cars_loaded):
                cars.append(cars_loaded[pos])
            else:
                cars.append([])
        if augment > 0:
            new_list = []
            for car_list in cars:
                new_list.append([])
                for car in car_list:
                    new_list[-1].append(car)
                    temp = car.copy()
                    if direction_aug == 0:
                        new_list[-1][-1][5] = self.reconstruction.shape[2] - temp[4]
                        new_list[-1][-1][4] = self.reconstruction.shape[2] - temp[5]
                    elif direction_aug == 1:
                        new_list[-1][-1][2] = self.reconstruction.shape[1] - temp[3]
                        new_list[-1][-1][3] = self.reconstruction.shape[1] - temp[2]
                    else:
                        for i in range(direction_aug - 1):
                            car_tmp = np.copy(new_list[-1][-1])
                            temp = np.copy(new_list[-1][-1])

                            car_tmp[4] = self.reconstruction.shape[1] - temp[3]
                            car_tmp[5] = self.reconstruction.shape[1] - temp[2]
                            car_tmp[2] = temp[4]
                            car_tmp[3] = temp[5]

                            new_list[-1][-1] = np.copy(car_tmp)
                            if augment == 2:
                                car_tmp = np.copy(new_list[-1][-1])

                                new_list[-1][-1][5] = self.reconstruction.shape[1] - car_tmp[4]
                                new_list[-1][-1][4] = self.reconstruction.shape[1] - car_tmp[5]
            cars = new_list
        return cars, direction, people, seq_len

    def check_valid_positions(self, ep):
        v_pos = ep.valid_positions.copy()
        # define breadth
        test_positions = []
        breadth = self.settings.agent_s[1:2]
        for x in range(v_pos.shape[0]):
            for y in range(v_pos.shape[1]):
                pos = [x, y]

                min_pos = np.zeros(2, np.int)
                max_pos = np.zeros(2, np.int)
                breadth = [breadth[0], breadth[1]]

                for i in range(2):
                    min_pos[i] = pos[i] - breadth[i]
                    max_pos[i] = pos[i] + breadth[i] + 1
                    if min_pos[i] < 0:  # agent below rectangle of visible.
                        min_pos[i] = 0

                    if max_pos[i] > self.reconstruction.shape[i]:
                        max_pos[i] = self.reconstruction.shape[i]
                if np.sum(v_pos[min_pos[0]:max_pos[0], min_pos[1]:max_pos[1]]) > 0:
                    ep.valid_positions[x, y] = 0
                if x > 0 and y > 0:
                    if ep.valid_positions[x - 1, y] == 1 and ep.valid_positions[x, y] == 0:
                        test_positions.append([x - 1, y])
                    elif x > 0 and y > 0 and ep.valid_positions[x - 1, y] == 0 and ep.valid_positions[x, y] == 1:
                        test_positions.append([x, y])
                    elif x > 0 and y > 0 and ep.valid_positions[x, y - 1] == 1 and ep.valid_positions[x, y] == 0:
                        test_positions.append([x, y - 1])
                    elif x > 0 and y > 0 and ep.valid_positions[x, y - 1] == 0 and ep.valid_positions[x, y] == 1:
                        test_positions.append([x, y])
            ep.test_positions = test_positions

    def create_person(self, y_p, x_p, w_p):
        # [self.settings.width / 2 - 2, self.settings.width / 2 + 2], [frame, frame + 2]]
        return np.array([[0, 8], [y_p - w_p, y_p + w_p], [x_p - w_p, x_p + w_p]]).reshape(3, 2)

    def set_values(self, indx, val):
        tmp = self.reconstruction[indx[0]:indx[1], indx[2]:indx[3], indx[4]:indx[5], 3].shape
        self.reconstruction[indx[0]:indx[1], indx[2]:indx[3], indx[4]:indx[5], 3] = val * np.ones(tmp)


class PeopleEnvironmentSupervised(CarEnvironmentSupervised):
    def __init__(self, cachePrefix, path,  sess, writer, gradBuffer, log, settings, net=None):
        super(PeopleEnvironmentSupervised, self).__init__(path, cachePrefix,  sess, writer, gradBuffer, log, settings, net=net)
        self.counter = 0
        self.settings = settings
        # self.init_methods = range(len(variations))
        # self.init_methods_train = range(len(variations))

    def statistics_file_name(self, file_agent, pos_x, pos_y, training, saved_files_counter, init_meth=-1):
        if not training:
            return file_agent + "_peopletest_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)
        return file_agent + "_people_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)

    # (agent, episode, itr, poses_db, training, init_m = initialization)
    def agent_initialization(self, agent, episode, itr, poses_db, training, init_m=-1, set_goal="", on_car=False):
        cars, direction, init_frames, people, people_dict, seq_len = self.get_variation(init_m, training)
        episode.people = people
        episode.people_dict = people_dict
        episode.init_frames = {0: 0}
        episode.reconstruction, episode.cars_predicted, episode.people_predicted, episode.reconstruction_2D= frame_reconstruction(episode.reconstruction, cars, people,temporal=self.settings.temporal , predict_future=self.settings.predict_future)
        episode.initialization_set_up(seq_len, adjust_first_frame=False)
        episode.test_positions = find_border(episode.valid_positions)
        episode.env_nbr = direction
        if not training and len(episode.valid_keys) > 0:
            print("Training people env")

        else:
            print("Testing people env")
        pos, indx, vel = episode.initial_position(poses_db, training=training, initialization=6)

        return pos, indx, vel

    def save_people_cars(self, ep_nbr, episode, people_list, car_list):
        for frame, cars in enumerate(episode.people):
            if len(cars) > 0:
                people_list[ep_nbr, frame, 0:2] = np.copy(cars[0][0])
                people_list[ep_nbr, frame, 2:4] = np.copy(cars[0][1])
                people_list[ep_nbr, frame, 4:] = np.copy(cars[0][2])

    # Set up toy episodes.
    def set_up_episode(self, cars, people, pos_x, pos_y, tensor, training, road_width=0, time_file=None,
                       people_dict={},
                       car_dict={}, valid_ids=None, direction=0, augment=0, seq_len_pfnn=-1):
        cars, direction, init_frames, people, people_dict, seq_len = self.get_variation(road_width,
                                                                                        training)

        ep = SupervisedEpisode(self.reconstruction, people, cars, pos_x, pos_y, self.settings.gamma, seq_len,
                               reward_weights=self.settings.reward_weights, people_dict=people_dict, init_frames=init_frames,
                               temporal=self.settings.temporal, predict_future=self.settings.predict_future,
                               run_2D=self.settings.run_2D, seq_len_pfnn=seq_len_pfnn)
        ep.test_positions = find_border(ep.valid_positions)

        return ep

    def get_variation(self, road_width, training):
        tmp = self.variations[road_width]
        direction = tmp[0]
        augment = tmp[1]
        direction_aug = tmp[2]
        print(("Direction " + str(direction) + " augnment " + str(augment) + " direction aug " + str(direction_aug)))
        # 0 - vertical
        # 1 - horizontal
        # 2- diagonal
        if training:
            seq_len = self.settings.seq_len_train
        else:
            seq_len = self.settings.seq_len_test

        # Local constants
        car_size = [8, 10]
        people = []
        cars = []
        self.reconstruction = np.zeros((self.settings.height, max(car_size) * 3, max(car_size) * 3, 6))

        if direction == 0:
            people_loaded = np.load(os.path.join(self.img_path, "people_horizontal.npy"))
        elif direction == 1:
            people_loaded = np.load(os.path.join(self.img_path, "people_vertical.npy"))
        else:
            people_loaded = np.load(os.path.join(self.img_path, "people_diagonal.npy"))
        for pos in range(seq_len):
            cars.append([])
            if len(people) < len(people_loaded):
                people.append(people_loaded[pos])
            else:
                people.append([])
        if augment > 0:
            new_list = []
            for car_list in people:
                new_list.append([])
                for car in car_list:
                    new_list[-1].append(car)
                    temp = car.copy()
                    if direction_aug == 0:
                        new_list[-1][-1][2][1] = self.reconstruction.shape[2] - temp[2][0]
                        new_list[-1][-1][2][0] = self.reconstruction.shape[2] - temp[2][1]
                    elif direction_aug == 1:
                        new_list[-1][-1][1][0] = self.reconstruction.shape[1] - temp[1][1]
                        new_list[-1][-1][1][1] = self.reconstruction.shape[1] - temp[1][0]
                    else:
                        for i in range(direction_aug - 1):
                            car_tmp = np.copy(new_list[-1][-1])
                            temp = np.copy(new_list[-1][-1])

                            car_tmp[2][0] = self.reconstruction.shape[1] - temp[1][1]
                            car_tmp[2][1] = self.reconstruction.shape[1] - temp[1][0]
                            car_tmp[1][0] = temp[2][0]
                            car_tmp[1][1] = temp[2][1]

                            new_list[-1][-1] = np.copy(car_tmp)
                            if augment == 2:
                                car_tmp = np.copy(new_list[-1][-1])

                                new_list[-1][-1][2][1] = self.reconstruction.shape[1] - car_tmp[2][0]
                                new_list[-1][-1][2][0] = self.reconstruction.shape[1] - car_tmp[2][1]
            people = new_list
        people_flat = []
        for frame_list in people:
            for person in frame_list:
                people_flat.append(person)
        people_dict = {}
        people_dict[0] = people_flat
        init_frames = {0: 0}
        return cars, direction, init_frames, people, people_dict, seq_len

    def create_person(self, y_p, x_p, w_p):
        # [self.settings.width / 2 - 2, self.settings.width / 2 + 2], [frame, frame + 2]]
        return np.array([[0, 8], [y_p - w_p, y_p + w_p], [x_p - w_p, x_p + w_p]]).reshape(3, 2)

    def set_values(self, indx, val):
        tmp = self.reconstruction[indx[0]:indx[1], indx[2]:indx[3], indx[4]:indx[5], 3].shape
        self.reconstruction[indx[0]:indx[1], indx[2]:indx[3], indx[4]:indx[5], 3] = val * np.ones(tmp)



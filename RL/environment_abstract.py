import os
import numpy as np
if True: # Set this to False! or comment it out
    from visualization import make_movie
from episode import SimpleEpisode
from extract_tensor import objects_in_range, extract_tensor, objects_in_range_map
from settings import USE_LOCAL_PATHS, NBR_MEASURES, NBR_MEASURES_CAR, NBR_STATS, NBR_POSES, NBR_MAPS, NBR_STATS_CAR, NBR_CAR_MAP,STATISTICS_INDX_MAP, NBR_MAP_STATS, STATISTICS_INDX_MAP_STAT,CAR_MEASURES_INDX,PEDESTRIAN_MEASURES_INDX,STATISTICS_INDX_CAR, STATISTICS_INDX,  CAR_REWARD_INDX,NUM_SEM_CLASSES, CHANNELS
import time
import pickle
import joblib
import scipy
from dotmap import DotMap
import colmap.reconstruct
from episode import SupervisedEpisode
import copy

from scipy import ndimage
from commonUtils import ReconstructionUtils
from commonUtils.ReconstructionUtils import  SIDEWALK_LABELS, ROAD_LABELS


class AbstractEnvironment(object):

    #(episode, file_name, statistics, training, poses,initialization_map, initialization_car, agent, statistics_car)
    def visualize(self, episode, file_name, statistics, training, poses,initialization_map,initialization_car, agent, statistics_car):

        seq_len=self.settings.seq_len_train
        if not training:
            seq_len =self.settings.seq_len_test
        jointsParents = None
        if self.settings.pfnn:
            jointsParents = agent.PFNN.getJointParents()
        self.frame_counter = make_movie(episode.people,
                                        episode.reconstruction,
                                        [statistics],
                                        self.settings.width,
                                        self.settings.depth,
                                        seq_len,
                                        self.frame_counter,
                                        self.settings.agent_shape,
                                        self.settings.agent_shape_s, episode,
                                        poses,
                                        initialization_map,
                                        initialization_car,
                                        statistics_car,
                                        car_goal=episode.car_goal,
                                        training=training,
                                        name_movie=self.settings.name_movie,
                                        path_results=self.settings.statistics_dir + "/agent/",
                                        episode_name=os.path.basename(file_name),
                                        action_reorder=self.settings.reorder_actions,
                                        velocity_show=self.settings.velocity,
                                        velocity_sigma=self.settings.velocity_sigmoid,
                                        jointsParents=jointsParents,
                                        continous=self.settings.continous,
                                        goal=self.settings.learn_goal,
                                        gaussian_goal=self.settings.goal_gaussian)

    def save_people_cars(self,ep_nbr, episode, people_list, car_list):
        pass


    # gets the default camera x,y pos in the world
    def default_camera_pos(self):
        return 0, (-self.settings.width // 2)

    def default_seq_lengths(self, training, evaluate):
        # Set the seq length
        if evaluate:
            seq_len = self.settings.seq_len_evaluate
        else:
            if training:
                seq_len = self.settings.seq_len_train
            else:
                seq_len = self.settings.seq_len_test

        frameRate, frameTime = self.settings.getFrameRateAndTime()
        seq_len_pfnn = -1
        if self.settings.pfnn:
            seq_len_pfnn = seq_len * 60 // frameRate


        print(("Environment seq Len {}. seq Len pfnn {}".format(seq_len, seq_len_pfnn)))
        return seq_len, seq_len_pfnn

    def default_initialize_agent(self, training, agent, episode, poses_db):
        succedeed = False

        # Initialize the agent at a position by the given parameters
        if training:
            test_init = 1 in self.init_methods_train or 3 in self.init_methods_train
        else:
            test_init = 1 in self.init_methods_train or 3 in self.init_methods_train

        isEpisodeRunValid = (episode.people_dict is not None) and len(episode.valid_keys) > 0
        #print len(episode.valid_keys)

        if (len(self.init_methods) == 1 and not training and self.init_methods[0] > 1) or (
                len(self.init_methods_train) == 1 and training and self.init_methods[0] > 1):
            initialization = self.get_next_initialization(training)
            if self.settings.learn_init:
                return test_init, True, isEpisodeRunValid
            pos, indx, vel = self.agent_initialization(agent, episode, 0, poses_db, training,
                                                       init_m=initialization)
            if len(pos) == 0:
                succedeed = False
                return test_init, False, isEpisodeRunValid

        succedeed = True
        return test_init, succedeed, isEpisodeRunValid

    def default_doActAndGetStats(self, training, number_of_runs_per_scene, agent, episode, poses_db, file_agent, file_name, saved_files_counter,
                                 outStatsGather = None, evaluate = False, car=None, save_stats=True, iterative_training=None,viz=False):
        if iterative_training:
            print("In default_doActAndGetStats train car:" + str(iterative_training.train_car) + "  train initializer " + str(
                iterative_training.train_initializer))
        for number_of_times in range(number_of_runs_per_scene):
            statistics, saved_files_counter, _, _, poses,initialization_map,initialization_car,statistics_car, = self.act_and_learn(agent, file_agent, episode,
                                                                                  poses_db, training,
                                                                                  saved_files_counter,car=car, save_stats=save_stats, iterative_training=iterative_training, evaluate=evaluate,viz=viz)

            if evaluate == False:
                if len(statistics) > 0:
                    if training:
                        self.scene_count = self.scene_count + 1
                    else:
                        self.scene_count_test = self.scene_count_test + 1
                    # if density >0.001* self.settings.height * self.settings.width * self.settings.depth:
                    if self.scene_count == 1 or (
                                self.scene_count // self.settings.vizualization_freq > self.viz_counter and training) or (
                                self.scene_count_test // self.settings.vizualization_freq_test > self.viz_counter_test and not training):
                        # print "Make movie "+str(training) # Debug
                        self.visualize(episode, file_name, statistics, training, poses,initialization_map, initialization_car, agent, statistics_car)
                        if training:
                            self.viz_counter = self.scene_count // self.settings.vizualization_freq
                        else:
                            self.viz_counter_test = self.scene_count_test // self.settings.vizualization_freq_test
            else:
                # COMMENT THIS OUT!
                # if  len(statistics)>0 and self.scene_count_test / self.settings.vizualization_freq_test > self.viz_counter_test:
                #     #if density >0.001* self.settings.height * self.settings.width * self.settings.depth:
                #     self.counter = make_movie_eval(episode.people,
                #                               episode.reconstruction,
                #                               [statistics],
                #                               self.settings.width,
                #                               self.settings.depth,
                #                               self.settings.seq_len_evaluate,
                #                               self.counter,
                #                               self.settings.agent_shape,
                #                               self.settings.agent_shape_s, episode,
                #                               poses,
                #                               name_movie=self.settings.name_movie_eval+self.settings.name_movie)
                self.scene_count_test += 1
                outStatsGather.append(statistics)
        if not training and self.settings.useRLToyCar or self.settings.useHeroCar:
            initializer_stats=DotMap()
            successes=np.zeros_like(statistics_car[:,0,STATISTICS_INDX_CAR.measures[0]+CAR_MEASURES_INDX.goal_reached])
            successes[np.any(statistics_car[:,:,STATISTICS_INDX_CAR.measures[0]+CAR_MEASURES_INDX.goal_reached]>0, axis=1)]=1
            initializer_stats.success_rate_car=np.mean(successes)
            collisions = np.zeros_like(statistics[:, 0, STATISTICS_INDX.measures[0]+PEDESTRIAN_MEASURES_INDX.hit_by_car])
            collisions[np.any(statistics[:, :, STATISTICS_INDX.measures[0]+PEDESTRIAN_MEASURES_INDX.hit_by_car] > 0, axis=1)] = 1
            initializer_stats.collision_rate_initializer = np.mean(collisions)
        else:
            initializer_stats=None

        return saved_files_counter, initializer_stats


    def  parseRealTimeEnvObservation(self, observation, episode):
        # We need pedestrians, cars and their velocities.
        # we also need hero cars position


        # Let's get some other simulation data if needed here
        # print (" Car init dir in env " + str(observation.car_init_dir))
        episode.update_pedestrians_and_cars(observation.frame, observation.heroCarPos, observation.heroCarVel,  observation.heroCarAngle,observation.heroCarGoal,observation.heroCarBBox,  observation.people_dict, observation.cars_dict, observation.pedestrian_vel_dict, observation.car_vel_dict, observation.measures, observation.reward, observation.probabilities,observation.car_init_dir , observation.heroCarAction)

    # Main loop is here! Perform actions in environment
    def act_and_learn(self, agent, agent_file, episode, poses_db, training,saved_files_counter,  road_width=0, curriculum_prob=0, time_file=None, pos_init=[], viz=False, set_goal="", viz_frame=-1, car=None,save_stats=True, iterative_training=None,evaluate=False):
        if iterative_training!=None:
            print("In act_and_learn train car:" + str(
                iterative_training.train_car) + "  train initializer " + str(
                iterative_training.train_initializer))
        # initialize folders
        num_episode = 0
        repeat_rep = self.get_repeat_reps_and_init(training)
        statistics = np.zeros((repeat_rep, (episode.seq_len - 1),NBR_STATS), dtype=np.float64)
        print(" stats shape "+str(statistics.shape))
        poses = np.zeros((repeat_rep, episode.seq_len * 60 // episode.frame_rate + episode.seq_len, NBR_POSES),
                             dtype=np.float64)
        initialization_map=[]
        initialization_car=[]

        if self.settings.learn_init:
            initialization_map=np.zeros((repeat_rep,self.settings.env_shape[1]*self.settings.env_shape[2] ,NBR_MAPS), dtype=np.float64)
            initialization_car = np.zeros((repeat_rep,  NBR_CAR_MAP),dtype=np.float64)

        if self.settings.useRealTimeEnv or self.settings.useRLToyCar:
            statistics_car = np.zeros((repeat_rep, (episode.seq_len - 1), NBR_STATS_CAR), dtype=np.float64)
            print(" stats car shape " + str(statistics_car.shape))
        else:
            statistics_car=[]
        people_list=np.zeros((repeat_rep, episode.seq_len ,6), dtype=np.float64)
        car_list = np.zeros((repeat_rep, episode.seq_len,6), dtype=np.int)
        self.counter=0
        # Go through all episodes in a gradient batch.
        jointsParents, labels_indx = self.get_joints_and_labels(agent)

        print (" Episode seq len "+str(episode.seq_len))
        for ep_itr in range(repeat_rep):
            # Reset episode
            initParams = DotMap()
            initParams.on_car = False
            if self.settings.useRealTimeEnv or self.settings.useRLToyCar:

                if iterative_training ==None or iterative_training.train_car:
                    if ep_itr%2==0 and training:
                        initParams.on_car=self.settings.supervised_and_rl_car

                print (" Train on car? Environment ")
                realTimeEnvObservation = self.realTimeEnv.reset(initParams)
                self.parseRealTimeEnvObservation(realTimeEnvObservation, episode)


            # Initialization
            cum_r = 0
            cum_r_car = 0
            pos=[]
            itr_counter=0
            episode.measures = np.zeros(episode.measures.shape)
            if ep_itr%1==0:
                while len(pos)==0:
                    initialization = self.get_next_initialization(training)

                    pos, indx, vel_init = self.agent_initialization(agent, episode, ep_itr, poses_db, training,init_m=initialization, set_goal=set_goal, on_car=initParams.on_car)

                    itr_counter=itr_counter+1
            else:
                episode.set_goal()
                pos=episode.agent[0]


            # print((str(pos)+" "+str(itr_counter)))
            # print(("Done with initialization "+str(episode.goal_person_id)+" "+str(pos)+" goal "+str(episode.goal)))


            self.save_people_cars(ep_itr, episode, people_list, car_list)

            # Initialize agent on the chosen initial position
            agent.initial_position(pos, episode.goal, vel=vel_init)
            if self.net:
                self.net.reset_mem()


            img_from_above = self.initialize_img_above()
            # print ("Before loop over frames " + str(episode.cars))
            # This is the main loop of what happens at each frame!
            # print(("sequence lenth episode: "+str(episode.seq_len)))

            for frame in range(0, episode.seq_len-1):
                # if viz_frame>0 and frame>viz_frame:
                #     value = agent.next_action(episode, training)#, viz=True)
                # else:
                self.print_input_for_manual_inspection(agent, episode, frame, img_from_above, jointsParents,
                                                       labels_indx,
                                                       training)
                # print ("Before agent's action " + str(episode.cars))

                value = agent.next_action(episode, training)

                # print ("After agent's action " + str(episode.cars))
                if self.settings.useRealTimeEnv or self.settings.useRLToyCar:
                    #print("Update real time")
                    # Action could be a sparse thing that's why we set it as a dictionary. In this example we give directly pos and velocity
                    ActionDict = self.update_agent_position_in_real_env(episode, frame)
                    # print ("Before car's action " + str(episode.cars))
                    realTimeEnvObservation = self.realTimeEnv.action(ActionDict, iterative_training ==None or iterative_training.train_car)

                    # print ("Parse input after car's action "+str(episode.cars))
                    self.parseRealTimeEnvObservation(realTimeEnvObservation,episode)
                    #print(f"Action on frame {frame}: {value}. velocity obtained {episode.velocity[frame]}")



                # print ("Perform action "+str(episode.cars))
                agent.perform_action(value, episode)

                self.print_agent_location(episode, frame, value)
                self.print_reward_for_manual_inspection(episode, frame, img_from_above, jointsParents, episode.reward)

            # Calculate rewards that can only be calculated at the end of the episode.
            # print("Computing some rewards at the end of episode")
            for frame in range(0, episode.seq_len - 1):
                #print("Reward for Frame " + str(frame))
                reward= episode.calculate_reward(frame, episode_done=True)  # Return to this!


            # Calculate discounted reward
            episode.discounted_reward()
            #print("Discounted rewards "+str(episode.reward_d))
            # Save all of the gathered statistics.

            episode.save(statistics, num_episode, poses, initialization_map,initialization_car,statistics_car)
            self.print_reward_after_discounting(cum_r, episode)

           # Print some statistics
            if np.sum(episode.measures[:,13])>0:
                self.successes =self.successes+1.0
            self.tries+=1
            # print(("Success rate "+str(self.successes/self.tries)+" nbr of tries: "+str(self.tries)))
            num_episode+=1
            self.num_sucessful_episodes += 1
            # if episode.useRealTimeEnv:
            #     cum_r_car += np.sum(np.copy(episode.reward_car))
            #     print(("  Reward car: " + str(cum_r_car / self.settings.update_frequency) + " "))
            #     if np.isnan(cum_r_car) or np.isinf(cum_r_car):
            #         raise KeyboardInterrupt
            # cum_r+=np.sum(np.copy(episode.reward))
            # print(("  Reward: " + str(cum_r/self.settings.update_frequency)+" "))
            # if np.isnan(cum_r) or np.isinf(cum_r):
            #     raise KeyboardInterrupt


            # Train the agent or evaluate it.
            if not viz:
                # print (" Save weights file")
                if training:
                    filename=self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1], training,saved_files_counter)
                    if self.settings.learn_init:
                        if (not self.settings.keep_init_net_constant  and iterative_training==None) or (iterative_training and iterative_training.train_initializer) or not self.settings.useRLToyCar:
                            print(" Train initializer")
                            statistics = agent.init_net_train(ep_itr, statistics, episode,filename+ '_init_net.pkl',filename + '_weights_init_net.pkl' , poses, initialization_map, initialization_car)
                            if self.settings.train_init_and_pedestrian:
                                print(" Train pedestrian")
                                statistics = agent.train(ep_itr, statistics, episode, filename + '.pkl',
                                                         filename + '_weights.pkl', poses)
                        # statistics, episode, filename, filename_weights, poses, priors, initialization_car
                        if (iterative_training ==None or iterative_training.train_car) and self.settings.useRLToyCar:
                            print(" Train car")
                            self.realTimeEnv.train(ep_itr, statistics,episode, filename + '_car_net.pkl',filename+ '_weights_car_net.pkl' , poses, initialization_map, statistics_car)
                    elif self.settings.useRLToyCar:
                        print(" Train car")
                        self.realTimeEnv.train(ep_itr, statistics, episode, filename + '_car_net.pkl',
                                               filename + '_weights_car_net.pkl', poses, initialization_map, statistics_car)
                    else:
                        print(" Train pedestrian")
                        # ep_itr, statistics, episode, filename, filename_weights, poses
                        statistics=agent.train(ep_itr, statistics, episode,filename + '.pkl',filename + '_weights.pkl' , poses)

                else:
                    if self.settings.learn_init:
                        statistics = agent.init_net_evaluate(ep_itr, statistics, episode, poses, initialization_map,
                                                          initialization_car)
                        # self.realTimeEnv.evaluate(ep_itr, statistics, episode, filename + '_car_net.pkl',
                        #                        filename + '_weights_car_net.pkl', poses, initialization_map,
                        #                        statistics_car)

                        if self.settings.useRLToyCar:
                            print(" Evaluate car")
                            self.realTimeEnv.evaluate(ep_itr, statistics, episode, poses, initialization_map,
                                                   statistics_car)
                    else:
                        statistics=agent.evaluate(ep_itr, statistics, episode, poses, initialization_map)


        # Save the loss of the agent.
        #episode.save_loss(statistics)
        # Save all statistics to a file.
        if save_stats:
            stat_file_name=self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1], training, saved_files_counter)
            print(("Statistics file: "+stat_file_name+".npy"))

            np.save(stat_file_name+".npy", statistics)
            if self.settings.pfnn and not training:
                np.save(stat_file_name+"_poses.npy",poses[:,:,:96])
            if self.settings.learn_init and (not self.settings.keep_init_net_constant or evaluate):
                if  viz :#or not self.settings.save_init_stats:
                    print ("Save init maps " +str(viz)+" "+str(evaluate)+" "+str(self.settings.save_init_stats))
                    np.save(stat_file_name+"_init_map.npy", initialization_map)
                else:
                    print("Save init stats " +str(viz)+" "+str(evaluate)+" "+str(self.settings.save_init_stats))
                    self.save_init_stats(stat_file_name + "_init_stat_map.npy", initialization_map)
                np.save(stat_file_name+ "_init_car.npy",initialization_car)
            if episode.useRealTimeEnv:
                np.save(stat_file_name+ "_learn_car.npy",statistics_car)


            # if np.sum(episode.reconstruction[0,:,:,3])>0:
            #     np.save(self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1], training, saved_files_counter)+"reconstruction", episode.reconstruction[0,:,:,3:5])
            saved_files_counter = saved_files_counter + 1
        return statistics, saved_files_counter, people_list, car_list, poses, initialization_map, initialization_car,statistics_car

    def update_agent_position_in_real_env(self, episode, frame):
        ActionDict = DotMap()
        ActionDict.frame = frame
        ActionDict.init_dir = episode.vel_init
        ActionDict.agentPos = episode.agent[frame]
        ActionDict.agentVel = episode.velocity[frame]
        ActionDict.inverse_dist_to_car = episode.measures[frame, 17]
        return ActionDict

    def initialize_img_above(self):
        return None

    def get_joints_and_labels(self, agent):
        return None, None

    def print_reward_for_manual_inspection(self, episode, frame, img_from_above, jointsParents, reward):
        pass

    def print_agent_location(self, episode, frame, value):
        pass

    def print_input_for_manual_inspection(self, agent, episode, frame, img_from_above, jointsParents, labels_indx,
                                          training):
        pass

    def print_reward_after_discounting(self, cum_r, episode):
        pass

    def get_repeat_reps_and_init(self, training):
        if training:
            repeat_rep = self.settings.update_frequency
            init_methods = self.init_methods  # self.init_methods_train
        else:
            repeat_rep = self.settings.update_frequency_test
            init_methods = self.init_methods  # self.init_methods_train
        return repeat_rep

    def save_init_stats(self, init_file_name, initialization_map):
        initialization_map_stats = np.zeros((initialization_map.shape[0], NBR_MAP_STATS), dtype=np.float64)
        distr = initialization_map[:, :, STATISTICS_INDX_MAP.init_distribution]

        prior = initialization_map[:, :, STATISTICS_INDX_MAP.prior]
        prior_non_zero=prior>0
        distr_goal = initialization_map[:, :, STATISTICS_INDX_MAP.goal_distribution]
        prior_goal = initialization_map[:, :, STATISTICS_INDX_MAP.goal_prior]
        prior_goal_non_zero = prior > 0

        entropys = []
        entropys_prior = []
        kls = []

        entropys_goal = []
        entropys_prior_goal = []
        kls_goal = []
        if self.settings.goal_gaussian:

            initialization_map_stats[:,
            STATISTICS_INDX_MAP_STAT.goal_position_mode[0]:STATISTICS_INDX_MAP_STAT.gaol_position_mode[1]] = distr_goal[:, :2]
            # print ("Goal position shape " + str(distr_goal[:, :2].shape)+" added shape "+str(initialization_map_stats[:,
            #STATISTICS_INDX_MAP_STAT.goal_position_mode[0]:STATISTICS_INDX_MAP_STAT.gaol_position_mode[1]].shape))

        else:
            if self.settings.learn_goal:
                product_goal = distr_goal * prior_goal
            goal_mode = []
            goal_prior = []
            for ep_itr in range(distr.shape[0]):
                entropys.append(scipy.stats.entropy(distr[ep_itr, :]))
                entropys_prior.append(scipy.stats.entropy(prior[ep_itr, :]))
                kls.append(scipy.stats.entropy(distr[ep_itr, :]*prior_non_zero[ep_itr, :], qk=prior[ep_itr, :]))
                if self.settings.learn_goal:
                    entropys_goal.append(scipy.stats.entropy(distr_goal[ep_itr, :]))
                    entropys_prior_goal.append(scipy.stats.entropy(distr_goal[ep_itr, :]))
                    kls_goal.append(scipy.stats.entropy(distr_goal[ep_itr, :]*prior_goal_non_zero[ep_itr, :], qk=prior_goal[ep_itr, :]))
                    goal_mode.append(np.unravel_index(np.argmax(product_goal[ep_itr, :]), self.settings.env_shape[1:]))
                    goal_prior.append(np.unravel_index(np.argmax(prior_goal[ep_itr, :]), self.settings.env_shape[1:]))
                # print ("distribution min "+str(min(distr[ep_itr, :]))+" max "+str(max(distr[ep_itr, :])))
                # print ("prior min "+str(min(prior[ep_itr, :]))+" max "+str(max(prior[ep_itr, :])))


            if self.settings.learn_goal:
                initialization_map_stats[:,
                STATISTICS_INDX_MAP_STAT.goal_position_mode[0]:STATISTICS_INDX_MAP_STAT.goal_position_mode[1]] = goal_mode
                # print (" Saved goal mode "+str(np.sum(goal_mode))+" saved: "+str(np.sum(initialization_map_stats[:,
                #STATISTICS_INDX_MAP_STAT.goal_position_mode[0]:STATISTICS_INDX_MAP_STAT.goal_position_mode[1]])))
                initialization_map_stats[:,
                STATISTICS_INDX_MAP_STAT.goal_prior_mode[0]:STATISTICS_INDX_MAP_STAT.goal_prior_mode[1]] = goal_prior
                #print (" Saved goal prior " + str(np.sum(goal_prior)) + " saved: " + str(np.sum(initialization_map_stats[:,STATISTICS_INDX_MAP_STAT.goal_prior_mode[0]:STATISTICS_INDX_MAP_STAT.goal_prior_mode[1]])))

        product = distr * prior
        init_mode = []
        init_prior = []
        for ep_itr in range(product.shape[0]):
            init_mode.append(np.unravel_index(np.argmax(product[ep_itr, :]), self.settings.env_shape[1:]))
            init_prior.append(np.unravel_index(np.argmax(prior[ep_itr, :]), self.settings.env_shape[1:]))
        # print (init_mode)
        initialization_map_stats[:,STATISTICS_INDX_MAP_STAT.init_position_mode[0]:STATISTICS_INDX_MAP_STAT.init_position_mode[1]]=init_mode
        #print (" Saved init mode " + str(np.sum(init_mode)) + " saved: " + str(np.sum(initialization_map_stats[:,STATISTICS_INDX_MAP_STAT.init_position_mode[0]:STATISTICS_INDX_MAP_STAT.init_position_mode[1]])))
        initialization_map_stats[:,STATISTICS_INDX_MAP_STAT.init_prior_mode[0]:STATISTICS_INDX_MAP_STAT.init_prior_mode[1]] = init_prior
        #print (" Saved init prior mode " + str(np.sum(init_prior)) + " saved: " + str(np.sum(initialization_map_stats[:,STATISTICS_INDX_MAP_STAT.init_prior_mode[0]:STATISTICS_INDX_MAP_STAT.init_prior_mode[1]])))

        initialization_map_stats[:,STATISTICS_INDX_MAP_STAT.entropy] = entropys
        #print (" Saved entropy " + str(np.sum(entropys)) + " saved: " + str(np.sum(initialization_map_stats[:,STATISTICS_INDX_MAP_STAT.entropy] )))

        initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.entropy_prior] = entropys_prior
        # print (" Saved prior entropy " + str(np.sum(entropys_prior)) + " saved: " + str(
        #     np.sum(initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.entropy_prior])))

        initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.kullback_leibler_divergence_init_and_prior] = kls
        # print (" Saved kl " + str(np.sum(kls)) + " saved: " + str(
        #     np.sum(initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.kullback_leibler_divergence_init_and_prior])))
        diff_to_prior = np.sum(np.abs(distr - prior), axis=1)
        initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.prior_init_difference] = diff_to_prior
        # print (" Difference to prior " + str(np.sum(diff_to_prior)) + " saved: " + str(
        #     np.sum(initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.prior_init_difference])))
        if self.settings.learn_goal:
            initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.entropy_goal] = entropys_goal
            # print (" Saved entropy goal " + str(np.sum(entropys_goal)) + " saved: " + str(
            #     np.sum(initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.entropy_goal])))
            initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.entropy_prior_goal] = entropys_prior_goal
            # print (" Saved entropy goal prior " + str(np.sum(entropys_prior_goal)) + " saved: " + str(
            #     np.sum(initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.entropy_prior_goal])))
            initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.kullback_leibler_divergence_init_and_prior_goal] = kls_goal
            # print (" Saved kl goal " + str(np.sum(kls_goal)) + " saved: " + str(
            #     np.sum(initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.kullback_leibler_divergence_init_and_prior_goal])))
            diff_to_prior_goal = np.sum(np.abs(distr_goal - prior_goal), axis=1)
            initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.prior_init_difference_goal] = diff_to_prior_goal
            # print (" Difference to prior goal " + str(np.sum(diff_to_prior_goal)) + " saved: " + str(
            #     np.sum(initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.prior_init_difference_goal])))
        np.save(init_file_name,initialization_map_stats)
        # print ("Saved init map stats "+init_file_name)




    def get_next_initialization(self, training):
        if training:
            print (" init methods "+str(self.init_methods_train)+" counter "+str(self.counter))
            self.counter=self.counter+1
            self.counter=self.counter%len(self.init_methods_train)
            if len(self.init_methods_train)==1:
                return self.init_methods[0]
            return self.init_methods_train[self.counter]
        else:

            self.counter = self.counter + 1
            self.counter = self.counter% len(self.init_methods)
        self.sem_counter = self.sem_counter + 1

        return self.init_methods[self.counter]



    # Get file name of statistics file.
    def statistics_file_name(self, file_agent, pos_x, pos_y, training,saved_files_counter, init_meth=-1):
        if not training:
            if init_meth>0:
                return file_agent + "_test"+str(init_meth)+"_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)
            return file_agent + "_test_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)
        return  file_agent + "_" + str(pos_x) + "_" + str(pos_y) + "_" + str(saved_files_counter)

    def getEpisodeCachePathByParams(self, episodeIndex, cameraPosX, cameraPosY):
        targetCacheFile = os.path.join(self.target_episodesCache_Path, "{0}_x{1:0.1f}_y{2:0.1f}.pkl".format(episodeIndex, cameraPosX, cameraPosY))
        return targetCacheFile

    def tryReloadFromCache(self, episodeIndex, cameraPosX, cameraPosY):
        episode_unpickled = None
        targetCacheFile = self.getEpisodeCachePathByParams(episodeIndex, cameraPosX, cameraPosY)
        print(targetCacheFile)
        try:
            if os.path.exists(targetCacheFile):
                print(("Loading {} from cache".format(targetCacheFile)))
                with open(targetCacheFile, "rb") as testFile:
                    start_load = time.time()
                    #episode_unpickled = pickle.load(testFile)
                    episode_unpickled = joblib.load(testFile)
                    load_time = time.time() - start_load
                    print(("Load time from cache (FILE) in s {:0.2f}s".format(load_time)))
            else:
                print(("File {} not found in cache...going to take a while to process".format(targetCacheFile)))
        except ImportError:
            print("import error ")

        return episode_unpickled

    def trySaveToCache(self, episode, episodeIndex, cameraPosX, cameraPosY):
        targetCacheFile = self.getEpisodeCachePathByParams(episodeIndex, cameraPosX, cameraPosY)
        with open(targetCacheFile, "wb") as testFile:
            print(("Storing {} to cache".format(targetCacheFile)))
            serial_start = time.time()
            #episode_serialized = pickle.dump(episode, testFile, protocol=pickle.HIGHEST_PROTOCOL)
            joblib.dump(episode, testFile, protocol=pickle.HIGHEST_PROTOCOL)
            serial_time = time.time() - serial_start
            print(("Serialization time (FILE) in s {:0.2f}s".format(serial_time)))

    # Set up episode given the ply file, camera position where the sample is taken from and few other parameters that need to be documented well :)
    def set_up_episode(self, cachePrefix, envPly, pos_x, pos_y, training, useCaching,evaluate=False,  time_file=None, seq_len_pfnn=-1, datasetOptions=None, supervised=False, car=None):
        print(" Car in set up episode " + str(car))
        cachePrefix += os.path.basename(envPly)

        # Check to see if caching is enabled and if it exists
        seq_len=self.settings.seq_len_train
        if not training:
            seq_len=self.settings.seq_len_test

        if useCaching:
            res = self.tryReloadFromCache(cachePrefix, pos_x, pos_y)
            if res is not None:
                if res is SupervisedEpisode and not supervised:
                    episode=SimpleEpisode([],
                                          res.people_e,
                                          res.cars_e,
                                          pos_x,
                                          pos_y,
                                          res.gamma,
                                          res.seq_len,
                                          res.reward_weights,
                                          res.agent_size,
                                          people_dict=res.people_dict,
                                          cars_dict=res.cars_dict,
                                          init_frames=res.init_frames,
                                          agent_height=self.settings.height_agent,
                                          multiplicative_reward=self.settings.multiplicative_reward,
                                          learn_goal=self.settings.learn_goal or self.settings.separate_goal_net,
                                          use_occlusion=self.settings.use_occlusion,
                                          useRealTimeEnv=self.settings.useRealTimeEnv or self.settings.useHeroCar,
                                          new_carla=self.settings.new_carla,
                                          lidar_occlusion=self.settings.lidar_occlusion)
                    print ("Wrong Episode class!")

                if res.seq_len>=seq_len and ((seq_len<=450 and self.settings.carla) or (seq_len<=450 and self.settings.waymo) or (seq_len<=30 and not self.settings.waymo and not self.settings.carla )):


                    print("set settings!")
                    res.set_correct_run_settings(self.settings.run_2D, seq_len,
                                                 self.settings.stop_on_goal,
                                                 self.settings.goal_dir,
                                                 self.settings.threshold_dist,
                                                 self.settings.gamma,
                                                 self.settings.reward_weights,
                                                 self.settings.end_on_bit_by_pedestrians,
                                                 self.settings.speed_input,
                                                 self.settings.waymo,
                                                 self.settings.reorder_actions,
                                                 seq_len_pfnn,
                                                 self.settings.velocity,
                                                 self.settings.height_agent,
                                                 evaluation=evaluate,
                                                 defaultSettings = self.settings,
                                                 multiplicative_reward=self.settings.multiplicative_reward,
                                                 learn_goal=self.settings.learn_goal or self.settings.separate_goal_net,
                                                 use_occlusion=self.settings.use_occlusion,
                                                 useRealTimeEnv=self.settings.useRealTimeEnv or self.settings.useHeroCar,
                                                 new_carla=self.settings.new_carla,
                                                 lidar_occlusion=self.settings.lidar_occlusion)

                    if self.settings.useRLToyCar:

                        from RealTimeRLCarEnvInteraction import RLCarRealTimeEnv
                        seq_len = self.settings.seq_len_train
                        if not training:
                            seq_len = self.settings.seq_len_test

                        self.realTimeEnv = RLCarRealTimeEnv(car, res.cars_dict, res.cars, res.init_frames,
                                                            res.init_frames_car, res.people_dict, res.people, res.reconstruction,
                                                            seq_len, self.settings.car_dim, self.settings.car_max_speed,
                                                            self.settings.car_input, True, self.settings.reward_weights_car[CAR_REWARD_INDX.reached_goal] )

                        observation_dict = self.realTimeEnv.get_cars_and_people(0)
                        cars_dict_sample1 = observation_dict.cars_dict
                        people_dict_sample1 = observation_dict.people_dict

                        cars_sample1 = [list(cars_dict_sample1.values())]
                        people_sample1 = [list(people_dict_sample1.values())]

                        for car_key in cars_dict_sample1.keys():
                            cars_dict_sample1[car_key] = [cars_dict_sample1[car_key]]

                        for person_key in people_dict_sample1.keys():
                            people_dict_sample1[person_key] = [people_dict_sample1[person_key]]

                        episode = self.init_episode(cars_dict_sample1, cars_sample1, self.realTimeEnv.init_frames,
                                                    self.realTimeEnv.init_frames_cars,
                                                    people_dict_sample1,
                                                    people_sample1, pos_x, pos_y, seq_len_pfnn, res.reconstruction, training,
                                                    res.heroCarDetails if self.settings.useHeroCar else None,
                                                    self.settings.useRealTimeEnv or self.settings.useRLToyCar or self.settings.useHeroCar,
                                                    observation_dict.car_vel_dict, observation_dict.pedestrian_vel_dict)


                        return episode
                    return res

        heroCarPos = None
        if datasetOptions is None or datasetOptions.isColmap == False:
            # Read the reconstruction from the ply episode file
            print (" Dataset  options is not None")
            # (filepath, local_scale_x=5, recalculate=False, find_poses=False, read_3D=True,datasetOptions=None, params=None)
            reconstruction, people, cars, scale, people_dict, cars_2D, people_2D, \
                valid_ids, car_dict, init_frames, init_frames_cars, heroCarDetails = ReconstructionUtils.reconstruct3D_ply(envPly, self.settings.scale_x, read_3D=True, datasetOptions=datasetOptions)

        else:
            # Returns reconstruction in cityscapes coordinate system.
            # reconstruction_rescaled, people, cars, scale, camera_locations_colmap, middle
            reconstruction, people, cars, scale, camera_locations_colmap, middle = colmap.reconstruct.reconstruct3D_ply(envPly, datasetOptions.envSettings, training)
            #reconstruction_rescaled, people, cars, scale, camera_locations_colmap, middle
            while len(people) < datasetOptions.LIMIT_FRAME_NUMBER:
                people.append([])
                cars.append([])

            init_frames = {}
            init_frames_cars = {}
            people_dict = {}
            car_dict = {}


        # objects = [self.reconstruction, people_rec, cars_rec, scale, ped_dict, cars_2D, people_2D, valid_ids, car_dict, init_frames, init_frames_cars]
        # mem_occupancy = deep_getsizeof(objects, set())

        # Create tensors of input from the total 3D reconstruction.
        tensor, density = extract_tensor(pos_x, pos_y, reconstruction, self.settings.height, self.settings.width, self.settings.depth)

        if datasetOptions != None and datasetOptions.filter2D == True:
            if datasetOptions.realtime_data:
                semantic=(tensor[:, :, :, CHANNELS.semantic]*NUM_SEM_CLASSES).astype(np.int)
                labels=[]
                for label in ROAD_LABELS:
                    labels.append(label)
                for label in SIDEWALK_LABELS:
                    labels.append(label)
                for label in  labels:
                    sidewalk=semantic==label

                    dilated_sidewalk=ndimage.binary_dilation(sidewalk,  iterations=4)
                    positions=np.where(dilated_sidewalk)

                    for i in range(len(positions[0])):
                        if tensor[positions[0][i], positions[1][i], positions[2][i], CHANNELS.semantic]==0:
                            tensor[positions[0][i], positions[1][i], positions[2][i], CHANNELS.semantic] = label/NUM_SEM_CLASSES
            else:
                for channel in range(4):
                    tensor[:,:,:,channel]=scipy.ndimage.filters.median_filter(tensor[:,:,:,channel], size=(1,4,4)) # (1,3,3), (2,3,3) or (1,4,4) which is more realistic?

        # print (" Before adding people "+str(len(people[0])) )
        cars_sample = objects_in_range(cars, pos_x, pos_y, self.settings.depth, self.settings.width, carla= self.settings.carla or self.settings.waymo)
        people_sample = objects_in_range(people, pos_x, pos_y, self.settings.depth, self.settings.width,  carla= self.settings.carla or self.settings.waymo)
        # print(" After adding people " + str(len(people_sample[0])))
        # people_dict_sample
        people_dict_sample={}
        if len(people_dict)>0:
            people_dict_sample, init_frames=objects_in_range_map(people_dict, pos_x, pos_y, self.settings.depth, self.settings.width, init_frames=init_frames)

        cars_dict_sample = {}
        if len(car_dict) > 0:
            cars_dict_sample, init_frames_cars= objects_in_range_map(car_dict, pos_x, pos_y, self.settings.depth,
                                                       self.settings.width, init_frames=init_frames_cars)

        if self.settings.useRealTimeEnv:
            import commonUtils.RealTimeEnv.RealTimeEnvInteraction
            self.realTimeEnv = commonUtils.RealTimeEnv.RealTimeEnvInteraction.CarlaRealTimeEnv()
        elif self.settings.useRLToyCar:
            print ("Set up toy car")
            from RealTimeRLCarEnvInteraction import RLCarRealTimeEnv
            seq_len=self.settings.seq_len_train
            if not training:
                seq_len = self.settings.seq_len_test

            self.realTimeEnv =RLCarRealTimeEnv(car,copy.deepcopy(cars_dict_sample), copy.deepcopy(cars_sample), copy.deepcopy(init_frames), copy.deepcopy(init_frames_cars), copy.deepcopy(people_dict_sample), copy.deepcopy(people_sample), copy.deepcopy(tensor), seq_len, self.settings.car_dim, self.settings.car_max_speed, self.settings.car_input, False,  self.settings.reward_weights_car[CAR_REWARD_INDX.reached_goal] )
            observation_dict=self.realTimeEnv.get_cars_and_people(0)
            cars_dict_sample1=observation_dict.cars_dict
            people_dict_sample1 = observation_dict.people_dict

            cars_sample1 = [list(cars_dict_sample1.values())]
            people_sample1 = [list(people_dict_sample1.values())]

            for car_key in cars_dict_sample1.keys():
                cars_dict_sample1[car_key]=[cars_dict_sample1[car_key]]

            for person_key in people_dict_sample1.keys():
                people_dict_sample1[person_key]=[people_dict_sample1[person_key]]


            episode1 = self.init_episode(cars_dict_sample1, cars_sample1, self.realTimeEnv.init_frames, self.realTimeEnv.init_frames_cars,
                                        people_dict_sample1,
                                        people_sample1, pos_x, pos_y, seq_len_pfnn, tensor, training,
                                        heroCarDetails if self.settings.useHeroCar else None,
                                        self.settings.useRealTimeEnv or self.settings.useRLToyCar,
                                        observation_dict.car_vel_dict,observation_dict.pedestrian_vel_dict )

            if useCaching:
                episode = self.init_episode(copy.deepcopy(cars_dict_sample), copy.deepcopy(cars_sample), copy.deepcopy(init_frames), copy.deepcopy(init_frames_cars),
                                            copy.deepcopy(people_dict_sample),
                                            copy.deepcopy(people_sample), pos_x, pos_y, seq_len_pfnn, copy.deepcopy(tensor), training,
                                            None,
                                            False)
                self.trySaveToCache(episode, cachePrefix, pos_x, pos_y)

            return episode1
        else:
            import commonUtils.RealTimeEnv.RealTimeNullEnvInteraction
            self.realTimeEnv = commonUtils.RealTimeEnv.RealTimeNullEnvInteraction.NullRealTimeEnv()

        episode = self.init_episode(cars_dict_sample, cars_sample, init_frames, init_frames_cars, people_dict_sample,
                                    people_sample, pos_x, pos_y, seq_len_pfnn, tensor, training,
                                     None,
                                    False)#self.settings.useRealTimeEnv or self.settings.useRLToyCar)# heroCarDetails if self.settings.useHeroCar else
        # print("Episode input cars "+str(cars_sample[0]))
        # print("Episode  cars " + str(episode.cars[0]))
        # for key, frame in episode.init_frames_car.items():
        #     print ("Frame "+str(frame)+" car "+str(key))
        #     if frame==0 and key in episode.cars_dict:
        #         print (" Car in 0-th frame "+str(episode.cars_dict[key][0]))
        #
        # print("Episode input people " + str(people_sample[0]))
        # print("Episode people " + str(episode.people[0]))
        # for key, frame in episode.init_frames.items():
        #     print("Frame " + str(frame) + " person " + str(key))
        #     if key in episode.people_dict and len(episode.people_dict[key])>=1 :
        #         print(" Pedestrian in 0-th frame " + str(episode.people_dict[key][0])+" frame "+str(frame))

        if useCaching:
            res = self.trySaveToCache(episode, cachePrefix, pos_x, pos_y)


        # print ("Episode not valid pos ")
        # print (np.where(episode.valid_positions!=episode1.valid_positions))
        #
        # input("Press Enter to continue...")
        return episode

    def init_episode(self, cars_dict_sample, cars_sample, init_frames, init_frames_cars, people_dict_sample,
                     people_sample, pos_x, pos_y, seq_len_pfnn, tensor, training, heroCarDetails, useRealTimeEnv, car_vel_dict=None, people_vel_dict=None):
        if training:
            seq_len=self.settings.seq_len_train
        else:
            while len(cars_sample) < self.settings.seq_len_test:
                if self.settings.carla  or self.settings.waymo:
                    cars_sample.append([])
                    people_sample.append([])
                else:
                    cars_sample.append(cars_sample[-1])
                    people_sample.append(people_sample[-1])
            seq_len=self.settings.seq_len_test
        episode = SimpleEpisode(tensor, people_sample, cars_sample, pos_x, pos_y, self.settings.gamma,
                                seq_len, self.settings.reward_weights,
                                agent_size=self.settings.agent_shape, people_dict=people_dict_sample,
                                cars_dict=cars_dict_sample,
                                init_frames=init_frames, follow_goal=self.settings.goal_dir,
                                action_reorder=self.settings.reorder_actions,
                                threshold_dist=self.settings.threshold_dist, init_frames_cars=init_frames_cars,
                                temporal=self.settings.temporal, predict_future=self.settings.predict_future,
                                run_2D=self.settings.run_2D, agent_init_velocity=self.settings.speed_input,
                                velocity_actions=self.settings.velocity or self.settings.continous,
                                seq_len_pfnn=seq_len_pfnn, end_collide_ped=self.settings.end_on_bit_by_pedestrians,
                                stop_on_goal=self.settings.stop_on_goal, waymo=self.settings.waymo,
                                defaultSettings=self.settings,multiplicative_reward=self.settings.multiplicative_reward,
                                learn_goal=self.settings.learn_goal or self.settings.separate_goal_net,
                                use_occlusion=self.settings.use_occlusion,
                                heroCarDetails = heroCarDetails, useRealTimeEnv=useRealTimeEnv,  car_vel_dict=car_vel_dict,
                                people_vel_dict=people_vel_dict, car_dim=self.settings.car_dim,
                                new_carla=self.settings.new_carla,lidar_occlusion=self.settings.lidar_occlusion)
        return episode

    # Initiaization.
    def agent_initialization(self, agent, episode, itr, poses_db, training, init_m=-1, set_goal="", on_car=False):

        if self.settings.learn_init and not on_car:
            print (" Learn init")
            return agent.init_agent(episode, training)
        init_key = itr
        # 1,6,2,4
        init_names = {1: "on ped", 6: "near ped", 2: "by car", 4: "random", -1: "train"}
        # aachen_000024
        init_pairs = {}
        init_pairs["aachen_000042"] = [([103, 68], [213, 38]), ([103, 68], [254, 45]), ([103, 68], [72, 70]),
                                       ([132, 42], [248, 46]), ([132, 42], [60, 23]), ([132, 42], [64, 32]),
                                       ([132, 42], [42, 38]),
                                       ([60, 25], [123, 29]), ([60, 25], [121, 2]), ([60, 25], [72, 73]),
                                       ([60, 25], [141, 70]),
                                       ([60, 25], [252, 49])]
        init_pairs["tubingen_000136"] = [([76, 16], [113, 62]), ([76, 16], [100, 83]), ([76, 16], [16, 62]),
                                         ([76, 16], [107, 4]), ([76, 16], [169, 78]), ([76, 16], [150, 112])]
        # init_pairs["tubingen_000112"] = [([131, 2], [7, 78]), ([131, 2], [69, 104]),([131, 2], [100, 78]), ([131, 2], [74, 123]), ([131, 2], [152, 64])]
        init_pairs["tubingen_000112"] = [(np.array([5, 79, 99]), [85, 123]), (np.array([5, 79, 99]), [21, 2]),
                                         (np.array([5, 79, 99]), [6, 124])]

        init_pairs["bremen_000028"] = [([40, 71], [145, 40]), ([40, 71], [166, 54]), ([12, 71], [157, 95]),
                                       ([12, 71], [169, 47]), ([12, 71], [86, 40])]
        init_pairs["bremen_000160"] = [([18, 42], [145, 82]), ([18, 42], [251, 57]), ([18, 42], [144, 83]),
                                       ([18, 42], [79, 84]),
                                       ([58, 54], [145, 82]), ([58, 54], [251, 57]), ([58, 54], [144, 83]),
                                       ([58, 54], [79, 84])]
        init_pairs["munster_000039"] = [([23, 57], [118, 97]), ([23, 57], [190, 10]), ([23, 57], [140, 42]),
                                        ([23, 57], [127, 94])]
        init_pairs["darmstadt_000019"] = [([27, 52], [191, 18]), ([27, 52], [80, 95]), ([27, 52], [243, 89]),
                                          ([27, 52], [59, 101]), ([59, 101], [191, 18]), ([59, 101], [243, 89])]

        if len(set_goal) > 0:
            goal = init_pairs[set_goal][itr % len(init_pairs[set_goal])][1]

            if "tubingen_000112" in set_goal:
                episode.agent[0] = init_pairs[set_goal][itr % len(init_pairs[set_goal])][0]
                episode.goal = np.array([episode.agent[0][0], 128 - goal[1], goal[0]])
            else:
                pos_x = init_pairs[set_goal][itr % len(init_pairs[set_goal])][0]
                h = episode.get_height_init()
                episode.agent[0] = np.array([h, 128 - pos_x[1], pos_x[0]])
                episode.goal = np.array([h, 128 - goal[1], goal[0]])
            print(("Set pos " + str(episode.agent[0]) + " " + str(goal) + " " + str(episode.goal)))
            return episode.agent[0], -1, episode.vel_init
        if itr >= len(episode.valid_keys) and len(episode.valid_keys) > 0:
            init_key = itr % len(episode.valid_keys)


        if init_m > 0:
            print(("init_m- initialization method " + str(init_m) + " " + str(init_key) + " " + str(training)))
            pos, indx, vel_init = episode.initial_position(poses_db, training=training, init_key=init_key, initialization=init_m)
        elif len(episode.valid_keys) > 0:
            print(("valid_keys" + " " + str(init_key) + " " + str(training)))
            pos, indx, vel_init = episode.initial_position(poses_db, training=training, init_key=init_key)
        else:
            print(("training " + str(training)))
            pos, indx, vel_init = episode.initial_position(poses_db, training=training)
        return pos, indx, vel_init


    def get_file_agent_path(self, file_name, eval_path=""):
        if self.settings.new_carla:
            basename=os.path.basename(file_name[0])
            city = basename
            seq_nbr = os.path.dirname(file_name[0])
        else:
            basename = os.path.basename(file_name)
            print (basename)
            parts = basename.split('_')
            city = parts[0]
            seq_nbr = parts[1]
        directory=self.settings.statistics_dir
        if eval_path:
            file_agent = os.path.join(eval_path, self.settings.timestamp + "agent_" + city + "_" + seq_nbr)
        else:

            file_agent = os.path.join(directory, self.settings.mode_name,
                                      self.settings.timestamp + "agent_" + city + "_" + seq_nbr)

        if not os.path.exists(file_agent):
            os.makedirs(file_agent)

        return file_agent



    def __init__(self, path, sess, writer, gradBuffer, log,settings, net=None):
        self.num_sucessful_episodes = 0
        self.accummulated_r = 0
        self.counter=0
        self.frame_counter_eval=0
        self.img_path = path  # images
        self.files = []
        self.episode_buffer = []
        self.gradBuffer = gradBuffer
        self.reconstruction={}
        self.writer=writer
        self.sess=sess
        self.log=log
        self.settings=settings
        self.net=net
        self.frame_counter=0
        self.sem_counter=0

        self.scene_count = 0
        self.scene_count_test = 0
        self.viz_counter_test = 0
        # self.goal_dist_reached=1
        self.init_methods = [1, 8, 6, 2, 4]  # [1,5]#[1, 3, 5, 9]#, 8, 6, 2, 4]#[1, 8, 6, 2, 4, 9,3,5, 10]#[1, 8, 6, 2, 4, 5, 9]#[8, 6, 2, 4] #<--- regular![1, 8, 6, 2, 4] [1]
        self.init_methods_train =[1, 8, 6, 2, 4] #[1, 8, 6, 2, 4, 1, 3, 5, 9]
        self.successes = 0.0
        self.tries = 0

        self.target_episodesCache_Path = settings.target_episodesCache_Path


        if not os.path.exists(self.target_episodesCache_Path) and USE_LOCAL_PATHS ==1:
            os.makedirs(self.target_episodesCache_Path)

        for path, dirs, files in os.walk(self.img_path):
            self.files = self.files + dirs  # + files

        self.scene_count_test=0
        self.viz_counter_test=0
        #self.goal_dist_reached=1

        self.successes = 0.0
        self.tries = 0






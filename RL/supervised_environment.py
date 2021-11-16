from carla_environment import CARLAEnvironment
from environment_abstract import AbstractEnvironment

from episode import SupervisedEpisode
from visualization import make_movie

from environment_waymo import WaymoEnvironment
from settings import NBR_MEASURES

import numpy as np
import os

class SupervisedEnvironment(AbstractEnvironment):

    def __init__(self, path, sess, writer, gradBuffer, log, settings, net=None):
        super(SupervisedEnvironment, self).__init__(path, sess, writer, gradBuffer, log, settings, net=net)
        self.init_methods_train = [-1]
        self.init_methods_test = [-1]
        self.init_methods_val=[-1]#[8, 6, 2, 4]

        print("Supervised Env ")

    def init_episode(self, cars_dict_sample, cars_sample, init_frames, init_frames_cars, people_dict_sample,
                     people_sample, pos_x, pos_y, seq_len_pfnn, tensor, training):
        if training:

            episode = SupervisedEpisode(tensor, people_sample, cars_sample, pos_x, pos_y, self.settings.gamma,
                                        self.settings.seq_len_train, self.settings.reward_weights,
                                        agent_size=self.settings.agent_shape, people_dict=people_dict_sample,
                                        cars_dict=cars_dict_sample,
                                        init_frames=init_frames, follow_goal=self.settings.goal_dir,
                                        action_reorder=self.settings.reorder_actions,
                                        threshold_dist=self.settings.threshold_dist, init_frames_cars=init_frames_cars,
                                        temporal=self.settings.temporal, predict_future=self.settings.predict_future,
                                        run_2D=self.settings.run_2D,
                                        velocity_actions=self.settings.velocity or self.settings.continous,
                                        seq_len_pfnn=seq_len_pfnn,
                                        end_collide_ped=self.settings.end_on_bit_by_pedestrians,
                                        stop_on_goal=self.settings.stop_on_goal, defaultSettings=self.settings)
        else:
            while len(cars_sample) < self.settings.seq_len_test:
                if self.settings.carla:
                    cars_sample.append([])
                    people_sample.append([])
                else:
                    cars_sample.append(cars_sample[-1])
                    people_sample.append(people_sample[-1])
            episode = SupervisedEpisode(tensor, people_sample, cars_sample, pos_x, pos_y, self.settings.gamma,
                                        self.settings.seq_len_test, self.settings.reward_weights,
                                        agent_size=self.settings.agent_shape, people_dict=people_dict_sample,
                                        cars_dict=cars_dict_sample,
                                        init_frames=init_frames, follow_goal=self.settings.goal_dir,
                                        action_reorder=self.settings.reorder_actions,
                                        threshold_dist=self.settings.threshold_dist, init_frames_cars=init_frames_cars,
                                        temporal=self.settings.temporal, predict_future=self.settings.predict_future,
                                        run_2D=self.settings.run_2D,
                                        velocity_actions=self.settings.velocity or self.settings.continous,
                                        seq_len_pfnn=seq_len_pfnn,
                                        end_collide_ped=self.settings.end_on_bit_by_pedestrians,
                                        stop_on_goal=self.settings.stop_on_goal, defaultSettings=self.settings)
        return episode
    def agent_initialization(self, agent, episode, itr,  poses_db, training, init_m=-1, set_goal=""):
        init_key = itr
        print(("Init key "+str(init_key)))
        # 1,6,2,4
        init_names = {1: "on ped", 6: "near ped", 2: "by car", 4: "random", -1: "train"}

        if itr >= len(episode.valid_keys) and len(episode.valid_keys) > 0:
            init_key = itr % len(episode.valid_keys)
            print(("Itr "+str(itr)+" init key "+str(init_key)+" "+str(len(episode.valid_keys))))

        if init_m > 0:
            print(("init_m- initialization method "+str(init_m)+" "+str(init_key)+" "+str(training)))
            pos, indx, vel = episode.initial_position(poses_db, training=training, init_key=init_key, initialization=init_m)
        elif len(episode.valid_keys) > 0:
            print(("valid_keys"+" "+str(init_key)+" "+str(training)+" len "+str(len(episode.valid_keys))))
            pos, indx, vel = episode.initial_position(poses_db, training=training, init_key=init_key)
        else:
            print(("training "+str(training)))
            pos, indx, vel = episode.initial_position(poses_db, training=training)

        return pos, indx, vel

    def get_repeat_reps_and_init(self, training,val=-1, validation=False):
        if validation:
            repeat_rep = self.settings.update_frequency_test
            self.init_methods = self.init_methods_val  # self.init_methods_train
            print(("Init methods "+str(self.init_methods)+" "+str(repeat_rep)))
        elif training:
            repeat_rep = min(val, self.settings.update_frequency)
            self.init_methods = self.init_methods_train  # self.init_methods_train
        else:
            repeat_rep = self.settings.update_frequency_test
            self.init_methods=self.init_methods_test  # self.init_methods_train
        return repeat_rep, self.init_methods

    def visualize(self, episode, file_name, statistics, training, poses, agent):
        return
        seq_len = self.settings.seq_len_train
        if not training:
            seq_len = self.settings.seq_len_test
        jointsParents=None
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
                                        training=training,
                                        name_movie=self.settings.name_movie,
                                        path_results=self.statistics_dir + "/agent/",
                                        episode_name=os.path.basename(file_name),
                                        action_reorder=self.settings.reorder_actions,
                                        inits_frames=True,
                                        velocity_show=self.settings.velocity,
                                        velocity_sigma=self.settings.velocity_sigmoid,
                                        jointsParents=jointsParents,
                                        continous=self.settings.continous)


    def act_and_learn(self, agent, agent_file, episode, poses_db, training,saved_files_counter,  road_width=0, curriculum_prob=0, time_file=None, pos_init=[], viz=False, set_goal="", viz_frame=-1):
    # (self, agent, agent_file, episode, poses_db, training, saved_files_counter, road_width=0,
    #                   curriculum_prob=0, time_file=None, pos_init=[], viz=False):

        num_episode = 0
        cum_r = 0
        init_methods = []


        repeat_rep, init_methods = self.get_repeat_reps_and_init(training, val=len(episode.valid_keys))
        poses = np.zeros((repeat_rep, episode.seq_len * 60 // episode.frame_rate + episode.seq_len, 93 + 2 + 1 + 1 + 512),
                     dtype=np.float64)
        statistics = np.zeros((repeat_rep, (episode.seq_len - 1), 38 + NBR_MEASURES + 1), dtype=np.float64)
        people_list = np.zeros((repeat_rep, episode.seq_len, 6), dtype=np.float64)
        car_list = np.zeros((repeat_rep, episode.seq_len, 6), dtype=np.int)
        self.counter = 0
        # Go through all episodes in a gradient batch.
        if True:
            for repeat_counter in range(len(episode.valid_keys)):#repeat_rep):

                cum_r = 0
                pos = []

                initialization = self.get_next_initialization(training)
                print(("Init "+str(initialization)))
                #print "Iteration "+str(repeat_counter)+" "+str(len(episode.valid_keys))
                pos, indx, vel = self.agent_initialization(agent, episode, repeat_counter, poses_db, training, init_m=initialization)
                #print "Inititialize of ped " + str(indx)
                if len(pos)>0:
                    #self.save_people_cars(itr, episode, people_list, car_list)
                    agent.initial_position(pos,episode.goal, episode.current_frame)
                    print(("Done with initialization " + str(episode.goal_person_id)))
                    if self.net:
                        self.net.reset_mem()
                    # This is the main loop of what happens at each frame!
                    correct_count=0
                    avg_error=0

                    number_frames=episode.seq_len - 1

                    person_key=episode.valid_keys[indx]
                    if len(episode.valid_people_tracks[person_key])-2<number_frames:
                        number_frames=len(episode.valid_people_tracks[person_key])-2
                    #print "Number of frames "+str(number_frames)+" len: "+str(number_frames)
                    for frame in range(0, number_frames):
                        #print "Enviornment frame "+str(frame)
                        value = agent.next_action(episode, training )#,filename=self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1], training, saved_files_counter) + '.pkl')#,filename_weights=
                                  # self.statistics_file_name(agent_file,  episode.pos[0], episode.pos[1], training, saved_files_counter) + '_weights.pkl'  )

                        desired=np.mean(episode.valid_people_tracks[person_key][agent.frame+1], axis=1)
                        coming_pos=agent.pos_exact+value
                        # print "Agent pos "+str(agent.pos_exact)+" desired pos "+str(desired)+" true pos "+str(coming_pos)+" "
                        # print "Agent takes action "+str(value[1:])+" desired action "+str(desired[1:]-agent.pos_exact[1:])+" error: "+str(coming_pos[1:]-desired[1:])
                        avg_error+=((desired[1:]-coming_pos[1:])/number_frames)
                        if (episode.get_valid_action(agent.frame, person_key)==episode.action[agent.frame]):
                            correct_count+=1
                        # print value
                        if episode.init_method==1 :
                            #print "Init method 1 "
                            agent.perform_action(value, episode, prob=curriculum_prob,person_id=indx, training=True)
                        else:
                            agent.perform_action(value, episode, prob=curriculum_prob, person_id=indx, training=training)

                        #print "Reward "+str(reward)
                        # print str(frame)+" Number non-zero" + str(len(np.nonzero(episode.reconstruction[:, :, :, 4])[0]))
                    number_frames=number_frames+1

                    while number_frames<episode.seq_len:

                        episode.agent[number_frames]= episode.agent[number_frames-1]
                        episode.velocity[number_frames-1] = np.zeros(3, dtype=int)
                        number_frames= number_frames+ 1
                    print(("Loss "+str(np.sum(episode.loss))+ "Error: "+str(avg_error)))#+" Accuracy: "+str(correct_count*1.0/number_frames)
                    #print "Error: "+str(avg_error)
                    for frame in range(0, episode.seq_len - 1):
                        reward = episode.calculate_reward(frame, person_id=indx)  # Return to this!
                    # Calculate discounted reward

                    episode.discounted_reward()
                    # Save all of the gathered statistics.
                    #  statistics, num_episode, poses, ped_seq_len=-1)
                    episode.save(statistics, num_episode, poses)
                    # if self.settings.curriculum_goal:
                    #     self.goal_dist_reached = episode.get_goal_dist_reached()
                    # else:
                    #     self.goal_dist_reached = -1
                    # Increase number of processed episodes.
                    num_episode += 1
                    self.num_sucessful_episodes += 1
                    cum_r += np.sum(np.copy(episode.reward))

                    # print "  Reward: " + str(cum_r / self.settings.update_frequency)

                #statistics = agent.evaluate(statistics, episode, poses)

                # Save the loss of the agent.

                # Save all statistics to a file.
                print(("Statistics file: " + self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1],
                                                                      training,
                                                                      saved_files_counter)))

                np.save(self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1], training, saved_files_counter)+".npy",
                        statistics)

                if np.sum(episode.reconstruction[0, :, :, 3]) > 0:
                    np.save(self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1], training,
                                                      saved_files_counter) + "reconstruction.npy",
                            episode.reconstruction[0, :, :, 3:5])
                saved_files_counter = saved_files_counter + 1
                return statistics, saved_files_counter, people_list, car_list, poses
                #return statistics, saved_files_counter, people_list, car_list
        else:

            statistics = np.zeros((repeat_rep, (episode.seq_len - 1), 38 + NBR_MEASURES + 1), dtype=np.float64)
            poses = np.zeros((repeat_rep, (episode.seq_len - 1), 93 + 2 + 1), dtype=np.float64)
            for itr in range(repeat_rep):
                print(("Repeat "+str(itr)+" of "+str(repeat_rep)))
                if len(episode.valid_keys) > 0:
                    self.save_people_cars(itr, episode, people_list, car_list)
                    frame_observed=12
                    pos, indx = episode.initial_position_n_frames(frame_observed, itr%len(episode.valid_keys))
                    while len(pos) > 0 and frame_observed<450:

                        print(("Initialize on frame " + str(frame_observed)))
                        agent.initial_position(pos, episode.goal, episode.current_frame)
                        agent.frame=frame_observed
                        print(("Done with initialization " + str(episode.goal_person_id)+" on frame "+str(frame_observed)+" "+str(episode.goal_person_id_val)))

                        if self.net:
                            self.net.reset_mem()
                        # This is the main loop of what happens at each frame!
                        correct_count = 0

                        number_frames = self.settings.N
                        if training and len(episode.valid_people_tracks[indx]) - frame_observed-1 < number_frames:
                            number_frames =min( len(episode.valid_people_tracks[indx]) -  frame_observed-1, len(episode.velocity)-frame_observed)
                        print(("Starting frame "+str(agent.frame)+" next: "+str(number_frames)))
                        for frame in range(frame_observed, min(frame_observed+number_frames,episode.ped_seq_len)):

                            value = agent.next_action(episode, training, filename=self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1], training, saved_files_counter) + '.pkl')#,filename_weights=
                                   #self.statistics_file_name(agent_file,  episode.pos[0], episode.pos[1], training, saved_files_counter) + '_weights.pkl'  )
                            # if len(value)>0:
                            #     if (episode.get_valid_action(agent.frame, indx) == episode.action[agent.frame]):
                            #         correct_count += 1

                            agent.perform_action(value, episode, prob=curriculum_prob, person_id=indx,training=False)
                        print(("Loss " + str(np.sum(episode.loss)) + " Accuracy: " + str( correct_count * 1.0 / number_frames)))

                        # print "Reward "+str(reward)
                        # print str(frame)+" Number non-zero" + str(len(np.nonzero(episode.reconstruction[:, :, :, 4])[0]))


                        # while frame in range(frame_observed, min(frame_observed+number_frames,episode.ped_seq_len)):
                        #     episode.time_dependent_evaluation(frame, episode.goal_person_id)  # Agent on Pavement?
                        #     episode.evaluate_measures(frame)  # Return to this!


                        # Calculate discounted reward

                        #episode.discounted_reward()
                        # Save all of the gathered statistics.
                        episode.save(statistics, num_episode, poses, episode.ped_seq_len)

                        # Increase number of processed episodes.
                        num_episode += 1
                        self.num_sucessful_episodes += 1
                        if self.num_sucessful_episodes% repeat_rep==0 and self.num_sucessful_episodes>0:
                            episode.save_loss(statistics)
                            np.save(self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1], training,
                                                              saved_files_counter)+".npy",
                                    statistics)
                            num_episode = 0
                            saved_files_counter = saved_files_counter + 1
                            print(("Statistics file: " + self.statistics_file_name(agent_file, episode.pos[0],
                                                                                  episode.pos[1], training,
                                                                                  saved_files_counter)))
                        cum_r += np.sum(np.copy(episode.reward))

                        print(("  Reward: " + str(cum_r / self.settings.update_frequency)))
                        frame_observed=frame_observed+1
                        pos, indx = episode.initial_position_n_frames(frame_observed, itr%len(episode.valid_keys))





        saved_files_counter = saved_files_counter + 1
        return statistics, saved_files_counter, people_list, car_list, poses


    def act_and_learn_eval(self, agent, agent_file, episode, poses_db, training, saved_files_counter, road_width=0,
                      curriculum_prob=0, time_file=None, pos_init=[], viz=False):


        print(training)
        num_episode = 0
        cum_r=0
        init_methods=[]
        repeat_rep,_ = self.get_repeat_reps_and_init(training, validation=True)

        statistics = np.zeros((repeat_rep, (episode.seq_len - 1), 38+NBR_MEASURES+1), dtype=np.float64)
        people_list=np.zeros((repeat_rep, episode.seq_len ,6), dtype=np.float64)
        car_list = np.zeros((repeat_rep, episode.seq_len,6), dtype=np.int)
        self.counter=0
        # Go through all episodes in a gradient batch.
        print((episode.seq_len))
        for ep_itr in range(repeat_rep):


            cum_r = 0
            pos=[]
            while len(pos)==0:
                initialization = self.get_next_initialization(training=False)
                pos, indx, vel = self.agent_initialization(agent, episode, ep_itr, poses_db, training, init_m=initialization)

            self.save_people_cars(ep_itr, episode, people_list, car_list)
            #print "Initial position "+str(pos)+" "
            agent.initial_position(pos, episode.goal)
            if self.net:
                self.net.reset_mem()
            # This is the main loop of what happens at each frame!

            for frame in range(0, episode.seq_len-1):
                value = agent.next_action(episode, training=False)
                #print str(frame) + " " + str(value) + " " + str(episode.agent[frame]) + " " + str(episode.agent[frame + 1])+ " "+str(agent.frame)+" "+str(agent.position)
                agent.perform_action(value, episode, training=False)
                #print str(frame) + " " + str(value)+" "+str(episode.agent[frame])+" "+str(episode.agent[frame+1])
            for frame in range(0, episode.seq_len - 1):
                reward= episode.calculate_reward(frame,episode_done=True )  # Return to this!
                #print str(frame)+" Number non-zero" + str(len(np.nonzero(episode.reconstruction[:, :, :, 4])[0]))

            # Calculate discounted reward
            episode.discounted_reward()
            # Save all of the gathered statistics.
            episode.save(statistics, num_episode)
            # if self.settings.curriculum_goal:
            #     self.goal_dist_reached=episode.get_goal_dist_reached()
            # else:
            #     self.goal_dist_reached =-1
            # Increase number of processed episodes.
            num_episode+=1
            self.num_sucessful_episodes += 1
            cum_r+=np.sum(np.copy(episode.reward))

            print(("  Reward: " + str(cum_r/self.settings.update_frequency)))


            # Train the agent or evaluate it.
            if training:
                statistics=agent.train(ep_itr, statistics, episode,self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1], training, saved_files_counter) + '.pkl',
                                       self.statistics_file_name(agent_file,  episode.pos[0], episode.pos[1], training, saved_files_counter) + '_weights.pkl' )
            else:
                statistics=agent.evaluate(ep_itr,statistics, episode)


            # Save the loss of the agent.
            #episode.save_loss(statistics)
            # Save all statistics to a file.
        print(("Statistics file: "+self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1], training, saved_files_counter)))

        np.save(self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1], training, saved_files_counter)+".npy", statistics)

        if np.sum(episode.reconstruction[0,:,:,3])>0:
            np.save(self.statistics_file_name(agent_file, episode.pos[0], episode.pos[1], training, saved_files_counter)+"reconstruction.npy", episode.reconstruction[0,:,:,3:5])
        saved_files_counter = saved_files_counter + 1
        return statistics, saved_files_counter, people_list, car_list

    # def set_up_episode(self, cachePrefix, envPly, pos_x, pos_y, training, useCaching, evaluate=False, time_file=None,
    #                    seq_len_pfnn=-1, datasetOptions=None,supervised=False):
    #     super(SupervisedEnvironment).set_up_episode(cachePrefix, envPly, pos_x, pos_y, training, useCaching,
    #                                                      evaluate,
    #                                                      time_file, seq_len_pfnn, datasetOptions,
    #                                                      supervised=True)




class SupervisedWaymoEnvironment(WaymoEnvironment, SupervisedEnvironment):
    def __init__(self, path, sess, writer, gradBuffer, log, settings, net=None):
        super(SupervisedWaymoEnvironment, self).__init__(path, sess, writer, gradBuffer, log, settings, net=net)
        self.init_methods_train = [-1]
        self.init_methods_test = [-1]
        self.init_methods_val = [-1]  # [8, 6, 2, 4]
        print("Supervised Env ")




class SupervisedCARLAEnvironment(SupervisedEnvironment, CARLAEnvironment):
    def __init__(self, path, sess, writer, gradBuffer, log, settings, net=None):
        super(SupervisedCARLAEnvironment, self).__init__(path, sess, writer, gradBuffer, log, settings, net=net)
        self.init_methods_train = [-1]
        self.init_methods_test = [-1]
        self.init_methods_val = [-1]  # [8, 6, 2, 4]
        print("Supervised Env ")
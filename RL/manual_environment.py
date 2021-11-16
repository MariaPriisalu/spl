import os
from datetime import datetime
external=False

import numpy as np
from environment import Environment
from carla_environment import CARLAEnvironment
from visualization import make_movie, make_movie_manual
from environment_waymo import WaymoEnvironment

from PFNNPythonLiveVisualizer import PFNNLiveVisualizer as PFNNViz
from environment_abstract import AbstractEnvironment

from RL.settings import PEDESTRIAN_MEASURES_INDX

def getBonePos(poseData, boneId):
    x = poseData[boneId*3 + 0]
    y = poseData[boneId*3 + 1]
    z = poseData[boneId*3 + 2]
    return x,y,z

def distance2D(x0, z0, x1, z1):
    return math.sqrt((x0-x1)**2 + (z0-z1)**2)

def prettyPrintBonePos(poseData, boneID):
    print(("Bone {0} has position: ({1:.2f},{2:.2f},{3:.2f})".format(boneID, *getBonePos(poseData, boneID))))


np.set_printoptions(precision=2)



class ManualAbstractEnvironment(AbstractEnvironment):


    def visualize(self,episode, file_name, statistics, training, poses,initialization_map, initialization_car, agent, statistics_car):
        pass

    def return_highest_object(self, tens):
        z = 0
        while z < tens.shape[0] - 1 and np.linalg.norm(tens[z])==0  :
            z = z + 1
        return z

    def return_highest_col(self, tens):
        z = 0
        while z < tens.shape[0] - 1 and np.linalg.norm(tens[z,:])==0  :
            z = z + 1
        return z



    def print_reward_after_discounting(self, cum_r, episode):

        print(("Reward after discounting: " + str(episode.reward)))
        print(("  Reward: " + str(cum_r / self.settings.update_frequency)))

    def initialize_img_above(self):
        img_from_above = np.zeros(1)
        return img_from_above

    def get_joints_and_labels(self, agent):
        #    label_mapping[0] = 11 # Building
        #    label_mapping[1] = 13 # Fence
        #    label_mapping[2] = 4 # Other/Static, 'guard rail' , dynamic, 23-'sky' ,'bridge', tunnel
        #    label_mapping[3] = 17 # Pole, polegroup
        #    label_mapping[4] = 7 # Road, 'road', 'parking', 6-'ground', 'railtrack'
        #    label_mapping[5] = 8 # Sidewalk
        #    label_mapping[6] = 21 # Vegetation, 'terrain'
        #    label_mapping[7] = 12 # Wall
        #    label_mapping[8] = 20 # Traffic sign, traffic light
        # Go through all episodes in a gradient batch.
        labels_indx = {11: 0, 13: 1, 14: 2, 4: 2, 5: 2, 15: 2, 16: 2, 17: 3, 18: 3, 7: 4, 9: 4, 6: 4, 10: 4,
                       8: 5, 21: 6, 22: 6, 12: 7, 20: 8, 19: 8}
        if agent.PFNN:
            poseData = agent.PFNN.getCurrentPose()  # Returns a numpy representing the pose
            jointsParents = agent.PFNN.getJointParents()  # Returns the parent of each of the numjoints bones (same numpy)
        else:
            jointsParents = None
        return jointsParents, labels_indx

    def print_reward_for_manual_inspection(self, episode, frame, img_from_above, jointsParents, reward):
        reward = episode.calculate_reward(frame, episode_done=False, print_reward_components=True)  # Return to this!
        print(("Reward \t" + " " + str(reward))+" goal "+str(episode.goal))
        if episode.useRealTimeEnv:
            print (" Car reward  "+str(episode.reward_car[frame]))
        if frame % self.settings.action_freq == 0:
            goal_r = 0
            print ("measures- agent frame "+str(frame))

            for key, value in PEDESTRIAN_MEASURES_INDX.items():
                print (str(key)+" agent "+str(episode.measures[frame, value]))#+" car "+str(episode.measures_car[frame, value]))

            print(("Penalty for hitting pedestrians: " + str(dir)))
            print(" reward "+str(reward)+" measures "+str(episode.measures[frame, 13]))
            if reward == 0 and episode.measures[frame, 13]:  # or frame>0:
                pass
            else:
                pass
                make_movie_manual(episode.people,
                                  episode.reconstruction,
                                  self.settings.width,
                                  self.settings.depth,
                                  episode.seq_len,
                                  self.settings.agent_shape,
                                  self.settings.agent_shape_s,
                                  episode,
                                  frame + 1,
                                  img_from_above,
                                  car_goal=episode.car_goal,
                                  name_movie='manual_agent',
                                  path_results=self.settings.LocalResultsPath,
                                  jointsParents=jointsParents)

    def print_agent_location(self, episode, frame, value):
        print(("Frame " + str(frame) + " agent took action: " + str(value) + " New position: " + str(
            episode.agent[frame + 1])))

    def print_input_for_manual_inspection(self, agent, episode, frame, img_from_above, jointsParents, labels_indx, training):
        #print("Car positions: ")
        # print((episode.cars[frame]))
        if frame % self.settings.action_freq == 0:
            make_movie_manual(episode.people,
                              episode.reconstruction,
                              self.settings.width,
                              self.settings.depth,
                              episode.seq_len,
                              self.settings.agent_shape,
                              self.settings.agent_shape_s,
                              episode,
                              frame,
                              img_from_above,
                              car_goal=episode.car_goal,
                              name_movie='manual_agent',
                              path_results=self.settings.LocalResultsPath,
                              jointsParents=jointsParents)
            if False:

                print("Agent training !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -------------------------------->")

                training_tmp = True
                input, people, cars, people_cv, cars_cv = episode.get_agent_neighbourhood(agent.position,
                                                                                          self.settings.agent_s,
                                                                                          frame,
                                                                                          training_tmp)
                if self.settings.run_2D:
                    sem = np.zeros((people.shape[0], people.shape[1], 2))
                else:
                    sem = np.zeros((people.shape[1], people.shape[2], 2))
                if self.settings.old:

                    sem = np.zeros((self.settings.net_size[1], self.settings.net_size[2], 9 + 7))

                    segmentation = (input[:, :, :, 3] * 33.0).astype(np.int)
                    for x in range(sem.shape[0]):
                        for y in range(sem.shape[1]):
                            sem[x, y, 0] = np.max(input[:, x, y, 0])
                            sem[x, y, 1] = np.max(input[:, x, y, 1])
                            sem[x, y, 2] = np.max(input[:, x, y, 2])
                            if training:
                                sem[x, y, 3] = self.settings.people_traj_gamma * np.max(input[:, x, y, 4])
                                sem[x, y, 4] = self.settings.people_traj_gamma * np.max(input[:, x, y, 5])

                            sem[x, y, 5] = np.max(people[:, x, y, 0])
                            sem[x, y, 6] = np.max(cars[:, x, y, 0])
                            for label in segmentation[:, x, y]:
                                if label > 0 and label != 23:
                                    sem[x, y, labels_indx[label] + 7] = 1
                else:
                    sem = np.zeros((self.settings.net_size[1], self.settings.net_size[2], 9 + 7))
                    segmentation = (input[:, :, 3] * 33.0).astype(np.int)
                    for x in range(sem.shape[0]):
                        for y in range(sem.shape[1]):

                            sem[x, y, 0] = input[x, y, 0]
                            sem[x, y, 1] = input[x, y, 1]
                            sem[x, y, 2] = input[x, y, 2]
                            if self.settings.predict_future:
                                # if training:
                                #     sem[x, y, 3] =tensor[self.return_highest_object(tensor[:,x,y,4]), x, y, 4]
                                #     sem[x, y, 4] =tensor[self.return_highest_object(tensor[:,x,y,5]), x, y, 5]
                                # else:
                                sem[x, y, 3] = people_cv[x, y, 0]
                                sem[x, y, 4] = cars_cv[x, y, 0]
                            elif training or self.settings.temporal:
                                sem[x, y, 3] = self.settings.people_traj_gamma * input[x, y, 4]
                                sem[x, y, 4] = self.settings.people_traj_gamma * input[x, y, 5]

                            sem[x, y, 5] = np.max(people[x, y, 0])
                            sem[x, y, 6] = np.max(cars[x, y, 0])

                            if segmentation[x, y] > 0 and segmentation[x, y] != 23 and segmentation[ x, y]in labels_indx:
                                sem[x, y, labels_indx[segmentation[x, y]] + 7] = 1
                print(("Input cars " + str(np.sum(sem[:, :, 5])) + " people " + str(
                    np.sum(sem[:, :, 6])) + " Input cars traj" + str(np.sum(sem[:, :, 4])) + " people traj " + str(
                    np.sum(sem[:, :, 3]))))
                # print "Output shape: "+str(sem[:, :, 0].shape)

                #    label_mapping[0] = 11 # Building
                #    label_mapping[1] = 13 # Fence
                #    label_mapping[2] = 4 # Other/Static, 'guard rail' , dynamic, 23-'sky' ,'bridge', tunnel
                #    label_mapping[3] = 17 # Pole, polegroup
                #    label_mapping[4] = 7 # Road, 'road', 'parking', 6-'ground', 'railtrack'
                #    label_mapping[5] = 8 # Sidewalk
                #    label_mapping[6] = 21 # Vegetation, 'terrain'
                #    label_mapping[7] = 12 # Wall
                #    label_mapping[8] = 20 # Traffic sign, traffic light

                print(("Input building " + str(np.sum(sem[:, :, 7])) + " fence " + str(
                    np.sum(sem[:, :, 7 + 1])) + " static " + str(np.sum(sem[:, :, 7 + 2])) + " pole " + str(
                    np.sum(sem[:, :, 7 + 3]))))
                print(("Input sidewalk " + str(np.sum(sem[:, :, 7 + 5])) + " road " + str(
                    np.sum(sem[:, :, 7 + 4])) + " veg. " + str(
                    np.sum(sem[:, :, 7 + 6])) + " wall " + str(np.sum(sem[:, :, 7 + 7])) + " sign " + str(
                    np.sum(sem[:, :, 7 + 8]))))

                # for x in range(sem.shape[0]):
                #     for y in range(sem.shape[1]):
                #         if self.settings.predict_future:
                #             if training_tmp:
                #                 if not self.settings.run_2D:
                #                     sem[x, y, 0] = input[self.return_highest_object(input[:, x, y, 4]), x, y, 4]
                #                     sem[x, y, 1] = input[self.return_highest_object(input[:, x, y, 5]), x, y, 5]
                #                 else:
                #                     sem[x, y, 0] = input[ x, y, 4]
                #                     sem[x, y, 1] = input[ x, y, 5]
                #             else:
                #                 if not self.settings.run_2D:
                #                     sem[x, y, 0] = people_cv[self.return_highest_object(people_cv[:, x, y, 0]),x, y, 0]
                #                     sem[x, y, 1] = cars_cv[self.return_highest_object(cars_cv[:, x, y, 0]),x, y, 0]
                #                 else:
                #                     sem[x, y, 0] = people_cv[x, y, 0]
                #                     sem[x, y, 1] = cars_cv[ x, y, 0]
                #         elif training_tmp:
                #             if not self.settings.run_2D:
                #                 sem[x, y, 0] = self.settings.people_traj_gamma * input[self.return_highest_object(input[:, x, y, 4]),x, y, 4]
                #                 sem[x, y, 1] = self.settings.people_traj_gamma * input[self.return_highest_object(input[:, x, y, 4]),x, y, 5]
                #             else:
                #                 sem[x, y, 0] = self.settings.people_traj_gamma * input[ x, y, 4]
                #                 sem[x, y, 1] = self.settings.people_traj_gamma * input[ x, y, 5]



                # print "Input cars " + str(sem[:,:,1])

                # print "Input people " + str(sem[:,:,5])

                # print "Agent testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "
                # training_tmp=False
                # input, people, cars, people_cv, cars_cv = episode.get_agent_neighbourhood(agent.position,
                #                                                                           self.settings.agent_s,
                #                                                                           frame,
                #                                                                           training_tmp)
                # #print "Output shape: " + str(sem[:, :, 0].shape)
                # if self.settings.run_2D:
                #     sem = np.zeros((people.shape[0], people.shape[1], 2))
                # else:
                #     sem = np.zeros((people.shape[1], people.shape[2], 2))
                # for x in range(sem.shape[0]):
                #     for y in range(sem.shape[1]):
                #         if self.settings.predict_future:
                #             if training_tmp:
                #                 if not self.settings.run_2D:
                #                     sem[x, y, 0] = input[self.return_highest_object(input[:, x, y, 4]), x, y, 4]
                #                     sem[x, y, 1] = input[self.return_highest_object(input[:, x, y, 5]), x, y, 5]
                #                 else:
                #                     sem[x, y, 0] = input[ x, y, 4]
                #                     sem[x, y, 1] = input[ x, y, 5]
                #             else:
                #                 if not self.settings.run_2D:
                #                     sem[x, y, 0] = people_cv[self.return_highest_object(people_cv[:, x, y, 0]),x, y, 0]
                #                     sem[x, y, 1] = cars_cv[self.return_highest_object(cars_cv[:, x, y, 0]),x, y, 0]
                #                 else:
                #                     sem[x, y, 0] = people_cv[x, y, 0]
                #                     sem[x, y, 1] = cars_cv[ x, y, 0]
                #         elif training_tmp:
                #             if not self.settings.run_2D:
                #                 sem[x, y, 0] = self.settings.people_traj_gamma * input[self.return_highest_object(input[:, x, y, 4]),x, y, 4]
                #                 sem[x, y, 1] = self.settings.people_traj_gamma * input[self.return_highest_object(input[:, x, y, 5]),x, y, 5]
                #             else:
                #                 sem[x, y, 0] = self.settings.people_traj_gamma * input[ x, y, 4]
                #                 sem[x, y, 1] = self.settings.people_traj_gamma * input[ x, y, 5]


                goal_dir = episode.get_goal_dir(episode.agent[frame], episode.goal)

                print(("Goal_dir " + str(goal_dir)))

                car_in = episode.get_input_cars(episode.agent[frame], frame)

                print(("Car dir " + str(car_in)))

                agent_frame = frame
                if self.settings.action_mem:
                    values = np.zeros((9 + 1) * self.settings.action_mem)
                    if self.settings.old:
                        for past_frame in range(1, self.settings.action_mem + 1):
                            if agent_frame - past_frame >= 0:
                                # print str(len(episode.action))+" "+str(max(agent_frame - past_frame, 0))
                                # print (past_frame-1)*(9+1)+int(episode.action[max(agent_frame - past_frame, 0)])
                                # print values.shape
                                values[(past_frame - 1) * (9 + 1) + int(
                                    episode.action[max(agent_frame - past_frame, 0)])] = 1
                                values[past_frame * past_frame - 1] = episode.measures[
                                    max(agent_frame - past_frame, 0), 3]
                        print(("Actions memory:" + str(np.sum(values))))
                        if self.settings.nbr_timesteps > 0:
                            values = np.zeros((9 + 1) * self.settings.nbr_timesteps)
                            for past_frame in range(self.settings.action_mem + 1, self.settings.nbr_timesteps + 1):
                                if agent_frame - past_frame >= 0:
                                    # print str(len(episode.action))+" "+str(max(agent_frame - past_frame, 0))
                                    # print (past_frame-1)*(9+1)+int(episode.action[max(agent_frame - past_frame, 0)])
                                    # print values.shape
                                    values[(past_frame - 1) * (9 + 1) + int(
                                        episode.action[max(agent_frame - past_frame, 0)])] = 1
                                    values[past_frame * past_frame - 1] = episode.measures[
                                        max(agent_frame - past_frame, 0), 3]
                            print(("Actions adiitional memory:" + str(np.sum(values))))
                    else:
                        for past_frame in range(1, self.settings.action_mem + 1):
                            if frame - past_frame >= 0:
                                pos = (past_frame - 1) * (9 + 1) + int(
                                    episode.action[max(frame - past_frame, 0)])
                                values[pos] = 1
                                values[past_frame * (9 + 1) - 1] = episode.measures[
                                    max(frame - past_frame, 0), 3]

                        print(("Actions memory:" + str(np.sum(values))))
                        if self.settings.nbr_timesteps > 0:
                            values = np.zeros((9 + 1) * self.settings.nbr_timesteps)
                            for past_frame in range(self.settings.action_mem + 1, self.settings.nbr_timesteps + 1):
                                if frame - past_frame >= 0:
                                    values[(past_frame - 2) * (9 + 1) + int(
                                        episode.action[max(frame - past_frame, 0)])] = 1
                                    values[(past_frame - 1) * (9 + 1) - 1] = \
                                        episode.measures[max(frame - past_frame - 1, 0), 3]
                            print(("Actions adiitional memory:" + str(np.sum(values))))


class ManualEnvironment(ManualAbstractEnvironment, Environment):
    def __init__(self, path, sess, writer, gradBuffer, log, settings, net=None):
        super(ManualEnvironment, self).__init__(path, sess, writer, gradBuffer, log, settings, net=net)
class ManualWaymoEnvironment(ManualAbstractEnvironment, WaymoEnvironment):
    def __init__(self, path, sess, writer, gradBuffer, log, settings, net=None):
        super(ManualWaymoEnvironment, self).__init__(path, sess, writer, gradBuffer, log,settings, net=net)
import random

class ManualCARLAEnvironment(ManualAbstractEnvironment, CARLAEnvironment):
    def __init__(self, path, sess, writer, gradBuffer, log, settings, net=None):
        super(ManualCARLAEnvironment, self).__init__(path, sess, writer, gradBuffer, log,settings, net=net)
        if self.settings.learn_init:
            self.init_methods = [7]  # [1,5]#[1, 3, 5, 9]#, 8, 6, 2, 4]#[1, 8, 6, 2, 4, 9,3,5, 10]#[1, 8, 6, 2, 4, 5, 9]#[8, 6, 2, 4] #<--- regular![1, 8, 6, 2, 4] [1]
            self.init_methods_train = [7]


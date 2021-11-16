import numpy as np

import tensorflow as tf
from settings import RANDOM_SEED
tf.set_random_seed(RANDOM_SEED)

from net_continous import GaussianNet
from supervised_net import SupervisedNet
from RL.settings import PEDESTRIAN_MEASURES_INDX

class Net_no_viz(GaussianNet):

    def __init__(self, settings):
        self.set_nbr_channels()
        super(Net_no_viz, self).__init__(settings)
        self.settings=settings
        self.actions = []

        v = [-1, 0, 1]
        j = 0
        if self.settings.actions_3d:
            for z in range(3):
                for y in range(3):
                    for x in range(3):
                        self.actions.append([v[x], v[y], v[z]])
                        j += 1
        else:
            for y in range(3):
                for x in range(3):
                    self.actions.append([0, v[y], v[x]])
                    j += 1
        j = 0
        if self.settings.reorder_actions:
            self.actions = [self.actions[k] for k in [4, 1, 0, 3, 6, 7, 8, 5, 2]]


    def define_conv_layers(self):
        return []


    def fully_connected(self, dim_p, prev_layer):
        self.cars = tf.placeholder(shape=[self.settings.batch_size, 2], dtype=self.DTYPE, name="cars")
        self.action_mem = tf.placeholder(dtype=self.DTYPE, shape=(self.settings.batch_size, 2), name="action_mem_1")
        value = [0, -1, 1, 0, 1, 0, 0, 1]
        init = tf.constant_initializer(value)
        self.weights = tf.get_variable('weights', [4, 2], self.DTYPE,
                                       initializer=init)#tf.contrib.layers.xavier_initializer())#i
        # self.biases = self.bias_variable('biases', [2])
        self.all_input=tf.concat([self.cars,self.action_mem ], axis=1)
        self.mu = tf.matmul(self.all_input, self.weights)
        self.mu_vel = tf.reshape(self.mu, [2])

    # Note scale agent memory if output is scaled!
    def construct_feed_dict(self, episode_in, frame, agent_frame, training=True, distracted=False):
        feed_dict = {}
        # print "Construct feed dict " + str(frame) + " " + str(agent_frame)
        # feed_dict[self.goal_dir] = episode_in.get_goal_dir_cont(episode_in.agent[agent_frame], episode_in.goal)
        feed_dict[self.cars] = episode_in.get_input_cars_cont_linear(episode_in.agent[agent_frame], agent_frame,distracted)
        values = np.zeros(2)
        for past_frame in range(1, 2):
            if agent_frame - past_frame >= 0:
                # print "Mem frame "+str(max(agent_frame - past_frame, 0))
                values[(past_frame - 1) * 2] = self.get_memory_vel1_episode(max(agent_frame - past_frame, 0),
                                                                            episode_in)
                values[(past_frame - 1) * 2 + 1] = self.get_memory_vel2_episode(max(agent_frame - past_frame, 0),
                                                                                episode_in)
            else:
                values[(past_frame - 1) * 2] =episode_in.vel_init[1]
                values[(past_frame - 1) * 2+1] = episode_in.vel_init[2]
        feed_dict[self.action_mem] = np.expand_dims(values, axis=0)
        # print "Sample"

        return feed_dict  # self.sample: episode_in.velocity[self.frame][0]

    def grad_feed_dict(self, agent_action, agent_measures, agent_pos, agent_velocity, ep_itr, episode_in, frame,
                       reward, statistics,poses,priors, agent_speed=None, training=True, agent_frame=-1, statistics_car=None):
        if agent_frame < 0 or not training:
            agent_frame = frame
        # print "Gradient feed dict " +str(frame)+" " + str(agent_frame)

        r = reward[ep_itr, agent_frame]
        # print  "Reward "+str(r)
        if sum(agent_measures[ep_itr, :agent_frame, 0]) > 0 or sum(agent_measures[ep_itr, :agent_frame, 13]) > 0:
            r = 0
        # print "Frame "+str(frame)+" Episode "+str(ep_itr)+" Reward "+str(r)+" hit by car:"+str(agent_measures[ep_itr,agent_frame, 0])+" Reached goal "+str(agent_measures[ep_itr,agent_frame, 13])

        feed_dict = {self.advantages: r,self.sample: self.get_sample(statistics, ep_itr, agent_frame)}

        feed_dict[self.cars] = episode_in.get_input_cars_cont_linear(agent_pos[ep_itr, agent_frame, :], agent_frame, agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.distracted])

        # print "Action mem size "+str(self.settings.action_mem)+" action size "+str(action_size)+" total size "+str(action_size* self.settings.action_mem)
        values = np.zeros(2 * self.settings.action_mem)
        vel_init = statistics[ep_itr, -1, 3:6]
        for past_frame in range(1, 2):
            if agent_frame - past_frame >= 0:
                values[(past_frame - 1) * 2] = self.get_memory_vel1(max(agent_frame - past_frame, 0), ep_itr,
                                                                    statistics)

                values[(past_frame - 1) * 2 + 1] = self.get_memory_vel2(
                    max(agent_frame - past_frame, 0), ep_itr, statistics)
            else:
                values[(past_frame - 1) * 2] = vel_init[1]
                values[(past_frame - 1) * 2 + 1] = vel_init[2]
        feed_dict[self.action_mem] = np.expand_dims(values, axis=0)
        return feed_dict

    def calc_probabilities(self, fc_size):
        self.probabilities = self.mu



class Net_no_viz_supervised(SupervisedNet):

    def __init__(self, settings):
        self.set_nbr_channels()

        super(Net_no_viz_supervised, self).__init__(settings)

        self.settings=settings

    def get_memory_vel2_episode(self, frame, episode_in):
        return episode_in.get_valid_vel(frame, episode_in.goal_person_id_val)[2]

    def get_memory_vel1_episode(self, frame, episode_in):
        return episode_in.get_valid_vel(frame, episode_in.goal_person_id_val)[1]


    def fully_connected(self,dim_p, prev_layer):
        self.action_mem = tf.placeholder(dtype=self.DTYPE, shape=( None,2), name="action_mem_1")
        self.weights = tf.get_variable('weights', [2, 2], self.DTYPE,
                                  initializer=tf.contrib.layers.xavier_initializer())
        #self.biases = self.bias_variable('biases', [2])

        self.mu = tf.matmul( self.action_mem , self.weights)
        #self.mu_b = tf.add(self.mu, self.biases)

        self.mu_vel=tf.reshape(self.mu,[2])

    def calc_probabilities(self, fc_size):

        return

    def construct_feed_dict(self, episode_in, frame, agent_frame,training=True, distracted=False):
        feed_dict ={}
        #print "Construct feed dict " + str(frame) + " " + str(agent_frame)
        #feed_dict[self.goal_dir] = episode_in.get_goal_dir_cont(episode_in.agent[agent_frame], episode_in.goal)
        values = np.zeros(2)
        for past_frame in range(1, 2):
            if agent_frame - past_frame >= 0:
                #print "Mem frame "+str(max(agent_frame - past_frame, 0))
                values[(past_frame - 1) * 2] = self.get_memory_vel1_episode( max(agent_frame - past_frame, 0), episode_in)
                values[(past_frame - 1) * 2 + 1] = self.get_memory_vel2_episode( max(agent_frame - past_frame, 0), episode_in)
        feed_dict[self.action_mem] = np.expand_dims(values, axis=0)
        #print "Sample"
        feed_dict[self.sample] =episode_in.get_valid_vel(agent_frame, episode_in.goal_person_id_val)[1:]
        return feed_dict #self.sample: episode_in.velocity[self.frame][0]

    # def get_feature_vectors_gradient(self, agent_action, agent_frame, agent_measures, agent_pos, agent_speed, ep_itr,episode_in, feed_dict, frame, poses, statistics, training):
    #         feed_dict = {}
    #         dim_p=2
    #         values = np.zeros(4)
    #         for past_frame in range(1, 3):
    #              if agent_frame - past_frame >= 0:
    #
    #                  values[(past_frame - 1) * self.fully_connected_size(dim_p)] = self.get_memory_vel1_episode( max(agent_frame - past_frame, 0))
    #                  values[(past_frame - 1) * self.fully_connected_size(dim_p) + 1] = self.get_memory_vel2_episode( max(agent_frame - past_frame, 0))
    #
    #         feed_dict[self.action_mem] = np.expand_dims(values, axis=0)
    #         feed_dict[self.sample]= self.get_sample(statistics, ep_itr, agent_frame)
    #         return feed_dict


    def define_conv_layers(self):
        return []


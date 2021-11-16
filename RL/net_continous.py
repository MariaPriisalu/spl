import numpy as np

import tensorflow as tf
from settings import RANDOM_SEED
tf.compat.v1.set_random_seed(RANDOM_SEED)

from net import  variable_summaries
import math as m
import copy


import sys
np.set_printoptions(threshold=sys.maxsize)
from nets_abstract import SoftMaxNet
class  GaussianNet(SoftMaxNet):
    def __init__(self,settings):
        settings.velocity=False
        super(GaussianNet, self).__init__( settings)

        tf.compat.v1.set_random_seed(settings.random_seed_tf)
        np.random.seed(settings.random_seed_np)

    def get_dim_p(self):
        return 2



    def fully_connected(self, dim_p, prev_layer):
        self.action_size_mem = 2
        super(GaussianNet, self).fully_connected( dim_p, prev_layer)

    def define_loss(self, dim_p):
        #self.entropy = - tf.reduce_sum(self.mu * tf.log(self.mu))
        self.sample = tf.placeholder(shape=[2], dtype=self.DTYPE,name="sample")
        pi = tf.constant(m.pi)
        if self.settings.sigma_vel<0:
            self.sigma_vel=tf.exp(self.logstd)
            self.normal_dist = tf.contrib.distributions.Normal(self.mu_vel, self.sigma_vel)
            self.responsible_output = self.normal_dist.prob(self.sample)
            self.loss=0.5 * tf.reduce_sum(tf.square((self.sample - self.mu_vel) / self.sigma_vel), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(self.sample)[-1]) + tf.reduce_sum(self.logstd, axis=-1)
            self.loss=self.loss*self.advantages
            self.loss=self.loss-(1e-1* tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1))
        else:
            self.normal_dist = tf.contrib.distributions.Normal(self.mu_vel, self.settings.sigma_vel)
            self.responsible_output = self.normal_dist.prob(self.sample)
            # self.loss = -tf.reduce_sum((self.normal_dist.log_prob(self.sample) * self.advantages))
            # self.loss -= 1e-1 * tf.reduce_sum(self.normal_dist.entropy())

            self.l2_loss=tf.nn.l2_loss(self.mu_vel-self.sample)#tf.nn.l2_loss
            self.loss = tf.reduce_mean(self.advantages*(2*tf.log(self.settings.sigma_vel)+tf.log(2*pi)+(self.l2_loss/(self.settings.sigma_vel*self.settings.sigma_vel))))
            #self.loss = tf.reduce_mean(self.advantages*self.l2_loss)




        return self.sample, self.loss

    def importance_sample_weight(self, responsible, statistics, ep_itr, frame, responsible_v=0):
        probabilities = statistics[ep_itr, frame, 7:34]
        mu = probabilities[0:2]
        sigma = probabilities[3]
        frameRate, frameTime = self.settings.getFrameRateAndTime()
        vel = statistics[ep_itr, frame, 4:6]*frameRate/5.0
        return np.exp(np.log(responsible[0]) -np.log( self.normalpdf(mu[1], sigma, vel[0])) +np.log( responsible[1])-np.log( self.normalpdf(mu[0], sigma, vel[1])))

    def apply_net(self, feed_dict, episode, frame, training, max_val=False, viz=False):
        sigma = [self.settings.sigma_vel, self.settings.sigma_vel]
        if self.settings.sigma_vel > 0:
            mean_vel, summary_train, mu_out, prob = self.sess.run([self.mu_vel, self.train_summaries, self.mu,   self.probabilities], feed_dict)

            #mu_out_3D=np.array([0,mean_vel[0], mean_vel[1]])
            #print " fc_out "+str(fc_out)+" prob: "+str(prob)

            # print episode.find_action_to_direction(mu_out_3D,np.linalg.norm(mean_vel))

        else:
            mean_vel, sigma, summary_train, log_sigma = self.sess.run([self.mu_vel, self.sigma_vel, self.train_summaries, self.logstd],
                                                           feed_dict)

        # print "Fully connected 1st layer: "+str(fc_1)
        # print "Fully connected out "+str(fc_out)
        # print "Mean output "+str(mean_vel)+" var "+str( self.settings.sigma_vel)
        self.take_step(episode, frame, max_val, mean_vel, sigma, training)

        return episode.velocity[frame]

    def take_step(self, episode, frame, max_val, mean_vel, sigma, training):

        if episode.goal_person_id >= 0 and frame + 1 < len(episode.valid_people_tracks[
                                                              episode.goal_person_id_val]) and training:  # np.linalg.norm(self.actions[value]) >0:

            track_length = len(episode.valid_people_tracks[episode.goal_person_id_val])
            next_frame = min(frame + 1, track_length - 1)
            current_frame = min(frame, track_length - 1)
            #print "Agent on pedestrian " + str(episode.valid_people_tracks[episode.goal_person_id_val][next_frame])
            episode.velocity[frame] = np.mean(episode.valid_people_tracks[episode.goal_person_id_val][next_frame] -
                                              episode.valid_people_tracks[episode.goal_person_id_val][current_frame],
                                              axis=1)
            episode.speed[frame] = np.linalg.norm(episode.velocity[frame])
            episode.action[frame] = episode.find_action_to_direction(episode.velocity[frame], episode.speed[frame])
            episode.probabilities[frame, 0:2] = np.copy(mean_vel)
            episode.probabilities[frame, 2:4] = np.copy(sigma)
            # print "Follow pedestrian action: " + str(episode.action[frame]) + " " + str(episode.velocity[frame])
            #print "Velocity "+str(episode.velocity[frame])
            # print "mode " + str(np.argmax(probabilities))+ " "+ "value "+str(value)
        else:

            if max_val and not training:
                speed_value_y = [mean_vel[0]]
                # print "Random y " + str(speed_value_y)
                speed_value_z = [mean_vel[1]]
            else:
                speed_value_y = np.random.normal(mean_vel[0], sigma[0], 1)
                #print "Random y " + str(speed_value_y)
                speed_value_z = np.random.normal(mean_vel[1], sigma[1], 1)
                #print "Random z " + str(speed_value_y)+" "+str(speed_value_z)+" mean: "+str(mean_vel[0])+" "+str(mean_vel[1])
            speed_value = np.sqrt(speed_value_y[0] ** 2 + speed_value_z[0] ** 2)
            episode.velocity[frame] = np.zeros(3)
            if speed_value > 1e-3:
                default_speed = 3.0
                # if episode.goal_person_id >= 0 and frame + 1 < len(
                #         episode.valid_people_tracks[episode.goal_person_id_val]):
                #     default_speed = np.linalg.norm(episode.valid_people_tracks[episode.goal_person_id_val][frame + 1] -
                #                                    episode.valid_people_tracks[episode.goal_person_id_val][frame])
                speed_value_y[0] = speed_value_y[0] * default_speed / speed_value
                speed_value_z[0] = speed_value_z[0] * default_speed / speed_value
                speed_value = default_speed

            episode.velocity[frame][1] = copy.copy(speed_value_y[0] * 5 / episode.frame_rate)
            episode.velocity[frame][2] = copy.copy(speed_value_z[0] * 5 / episode.frame_rate)
            episode.speed[frame] = speed_value * 5 / episode.frame_rate

            episode.action[frame] = episode.find_action_to_direction(episode.velocity[frame], episode.speed[frame])
            episode.probabilities[frame, 0:2] = np.copy(mean_vel)
            episode.probabilities[frame, 2:4] = np.copy(sigma)

    def fully_connected_size(self, dim_p):
        if self.settings.sigma_vel<0:
            return 4
        return 9#2

    def calc_probabilities(self, fc_size):
        if not self.settings.old_fc:

            #if not self.settings.lstm:
            if self.settings.goal_dir:
                self.fc_out = self.mu_dir
            elif self.settings.pose:
                print('Mu pose')
                self.fc_out = self.mu_pose
            elif self.settings.old_lstm:
                self.fc_out = self.mu_rnn
            else:
                self.fc_out = self.mu
            # else:
            #     self.fc_out = self.mu
            weights = tf.get_variable('weights_last', [9, 2], self.DTYPE,
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = self.bias_variable('biases_last', [2])
            self.fc_out_1=tf.matmul(self.fc_out, weights)
            self.fc_out_1 = tf.add(self.fc_out_1, biases)
            if self.settings.sigma_vel<0:
                mu_basis, logstd = tf.split(self.fc_out_1, [2,2],axis=0)
                self.probabilities = tf.reshape(tf.sigmoid(mu_basis + np.finfo(np.float32).eps), [4])
                self.logstd=self.probabilities * 0.0 + logstd
                self.mu_vel=self.probabilities** 6 - 3
            else:
                self.probabilities = tf.reshape(
                    tf.nn.softmax(self.fc_out_1 + np.finfo(np.float32).eps + tf.slice(self.action_mem, [0, 0], [1, 2])),
                    [2])  # +tf.slice(self.action_mem, [0,0],[1,9])

                #self.probabilities = tf.reshape(tf.sigmoid(self.fc_out_1 + np.finfo(np.float32).eps), [2])
                self.mu_vel=self.probabilities*6-3
        else:
            weights = tf.get_variable('weights_last', [9, 2], self.DTYPE,
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = self.bias_variable('biases_last', [2])
            self.fc_out_1 = tf.matmul(self.fc_out, weights)
            self.fc_out_1 = tf.add(self.fc_out_1, biases)
            self.mu_vel = tf.reshape(tf.sigmoid(self.fc_out_1 + np.finfo(np.float32).eps), [2]) * 6-3#tf.nn.relu(self.fc_out + np.finfo(np.float32).eps), [2])#
            variable_summaries(self.mu_vel , name='softmax', summaries=self.train_summaries)

    def get_velocity(self, velocity, action, frame):
        return [velocity[frame]]

    # def get_feature_vectors_gradient(self, agent_action, agent_frame, agent_measures, agent_pos, agent_speed, ep_itr,
    #                                  episode_in, feed_dict, frame, poses, statistics, training):
    #     dim_p = self.get_dim_p()
    #     super(GaussianNet, self).get_feature_vectors_gradient(agent_action, agent_frame, agent_measures, agent_pos, agent_speed, ep_itr,
    #                                  episode_in, feed_dict, frame, poses, statistics, training)
    #     action_size = 2
    #
    #     # print "Action mem size "+str(self.settings.action_mem)+" action size "+str(action_size)+" total size "+str(action_size* self.settings.action_mem)
    #     values = np.zeros(action_size * self.settings.action_mem)
    #     vel_init = statistics[ep_itr, -1, 3:6]
    #     for past_frame in range(1, self.settings.action_mem + 1):
    #         if agent_frame - past_frame >= 0:
    #             values[(past_frame - 1) * action_size] = self.get_memory_vel1(max(agent_frame - past_frame, 0),ep_itr, statistics)
    #
    #             values[(past_frame - 1) * action_size + 1] = self.get_memory_vel2(
    #                 max(agent_frame - past_frame, 0), ep_itr, statistics)
    #         else:
    #             values[(past_frame - 1) * action_size] =vel_init[1]
    #             values[(past_frame - 1) * action_size+1] =vel_init[2]
    #
    #     feed_dict[self.action_mem] = np.expand_dims(values, axis=0)
    #     if self.settings.nbr_timesteps > 0:
    #         values = np.zeros(action_size * self.settings.nbr_timesteps)
    #         for past_frame in range(self.settings.action_mem + 1, self.settings.nbr_timesteps + 2):
    #             if agent_frame - past_frame >= 0:
    #                 values[(past_frame - 2) * action_size] = self.get_memory_vel1(
    #                     max(agent_frame - past_frame, 0), ep_itr, statistics)
    #                 values[(past_frame - 2) * action_size + 1] = self.get_memory_vel2(
    #                     max(agent_frame - past_frame, 0),ep_itr, statistics)
    #             else:
    #                 values[(past_frame - 2) * action_size] = vel_init[1]
    #                 values[(past_frame - 2) * action_size + 1] = vel_init[2]
    #         feed_dict[self.action_mem_further] = np.expand_dims(values, axis=0)
    #
    #
    # def get_memory_vel2(self, frame, ep_itr, statistics):
    #     return statistics[ep_itr, frame, 5]
    #
    # def get_memory_vel1(self, frame, ep_itr,statistics):
    #
    #     return statistics[ep_itr,frame, 4]
    #
    # def get_feature_vectors(self, agent_frame, episode_in, feed_dict, frame, distracted=False):
    #     dim_p = self.get_dim_p()
    #     action_size = 2
    #     super(GaussianNet, self).get_feature_vectors(agent_frame, episode_in, feed_dict, frame,distracted)
    #     values = np.zeros(action_size * self.settings.action_mem)
    #     for past_frame in range(1, self.settings.action_mem + 1):
    #         if agent_frame - past_frame >= 0:
    #
    #             values[(past_frame - 1) * action_size] = self.get_memory_vel1_episode(max(agent_frame - past_frame, 0),
    #                                                                                   episode_in)
    #             values[(past_frame - 1) * action_size + 1] = self.get_memory_vel2_episode(
    #                 max(agent_frame - past_frame, 0), episode_in)
    #         else:
    #
    #             values[(past_frame - 1) * action_size] =episode_in.vel_init[1]
    #             values[(past_frame - 1) * action_size+1] = episode_in.vel_init[2]
    #
    #     feed_dict[self.action_mem] = np.expand_dims(values, axis=0)
    #     if self.settings.nbr_timesteps > 0:
    #         values = np.zeros(action_size * self.settings.nbr_timesteps)
    #         for past_frame in range(self.settings.action_mem + 1, self.settings.nbr_timesteps + 2):
    #             if agent_frame - past_frame >= 0:
    #
    #                 values[(past_frame - 2) * action_size] = self.get_memory_vel1_episode(
    #                     max(agent_frame - past_frame, 0), episode_in)
    #                 values[(past_frame - 2) * action_size + 1] = self.get_memory_vel2_episode(
    #                     max(agent_frame - past_frame, 0), episode_in)
    #             else:
    #
    #                 values[(past_frame - 2) * action_size] = episode_in.vel_init[1]
    #                 values[(past_frame - 2) * action_size+1] = episode_in.vel_init[2]
    #         feed_dict[self.action_mem_further] = np.expand_dims(values, axis=0)
    #
    #
    # def get_memory_vel2_episode(self, frame, episode_in):
    #     return episode_in.velocity[frame][2]
    #
    # def get_memory_vel1_episode(self, frame, episode_in):
    #     return episode_in.velocity[frame][1]

    def get_sample(self, statistics, ep_itr, agent_frame):
        frameRate, frameTime = self.settings.getFrameRateAndTime()
        return statistics[ep_itr, agent_frame, 4:6] * frameRate / 5.0


class Angular_Net(GaussianNet):
    def define_loss(self, dim_p):
        # self.entropy = - tf.reduce_sum(self.mu * tf.log(self.mu))
        self.sample = tf.placeholder(shape=[2], dtype=self.DTYPE, name="sample")
        # pi = tf.constant(m.pi)
        # self.l2_loss_vel = tf.nn.l2_loss(self.mu_vel - self.sample)  # tf.nn.l2_loss
        # self.l2_loss_angle = tf.nn.l2_loss(self.mu_vel - self.sample)  # tf.nn.l2_loss
        # self.loss = tf.reduce_mean(self.advantages * (2 * tf.log(self.settings.sigma_vel) + tf.log(2 * pi) + (
        # self.l2_loss / (self.settings.sigma_vel * self.settings.sigma_vel))))
        # self.loss = tf.reduce_mean(self.advantages*self.l2_loss)
        self.normal_dist = tf.contrib.distributions.Normal(self.mu_vel, self.settings.sigma_vel)
        self.responsible_output = self.normal_dist.prob(self.sample)
        self.loss = -tf.reduce_sum((self.normal_dist.log_prob(self.sample) * self.advantages))
        self.loss -= 1e-1 * tf.reduce_sum(self.normal_dist.entropy())

        return self.sample, self.loss

    def calc_probabilities(self, fc_size):
        if self.settings.old_fc:
            if self.settings.lstm:
                self.probabilities = tf.reshape(tf.nn.softmax(self.mu + np.finfo(np.float32).eps), [fc_size])
            else:
                self.probabilities = tf.reshape(tf.nn.softmax(self.mu + np.finfo(np.float32).eps), [fc_size])
                if self.settings.goal_dir:
                    self.probabilities = tf.reshape(tf.nn.softmax(self.mu_dir + np.finfo(np.float32).eps), [fc_size])
                if self.settings.pose:
                    self.probabilities = tf.reshape(tf.nn.softmax(self.mu_pose + np.finfo(np.float32).eps), [fc_size])
                if self.settings.old_lstm:
                    self.probabilities = tf.reshape(tf.nn.softmax(self.mu_rnn + np.finfo(np.float32).eps), [fc_size])
            self.mu_vel = self.probabilities
        else:
            self.mu_vel = tf.reshape(tf.sigmoid(self.fc_out + np.finfo(np.float32).eps),
                                     [2])   # tf.nn.relu(self.fc_out + np.finfo(np.float32).eps), [2])#
            variable_summaries(self.mu_vel, name='softmax', summaries=self.train_summaries)


    def apply_net(self, feed_dict, episode, frame, training, max_val=False):
        sigma=[self.settings.sigma_vel, self.settings.sigma_vel]
        if self.settings.sigma_vel>0:
            mean_vel, summary_train = self.sess.run([self.mu_vel, self.train_summaries], feed_dict)
        else:
            mean_vel,sigma, summary_train = self.sess.run([self.mu_vel,self.sigma_vel, self.train_summaries], feed_dict)

        self.take_step(episode, frame, max_val, mean_vel, sigma, training)

        return episode.velocity[frame]

    def take_step(self, episode, frame, max_val, mean_vel, sigma, training):
        # print "Model output "+str(mean_vel)
        if max_val and not training:
            stoch_speed = mean_vel[0]
            stoch_angle = mean_vel[1]

        else:
            stoch_speed = np.random.normal(mean_vel[0], sigma[0], 1)
            stoch_angle = np.random.normal(mean_vel[1], sigma[1], 1)
        speed_value = 2.3#max(min(stoch_speed, 1), 0) * 3
        angle = max(min(stoch_angle, 1), -1) * np.pi / 2
        # print "Speed before "+str(stoch_speed)+"  after: " + str(speed_value)+" angle before: "+str(stoch_angle)+" "+str(angle)

        episode.speed[frame] = speed_value * 5 / episode.frame_rate
        episode.action[frame] = angle
        episode.velocity[frame] = np.zeros(3)
        episode.velocity[frame][1] = copy.copy(episode.speed[frame] * np.cos(angle))
        episode.velocity[frame][2] = copy.copy(-episode.speed[frame] * np.sin(angle))
        # print "Action "+str(episode.velocity[frame])
        episode.action[frame] = 0
        episode.probabilities[frame, 0:2] = np.copy(mean_vel)
        episode.probabilities[frame, 2:4] = np.copy(sigma)

    # def fc_layer(self, dim_fc, dim_p, last_layer):
    #     self.angle = tf.placeholder(shape=[1,1], dtype=self.DTYPE,
    #                                    name="angle")
    #     print "Agent angle"
    #     last_layer = tf.concat([last_layer, self.angle], axis=1)
    #
    #     dim_fc = dim_fc + 1
    #     weights_fc_out = tf.get_variable('weights_fc_out', [dim_fc, self.fully_connected_size(dim_p)], self.DTYPE,
    #                                      initializer=tf.contrib.layers.xavier_initializer())
    #     self.fc_out = tf.matmul(last_layer, weights_fc_out)  # , bias_fc_out)


    def get_velocity(self, velocity, action, frame):
        return [velocity[frame]]

    def get_sample(self, statistics, ep_itr, agent_frame):
        frameRate, frameTime = self.settings.getFrameRateAndTime()
        sample = copy.copy(statistics[ep_itr, agent_frame, 4:6] * frameRate / 5.0) / 3.0
        sample[0]=np.sqrt(statistics[ep_itr, agent_frame, 4]**2+ statistics[ep_itr, agent_frame, 5]**2)
        sample[1]=statistics[ep_itr, agent_frame, 6]*2/np.pi#2/np.pi
        return sample

    # def get_memory_vel2_episode(self, agent_frame, episode_in, past_frame):
    #     return (episode_in.action[max(agent_frame - past_frame, 0), 6] )* 2/ np.pi
    #
    # def get_memory_vel1_episode(self, agent_frame, episode_in, past_frame):
    #     return (episode_in.speed[max( agent_frame - past_frame, 0)] * 17.0 / 5.0 ) / 3.0
    #
    #
    # def get_memory_vel2(self, agent_frame, ep_itr, past_frame, statistics):
    #     return (statistics[ep_itr, max(agent_frame - past_frame, 0), 6] *2/np.pi)
    #
    # def get_memory_vel1(self, agent_frame, ep_itr, past_frame, statistics):
    #     return (np.sqrt(statistics[ep_itr, max(agent_frame - past_frame, 0), 4]**2+ statistics[ep_itr, max(agent_frame - past_frame, 0), 5]**2))

    def get_memory_vel2_episode(self, frame, episode_in):
        if len(episode_in.agent[frame + 1]) > 0:
            # print "Frame "+str(frame +1 )+" "+str(episode_in.agent[frame+1])+" "+str(episode_in.agent[frame])
            # print " 2nd: "+str(episode_in.agent[frame+1]-episode_in.agent[frame])+" "+str(episode_in.agent[frame+1][2]-episode_in.agent[frame][2])
            return (episode_in.agent[frame+1][2]-episode_in.agent[frame][2])
        return 0

    def get_memory_vel1_episode(self, frame, episode_in):
        if len(episode_in.agent[frame + 1])>0:
            # print "Frame " + str(frame + 1) + " " + str(episode_in.agent[frame + 1]) + " " + str(
            #     episode_in.agent[frame])
            # print " 1st: " + str(episode_in.agent[frame + 1] - episode_in.agent[frame]) + " " + str(
            #    episode_in.agent[frame + 1][1] - episode_in.agent[frame][1])
            return ( episode_in.agent[frame + 1][1]-episode_in.agent[frame][1])
        return 0


    def get_memory_vel2(self, agent_frame, ep_itr,statistics):
        if statistics.shape[0]>agent_frame+1:
            return (statistics[ep_itr,agent_frame+1, 2] -statistics[ep_itr,agent_frame, 2])
        return 0

    def get_memory_vel1(self, agent_frame, ep_itr, statistics):
        if statistics.shape[0] > agent_frame + 1:
            return (statistics[ep_itr,agent_frame+1, 1] -statistics[ep_itr,agent_frame, 1])
        return 0

    def get_feature_vectors_gradient(self, agent_action, agent_frame, agent_measures, agent_pos, agent_speed, ep_itr,
                                     episode_in, feed_dict, frame, poses, statistics, training):
        super(Angular_Net, self).get_feature_vectors_gradient(agent_action, agent_frame, agent_measures, agent_pos, agent_speed, ep_itr,
                                     episode_in, feed_dict, frame, poses, statistics, training)
        feed_dict[self.angle]=np.expand_dims(np.array([statistics[ep_itr, max(agent_frame, 0), int(NUM_SEM_CLASSES)]/np.pi]), axis=0)
        # rotation_matrix = np.identity(2)
        # rotation_matrix[0,0] = np.cos(statistics[ep_itr, agent_frame , int(NUM_SEM_CLASSES)])
        # rotation_matrix[1,1]= np.cos(statistics[ep_itr, agent_frame, int(NUM_SEM_CLASSES)])
        # rotation_matrix[0, 1] = -np.sin(statistics[ep_itr, agent_frame, int(NUM_SEM_CLASSES)])
        # rotation_matrix[1, 0] = np.sin(statistics[ep_itr, agent_frame , int(NUM_SEM_CLASSES)])
        # print rotation_matrix
        # if self.settings.car_var:
        #     feed_dict[self.cars] = np.transpose(
        #         np.matmul(rotation_matrix, np.transpose(np.array(feed_dict[self.cars]))))
        # if self.settings.goal_dir:
        #     feed_dict[self.goal_dir] = np.transpose(
        #         np.matmul(rotation_matrix, np.transpose(np.array(feed_dict[self.goal_dir]))))

        return feed_dict


    def get_feature_vectors(self, agent_frame, episode_in, feed_dict, frame):
        super(Angular_Net, self).get_feature_vectors(agent_frame, episode_in, feed_dict, frame)
        rotation_matrix = np.identity(2)
        feed_dict[self.angle] = np.expand_dims(np.array([episode_in.angle[agent_frame] / np.pi]), axis=0)
        # print "Angle "+str(episode_in.angle[agent_frame])+" "+str(agent_frame)
        # rotation_matrix[0, 0] = np.cos(episode_in.angle[agent_frame])
        # rotation_matrix[1, 1] = np.cos(episode_in.angle[agent_frame])
        # rotation_matrix[0, 1] = -np.sin(episode_in.angle[agent_frame])
        # rotation_matrix[1, 0] = np.sin(episode_in.angle[agent_frame])
        # print rotation_matrix
        # if self.settings.car_var:
        #     #print "car var before rotation " +str(feed_dict[self.cars]* np.sqrt(episode_in.reconstruction.shape[1] ** 2 + episode_in.reconstruction.shape[2] ** 2))
        #     feed_dict[self.cars] = np.transpose(np.matmul(rotation_matrix, np.transpose(np.array(feed_dict[self.cars]))))
        #     #print "car var after rotation: "+str(feed_dict[self.cars]* np.sqrt(episode_in.reconstruction.shape[1] ** 2 + episode_in.reconstruction.shape[2] ** 2))
        # if self.settings.goal_dir:
        #     #print "goal before rotation " + str(feed_dict[self.goal_dir]* np.sqrt(episode_in.reconstruction.shape[1] ** 2 + episode_in.reconstruction.shape[2] ** 2))
        #     feed_dict[self.goal_dir] = np.transpose(np.matmul(rotation_matrix, np.transpose(np.array(feed_dict[self.goal_dir]))))
        #     #print "goal after rotation " + str(feed_dict[self.goal_dir]* np.sqrt(episode_in.reconstruction.shape[1] ** 2 + episode_in.reconstruction.shape[2] ** 2))

        return feed_dict


class Angular_No_vel_Net(Angular_Net):


    def apply_net(self, feed_dict, episode, frame, training, max_val=False):
        mean_vel, summary_train = self.sess.run([self.mu_vel, self.train_summaries], feed_dict)
        print(("Model output "+str(mean_vel)))
        #stoch_speed=np.random.normal(mean_vel[0], self.settings.sigma_vel, 1)
        stoch_angle=np.random.normal(mean_vel[1], self.settings.sigma_vel, 1)
        speed_value= 3#max(min(stoch_speed,1),0)*3
        angle =max( min(stoch_angle,1),-1)*np.pi/2
        print(("Speed before "+str(stoch_speed)+"  after: " + str(speed_value)+" angle before: "+str(stoch_angle)+" "+str(angle)))

        episode.speed[frame] = speed_value * 5 / episode.frame_rate
        episode.action[frame]=angle
        episode.velocity[frame] = np.zeros(3)

        episode.velocity[frame][1] = copy.copy(episode.speed[frame]*np.cos(angle))
        episode.velocity[frame][2] = copy.copy(-episode.speed[frame]*np.sin(angle))
        print(("Action "+str(episode.velocity[frame])))
        episode.action[frame] = episode.find_action_to_direction(episode.velocity[frame], episode.speed[frame] )
        episode.probabilities[frame, 0:2] = np.copy(mean_vel)
        episode.probabilities[frame, 3] = np.copy(self.settings.sigma_vel)

        return episode.velocity[frame]



import tensorflow as tf
tf.compat.v1.set_random_seed(RANDOM_SEED)
from net_continous import GaussianNet

import numpy as np
class ContinousMemNet(GaussianNet):
    def __init__(self, settings):
        self.action_size_mem = 2
        super(ContinousMemNet, self).__init__(settings)
        tf.compat.v1.set_random_seed(settings.random_seed_tf)
        np.random.seed(settings.random_seed_np)

    def get_dim_p(self):
        return 2

    def fully_connected(self, dim_p, prev_layer):
        self.action_size_mem = 2
        super(ContinousMemNet, self).fully_connected( dim_p, prev_layer)


    def get_feature_vectors(self, agent_frame, episode_in, feed_dict, frame):
        super(ContinousMemNet, self).get_feature_vectors( agent_frame, episode_in, feed_dict, frame)
        # if episode_in.goal_person_id>0:
        #     #print str(len(episode_in.valid_people_tracks[episode_in.goal_person_id]))+" "+str(agent_frame)+" "+str(frame)
        #     agent_pos = np.mean(episode_in.valid_people_tracks[episode_in.goal_person_id][agent_frame], axis=1)
        dim_p = self.get_dim_p()

        if self.settings.action_mem:
            action_size = 2
            #print "Action mem size "+str(self.settings.action_mem)+" action size "+str(action_size)+" total size "+str(action_size* self.settings.action_mem)
            values = np.zeros(action_size* self.settings.action_mem)
            for past_frame in range(1, self.settings.action_mem+1):
                if agent_frame - past_frame >= 0:
                    values[(past_frame - 1) * action_size] = self.get_memory_vel1_episode(max(agent_frame - past_frame, 0), episode_in)
                    values[(past_frame - 1) * action_size + 1] = self.get_memory_vel2_episode(max(agent_frame - past_frame, 0), episode_in)

            feed_dict[self.action_mem] = np.expand_dims(values, axis=0)
            if self.settings.nbr_timesteps > 0:
                values = np.zeros(action_size* self.settings.nbr_timesteps)
                for past_frame in range(self.settings.action_mem + 1, self.settings.nbr_timesteps + 2):
                    if agent_frame - past_frame >= 0:

                        values[(past_frame - 2) * action_size] = self.get_memory_vel1_episode(max(agent_frame - past_frame, 0), episode_in)
                        values[(past_frame - 2) * action_size + 1] = self.get_memory_vel2_episode(max(agent_frame - past_frame, 0), episode_in)
                feed_dict[self.action_mem_further] = np.expand_dims(values, axis=0)



    def get_memory_vel2_episode(self, frame, episode_in):

        return episode_in.agent[min(frame + 1, len(episode_in.agent)-1)][2] - episode_in.agent[frame][2]

    def get_memory_vel1_episode(self, frame, episode_in):

        return episode_in.agent[min(frame+1, len(episode_in.agent)-1)][1]-episode_in.agent[frame][1]


    def get_feature_vectors_gradient(self, agent_action, agent_frame, agent_measures, agent_pos, agent_speed, ep_itr,
                                                          episode_in, feed_dict, frame, poses, statistics, training):
        dim_p = self.get_dim_p()
        super(ContinousMemNet, self).get_feature_vectors_gradient( agent_action, agent_frame, agent_measures, agent_pos, agent_speed, ep_itr,
                                                          episode_in, feed_dict, frame, poses, statistics, training)
        if self.settings.action_mem:
            #print dim_p
            values = np.zeros(dim_p  * self.settings.action_mem)
            for past_frame in range(1, self.settings.action_mem+1):
                if agent_frame - past_frame >= 0:

                        values[(past_frame - 1) * dim_p] = self.get_memory_vel1( max(agent_frame - past_frame, 0),
                                                                                                       ep_itr,
                                                                                                       statistics)
                        values[(past_frame - 1) * dim_p + 1] = self.get_memory_vel2(
                         max(agent_frame - past_frame, 0), ep_itr, statistics)

            feed_dict[self.action_mem] = np.expand_dims(values, axis=0)
            if self.settings.nbr_timesteps > 0:
                values = np.zeros(dim_p  * self.settings.nbr_timesteps)
                for past_frame in range(self.settings.action_mem+1, self.settings.nbr_timesteps + 1):
                    if agent_frame - past_frame >= 0:

                        values[(past_frame - 2) * dim_p] = self.get_memory_vel1(
                             max(agent_frame - past_frame, 0),
                             ep_itr,
                             statistics)
                        values[(past_frame - 2) * dim_p + 1] = self.get_memory_vel2(
                            max(agent_frame - past_frame, 0), ep_itr, statistics)
                feed_dict[self.action_mem_further] = np.expand_dims(values, axis=0)

    def get_memory_vel2(self, frame, ep_itr, statistics):
        #print "Frame "+str(frame )+" "+str(statistics[ep_itr, frame+1, :3])+" "+str(statistics[ep_itr, frame, :3])+" "+str(statistics[ep_itr, frame+1, 2]-statistics[ep_itr, frame, 2])
        return statistics[ep_itr, min(frame+1,statistics.shape[1]-1 ), 2]-statistics[ep_itr, frame, 2]

    def get_memory_vel1(self, frame, ep_itr,statistics):
        #print "Frame " + str(frame) + " " + str(statistics[ep_itr, frame + 1, :3]) + " " + str(
        #    statistics[ep_itr, frame, :3]) + " " + str(statistics[ep_itr, frame + 1, 1] - statistics[ep_itr, frame, 1])

        return statistics[ep_itr, min(frame+1,statistics.shape[1] -1), 1] - statistics[ep_itr, frame, 1]


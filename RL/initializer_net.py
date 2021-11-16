import numpy as np

from net import Net
from settings import NBR_MEASURES,RANDOM_SEED

import tensorflow as tf

tf.compat.v1.set_random_seed(RANDOM_SEED)
import tensorflow_probability as tfp
tfd = tfp.distributions

import pickle

import copy





class InitializerNet(Net):
    # Softmax Simplified
    def __init__(self, settings, weights_name="init") :
        self.labels_indx = {11: 0, 13: 1, 14: 2, 4: 2, 5: 2, 15: 2, 16: 2, 17: 3, 18: 3, 7: 4, 9: 4, 6: 4, 10: 4,
                            8: 5, 21: 6, 22: 6, 12: 7, 20: 8, 19: 8}
        self.feed_dict=[]
        self.valid_pos=[]
        self.probabilities_saved=[]
        self.carla = settings.carla
        self.set_nbr_channels()

        #self.temporal_scaling = 0.1 * 0.3

        super(InitializerNet, self).__init__(settings, weights_name=weights_name)

    def get_goal(self, statistics, ep_itr, agent_frame, initialization_car):
        #print ("Get goal "+str(statistics[ep_itr, 3:5, 38 + NBR_MEASURES].astype(int))+" size "+str(self.settings.env_shape[1:]) )
        if self.settings.goal_gaussian:
            goal=statistics[ep_itr, 3:5, 38 + NBR_MEASURES]
            agent_pos=statistics[ep_itr, 0,1:3]
            manual_goal=initialization_car[ep_itr,5:]
            goal_dir=goal-manual_goal#agent_pos
            goal_dir[0]=goal_dir[0]#/self.settings.env_shape[1]
            goal_dir[1] = goal_dir[1]# / self.settings.env_shape[2]
            # print ("Get goal " + str(goal_dir))
            return goal_dir
        else:
            # print ("Get goal " + str(
            #     np.ravel_multi_index( statistics[ep_itr, 3:5, 38 + NBR_MEASURES].astype(int),
            #                          self.settings.env_shape[1:])))
            return [np.ravel_multi_index( statistics[ep_itr, 3:5, 38 + NBR_MEASURES].astype(int),
                                    self.settings.env_shape[1:])]

    def set_nbr_channels(self):
        self.nbr_channels = 9 + 7

    def define_conv_layers(self):
        #tf.reset_default_graph()
        output_channels=1

        self.state_in = tf.compat.v1.placeholder(shape=[ self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2], self.nbr_channels],
                                       dtype=self.DTYPE,
                                       name="reconstruction")
        self.prior = tf.compat.v1.placeholder(
            shape=[self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2],1],
            dtype=self.DTYPE,
            name="prior")

        #mean = tf.constant(self.mean, dtype=self.DTYPE)  # (3)
        padding = 'SAME'
        if self.settings.inilitializer_interpolate:
            padding = 'VALID'
        prev_layer = tf.concat([self.state_in, self.prior], axis=3)  # - mean
        if self.settings.num_layers_init==1:
            with tf.compat.v1.variable_scope('conv1') as scope:
                out_filters = self.settings.outfilters[0]
                print("Define 2D weights") # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
                #kernel1 = tf.get_variable('weights', self.DTYPE,[ 3, 3, self.nbr_channels, out_filters],initializer=#tf.constant_initializer(1))#tf.contrib.layers.xavier_initializer())
                kernel1 = tf.get_variable('weights', [self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, output_channels], self.DTYPE,
                                          initializer=tf.contrib.layers.xavier_initializer())
                self.conv1 = tf.nn.conv2d(prev_layer, kernel1, padding=padding)  # [1, 2, 2, 2, 1]
                #biases1 = self.bias_variable('biases', [out_filters])

                #self.bias1 =self.conv_out1#tf.nn.bias_add(self.conv_out1, biases1)
                #self.conv1 #=tf.nn.relu(self.bias1, name=scope.name)

                if self.settings.learn_goal and not self.settings.goal_gaussian:
                    kernel1_goal = tf.get_variable('weights_goal',
                                              [self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels + 1,output_channels], self.DTYPE,
                                              initializer=tf.contrib.layers.xavier_initializer())
                    self.conv1_goal = tf.nn.conv2d(prev_layer, kernel1_goal, padding=padding)

                    prev_layer = tf.concat([self.conv1, self.conv1_goal],axis=3)
                else:
                    prev_layer = self.conv1

                if self.settings.inilitializer_interpolate:
                    prev_layer = tf.image.resize_images(prev_layer,[self.settings.env_shape[1],
                                                                  self.settings.env_shape[2]])
        else:

            print ("Two layers!")
            with tf.compat.v1.variable_scope('conv1') as scope:
                out_filters = self.settings.outfilters[0]
                print("Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
                # kernel1 = tf.get_variable('weights', self.DTYPE,[ 3, 3, self.nbr_channels, out_filters],initializer=#tf.constant_initializer(1))#tf.contrib.layers.xavier_initializer())
                kernel1 = tf.get_variable('weights',
                                          [3, 3, self.nbr_channels + 1,
                                           output_channels], self.DTYPE,
                                          initializer=tf.contrib.layers.xavier_initializer())
                self.conv1 = tf.nn.conv2d(prev_layer, kernel1, padding=padding)  # [1, 2, 2, 2, 1]
                prev_layer = self.conv1
                if 1 in self.settings.pooling:
                    self.pooled = tf.nn.max_pool(prev_layer, [1, 2, 2, 1], [1, 1, 1, 1], padding)  # , 'SAME')

                    prev_layer = self.pooled
                    print("Pooling 1")
                if self.settings.inilitializer_interpolate and self.settings.inilitializer_interpolated_add:
                    layer_1_interpolated = tf.image.resize_images(prev_layer,[self.settings.env_shape[1],
                                                                  self.settings.env_shape[2]])
                    print(" After reshaping size " + str(layer_1_interpolated.shape))
                in_filters = out_filters

                with tf.compat.v1.variable_scope('conv2') as scope:
                    out_filters = self.settings.outfilters[1]
                    kernel2 = tf.get_variable('weights_conv2', [2, 2, in_filters, out_filters], self.DTYPE,
                                              initializer=tf.contrib.layers.xavier_initializer())

                    self.conv_out2 = tf.nn.conv2d(prev_layer, kernel2, [1, 1, 1, 1], padding=padding)  # [1, 2, 2, 2, 1]
                    if self.settings.conv_bias:
                        biases2 = self.bias_variable('biases', [out_filters])

                        self.bias2 = tf.nn.bias_add(self.conv_out2, biases2)
                    else:
                        self.bias2 = self.conv_out2

                    self.conv2 = tf.nn.relu(self.bias2, name=scope.name)
                    # variable_summaries(biases2, name='conv2b', summaries=self.conv_summaries)

                prev_layer = self.conv2
                if 2 in self.settings.pooling:
                    self.pooled2 = tf.nn.max_pool(prev_layer, [1, 2, 2, 1], [1, 1, 1, 1], padding)  # , 'SAME')

                    prev_layer = self.pooled2
                    print("Pooling 2")

                if self.settings.inilitializer_interpolate:
                    layer_2_interpolated = tf.image.resize_images(prev_layer, [self.settings.env_shape[1],
                                                                              self.settings.env_shape[2]])
                    print(" After reshaping size "+str(layer_2_interpolated.shape))
                    if self.settings.inilitializer_interpolated_add:
                        self.conv_output = layer_2_interpolated+ layer_1_interpolated
                    else:
                        self.conv_output = layer_2_interpolated
                    prev_layer=self.conv_output
                else:
                    self.conv_output = prev_layer

        return prev_layer


    def define_loss(self, dim_p):

        self.sample = tf.compat.v1.placeholder(shape=[1], dtype=tf.int32, name="sample")
        self.responsible_output = tf.slice(self.probabilities, self.sample, [1]) + np.finfo(
            np.float32).eps
        if self.settings.learn_goal and not self.settings.goal_gaussian:
            self.prior_goal = tf.compat.v1.placeholder(
                shape=[self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2], 1],
                dtype=self.DTYPE,
                name="prior_goal")
            self.prior_flat_goal = tf.reshape(self.prior_goal, [
                self.settings.batch_size * self.settings.env_shape[1] * self.settings.env_shape[2]])
            self.goal = tf.compat.v1.placeholder(shape=[1], dtype=tf.int32, name="goal")
            self.responsible_output_goal = tf.slice(self.probabilities_goal, self.goal, [1]) + np.finfo(
                np.float32).eps
            self.responsible_output_prior_goal = tf.slice(self.prior_flat_goal, self.goal, [1]) + np.finfo(
                np.float32).eps
        self.prior_flat= tf.reshape(self.prior, [self.settings.batch_size*self.settings.env_shape[1]*self.settings.env_shape[2]])
        self.responsible_output_prior=tf.slice(self.prior_flat, self.sample, [1]) + np.finfo(
            np.float32).eps
        self.distribution=self.prior_flat*self.probabilities


        self.loss = -tf.reduce_mean((tf.log(self.responsible_output)+tf.log(self.responsible_output_prior) )* self.advantages)
        if self.settings.learn_goal and not self.settings.goal_gaussian:
            self.loss -= tf.reduce_mean(
                (tf.log(self.responsible_output_goal) + tf.log(self.responsible_output_prior_goal)) * self.advantages)

        if self.settings.entr_par_init: # To do: add entropy for goal!
            y_zeros = tf.zeros_like(self.distribution)
            y_mask = tf.math.greater(self.distribution, y_zeros)
            res = tf.boolean_mask(self.distribution, y_mask)
            logres = tf.math.log(res)

            self.entropy = tf.nn.softmax_cross_entropy_with_logits(labels=res, logits=logres)
            self.loss =self.loss -tf.reduce_mean(self.settings.entr_par_init*self.entropy)

        if self.settings.learn_goal and self.settings.goal_gaussian:
            self.goal = tf.compat.v1.placeholder(shape=[2], dtype=self.DTYPE, name="goal")
            self.normal_dist = tf.contrib.distributions.Normal(self.probabilities_goal, self.settings.goal_std)
            self.responsible_output = self.normal_dist.prob(self.goal)

            self.l2_loss = tf.nn.l2_loss(self.probabilities_goal - self.goal)  # tf.nn.l2_loss
            # self.loss = tf.reduce_mean(self.advantages * (2 * tf.log(self.settings.sigma_vel) + tf.log(2 * tf.pi) + (
            # self.l2_loss / (self.settings.sigma_vel * self.settings.sigma_vel))))
            self.loss =self.loss + tf.reduce_mean(self.advantages * self.l2_loss)

        if self.settings.learn_time and self.goal_net:
            self.time_requirement = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 1], dtype=self.DTYPE,
                                                        name="time_requirement")
            self.beta_distr=tfd.Beta(self.alpha, self.beta)
            self.time_prob=self.beta_distr.prob(self.time_requirement)
            self.loss = self.loss - tf.reduce_mean(self.advantages * tf.log(self.time_prob))

        return self.sample, self.loss

    def get_feature_vectors_gradient(self, agent_action, agent_frame, agent_measures, agent_pos, agent_speed, ep_itr,
                                     episode_in, feed_dict, frame, poses, statistics, training):
        pass

    def fully_connected(self, dim_p, prev_layer):
        # print ("Fully connected flattened layer: "+str(prev_layer))

        if self.settings.learn_goal and not self.settings.goal_gaussian:
            dim = np.prod(prev_layer.get_shape().as_list()[1:-1])
            # print ("Flattened size "+str(dim) +" shape "+str(prev_layer.get_shape()))
            self.flattened_layer = tf.reshape(prev_layer[:,:,:,0], [dim])
            self.flattened_layer_goal = tf.reshape(prev_layer[:, :, :, 1], [ dim])
        else:
            dim = np.prod(prev_layer.get_shape().as_list()[1:])
            self.flattened_layer=tf.reshape(prev_layer, [-1])
            if self.settings.learn_goal:

                weights = tf.get_variable('weights_goal', [dim, 2], self.DTYPE,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = self.bias_variable('biases_goal', [2])
                self.flattened_layer_goal = tf.matmul(tf.reshape(self.flattened_layer, [1,dim]), weights)
                self.flattened_layer_goal = tf.add(self.flattened_layer_goal, biases)

        if self.settings.learn_time and self.goal_net:
            weights_time = tf.get_variable('weights_time', [dim, 2], self.DTYPE,
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases_time = self.bias_variable('biases_time', [2])
            self.flattened_layer_time = tf.matmul(tf.reshape(self.flattened_layer, [1, dim]), weights_time)
            self.flattened_layer_time = tf.add(self.flattened_layer_time, biases_time)


    #return [statistics[ep_itr, agent_frame, 6]]
    def get_sample(self, statistics,ep_itr, agent_frame,  initialization_car):
        # print ("Get sample "+str(np.ravel_multi_index(([int(statistics[ep_itr, 0, 1])], [int(statistics[ep_itr, 0, 2])]), self.settings.env_shape[1:] )))
        return np.ravel_multi_index(([int(statistics[ep_itr, 0, 1])], [int(statistics[ep_itr, 0, 2])]), self.settings.env_shape[1:] )


    def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):
        return episode_in.reconstruction

    def calc_probabilities(self, fc_size):
        #print ("Define probabilities: " + str(self.flattened_layer))
        if self.settings.learn_goal :
            if not self.settings.goal_gaussian:
                self.probabilities_goal=tf.nn.softmax(self.flattened_layer_goal)
            else:
                self.probabilities_goal = tf.nn.sigmoid(self.flattened_layer_goal)

            if self.settings.learn_time and self.goal_net:
                self.probabilities_time = self.flattened_layer_time
                #print (self.probabilities_time.shape)
                self.alpha=tf.math.abs(self.probabilities_time[0,0])+1e-5
                self.beta = tf.nn.relu(self.probabilities_time[0,1])+1e-5

        self.probabilities = tf.nn.softmax(self.flattened_layer)

    def importance_sample_weight(self,responsible, statistics, ep_itr, frame, responsible_v=0):
        pass

    def get_feature_vectors(self, agent_frame, episode_in, feed_dict, frame):
        pass

    def apply_net(self, feed_dict, episode, frame, training, max_val=False, viz=False):
        # Choose car
        print(" Apply initializer")
        episode.init_method = 7
        if episode.useRealTimeEnv:
            print(" Use ReaL TIME ")
            episode.init_car_id = 0
            episode.init_car_pos = episode.car[0][1:]
            episode.init_car_vel = episode.car_dir[1:] * episode.frame_rate
            print ("Car pos " + str(episode.init_car_pos))
            print ("Car vel " + str(episode.init_car_vel)+" m/s "+str(episode.init_car_vel*.2))
            car_dims = episode.car_dims[1:]
            if abs(episode.init_car_vel[0]) > abs(episode.init_car_vel[1]):
                car_dims = [episode.car_dims[1], episode.car_dims[0]]
            car_max_dims = max(car_dims)
            car_min_dims = min(car_dims)
        else:
            episode.init_car_id= np.random.choice(episode.init_cars)
            episode.init_car_pos = np.array([np.mean(episode.cars_dict[episode.init_car_id][0][2:4]), np.mean(episode.cars_dict[episode.init_car_id][0][4:])])
            car_pos_next = np.array([np.mean(episode.cars_dict[episode.init_car_id][1][2:4]), np.mean(episode.cars_dict[episode.init_car_id][1][4:])])
            episode.init_car_vel = (car_pos_next -episode.init_car_pos) / episode.frame_time
            # print ("Car pos "+str(episode.init_car_pos))
            # print ("Car vel " + str(episode.init_car_vel))
            car_dims = [episode.cars_dict[episode.init_car_id][0][3] - episode.cars_dict[episode.init_car_id][0][2],
                        episode.cars_dict[episode.init_car_id][0][5] - episode.cars_dict[episode.init_car_id][0][4]]
            car_max_dims = max(car_dims)
            car_min_dims = min(car_dims)

        episode.calculate_prior(episode.init_car_pos, episode.init_car_vel, car_max_dims, car_min_dims)
        #episode.calculate_occlusions(episode.init_car_pos, episode.init_car_vel, car_max_dims, car_min_dims)

        flat_prior=episode.prior.flatten()
        feed_dict[self.prior]=np.expand_dims(np.expand_dims(episode.prior*(1/max(flat_prior)), axis=0), axis=-1)

        # print ("Feed dict")
        # for key, value in feed_dict.items():
        #     if key == self.state_in:
        #         print (value.shape)
        #
        #         print ("Input cars " + str(np.sum(value[0, :, :, 6])) + " people " + str(
        #             np.sum(value[0, :, :, 5])) + " Input cars traj" + str(
        #             np.sum(value[0, :, :, 4])) + " people traj " + str(
        #             np.sum(value[0, :, :, 3])))
        #         print ("Input building " + str(np.sum(value[0, :, :, 7])) + " fence " + str(
        #             np.sum(value[0, :, :, 7 + 1])) + " static " + str(
        #             np.sum(value[0, :, :, 7 + 2])) + " pole " + str(
        #             np.sum(value[0, :, :, 7 + 3])))
        #         print ("Input sidewalk " + str(np.sum(value[0, :, :, 7 + 5])) + " road " + str(
        #             np.sum(value[0, :, :, 7 + 4])) + " veg. " + str(
        #             np.sum(value[0, :, :, 7 + 6])) + " wall " + str(np.sum(value[0, :, :, 7 + 7])) + " sign " + str(
        #             np.sum(value[0, :, :, 7 + 8])))
        #
        #         # print ("Cars: ")
        #         # print (value[0, :, :, 6])
        #         # print ("Cars traj : ")
        #         # print (value[0, :, :, 4])
        #         # print ("People:")
        #         # print (value[0, :, :, 5])
        #         # print ("People traj : ")
        #         # print (value[0, :, :, 3])
        #     elif key == self.prior:
        #         print ("prior " + str(np.sum(value[0, :, :])))
        #
        #         # print ("Equal to feed dict ")
        #         # print (np.array_equal(value, self.feed_dict[self.prior]))
        #         # indxs=self.feed_dict[self.prior]-value
        #         # print ("nonzero  grad dict "+str(np.sum(value)))
        #         # print ("nonzero  feed dict "+str(np.sum(self.feed_dict[self.prior])))
        #     else:
        #         print (key)
        #         print (value)

        if not self.settings.evaluate_prior:
            if self.settings.learn_goal:
                if self.settings.learn_time:
                    probabilities, probabilities_goal, flattend_layer, conv_1,alpha, beta, summary_train = self.sess.run(
                        [self.probabilities, self.probabilities_goal, self.flattened_layer, self.conv1,self.alpha, self.beta,
                         self.train_summaries, ], feed_dict)
                    episode.goal_time=np.random.beta(alpha,beta)
                    # print ("Model outputs alpha "+str(alpha)+" beta "+str(beta)+" factor "+str(episode.goal_time))

                else:
                    probabilities,probabilities_goal, flattend_layer, conv_1, summary_train = self.sess.run(
                        [self.probabilities,self.probabilities_goal, self.flattened_layer, self.conv1, self.train_summaries], feed_dict)
                if self.settings.goal_gaussian:
                    # print ( "Gaussian model output: "+str(probabilities_goal))
                    episode.goal_distribution = np.copy(probabilities_goal)
                    # probabilities_goal[0][0]=probabilities_goal[0][0]
                    # probabilities_goal[0][1] = probabilities_goal[0][1] * self.settings.env_shape[2]
                    # print ("After scaling: " + str(probabilities_goal))
                else:
                    episode.goal_distribution = np.copy(probabilities_goal)
                    flat_prior_goal=episode.goal_prior.flatten()
                    probabilities_goal=probabilities_goal*flat_prior_goal
                    probabilities_goal = probabilities_goal * (1 / np.sum(probabilities_goal))

            else:
                probabilities,flattend_layer, conv_1,conv_2,pooled,pooled_2, summary_train = self.sess.run([self.probabilities,self.flattened_layer,self.conv1,self.conv2,self.pooled, self.pooled2, self.train_summaries], feed_dict)
        else:
            probabilities=np.ones_like(flat_prior)

        print (" Prior max pos " + str(
            np.unravel_index(np.where(flat_prior == np.max(flat_prior[:])), self.settings.env_shape[1:])))

        episode.init_distribution = np.copy(probabilities)
        # print (" Prior  " + str(np.unravel_index(np.where(flat_prior<0), self.settings.env_shape[1:])))
        probabilities = probabilities*flat_prior#episode.prior.flatten()
        probabilities=probabilities*(1/np.sum(probabilities))
        # print ("After multiplication with prior probabilities max pos " + str(
        #     np.unravel_index(np.where(probabilities == np.max(probabilities[:])), self.settings.env_shape[1:])))

        indx = np.random.choice(range(len(probabilities)), p=np.copy(probabilities))

        pos = np.unravel_index(indx, self.settings.env_shape[1:])
        episode.agent[0][0] = episode.get_height_init()
        episode.agent[0][1] = pos[0]
        episode.agent[0][2] = pos[1]
        # print ("Initialize pedestrian pos in 2D: " + str(episode.agent[0][1:])+" pos "+str(indx))

        # Vector from pedestrian to car in voxels
        vector_car_to_pedestrian = episode.agent[0][1:] - episode.init_car_pos

        # print ("Car velocity "+str(episode.init_car_vel)+ " vector car to pedestrian "+str(vector_car_to_pedestrian))
        # print ("Vector car to pedestrian "+str(vector_car_to_pedestrian))
        if np.linalg.norm(episode.init_car_vel) < 0.01:
            speed=3 * 5  # Speed voxels/second
            # print ("Desired speed pedestrian " + str(speed * .2) + " car vel " + str(episode.init_car_vel * .2))

            # Unit orthogonal direction
            unit = -vector_car_to_pedestrian * (
                1 / np.linalg.norm(vector_car_to_pedestrian))

            # Set ortogonal direction to car
            episode.vel_init = np.array([0, unit[0], unit[1]]) * speed * episode.frame_time  # set this correctly
            episode.speed_init = np.linalg.norm(episode.vel_init)
            if episode.follow_goal:
                episode.goal=episode.agent[0].copy()
                episode.goal[1:] = episode.init_car_pos
                # print (" car still Pedestrian goal init " + str(episode.goal))
                episode.manual_goal = episode.goal[1:].copy()
                if self.settings.learn_goal:
                    if self.settings.goal_gaussian:


                        # print ("Before adding random goal : " + str(episode.goal)+" model output "+str(probabilities_goal))
                        random_numbers = [np.random.normal(probabilities_goal[0][0], self.settings.goal_std, 1),np.random.normal(probabilities_goal[0][1], self.settings.goal_std, 1)]
                        # print ("Randon numbers " + str(random_numbers)+ " mean "+str(probabilities_goal[0][0]))
                        random_numbers[0] = random_numbers[0]#* self.settings.env_shape[1]
                        random_numbers[1] = random_numbers[1] #* self.settings.env_shape[2]
                        # print ("Before adding random goal : " + str(episode.goal) + " model output " + str(probabilities_goal))
                        # print ("Randon numbers after scaling " + str(random_numbers)+ "  scaled mean "+str(probabilities_goal[0][0]* self.settings.env_shape[1])+ "  scaled mean "+str(probabilities_goal[0][1]* self.settings.env_shape[2]))
                        # print ("Final mean " + str(
                        #     probabilities_goal[0][0] +episode.goal[1]) + "  scaled mean " + str(
                        #     probabilities_goal[0][1] +episode.goal[2]))

                        episode.goal[1]+=random_numbers[0]
                        episode.goal[2]+= random_numbers[1]
                        # print ("After addition: "+str(episode.goal))
                    else:
                        indx = np.random.choice(range(len(probabilities)), p=np.copy(probabilities_goal))
                        pos = np.unravel_index(indx, self.settings.env_shape[1:])
                        episode.goal[0] = episode.get_height_init()
                        episode.goal[1] = pos[0]
                        episode.goal[2] = pos[1]
                #episode.goal[2] += goal[1][0] * 256
                # print (" car still Pedestrian goal " + str(episode.goal))
            if self.settings.speed_input:
                if self.settings.learn_time:
                    episode.goal_time = episode.goal_time*3*5*episode.frame_time
                    # print ("Episode speed "+str( episode.goal_time)+" factor "+str(15*episode.frame_time)+" frametime "+str(episode.frame_time))
                    episode.goal_time = np.linalg.norm(episode.goal[1:] - episode.agent[0][1:])/episode.goal_time
                    # print ("Episode goal time " + str(episode.goal_time ))
                else:

                    episode.goal_time = min(np.linalg.norm(vector_car_to_pedestrian)/episode.speed_init, episode.seq_len-2)
                # print ("Car standing, Pedestrian initial speed: voxels / frame " + str(episode.speed_init) + " dist " + str(
                #     np.linalg.norm(vector_car_to_pedestrian)) + " timeframe " + str(
                #     episode.goal_time))
                # if self.settings.learn_goal:
                #     episode.goal_time = episode.seq_len - 2
            self.feed_dict.append({})
            for key, value in feed_dict.items():
                self.feed_dict[-1][key]=copy.deepcopy(value)

            #print (" prior "+str(np.sum(feed_dict[self.prior])))

            return episode.agent[0], 11, episode.vel_init  # pos, indx, vel_init
        # set initial pedestrian velocity orthogonal to car!- voxels
        # time to collision in seconds

        time_to_collision=np.dot(vector_car_to_pedestrian, episode.init_car_vel) / (np.linalg.norm(episode.init_car_vel)** 2 )
        vector_travelled_by_car_to_collision = time_to_collision* episode.init_car_vel
        #time to collision in frames:
        time_to_collision = time_to_collision*episode.frame_rate

        # print ("scalar product between car direction "+str(episode.init_car_vel)+" and vector car to pedestrian "+str(vector_car_to_pedestrian)+ " : "+ str(np.dot(vector_car_to_pedestrian, episode.init_car_vel)))
        # # #
        # print ("Car speed "+str(np.linalg.norm(episode.init_car_vel))+" time to collision car "+str(time_to_collision)+" in s "+str(time_to_collision/episode.frame_rate))
        vector_travelled_by_pedestrian_to_collision =  vector_travelled_by_car_to_collision-vector_car_to_pedestrian

        # print ("Vector to collision by car "+str(vector_travelled_by_car_to_collision)+" pos at collision "+str(vector_travelled_by_car_to_collision+episode.init_car_pos))
        # print ("Vector to collision by ped " + str(vector_travelled_by_pedestrian_to_collision) + " pos at collision " + str(
        #     vector_travelled_by_pedestrian_to_collision + episode.agent[0][1:]))

        # pedestrian speed in voxels per frame
        speed =np.linalg.norm(vector_travelled_by_pedestrian_to_collision)/time_to_collision
        if speed>0.01:

            # print ("Desired speed pedestrian " + str(speed ) + " dist to travel "+str(np.linalg.norm(vector_travelled_by_pedestrian_to_collision)) )

            speed=min(speed, 3*5*episode.frame_time)  # Speed voxels/second, max speed= 3m/s, 5 voxels per meter

            time_to_collision=np.linalg.norm(vector_travelled_by_pedestrian_to_collision)/speed

            # Unit orthogonal direction
            unit = vector_travelled_by_pedestrian_to_collision * (1 / np.linalg.norm(vector_travelled_by_pedestrian_to_collision))

            # Set ortogonal direction to car
            episode.vel_init = np.array([0, unit[0], unit[1]]) * speed  # set this correctly

            # print ("Pedestrian intercepting car? "+str(episode.intercept_car(0, all_frames=False)))
            episode.speed_init = np.linalg.norm(episode.vel_init)
        else:
            speed = 3 * 5  # Speed voxels/second

            # Set agent movement to car position
            unit = -vector_car_to_pedestrian * (
                1 / np.linalg.norm(vector_car_to_pedestrian))

            # Set ortogonal direction to car
            episode.vel_init = np.array(
                [0, unit[0], unit[1]]) * speed * episode.frame_time  # set this correctly
            episode.speed_init = np.linalg.norm(episode.vel_init)

        if episode.follow_goal:
            episode.goal=episode.agent[0]+(2*episode.vel_init*time_to_collision)
            if self.settings.learn_goal:
                episode.goal = episode.agent[0] + (episode.vel_init * time_to_collision)
            episode.manual_goal = episode.goal[1:].copy()
            # print ("Pedestrian goal init " + str(episode.goal))
            if self.settings.learn_goal:

                if self.settings.goal_gaussian:

                    # print (
                    # "Before adding random goal : " + str(episode.goal) + " model output " + str(probabilities_goal))
                    random_numbers = [np.random.normal(probabilities_goal[0][0], self.settings.goal_std, 1),
                                      np.random.normal(probabilities_goal[0][1], self.settings.goal_std, 1)]
                    # print ("Randon numbers " + str(random_numbers) + " mean " + str(probabilities_goal[0]))
                    random_numbers[0] = random_numbers[0]#"* self.settings.env_shape[1]
                    random_numbers[1] = random_numbers[1]# * self.settings.env_shape[2]
                    # print ("Init goal : " + str(episode.goal))
                    # print ("Randon numbers after scaling " + str(random_numbers) + "  scaled mean " + str(
                    #     probabilities_goal[0][0]) + "  scaled mean " + str(
                    #     probabilities_goal[0][1] ))
                    # print ("Final mean " + str(
                    #     probabilities_goal[0][0]  + episode.goal[
                    #         1]) + "  scaled mean " + str(
                    #     probabilities_goal[0][1]  + episode.goal[2]))

                    episode.goal[1] += random_numbers[0]
                    episode.goal[2] += random_numbers[1]
                else:
                    indx = np.random.choice(range(len(probabilities_goal)), p=np.copy(probabilities_goal))
                    pos = np.unravel_index(indx, self.settings.env_shape[1:])
                    episode.goal[0] = episode.get_height_init()
                    episode.goal[1] = pos[0]
                    episode.goal[2] = pos[1]
                    # print ("Goal pos "+str(indx)+" in pos "+str(pos))

            # print ("Pedestrian goal "+str(episode.goal))
        if self.settings.speed_input:
            if self.settings.learn_time:
                episode.goal_time = episode.goal_time * 3 * 5 * episode.frame_time
                # print ("Episode speed " + str(episode.goal_time) + " factor " + str(
                #     15 * episode.frame_time) + " frametime " + str(episode.frame_time))
                if episode.goal_time!=0:
                    episode.goal_time =  np.linalg.norm(episode.goal[1:] - episode.agent[0][1:])/episode.goal_time
                # print ("Episode goal time " + str(episode.goal_time)+" dist "+str(np.linalg.norm(episode.goal[1:] - episode.agent[0][1:])))
            else:
                episode.goal_time = min(time_to_collision*2,  episode.seq_len-2)
                if self.settings.learn_goal:
                    episode.goal_time =  episode.seq_len - 2
        #     print ("Pedestrian initial speed: voxels / frame "+str(episode.speed_init)+" dist "+str(np.linalg.norm(vector_travelled_by_pedestrian_to_collision) )+" timeframe "+str(episode.goal_time))
        # print ("Pedestrian vel " + str(episode.vel_init[1:])+" time to goal "+str(episode.goal_time)+" dist travelled "+str(episode.goal_time*episode.vel_init)+" final pos "+str(episode.agent[0]+episode.goal_time*episode.vel_init) )
        self.feed_dict.append({})
        for key, value in feed_dict.items():
            self.feed_dict[-1][key] = copy.deepcopy(value)
        #print (" prior " + str(np.sum(feed_dict[self.prior])))
        return episode.agent[0], 11,episode.vel_init #  pos, indx, vel_init


    def get_vel(self, episode_in, frame):
        pass

    def fully_connected_size(self, dim_p):
        return 0

    def construct_feed_dict(self, episode_in, frame, agent_frame,training=True,distracted=False):
        feed_dict ={}
        feed_dict[self.state_in] = self.get_input_init(episode_in)
        return feed_dict

    def get_input_init(self, episode_in):
        sem = np.zeros(( self.settings.env_shape[1], self.settings.env_shape[2], self.nbr_channels))
        segmentation = (episode_in.reconstruction_2D[ :, :, 3] * 33.0).astype(np.int)
        sem[:,:,0:3]=episode_in.reconstruction_2D[ :, :, 0:3].copy()

        if not episode_in.useRealTimeEnv:
            print(" Not using real time env")
            temp_people = episode_in.people_predicted[1].copy()
            temp_cars = episode_in.cars_predicted[1].copy()

        else:
            print (" People predicted init ")
            temp_people = episode_in.people_predicted_init.copy()
            temp_cars = episode_in.cars_predicted_init.copy()
        # print ("People predicted shape "+str(temp_people.shape)+" frame "+str(0)+ " sum "+str(np.sum(np.abs(temp_people))))
        # print ("Cars predicted shape " + str(temp_cars.shape) + " frame " + str(0) + " sum " + str(np.sum(np.abs(temp_cars))))
        temp_people[temp_people!=0]=1.0/temp_people[temp_people!=0]
        sem[:, :, 3]=temp_people.copy()#*self.temporal_scaling

        temp_cars[temp_cars != 0] = 1.0 / temp_cars[temp_cars != 0]
        sem[:, :, 4] = temp_cars.copy()

        # Not necessary any longer to have a separate channel!
        for person in episode_in.people[0]:
            person_index=np.round(person).astype(int)
            #print ("Pedestrian " + str([person[1][0], person[1][1], person[2][0], person[2][1]]))
            sem[person_index[1][0]:person_index[1][1]+1,person_index[2][0]:person_index[2][1]+1,5] = np.ones_like(sem[person_index[1][0]:person_index[1][1]+1, person_index[2][0]:person_index[2][1]+1,5])

        #print(" Episode cars "+str(episode_in.cars[0]))
        for car in episode_in.cars[0]:
            car_index = np.round(car).astype(int)
            sem[car_index[2]:car_index[3], car_index[4]:car_index[5],6] = np.ones_like(sem[car_index[2]:car_index[3], car_index[4]:car_index[5],6])


        # Do this faster somewhoe.
        for x in range(sem.shape[0]):
            for y in range(sem.shape[1]):
                if segmentation[ x, y]>0 and segmentation[ x, y]!=23:
                    if segmentation[ x, y] in self.labels_indx :
                        sem[x, y,self.labels_indx[segmentation[ x, y]] + 7] = 1
        return np.expand_dims( sem, axis=0)

    #construct_feed_dict(self, episode_in, frame, agent_frame, training=True):
    def grad_feed_dict(self, agent_action, agent_measures, agent_pos, agent_velocity, ep_itr, episode_in, frame,
                       reward, statistics,poses,priors,initialization_car, agent_speed=None, training=True, agent_frame=-1):
        if agent_frame < 0 or not training:
            agent_frame = frame
        r=reward[ep_itr, 0]
        #print  ("Reward "+str(r)+" rewards "+str(reward[ep_itr, :]))
        if sum(agent_measures[ep_itr, :agent_frame, 0])>0 or sum(agent_measures[ep_itr, :agent_frame, 13])>0:
            r=0
        # print ("Frame "+str(frame)+" Episode "+str(ep_itr)+" Reward "+str(r)+" hit by car:"+str(agent_measures[ep_itr,agent_frame, 0])+" Reached goal "+str(agent_measures[ep_itr,agent_frame, 13]))

        feed_dict = {self.state_in: self.get_input_init(episode_in),
                     self.advantages: r,
                     self.sample: self.get_sample(statistics, ep_itr, agent_frame,  initialization_car)}

        if self.settings.learn_goal and self.settings.goal_gaussian:
            feed_dict[self.goal]=self.get_goal(statistics, ep_itr, agent_frame, initialization_car)

        feed_dict[self.prior] = np.reshape(priors[ep_itr,:,0]*(1.0/max(priors[ep_itr,:,0])), (self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2],1))
        if self.settings.learn_goal and not self.settings.goal_gaussian:
            feed_dict[self.prior_goal]=np.reshape(priors[ep_itr,:,0]*(1.0/max(priors[ep_itr,:,2])), (self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2],1))

        if self.settings.learn_time and self.goal_net:
            goal = statistics[ep_itr, 3:5, 38 + NBR_MEASURES]
            agent_pos = statistics[ep_itr, 0,1:3]
            goal_dist =np.linalg.norm( goal - agent_pos)
            goal_time=statistics[ep_itr, 5, 38 + NBR_MEASURES]
            # episode.goal_time = episode.goal_time * 3 * 5 * episode.frame_time * np.linalg.norm(
            #     episode.goal[1:] - episode.agent[1:])
            # print("Agent position "+str(agent_pos)+" goal "+str(goal))
            # print ("Episode goal time " + str(goal_time)+" goal time "+str(goal_dist)+" fraction "+str(goal_dist/goal_time)+" ratio "+str(17/15))
            if goal_time==0:
                feed_dict[self.time_requirement] = np.array([[0]])
            else:
                feed_dict[self.time_requirement]=np.array([[goal_dist/goal_time*(17/15)]])
            # print ("Feed dict input " + str(feed_dict[self.time_requirement]))

        return feed_dict

    def evaluate(self, ep_itr, statistics, episode_in, poses, priors,initialization_car, statistics_car, seq_len=-1):
        if seq_len == -1:
            seq_len = self.settings.seq_len_test
        # print "Evaluate"
        agent_action, agent_measures, agent_pos, agent_reward, agent_reward_d, agent_velocity, agent_vel = self.stats(
            statistics, statistics_car)
        reward = agent_reward_d  # np.zeros_like(agent_reward)


        self.reset_mem()
        feed_dict = self.grad_feed_dict(agent_action, agent_measures, agent_pos, agent_velocity, ep_itr,
                                     episode_in, 0, reward, statistics,poses,priors,  initialization_car,agent_speed=agent_vel)
        loss, summaries_loss = self.sess.run([self.loss, self.loss_summaries], feed_dict)  # self.merged
        # episode_in.loss[frame] = loss
        #statistics[ep_itr, 0, STATISTICS_INDX.loss] = loss
        if self.writer:
            self.test_writer.add_summary(summaries_loss, global_step=self.num_grad_itrs)
        return statistics

    def train(self, ep_itr, statistics, episode_in, filename, filename_weights,poses, priors,initialization_car,statistics_car, seq_len=-1):

        agent_action, agent_measures, agent_pos, agent_reward, agent_reward_d, agent_velocity, agent_vel = self.stats(statistics,statistics_car)


        if self.settings.normalize:
            reward=self.normalize_reward(agent_reward_d, agent_measures)
        else:
            reward=agent_reward_d

            #print "Gradient: "+str(seq_len-1)

        self.reset_mem()


        feed_dict = self.grad_feed_dict(agent_action, agent_measures, agent_pos, agent_velocity, ep_itr,
                                         episode_in, 0, reward, statistics,poses,priors,  initialization_car,agent_speed=agent_vel)

        # print ("Grad Feed dict---------------------------------------------------------")
        # print(" Feed dict len " + str(len(self.feed_dict)))
        # for key, value in feed_dict.items():
        #     if key == self.state_in:
        #         print (value.shape)
        #
        #         print ("Input cars " + str(np.sum(value[0, :, :, 6])) + " people " + str(
        #             np.sum(value[0, :, :, 5])) + " Input cars traj" + str(
        #             np.sum(value[0, :, :, 4])) + " people traj " + str(
        #             np.sum(value[0, :, :, 3])))
        #         print ("Input building " + str(np.sum(value[0, :, :, 7])) + " fence " + str(
        #             np.sum(value[0, :, :, 7 + 1])) + " static " + str(
        #             np.sum(value[0, :, :, 7 + 2])) + " pole " + str(
        #             np.sum(value[0, :, :, 7 + 3])))
        #         print ("Input sidewalk " + str(np.sum(value[0, :, :, 7 + 5])) + " road " + str(
        #             np.sum(value[0, :, :, 7 + 4])) + " veg. " + str(
        #             np.sum(value[0, :, :, 7 + 6])) + " wall " + str(np.sum(value[0, :, :, 7 + 7])) + " sign " + str(
        #             np.sum(value[0, :, :, 7 + 8])))
        #         equal=np.array_equal(value, self.feed_dict[ep_itr][self.state_in])
        #         print ("Equal to feed dict state "+str(equal))
        #         if not equal:
        #             diff=self.feed_dict[ep_itr][self.state_in]-value
        #             not_equal_pos=np.where(diff!=0)
        #
        #             print ("Not equal: "+str(np.sum(self.feed_dict[ep_itr][self.state_in]-value)))
        #             print (" Size "+str(len(not_equal_pos))+" diff size "+str(diff.shape))
        #             print (" Not equal x " + str(np.unique(not_equal_pos[1])))
        #             print (" Not equal y " + str(np.unique(not_equal_pos[2])))
        #             print ( "Not equal at "+str(np.unique(not_equal_pos[3])))
        #             print(" Value in orig feed dict "+str(self.feed_dict[ep_itr][self.state_in][not_equal_pos]))
        #             print(" Value in grad feed dict " + str(value[not_equal_pos]))
        #
        #         # print ("Cars: ")
        #         # print (value[0, :, :, 6])
        #         # print ("Cars traj : ")
        #         # print (value[0, :, :, 4])
        #         # print ("People:")
        #         # print (value[0, :, :, 5])
        #         # print ("People traj : ")
        #         # print (value[0, :, :, 3])
        #     elif key == self.prior:
        #         print ("prior " + str(np.sum(value[0, :, :])))
        #         print ("Equal to feed dict "+str(np.array_equal(value, self.feed_dict[ep_itr][self.prior])))
        #         # print ("Equal to feed dict ")
        #         # print (np.array_equal(value, self.feed_dict[self.prior]))
        #         # indxs=self.feed_dict[self.prior]-value
        #         # print ("nonzero  grad dict "+str(np.sum(value)))
        #         # print ("nonzero  feed dict "+str(np.sum(self.feed_dict[self.prior])))
        #     else:
        #         print (str(key)+" "+str( value))
        #         if key in self.feed_dict[ep_itr]:
        #             print ("Equal to feed dict "+str(np.array_equal(value, self.feed_dict[ep_itr][key])))

        self.calculate_gradients( feed_dict, statistics, 0, ep_itr) #                 self.calculate_gradients(episode_in, frame, feed_dict) %

        if ep_itr==statistics.shape[0]-1:
            if not self.settings.overfit or( self.settings.overfit and self.num_grad_itrs%20==0):
                with open(filename, 'wb') as f:
                    pickle.dump(self.gradBuffer, f, pickle.HIGHEST_PROTOCOL)


                [weights]=self.sess.run([self.tvars])
                with open(filename_weights, 'wb') as f:
                    pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)
            else:
                [weights] = self.sess.run([self.tvars])
                # print "Weights "
                # print weights
            #if self.num_grad_itrs % (self.update_frequency) == 0 and self.num_grad_itrs != 0:
            self.update_gradients()
            self.feed_dict=[]
            #print(" Feed dict len "+str(self.feed_dict))

        return statistics

class InitializerGaussianNet(InitializerNet):
    def __init__(self, settings, weights_name="init") :
        super(InitializerGaussianNet, self).__init__(settings, weights_name=weights_name)


    def define_loss(self, dim_p):
        self.sample = tf.compat.v1.placeholder(shape=[2], dtype=self.DTYPE, name="sample")
        self.normal_dist = tf.contrib.distributions.Normal(self.probabilities, self.settings.init_std)
        self.responsible_output = self.normal_dist.prob(self.sample)


        self.l2_loss = tf.nn.l2_loss(self.probabilities - self.sample)  # tf.nn.l2_loss
        #self.loss = tf.reduce_mean(self.advantages * (2 * tf.log(self.settings.sigma_vel) + tf.log(2 * tf.pi) + (
        #self.l2_loss / (self.settings.sigma_vel * self.settings.sigma_vel))))
        self.loss = tf.reduce_mean(self.advantages*self.l2_loss)

        return self.sample, self.loss

    def get_feature_vectors_gradient(self, agent_action, agent_frame, agent_measures, agent_pos, agent_speed, ep_itr,
                                     episode_in, feed_dict, frame, poses, statistics, training):
        pass

    def fully_connected(self, dim_p, prev_layer):
        # print ("Fully connected flattened layer: "+str(prev_layer))
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        self.flattened_layer_conv = tf.reshape(prev_layer, [-1])
        weights = tf.get_variable('weights_goal', [dim, 2], self.DTYPE,
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = self.bias_variable('biases_goal', [2])
        self.flattened_layer = tf.matmul(tf.reshape(self.flattened_layer_conv, [1, dim]), weights)
        self.flattened_layer = tf.add(self.flattened_layer, biases)



    #return [statistics[ep_itr, agent_frame, 6]]
    def get_sample(self, statistics,ep_itr, agent_frame, init_car):
        #print ([statistics[ep_itr, 0, 1]], [statistics[ep_itr, 0, 2]])
        #print ("Get Sample "+str(statistics[ep_itr, 0, 1]*self.settings.env_shape[1]+statistics[ep_itr, 0, 2]))
        # np.ravel_multi_index(([statistics[ep_itr, 0, 1]], [statistics[ep_itr, 0, 2]]), self.settings.env_shape[1:] )
        # print ("Get Sample "+str([statistics[ep_itr, 0, 1:2]])+" flattened: "+str(np.ravel_multi_index(([int(statistics[ep_itr, 0, 1])], [int(statistics[ep_itr, 0, 2])]), self.settings.env_shape[1:] )))
        sample=statistics[ep_itr, 0,  1:3]-init_car[ep_itr,1:3]
        # sample[0]=sample[0]/self.settings.env_shape[1]
        # sample[1] = sample[1] / self.settings.env_shape[2]
        return sample
        #[statistics[ep_itr, 0, 1]*self.settings.env_shape[1]+statistics[ep_itr, 0, 2]]

    # def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):
    #     return episode_in.reconstruction

    def calc_probabilities(self, fc_size):
        #print ("Define probabilities: " + str(self.flattened_layer))
        self.probabilities =self.flattened_layer#tf.sigmoid(self.mu)

    def importance_sample_weight(self,responsible, statistics, ep_itr, frame, responsible_v=0):
        pass

    def get_feature_vectors(self, agent_frame, episode_in, feed_dict, frame):
        pass

    def apply_net(self, feed_dict, episode, frame, training, max_val=False, viz=False):
        # Choose car
        episode.init_car_id= np.random.choice(episode.init_cars)
        episode.init_car_pos = np.array([np.mean(episode.cars_dict[episode.init_car_id][0][2:4]), np.mean(episode.cars_dict[episode.init_car_id][0][4:])])
        car_pos_next = np.array([np.mean(episode.cars_dict[episode.init_car_id][1][2:4]), np.mean(episode.cars_dict[episode.init_car_id][1][4:])])
        episode.init_car_vel = (car_pos_next -episode.init_car_pos) / episode.frame_time



        episode.init_method=7

        car_dims = [episode.cars_dict[episode.init_car_id][0][3] - episode.cars_dict[episode.init_car_id][0][2],
                    episode.cars_dict[episode.init_car_id][0][5] - episode.cars_dict[episode.init_car_id][0][4]]
        car_max_dims = max(car_dims)
        car_min_dims = min(car_dims)

        episode.calculate_prior(episode.init_car_pos, episode.init_car_vel, car_max_dims, car_min_dims)


        flat_prior=episode.prior.flatten()
        feed_dict[self.prior]=np.expand_dims(np.expand_dims(episode.prior*(1/max(flat_prior)), axis=0), axis=-1)



        mean_vel, summary_train = self.sess.run([self.mu, self.train_summaries], feed_dict)
        # print ("Mean vel "+str(mean_vel))
        episode.init_distribution=copy.copy(mean_vel[0])

        episode.agent[0]=np.zeros(3)
        episode.agent[0][0] = episode.get_height_init()
        episode.agent[0][1] = episode.init_car_pos[0]+ np.random.normal(mean_vel[0][0], self.settings.init_std, 1)
        episode.agent[0][2] = episode.init_car_pos[1]+np.random.normal(mean_vel[0][1], self.settings.init_std, 1)
        episode.agent[0][1] = max(min(episode.agent[0][1], self.settings.env_shape[1] - 1), 0)
        episode.agent[0][2] = max(min(episode.agent[0][2], self.settings.env_shape[2] - 1), 0)

        while(episode.prior[int(episode.agent[0][1]),int(episode.agent[0][1])]==0):
            episode.agent[0][1] = episode.init_car_pos[0] + np.random.normal(mean_vel[0][0], self.settings.init_std, 1)
            episode.agent[0][2] = episode.init_car_pos[1] + np.random.normal(mean_vel[0][1], self.settings.init_std, 1)
            episode.agent[0][1] = max(min(episode.agent[0][1], self.settings.env_shape[1] - 1), 0)
            episode.agent[0][2] = max(min(episode.agent[0][2], self.settings.env_shape[2] - 1), 0)
        # make sure agent is in environment borders
        episode.agent[0][1]= max(min(episode.agent[0][1], self.settings.env_shape[1]-1),0)
        episode.agent[0][2] = max(min(episode.agent[0][2], self.settings.env_shape[2] - 1), 0)

        # print ("Initialize pedestrian "  + str( indx) + " pos in 2D: " + str(episode.agent[0][1:])+" no prior probability: "+str( episode.init_distribution[indx])+" prior probability"+str(probabilities[[indx]]))

        # Vector from pedestrian to car in voxels
        vector_car_to_pedestrian = episode.agent[0][1:] - episode.init_car_pos
        if np.linalg.norm(episode.init_car_vel) < 0.01:
            speed=3 * 5  # Speed voxels/second
            # print ("Desired speed pedestrian " + str(speed * .2) + " car vel " + str(episode.init_car_vel * .2))

            # Unit orthogonal direction
            unit = -vector_car_to_pedestrian * (
                1 / np.linalg.norm(vector_car_to_pedestrian))

            # Set ortogonal direction to car
            episode.vel_init = np.array([0, unit[0], unit[1]]) * speed * episode.frame_time  # set this correctly
            return episode.agent[0], 11, episode.vel_init  # pos, indx, vel_init
        # set initial pedestrian velocity orthogonal to car!- voxels
        vector_travelled_by_car_to_collision = np.dot(vector_car_to_pedestrian, episode.init_car_vel) / np.linalg.norm(
            episode.init_car_vel) * episode.init_car_vel

        vector_travelled_by_pedestrian_to_collision = vector_car_to_pedestrian - vector_travelled_by_car_to_collision

        ratio_pedestrian_to_car_dist_to_collision = np.linalg.norm(
            vector_travelled_by_pedestrian_to_collision) / np.linalg.norm(vector_travelled_by_car_to_collision)

        speed = min(ratio_pedestrian_to_car_dist_to_collision * np.linalg.norm(episode.init_car_vel), 3*5)  # Speed voxels/second
        # print ("Desired speed pedestrian "+str(speed*.2)+" ratio_pedestrian_to_car_dist_to_collision "+str(ratio_pedestrian_to_car_dist_to_collision)+" "+str(episode.init_car_vel*.2))

        # Unit orthogonal direction
        unit = -vector_travelled_by_pedestrian_to_collision * (
        1 / np.linalg.norm(vector_travelled_by_pedestrian_to_collision))

        # Set ortogonal direction to car
        episode.vel_init = np.array([0, unit[0], unit[1]]) * speed * episode.frame_time  # set this correctly

        # print ("Pedestrian intercepting car? "+str(episode.intercept_car(0, all_frames=False)))

        return episode.agent[0], 11,episode.vel_init #  pos, indx, vel_init




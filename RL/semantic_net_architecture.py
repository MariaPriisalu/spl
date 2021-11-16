import numpy as np
from settings import NBR_MEASURES,RANDOM_SEED

from initializer_net import InitializerNet

import tensorflow as tf
tf.compat.v1.set_random_seed(RANDOM_SEED)


import copy


#
# np.set_printoptions(precision=5)




class InitializerArchNet(InitializerNet):
    # Softmax Simplified
    def __init__(self, settings, weights_name="policy"):
        self.labels_indx = {11: 0, 13: 1, 14: 2, 4: 2, 5: 2, 15: 2, 16: 2, 17: 3, 18: 3, 7: 4, 9: 4, 6: 4, 10: 4,
                            8: 5, 21: 6, 22: 6, 12: 7, 20: 8, 19: 8}
        self.feed_dict = {}
        self.valid_pos = []
        self.probabilities_saved = []
        self.carla = settings.carla
        self.set_nbr_channels()
        # self.temporal_scaling = 0.1 * 0.3

        super(InitializerArchNet, self).__init__(settings, weights_name="init")

    # def get_goal(self, statistics, ep_itr, agent_frame, initialization_car):
    #     # print ("Get goal "+str(statistics[ep_itr, 3:5, 38 + NBR_MEASURES].astype(int))+" size "+str(self.settings.env_shape[1:]) )
    #     if self.settings.goal_gaussian:
    #         goal = statistics[ep_itr, 3:5, 38 + NBR_MEASURES]
    #         agent_pos = initialization_car[ep_itr, 5:]
    #         goal_dir = goal - agent_pos
    #         goal_dir[0] = goal_dir[0] / self.settings.env_shape[1]
    #         goal_dir[1] = goal_dir[1] / self.settings.env_shape[2]
    #         print ("Get goal " + str(goal_dir))
    #         return goal_dir
    #     else:
    #         print ("Get goal " + str(
    #             np.ravel_multi_index(statistics[ep_itr, 3:5, 38 + NBR_MEASURES].astype(int),
    #                                  self.settings.env_shape[1:])))
    #         return [np.ravel_multi_index(statistics[ep_itr, 3:5, 38 + NBR_MEASURES].astype(int),
    #                                      self.settings.env_shape[1:])]
    #
    # def set_nbr_channels(self):
    #     self.nbr_channels = 9 + 7

    def define_conv_layers(self):
        # tf.reset_default_graph()
        output_channels = 1

        self.state_in = tf.compat.v1.placeholder(
            shape=[self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2], self.nbr_channels],
            dtype=self.DTYPE,
            name="reconstruction")
        self.prior = tf.compat.v1.placeholder(
            shape=[self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2], 1],
            dtype=self.DTYPE,
            name="prior")

        # mean = tf.constant(self.mean, dtype=self.DTYPE)  # (3)

        prev_layer = tf.concat([self.state_in, self.prior], axis=3)  # - mean
        out_channels = (self.nbr_channels + 1)
        with tf.compat.v1.variable_scope('conv1_1') as scope:
            out_channels = (self.nbr_channels + 1) * 2

            print(
            "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel1 = tf.get_variable('weights', [3, 3, self.nbr_channels + 1, out_channels], self.DTYPE,
                                      initializer=tf.contrib.layers.xavier_initializer())
            prev_layer = tf.nn.conv2d(prev_layer, kernel1, strides=[1, 2, 2, 1], padding='SAME')  # [1, 2, 2, 2, 1]

        with tf.compat.v1.variable_scope('conv2_1') as scope:
            print(
            "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel1_2 = tf.get_variable('weights', [3, 3, out_channels, out_channels], self.DTYPE,
                                        initializer=tf.contrib.layers.xavier_initializer())
            prev_layer = tf.nn.conv2d(prev_layer, kernel1_2, padding='SAME')  # [1, 2, 2, 2, 1]

        with tf.compat.v1.variable_scope('conv3_1') as scope:
            print(
            "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel1_3 = tf.get_variable('weights', [3, 3, out_channels, out_channels], self.DTYPE,
                                        initializer=tf.contrib.layers.xavier_initializer())
            prev_layer = tf.nn.conv2d(prev_layer, kernel1_3, padding='SAME')  # [1, 2, 2, 2, 1]
        self.block1=copy.copy(prev_layer)
        self.out_channels_1 = copy.copy(out_channels)

        # Second Block
        with tf.compat.v1.variable_scope('conv1_2') as scope:
            out_channels = out_channels * 2

            print(
                "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel2 = tf.get_variable('weights', [3, 3, self.nbr_channels + 1, out_channels], self.DTYPE,
                                      initializer=tf.contrib.layers.xavier_initializer())
            prev_layer = tf.nn.conv2d(prev_layer, kernel2, strides=[1, 2, 2, 1], padding='SAME')  # [1, 2, 2, 2, 1]

        with tf.compat.v1.variable_scope('conv2_2') as scope:
            print(
                "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel2_2 = tf.get_variable('weights', [3, 3, out_channels, out_channels], self.DTYPE,
                                        initializer=tf.contrib.layers.xavier_initializer())
            prev_layer = tf.nn.conv2d(prev_layer, kernel2_2, padding='SAME')  # [1, 2, 2, 2, 1]

        with tf.compat.v1.variable_scope('conv3_2') as scope:
            print(
                "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel2_3 = tf.get_variable('weights', [3, 3, out_channels, out_channels], self.DTYPE,
                                        initializer=tf.contrib.layers.xavier_initializer())
            prev_layer = tf.nn.conv2d(prev_layer, kernel2_3, padding='SAME')  # [1, 2, 2, 2, 1]
        self.block2 = copy.copy(prev_layer)
        self.out_channels_2=copy.copy(out_channels)

        # Third Block
        with tf.compat.v1.variable_scope('conv1_3') as scope:
            out_channels = out_channels * 2

            print(
                "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel3 = tf.get_variable('weights', [3, 3, self.nbr_channels + 1, out_channels], self.DTYPE,
                                      initializer=tf.contrib.layers.xavier_initializer())
            prev_layer = tf.nn.conv2d(prev_layer, kernel3, strides=[1, 2, 2, 1], padding='SAME')  # [1, 2, 2, 2, 1]

        with tf.compat.v1.variable_scope('conv2_3') as scope:
            print(
                "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel3_2 = tf.get_variable('weights', [3, 3, out_channels, out_channels], self.DTYPE,
                                        initializer=tf.contrib.layers.xavier_initializer())
            prev_layer = tf.nn.conv2d(prev_layer, kernel3_2, padding='SAME')  # [1, 2, 2, 2, 1]

        with tf.compat.v1.variable_scope('conv3_3') as scope:
            print(
                "Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel3_3 = tf.get_variable('weights', [3, 3, out_channels, out_channels], self.DTYPE,
                                        initializer=tf.contrib.layers.xavier_initializer())
            prev_layer = tf.nn.conv2d(prev_layer, kernel3_3, padding='SAME')  # [1, 2, 2, 2, 1]
        self.block3 = copy.copy(prev_layer)
        self.out_channels_3 = copy.copy(out_channels)


        self.block1_resized=tf.image.resize_images(self.block1, [self.settings.env_shape[1], self.settings.env_shape[2]])

        print ("Block 1 before resizing "+str(self.block1.shape)+" block 1 after resizing "+str(self.block1_resized.shape))

        self.block2_resized = tf.image.resize_images(self.block2,
                                                     [self.settings.env_shape[1], self.settings.env_shape[2]])

        print ("Block 2 before resizing " + str(self.block2.shape) + " block 2 after resizing " + str(self.block2_resized.shape))

        self.block3_resized = tf.image.resize_images(self.block3,
                                                     [self.settings.env_shape[1], self.settings.env_shape[2]])

        print ("Block 3 before resizing " + str(self.block3.shape) + " block 3 after resizing " + str(self.block3_resized.shape))

        self.out_channels=tf.concat([self.block1_resized, self.block2_resized, self.block3_resized], axis=3)

        print ("Size after concatenation " + str(self.out_channels.shape))
        out_channels=self.out_channels_1+self.out_channels_2+self.out_channels_3
        with tf.compat.v1.variable_scope('out_conv') as scope:
            print("Define 2D weights")  # self.settings.net_size, self.settings.net_size# [ self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels+1, out_filters]
            kernel_out = tf.get_variable('weights', [1, 1, out_channels, 1], self.DTYPE,
                                        initializer=tf.contrib.layers.xavier_initializer())
            prev_layer = tf.nn.conv2d(self.out_channels, kernel_out, padding='SAME')  # [1, 2, 2, 2, 1]

        print ("Size after 1-conv " + str(prev_layer.shape))
        return prev_layer

    # def define_loss(self, dim_p):
    #
    #     self.sample = tf.compat.v1.placeholder(shape=[1], dtype=tf.int32, name="sample")
    #     self.responsible_output = tf.slice(self.probabilities, self.sample, [1]) + np.finfo(
    #         np.float32).eps
    #     if self.settings.learn_goal and not self.settings.goal_gaussian:
    #         self.prior_goal = tf.compat.v1.placeholder(
    #             shape=[self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2], 1],
    #             dtype=self.DTYPE,
    #             name="prior_goal")
    #         self.prior_flat_goal = tf.reshape(self.prior_goal, [
    #             self.settings.batch_size * self.settings.env_shape[1] * self.settings.env_shape[2]])
    #         self.goal = tf.compat.v1.placeholder(shape=[1], dtype=tf.int32, name="goal")
    #         self.responsible_output_goal = tf.slice(self.probabilities_goal, self.goal, [1]) + np.finfo(
    #             np.float32).eps
    #         self.responsible_output_prior_goal = tf.slice(self.prior_flat_goal, self.goal, [1]) + np.finfo(
    #             np.float32).eps
    #     self.prior_flat = tf.reshape(self.prior, [
    #         self.settings.batch_size * self.settings.env_shape[1] * self.settings.env_shape[2]])
    #     self.responsible_output_prior = tf.slice(self.prior_flat, self.sample, [1]) + np.finfo(
    #         np.float32).eps
    #     self.distribution = self.prior_flat * self.probabilities
    #
    #     self.loss = -tf.reduce_mean(
    #         (tf.log(self.responsible_output) + tf.log(self.responsible_output_prior)) * self.advantages)
    #     if self.settings.learn_goal and not self.settings.goal_gaussian:
    #         self.loss -= tf.reduce_mean(
    #             (tf.log(self.responsible_output_goal) + tf.log(self.responsible_output_prior_goal)) * self.advantages)
    #
    #     if self.settings.entr_par_init:  # To do: add entropy for goal!
    #         y_zeros = tf.zeros_like(self.distribution)
    #         y_mask = tf.math.greater(self.distribution, y_zeros)
    #         res = tf.boolean_mask(self.distribution, y_mask)
    #         logres = tf.math.log(res)
    #
    #         self.entropy = tf.nn.softmax_cross_entropy_with_logits(labels=res, logits=logres)
    #         self.loss = -tf.reduce_mean(self.settings.entr_par_init * self.entropy)
    #
    #     if self.settings.learn_goal and self.settings.goal_gaussian:
    #         self.goal = tf.compat.v1.placeholder(shape=[2], dtype=self.DTYPE, name="goal")
    #         self.normal_dist = tf.contrib.distributions.Normal(self.probabilities_goal, self.settings.goal_std)
    #         self.responsible_output = self.normal_dist.prob(self.goal)
    #
    #         self.l2_loss = tf.nn.l2_loss(self.probabilities_goal - self.goal)  # tf.nn.l2_loss
    #         # self.loss = tf.reduce_mean(self.advantages * (2 * tf.log(self.settings.sigma_vel) + tf.log(2 * tf.pi) + (
    #         # self.l2_loss / (self.settings.sigma_vel * self.settings.sigma_vel))))
    #         self.loss = self.loss + tf.reduce_mean(self.advantages * self.l2_loss)
    #
    #     return self.sample, self.loss
    #
    # def get_feature_vectors_gradient(self, agent_action, agent_frame, agent_measures, agent_pos, agent_speed, ep_itr,
    #                                  episode_in, feed_dict, frame, poses, statistics, training):
    #     pass
    #
    # def fully_connected(self, dim_p, prev_layer):
    #     print ("Fully connected flattened layer: " + str(prev_layer))
    #
    #     if self.settings.learn_goal and not self.settings.goal_gaussian:
    #         dim = np.prod(prev_layer.get_shape().as_list()[1:-1])
    #         print ("Flattened size " + str(dim) + " shape " + str(prev_layer.get_shape()))
    #         self.flattened_layer = tf.reshape(prev_layer[:, :, :, 0], [dim])
    #         self.flattened_layer_goal = tf.reshape(prev_layer[:, :, :, 1], [dim])
    #     else:
    #         dim = np.prod(prev_layer.get_shape().as_list()[1:])
    #         self.flattened_layer = tf.reshape(prev_layer, [-1])
    #         if self.settings.learn_goal:
    #             weights = tf.get_variable('weights_goal', [dim, 2], self.DTYPE,
    #                                       initializer=tf.contrib.layers.xavier_initializer())
    #             biases = self.bias_variable('biases_goal', [2])
    #             self.flattened_layer_goal = tf.matmul(tf.reshape(self.flattened_layer, [1, dim]), weights)
    #             self.flattened_layer_goal = tf.add(self.flattened_layer_goal, biases)
    #
    # # return [statistics[ep_itr, agent_frame, 6]]
    # def get_sample(self, statistics, ep_itr, agent_frame, initialization_car):
    #     print ("Get sample " + str(
    #         np.ravel_multi_index(([int(statistics[ep_itr, 0, 1])], [int(statistics[ep_itr, 0, 2])]),
    #                              self.settings.env_shape[1:])))
    #     return np.ravel_multi_index(([int(statistics[ep_itr, 0, 1])], [int(statistics[ep_itr, 0, 2])]),
    #                                 self.settings.env_shape[1:])
    #
    # def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):
    #     return episode_in.reconstruction
    #
    # def calc_probabilities(self, fc_size):
    #     # print ("Define probabilities: " + str(self.flattened_layer))
    #     if self.settings.learn_goal:
    #         if not self.settings.goal_gaussian:
    #             self.probabilities_goal = tf.nn.softmax(self.flattened_layer_goal)
    #         else:
    #             self.probabilities_goal = tf.nn.sigmoid(self.flattened_layer_goal)
    #
    #     self.probabilities = tf.nn.softmax(self.flattened_layer)
    #
    # def importance_sample_weight(self, responsible, statistics, ep_itr, frame, responsible_v=0):
    #     pass
    #
    # def get_feature_vectors(self, agent_frame, episode_in, feed_dict, frame):
    #     pass
    #
    # def apply_net(self, feed_dict, episode, frame, training, max_val=False, viz=False):
    #     # Choose car
    #     episode.init_car_id = np.random.choice(episode.init_cars)
    #     episode.init_car_pos = np.array([np.mean(episode.cars_dict[episode.init_car_id][0][2:4]),
    #                                      np.mean(episode.cars_dict[episode.init_car_id][0][4:])])
    #     car_pos_next = np.array([np.mean(episode.cars_dict[episode.init_car_id][1][2:4]),
    #                              np.mean(episode.cars_dict[episode.init_car_id][1][4:])])
    #     episode.init_car_vel = (car_pos_next - episode.init_car_pos) / episode.frame_time
    #     print ("Car pos " + str(episode.init_car_pos))
    #     print ("Car vel " + str(episode.init_car_vel))
    #
    #     episode.init_method = 7
    #
    #     car_dims = [episode.cars_dict[episode.init_car_id][0][3] - episode.cars_dict[episode.init_car_id][0][2],
    #                 episode.cars_dict[episode.init_car_id][0][5] - episode.cars_dict[episode.init_car_id][0][4]]
    #     car_max_dims = max(car_dims)
    #     car_min_dims = min(car_dims)
    #
    #     episode.calculate_prior(episode.init_car_pos, episode.init_car_vel, car_max_dims, car_min_dims)
    #
    #     flat_prior = episode.prior.flatten()
    #     feed_dict[self.prior] = np.expand_dims(np.expand_dims(episode.prior * (1 / max(flat_prior)), axis=0), axis=-1)
    #
    #     if self.settings.learn_goal:
    #
    #         probabilities, probabilities_goal, flattend_layer, conv_1, summary_train = self.sess.run(
    #             [self.probabilities, self.probabilities_goal, self.flattened_layer, self.conv1, self.train_summaries],
    #             feed_dict)
    #         if self.settings.goal_gaussian:
    #             # print ( "Gaussian model output: "+str(probabilities_goal))
    #             episode.goal_distribution = np.copy(probabilities_goal)
    #             # probabilities_goal[0][0]=probabilities_goal[0][0]
    #             # probabilities_goal[0][1] = probabilities_goal[0][1] * self.settings.env_shape[2]
    #             # print ("After scaling: " + str(probabilities_goal))
    #         else:
    #             episode.goal_distribution = np.copy(probabilities_goal)
    #             flat_prior_goal = episode.goal_prior.flatten()
    #             probabilities_goal = probabilities_goal * flat_prior_goal
    #             probabilities_goal = probabilities_goal * (1 / np.sum(probabilities_goal))
    #
    #     else:
    #         probabilities, flattend_layer, conv_1, summary_train = self.sess.run(
    #             [self.probabilities, self.flattened_layer, self.conv1, self.train_summaries], feed_dict)
    #
    #     episode.init_distribution = np.copy(probabilities)
    #
    #     pos_max = np.argmax(episode.init_distribution)
    #     pos_min = np.argmin(episode.init_distribution)
    #
    #     # print (" max value "+str(episode.init_distribution[pos_max])+"  min value "+str(episode.init_distribution[pos_min]) )
    #
    #     reshaped_init = np.reshape(episode.init_distribution, episode.prior.shape)
    #
    #     pos_max = np.argmax(flat_prior)
    #     pos_min = np.argmin(flat_prior)
    #
    #     probabilities = probabilities * flat_prior  # episode.prior.flatten()
    #     probabilities_reshaped = np.reshape(probabilities, episode.prior.shape)
    #
    #     pos_max = np.argmax(probabilities)
    #     pos_min = np.argmin(probabilities)
    #
    #     # for car in episode.cars[0]:
    #     #     print ("Probabilities reshaped " + str(car))
    #     #     print (np.sum(np.abs(probabilities_reshaped[car[2]:car[3], car[4]:car[5]])))
    #
    #     pos_max = np.argmax(probabilities)
    #     # print ("After prior Init distr " + str(np.sum(np.abs(probabilities))) + "  max: " + str(
    #     #     pos_max) + " pos in 2D: " + str(np.unravel_index(pos_max, self.settings.env_shape[1:])))
    #
    #     probabilities = probabilities * (1 / np.sum(probabilities))
    #
    #     # print ("Final distr " + str(np.sum(np.abs(probabilities))) + "  max: " + str(
    #     #     pos_max) + " pos in 2D: " + str(np.unravel_index(pos_max, self.settings.env_shape[1:])))
    #     # print ("Final distr  min pos " + str(pos_min) + " pos in 2D: " + str(
    #     #     np.unravel_index(pos_min, self.settings.env_shape[1:])))
    #     # print (" max value " + str(probabilities[pos_max]) + "  min value " + str(probabilities[pos_min]))
    #     #
    #     # print ("10th row:" + str(probabilities_reshaped[10, :10]))
    #
    #     pos_max = np.argmax(probabilities)
    #     # if max_val and not training:
    #     #     # print ("Maximal evaluation init net")
    #     #     pos = np.unravel_index(pos_max, self.settings.env_shape[1:])
    #     #     episode.agent[0][0] = episode.get_height_init()
    #     #     episode.agent[0][1] = pos[0]
    #     #     episode.agent[0][2] = pos[1]
    #     # else:
    #     # print ("After normalization " + str(np.sum(np.abs(probabilities))) + "  max: " + str(
    #     #     pos_max) + " pos in 2D: " + str(np.unravel_index(pos_max, self.settings.env_shape[1:]))+" no prior max "+str(episode.init_distribution[pos_max]))
    #
    #     indx = np.random.choice(range(len(probabilities)), p=np.copy(probabilities))
    #
    #     pos = np.unravel_index(indx, self.settings.env_shape[1:])
    #     episode.agent[0][0] = episode.get_height_init()
    #     episode.agent[0][1] = pos[0]
    #     episode.agent[0][2] = pos[1]
    #     # print ("Initialize pedestrian pos in 2D: " + str(episode.agent[0][1:]))
    #
    #     # Vector from pedestrian to car in voxels
    #     vector_car_to_pedestrian = episode.agent[0][1:] - episode.init_car_pos
    #
    #     # print ("Vector car to pedestrian "+str(vector_car_to_pedestrian))
    #     if np.linalg.norm(episode.init_car_vel) < 0.01:
    #         speed = 3 * 5  # Speed voxels/second
    #         # print ("Desired speed pedestrian " + str(speed * .2) + " car vel " + str(episode.init_car_vel * .2))
    #
    #         # Unit orthogonal direction
    #         unit = -vector_car_to_pedestrian * (
    #             1 / np.linalg.norm(vector_car_to_pedestrian))
    #
    #         # Set ortogonal direction to car
    #         episode.vel_init = np.array([0, unit[0], unit[1]]) * speed * episode.frame_time  # set this correctly
    #         episode.speed_init = np.linalg.norm(episode.vel_init)
    #         if episode.follow_goal:
    #             episode.goal = episode.agent[0].copy()
    #             episode.goal[1:] = episode.init_car_pos
    #             print (" car still Pedestrian goal init " + str(episode.goal))
    #             if self.settings.learn_goal:
    #                 episode.manual_goal = episode.goal[1:].copy()
    #                 if self.settings.goal_gaussian:
    #                     # episode.goal=episode.agent[0].copy()
    #
    #                     # print ("Before adding random goal : " + str(episode.goal))
    #                     random_numbers = [np.random.normal(probabilities_goal[0][0], self.settings.goal_std, 1),
    #                                       np.random.normal(probabilities_goal[0][1], self.settings.goal_std, 1)]
    #                     # print ("Randon numbers " + str(random_numbers)+ " mean "+str(probabilities_goal[0][0]))
    #                     random_numbers[0] = random_numbers[0] * self.settings.env_shape[1]
    #                     random_numbers[1] = random_numbers[1] * self.settings.env_shape[2]
    #                     # print ("Randon numbers after scaling " + str(random_numbers))
    #                     episode.goal[1] += random_numbers[0]
    #                     episode.goal[2] += random_numbers[1]
    #                     # print ("After addition: "+str(episode.goal))
    #                 else:
    #                     indx = np.random.choice(range(len(probabilities)), p=np.copy(probabilities_goal))
    #                     pos = np.unravel_index(indx, self.settings.env_shape[1:])
    #                     episode.goal[0] = episode.get_height_init()
    #                     episode.goal[1] = pos[0]
    #                     episode.goal[2] = pos[1]
    #             # episode.goal[2] += goal[1][0] * 256
    #             print (" car still Pedestrian goal " + str(episode.goal))
    #         if self.settings.speed_input:
    #             episode.goal_time = min(np.linalg.norm(vector_car_to_pedestrian) / episode.speed_init,
    #                                     episode.seq_len - 2)
    #             # print ("Car standing, Pedestrian initial speed: voxels / frame " + str(episode.speed_init) + " dist " + str(
    #             #     np.linalg.norm(vector_car_to_pedestrian)) + " timeframe " + str(
    #             #     episode.goal_time))
    #             if self.settings.learn_goal:
    #                 episode.goal_time = episode.seq_len - 2
    #         return episode.agent[0], 11, episode.vel_init  # pos, indx, vel_init
    #     # set initial pedestrian velocity orthogonal to car!- voxels
    #     # time to collision in seconds
    #
    #     time_to_collision = np.dot(vector_car_to_pedestrian, episode.init_car_vel) / (
    #     np.linalg.norm(episode.init_car_vel) ** 2)
    #     vector_travelled_by_car_to_collision = time_to_collision * episode.init_car_vel
    #     # time to collision in frames:
    #     time_to_collision = time_to_collision * episode.frame_rate
    #
    #     # print ("scalar product between car direction "+str(episode.init_car_vel)+" and vector car to pedestrian "+str(vector_car_to_pedestrian)+ " : "+ str(np.dot(vector_car_to_pedestrian, episode.init_car_vel)))
    #     #
    #     # print ("Norm on v "+str(np.linalg.norm(episode.init_car_vel)**2)+" time to collision car "+str(time_to_collision)+" in s "+str(time_to_collision/episode.frame_rate))
    #     vector_travelled_by_pedestrian_to_collision = vector_travelled_by_car_to_collision - vector_car_to_pedestrian
    #
    #     # print ("Vector to collision by car "+str(vector_travelled_by_car_to_collision)+" pos at collision "+str(vector_travelled_by_car_to_collision+episode.init_car_pos))
    #     # print ("Vector to collision by ped " + str(vector_travelled_by_pedestrian_to_collision) + " pos at collision " + str(
    #     #     vector_travelled_by_pedestrian_to_collision + episode.agent[0][1:]))
    #
    #     # pedestrian speed in voxels per frame
    #     speed = np.linalg.norm(vector_travelled_by_pedestrian_to_collision) / time_to_collision
    #
    #     # print ("Desired speed pedestrian " + str(speed ) + " dist to travel "+str(np.linalg.norm(vector_travelled_by_pedestrian_to_collision)) )
    #
    #     speed = min(speed, 3 * 5 * episode.frame_time)  # Speed voxels/second, max speed= 3m/s, 5 voxels per meter
    #
    #     # Unit orthogonal direction
    #     unit = vector_travelled_by_pedestrian_to_collision * (
    #     1 / np.linalg.norm(vector_travelled_by_pedestrian_to_collision))
    #
    #     # Set ortogonal direction to car
    #     episode.vel_init = np.array([0, unit[0], unit[1]]) * speed  # set this correctly
    #
    #     # print ("Pedestrian intercepting car? "+str(episode.intercept_car(0, all_frames=False)))
    #     episode.speed_init = np.linalg.norm(episode.vel_init)
    #     if episode.follow_goal:
    #
    #         episode.goal = episode.agent[0] + (2 * episode.vel_init * time_to_collision)
    #         print ("Pedestrian goal init " + str(episode.goal))
    #         if self.settings.learn_goal:
    #             episode.goal = episode.agent[0] + (episode.vel_init * time_to_collision)
    #             episode.manual_goal = episode.goal[1:].copy()
    #             if self.settings.goal_gaussian:
    #                 # print ("Before adding random goal : " + str(episode.goal))
    #                 random_numbers = [np.random.normal(probabilities_goal[0][0], self.settings.goal_std, 1),
    #                                   np.random.normal(probabilities_goal[0][1], self.settings.goal_std, 1)]
    #                 # print ("Randon numbers " + str(random_numbers) + " mean " + str(probabilities_goal[0]))
    #                 random_numbers[0] = random_numbers[0] * self.settings.env_shape[1]
    #                 random_numbers[1] = random_numbers[1] * self.settings.env_shape[2]
    #                 # print ("Randon numbers after scaling " + str(random_numbers))
    #                 episode.goal[1] += random_numbers[0]
    #                 episode.goal[2] += random_numbers[1]
    #                 # print ("After addition: " + str(episode.goal))
    #             else:
    #                 indx = np.random.choice(range(len(probabilities_goal)), p=np.copy(probabilities_goal))
    #                 pos = np.unravel_index(indx, self.settings.env_shape[1:])
    #                 episode.goal[0] = episode.get_height_init()
    #                 episode.goal[1] = pos[0]
    #                 episode.goal[2] = pos[1]
    #                 # print ("Goal pos "+str(indx)+" in pos "+str(pos))
    #
    #         print ("Pedestrian goal " + str(episode.goal))
    #     if self.settings.speed_input:
    #         episode.goal_time = min(time_to_collision * 2, episode.seq_len - 2)
    #         if self.settings.learn_goal:
    #             episode.goal_time = episode.seq_len - 2
    #     # print ("Pedestrian initial speed: voxels / frame "+str(episode.speed_init)+" dist "+str(np.linalg.norm(vector_travelled_by_pedestrian_to_collision) )+" timeframe "+str(episode.goal_time))
    #     # print ("Pedestrian vel " + str(episode.vel_init[1:])+" time to goal "+str(episode.goal_time)+" dist travelled "+str(episode.goal_time*episode.vel_init)+" final pos "+str(episode.agent[0]+episode.goal_time*episode.vel_init) )
    #     return episode.agent[0], 11, episode.vel_init  # pos, indx, vel_init
    #
    # def get_vel(self, episode_in, frame):
    #     pass
    #
    # def fully_connected_size(self, dim_p):
    #     return 0
    #
    # def construct_feed_dict(self, episode_in, frame, agent_frame, training=True):
    #     feed_dict = {}
    #     feed_dict[self.state_in] = self.get_input_init(episode_in)
    #     return feed_dict
    #
    # def get_input_init(self, episode_in):
    #     sem = np.zeros((self.settings.env_shape[1], self.settings.env_shape[2], self.nbr_channels))
    #     segmentation = (episode_in.reconstruction_2D[:, :, 3] * 33.0).astype(np.int)
    #     sem[:, :, 0:3] = episode_in.reconstruction_2D[:, :, 0:3].copy()
    #
    #     temp = episode_in.people_predicted[1].copy()
    #     temp[temp != 0] = 1.0 / temp[temp != 0]
    #     sem[:, :, 3] = temp.copy()  # *self.temporal_scaling
    #     temp = episode_in.car_predicted[1].copy()
    #     temp[temp != 0] = 1.0 / temp[temp != 0]
    #     sem[:, :, 4] = temp.copy()
    #
    #     for person in episode_in.people[0]:
    #         print ("Pedestrian " + str([person[1][0], person[1][1], person[2][0], person[2][1]]))
    #         sem[person[1][0]:person[1][1] + 1, person[2][0]:person[2][1] + 1, 5] = np.ones_like(
    #             sem[person[1][0]:person[1][1] + 1, person[2][0]:person[2][1] + 1, 5])
    #
    #     for car in episode_in.cars[0]:
    #         sem[car[2]:car[3], car[4]:car[5], 6] = np.ones_like(sem[car[2]:car[3], car[4]:car[5], 6])
    #
    #     # Do this faster somewhoe.
    #     for x in range(sem.shape[0]):
    #         for y in range(sem.shape[1]):
    #             if segmentation[x, y] > 0 and segmentation[x, y] != 23:
    #                 sem[x, y, self.labels_indx[segmentation[x, y]] + 7] = 1
    #     return np.expand_dims(sem, axis=0)
    #
    # # construct_feed_dict(self, episode_in, frame, agent_frame, training=True):
    # def grad_feed_dict(self, agent_action, agent_measures, agent_pos, agent_velocity, ep_itr, episode_in, frame,
    #                    reward, statistics, poses, priors, initialization_car, agent_speed=None, training=True, agent_frame=-1):
    #     if agent_frame < 0 or not training:
    #         agent_frame = frame
    #     r = reward[ep_itr, 0]
    #     # print  ("Reward "+str(r)+" rewards "+str(reward[ep_itr, :]))
    #     if sum(agent_measures[ep_itr, :agent_frame, 0]) > 0 or sum(agent_measures[ep_itr, :agent_frame, 13]) > 0:
    #         r = 0
    #     # print ("Frame "+str(frame)+" Episode "+str(ep_itr)+" Reward "+str(r)+" hit by car:"+str(agent_measures[ep_itr,agent_frame, 0])+" Reached goal "+str(agent_measures[ep_itr,agent_frame, 13]))
    #
    #     feed_dict = {self.state_in: self.get_input_init(episode_in),
    #                  self.advantages: r,
    #                  self.sample: self.get_sample(statistics, ep_itr, agent_frame, initialization_car),
    #                  self.goal: self.get_goal(statistics, ep_itr, agent_frame, initialization_car)}
    #
    #     feed_dict[self.prior] = np.reshape(priors[ep_itr, :, 0] * (1.0 / max(priors[ep_itr, :, 0])), (
    #     self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2], 1))
    #     if self.settings.learn_goal and not self.settings.goal_gaussian:
    #         feed_dict[self.prior_goal] = np.reshape(priors[ep_itr, :, 0] * (1.0 / max(priors[ep_itr, :, 2])), (
    #         self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2], 1))
    #
    #     return feed_dict
    #
    # def train(self, statistics, episode_in, filename, filename_weights, poses, priors, initialization_car, seq_len=-1):
    #     if seq_len == -1:
    #         seq_len = self.settings.seq_len_train
    #     agent_action, agent_measures, agent_pos, agent_reward, agent_reward_d, agent_velocity, agent_vel = self.stats(
    #         statistics)
    #
    #     if self.settings.normalize:
    #         reward = self.normalize_reward(agent_reward_d, agent_measures)
    #     else:
    #         reward = agent_reward_d
    #
    #         # print "Gradient: "+str(seq_len-1)
    #     for ep_itr in range(statistics.shape[0]):
    #         self.reset_mem()
    #
    #         feed_dict = self.grad_feed_dict(agent_action, agent_measures, agent_pos, agent_velocity, ep_itr,
    #                                         episode_in, 0, reward, statistics, poses, priors, initialization_car,
    #                                         agent_speed=agent_vel)
    #
    #         print ("Grad Feed dict")
    #         # for key, value in feed_dict.items():
    #         #     if key == self.state_in:
    #         #         print (value.shape)
    #         #
    #         #         print ("Input cars " + str(np.sum(value[0, :, :, 6])) + " people " + str(
    #         #             np.sum(value[0, :, :, 5])) + " Input cars traj" + str(
    #         #             np.sum(value[0, :, :, 4])) + " people traj " + str(
    #         #             np.sum(value[0, :, :, 3])))
    #         #         print ("Input building " + str(np.sum(value[0, :, :, 7])) + " fence " + str(
    #         #             np.sum(value[0, :, :, 7 + 1])) + " static " + str(
    #         #             np.sum(value[0, :, :, 7 + 2])) + " pole " + str(
    #         #             np.sum(value[0, :, :, 7 + 3])))
    #         #         print ("Input sidewalk " + str(np.sum(value[0, :, :, 7 + 5])) + " road " + str(
    #         #             np.sum(value[0, :, :, 7 + 4])) + " veg. " + str(
    #         #             np.sum(value[0, :, :, 7 + 6])) + " wall " + str(np.sum(value[0, :, :, 7 + 7])) + " sign " + str(
    #         #             np.sum(value[0, :, :, 7 + 8])))
    #         #         # print ("Equal to feed dict ")
    #         #         # print (np.array_equal(value, self.feed_dict[self.state_in]))
    #         #         # print ("Cars: ")
    #         #         # print (value[0, :, :, 6])
    #         #         # print ("Cars traj : ")
    #         #         # print (value[0, :, :, 4])
    #         #         # print ("People:")
    #         #         # print (value[0, :, :, 5])
    #         #         # print ("People traj : ")
    #         #         # print (value[0, :, :, 3])
    #         #     elif key == self.prior:
    #         #         print ("prior " + str(np.sum(value[0, :, :])))
    #         #         # print ("Equal to feed dict ")
    #         #         # print (np.array_equal(value, self.feed_dict[self.prior]))
    #         #         # indxs=self.feed_dict[self.prior]-value
    #         #         # print ("nonzero  grad dict "+str(np.sum(value)))
    #         #         # print ("nonzero  feed dict "+str(np.sum(self.feed_dict[self.prior])))
    #         #     else:
    #         #         print (key)
    #         #         print (value)
    #
    #         self.calculate_gradients(feed_dict, statistics, 0,
    #                                  ep_itr)  # self.calculate_gradients(episode_in, frame, feed_dict) %
    #
    #     if not self.settings.overfit or (self.settings.overfit and self.num_grad_itrs % 20 == 0):
    #         with open(filename, 'wb') as f:
    #             pickle.dump(self.gradBuffer, f, pickle.HIGHEST_PROTOCOL)
    #
    #         [weights] = self.sess.run([self.tvars])
    #         with open(filename_weights, 'wb') as f:
    #             pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)
    #     else:
    #         [weights] = self.sess.run([self.tvars])
    #         # print "Weights "
    #         # print weights
    #     # if self.num_grad_itrs % (self.update_frequency) == 0 and self.num_grad_itrs != 0:
    #     self.update_gradients()
    #
    #     return statistics


class InitializerArchGaussianNet(InitializerNet):
    def __init__(self, settings, weights_name="policy"):
        super(InitializerArchGaussianNet, self).__init__(settings, weights_name="init")

    def define_loss(self, dim_p):
        self.sample = tf.compat.v1.placeholder(shape=[2], dtype=self.DTYPE, name="sample")
        self.normal_dist = tf.contrib.distributions.Normal(self.probabilities, self.settings.init_std)
        self.responsible_output = self.normal_dist.prob(self.sample)

        self.l2_loss = tf.nn.l2_loss(self.probabilities - self.sample)  # tf.nn.l2_loss
        # self.loss = tf.reduce_mean(self.advantages * (2 * tf.log(self.settings.sigma_vel) + tf.log(2 * tf.pi) + (
        # self.l2_loss / (self.settings.sigma_vel * self.settings.sigma_vel))))
        self.loss = tf.reduce_mean(self.advantages * self.l2_loss)

        return self.sample, self.loss

    def get_feature_vectors_gradient(self, agent_action, agent_frame, agent_measures, agent_pos, agent_speed, ep_itr,
                                     episode_in, feed_dict, frame, poses, statistics, training):
        pass

    def fully_connected(self, dim_p, prev_layer):
        print ("Fully connected flattened layer: " + str(prev_layer))
        self.flattened_layer = tf.reshape(prev_layer, [-1])
        # dim = np.prod(prev_layer.get_shape().as_list()[1:])
        # prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        # weights = tf.get_variable('weights', [dim, 2], self.DTYPE,
        #                           initializer=tf.contrib.layers.xavier_initializer())
        # biases = self.bias_variable('biases', [2])

        # self.mu = tf.matmul(tf.expand_dims(prev_layer_flat,0), weights)
        # self.mu = tf.add(self.mu, biases)

    # return [statistics[ep_itr, agent_frame, 6]]
    def get_sample(self, statistics, ep_itr, agent_frame, init_car):
        # print ([statistics[ep_itr, 0, 1]], [statistics[ep_itr, 0, 2]])
        # print ("Get Sample "+str(statistics[ep_itr, 0, 1]*self.settings.env_shape[1]+statistics[ep_itr, 0, 2]))
        # np.ravel_multi_index(([statistics[ep_itr, 0, 1]], [statistics[ep_itr, 0, 2]]), self.settings.env_shape[1:] )
        # print ("Get Sample "+str([statistics[ep_itr, 0, 1:2]])+" flattened: "+str(np.ravel_multi_index(([int(statistics[ep_itr, 0, 1])], [int(statistics[ep_itr, 0, 2])]), self.settings.env_shape[1:] )))
        sample = statistics[ep_itr, 0, 1:3] - init_car[ep_itr, 1:3]
        # sample[0]=sample[0]/self.settings.env_shape[1]
        # sample[1] = sample[1] / self.settings.env_shape[2]
        return sample
        # [statistics[ep_itr, 0, 1]*self.settings.env_shape[1]+statistics[ep_itr, 0, 2]]

    # def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):
    #     return episode_in.reconstruction

    def calc_probabilities(self, fc_size):
        # print ("Define probabilities: " + str(self.flattened_layer))
        self.probabilities = self.mu  # tf.sigmoid(self.mu)

    def importance_sample_weight(self, responsible, statistics, ep_itr, frame, responsible_v=0):
        pass

    def get_feature_vectors(self, agent_frame, episode_in, feed_dict, frame):
        pass

    def apply_net(self, feed_dict, episode, frame, training, max_val=False, viz=False):
        # Choose car
        episode.init_car_id = np.random.choice(episode.init_cars)
        episode.init_car_pos = np.array([np.mean(episode.cars_dict[episode.init_car_id][0][2:4]),
                                         np.mean(episode.cars_dict[episode.init_car_id][0][4:])])
        car_pos_next = np.array([np.mean(episode.cars_dict[episode.init_car_id][1][2:4]),
                                 np.mean(episode.cars_dict[episode.init_car_id][1][4:])])
        episode.init_car_vel = (car_pos_next - episode.init_car_pos) / episode.frame_time

        episode.init_method = 7

        car_dims = [episode.cars_dict[episode.init_car_id][0][3] - episode.cars_dict[episode.init_car_id][0][2],
                    episode.cars_dict[episode.init_car_id][0][5] - episode.cars_dict[episode.init_car_id][0][4]]
        car_max_dims = max(car_dims)
        car_min_dims = min(car_dims)

        episode.calculate_prior(episode.init_car_pos, episode.init_car_vel, car_max_dims, car_min_dims)

        flat_prior = episode.prior.flatten()
        feed_dict[self.prior] = np.expand_dims(np.expand_dims(episode.prior * (1 / max(flat_prior)), axis=0), axis=-1)

        mean_vel, summary_train = self.sess.run([self.mu, self.train_summaries], feed_dict)
        print ("Mean vel " + str(mean_vel))
        episode.init_distribution = copy.copy(mean_vel[0])

        episode.agent[0] = np.zeros(3)
        episode.agent[0][0] = episode.get_height_init()
        episode.agent[0][1] = episode.init_car_pos[0] + np.random.normal(mean_vel[0][0], self.settings.init_std, 1)
        episode.agent[0][2] = episode.init_car_pos[1] + np.random.normal(mean_vel[0][1], self.settings.init_std, 1)
        episode.agent[0][1] = max(min(episode.agent[0][1], self.settings.env_shape[1] - 1), 0)
        episode.agent[0][2] = max(min(episode.agent[0][2], self.settings.env_shape[2] - 1), 0)

        while (episode.prior[int(episode.agent[0][1]), int(episode.agent[0][1])] == 0):
            episode.agent[0][1] = episode.init_car_pos[0] + np.random.normal(mean_vel[0][0], self.settings.init_std, 1)
            episode.agent[0][2] = episode.init_car_pos[1] + np.random.normal(mean_vel[0][1], self.settings.init_std, 1)
            episode.agent[0][1] = max(min(episode.agent[0][1], self.settings.env_shape[1] - 1), 0)
            episode.agent[0][2] = max(min(episode.agent[0][2], self.settings.env_shape[2] - 1), 0)
        # make sure agent is in environment borders
        episode.agent[0][1] = max(min(episode.agent[0][1], self.settings.env_shape[1] - 1), 0)
        episode.agent[0][2] = max(min(episode.agent[0][2], self.settings.env_shape[2] - 1), 0)

        # print ("Initialize pedestrian "  + str( indx) + " pos in 2D: " + str(episode.agent[0][1:])+" no prior probability: "+str( episode.init_distribution[indx])+" prior probability"+str(probabilities[[indx]]))

        # Vector from pedestrian to car in voxels
        vector_car_to_pedestrian = episode.agent[0][1:] - episode.init_car_pos
        if np.linalg.norm(episode.init_car_vel) < 0.01:
            speed = 3 * 5  # Speed voxels/second
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

        speed = min(ratio_pedestrian_to_car_dist_to_collision * np.linalg.norm(episode.init_car_vel),
                    3 * 5)  # Speed voxels/second
        # print ("Desired speed pedestrian "+str(speed*.2)+" ratio_pedestrian_to_car_dist_to_collision "+str(ratio_pedestrian_to_car_dist_to_collision)+" "+str(episode.init_car_vel*.2))

        # Unit orthogonal direction
        unit = -vector_travelled_by_pedestrian_to_collision * (
            1 / np.linalg.norm(vector_travelled_by_pedestrian_to_collision))

        # Set ortogonal direction to car
        episode.vel_init = np.array([0, unit[0], unit[1]]) * speed * episode.frame_time  # set this correctly

        # print ("Pedestrian intercepting car? "+str(episode.intercept_car(0, all_frames=False)))

        return episode.agent[0], 11, episode.vel_init  # pos, indx, vel_init




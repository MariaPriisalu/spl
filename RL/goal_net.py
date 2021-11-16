import numpy as np

from initializer_net import InitializerNet, InitializerGaussianNet
from settings import NBR_MEASURES,STATISTICS_INDX,STATISTICS_INDX_MAP,PEDESTRIAN_MEASURES_INDX
import tensorflow as tf
from settings import RANDOM_SEED
tf.compat.v1.set_random_seed(RANDOM_SEED)
import tensorflow_probability as tfp
tfd = tfp.distributions



class GoalNet(InitializerNet):
    # Softmax Simplified
    def __init__(self, settings, weights_name="policy") :
        #self.temporal_scaling = 0.1 * 0.3

        super(GoalNet, self).__init__(settings, weights_name="goal")



    def define_loss(self, dim_p):
        self.goal = tf.compat.v1.placeholder(shape=[1], dtype=tf.int32, name="sample")
        self.responsible_output = tf.slice(self.probabilities, self.goal, [1]) + np.finfo(
            np.float32).eps
        self.prior_flat = tf.reshape(self.prior, [
            self.settings.batch_size * self.settings.env_shape[1] * self.settings.env_shape[2]])
        self.responsible_output_prior = tf.slice(self.prior_flat, self.goal, [1]) + np.finfo(
            np.float32).eps
        self.distribution = self.prior_flat * self.probabilities

        self.loss = -tf.reduce_mean(
            (tf.log(self.responsible_output) + tf.log(self.responsible_output_prior)) * self.advantages)

        if self.settings.entr_par_goal:
            y_zeros = tf.zeros_like(self.distribution)
            y_mask = tf.math.greater(self.distribution, y_zeros)
            res = tf.boolean_mask(self.distribution, y_mask)
            logres = tf.math.log(res)

            self.entropy = tf.nn.softmax_cross_entropy_with_logits(labels=res, logits=logres)
            self.loss = -tf.reduce_mean(self.settings.entr_par_init * self.entropy)
        if self.settings.learn_time :
            self.time_requirement = tf.compat.v1.placeholder(shape=[self.settings.batch_size, 1], dtype=self.DTYPE,
                                                        name="time_requirement")
            self.beta_distr=tfd.Beta(self.alpha, self.beta)
            self.time_prob=self.beta_distr.prob(self.time_requirement)
            self.loss = self.loss - tf.reduce_mean(self.advantages * tf.log(self.time_prob))
        return self.goal, self.loss


    #return [statistics[ep_itr, agent_frame, 6]]
    def get_sample(self, statistics,ep_itr, agent_frame,  initialization_car):
        print ("Get sample "+str([np.ravel_multi_index( statistics[ep_itr, 3:5, STATISTICS_INDX.goal].astype(int),
                                    self.settings.env_shape[1:])]))
        return [np.ravel_multi_index( statistics[ep_itr, 3:5, STATISTICS_INDX.goal].astype(int),
                                    self.settings.env_shape[1:])]

    def apply_net(self, feed_dict, episode, frame, training, max_val=False, viz=False):

        flat_prior=episode.calculate_goal_prior( episode.init_car_pos, episode.init_car_vel,episode.agent[0][1:], max(self.settings.car_dim[1:]), max(self.settings.car_dim[1:])).flatten()
        print ("Max value of goal prior "+str( max(flat_prior)))
        feed_dict[self.prior] = np.expand_dims(np.expand_dims(episode.goal_prior * (1 / max(flat_prior)), axis=0), axis=-1)
        # print ("goal prior shape " + str(feed_dict[self.prior].shape))
        # print ("Goal prior min "+str(np.min(flat_prior))+" max "+str(np.max(flat_prior)))
        # print ("After normalizing goal prior min " + str(np.min(feed_dict[self.prior][0, :, :, 0])) + " max " + str(np.max(feed_dict[self.prior][0, :, :, 0])))
        if self.settings.learn_time:
            probabilities, flattend_layer, conv_1,alpha, beta, summary_train = self.sess.run(
                [self.probabilities, self.flattened_layer, self.conv1,self.alpha, self.beta, self.train_summaries], feed_dict)
            episode.goal_time = np.random.beta(alpha, beta)
            # print ("Model outputs alpha " + str(alpha) + " beta " + str(beta) + " factor " + str(episode.goal_time))

        else:
            probabilities,flattend_layer,conv_1, summary_train = self.sess.run([self.probabilities,self.flattened_layer,self.conv1, self.train_summaries], feed_dict)

        episode.goal_distribution = np.copy(probabilities)
        pos_max=np.argmax(episode.goal_distribution)
        pos_min = np.argmin(episode.goal_distribution)
        #
        # print ("Init distr "+str(np.sum(np.abs(episode.init_distribution)))+"  max: "+str(pos_max)+" pos in 2D: "+str(np.unravel_index(pos_max, self.settings.env_shape[1:])))
        # print ("Init distr  min pos "+ str(pos_min) +" pos in 2D: "+str(np.unravel_index(pos_min, self.settings.env_shape[1:])))
        print (" max value "+str(episode.goal_distribution[pos_max])+"  min value "+str(episode.goal_distribution[pos_min]) )

        reshaped_init=np.reshape(episode.goal_distribution, episode.prior.shape)

        # print ("10th row:"+str(reshaped_init[10,:10]))
        # print("Max of flattened layer "+str(max(flattend_layer))+" min of flattened layer "+str(min(flattend_layer)))
        # print ("10th row of conv "+str(conv_1[0,10,:10,0]))
        #

        pos_max = np.argmax(flat_prior)
        pos_min = np.argmin(flat_prior)
        # print ("Prior distr " + str(np.sum(np.abs(flat_prior))) + "  max: " + str(
        #     pos_max) + " pos in 2D: " + str(np.unravel_index(pos_max, self.settings.env_shape[1:])))
        # print ("Prior distr  min pos " + str(pos_min) + " pos in 2D: " + str(
        #     np.unravel_index(pos_min, self.settings.env_shape[1:])))
        # print (" max value " + str(flat_prior[pos_max]) + "  min value " + str(flat_prior[pos_min]))
        #
        # print ("10th row:" + str(episode.prior[10, :10]))

        probabilities = probabilities*flat_prior#episode.prior.flatten()
        probabilities_reshaped=np.reshape(probabilities, episode.prior.shape)

        pos_max = np.argmax(probabilities)
        pos_min = np.argmin(probabilities)

        # for car in episode.cars[0]:
        #     print ("Probabilities reshaped " + str(car))
        #     print (np.sum(np.abs(probabilities_reshaped[car[2]:car[3], car[4]:car[5]])))

        pos_max = np.argmax(probabilities)
        print ("After prior Init distr " + str(np.sum(np.abs(probabilities))) + "  max: " + str(
            pos_max) + " pos in 2D: " + str(np.unravel_index(pos_max, self.settings.env_shape[1:])))

        probabilities=probabilities*(1/np.sum(probabilities))

        print ("Final distr " + str(np.sum(np.abs(probabilities))) + "  max: " + str(
            pos_max) + " pos in 2D: " + str(np.unravel_index(pos_max, self.settings.env_shape[1:])))
        print ("Final distr  min pos " + str(pos_min) + " pos in 2D: " + str(
            np.unravel_index(pos_min, self.settings.env_shape[1:])))
        print (" max value " + str(probabilities[pos_max]) + "  min value " + str(probabilities[pos_min]))
        #
        # print ("10th row:" + str(probabilities_reshaped[10, :10]))

        pos_max = np.argmax(probabilities)
        # if max_val and not training:
        #     #print ("Maximal evaluation init net")
        #     pos = np.unravel_index(pos_max, self.settings.env_shape[1:])
        #     episode.goal[0] = episode.get_height_init()
        #     episode.goal[1] = pos[0]
        #     episode.goal[2] = pos[1]
        # else:
        # print ("After normalization " + str(np.sum(np.abs(probabilities))) + "  max: " + str(
        #     pos_max) + " pos in 2D: " + str(np.unravel_index(pos_max, self.settings.env_shape[1:]))+" no prior max "+str(episode.init_distribution[pos_max]))

        indx = np.random.choice(range(len(probabilities)), p=np.copy(probabilities))

        pos = np.unravel_index(indx, self.settings.env_shape[1:])
        episode.goal[0] = episode.get_height_init()
        episode.goal[1] = pos[0]
        episode.goal[2] = pos[1]
        if self.settings.speed_input:
            if self.settings.learn_time:
                episode.goal_time = episode.goal_time * 3 * 5 * episode.frame_time
                print ("Episode speed " + str(episode.goal_time) + " factor " + str(
                    15 * episode.frame_time) + " frametime " + str(episode.frame_time))
                episode.goal_time = np.linalg.norm(episode.goal[1:] - episode.agent[0][1:]) / episode.goal_time
                print ("Episode goal time " + str(episode.goal_time))
            else:
                episode.goal_time = min(np.linalg.norm(episode.goal[1:] - episode.agent[0][1:]) / episode.speed_init,
                                        episode.seq_len - 2)
        print ("Goal "+str(episode.goal))
        return episode.agent[0], 11,episode.vel_init #  pos, indx, vel_init




      #construct_feed_dict(self, episode_in, frame, agent_frame, training=True):
    def grad_feed_dict(self, agent_action, agent_measures, agent_pos, agent_velocity, ep_itr, episode_in, frame,
                       reward, statistics,poses,priors,initialization_car, agent_speed=None, training=True, agent_frame=-1):
        if agent_frame < 0 or not training:
            agent_frame = frame
        r=reward[ep_itr, 0]
        # print  ("Reward "+str(r)+" rewards "+str(reward[ep_itr, :]))
        if sum(agent_measures[ep_itr, :agent_frame, PEDESTRIAN_MEASURES_INDX.hit_by_car])>0 or sum(agent_measures[ep_itr, :agent_frame, PEDESTRIAN_MEASURES_INDX.goal_reached])>0:
            r=0
        # print ("Frame "+str(frame)+" Episode "+str(ep_itr)+" Reward "+str(r)+" hit by car:"+str(agent_measures[ep_itr,agent_frame, 0])+" Reached goal "+str(agent_measures[ep_itr,agent_frame, 13]))

        feed_dict = {self.state_in: self.get_input_init(episode_in),
                     self.advantages: r,
                     self.goal: self.get_sample(statistics, ep_itr, agent_frame,  initialization_car)}


        #feed_dict[self.prior] = np.zeros((1, self.settings.env_shape[1], self.settings.env_shape[2]))

        feed_dict[self.prior]= np.reshape(priors[ep_itr,:,2]*(1.0/max(priors[ep_itr,:,2])), (self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2],1))
        # feed_dict[self.prior][0, episode.agent[0][1], episode.agent[0][2], 0] = 2
        # feed_dict[self.prior][0, :, :, 1] = episode.goal_prior * (1 / max(flat_prior))
        if self.settings.learn_time and self.goal_net:
            goal = statistics[ep_itr, 3:5, STATISTICS_INDX.goal]
            agent_pos = statistics[ep_itr, 0, STATISTICS_INDX.agent_pos[0]:STATISTICS_INDX.agent_pos[1]]
            goal_dist = np.linalg.norm(goal - agent_pos)
            goal_time = statistics[ep_itr, 5, STATISTICS_INDX.goal]
            # episode.goal_time = episode.goal_time * 3 * 5 * episode.frame_time * np.linalg.norm(
            #     episode.goal[1:] - episode.agent[1:])
            print("Agent position " + str(agent_pos) + " goal " + str(goal))
            print ("Episode goal time " + str(goal_time) + " goal time " + str(goal_dist) + " fraction " + str(
                goal_dist / goal_time) + " ratio " + str(17 / 15))
            if goal_time == 0:
                feed_dict[self.time_requirement] = np.array([[0]])
            else:
                feed_dict[self.time_requirement] = np.array([[goal_dist / goal_time * (17 / 15)]])
            print ("Feed dict input " + str(feed_dict[self.time_requirement]))
        return feed_dict


class GoalGaussianNet(GoalNet):
    def __init__(self, settings, weights_name="goal") :
        super(GoalGaussianNet, self).__init__(settings, weights_name="goal")


    def define_loss(self, dim_p):
        self.goal = tf.compat.v1.placeholder(shape=[2], dtype=self.DTYPE, name="goal")
        self.normal_dist = tf.contrib.distributions.Normal(self.probabilities, self.settings.goal_std)
        self.responsible_output = self.normal_dist.prob(self.goal)

        self.l2_loss = tf.nn.l2_loss(self.probabilities - self.goal)  # tf.nn.l2_loss
        # self.loss = tf.reduce_mean(self.advantages * (2 * tf.log(self.settings.sigma_vel) + tf.log(2 * tf.pi) + (
        # self.l2_loss / (self.settings.sigma_vel * self.settings.sigma_vel))))
        self.loss = tf.reduce_mean(self.advantages * self.l2_loss)

        return self.goal, self.loss

    def get_feature_vectors_gradient(self, agent_action, agent_frame, agent_measures, agent_pos, agent_speed, ep_itr,
                                     episode_in, feed_dict, frame, poses, statistics, training):
        pass

    def fully_connected(self, dim_p, prev_layer):
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        self.flattened_layer_conv = tf.reshape(prev_layer, [-1])
        weights = tf.get_variable('weights_goal', [dim, 2], self.DTYPE,
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = self.bias_variable('biases_goal', [2])
        self.flattened_layer = tf.matmul(tf.reshape(self.flattened_layer_conv, [1, dim]), weights)
        self.flattened_layer = tf.add(self.flattened_layer, biases)




    #return [statistics[ep_itr, agent_frame, 6]]
    def get_sample(self, statistics, ep_itr, agent_frame, initialization_car):
        goal = statistics[ep_itr, 3:5, 38 + NBR_MEASURES]
        agent_pos = statistics[ep_itr, 0, 1:3]
        manual_goal = initialization_car[ep_itr, 5:]
        goal_dir = goal - manual_goal  # agent_pos
        goal_dir[0] = goal_dir[0]  # /self.settings.env_shape[1]
        goal_dir[1] = goal_dir[1]  # / self.settings.env_shape[2]
        print ("Get goal " + str(goal_dir))
        return goal_dir
        #[statistics[ep_itr, 0, 1]*self.settings.env_shape[1]+statistics[ep_itr, 0, 2]]

    # def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):
    #     return episode_in.reconstruction

    def calc_probabilities(self, fc_size):
        #print ("Define probabilities: " + str(self.flattened_layer))
        self.probabilities = tf.nn.sigmoid(self.flattened_layer)

    def importance_sample_weight(self,responsible, statistics, ep_itr, frame, responsible_v=0):
        pass

    def get_feature_vectors(self, agent_frame, episode_in, feed_dict, frame):
        pass

    def apply_net(self, feed_dict, episode, frame, training, max_val=False, viz=False):

        car_dims = [episode.cars_dict[episode.init_car_id][0][3] - episode.cars_dict[episode.init_car_id][0][2],
                    episode.cars_dict[episode.init_car_id][0][5] - episode.cars_dict[episode.init_car_id][0][4]]
        car_max_dims = max(car_dims)
        car_min_dims = min(car_dims)

        flat_prior = episode.calculate_goal_prior(episode.init_car_pos, episode.init_car_vel,
                                                  episode.agent[0][1:]).flatten()
        print ("Max value " + str(max(flat_prior)))
        feed_dict[self.prior] = np.expand_dims(np.expand_dims(episode.goal_prior * (1 / max(flat_prior)), axis=0),
                                               axis=-1)

        probabilities, flattend_layer, conv_1, summary_train = self.sess.run(
                    [self.probabilities, self.flattened_layer, self.conv1,
                     self.train_summaries], feed_dict)

        episode.goal_distribution = np.copy(probabilities)




        episode.goal[1:] = episode.manual_goal

        print (
        "Before adding random goal : " + str(episode.goal) + " model output " + str(probabilities))
        random_numbers = [np.random.normal(probabilities[0][0], self.settings.goal_std, 1),
                          np.random.normal(probabilities[0][1], self.settings.goal_std, 1)]
        print ("Randon numbers " + str(random_numbers) + " mean " + str(probabilities[0][0]))
        random_numbers[0] = random_numbers[0]  # * self.settings.env_shape[1]
        random_numbers[1] = random_numbers[1]  # * self.settings.env_shape[2]
        print (
        "Before adding random goal : " + str(episode.goal) + " model output " + str(probabilities))
        print ("Randon numbers after scaling " + str(random_numbers) + "  scaled mean " + str(
            probabilities[0][0] * self.settings.env_shape[1]) + "  scaled mean " + str(
            probabilities[0][1] * self.settings.env_shape[2]))
        print ("Final mean " + str(
            probabilities[0][0] + episode.goal[1]) + "  scaled mean " + str(
            probabilities[0][1] + episode.goal[2]))

        episode.goal[1] += random_numbers[0]
        episode.goal[2] += random_numbers[1]

        return episode.agent[0], 11, episode.vel_init  # pos, indx, vel_init

    def grad_feed_dict(self, agent_action, agent_measures, agent_pos, agent_velocity, ep_itr, episode_in, frame,
                       reward, statistics, poses, priors, initialization_car, agent_speed=None, training=True, agent_frame=-1):
        if agent_frame < 0 or not training:
            agent_frame = frame
        r = reward[ep_itr, 0]
        # print  ("Reward "+str(r)+" rewards "+str(reward[ep_itr, :]))
        if sum(agent_measures[ep_itr, :agent_frame, 0]) > 0 or sum(agent_measures[ep_itr, :agent_frame, 13]) > 0:
            r = 0
        # print ("Frame "+str(frame)+" Episode "+str(ep_itr)+" Reward "+str(r)+" hit by car:"+str(agent_measures[ep_itr,agent_frame, 0])+" Reached goal "+str(agent_measures[ep_itr,agent_frame, 13]))

        feed_dict = {self.state_in: self.get_input_init(episode_in),
                     self.advantages: r,
                     self.goal: self.get_sample(statistics, ep_itr, agent_frame, initialization_car)}

        # feed_dict[self.prior] = np.zeros((1, self.settings.env_shape[1], self.settings.env_shape[2]))
        prior=priors[ep_itr, :, STATISTICS_INDX_MAP.goal_prior]
        feed_dict[self.prior] = np.reshape(prior* (1.0 / max(prior)), (self.settings.batch_size, self.settings.env_shape[1], self.settings.env_shape[2], 1))
        # feed_dict[self.prior][0, episode.agent[0][1], episode.agent[0][2], 0] = 2
        # feed_dict[self.prior][0, :, :, 1] = episode.goal_prior * (1 / max(flat_prior))
        return feed_dict


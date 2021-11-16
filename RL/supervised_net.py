
import tensorflow as tf
from settings import RANDOM_SEED
tf.compat.v1.set_random_seed(RANDOM_SEED)
from net_continous import GaussianNet

import numpy as np
class SupervisedNet(GaussianNet):
    def __init__(self, settings):
        self.action_size_mem = 2
        super(SupervisedNet, self).__init__(settings)
        tf.compat.v1.set_random_seed(settings.random_seed_tf)
        np.random.seed(settings.random_seed_np)

    def get_dim_p(self):
        return 2

    def define_loss(self, dim_p):
        self.sample = tf.placeholder(shape=[2], dtype=self.DTYPE, name="sample")
        self.diff=self.sample-self.mu_vel
        self.squared_diff=tf.squared_difference(self.sample, self.mu_vel)
        self.loss =tf.reduce_mean(self.squared_diff)#tf.reduce_mean(tf.losses.mean_squared_error(self.sample,self.mu_vel))# tf.reduce_mean(tf.square(self.diff))#
        return self.sample, self.loss

    def fully_connected(self, dim_p, prev_layer):
        self.action_size_mem = 2
        super(SupervisedNet, self).fully_connected( dim_p, prev_layer)

    def feed_forward(self, episode_in, frame, training, agent_frame=-1, filename="",  filename_weights=""):
        if agent_frame < 0 or not training:
            agent_frame = frame
        # print "Net feed forward "+str(frame)+" "+str(agent_frame)
        feed_dict = self.construct_feed_dict(episode_in, frame, agent_frame, training=training)

        value = self.apply_net(feed_dict, episode_in, agent_frame, training, max_val=self.settings.deterministic_test, filename=filename,  filename_weights=filename_weights)
        # print "model out: "+str(value)
        self.num_iterations += 1
        return value


    def apply_net(self, feed_dict, episode, frame, training, max_val=False,no_labels=True,  filename="",  filename_weights=""):

        if training:
            # for key, value in feed_dict.items():
            #
            #     print key
            #     print value
            mean_vel, loss, grads, weights,diff, squared_diff, summaries_loss, summaries_grad = self.sess.run([self.mu_vel, self.loss, self.gradients,self.tvars, self.diff,self.squared_diff,self.loss_summaries, self.grad_summaries], feed_dict)
            episode.loss[frame] = loss
            episode.velocity[frame]=np.zeros_like(episode.probabilities[frame, 0:3])#mean_vel
            episode.velocity[frame][1]=mean_vel[0]
            episode.velocity[frame][2] = mean_vel[1]

            episode.action[frame] = episode.find_action_to_direction([0, mean_vel[0], mean_vel[1]], np.sqrt(mean_vel[0] ** 2 + mean_vel[1] ** 2))
            episode.speed[frame] =np.sqrt(mean_vel[0] ** 2 + mean_vel[1] ** 2) # ???????
            episode.probabilities[frame, 0:9] = np.zeros_like(episode.probabilities[frame, 0:9])
            episode.probabilities[frame, episode.action[frame]] = 1

            self.num_grad_itrs += 1
            # print "Net output "+str(mean_vel)+" diff: "+str(diff)+" loss: "+str(loss)
            # print "Sguared diff "+str(squared_diff)
            for idx, gradient in enumerate(grads):
                if np.isnan(gradient).any() or np.isnan(gradient).any():
                    logging.warning('Gradient is ' + str(gradient) + " given input " + str(feed_dict))
                else:
                    if frame>0:
                        self.gradBuffer[idx] += gradient/self.settings.update_frequency
                    # print "Weight "
                    # print str(weights[idx])
                    # print " Gradient "
                    # print str(gradient)
            if self.num_grad_itrs % self.settings.update_frequency == 0 and self.num_grad_itrs != 0:
                print("Update Gradients!----------------------------------------------------------")
                self.update_gradients()
                # with open(filename, 'wb') as f:
                #     pickle.dump(self.gradBuffer, f, pickle.HIGHEST_PROTOCOL)
                #
                # [weights] = self.sess.run([self.tvars])
                #
                # with open(filename_weights, 'wb') as f:
                #     pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)
                #     # if self.num_grad_itrs % (self.update_frequency) == 0 and self.num_grad_itrs != 0:

        elif no_labels:
            mean_vel, summary_train = self.sess.run([self.mu_vel, self.train_summaries], feed_dict)
            episode.velocity[frame] = np.zeros_like(episode.probabilities[frame, 0:3])  # mean_vel
            episode.velocity[frame][1] = mean_vel[0]
            episode.velocity[frame][2] = mean_vel[1]

            episode.action[frame] = episode.find_action_to_direction([0, mean_vel[0], mean_vel[1]],
                                                                     np.sqrt(mean_vel[0] ** 2 + mean_vel[1] ** 2))
            episode.speed[frame] = np.sqrt(mean_vel[0] ** 2 + mean_vel[1] ** 2)  # ???????
            episode.probabilities[frame, 0:9] = np.zeros_like(episode.probabilities[frame, 0:9])
            episode.probabilities[frame, episode.action[frame]] = 1
        else:

            mean_vel, loss, grads, summaries_loss, summaries_grad = self.sess.run(
                [self.mu_vel, self.loss, self.gradients, self.loss_summaries, self.grad_summaries], feed_dict)
            episode.loss[frame] = loss
            episode.velocity[frame] = np.zeros_like(episode.probabilities[frame, 0:3])  # mean_vel
            episode.velocity[frame][1] = mean_vel[0]
            episode.velocity[frame][2] = mean_vel[1]

            episode.action[frame] = episode.find_action_to_direction([0, mean_vel[0], mean_vel[1]],
                                                                     np.sqrt(mean_vel[0] ** 2 + mean_vel[1] ** 2))
            episode.speed[frame] = np.sqrt(mean_vel[0] ** 2 + mean_vel[1] ** 2)  # ???????
            episode.probabilities[frame, 0:9] = np.zeros_like(episode.probabilities[frame, 0:9])
            episode.probabilities[frame, episode.action[frame]] = 1

        #super(SupervisedNet, self).take_step(episode, frame, True, mean_vel, [0,0], False)
        return episode.velocity[frame]

    def construct_feed_dict(self, episode_in, frame, agent_frame,training=True, distracted=False):
        feed_dict = {}
        agent_pos=episode_in.agent[frame]#np.mean(episode_in.valid_people_tracks[episode_in.goal_person_id_val][agent_frame], axis=1)
        #print "Construct feed dict " + str(frame) + " " + str(agent_frame) + " " + str(agent_pos)
        feed_dict[self.state_in] = self.get_input(episode_in, agent_pos, frame_in=frame,
                                                  training=training)

        self.get_feature_vectors(agent_frame, episode_in, feed_dict, frame,distracted)
        feed_dict[self.sample]=episode_in.get_valid_vel(agent_frame, episode_in.goal_person_id_val)[1:]
        return feed_dict

    def get_feature_vectors(self, agent_frame, episode_in, feed_dict, frame, distracted=False):
        agent_pos =episode_in.agent[frame] #np.mean(episode_in.valid_people_tracks[episode_in.goal_person_id_val][agent_frame], axis=1)
        dim_p = self.get_dim_p()
        if self.settings.car_var:
            feed_dict[self.cars] = episode_in.get_input_cars_smooth(agent_pos, agent_frame,distracted)
        if self.settings.pose:
            itr = int( episode_in.agent_pose_frames[agent_frame])
            feed_dict[self.pose] = np.expand_dims(np.array(episode_in.agent_pose_hidden[itr, :]), axis=0)
            #print "Hidden layer: "+str(feed_dict[self.pose][0,:5])
        if self.settings.goal_dir:
            feed_dict[self.goal_dir] = episode_in.get_goal_dir_smooth(agent_pos, episode_in.goal)
            if self.settings.speed_input:
                feed_dict[self.time_alloted] = np.expand_dims(
                    [episode_in.get_time(agent_pos, episode_in.goal, agent_frame)], axis=0)


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
        return episode_in.velocity[frame][2]#episode_in.get_valid_vel(frame, episode_in.goal_person_id_val)[2]

    def get_memory_vel1_episode(self, frame, episode_in):
        return episode_in.velocity[frame][1]#episode_in.get_valid_vel(frame, episode_in.goal_person_id_val)[1]

    def train(self,ep_itr, statistics, episode_in, filename, filename_weights,poses, seq_len=-1):
        self.update_gradients()
        return statistics

    def evaluate(self,ep_itr, statistics, episode_in, poses, seq_len=-1):
        return statistics



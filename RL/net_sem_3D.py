
from net_3d import SimpleSoftMaxNet
from settings import NBR_MEASURES, NUM_SEM_CLASSES_ASINT,STATISTICS_INDX,NUM_SEM_CLASSES,PEDESTRIAN_MEASURES_INDX,RANDOM_SEED
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)
import numpy as np
from settings import CHANNELS
from dotmap import DotMap



labels=['unlabeled','ego vehicle' ,'rectification border' ,'out of roi' , 'static','dynamic','ground' ,'road' ,'sidewalk' ,
        'parking' ,'rail track','building' ,'wall' ,'fence', 'guard rail', 'bridge', 'tunnel' ,'pole', 'polegroup',
        'traffic light','traffic sign' ,'vegetation', 'terrain' ,'sky','person' ,'rider', 'car','truck' ,'bus',
        'caravan' ,'trailer', 'train' , 'motorcycle' ,'bicycle','license plate' ]
class Seg_3d(SimpleSoftMaxNet):
    def __init__(self, settings):
        self.channels = DotMap()
        self.channels.pedestrian_trajectory = 0
        self.channels.cars_trajectory = 1
        self.channels.pedestrians = 2
        self.channels.cars = 3
        self.channels.semantic = 4
        self.set_nbr_channels()
        super(Seg_3d, self).__init__(settings)


    def set_nbr_channels(self):
        self.nbr_channels = NUM_SEM_CLASSES_ASINT+4

    def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):
        tensor, people, cars,people_cv,cars_cv = episode_in.get_agent_neighbourhood(agent_pos_cur, self.settings.agent_s, frame_in)
        sem = np.zeros((self.settings.net_size[0], self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))
        segmentation = (tensor[:, :, :, CHANNELS.semantic] * NUM_SEM_CLASSES).astype(np.int)
        for x in range(sem.shape[0]):
            for y in range(sem.shape[1]):
                for z in range(sem.shape[2]):
                    if self.settings.predict_future:
                        # if training :
                        #     sem[x, y,z, 0] = tensor[ x, y,z, 4]
                        #     sem[x, y,z, 1] = tensor[ x, y,z, 5]
                        #if self.settings.predict_future:
                        sem[x, y, z, self.channels.pedestrian_trajectory] = people_cv[ x, y,z,0]
                        sem[x, y, z, self.channels.cars_trajectory] = cars_cv[ x, y,z,0]
                    elif training or self.settings.temporal:
                        sem[x, y, z, self.channels.pedestrian_trajectory] = tensor[x, y, z, CHANNELS.pedestrian_trajectory] * self.traj_forget_rate
                        sem[x, y, z, self.channels.cars_trajectory] = tensor[x, y, z, CHANNELS.cars_trajectory] * self.traj_forget_rate
                    sem[x, y, z, self.channels.pedestrians] = people[x, y, z, 0]
                    sem[x, y, z, self.channels.cars] = cars[x, y, z, 0]
                    if segmentation[ x, y,z]>0:
                        sem[x, y,z, segmentation[ x, y,z] + self.channels.semantic] = 1


        return np.expand_dims(sem, axis=0)


    def reset_mem(self):
        pass


class Seg_3d_RGB(SimpleSoftMaxNet):
    def __init__(self, settings):
        self.channels = DotMap()
        self.channels.rgb = [0, 3]
        self.channels.pedestrian_trajectory = 3
        self.channels.cars_trajectory = 4
        self.channels.pedestrians = 5
        self.channels.cars = 6
        self.channels.semantic = 7
        self.set_nbr_channels()
        super(Seg_3d_RGB, self).__init__(settings)


    def set_nbr_channels(self):
        self.nbr_channels = 7#+len(labels)

    def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):
        tensor , people, cars, people_cv, cars_cv= episode_in.get_agent_neighbourhood(agent_pos_cur, self.settings.agent_s, frame_in)
        sem = np.zeros(
            (self.settings.net_size[0], self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))
        segmentation = (tensor[:, :, :, 3] * NUM_SEM_CLASSES).astype(np.int)
        for x in range(sem.shape[0]):
            for y in range(sem.shape[1]):
                for z in range(sem.shape[2]):
                    sem[x, y, z, self.channels.rgb[0]] = tensor[x, y, z, CHANNELS.rgb[0]]
                    sem[x, y, z, self.channels.rgb[0]+1] = tensor[x, y, z, CHANNELS.rgb[0]+1]
                    sem[x, y, z, self.channels.rgb[0]+2] = tensor[x, y, z, CHANNELS.rgb[0]+2]
                    if self.settings.predict_future:
                        # if training:
                        #     sem[x, y, z, 3] = tensor[x, y, z, 4]
                        #     sem[x, y, z, 4] = tensor[x, y, z, 5]
                        # else:
                        sem[x, y, z, self.channels.cars_trajectory] = people_cv[ x, y,z,0]
                        sem[x, y, z, self.channels.cars_trajectory] =cars_cv[ x, y,z,0]
                    elif  training or self.settings.temporal:
                        sem[x, y, z, self.channels.cars_trajectory] = tensor[x, y, z, CHANNELS.pedestrian_trajectory] * self.traj_forget_rate
                        sem[x, y, z, self.channels.cars_trajectory] = tensor[x, y, z, CHANNELS.cars_trajectory] * self.traj_forget_rate
                    sem[x, y, z, self.channels.pedestrians] = people[x, y, z, 0]
                    sem[x, y, z, self.channels.cars] = cars[x, y, z, 0]
                    # if segmentation[x, y, z] > 0:
                    #     sem[x, y, z, segmentation[x, y, z] + 7] = 1

        return np.expand_dims(sem, axis=0)

    def reset_mem(self):
        pass

 # label_mapping[0] = 0 # None
 #    label_mapping[1] = 11 # Building
 #    label_mapping[2] = 13 # Fence
 #    label_mapping[3] = 4 # Other/Static
 #    label_mapping[4] = 24 # Pedestrian
 #    label_mapping[5] = 17 # Pole
 #    label_mapping[6] = 7 # RoadLines
 #    label_mapping[7] = 7 # Road
 #    label_mapping[8] = 8 # Sidewlk
 #    label_mapping[9] = 21 # Vegetation
 #    label_mapping[10] = 26 # Vehicles
 #    label_mapping[11] = 12 # Wall
 #    label_mapping[12] = 20 # Traffic sign
class Seg_3d_min(SimpleSoftMaxNet):
    def __init__(self, settings):
        self.labels_indx =  {11:0,13:1,14:2,4:2,5:2,15:2,16:2,17:3,18:3,7:4,9:4,6:4,10:4,8:5,21:6,22:6,12:7,20:8,19:8}
        self.carla=settings.carla
        self.channels = DotMap()
        self.channels.rgb = [0, 3]
        self.channels.pedestrian_trajectory = 3
        self.channels.cars_trajectory = 4
        self.channels.pedestrians = 5
        self.channels.cars = 6
        self.channels.semantic = 7

        self.set_nbr_channels()
        super(Seg_3d_min, self).__init__(settings)

    def set_nbr_channels(self):
        # if not self.carla:
        #     self.labels_indx = []
        #     labels_mini=['ground' ,'road' ,'sidewalk' ,'parking' ,'rail track','building' ,'wall' ,'fence','pole','traffic light','traffic sign' ,'vegetation', 'terrain' ]
        #     for i,label in enumerate(labels):
        #         if label in labels_mini:
        #             self.labels_indx.append(i)
        self.nbr_channels = 9+7

    def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):
        tensor, people, cars,people_cv, cars_cv  = episode_in.get_agent_neighbourhood(agent_pos_cur, self.settings.agent_s,frame_in)
        sem = np.zeros((self.settings.net_size[0], self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))
        segmentation = (tensor[:, :, :, CHANNELS.semantic] * NUM_SEM_CLASSES).astype(np.int)

        for x in range(sem.shape[0]):
            for y in range(sem.shape[1]):
                for z in range(sem.shape[2]):
                    sem[x, y, z, self.channels.rgb[0]]=tensor[x, y, z, CHANNELS.rgb[0]]
                    sem[x, y, z, self.channels.rgb[0]+1] = tensor[x, y, z, CHANNELS.rgb[0]+1]
                    sem[x, y, z, self.channels.rgb[0]+2] = tensor[x, y, z, CHANNELS.rgb[0]+2]
                    if self.settings.predict_future:
                        # if training:
                        #
                        #     sem[x, y,z, 3] = tensor[ x, y,z, 4]
                        #     sem[x, y,z, 4] = tensor[ x, y,z, 5]
                        # else:
                        sem[x, y, z, self.channels.pedestrian_trajectory] =people_cv[ x, y,z,0]
                        sem[x, y, z, self.channels.cars_trajectory] =cars_cv[ x, y,z,0]
                    elif training or self.settings.temporal:
                            sem[x, y,z, self.channels.pedestrian_trajectory] = tensor[ x, y,z, CHANNELS.pedestrian_trajectory]*self.traj_forget_rate
                            sem[x, y,z, self.channels.cars_trajectory] = tensor[ x, y,z, CHANNELS.cars_trajectory]*self.traj_forget_rate

                    sem[x, y, z, self.channels.pedestrians] = people[x, y, z,0]
                    sem[x, y, z, self.channels.cars] = cars[x, y, z, 0]
                    if segmentation[ x, y,z]in self.labels_indx:

                        sem[x, y,z, self.labels_indx[segmentation[ x, y,z]] + self.channels.semantic] = 1

        return np.expand_dims(sem, axis=0)


    def reset_mem(self):
        pass

class Seg_3d_min_pop(Seg_3d_min):
    def __init__(self, settings):
        #self.epsilon = settings.epsilon
        super(Seg_3d_min_pop, self).__init__(settings)

        #print((self.epsilon))

    def define_loss(self, dim_p):
        self.policy_old_log = tf.placeholder(shape=[1], dtype=tf.float32, name="policy_old_log")
        self.prob_log = tf.log(self.probabilities)
        self.sample = tf.placeholder(shape=[1], dtype=tf.int32, name="sample")
        self.responsible_output_log = tf.slice(self.prob_log, self.sample, [1]) + np.finfo(
            np.float32).eps
        self.ratio = tf.exp(self.responsible_output_log - self.policy_old_log)
        self.loss1 = -self.ratio * self.advantages
        self.loss2 = -tf.clip_by_value(self.ratio, 1 - self.epsilon, 1 + self.epsilon) * self.advantages
        self.entropy=- tf.reduce_sum(self.mu * tf.log(self.mu))

        self.R =tf.placeholder(shape=[1], dtype=tf.float32, name="R")
        vpred=self.val
        v_clipped=self.oldvpred+tf.clip_by_value(self.val - self.epsilon, -  self.epsilon,  self.epsilon)
        vf_losses1 = tf.square(vpred - self.R)
        # Clipped value
        vf_losses2 = tf.square(v_clipped - self.R)
        self.vf_loss=.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        self.loss = tf.reduce_mean(tf.maximum(self.loss1, self.loss2))- (self.settings.entr_par*self.entropy)+(self.settings.vf_par*self.vf_loss)
        return self.sample, self.loss



    # def construct_feed_dict(self, episode_in, frame,  training=True, agent_frame=-1):
    #     if agent_frame<0:
    #         agent_frame=frame
    #     feed_dict = super(Seg_3d_min, self).construct_feed_dict(episode_in, frame, training=training, agent_frame=agent_frame)
    #
    #     # feed_dict[self.policy_old_log] = np.reshape(np.log(
    #     #     episode_in.probabilities[max(agent_frame - 2,0), episode_in.action[max(agent_frame - 1, 0)]]), (1,))
    #     return feed_dict

        # self, agent_action, agent_measures, agent_pos, agent_velocity, ep_itr, episode_in, frame,
        # reward, statistics, agent_speed = None, training = True, agent_frame = -1):

    def grad_feed_dict(self, agent_action, agent_measures, agent_pos, agent_velocity, ep_itr, episode_in, frame,
                       reward, statistics,poses,priors, agent_speed=None, training=True, agent_frame=-1, statistics_car=None):

        grad_feed_dict = super(Seg_3d_min_pop, self).grad_feed_dict(agent_action, agent_measures, agent_pos, agent_velocity, ep_itr, episode_in, frame,
                               reward, statistics,poses, priors, agent_speed=agent_speed, training=training, agent_frame=agent_frame)
        if agent_frame < 0:
            agent_frame = frame
        probabilities = statistics[ep_itr, max(agent_frame - 1, 0), STATISTICS_INDX.probabilities[0]:STATISTICS_INDX.probabilities[1]]
        sample = int(agent_action[ep_itr, agent_frame])
        grad_feed_dict[self.policy_old_log] = np.reshape(np.log(probabilities[sample]), (1,))
        return grad_feed_dict

class Seg_3d_min_still_actions(Seg_3d_min):
    def __init__(self, settings):
        self.set_nbr_channels()
        super(Seg_3d_min_still_actions, self).__init__(settings)
        self.actions = [[0,1,0],[0,-1,0], [0,1,1],[0,-1,1],[0,0,0]]
        v = [-1, 0, 1]

        # j = 0
        # if self.settings.actions_3d:
        #     for z in range(3):
        #         for y in range(3):
        #             for x in range(3):
        #                 self.actions.append([v[x], v[y], v[z]])
        #                 j += 1
        #else:
        # for y in range(3):
        #     for x in range(3):
        #         self.actions.append([0, v[y], v[x]])
        #         j += 1
        #j = 0

    def fully_connected_size(self, dim_p):
        return 5



class Seg_3d_no_sem(SimpleSoftMaxNet):
    def __init__(self, settings):
        self.channels = DotMap()
        self.channels.rgb = [0, 3]
        self.channels.pedestrian_trajectory = 3
        self.channels.cars_trajectory = 4
        self.channels.pedestrians = 5
        self.channels.cars = 6
        self.channels.pedestrians = 7
        self.set_nbr_channels()
        super(Seg_3d_no_sem, self).__init__(settings)


    def set_nbr_channels(self):
        self.nbr_channels = 7

    def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):
        tensor, people, cars, people_cv, cars_cv = episode_in.get_agent_neighbourhood(agent_pos_cur, self.settings.agent_s, frame_in)
        sem = np.zeros((self.settings.net_size[0], self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))
        sem[:,:,:,self.channels.rgb[0]:self.channels.rgb[1]]=tensor[:,:,:,CHANNELS.rgb[0]:CHANNELS.rgb[1]]
        # if not training:
        #     sem[:,:,:,3]=people[:,:,:,0]
        #     sem[:, :, :, 4] = cars[:,:,:,0]
        if self.settings.predict_future:
            # if training:
            #     sem[:, :,:, 3] = tensor[:, :, :, 4]
            #     sem[:, :,:, 4] = tensor[:, :, :, 5]
            # else:
            sem[:, :,:, self.channels.pedestrian_trajectory] = people_cv
            sem[:, :,:, self.channels.cars_trajectory] = cars_cv
        elif training or self.settings.temporal:
                sem[:, :,:, self.channels.pedestrian_trajectory] = self.traj_forget_rate *tensor[:, :, :, CHANNELS.pedestrian_trajectory]
                sem[:, :,:, self.channels.cars_trajectory] = self.traj_forget_rate *tensor[:, :, :, CHANNELS.cars_trajectory]


        sem[:, :,:, self.channels.pedestrians] = people[:, :, :, 0]
        sem[:, :,:, self.channels.pedestrians] =cars[:, :, :, 0]


        return np.expand_dims(sem, axis=0)


    def reset_mem(self):
        pass

class Seg_3d_no_vis(SimpleSoftMaxNet):
    def __init__(self, settings):

        self.set_nbr_channels()
        super(Seg_3d_no_vis, self).__init__(settings)

    def define_conv_layers(self):
        return
        #tf.reset_default_graph()

    def construct_feed_dict(self, episode_in, frame, agent_frame,training=True, distracted=False):

        feed_dict ={}

        if self.settings.car_var:
            feed_dict[self.cars] = episode_in.get_input_cars( episode_in.agent[agent_frame],frame, training, distracted)


        if len(self.settings.sem_class) > 0:
            feed_dict[self.sem_class]=episode_in.get_sem_class()

        if self.settings.goal_dir:
            feed_dict[self.goal_dir] = episode_in.get_goal_dir(episode_in.agent[agent_frame], episode_in.goal)

        if self.settings.action_mem:
            values = np.zeros((self.fully_connected_size(0)+1)*self.settings.action_mem)
            for past_frame in range(1, self.settings.action_mem+1):
                if agent_frame-past_frame>=0:

                    values[(past_frame-1)*(self.fully_connected_size(0)+1)+int(episode_in.action[max(agent_frame - past_frame, 0)])] = 1
                    values[past_frame*past_frame-1] = episode_in.measures[max(agent_frame - past_frame, 0), 3]
            feed_dict[self.action_mem] = np.expand_dims(values, axis=0)
        return feed_dict

    def grad_feed_dict(self,agent_action, agent_measures, agent_pos, agent_velocity, ep_itr, episode_in, frame,
                       reward, statistics,poses,priors, agent_speed=None, training=True, agent_frame=-1, statistics_car=None):
        if agent_frame < 0 or not training:
            agent_frame = frame
        #print "Gradient feed dict " +str(frame)+" " + str(agent_frame)
        r=reward[ep_itr, agent_frame]
        if sum(agent_measures[ep_itr, :agent_frame, 0])>0:
            r=0
        feed_dict = {self.advantages: r,
                     self.sample: [agent_action[ep_itr, agent_frame]]}
        if self.settings.controller_len>0:
            if np.isnan(agent_speed[ep_itr, agent_frame]):
                feed_dict[self.sample_c] = [0]
            else:
                feed_dict[self.sample_c]=[int(agent_speed[ep_itr, agent_frame])]
        if self.writer:
            feed_dict+={
                     self.num_cars: agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.hit_by_car].copy(),
                     self.num_people: agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory].copy(),
                     self.pavement: agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.iou_pavement].copy(),
                     self.num_of_obj: agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles].copy(),
                     self.dist_travelled: agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init].copy(),
                     self.out_of_axis: agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.out_of_axis].copy(),
                     self.tot_reward: np.sum(statistics[ep_itr, :agent_frame, STATISTICS_INDX.reward]) #  self.pose: poses,
                     }
        if self.settings.car_var:
            feed_dict[self.cars] = episode_in.get_input_cars( agent_pos[ep_itr, agent_frame, :],frame, training,agent_measures[ep_itr, agent_frame, PEDESTRIAN_MEASURES_INDX.distracted].copy())
        if len(self.settings.sem_class)>0:
            feed_dict[self.sem_class]=np.zeros((1,9))
            if statistics[ep_itr, 2, STATISTICS_INDX.goal]>0:
                feed_dict[self.sem_class][0,self.settings.sem_dict[int(statistics[ep_itr, 2, STATISTICS_INDX.goal])]]=1
        if self.settings.goal_dir:
            feed_dict[self.goal_dir] = np.zeros((1, 10))
            goal=[0,0,0]
            goal[1]=statistics[ep_itr, 3, STATISTICS_INDX.goal]
            goal[2] = statistics[ep_itr, 4, STATISTICS_INDX.goal]
            feed_dict[self.goal_dir] = episode_in.get_goal_dir(agent_pos[ep_itr, agent_frame, :], goal)
        if self.settings.action_mem:
            values = np.zeros((self.fully_connected_size(0)+1)*self.settings.action_mem)
            for past_frame in range(1, self.settings.action_mem+1):
                if agent_frame -past_frame>=0:
                    values[(past_frame-1)*(self.fully_connected_size(0)+1)+int(agent_action[ep_itr,max(agent_frame - past_frame, 0)])] = 1
                    values[past_frame*past_frame-1] = agent_measures[ep_itr,max(agent_frame - past_frame, 0), PEDESTRIAN_MEASURES_INDX.hit_obstacles]
            feed_dict[self.action_mem] = np.expand_dims(values, axis=0)
        return feed_dict

    def reset_mem(self):
        pass



#
# class SupervisedNet_3D(SupervisedNet):
#     def __init__(self, settings):
#         self.labels_indx =  {11:0,13:1,14:2,4:2,5:2,15:2,16:2,17:3,18:3,7:4,9:4,6:4,10:4,8:5,21:6,22:6,12:7,20:8,19:8}
#         self.carla=settings.carla
#         self.set_nbr_channels()
#         super(SupervisedNet_3D, self).__init__(settings)
#
#     def set_nbr_channels(self):
#
#         self.nbr_channels = 9+7
#
#     def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):
#         tensor, people, cars, people_cv, cars_cv = episode_in.get_agent_neighbourhood(agent_pos_cur, self.settings.agent_s,frame_in)
#         sem = np.zeros((self.settings.net_size[0], self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))
#         segmentation = (tensor[:, :, :, 3] * NUM_SEM_CLASSES).astype(np.int)
#
#         for x in range(sem.shape[0]):
#             for y in range(sem.shape[1]):
#                 for z in range(sem.shape[2]):
#                     sem[x, y, z, 0]=tensor[x, y, z, 0]
#                     sem[x, y, z, 1] = tensor[x, y, z, 1]
#                     sem[x, y, z, 2] = tensor[x, y, z, 2]
#                     if self.settings.predict_future:
#                         # if training:
#                         #     sem[x, y,z, 3] = tensor[ x, y,z, 4]
#                         #     sem[x, y,z, 4] = tensor[ x, y,z, 5]
#                         # else:
#                         sem[x, y, z, 3] = people_cv[x, y, z,0]
#                         sem[x, y, z, 4] = cars_cv[x, y, z,0]
#                     elif training or self.settings.temporal:
#                         sem[x, y, z, 3] = tensor[x, y, z, 4] * self.traj_forget_rate
#                         sem[x, y, z, 4] = tensor[x, y, z, 5] * self.traj_forget_rate
#                     sem[x, y, z, 5] = people[x, y, z,0]
#                     sem[x, y, z, 6] = cars[x, y, z, 0]
#                     if segmentation[ x, y,z]in self.labels_indx:
#
#                         sem[x, y,z, self.labels_indx[segmentation[ x, y,z]] + 7] = 1
#
#         return np.expand_dims(sem, axis=0)
#
#
#     def reset_mem(self):
#         pass


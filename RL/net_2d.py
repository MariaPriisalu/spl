import numpy as np
import tensorflow as tf
from settings import RANDOM_SEED
tf.set_random_seed(RANDOM_SEED)
from net import Net, variable_summaries
from net_continous import GaussianNet, Angular_Net
from nets_abstract import SoftMaxNet
from settings import CHANNELS


class Net_2d(Net):

    def __init__(self, settings):
        self.set_nbr_channels()

        super(Net_2d, self).__init__(settings)

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
        #tf.reset_default_graph()

        self.state_in = tf.compat.v1.placeholder(shape=[ self.settings.batch_size, self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels],
                                       dtype=self.DTYPE,
                                       name="reconstruction")
        #mean = tf.constant(self.mean, dtype=self.DTYPE)  # (3)

        prev_layer = self.state_in  # - mean
        variable_summaries(prev_layer, name='input', summaries=self.conv_summaries)
        with tf.compat.v1.variable_scope('conv1') as scope:
            out_filters = self.settings.outfilters[0]
            print("Define 2D weights")
            kernel1 = tf.get_variable('weights', [ 3, 3, self.nbr_channels, out_filters], self.DTYPE,
                                      initializer=tf.contrib.layers.xavier_initializer())

            self.conv_out1 = tf.nn.conv2d(prev_layer, kernel1, [1, 1, 1, 1], padding='SAME')  # [1, 2, 2, 2, 1]
            if self.settings.conv_bias:
                biases1 = self.bias_variable('biases', [out_filters])

                self.bias1 =tf.nn.bias_add(self.conv_out1, biases1)
            else:
                self.bias1=self.conv_out1
            self.conv1 =tf.nn.relu(self.bias1, name=scope.name)

        prev_layer = self.conv1
        if 1 in self.settings.pooling:
            pooled = tf.nn.max_pool(prev_layer, [1, 2, 2, 1], [1, 1, 1, 1],'SAME')#, 'SAME')

            prev_layer=pooled

        in_filters = out_filters
        if self.settings.num_layers==2:
            with tf.compat.v1.variable_scope('conv2') as scope:
                out_filters = self.settings.outfilters[1]
                kernel2 = tf.get_variable('weights_conv2', [2,2,  in_filters, out_filters], self.DTYPE,
                                          initializer=tf.contrib.layers.xavier_initializer())

                self.conv_out2 = tf.nn.conv2d(prev_layer, kernel2, [1, 1, 1, 1], padding='SAME')  # [1, 2, 2, 2, 1]
                if self.settings.conv_bias:
                    biases2 = self.bias_variable('biases', [out_filters])

                    self.bias2 = tf.nn.bias_add(self.conv_out2, biases2)
                else:
                    self.bias2=self.conv_out2

                self.conv2 = tf.nn.relu(self.bias2, name=scope.name)
                #variable_summaries(biases2, name='conv2b', summaries=self.conv_summaries)
                variable_summaries(kernel2, name='conv2k', summaries=self.conv_summaries)
                variable_summaries(self.conv2, name='conv2_out', summaries=self.conv_summaries)
            prev_layer = self.conv2
        if 2 in self.settings.pooling:
            pooled2 = tf.nn.max_pool(prev_layer, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')#, 'SAME')
            variable_summaries(pooled2, name='pooled', summaries=self.conv_summaries)
            prev_layer=pooled2
        self.conv_output = prev_layer

        return prev_layer


    def get_input(self, episode_in, agent_pos_cur, frame_in=-1,training=True):
        tensor,people,cars, people_cv, cars_cv= episode_in.get_agent_neighbourhood(agent_pos_cur, self.settings.agent_s, frame_in) # TODO!
        img_from_above = np.zeros((tensor.shape[0], tensor.shape[1], ), dtype=np.float)
        if self.settings.predict_future:
            if not training:
                img_from_above=tensor[:,:,:]
                img_from_above[:,:,CHANNELS.pedestrian_trajectory]=people[:,:,:,0]
                img_from_above[:, :, CHANNELS.cars_trajectory] = cars[ :, :, :, 0]
            else:
                img_from_above = tensor[ :, :, :]
                img_from_above[:, :, CHANNELS.pedestrian_trajectory] = img_from_above[ :, :, CHANNELS.pedestrian_trajectory]
                img_from_above[:,  :, CHANNELS.cars_trajectory] = img_from_above[ :, :, CHANNELS.cars_trajectory]
        elif training or self.settings.temporal:
            img_from_above = tensor[ :, :, :]
            print(("Forgetting rate: "+str( self.traj_forget_rate)))
            img_from_above[ :, :, CHANNELS.pedestrian_trajectory] = self.traj_forget_rate * img_from_above[ :, :, CHANNELS.pedestrian_trajectory]
            img_from_above[ :, :, CHANNELS.cars_trajectory] = self.traj_forget_rate * img_from_above[ :, :, CHANNELS.cars_trajectory]

        return np.expand_dims(img_from_above, axis=0)



    def return_highest_object(self, tens):
        z = 0
        while z < tens.shape[0] - 1 and np.linalg.norm(tens[z]) == 0:
            z = z + 1
        return z

    def return_highest_col(self, tens):
        z = 0
        while z < tens.shape[0] - 1 and np.linalg.norm(tens[z, :]) == 0:
            z = z + 1
        return z

    def fully_connected(self, dim_p, prev_layer):
        super(Net_2d, self).fully_connected(dim_p, prev_layer)

class SimpleSoftMaxNet_2D(SoftMaxNet, Net_2d):
    def __init__(self, settings):
        super(SimpleSoftMaxNet_2D, self).__init__(settings)

            # def get_pose(self, poses, frame):
class ContinousNet_2D(GaussianNet, Net_2d):
    def __init__(self, settings):
        super(ContinousNet_2D, self).__init__(settings)

class ContinousNet_2D_angular(Angular_Net, Net_2d):
    def __init__(self, settings):
        super(ContinousNet_2D_angular, self).__init__(settings)

            # def get_pose(self, poses, frame):


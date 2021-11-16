import numpy as np
import tensorflow as tf
from settings import RANDOM_SEED
tf.set_random_seed(RANDOM_SEED)

from nets_abstract import SoftMaxNet
from net import Net
from settings import CHANNELS

class Net_3D(Net):

    def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):
        input, people,cars, people_cv, cars_cv= episode_in.get_agent_neighbourhood(agent_pos_cur, self.settings.agent_s,frame_in, training)
        if self.settings.predict_future:
            if not training:
                input[:,:,:,CHANNELS.pedestrian_trajectory]=people
                input[:, :, :, CHANNELS.cars_trajectory] = cars
            else:
                input[:, :, :, CHANNELS.pedestrian_trajectory] = input[:, :, :, CHANNELS.pedestrian_trajectory]
                input[:, :, :, CHANNELS.cars_trajectory] = input[:, :, :, CHANNELS.cars_trajectory]
        else:
            if not training:
                input[:,:,:,CHANNELS.pedestrian_trajectory]=people
                input[:, :, :, CHANNELS.cars_trajectory] = cars
            else:
                input[:, :, :, CHANNELS.pedestrian_trajectory] = self.traj_forget_rate * input[:, :, :, CHANNELS.pedestrian_trajectory]
                input[:, :, :, CHANNELS.cars_trajectory] = self.traj_forget_rate * input[:, :, :, CHANNELS.cars_trajectory]

        return np.expand_dims(
            input, axis=0)


    def define_update_gradients(self,  optimizer):
        if self.settings.batch_normalize:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.settings.update_batch = optimizer.apply_gradients(list(zip( self.gradient_holders, self.tvars)))
        else:
            self.settings.update_batch = optimizer.apply_gradients(list(zip(self.gradient_holders, self.tvars)))

    def merge_summaries(self):
        self.train_summaries.extend(self.conv_summaries)
        self.train_summaries = tf.summary.merge(self.train_summaries)
        self.loss_summaries = tf.summary.merge(self.loss_summaries)
        self.grad_summaries = tf.summary.merge(self.grad_summaries)

    def define_conv_layers(self):
        #tf.reset_default_graph()
        # with tf.device('/gpu:0'):
        self.state_in = tf.compat.v1.placeholder(shape=[self.settings.batch_size, self.settings.net_size[0], self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels],
                                       dtype=self.DTYPE,
                                       name="reconstruction")
        mean = tf.constant(self.mean, dtype=self.DTYPE)  # (3)
        #mean = tf.reshape(mean, [1, 1, 1, 1, self.nbr_channels])
        prev_layer = self.state_in  # - mean
        if self.writer:
            variable_summaries(prev_layer, name='input', summaries=self.conv_summaries)
        with tf.compat.v1.variable_scope('conv1') as scope:
            out_filters = self.settings.outfilters[0]
            kernel1 = tf.compat.v1.variable_scope('weights_conv1', [3,3,3, self.nbr_channels, out_filters], self.DTYPE,
                                      initializer=tf.contrib.layers.xavier_initializer())

            conv_out1 = tf.nn.conv3d(prev_layer, kernel1, [1, 1, 1, 1, 1], padding='SAME')  # [1, 2, 2, 2, 1]
            biases1 = self.bias_variable('biases', [out_filters])

            bias1 = tf.nn.bias_add(conv_out1, biases1)
            conv1 = tf.nn.relu(bias1, name=scope.name)
            if self.writer:
                variable_summaries(biases1, name='conv1b', summaries=self.conv_summaries)
                variable_summaries(kernel1, name='conv1', summaries=self.conv_summaries)
                variable_summaries(conv1, name='conv1_out', summaries=self.conv_summaries)
        prev_layer = conv1
        if 1 in self.settings.pooling:
            with tf.compat.v1.variable_scope('pooling1') as scope:
                pooled1 = tf.nn.max_pool3d(prev_layer, [1, 2, 2, 2, 1], [1, 1, 1, 1, 1], 'SAME')
                if self.writer:
                    variable_summaries(pooled1, name=scope.name, summaries=self.conv_summaries)
            prev_layer = pooled1
        in_filters = out_filters

        if self.settings.num_layers == 2:
            with tf.compat.v1.variable_scope('conv2') as scope:
                out_filters = self.settings.outfilters[1]
                kernel2 = tf.compat.v1.variable_scope('weights_conv2', [2, 2, 2, in_filters, out_filters], self.DTYPE,
                                          initializer=tf.contrib.layers.xavier_initializer())

                conv_out2 = tf.nn.conv3d(prev_layer, kernel2, [1, 1, 1, 1, 1], padding='SAME')  # [1, 2, 2, 2, 1]
                biases2 = self.bias_variable('biases', [out_filters])

                bias2 = tf.nn.bias_add(conv_out2, biases2)
                # if self.batch_normalize:
                #     h2 = tf.contrib.layers.batch_norm(bias2,center=True, scale=True,is_training=True)
                # else:
                #     h2=bias2
                conv2 = tf.nn.relu(bias2, name=scope.name)
                # variable_summaries(biases2, name='conv2b', summaries=self.conv_summaries)
                # variable_summaries(kernel2, name='conv2k', summaries=self.conv_summaries)
                # variable_summaries(conv2, name='conv2_out', summaries=self.conv_summaries)
            prev_layer = conv2
        if 2 in self.settings.pooling:
            with tf.compat.v1.variable_scope('pooling2') as scope:
                pooled2 = tf.nn.max_pool3d(prev_layer, [1, 2, 2, 2, 1], [1, 1, 1, 1, 1], 'SAME')
                #variable_summaries(pooled2, name=scope.name, summaries=self.conv_summaries)
            prev_layer = pooled2
        # if self.layers == 3:
        #     prev_layer = conv2
        #     in_filters = out_filters
        #     with tf.compat.v1.variable_scope('conv3') as scope:
        #         out_filters = 64
        #         kernel = tf.compat.v1.variable_scope('weights_conv3', [5, 5, 5, in_filters, out_filters], self.DTYPE,
        #                                  initializer=tf.contrib.layers.xavier_initializer())  # batch_size, depth,height,width, channels
        #         conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        #         biases = self.bias_variable('biases', [out_filters])
        #         bias = tf.nn.bias_add(conv, biases)
        #         conv3 = tf.nn.relu(bias, name=scope.name)
        #     in_filters = out_filters
        self.conv_output = prev_layer
        return prev_layer

class SimpleSoftMaxNet(SoftMaxNet, Net_3D):
    def __init__(self, settings):
        super(SimpleSoftMaxNet, self).__init__(settings)

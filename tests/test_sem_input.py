import unittest
import numpy as np

# import sys
# sys.path.append("RL/")
from RL.episode import SimpleEpisode
from RL.agent import SimplifiedAgent
from RL.settings import run_settings
from RL.net_sem_2d import Seg_2d_softmax
from RL.agent_net import NetAgent



class TestNet(unittest.TestCase):

    # Help function. Setup for tests.
    def initialize_episode(self, cars, gamma, people, pos_x, pos_y, tensor,seq_len=30, rewards=(1, 1, 1, 1, 1, 1),  width=0):
        #  tensor, people_e, cars_e, pos_x, pos_y, gamma, seq_len, reward_weights
        setup = run_settings()
        setup.run_2D = True
        episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, seq_len, rewards,
                                agent_size=[width, width, width], adjust_first_frame=False, run_2D=True)


        setup.shape_s=[0,0,0]
        setup.agent_shape=[width, width, width]
        setup.net_size=[1+2*width,1+2*width,1+2*width]
        setup.agent_s=[width, width, width]
        #tf.reset_default_graph()
        net = Seg_2d_softmax(setup)
        agent=SimplifiedAgent(setup)
        agent = NetAgent( setup,net, None)

        return agent, episode, net

    def initialize_tensor(self, seq_len, size=3):
        tensor = np.zeros((size,size, size, 6))
        people = []
        cars = []
        for i in range(seq_len):
            people.append([])
            cars.append([])
        gamma = 0.99
        pos_x = 0
        pos_y = 0
        return cars, gamma, people, pos_x, pos_y, tensor

    def initialize_pos(self, agent, episode):
        pos, i , vel= episode.initial_position(None)

        agent.initial_position(pos, episode.goal)

    # Test correct empty initialization.
    def test_walk_into_objs(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(11)
        tensor[1,1,1,3]=8/33.0 # pavement
        tensor[2, 1, 2, 3] = 11.0 / 33.0 # person/obj?
        agent, episode, net=self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=11, rewards=(0, 0, 0, 1, 0, 0), width=0)
        episode.agent_size = [0, 0, 0]
        self.initialize_pos(agent, episode)
        tensor=net.get_input(episode, [1,1.2,.9])
        expected=np.zeros((1,1,1,40))
        expected[:,:,:,8+7]=1
        #expected[:, :, :, 2] = 1
        np.testing.assert_array_equal(tensor, expected)

        tensor = net.get_input(episode, [0,.9,2.1])
        expected = np.zeros((1, 1, 1, 40))
        expected[:, :, :, 11+7] = 1
        #expected[:, :, :, 2] = 1
        np.testing.assert_array_equal(tensor, expected)

        # Test correct empty initialization.

    def test_walk_into_objs2(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(11)
        tensor[1, 1, 1, 3] = 8 / 33.0
        tensor[2, 1, 2, 3] = 11.0 / 33.0
        agent, episode, net = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=11,
                                                      width=1)
        episode.agent_size = [1, 1, 1]
        self.initialize_pos(agent, episode)
        tensor = net.get_input(episode, [1, 0.8, 1.1])
        expected = np.zeros((1, 3, 3, 40))
        expected[:, 1, 1, 8+7] = 1
        expected[0, 1, 2, 11+7] = 1
        #expected[:, :, :, 2] = 1
        np.testing.assert_array_equal(tensor, expected)

        tensor = net.get_input(episode, [0, 1.1, 2+1e-15])
        expected = np.zeros((1, 3, 3, 40))
        expected[:, 1, 1, 11+7] = 1
        #expected[:, :, :, 2] = 1
        expected[:, 1, 0, 8+7] = 1
        np.testing.assert_array_equal(tensor, expected)

import unittest
import numpy as np
import sys
sys.path.append("RL/")
from RL.episode import SimpleEpisode
from RL.agent import SimplifiedAgent
from RL.settings import run_settings, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX

class TestEnv(unittest.TestCase):
    def get_reward(self, all_zeros=False):

        rewards = np.zeros(NBR_REWARD_WEIGHTS)
        if not all_zeros:
            rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
            rewards[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] = 1
            rewards[PEDESTRIAN_REWARD_INDX.on_pavement] = 1
            rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
            rewards[PEDESTRIAN_REWARD_INDX.distance_travelled] = 1
            rewards[PEDESTRIAN_REWARD_INDX.out_of_axis] = -1
            rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
            rewards[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] = 1
        return rewards

    # Help function. Setup for tests.
    def initialize_episode(self, cars, gamma, people, pos_x, pos_y, tensor,seq_len=30, rewards=np.ones(NBR_REWARD_WEIGHTS), agent_size=(0,0,0),people_dict={}, init_frames={}):
        #  tensor, people_e, cars_e, pos_x, pos_y, gamma, seq_len, reward_weights
        setup = run_settings()
        setup.useHeroCar = False
        setup.useRLToyCar = False
        episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, seq_len, rewards, agent_size=agent_size,
                                people_dict=people_dict, init_frames=init_frames, adjust_first_frame=False,
                                velocity_actions=False, defaultSettings=setup)
        episode.action=np.zeros(len(episode.action))
        agent=SimplifiedAgent(setup)

        return agent, episode

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
        pos, i,vel = episode.initial_position(None, initialization=8)

        agent.initial_position(pos,episode.goal, vel=vel)

    # Test correct empty initialization.
    def test_walk_into_objs(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(11)
        tensor[1,1,1,3]=8/33.0
        tensor[2, 1, 2, 3] = 11.0 / 33.0
        rewards=self.get_reward(True)

        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects]=-1
        agent, episode=self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=11, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        self.initialize_pos(agent, episode)

        np.testing.assert_array_equal(episode.agent[0], [1,1,1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])

        # Invalid move
        agent.perform_action([0,0,1], episode)
        episode.action[0] = 5
        np.testing.assert_array_equal(episode.agent[1], [1,1,1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(0), -1)

        # Invalid move
        agent.perform_action([0, 0, 1], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], [1, 1, 1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(1), -1)

        # Valid move
        agent.perform_action([0, 1, 0], episode)
        episode.action[2] = 5
        np.testing.assert_array_equal(episode.agent[3], [1, 2, 1])
        np.testing.assert_array_equal(agent.position, [1, 2, 1])
        np.testing.assert_array_equal(episode.calculate_reward(2), 0)

        # Valid move
        agent.perform_action([0, 0, 1], episode)
        episode.action[3] = 5
        np.testing.assert_array_equal(episode.agent[4], [1, 2, 2])
        np.testing.assert_array_equal(agent.position, [1, 2, 2])
        np.testing.assert_array_equal(episode.calculate_reward(3), 0)

        # Invalid move
        agent.perform_action([0, -1, 0], episode)
        episode.action[4] = 5
        np.testing.assert_array_equal(episode.agent[5], [1, 2, 2])
        np.testing.assert_array_equal(agent.position, [1, 2, 2])
        np.testing.assert_array_equal(episode.calculate_reward(4), -1)

        # Valid move
        agent.perform_action([0, 0, -1], episode)
        episode.action[5] = 5
        np.testing.assert_array_equal(episode.agent[6], [1, 2, 1])
        np.testing.assert_array_equal(agent.position, [1, 2, 1])
        np.testing.assert_array_equal(episode.calculate_reward(5), 0)

        # Valid move
        agent.perform_action([0, -1,0], episode)
        episode.action[6] = 5
        np.testing.assert_array_equal(episode.agent[7], [1, 1, 1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(6), 0)

        # Valid move
        agent.perform_action([0, -1, 0], episode)
        episode.action[7] = 5
        np.testing.assert_array_equal(episode.agent[8], [1, 0, 1])
        np.testing.assert_array_equal(agent.position, [1, 0, 1])
        np.testing.assert_array_equal(episode.calculate_reward(7), 0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[8] = 5
        np.testing.assert_array_equal(episode.agent[9], [1, 0, 2])
        np.testing.assert_array_equal(agent.position, [1, 0, 2])
        np.testing.assert_array_equal(episode.calculate_reward(8), 0)

        agent.perform_action([0, 1, 0], episode)
        episode.action[9] = 5
        np.testing.assert_array_equal(episode.agent[10], [1, 0, 2])
        np.testing.assert_array_equal(agent.position, [1, 0, 2])

    # Test correct empty initialization.
    def test_walk_into_objs_varied_vel(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(11)
        tensor[1, 1, 1, 3] = 8 / 33.0
        tensor[2, 1, 2, 3] = 11.0 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=11,
                                                 rewards=rewards)
        episode.agent_size = [0, 0, 0]
        self.initialize_pos(agent, episode)

        np.testing.assert_array_equal(episode.agent[0], [1, 1, 1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])

        # Invalid move
        agent.perform_action([0, 0, 0.9], episode)
        episode.action[0] = 5
        np.testing.assert_array_equal(episode.agent[1], [1, 1, 1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(0), -1)

        # Invalid move
        agent.perform_action([0, 0, 0.7], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], [1, 1, 1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(1), -1)

        # Valid move
        agent.perform_action([0, 0.8, 0.1], episode)
        episode.action[2] = 5
        #np.testing.assert_array_equal(episode.agent[3], [1, 2, 1])
        np.testing.assert_array_almost_equal(episode.agent[3], [1, 1.8, 1.1])
        np.testing.assert_array_equal(agent.position, [1, 2, 1])
        np.testing.assert_array_equal(episode.calculate_reward(2), 0)

        # Valid move
        agent.perform_action([0, 0.1, 1.2], episode)
        episode.action[3] = 5
        #np.testing.assert_array_equal(episode.agent[4], [1, 2, 2])
        np.testing.assert_array_almost_equal(episode.agent[4], [1, 1.9, 2.3])
        np.testing.assert_array_equal(agent.position, [1, 2, 2])
        np.testing.assert_array_equal(episode.calculate_reward(3), 0)

        # Invalid move
        agent.perform_action([0, -1.3, 0], episode)
        episode.action[4] = 5
        #np.testing.assert_array_equal(episode.agent[5], [1, 2, 2])
        np.testing.assert_array_almost_equal(episode.agent[5], [1, 1.9, 2.3])
        np.testing.assert_array_equal(agent.position, [1, 2, 2])
        np.testing.assert_array_equal(episode.calculate_reward(4), -1)

        # Valid move
        agent.perform_action([0, 0, -1.3], episode)
        episode.action[5] = 5
        #np.testing.assert_array_equal(episode.agent[6], [1, 2, 1])
        np.testing.assert_array_almost_equal(episode.agent[6], [1, 1.9, 1.0])
        np.testing.assert_array_equal(agent.position, [1, 2, 1])
        np.testing.assert_array_equal(episode.calculate_reward(5), 0)

        # Valid move
        agent.perform_action([0, -1.4, 0], episode)
        episode.action[6] = 5
        #np.testing.assert_array_equal(episode.agent[7], [1, 1, 1])
        np.testing.assert_array_almost_equal(episode.agent[7], [1, .5, 1.0])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(6), 0)

        # Valid move
        agent.perform_action([0, -.7, 0], episode)
        episode.action[7] = 5
        #np.testing.assert_array_equal(episode.agent[8], [1, 0, 1])
        np.testing.assert_array_almost_equal(episode.agent[8], [1, -.2, 1.0])
        np.testing.assert_array_equal(agent.position, [1, 0, 1])
        np.testing.assert_array_equal(episode.calculate_reward(7), 0)

        agent.perform_action([0, 0, 1.1], episode)
        episode.action[8] = 5
        #np.testing.assert_array_equal(episode.agent[9], [1, 0, 2])
        np.testing.assert_array_almost_equal(episode.agent[9], [1, -.2, 2.1])
        np.testing.assert_array_equal(agent.position, [1, 0, 2])
        np.testing.assert_array_equal(episode.calculate_reward(8), 0)

        agent.perform_action([0, .7, 0], episode)
        episode.action[9] = 5
        #np.testing.assert_array_equal(episode.agent[10], [1, 0, 2])
        np.testing.assert_array_almost_equal(episode.agent[10], [1, -.2, 2.1])
        np.testing.assert_array_equal(agent.position, [1, 0, 2])

    # Test correct empty initialization.
    def test_walk_into_objs_wide(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(14, size=7)
        tensor[1, 3, 3, 3] = 8 / 33.0
        tensor[2, 3, 5, 3] = 11.0 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=14, rewards=rewards)
        episode.agent_size = [0, 1, 1]

        self.initialize_pos(agent, episode)
        np.testing.assert_array_equal(episode.agent[0], [1, 3, 3])
        np.testing.assert_array_equal(agent.position, [1, 3, 3])

        # Invalid move
        agent.perform_action([0, 0, 1], episode)
        episode.action[0] = 5
        np.testing.assert_array_equal(episode.agent[1], [1, 3, 3])
        np.testing.assert_array_equal(agent.position, [1, 3, 3])
        np.testing.assert_array_equal(episode.calculate_reward(0), -1)

        # Invalid move
        agent.perform_action([0, 0, 1], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], [1, 3, 3])
        np.testing.assert_array_equal(agent.position, [1, 3, 3])
        np.testing.assert_array_equal(episode.calculate_reward(1), -1.0)

        agent.perform_action([0, 1, 0], episode)
        episode.action[2] = 5
        np.testing.assert_array_equal(episode.agent[3], [1, 4, 3])
        np.testing.assert_array_equal(agent.position, [1, 4, 3])
        np.testing.assert_array_equal(episode.calculate_reward(2),0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[3] = 5
        np.testing.assert_array_equal(episode.agent[4], [1, 4, 3])
        np.testing.assert_array_equal(agent.position, [1, 4, 3])
        np.testing.assert_array_equal(episode.calculate_reward(3), -1)

        agent.perform_action([0, 1, 0], episode)
        episode.action[4] = 5
        np.testing.assert_array_equal(episode.agent[5], [1, 5, 3])
        np.testing.assert_array_equal(agent.position, [1, 5, 3])
        np.testing.assert_array_equal(episode.calculate_reward(4), 0)


        agent.perform_action([0, 0, 1], episode)
        episode.action[5] = 5
        np.testing.assert_array_equal(episode.agent[6], [1, 5, 4])
        np.testing.assert_array_equal(agent.position, [1, 5, 4])
        np.testing.assert_array_equal(episode.calculate_reward(5), 0)

        agent.perform_action([0, -1, 0], episode)
        episode.action[6] = 5
        np.testing.assert_array_equal(episode.agent[7], [1, 5, 4])
        np.testing.assert_array_equal(agent.position, [1, 5, 4])
        np.testing.assert_array_equal(episode.calculate_reward(6), -1)

        agent.perform_action([0, 0, 1], episode)
        episode.action[7] = 5
        np.testing.assert_array_equal(episode.agent[8], [1, 5, 5])
        np.testing.assert_array_equal(agent.position, [1, 5, 5])
        np.testing.assert_array_equal(episode.calculate_reward(7), 0)


        agent.perform_action([0, -1, 0], episode)
        episode.action[8] = 5
        np.testing.assert_array_equal(episode.agent[9], [1, 5, 5])
        np.testing.assert_array_equal(agent.position, [1, 5, 5])
        np.testing.assert_array_equal(episode.calculate_reward(8), -1)

        agent.perform_action([0,0, 1], episode)
        episode.action[9] = 5
        np.testing.assert_array_equal(episode.agent[10], [1, 5,6])
        np.testing.assert_array_equal(agent.position, [1, 5, 6])
        np.testing.assert_array_equal(episode.calculate_reward(9), 0)

        agent.perform_action([0, -1, 0], episode)
        episode.action[10] = 5
        np.testing.assert_array_equal(episode.agent[11], [1, 5, 6])
        np.testing.assert_array_equal(agent.position, [1, 5, 6])
        np.testing.assert_array_equal(episode.calculate_reward(10), -1)

        agent.perform_action([0, 0, 1], episode)
        episode.action[11] = 5
        np.testing.assert_array_equal(episode.agent[12], [1, 5, 7])
        np.testing.assert_array_equal(agent.position, [1, 5, 7])
        np.testing.assert_array_equal(episode.calculate_reward(11), 0)

        agent.perform_action([0, -1, 0], episode)
        episode.action[12] = 5
        np.testing.assert_array_equal(episode.agent[13], [1, 4, 7])
        np.testing.assert_array_equal(agent.position, [1, 4, 7])
        np.testing.assert_array_equal(episode.calculate_reward(12), 0)

        # Test correct empty initialization.

    def test_walk_into_objs_wide_varied_vel(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(14, size=7)
        tensor[1, 3, 3, 3] = 8 / 33.0
        tensor[2, 3, 5, 3] = 11.0 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=14,
                                                 rewards=rewards)
        episode.agent_size = [0, 1, 1]

        self.initialize_pos(agent, episode)
        np.testing.assert_array_equal(episode.agent[0], [1, 3, 3])
        np.testing.assert_array_equal(agent.position, [1, 3, 3])

        # Invalid move
        agent.perform_action([0, 0, .9], episode)
        episode.action[0] = 5
        np.testing.assert_array_equal(episode.agent[1], [1, 3, 3])
        #np.testing.assert_array_equal(episode.agent[1], [1, 3, 3])
        np.testing.assert_array_equal(agent.position, [1, 3, 3])
        np.testing.assert_array_equal(episode.calculate_reward(0), -1)

        # Invalid move
        agent.perform_action([0, 0, 1.2], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], [1, 3, 3])
        np.testing.assert_array_equal(agent.position, [1, 3, 3])
        np.testing.assert_array_equal(episode.calculate_reward(1), -1.0)

        agent.perform_action([0, .8, 0], episode)
        episode.action[2] = 5
        # np.testing.assert_array_equal(episode.agent[3], [1, 4, 3])
        np.testing.assert_array_almost_equal(episode.agent[3], [1, 3.8, 3])
        np.testing.assert_array_equal(agent.position, [1, 4, 3])
        np.testing.assert_array_equal(episode.calculate_reward(2), 0)

        agent.perform_action([0, 0, .7], episode)
        episode.action[3] = 5
        #np.testing.assert_array_equal(episode.agent[4], [1, 4, 3])
        np.testing.assert_array_almost_equal(episode.agent[4], [1, 3.8, 3])
        np.testing.assert_array_equal(agent.position, [1, 4, 3])
        np.testing.assert_array_equal(episode.calculate_reward(3), -1)

        agent.perform_action([0, .9, 0], episode)
        episode.action[4] = 5
        #np.testing.assert_array_equal(episode.agent[5], [1, 5, 3])
        np.testing.assert_array_almost_equal(episode.agent[5], [1, 4.7, 3])
        np.testing.assert_array_equal(agent.position, [1, 5, 3])
        np.testing.assert_array_equal(episode.calculate_reward(4), 0)

        agent.perform_action([0, 0, .8], episode)
        episode.action[5] = 5
        #np.testing.assert_array_equal(episode.agent[6], [1, 5, 4])
        np.testing.assert_array_almost_equal(episode.agent[6], [1, 4.7, 3.8])
        np.testing.assert_array_equal(agent.position, [1, 5, 4])
        np.testing.assert_array_equal(episode.calculate_reward(5), 0)

        agent.perform_action([0, -.7, 0], episode)
        episode.action[6] = 5
        #np.testing.assert_array_equal(episode.agent[7], [1, 5, 4])
        np.testing.assert_array_almost_equal(episode.agent[7], [1, 4.7, 3.8])
        np.testing.assert_array_equal(agent.position, [1, 5, 4])
        np.testing.assert_array_equal(episode.calculate_reward(6), -1)

        agent.perform_action([0, 0, .9], episode)
        episode.action[7] = 5
        #np.testing.assert_array_equal(episode.agent[8], [1, 5, 5])
        np.testing.assert_array_almost_equal(episode.agent[8], [1, 4.7, 4.7])
        np.testing.assert_array_equal(agent.position, [1, 5, 5])
        np.testing.assert_array_equal(episode.calculate_reward(7), 0)

        agent.perform_action([0, -1.3, 0], episode)
        episode.action[8] = 5
        #np.testing.assert_array_equal(episode.agent[9], [1, 5, 5])
        np.testing.assert_array_almost_equal(episode.agent[9], [1, 4.7, 4.7])
        np.testing.assert_array_equal(agent.position, [1, 5, 5])
        np.testing.assert_array_equal(episode.calculate_reward(8), -1)

        agent.perform_action([0, 0, 1.3], episode)
        episode.action[9] = 5
        #np.testing.assert_array_equal(episode.agent[10], [1, 5, 6])
        np.testing.assert_array_almost_equal(episode.agent[10], [1,4.7, 6])
        np.testing.assert_array_equal(agent.position, [1, 5, 6])
        np.testing.assert_array_equal(episode.calculate_reward(9), 0)

        agent.perform_action([0, -1.1, 0], episode)
        episode.action[10] = 5
        #np.testing.assert_array_equal(episode.agent[11], [1, 5, 6])
        np.testing.assert_array_almost_equal(episode.agent[11], [1, 4.7, 6])
        np.testing.assert_array_equal(agent.position, [1, 5, 6])
        np.testing.assert_array_equal(episode.calculate_reward(10), -1)

        agent.perform_action([0, 0, 1.1], episode)
        episode.action[11] = 5
        #np.testing.assert_array_equal(episode.agent[12], [1, 5, 7])
        np.testing.assert_array_almost_equal(episode.agent[12], [1, 4.7, 7.1])
        np.testing.assert_array_equal(agent.position, [1, 5, 7])
        np.testing.assert_array_equal(episode.calculate_reward(11), 0)

        agent.perform_action([0, -.7, 0], episode)
        episode.action[12] = 5
        #np.testing.assert_array_equal(episode.agent[13], [1, 4, 7])
        np.testing.assert_array_almost_equal(episode.agent[13], [1, 4, 7.1])
        np.testing.assert_array_equal(agent.position, [1, 4, 7])
        np.testing.assert_array_equal(episode.calculate_reward(12), 0)


    def test_pedestrians(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(7)
        people[0].append(np.array([0,2, 0,0,0,0]).reshape((3,2)))
        people[1].append(np.array([0, 2, 1, 1, 1, 1]).reshape((3,2)))
        people[0].append(np.array([0, 2, 2, 2, 2, 2]).reshape((3,2)))
        people[0].append(np.array([0, 2, 0, 0, 2, 2]).reshape((3, 2)))

        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=7,  rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0]=[0,0,0]

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 1, 1], episode)
        episode.action[0] = 5
        np.testing.assert_array_equal(episode.agent[1], [0, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(0), -1)

        agent.perform_action([0, 1, 1], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], [0, 2, 2])
        np.testing.assert_array_equal(episode.calculate_reward(1), 0)

        agent.perform_action([0, -1, -1], episode)
        episode.action[2] = 5
        np.testing.assert_array_equal(episode.agent[3], [0, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(2), 0)

        agent.perform_action([0, -1, -1], episode)
        episode.action[3] = 5
        np.testing.assert_array_equal(episode.agent[4], [0, 0, 0])
        np.testing.assert_array_equal(episode.calculate_reward(3), 0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[4] = 5
        np.testing.assert_array_equal(episode.agent[5], [0, 0, 1])
        np.testing.assert_array_equal(episode.calculate_reward(4), 0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[5] = 5
        np.testing.assert_array_equal(episode.agent[6], [0, 0, 2])
        np.testing.assert_array_equal(episode.calculate_reward(5), 0)

    def test_pedestrians_varied_vel(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(7)
        people[0].append(np.array([0,2, 0,0,0,0]).reshape((3,2)))
        people[1].append(np.array([0, 2, 1, 1, 1, 1]).reshape((3,2)))
        people[0].append(np.array([0, 2, 2, 2, 2, 2]).reshape((3,2)))
        people[0].append(np.array([0, 2, 0, 0, 2, 2]).reshape((3, 2)))

        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=7,  rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0]=[0,0,0]

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 0.9, 0.9], episode)
        episode.action[0] = 5

        #np.testing.assert_array_equal(episode.agent[1], [0, 1, 1])
        np.testing.assert_array_almost_equal(episode.agent[1], [0, .9, .9])
        np.testing.assert_array_equal(episode.calculate_reward(0), -1)

        agent.perform_action([0, 1.1, 1.2], episode)
        episode.action[1] = 5
        np.testing.assert_array_almost_equal(episode.agent[2], [0, 2.0, 2.1])
        #np.testing.assert_array_equal(episode.agent[2], [0, 2, 2])
        np.testing.assert_array_equal(episode.calculate_reward(1), 0)

        agent.perform_action([0, -.9, -1], episode)
        episode.action[2] = 5
        #np.testing.assert_array_equal(episode.agent[3], [0, 1, 1])
        np.testing.assert_array_almost_equal(episode.agent[3], [0, 1.1, 1.1])
        np.testing.assert_array_equal(episode.calculate_reward(2), 0)

        agent.perform_action([0, -.9, -1.1], episode)
        episode.action[3] = 5
        #np.testing.assert_array_equal(episode.agent[4], [0, 0, 0])
        np.testing.assert_array_almost_equal(episode.agent[4], [0, .2, 0])
        np.testing.assert_array_equal(episode.calculate_reward(3), 0)

        agent.perform_action([0, 0, 1.2], episode)
        episode.action[4] = 5
        #np.testing.assert_array_equal(episode.agent[5], [0, 0, 1])
        np.testing.assert_array_almost_equal(episode.agent[5], [0, .2, 1.2])
        np.testing.assert_array_equal(episode.calculate_reward(4), 0)

        agent.perform_action([0, 0, .75], episode)
        episode.action[5] = 5
        #np.testing.assert_array_equal(episode.agent[6], [0, 0, 2])
        np.testing.assert_array_almost_equal(episode.agent[6], [0, 0.2, 1.95])
        np.testing.assert_array_equal(episode.calculate_reward(5), 0)



    def test_pedestrians_wide(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(14, size=7)
        for i in range(7):
            people[0].append(np.array([0, 2, i, i, i, i]).reshape((3, 2)))

        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.pedestrian_heatmap] = 1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=14, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = [0, 0, 0]

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        for i in range(1,7):
            agent.perform_action([0, .5, .5], episode)
            episode.action[i-1] = 5
            np.testing.assert_array_equal(episode.agent[i], [0, i*0.5,i*0.5])
            np.testing.assert_array_equal(episode.calculate_reward(i-1), 1)

        episode.agent[0] = np.array([0, 6, 6])
        agent.initial_position(episode.agent[0], episode.goal)
        for i in range(1,7):
            agent.perform_action([0, -.5, -.5], episode)
            episode.action[i - 1] = 5
            np.testing.assert_array_equal(episode.agent[i], [0, 6-(i*.5), 6-(i*.5)])
            np.testing.assert_array_equal(episode.calculate_reward(i-1), 1)




    def test_dist_travelled(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        tensor=np.ones(tensor.shape)*8/33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.distance_travelled] = 1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        for step in range(seq_len-2):
            agent.perform_action([0, .75, 0.75], episode)
            episode.action[step] = 5
            np.testing.assert_array_equal(episode.agent[step+1], [0, 0.75*(step+1), 0.75*(step+1)])
            np.testing.assert_array_equal(episode.calculate_reward(step), 0)
        agent.perform_action([0, .75, .75], episode)
        episode.action[seq_len-2] = 5
        np.testing.assert_array_equal(episode.agent[seq_len-1], [0, (seq_len-1)*.75, (seq_len-1)*.75])
        np.testing.assert_approx_equal(episode.calculate_reward(seq_len-2), (seq_len-1)*np.sqrt(2)*.75)



    def test_dist_travelled_vel(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        tensor=np.ones(tensor.shape)*8/33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.distance_travelled] = 1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        for step in range(seq_len-2):
            agent.perform_action([0, 1/np.sqrt(2), 1/np.sqrt(2)], episode)
            episode.action[step] = 5
            np.testing.assert_array_almost_equal(episode.agent[step+1], [0, 1/np.sqrt(2)*(step+1), 1/np.sqrt(2)*(step+1)])
            np.testing.assert_array_equal(episode.calculate_reward(step), 0)
        agent.perform_action([0,1/np.sqrt(2), 1/np.sqrt(2)], episode)
        episode.action[seq_len-2] = 5
        np.testing.assert_array_almost_equal(episode.agent[seq_len-1], [0, 1/np.sqrt(2)*(seq_len-1), 1/np.sqrt(2)*(seq_len-1)])
        np.testing.assert_approx_equal(episode.calculate_reward(seq_len-2), seq_len-1)

    def test_cars_hit_on_one(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars[1].append([0,2,0,0,1,1])
        cars[2].append([0, 2, 1, 1, 1, 1])
        cars[3].append([0, 2, 2, 2, 1, 1])
        tensor=np.ones(tensor.shape)*8/33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0.1])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0.1])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 0, .9], episode)
        episode.action[0] = 5
        np.testing.assert_array_equal(episode.agent[1], [0, 0,1])
        np.testing.assert_approx_equal(episode.calculate_reward(0), -1)

        agent.perform_action([0, 0, 1], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], [0, 0, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1), 0)


    def test_cars_hit_on_one_vel(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars[1].append([0,2,0,0,1,1])
        cars[2].append([0, 2, 1, 1, 1, 1])
        cars[3].append([0, 2, 2, 2, 1, 1])
        tensor=np.ones(tensor.shape)*8/33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 0, .9], episode)
        episode.action[0] = 5
        #np.testing.assert_array_equal(episode.agent[1], [0, 0, 1])
        np.testing.assert_array_almost_equal(episode.agent[1], [0, 0, .9])
        np.testing.assert_approx_equal(episode.calculate_reward(0), -1)

        agent.perform_action([0, 0, .8], episode)
        episode.action[1] = 5
        #np.testing.assert_array_equal(episode.agent[2], [0, 0, 1])
        np.testing.assert_array_almost_equal(episode.agent[2], [0, 0, .9])
        np.testing.assert_approx_equal(episode.calculate_reward(1), 0)


    def test_cars_hit_on_two(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars[1].append([0,2,0,0,1,1])
        cars[2].append([0, 2, 1, 1, 1, 1])
        cars[3].append([0, 2, 2, 2, 1, 1])
        tensor=np.ones(tensor.shape)*8/33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 1, 0.1], episode)
        episode.action[0] = 5
        np.testing.assert_array_equal(episode.agent[1], [0, 1, 0.1])
        np.testing.assert_approx_equal(episode.calculate_reward(0), 0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], [0, 1, 1.1])
        np.testing.assert_approx_equal(episode.calculate_reward(1), -1)

        agent.perform_action([0, 0, -1], episode)
        episode.action[2] = 5
        np.testing.assert_array_equal(episode.agent[3], [0, 1, 1.1])
        np.testing.assert_approx_equal(episode.calculate_reward(2), 0)

        for itr in range(3, seq_len-1):
            agent.perform_action([0, 0, -1], episode)
            episode.action[itr] = 5
            np.testing.assert_array_equal(episode.agent[3], [0, 1, 1.1])
            np.testing.assert_approx_equal(episode.calculate_reward(itr), 0)


    def test_cars_hit_on_two_vel(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars[1].append([0,2,0,0,1,1])
        cars[2].append([0, 2, 1, 1, 1, 1])
        cars[3].append([0, 2, 2, 2, 1, 1])
        tensor=np.ones(tensor.shape)*8/33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 1.1, 0], episode)
        episode.action[0] = 5

        #np.testing.assert_array_equal(episode.agent[1], [0, 1, 0])
        np.testing.assert_array_almost_equal(episode.agent[1], [0, 1.1, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(0), 0)

        agent.perform_action([0, 0, 1.2], episode)
        episode.action[1] = 5
        #np.testing.assert_array_equal(episode.agent[2], [0, 1, 1])
        np.testing.assert_array_almost_equal(episode.agent[2], [0, 1.1, 1.2])
        np.testing.assert_approx_equal(episode.calculate_reward(1), -1)

        agent.perform_action([0, 0, -.9], episode)
        episode.action[2] = 5
        np.testing.assert_array_almost_equal(episode.agent[3], [0, 1.1, 1.2])
        #np.testing.assert_array_equal(episode.agent[3], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(2), 0)

        for itr in range(3, seq_len-1):
            agent.perform_action([0, 0, -.96], episode)
            episode.action[itr] = 5
            #np.testing.assert_array_equal(episode.agent[3], [0, 1, 1])
            np.testing.assert_array_almost_equal(episode.agent[itr+1], [0, 1.1, 1.2])
            np.testing.assert_approx_equal(episode.calculate_reward(itr), 0)

    def test_cars_follow_car(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars[1].append([0,2,0,0,1,1])
        cars[2].append([0, 2, 1, 1, 1, 1])
        cars[3].append([0, 2, 2, 2, 1, 1])
        tensor=np.ones(tensor.shape)*8/33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 0, 0], episode)
        episode.action[0] = 5
        np.testing.assert_array_equal(episode.agent[1], [0, 0, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(0), 0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], [0, 0, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1), 0)

        agent.perform_action([0, 1, 0], episode)
        episode.action[2] = 5
        np.testing.assert_array_equal(episode.agent[3], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(2), 0)

        agent.perform_action([0, 1, 0], episode)
        episode.action[3] = 5
        np.testing.assert_array_equal(episode.agent[4], [0, 2, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(3), 0)

    def test_cars_follow_car_vel(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars[1].append([0, 2, 0, 0, 1, 1])
        cars[2].append([0, 2, 1, 1, 1, 1])
        cars[3].append([0, 2, 2, 2, 1, 1])
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        tensor = np.ones(tensor.shape) * 8 / 33.0
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 0, 0.1], episode)
        episode.action[0] = 5
        np.testing.assert_array_almost_equal(episode.agent[1], [0, 0, 0.1])
        np.testing.assert_approx_equal(episode.calculate_reward(0), 0)

        agent.perform_action([0, 0, .8], episode)
        episode.action[1] = 5
        np.testing.assert_array_almost_equal(episode.agent[2], [0, 0, .9])
        np.testing.assert_approx_equal(episode.calculate_reward(1), 0)

        agent.perform_action([0, 1.1, 0], episode)
        episode.action[2] = 5
        np.testing.assert_array_equal(episode.agent[3], [0, 1.1,.9])
        np.testing.assert_approx_equal(episode.calculate_reward(2), 0)

        agent.perform_action([0, 1.2, 0.2], episode)
        episode.action[3] = 5
        np.testing.assert_array_equal(episode.agent[4], [0, 2.3, 1.1])
        np.testing.assert_approx_equal(episode.calculate_reward(3), 0)

    def test_cars_hit_on_three(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars[1].append([0, 2, 0, 0, 1, 1])
        cars[2].append([0, 2, 1, 1, 1, 1])
        cars[3].append([0, 2, 2, 2, 1, 1])
        tensor = np.ones(tensor.shape) * 8 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 1, 1], episode)
        episode.action[0] = 5
        np.testing.assert_array_equal(episode.agent[1], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(0), 0)

        agent.perform_action([0, 0, 0], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1), -1)

    def test_cars_hit_on_three_vel(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars[1].append([0, 2, 0, 0, 1, 1])
        cars[2].append([0, 2, 1, 1, 1, 1])
        cars[3].append([0, 2, 2, 2, 1, 1])
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        tensor = np.ones(tensor.shape) * 8 / 33.0
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 0.9, 1.1], episode)
        episode.action[0] = 5
        np.testing.assert_array_almost_equal(episode.agent[1], [0, 0.9, 1.1])
        np.testing.assert_approx_equal(episode.calculate_reward(0), 0)

        agent.perform_action([0, 0, 0], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], [0, .9, 1.1])
        np.testing.assert_approx_equal(episode.calculate_reward(1), -1)

    def test_goal_reached_on_two(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) * 8 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.reached_goal] = 1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)
        episode.follow_goal=True

        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])
        episode.goal[1] = 2
        episode.goal[2] = 2

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        #np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 0, 1], episode)
        episode.action[0] = 5
        np.testing.assert_array_equal(episode.agent[1], [0, 0, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(0), 0)
        np.testing.assert_approx_equal(episode.measures[0, 7], np.sqrt(5))
        np.testing.assert_approx_equal(episode.measures[0,13], 0)

        agent.perform_action([0, 1, 0], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1), 1)
        np.testing.assert_approx_equal(episode.measures[1, 7], np.sqrt(2))
        np.testing.assert_approx_equal(episode.measures[1, 13], 1)

        agent.perform_action([0, 1, 1], episode)
        episode.action[2] = 5
        np.testing.assert_array_equal(episode.agent[3], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(2), 0)
        np.testing.assert_approx_equal(episode.measures[2, 7], np.sqrt(2))
        np.testing.assert_approx_equal(episode.measures[2, 13], 1)

    def test_goal_reached_on_two_vel(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) * 8 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.reached_goal] = 1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)
        episode.follow_goal = True

        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])
        episode.goal[1] = 2
        episode.goal[2] = 2

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        # np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 0, 1.1], episode)
        episode.action[0] = 5
        np.testing.assert_array_equal(episode.agent[1], [0, 0, 1.1])
        np.testing.assert_approx_equal(episode.calculate_reward(0), 0)
        np.testing.assert_approx_equal(episode.measures[0, 7], np.sqrt(4.81))
        np.testing.assert_approx_equal(episode.measures[0, 13], 0)

        agent.perform_action([0, .9, 0], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], [0, .9, 1.1])
        np.testing.assert_approx_equal(episode.calculate_reward(1), 0)
        np.testing.assert_approx_equal(episode.measures[1, 7], np.sqrt(2.02))
        np.testing.assert_approx_equal(episode.measures[1, 13], 0)

        agent.perform_action([0, 1, 1], episode)
        episode.action[2] = 5
        np.testing.assert_array_equal(episode.agent[3], [0, 1.9, 2.1])
        np.testing.assert_approx_equal(episode.calculate_reward(2), 1)
        np.testing.assert_approx_equal(episode.measures[2, 7], np.sqrt(0.02))
        np.testing.assert_approx_equal(episode.measures[2, 13], 1)

    def test_neg_dist_reward(self):
        seq_len = 4
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) * 8 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.total_distance_vrs_birds_flight_distance] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)

        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        #np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 0, 1], episode)
        episode.action[0] = 5
        agent.perform_action([0, 1, 0], episode)
        episode.action[1] = 5
        agent.perform_action([0, 1, 1], episode)
        episode.action[2] = 5
        np.testing.assert_array_equal(episode.agent[1], [0, 0, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(0, episode_done=True), 1-((2+np.sqrt(2))/np.sqrt(8)))
        np.testing.assert_approx_equal(episode.measures[0, 12], 2 + np.sqrt(2))
        np.testing.assert_approx_equal(episode.measures[0,6], np.sqrt(8))



        np.testing.assert_array_equal(episode.agent[2], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1, episode_done=True), 1-((1+np.sqrt(2))/np.sqrt(5)))
        np.testing.assert_approx_equal(episode.measures[1, 12],  1+np.sqrt(2))
        np.testing.assert_approx_equal(episode.measures[1, 6], np.sqrt(5))


        np.testing.assert_array_equal(episode.agent[3], [0, 2, 2])
        np.testing.assert_approx_equal(episode.calculate_reward(2, episode_done=True),0)
        np.testing.assert_approx_equal(episode.measures[2, 12],np.sqrt(2) )
        np.testing.assert_approx_equal(episode.measures[2, 6], np.sqrt(2))


    def test_follow_agent_reward(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        people[0]=[np.array([[0,0],[0,0], [0,0]]) ]
        people[1] = [np.array([[0, 0], [1, 1], [1, 1]]),np.array([[0,0],[0,1], [0,0]])]
        people[2] = [np.array([[0, 0], [2, 2], [2, 2]]), np.array([[0, 0], [1, 2], [0, 1]]) ]
        people[3] = [np.array([[0, 0], [3, 3],[3, 3]]), np.array([[0, 0], [2, 3], [1, 2]])]
        people[4] = [np.array([[0, 0], [4, 4], [4, 4]]), np.array([[0, 0], [3, 4], [2, 3]])]

        init_frames={1:0, 2:1}
        agent_1=[np.array([[0,0],[0,0], [0,0]]),np.array([[0, 0], [1, 1], [1, 1]]), np.array([[0, 0], [2, 2], [2, 2]]), np.array([[0, 0], [3, 3],[3, 3]]),np.array([[0, 0], [4, 4], [4, 4]])]
        agent_2 = [ np.array([[0, 0], [0, 1], [0,0]]), np.array([[0, 0], [1, 2], [0, 1]]),
                   np.array([[0, 0], [2, 3], [1, 2]]), np.array([[0, 0], [3, 4], [2, 3]])]

        people_map={1:agent_1, 2:agent_2}

        tensor = np.ones(tensor.shape) * 8 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.one_step_prediction_error] = 1
        #(0-0, 0-1, 0-2, 0-3,0-4, 0-5, 0-6,0-7,0-8,0-9,0-10,1-11,0,0,0)
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards, people_dict=people_map, init_frames=init_frames)

        episode.agent_size = [0, 0, 0]
        episode.initial_position(None, initialization=1, init_key=0)
        print (episode.goal_person_id)
        np.testing.assert_array_equal(episode.goal_person_id_val, 1)
        #episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        #np.testing.assert_array_equal(agent.position, [0, 0, 0])
        #"Next action: [ 1:'downL', 2:'down', 3:'downR', 4:'left', 5:'stand', 6:'right',7:'upL', 8:'up', 9:'upR', 0: stop excecution] "
        agent.perform_action([0, 1, 1], episode)
        episode.action[0]=8

        np.testing.assert_array_equal(episode.agent[1], [0, 1, 1])

        agent.perform_action([0, 1, 0], episode)
        episode.action[1] = 7

        np.testing.assert_approx_equal(episode.calculate_reward(0), 1)
        #np.testing.assert_approx_equal(episode.measures[0, 12], np.sqrt(2))
        np.testing.assert_approx_equal(episode.measures[0,4], np.sqrt(2))
        np.testing.assert_approx_equal(episode.measures[0, 8], 0)
        np.testing.assert_approx_equal(episode.measures[0, 9], 0)


        np.testing.assert_array_equal(episode.agent[2], [0, 2, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1),1- (1/np.sqrt(32)))
        #np.testing.assert_approx_equal(episode.measures[1, 12], np.sqrt(2)+1)
        np.testing.assert_approx_equal(episode.measures[1, 4], np.sqrt(5))
        np.testing.assert_approx_equal(episode.measures[1, 8], 1)
        np.testing.assert_approx_equal(episode.measures[1, 9], 1)
        #
        # agent.perform_action([0, 1, 1], episode)
        # np.testing.assert_array_equal(episode.agent[3], [0, 2, 2])
        # np.testing.assert_approx_equal(episode.calculate_reward(2),1-((2+np.sqrt(2))/np.sqrt(8)))
        # np.testing.assert_approx_equal(episode.measures[2, 12], 2 + np.sqrt(2))
        # np.testing.assert_approx_equal(episode.measures[2, 4], 2*np.sqrt(2))


    def test_mov_penalty_reward(self):
        seq_len = 10
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) * 8 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,rewards=rewards)

        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        # np.testing.assert_array_equal(agent.position, [0, 0, 0])

        for frame in range(4):
            agent.perform_action([0, 0, 1], episode)
            episode.action[frame] = 5
            np.testing.assert_array_equal(episode.agent[frame], [0, 0, frame])
            np.testing.assert_approx_equal(episode.calculate_reward(frame),0)
            np.testing.assert_approx_equal(episode.measures[frame, 11], 0)

        for frame in range(4, 8):
            agent.perform_action([0, 1, 0], episode)
            episode.action[frame] = 7
            np.testing.assert_array_equal(episode.agent[frame], [0, frame-4,4 ])

            np.testing.assert_approx_equal(episode.calculate_reward(frame), 0)
            np.testing.assert_approx_equal(episode.measures[frame, 11], 0)

    # [0:[0, -1, -1], 1:[0, -1, 0], 2:[0, -1, 1], 3:[0, 0, -1], 4:[0, 0, 0], 5:[0, 0, 1], 6:[0, 1, -1], 7:[0, 1, 0], 8:[0, 1, 1]]
    def test_mov_penalty_reward_two(self):
        seq_len = 14
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) * 8 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)

        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        # np.testing.assert_array_equal(agent.position, [0, 0, 0])


        agent.perform_action([0, 1, 0], episode)
        episode.action[0] = 7
        episode.velocity[0] = np.array([0, 1, 0])
        np.testing.assert_array_equal(episode.agent[1], [0, 1,0])
        np.testing.assert_approx_equal(episode.calculate_reward(0), 0)
        np.testing.assert_approx_equal(episode.measures[0, 11], 0)

        agent.perform_action([0, 1, 1], episode)
        episode.action[1] = 8
        episode.velocity[1] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.agent[2], [0,2, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1), 0)
        np.testing.assert_approx_equal(episode.measures[1, 11], 0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[2] = 5
        episode.velocity[2] = np.array([0, 0, 1])
        np.testing.assert_array_equal(episode.agent[3], [0, 2, 2])
        np.testing.assert_approx_equal(episode.calculate_reward(2), 0)
        np.testing.assert_approx_equal(episode.measures[2, 11], 0)

        agent.perform_action([0, -1, 1], episode)
        episode.action[3] = 2
        episode.velocity[3] = np.array([0, -1, 1])
        np.testing.assert_array_equal(episode.agent[4], [0, 1, 3])
        np.testing.assert_approx_equal(episode.calculate_reward(3), 0)
        np.testing.assert_approx_equal(episode.measures[3, 11], 0)

        agent.perform_action([0, -1, 0], episode)
        episode.action[4] = 1
        episode.velocity[4] = np.array([0, -1, 0])
        np.testing.assert_array_equal(episode.agent[5], [0, 0, 3])
        np.testing.assert_approx_equal(episode.calculate_reward(4), 0)
        np.testing.assert_approx_equal(episode.measures[4, 11], 0)

        agent.perform_action([0, -1, -1], episode)
        episode.action[5] = 0
        episode.velocity[5] = np.array([0, -1, -1])
        np.testing.assert_array_equal(episode.agent[6], [0, -1, 2])
        np.testing.assert_approx_equal(episode.calculate_reward(5), 0)
        np.testing.assert_approx_equal(episode.measures[5, 11], 0)

        agent.perform_action([0, 0, -1], episode)
        episode.action[6] = 3
        episode.velocity[6] = np.array([0, 0, -1])
        np.testing.assert_array_equal(episode.agent[7], [0, -1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(6), 0)
        np.testing.assert_approx_equal(episode.measures[6, 11], 0)

        agent.perform_action([0, 1, -1], episode)
        episode.action[7] = 6
        episode.velocity[7] = np.array([0, 1, -1])
        np.testing.assert_array_equal(episode.agent[8], [0, 0, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(7), 0)
        np.testing.assert_approx_equal(episode.measures[7, 11], 0)

        agent.perform_action([0, 1, 0], episode)
        episode.action[8] = 7
        episode.velocity[8] = np.array([0, 1, 0])
        np.testing.assert_array_equal(episode.agent[9], [0, 1, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(8), 0)
        np.testing.assert_approx_equal(episode.measures[8, 11], 0)

        agent.perform_action([0, 0, 0], episode)
        episode.action[9] = 4
        episode.velocity[9] = np.array([0, 0, 0])
        np.testing.assert_array_equal(episode.agent[10], [0, 1, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(9), 0)
        np.testing.assert_approx_equal(episode.measures[9, 11],0)

        agent.perform_action([0, 0, 0], episode)
        episode.action[10] = 4
        episode.velocity[10] = np.array([0, 0, 0])
        np.testing.assert_array_equal(episode.agent[11], [0, 1, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(10), 0)
        np.testing.assert_approx_equal(episode.measures[10, 11], 0)

        agent.perform_action([0, 1, 1], episode)
        episode.action[11] = 8
        episode.velocity[11] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.agent[12], [0, 2, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(11),0)
        np.testing.assert_approx_equal(episode.measures[11, 11], 0)


        # [0:[0, -1, -1], 1:[0, -1, 0], 2:[0, -1, 1], 3:[0, 0, -1], 4:[0, 0, 0], 5:[0, 0, 1], 6:[0, 1, -1], 7:[0, 1, 0], 8:[0, 1, 1]]

    def test_mov_penalty_reward_three(self):
        seq_len = 14
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) * 8 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)

        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        # np.testing.assert_array_equal(agent.position, [0, 0, 0])


        agent.perform_action([0, 1, 0], episode)
        episode.action[0] = 7
        episode.velocity[0] = np.array([0, 1, 0])
        np.testing.assert_array_equal(episode.agent[1], [0, 1, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(0), 0)
        np.testing.assert_approx_equal(episode.measures[0, 11], 0)

        agent.perform_action([0, 1, -1], episode)
        episode.action[1] = 6
        episode.velocity[1] = np.array([0, 1, -1])
        np.testing.assert_array_equal(episode.agent[2], [0, 2, -1])
        np.testing.assert_approx_equal(episode.calculate_reward(1), 0)
        np.testing.assert_approx_equal(episode.measures[1, 11], 0)

        agent.perform_action([0, 0, -1], episode)
        episode.action[2] = 3
        episode.velocity[2] = np.array([0, 0, -1])
        np.testing.assert_array_equal(episode.agent[3], [0, 2, -2])
        np.testing.assert_approx_equal(episode.calculate_reward(2), 0)
        np.testing.assert_approx_equal(episode.measures[2, 11], 0)

        agent.perform_action([0, -1, -1], episode)
        episode.action[3] = 0
        episode.velocity[3] = np.array([0, -1, -1])
        np.testing.assert_array_equal(episode.agent[4], [0, 1, -3])
        np.testing.assert_approx_equal(episode.calculate_reward(3), 0)
        np.testing.assert_approx_equal(episode.measures[3, 11], 0)

        agent.perform_action([0, -1, 0], episode)
        episode.action[4] = 1
        episode.velocity[4] = np.array([0, -1, 0])
        np.testing.assert_array_equal(episode.agent[5], [0, 0, -3])
        np.testing.assert_approx_equal(episode.calculate_reward(4), 0)
        np.testing.assert_approx_equal(episode.measures[4, 11], 0)

        agent.perform_action([0, -1, +1], episode)
        episode.action[5] = 2
        episode.velocity[5] = np.array([0, -1, 1])
        np.testing.assert_array_equal(episode.agent[6], [0, -1, -2])
        np.testing.assert_approx_equal(episode.calculate_reward(5), 0)
        np.testing.assert_approx_equal(episode.measures[5, 11], 0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[6] = 5
        episode.velocity[6] = np.array([0, 0, 1])
        np.testing.assert_array_equal(episode.agent[7], [0, -1, -1])
        np.testing.assert_approx_equal(episode.calculate_reward(6), 0)
        np.testing.assert_approx_equal(episode.measures[6, 11], 0)

        agent.perform_action([0, 1, 1], episode)
        episode.action[7] = 8
        episode.velocity[7] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.agent[8], [0, 0, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(7), 0)
        np.testing.assert_approx_equal(episode.measures[7, 11], 0)

        agent.perform_action([0, 1, 0], episode)
        episode.action[8] = 7
        episode.velocity[8] = np.array([0, 1, 0])
        np.testing.assert_array_equal(episode.agent[9], [0, 1, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(8), 0)
        np.testing.assert_approx_equal(episode.measures[8, 11], 0)

        agent.perform_action([0, 0, 0], episode)
        episode.action[9] = 4
        episode.velocity[9] = np.array([0, 0, 0])
        np.testing.assert_array_equal(episode.agent[10], [0, 1, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(9), 0)
        np.testing.assert_approx_equal(episode.measures[9, 11], 0)


        agent.perform_action([0, 1, 1], episode)
        episode.action[10] = 8
        episode.velocity[10] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.agent[11], [0, 2, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(10), -np.sin(np.pi/8))
        np.testing.assert_approx_equal(episode.measures[10, 11], np.sin(np.pi/8))

    def test_mov_penalty_zig_zag(self):
        seq_len = 14
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) * 8 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.changing_movement_directions_frequently] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)

        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        # np.testing.assert_array_equal(agent.position, [0, 0, 0])

        # [0:[0, -1, -1], 1:[0, -1, 0], 2:[0, -1, 1], 3:[0, 0, -1], 4:[0, 0, 0], 5:[0, 0, 1], 6:[0, 1, -1], 7:[0, 1, 0], 8:[0, 1, 1]]
        agent.perform_action([0, 1, 0], episode)
        episode.action[0] = 7
        episode.velocity[0]=np.array([0, 1, 0])
        np.testing.assert_array_equal(episode.agent[1], [0, 1, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(0), 0)
        np.testing.assert_approx_equal(episode.measures[0, 11], 0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[1] = 5
        episode.velocity[1] = np.array([0, 0, 1])
        np.testing.assert_array_equal(episode.agent[2], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1), 0)
        np.testing.assert_approx_equal(episode.measures[1, 11], 0)

        agent.perform_action([0, 1, 0], episode)
        episode.action[2] = 7
        episode.velocity[2] = np.array([0, 1, 0])
        np.testing.assert_array_equal(episode.agent[3], [0, 2, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(2), -1/np.sqrt(2))
        np.testing.assert_approx_equal(episode.measures[2, 11], 1/np.sqrt(2))

        agent.perform_action([0, 1, 0], episode)
        episode.action[3] = 7
        episode.velocity[3] = np.array([0, 1, 0])
        np.testing.assert_array_equal(episode.agent[4], [0, 3, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(3), -1 / np.sqrt(2))
        np.testing.assert_approx_equal(episode.measures[3, 11], 1 / np.sqrt(2))

        agent.perform_action([0, 0, 1], episode)
        episode.action[4] = 5
        episode.velocity[4] = np.array([0, 0, 1])
        np.testing.assert_array_equal(episode.agent[5], [0, 3, 2])
        np.testing.assert_approx_equal(episode.calculate_reward(4), -1 / np.sqrt(2))
        np.testing.assert_approx_equal(episode.measures[4, 11], 1 / np.sqrt(2))

        agent.perform_action([0, 1, 1], episode)
        episode.action[5] = 8
        episode.velocity[5] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.agent[6], [0, 4, 3])
        np.testing.assert_approx_equal(episode.calculate_reward(5), -np.sin(np.pi/8))
        np.testing.assert_approx_equal(episode.measures[5, 11], np.sin(np.pi/8))

        agent.perform_action([0, 1, 1], episode)
        episode.action[6] = 8
        episode.velocity[6] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.agent[7], [0, 5, 4])
        np.testing.assert_approx_equal(episode.calculate_reward(6),  -np.sin(np.pi/8))
        np.testing.assert_approx_equal(episode.measures[6, 11], np.sin(np.pi/8))

        agent.perform_action([0, 1, 1], episode)
        episode.action[7] = 8
        episode.velocity[7] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.agent[8], [0, 6, 5])
        np.testing.assert_approx_equal(episode.calculate_reward(7), 0)
        np.testing.assert_approx_equal(episode.measures[7, 11], 0)

        agent.perform_action([0, 1, 1], episode)
        episode.action[8] = 8
        episode.velocity[8] = np.array([0, 1, 1])
        np.testing.assert_array_equal(episode.agent[9], [0, 7, 6])
        np.testing.assert_approx_equal(episode.calculate_reward(8), 0)
        np.testing.assert_approx_equal(episode.measures[8, 11], 0)








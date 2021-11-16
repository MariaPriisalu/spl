import unittest
import numpy as np
import sys
sys.path.append("RL/") #
from RL.episode import SimpleEpisode
from RL.agent import SimplifiedAgent
from RL.settings import run_settings, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX


class TestEnv(unittest.TestCase):

    def get_reward(self, all_zeros=False):

        rewards = np.zeros(NBR_REWARD_WEIGHTS)
        if all_zeros:
            return rewards

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
    def initialize_episode(self, cars, gamma, people, pos_x, pos_y, tensor,seq_len=30, rewards=[], agent_size=(0,0,0), people_dict={}, init_frames={}):
        settings = run_settings()
        settings.useHeroCar = False
        settings.useRLToyCar=False
        settings.multiplicative_reward=False
        if len(rewards)==0:
            rewards = self.get_reward()
        episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, seq_len, rewards, agent_size=agent_size,
                                people_dict=people_dict, init_frames=init_frames, defaultSettings=settings)
        agent=SimplifiedAgent(settings)

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
        pos, i, vel = episode.initial_position(None, initialization=8)

        agent.initial_position(pos,episode.goal)

    # Test correct empty initialization.
    def test_walk_into_objs(self):

        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(11)
        tensor[1,1,1,3]=8/33.0
        tensor[2, 1, 2, 3] = 11.0 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        agent, episode=self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=11, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        self.initialize_pos(agent, episode)
        np.testing.assert_array_equal(episode.agent[0], [1,1,1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])

        # Invalid move
        agent.perform_action([0,0,1], episode)
        episode.action[0]=5
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
        episode.action[4] = 1
        np.testing.assert_array_equal(episode.agent[5], [1, 2, 2])
        np.testing.assert_array_equal(agent.position, [1, 2, 2])
        np.testing.assert_array_equal(episode.calculate_reward(4), -1)

        # Valid move
        agent.perform_action([0, 0, -1], episode)
        episode.action[5] = 3
        np.testing.assert_array_equal(episode.agent[6], [1, 2, 1])
        np.testing.assert_array_equal(agent.position, [1, 2, 1])
        np.testing.assert_array_equal(episode.calculate_reward(5), 0)

        # Valid move
        agent.perform_action([0, -1,0], episode)
        episode.action[6] = 1
        np.testing.assert_array_equal(episode.agent[7], [1, 1, 1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(6), 0)

        # Valid move
        agent.perform_action([0, -1, 0], episode)
        episode.action[7] = 1
        np.testing.assert_array_equal(episode.agent[8], [1, 0, 1])
        np.testing.assert_array_equal(agent.position, [1, 0, 1])
        np.testing.assert_array_equal(episode.calculate_reward(7), 0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[8] = 5
        np.testing.assert_array_equal(episode.agent[9], [1, 0, 2])
        np.testing.assert_array_equal(agent.position, [1, 0, 2])
        np.testing.assert_array_equal(episode.calculate_reward(8), 0)

        agent.perform_action([0, 1, 0], episode)
        episode.action[9] = 7
        np.testing.assert_array_equal(episode.agent[10], [1, 0, 2])
        np.testing.assert_array_equal(agent.position, [1, 0, 2])

    # Test correct empty initialization.
    #[0:[0, -1, -1], 1:[0, -1, 0], 2:[0, -1, 1], 3:[0, 0, -1], 4:[0, 0, 0], 5:[0, 0, 1], 6:[0, 1, -1], 7:[0, 1, 0], 8:[0, 1, 1]]
    def test_walk_into_objs_wide(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(14, size=7)
        tensor[1, 3, 3, 3] = 8 / 33.0
        tensor[2, 3, 5, 3] = 11.0 / 33.0
        rewards = self.get_reward(all_zeros=True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        print(" rewards "+str(rewards))
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
        episode.action[2] = 7
        np.testing.assert_array_equal(episode.agent[3], [1, 4, 3])
        np.testing.assert_array_equal(agent.position, [1, 4, 3])
        np.testing.assert_array_equal(episode.calculate_reward(2),0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[3] = 5
        np.testing.assert_array_equal(episode.agent[4], [1, 4, 3])
        np.testing.assert_array_equal(agent.position, [1, 4, 3])
        np.testing.assert_array_equal(episode.calculate_reward(3), -1)

        agent.perform_action([0, 1, 0], episode)
        episode.action[4] = 7
        np.testing.assert_array_equal(episode.agent[5], [1, 5, 3])
        np.testing.assert_array_equal(agent.position, [1, 5, 3])
        np.testing.assert_array_equal(episode.calculate_reward(4), 0)


        agent.perform_action([0, 0, 1], episode)
        episode.action[5] = 5
        np.testing.assert_array_equal(episode.agent[6], [1, 5, 4])
        np.testing.assert_array_equal(agent.position, [1, 5, 4])
        np.testing.assert_array_equal(episode.calculate_reward(5), 0)

        agent.perform_action([0, -1, 0], episode)
        episode.action[6] = 1
        np.testing.assert_array_equal(episode.agent[7], [1, 5, 4])
        np.testing.assert_array_equal(agent.position, [1, 5, 4])
        np.testing.assert_array_equal(episode.calculate_reward(6), -1)

        agent.perform_action([0, 0, 1], episode)
        episode.action[7] = 5
        np.testing.assert_array_equal(episode.agent[8], [1, 5, 5])
        np.testing.assert_array_equal(agent.position, [1, 5, 5])
        np.testing.assert_array_equal(episode.calculate_reward(7), 0)

        agent.perform_action([0, -1, 0], episode)
        episode.action[8] = 1
        np.testing.assert_array_equal(episode.agent[9], [1, 5, 5])
        np.testing.assert_array_equal(agent.position, [1, 5, 5])
        np.testing.assert_array_equal(episode.calculate_reward(8), -1)

        agent.perform_action([0,0, 1], episode)
        episode.action[9] = 5
        np.testing.assert_array_equal(episode.agent[10], [1, 5,6])
        np.testing.assert_array_equal(agent.position, [1, 5, 6])
        np.testing.assert_array_equal(episode.calculate_reward(9), 0)

        agent.perform_action([0, -1, 0], episode)
        episode.action[10] = 1
        np.testing.assert_array_equal(episode.agent[11], [1, 5, 6])
        np.testing.assert_array_equal(agent.position, [1, 5, 6])
        np.testing.assert_array_equal(episode.calculate_reward(10), -1)

        agent.perform_action([0, 0, 1], episode)
        episode.action[11] = 5
        np.testing.assert_array_equal(episode.agent[12], [1, 5, 7])
        np.testing.assert_array_equal(agent.position, [1, 5, 7])
        np.testing.assert_array_equal(episode.calculate_reward(11), 0)

        agent.perform_action([0, -1, 0], episode)
        episode.action[12] = 1
        np.testing.assert_array_equal(episode.agent[13], [1, 4, 7])
        np.testing.assert_array_equal(agent.position, [1, 4, 7])
        np.testing.assert_array_equal(episode.calculate_reward(12), 0)



    def test_pedestrians(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(7)
        people[0].append(np.array([0,2, 0,0,0,0]).reshape((3,2)))
        people[1].append(np.array([0, 2, 1, 1, 1, 1]).reshape((3,2)))
        people[0].append(np.array([0, 2, 2, 2, 2, 2]).reshape((3,2)))
        people[0].append(np.array([0, 2, 0, 0, 2, 2]).reshape((3, 2)))
        # rewards=np.zeros(15)
        # rewards[9]=1
        # rewards[7] =-1
        people_dict={0:[people[0][0],people[1][0]],1:[people[0][1]],2:[people[0][2]]}
        init_frames={0:0,1:0,2:0}
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] = 1
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=7,  rewards=rewards)#, people_dict=people_dict, init_frames=init_frames)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None, initialization=1, init_key=0)

        episode.agent[0]=[0,0,0]
        # [0:[0, -1, -1], 1:[0, -1, 0], 2:[0, -1, 1], 3:[0, 0, -1], 4:[0, 0, 0], 5:[0, 0, 1], 6:[0, 1, -1], 7:[0, 1, 0], 8:[0, 1, 1]]


        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 1, 1], episode)
        episode.action[0] = 8
        np.testing.assert_array_equal(episode.agent[1], [0, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(0), -1)

        agent.perform_action([0, 1, 1], episode)
        episode.action[1] = 8
        np.testing.assert_array_equal(episode.agent[2], [0, 2, 2])
        np.testing.assert_array_equal(episode.calculate_reward(1), 1)

        agent.perform_action([0, -1, -1], episode)
        episode.action[2] = 8
        np.testing.assert_array_equal(episode.agent[3], [0, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(2), 0)

        agent.perform_action([0, -1, -1], episode)
        episode.action[3] = 4
        np.testing.assert_array_equal(episode.agent[4], [0, 0, 0])
        np.testing.assert_array_equal(episode.calculate_reward(3), 0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[4] = 5
        np.testing.assert_array_equal(episode.agent[5], [0, 0, 1])
        np.testing.assert_array_equal(episode.calculate_reward(4), 0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[5] = 5
        np.testing.assert_array_equal(episode.agent[6], [0, 0, 2])
        np.testing.assert_array_equal(episode.calculate_reward(5), 1)


    def test_pedestrians_wide(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(14, size=7)
        for i in range(7):
            people[0].append(np.array([0, 2, i, i, i, i]).reshape((3, 2)))

        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] = 1
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=14, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None, initialization=1)
        episode.agent[0] = [0, 0, 0]

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        for i in range(1,7):
            agent.perform_action([0, 1, 1], episode)
            episode.action[i-1]=8

            np.testing.assert_array_equal(episode.agent[i], [0, i,i])
            np.testing.assert_array_equal(episode.calculate_reward(i-1), 1)

        episode.agent[0] = [0, 6, 6]
        agent.initial_position(episode.agent[0], episode.goal)

        for i in range( 1,7):
            agent.perform_action([0, -1, -1], episode)
            episode.action[i-1] = 0
            np.testing.assert_array_equal(episode.agent[i], [0, 6-i, 6-i])

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
            agent.perform_action([0, 1, 1], episode)
            episode.action[step ] = 8
            np.testing.assert_array_equal(episode.agent[step+1], [0, step+1, step+1])
            np.testing.assert_array_equal(episode.calculate_reward(step), 0)
        agent.perform_action([0, 1, 1], episode)
        episode.action[seq_len-2] = 8
        np.testing.assert_array_equal(episode.agent[seq_len-1], [0, seq_len-1, seq_len-1])
        np.testing.assert_approx_equal(episode.calculate_reward(seq_len-2), (seq_len-1)*np.sqrt(2))


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
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0],episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 0, 1], episode)
        episode.action[0] = 5
        np.testing.assert_array_equal(episode.agent[1], [0, 0, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(0), -1)

        agent.perform_action([0, 0, 1], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], [0, 0, 1])
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
        # [0:[0, -1, -1], 1:[0, -1, 0], 2:[0, -1, 1], 3:[0, 0, -1], 4:[0, 0, 0], 5:[0, 0, 1], 6:[0, 1, -1], 7:[0, 1, 0], 8:[0, 1, 1]]
        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 1, 0], episode)
        episode.action[0] = 7
        np.testing.assert_array_equal(episode.agent[1], [0, 1, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(0), 0)
        np.testing.assert_approx_equal(episode.measures[0,0], 0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1), -1)
        np.testing.assert_approx_equal(episode.measures[1, 0], 1)

        agent.perform_action([0, 0, -1], episode)
        episode.action[2] = 3
        np.testing.assert_array_equal(episode.agent[3], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(2), 0)
        np.testing.assert_approx_equal(episode.measures[2, 0], 1)

        for itr in range(3, seq_len-1):
            agent.perform_action([0, 0, -1], episode)
            episode.action[itr] = 3
            np.testing.assert_array_equal(episode.agent[itr+1], [0, 1, 1])
            np.testing.assert_approx_equal(episode.calculate_reward(3), 0)

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

        #0
        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        #1
        agent.perform_action([0, 0, 0], episode)
        episode.action[0] = 4
        np.testing.assert_array_equal(episode.agent[1], [0, 0, 0])
        np.testing.assert_approx_equal(episode.calculate_reward(0),0)

        #2
        agent.perform_action([0, 0, 1], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], [0, 0, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1),0)#0

        #3
        agent.perform_action([0, 1, 0], episode)
        episode.action[2] = 7
        np.testing.assert_array_equal(episode.agent[3], [0, 1, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(2), 0)#0

        #4
        agent.perform_action([0, 1, 0], episode)
        episode.action[3] = 7
        np.testing.assert_array_equal(episode.agent[4], [0, 2, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(3), 0)#0

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

        agent.perform_action([0, 0, 1], episode)
        episode.action[0] = 5
        np.testing.assert_array_equal(episode.agent[1], [0, 0, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(0), -1)

        agent.perform_action([0, 1, 1], episode)
        episode.action[1] = 8
        np.testing.assert_array_equal(episode.agent[1], [0, 0, 1])
        np.testing.assert_approx_equal(episode.calculate_reward(1),0)


import unittest
import numpy as np
import sys
sys.path.append("RL/")
from RL.episode import SimpleEpisode
from RL.agent_pfnn import AgentPFNN
from RL.agent import ContinousAgent, SimplifiedAgent
from RL.settings import run_settings,NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX

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
    def initialize_episode(self, cars, gamma, people, pos_x, pos_y, tensor,seq_len=30, rewards=np.zeros(NBR_REWARD_WEIGHTS), agent_size=(0,0,0),people_dict={}, init_frames={}):
        #  tensor, people_e, cars_e, pos_x, pos_y, gamma, seq_len, reward_weights
        setup = run_settings()
        setup.action_freq=1
        setup.useHeroCar = False
        setup.useRLToyCar = False
        frameRate, frameTime = setup.getFrameRateAndTime()
        episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, seq_len, rewards, agent_size=agent_size,
                                people_dict=people_dict, init_frames=init_frames, adjust_first_frame=False,
                                seq_len_pfnn=seq_len * 60 // frameRate, defaultSettings=setup)
        episode.action=np.zeros(len(episode.action))
        episode.vel_init = np.zeros(3)
        agent=AgentPFNN(setup, None)#, None, None, None)

        return agent, episode

    # Help function. Setup for tests.
    def initialize_episode_cont(self, cars, gamma, people, pos_x, pos_y, tensor, seq_len=30,
                                rewards=np.zeros(NBR_REWARD_WEIGHTS), agent_size=(0, 0, 0), people_dict={},
                           init_frames={}):
        setup = run_settings()
        setup.action_freq = 1
        setup.useHeroCar = False
        setup.useRLToyCar = False
        #  tensor, people_e, cars_e, pos_x, pos_y, gamma, seq_len, reward_weights
        agent, episode=self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor,seq_len, rewards, agent_size,people_dict, init_frames)
        episode.vel_init=np.zeros(3)
        agent=ContinousAgent(setup)

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
        episode.vel_init = np.zeros(3)
        agent.initial_position(pos,episode.goal)


    # Test correct empty initialization.
    def test_walk_into_objs(self):
        print ("----------------------------------------------------------------test walk into objs start")
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(11)
        tensor[1,1,1,3]=8/33.0
        tensor[2, 1, 2, 3] = 11.0 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        agent, episode=self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=11, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        self.initialize_pos(agent, episode)
        print ("Agent pos "+str(episode.agent[0]) )
        np.testing.assert_array_equal(episode.agent[0], [1,1,1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])
        np.testing.assert_array_equal(agent.pos_exact, [1, 1, 1])

        # Invalid move
        agent.perform_action([0,0,1], episode)
        episode.action[0] = 5
        print ("Agent pos " + str(episode.agent[1]))
        print ("----------------------------------------------------------------test walk into objs end")
        np.testing.assert_array_equal( [1,1],episode.agent[1][1:])
        np.testing.assert_array_equal([ 1, 1],agent.pos_exact[1:])
        np.testing.assert_array_equal(episode.calculate_reward(0), -1)

        # Invalid move
        agent.perform_action([0, 0, 1], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal( [1, 1], episode.agent[2][1:])
        np.testing.assert_array_equal([1, 1], agent.pos_exact[1:])
        np.testing.assert_array_equal(episode.calculate_reward(1), -1)

        # Valid move
        agent.perform_action([0, 1, 0], episode)
        episode.action[2] = 5
        np.testing.assert_array_less(  episode.agent[2][1], episode.agent[3][1])
        np.testing.assert_array_almost_equal( episode.agent[2][0], episode.agent[3][0],decimal=0)
        np.testing.assert_array_equal(episode.calculate_reward(2), 0)


        # Invalid move
        agent.perform_action([0, -1, 1], episode)
        episode.action[3] = 5
        print (episode.agent)
        #np.testing.assert_array_equal(episode.agent[4], episode.agent[3])
        #np.testing.assert_array_equal(agent.pos_exact,episode.agent[3])
        np.testing.assert_array_equal(episode.calculate_reward(3), -1)

        # valid move
        agent.perform_action([0, -1, 0], episode)
        episode.action[4] = 5

        np.testing.assert_array_less(episode.agent[5][1], episode.agent[4][1])
        np.testing.assert_array_almost_equal(episode.agent[4][0], episode.agent[5][0], decimal=0)
        np.testing.assert_array_equal(episode.calculate_reward(4), -1)
        print ("----------------------------------------------------------------test walk into objs end")



    def test_pedestrians(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(7)
        people[0].append(np.array([0,2, 0,0,0,0]).reshape((3,2)))
        people[2].append(np.array([0, 2, 1, 1, 1, 1]).reshape((3,2)))
        people[0].append(np.array([0, 2, 2, 2, 2, 2]).reshape((3,2)))
        people[0].append(np.array([0, 2, 0, 0, 2, 2]).reshape((3, 2)))

        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=7,  rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.vel_init = np.zeros(3)
        episode.agent[0]=[0,0,0]

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 1, 1], episode)
        episode.action[0] = 5
        np.testing.assert_array_less( [ 0, 0], episode.agent[1][1:])
        np.testing.assert_array_equal(episode.calculate_reward(0), 0)

        agent.perform_action([0, 1, 1], episode)
        episode.action[1] = 5

        np.testing.assert_array_less(episode.agent[1][1:], episode.agent[2][1:])
        np.testing.assert_array_equal(episode.calculate_reward(1), -1)

        agent.perform_action([0, -1, -1], episode)
        episode.action[2] = 5
        np.testing.assert_array_less( episode.agent[3][1:], episode.agent[2][1:])
        np.testing.assert_array_equal(episode.calculate_reward(2), 0)

        agent.perform_action([0, -1, -1], episode)
        episode.action[3] = 5
        np.testing.assert_array_less(episode.agent[4][1:], episode.agent[3][1:])
        np.testing.assert_array_equal(episode.calculate_reward(3), 0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[4] = 5
        np.testing.assert_array_less(episode.agent[4][2], episode.agent[5][2])
        np.testing.assert_array_almost_equal(episode.agent[4][1], episode.agent[5][1], decimal=0)
        np.testing.assert_array_equal(episode.calculate_reward(4), 0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[5] = 5
        np.testing.assert_array_less(episode.agent[5][2], episode.agent[6][2])
        np.testing.assert_array_almost_equal(episode.agent[5][1], episode.agent[6][1], decimal=0)
        np.testing.assert_array_equal(episode.calculate_reward(5), 0)

    def test_cars_hit_on_one(self):
        seq_len=7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=7)
        cars[1].append([0,2,0,0,1,1])
        cars[1].append([0, 2, 0, 0, 0, 0])
        cars[2].append([0, 2, 0, 1, 1, 2])
        cars[2].append([0, 2, 1, 1, 1, 1])
        cars[3].append([0, 2, 2, 2, 1, 1])
        cars[3].append([0, 2, 0, 0, 1, 1])
        tensor=np.ones(tensor.shape)*8/33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.vel_init = np.zeros(3)
        episode.agent[0] = np.array([0, 0, 0.1])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0.1])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 0, 1], episode)
        episode.action[0] = 5

        np.testing.assert_array_less(episode.agent[0][2], episode.agent[1][2])
        np.testing.assert_array_almost_equal(episode.agent[0][1], episode.agent[1][1], decimal=0)
        np.testing.assert_approx_equal(episode.calculate_reward(0),-1)

        agent.perform_action([0, 0, 1], episode)
        episode.action[1] = 5
        np.testing.assert_array_equal(episode.agent[2], episode.agent[1])
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
        episode.vel_init = np.zeros(3)
        episode.agent[0] = np.array([0, 0, 0])

        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 1, 0.1], episode)
        episode.action[0] = 5
        np.testing.assert_array_less(episode.agent[0][1], episode.agent[1][1])
        np.testing.assert_array_almost_equal(episode.agent[0][2], episode.agent[1][2], decimal=0)
        np.testing.assert_approx_equal(episode.calculate_reward(0), 0)

        agent.perform_action([0, 0, 1], episode)
        episode.action[1] = 5
        np.testing.assert_array_less(episode.agent[1][2], episode.agent[2][2])
        np.testing.assert_array_almost_equal(episode.agent[1][1], episode.agent[2][1], decimal=0)
        print(episode.agent)
        np.testing.assert_approx_equal(episode.calculate_reward(1), -1)
        np.testing.assert_approx_equal(episode.measures[1,0], 1)

        agent.perform_action([0, 0, -1], episode)
        episode.action[2] = 5
        np.testing.assert_array_equal(episode.agent[3], episode.agent[2])
        np.testing.assert_approx_equal(episode.calculate_reward(2), 0)
        np.testing.assert_approx_equal(episode.measures[2, 0], 1)

        for itr in range(3, seq_len-1):
            agent.perform_action([0, 0, -1], episode)
            episode.action[itr] = 5
            np.testing.assert_array_equal(episode.agent[itr],episode.agent[2])
            np.testing.assert_approx_equal(episode.calculate_reward(itr), 0)

        print(episode.agent)



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
        episode.vel_init = np.zeros(3)
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
        np.testing.assert_array_less(episode.agent[1][2], episode.agent[2][2])
        np.testing.assert_array_almost_equal(episode.agent[1][1], episode.agent[2][1], decimal=0)
        np.testing.assert_approx_equal(episode.calculate_reward(1), 0)

        agent.perform_action([0, 1, 0], episode)
        episode.action[2] = 5
        np.testing.assert_array_less(episode.agent[2][1], episode.agent[3][1])
        np.testing.assert_array_almost_equal(episode.agent[2][2], episode.agent[3][2], decimal=0)
        np.testing.assert_approx_equal(episode.calculate_reward(2), 0)

        agent.perform_action([0, 1, 0], episode)
        episode.action[3] = 5
        np.testing.assert_array_less(episode.agent[3][1], episode.agent[4][1])
        np.testing.assert_array_almost_equal(episode.agent[3][2], episode.agent[4][2], decimal=0)
        np.testing.assert_approx_equal(episode.calculate_reward(3), 0)


    def test_goal_reached_on_two(self):
        seq_len = 7
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
        tensor = np.ones(tensor.shape) * 8 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.reached_goal] = 1
        agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
                                                 rewards=rewards)
        episode.follow_goal=True
        episode.vel_init = np.zeros(3)
        episode.agent_size = [0, 0, 0]
        episode.initial_position(None)
        episode.vel_init = np.zeros(3)
        episode.agent[0] = np.array([0, 0, 0])
        episode.goal[1] = 2
        episode.goal[2] = 2


        agent.initial_position(episode.agent[0], episode.goal)
        np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
        np.testing.assert_array_equal(agent.position, [0, 0, 0])
        #np.testing.assert_array_equal(agent.position, [0, 0, 0])

        agent.perform_action([0, 0, 1], episode)
        episode.action[0] = 5
        np.testing.assert_array_less(episode.agent[0][2], episode.agent[1][2])
        np.testing.assert_array_almost_equal(episode.agent[0][1], episode.agent[1][1], decimal=0)
        np.testing.assert_approx_equal(episode.calculate_reward(0), 0)
        #np.testing.assert_approx_equal(episode.measures[0, 7], np.sqrt(5))
        np.testing.assert_approx_equal(episode.measures[0,13], 0)

        agent.perform_action([0, 1, 0], episode)
        episode.action[1] = 5
        np.testing.assert_array_less(episode.agent[1][1], episode.agent[2][1])
        np.testing.assert_array_almost_equal(episode.agent[1][2], episode.agent[2][2], decimal=0)

        np.testing.assert_approx_equal(episode.calculate_reward(1), 0)
        #np.testing.assert_approx_equal(episode.measures[1, 7], np.sqrt(2))
        np.testing.assert_approx_equal(episode.measures[1, 13], 0)

        agent.perform_action([0, 1, 1], episode)
        episode.action[2] = 5
        np.testing.assert_array_less(episode.agent[2][1:], episode.agent[3][1:])

        print (np.linalg.norm(episode.agent[3][1:]-episode.goal[1:]))
        np.testing.assert_approx_equal(episode.calculate_reward(2), 1)
        #np.testing.assert_approx_equal(episode.measures[2, 7], np.sqrt(2))
        np.testing.assert_approx_equal(episode.measures[2, 13], 1)


        agent.perform_action([0, 1, 1], episode)
        episode.action[2] = 5
        np.testing.assert_approx_equal(episode.agent[3][1], episode.agent[4][1])
        np.testing.assert_approx_equal(episode.agent[3][2], episode.agent[4][2])

        np.testing.assert_approx_equal(episode.calculate_reward(3), 0)
        #np.testing.assert_approx_equal(episode.measures[2, 7], np.sqrt(2))
        np.testing.assert_approx_equal(episode.measures[3, 13], 1)


        agent.perform_action([0, 1, 1], episode)
        episode.action[2] = 5
        np.testing.assert_array_equal(episode.agent[4], episode.agent[5])

        np.testing.assert_approx_equal(episode.calculate_reward(4), 0)
        #np.testing.assert_approx_equal(episode.measures[2, 7], np.sqrt(2))
        np.testing.assert_approx_equal(episode.measures[4, 13], 1)

    # def test_follow_agent_reward(self):
    #     seq_len = 7
    #     cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(seq_len, size=5)
    #     people[0]=[np.array([[0,0],[0,0], [0,0]]) ]
    #     people[1] = [np.array([[0, 0], [1, 1], [1, 1]]),np.array([[0,0],[0,1], [0,0]])]
    #     people[2] = [np.array([[0, 0], [2, 2], [2, 2]]), np.array([[0, 0], [1, 2], [0, 1]]) ]
    #     people[3] = [np.array([[0, 0], [3, 3],[3, 3]]), np.array([[0, 0], [2, 3], [1, 2]])]
    #     people[4] = [np.array([[0, 0], [4, 4], [4, 4]]), np.array([[0, 0], [3, 4], [2, 3]])]
    #
    #     init_frames={1:0, 2:1}
    #     agent_1=[np.array([[0,0],[0,0], [0,0]]),np.array([[0, 0], [1, 1], [1, 1]]), np.array([[0, 0], [2, 2], [2, 2]]), np.array([[0, 0], [3, 3],[3, 3]]),np.array([[0, 0], [4, 4], [4, 4]])]
    #     agent_2 = [ np.array([[0, 0], [0, 1], [0,0]]), np.array([[0, 0], [1, 2], [0, 1]]),
    #                np.array([[0, 0], [2, 3], [1, 2]]), np.array([[0, 0], [3, 4], [2, 3]])]
    #
    #     people_map={1:agent_1, 2:agent_2}
    #
    #     tensor = np.ones(tensor.shape) * 8 / 33.0
    #     agent, episode = self.initialize_episode(cars, gamma, people, pos_x, pos_y, tensor, seq_len=seq_len,
    #                                              rewards=(0, 0, 0, 0,0, 0, 0,0,0,0,0,1,0, 0,0,0), people_dict=people_map, init_frames=init_frames)
    #
    #     episode.agent_size = [0, 0, 0]
    #     episode.initial_position(None, initialization=1, init_key=0)
    #     np.testing.assert_array_equal(episode.goal_person_id_val, 1)
    #
    #
    #     agent.initial_position(episode.agent[0], episode.goal)
    #     np.testing.assert_array_equal(episode.agent[0], [0, 0, 0])
    #     np.testing.assert_array_equal(agent.position, [0, 0, 0])
    #
    #     agent.perform_action([0, 1, 1], episode)
    #     episode.action[0]=8
    #
    #     np.testing.assert_array_less(episode.agent[0][1:], episode.agent[1][1:])
    #     np.testing.assert_array_less(0, episode.calculate_reward(0))
    #     # np.testing.assert_approx_equal(episode.measures[0, 12], np.sqrt(2))
    #     np.testing.assert_approx_equal(episode.measures[0, 4],
    #                                    np.sqrt(episode.agent[1][1] ** 2 + episode.agent[1][2] ** 2))
    #
    #     np.testing.assert_approx_equal(episode.measures[0, 8], 1)
    #     np.testing.assert_array_less( 0,episode.measures[0, 9])
    #
    #     agent.perform_action([0, 1, 0], episode)
    #     episode.action[1] = 7
    #
    #     np.testing.assert_array_less(0, episode.calculate_reward(1))
    #     np.testing.assert_array_less(episode.agent[1][1],episode.agent[2][1])
    #     #np.testing.assert_approx_equal(episode.calculate_reward(1),1- (1/np.sqrt(32)))
    #     #np.testing.assert_approx_equal(episode.measures[1, 12], np.sqrt(2)+1)
    #     #np.testing.assert_approx_equal(episode.measures[1, 4], np.sqrt(5))
    #     np.testing.assert_approx_equal(episode.measures[1, 8], 1)
    #     #np.testing.assert_array_less(0, episode.measures[1, 9])
    #
    #     agent.perform_action([0, 1, 0], episode)
    #     episode.action[2] = 7
    #     np.testing.assert_array_less(0, episode.calculate_reward(2))
    #     np.testing.assert_array_less( episode.agent[2][1],episode.agent[3][1])
    #     print(episode.agent)
    #     # np.testing.assert_approx_equal(episode.calculate_reward(1),1- (1/np.sqrt(32)))
    #     # np.testing.assert_approx_equal(episode.measures[1, 12], np.sqrt(2)+1)
    #     # np.testing.assert_approx_equal(episode.measures[1, 4], np.sqrt(5))
    #     np.testing.assert_approx_equal(episode.measures[2, 8], 1)
    #     np.testing.assert_array_less(0, episode.measures[2, 9])
    #
    #
    #     # agent.perform_action([0, 1, 1], episode)
    #     # np.testing.assert_array_equal(episode.agent[3], [0, 2, 2])
    #     # np.testing.assert_approx_equal(episode.calculate_reward(2),1-((2+np.sqrt(2))/np.sqrt(8)))
    #     # np.testing.assert_approx_equal(episode.measures[2, 12], 2 + np.sqrt(2))
    #     # np.testing.assert_approx_equal(episode.measures[2, 4], 2*np.sqrt(2))


    def test_continous_agent(self):
        cars, gamma, people, pos_x, pos_y, tensor = self.initialize_tensor(11)
        tensor[1,0,0,3]=8/33.0
        tensor[2, 2, 1, 3] = 11.0 / 33.0
        rewards = self.get_reward(True)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        agent, episode=self.initialize_episode_cont(cars, gamma, people, pos_x, pos_y, tensor, seq_len=11, rewards=rewards)
        episode.agent_size = [0, 0, 0]
        self.initialize_pos(agent, episode)

        np.testing.assert_array_equal(episode.agent[0], [1,0,0])
        np.testing.assert_array_equal(agent.position, [1, 0, 0])

        # valid move
        agent.perform_action([0,1,1], episode)
        episode.action[0] = -np.pi/4
        np.testing.assert_array_equal(episode.agent[1], [1,1,1])
        np.testing.assert_array_equal(agent.position, [1, 1, 1])
        np.testing.assert_array_equal(episode.calculate_reward(0), 0)
        np.testing.assert_equal(agent.angle, - np.pi/4)

        # valid move
        agent.perform_action([0, 1, 1], episode)
        episode.action[1] =  -np.pi/4
        np.testing.assert_array_almost_equal(episode.agent[2], [1, 1,1+ np.sqrt(2)])
        np.testing.assert_array_almost_equal(agent.pos_exact, [1, 1, 1+np.sqrt(2)])
        np.testing.assert_array_equal(episode.calculate_reward(1), 0)
        np.testing.assert_equal(agent.angle,  -np.pi / 2)

        # Valid move
        agent.perform_action([0, 1, 1], episode)
        episode.action[2] = -np.pi/4
        np.testing.assert_array_almost_equal(  episode.agent[3],[1, 0,2+np.sqrt(2)])
        np.testing.assert_array_almost_equal( agent.pos_exact,[1, 0,2+np.sqrt(2)])
        np.testing.assert_array_equal(episode.calculate_reward(2), 0)
        np.testing.assert_equal(agent.angle,  -3*np.pi / 4)

        # Valid move
        agent.perform_action([0, 1, 1], episode)
        episode.action[3] =  -np.pi/4
        np.testing.assert_array_almost_equal(episode.agent[4], [1, - np.sqrt(2),2+np.sqrt(2)])
        np.testing.assert_array_almost_equal(agent.pos_exact, [1, - np.sqrt(2), 2+np.sqrt(2)])
        np.testing.assert_array_equal(episode.calculate_reward(3), 0)
        np.testing.assert_equal(agent.angle,  np.pi )

        # Valid move
        agent.perform_action([0, 1, 1], episode)
        episode.action[4] =  -np.pi/4
        np.testing.assert_array_almost_equal(episode.agent[5], [1, - np.sqrt(2)-1, 1+np.sqrt(2) ])
        np.testing.assert_array_almost_equal(agent.pos_exact, [1, - np.sqrt(2)-1, 1+np.sqrt(2) ])
        np.testing.assert_array_equal(episode.calculate_reward(4), 0)
        np.testing.assert_approx_equal(agent.angle, 3*np.pi / 4)

        # Valid move
        agent.perform_action([0, 1, 1], episode)
        episode.action[5] =  -np.pi/4
        np.testing.assert_array_almost_equal(episode.agent[6], [1, - np.sqrt(2) - 1, 1 ])
        np.testing.assert_array_almost_equal(agent.pos_exact, [1, - np.sqrt(2) - 1, 1 ])
        np.testing.assert_array_equal(episode.calculate_reward(5), 0)
        np.testing.assert_approx_equal(agent.angle, np.pi / 2)
        # Valid move
        agent.perform_action([0, 1, 1], episode)
        episode.action[6] =  -np.pi/4
        np.testing.assert_array_almost_equal(episode.agent[7], [1, - np.sqrt(2) , 0])
        np.testing.assert_array_almost_equal(agent.pos_exact, [1, - np.sqrt(2) , 0])
        np.testing.assert_array_equal(episode.calculate_reward(6), 0)
        np.testing.assert_approx_equal(agent.angle, np.pi / 4)
        # Valid move
        agent.perform_action([0, 1, 1], episode)
        episode.action[7] =  -np.pi/4
        np.testing.assert_array_almost_equal(episode.agent[8], [1, 0, 0])
        np.testing.assert_array_almost_equal(agent.pos_exact, [1, 0, 0])
        np.testing.assert_array_equal(episode.calculate_reward(7), 0)
        self.assertLess(abs(agent.angle),1e-12)

        agent.perform_action([0, -1, -1], episode)
        episode.action[8] =  np.pi/4
        np.testing.assert_array_almost_equal(episode.agent[9], [1, -1, -1])
        np.testing.assert_array_almost_equal(agent.pos_exact, [1, -1, -1])
        np.testing.assert_array_equal(episode.calculate_reward(8), 0)
        print(agent.angle/np.pi)
        np.testing.assert_approx_equal(agent.angle, 3*np.pi / 4)





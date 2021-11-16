import unittest
import numpy as np
import sys
sys.path.append("RL/") #
from RL.episode import SimpleEpisode
from RL.settings import run_settings, NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, NBR_REWARD_WEIGHTS, PEDESTRIAN_REWARD_INDX
# Test methods in episode.

class TestEpisode(unittest.TestCase):

    # Test correct empty initialization.
    def test_initialize(self):
        cars, people, tensor, gamma=self.initialize()
        episode = self.get_episode(cars, gamma, people, tensor)
        pos, i, vel=episode.initial_position(None)
        # print( pos)
        # np.testing.assert_array_equal(episode.valid_positions[pos[1],pos[2] ], 1)
        #np.testing.assert_array_equal(i, -1)

    def get_reward(self):

        rewards = np.zeros(NBR_REWARD_WEIGHTS)
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_car] = -1
        rewards[PEDESTRIAN_REWARD_INDX.on_pavement] = 1
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_objects] = -1
        rewards[PEDESTRIAN_REWARD_INDX.distance_travelled] = 1
        rewards[PEDESTRIAN_REWARD_INDX.out_of_axis] = -1
        rewards[PEDESTRIAN_REWARD_INDX.collision_with_pedestrian] = -1
        rewards[PEDESTRIAN_REWARD_INDX.on_pedestrian_trajectory] = 1
        return rewards

    # Test correct initialization when one voxel of Simple exists.
    def test_initialize_one_Simple(self):
        cars, people, tensor, gamma = self.initialize()
        for i in [8]:#[6,7,8,9,10]:
            tensor[1,1,1,3]=i/NUM_SEM_CLASSES
            episode = self.get_episode(cars, gamma, people, tensor)
            Simple=episode.find_sidewalk(False)
            np.testing.assert_array_equal(Simple[0], [1])
            np.testing.assert_array_equal(Simple[1], [1])
            np.testing.assert_array_equal(Simple[2], [1])
            pos, i, vel = episode.initial_position(None, initialization=8)
            np.testing.assert_array_equal(pos, [1, 1, 1])
            np.testing.assert_array_equal(i, -1)

    def get_episode(self, cars, gamma, people, tensor, seq_len=30):
        pos_x = 0
        pos_y = 0
        rewards = self.get_reward()
        episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, seq_len, rewards, agent_size=(0, 0, 0),
                                defaultSettings=run_settings())
        return episode

    # Test correct initialization when two voxels of road exists.
    def test_initialize_two_Simple(self):
        cars, people, tensor, gamma = self.initialize()
        tensor[:,:,:,3] = 6 / NUM_SEM_CLASSES*np.ones((3,3,3))
        tensor[1,1,1,3]=8/NUM_SEM_CLASSES
        episode = self.get_episode(cars, gamma, people, tensor)
        Simple=episode.find_sidewalk(False) # test Simple function.
        np.testing.assert_array_equal(Simple[0], [1])
        np.testing.assert_array_equal(Simple[1], [1])
        np.testing.assert_array_equal(Simple[2], [1])
        pos, i , vel= episode.initial_position(None, initialization=8)
        np.testing.assert_array_equal(pos, [1, 1, 1])
        #self.assertEqual(i, -1)

    # Test initialization when there is one block of Simple.
    def test_initialize_one_block_Simple(self):
        cars, people, tensor, gamma = self.initialize()
        for i in [8]:
            tensor[:,:,:,3] = i / NUM_SEM_CLASSES*np.ones((3,3,3))
            episode = self.get_episode(cars, gamma, people, tensor)
            pos, i, vel = episode.initial_position(None, initialization=8)
            np.testing.assert_array_less(np.sort(pos), 3)
            #np.testing.assert_array_equal(i, -1)

    # Test initialization when there is one block of Simple.
    def test_initialize_one_small_block_Simple(self):
        cars, people, tensor, gamma = self.initialize()
        for i in [8]:
            tensor[:, :, 0, 3] = i / NUM_SEM_CLASSES * np.ones((3, 3))
            episode = self.get_episode(cars, gamma, people, tensor)
            pos, i, vel = episode.initial_position(None, initialization=8)
            self.assertLess(pos[0], 3)
            self.assertLess(pos[1], 3)
            self.assertEqual(pos[2], 0)
            #self.assertEqual(i, -1)

    # Test initialization when there is two voxels of Simple.
    def test_initialize_two_block_Simple(self):
        cars, people, tensor, gamma = self.initialize()
        for i in [8]:
            tensor[0, 2, 0, 3] = i / NUM_SEM_CLASSES
            tensor[0, 0, 0, 3] = i / NUM_SEM_CLASSES
            episode = self.get_episode(cars, gamma, people, tensor)
            pos, i, vel = episode.initial_position(None, initialization=8)
            self.assertEqual(pos[0], 0)
            self.assertTrue(pos[1]== 2 or pos[1]==0)
            self.assertEqual(pos[2], 0)
            #self.assertEqual(i, -1)

    # Test initialization when there is a pedestrian.

    # def test_one_person(self):
    #     cars, people, tensor, gamma = self.initialize()
    #     people[29].append(np.random.randint(2,size=(3,2)))
    #     mean=np.mean(people[29][0], axis=1).astype(int)
    #     print (mean)
    #     for i in [8]:
    #         tensor[0, 2, 0, 3] = i / 33.0
    #         tensor[0, 0, 0, 3] = i / 33.0
    #         pos_x = 0
    #         pos_y = 0
    #         episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, 30,
    #                                 (-1, 0, 1, -1, 1, -1, 0, -1, 0, 1, 0, 0, 0), agent_size=(0, 0, 0), defaultSettings=run_settings())
    #         episode.agent_size = [0,0,0]
    #         pos, i , vel= episode.initial_position(None, initialization=10)
    #         np.testing.assert_array_equal(pos, mean)
    #         # self.assertEqual(i, 29)
    #         self.assertEqual(len(episode.intercept_person(0)),1)
    #         episode.agent[1]=[2,2,2]
    #         self.assertEqual(len(episode.intercept_person(1)), 0)
    #
    # # Test initialization when there are two pedestrians.
    # def test_two_person(self):
    #     cars, people, tensor, gamma = self.initialize()
    #     people[29].append(np.array([0,0,0,0,0,0]).reshape((3,2)))
    #     people[10].append(np.array([0, 1, 1,0,1,1]).reshape((3,2)))
    #
    #     pos_x =0
    #     pos_y = 0
    #     episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, 30,
    #                             (-1, 0, 1, -1, 1, -1, 0, -1, 0, 1, 0, 0, 0), agent_size=(0, 0, 0), defaultSettings=run_settings())
    #     episode.agent_size = [0,0,0]
    #     pos, i , vel= episode.initial_position(None, initialization=10)
    #     episode.agent[0]=[0,0,0]
    #
    #     self.assertEqual(len(episode.intercept_person(0)), 1)
    #     episode.agent[1] = [2, 2, 2]
    #     self.assertEqual(len(episode.intercept_person(1)), 0)
    #     episode.agent[2] = [0, 1, 1]
    #     self.assertEqual(len(episode.intercept_person(2)), 1)

    # Help function. Setup for tests.
    def initialize(self, seq_len=30):
        tensor = np.zeros((3, 3, 3, 6))
        people = []
        cars = []
        for i in range(seq_len):
            people.append([])
            cars.append([])
        gamma=0.99
        return cars, people, tensor, gamma

    # Test method intercept_car with one car.
    def test_intercept_car(self):
        cars, people, tensor, gamma = self.initialize()
        for frame in range(1,30):
            cars[frame].append([1,1,1,1,1,1])
        episode = self.get_episode(cars, gamma, people, tensor)
        episode.agent_size=[0,0,0]
        episode.agent[0]=[0,0,0]
        episode.agent[0] = [1,0.8,1.1]
        self.assertEqual(episode.intercept_car(0), 0)
        episode.agent[1] = [1, 0, 0]
        self.assertEqual(episode.intercept_car(1), 0)
        episode.agent[2] = [1, .9, 0]
        self.assertEqual(episode.intercept_car(2), 0)
        episode.agent[3] = [1, 0, .7]
        self.assertEqual(episode.intercept_car(3), 0)
        episode.agent[4] = [0, 0, 1.2]
        self.assertEqual(episode.intercept_car(4), 0)
        episode.agent[5] = [0, 1.3, 0]
        self.assertEqual(episode.intercept_car(5), 0)
        episode.agent[6] = [1, 1.1, .9]  # True
        self.assertEqual(episode.intercept_car(6), 1)
        episode.agent[7] = [1, 1.2, 2.1]
        self.assertEqual(episode.intercept_car(7), 0)
        episode.agent[8] = [0, .9, 1.8]
        self.assertEqual(episode.intercept_car(8), 0)
        episode.agent[9] = [2, 1.1, 2.1]
        self.assertEqual(episode.intercept_car(9), 0)

    # Test method intercept_car with two cars.
    def test_intercept_two_car(self):
        cars, people, tensor, gamma = self.initialize()
        for frame in range(30):
            cars[frame].append([0,0,0,1,0,0])
            cars[frame].append([2, 2, 0, 0, 0, 0])
        episode = self.get_episode(cars, gamma, people, tensor)
        episode.agent_size=[0,0,0]
        episode.agent[0]=[0,0,0]
        self.assertEqual(episode.intercept_car(0), 2)
        episode.agent[0] = [0, .9, 0]
        self.assertEqual(episode.intercept_car(0),1)
        episode.agent[0] = [0, 0, .9]
        self.assertEqual(episode.intercept_car(0), 0)
        episode.agent[0] = [1, 0, 0.1]
        self.assertEqual(episode.intercept_car(0), 2)
        episode.agent[0] = [2, 0.2, 0.1]
        self.assertEqual(episode.intercept_car(0), 2)
        episode.agent[0] = [1, .9, 1.1]
        self.assertEqual(episode.intercept_car(0), 0)
        for frame in range(30):
            cars[frame]=[]
        cars[1] = []
        cars[1].append([0, 1, 0, 1, 0, 1])
        episode = self.get_episode(cars, gamma, people, tensor)
        episode.agent_size = [1,1,1]
        episode.agent[1] = [0, 0, 0]
        self.assertEqual(episode.intercept_car(1), 1)
        episode.agent[1] = [0, 1.1, 0]
        self.assertEqual(episode.intercept_car(1), 1)
        episode.agent[1] = [2, 2.1, 2.1]
        self.assertEqual(episode.intercept_car(1), 1)
        episode.agent[0] = [2, 1.9, 1.8]
        self.assertEqual(episode.intercept_car(0), 0)

        episode.agent[1] = [3, 2.9, 3.1]
        self.assertEqual(episode.intercept_car(0), 0)

    # Test method intercep_obj with one object.
    def test_intercept_obj(self):
        objs=range(11,21)#[1,4,5,11,12,13,14,15,16,17,18,19,20,21,24,25,26,27,28,29,30,31,32,33,34,35]
        cars, people, tensor, gamma = self.initialize()
        for val in objs:
            tensor[0,0,0,3]=val/NUM_SEM_CLASSES
            episode = self.get_episode(cars, gamma, people, tensor)
            episode.agent_size = [1,1,1]
            episode.agent[0] = [0, 0.1, 0.2]
            self.assertEqual(episode.intercept_objects(0), 1/9.0)

    # Test method intercep_obj with two objects.
    def test_intercept_2_obj(self):
        objs = range(11,21)#[1, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,34,35]
        cars, people, tensor, gamma = self.initialize()
        prev_val=objs[-1]
        for val in objs:
            tensor[0, 0, 0, 3] = val / NUM_SEM_CLASSES
            tensor[0,1,0,3]=prev_val/NUM_SEM_CLASSES
            prev_val=val
            episode = self.get_episode(cars, gamma, people, tensor)
            episode.agent_size = [1,1,1]
            episode.agent[0] = [0, 0.2, 0.1]
            self.assertEqual(episode.intercept_objects(0), 2 / 9.0)

    # Test method intercep_obj with no object.
    def test_intercept_no_obj(self):
        not_objs = [2,3,6,7,8,9,10,22,23]
        cars, people, tensor, gamma = self.initialize()
        prev_val=not_objs[-1]
        for val in not_objs:
            tensor[0, 0, 0, 3] = val / NUM_SEM_CLASSES
            tensor[0,1,0,3]=prev_val/NUM_SEM_CLASSES
            prev_val=val
            episode = self.get_episode(cars, gamma, people, tensor)
            episode.agent_size = [1,1,1]
            episode.agent[0] = [0, 0.2, 0.3]
            self.assertEqual(episode.intercept_objects(0), 0 / 9.0)

    # Test method intercep_obj when not intercepring object.
    def test_intercept_no_inter(self):
        objs = [1, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
        cars, people, tensor, gamma = self.initialize()
        prev_val = objs[-1]
        for val in objs:
            tensor[2, 2,0, 3] = val / NUM_SEM_CLASSES
            tensor[2,2, 0, 3] = prev_val / NUM_SEM_CLASSES
            prev_val = val
            episode = self.get_episode(cars, gamma, people, tensor)
            episode.agent_size = [1,1,1]
            episode.agent[0] = [0, -0.1, -0.2]
            self.assertEqual(episode.intercept_objects(0), 0 / 9.0)

    # Test method on Simple
    def test_on_pavement(self):
        Simple=[6,7,8,9,10]
        cars, people, tensor, gamma = self.initialize()
        for val in Simple:
            tensor[0,1,1]=val/NUM_SEM_CLASSES
            episode = self.get_episode(cars, gamma, people, tensor)
            episode.agent_size = [0,0,0]
            episode.agent[0] = [0, 0.1, -0.1]
            self.assertFalse(episode.on_pavement(0))
            episode.agent[1] = [0, 1.1, 0.1]
            self.assertFalse(episode.on_pavement(1))
            episode.agent[2] = [0, .9, .8]
            self.assertTrue(episode.on_pavement(2))
            episode.agent[3] = [1, 1.1, .9]
            self.assertTrue(episode.on_pavement(3))

    def test_iou_pavement(self):
        Simple = [8]#[6, 7, 8, 9, 10]
        cars, people, tensor, gamma = self.initialize()
        for val in Simple:
            tensor[0, 1, 1] = val / NUM_SEM_CLASSES
            episode = self.get_episode(cars, gamma, people, tensor)
            episode.agent_size = [0, 0, 0]
            episode.agent[0] = [0, 0, 0.1]
            self.assertEqual(episode.iou_sidewalk(0), 0)
            episode.agent[1] = [0, 1.2, 0.1]
            self.assertEqual(episode.iou_sidewalk(1), 0)
            episode.agent[2] = [0, .9, 1.2]
            self.assertEqual(episode.iou_sidewalk(2), 1)
            episode.agent[3] = [1, .9, 1.1]
            self.assertEqual(episode.iou_sidewalk(3), 1)

    def test_reward_pavement_and_obstacle(self):
        seq_len=4
        cars, people, tensor, gamma = self.initialize(seq_len)
        for i in [8]:
            tensor[1, 1, 1, 3] = i / NUM_SEM_CLASSES
            tensor[2, 2, 2, 3]=11/NUM_SEM_CLASSES
            pos_x = 0
            pos_y = 0
            episode = self.get_episode(cars, gamma, people, tensor,seq_len)
            Simple = episode.find_sidewalk(False)
            episode.agent_size = [0, 0, 0]
            np.testing.assert_array_equal(Simple[0], [1])
            np.testing.assert_array_equal(Simple[1], [1])
            np.testing.assert_array_equal(Simple[2], [1])
            pos, i , vel= episode.initial_position(None, initialization=8)
            np.testing.assert_array_equal(pos, [1, 1, 1])
            self.assertFalse(episode.out_of_axis(0))
            episode.agent[1] = np.array([1,1.2,1.3])
            episode.action[0]=4
            np.testing.assert_array_equal(episode.calculate_reward(0), 1)
            episode.action[1] = 4
            episode.agent[2] = np.array([0, 0.1, -0.3])
            np.testing.assert_array_equal(episode.calculate_reward(1), 0)
            episode.agent[2] = np.array([2, 1.8, 2.2])
            np.testing.assert_array_equal(episode.calculate_reward(1), -1/4.0)
            episode.action[2] = 4
            episode.agent[3] = np.array([0, 0.1, -0.1])
            np.testing.assert_array_equal(episode.calculate_reward(2), 0)
            episode.agent[3] = np.array([1, .9, 1.2])
            np.testing.assert_almost_equal(episode.calculate_reward(2), 1+(0.1*np.sqrt(5)))

    def test_two_pavement_and_obstacle(self):
        seq_len=4
        cars, people, tensor, gamma = self.initialize(seq_len)
        for i in [8]:
            tensor[1, 1, 1, 3] = i / NUM_SEM_CLASSES
            tensor[1, 0, 1, 3] = i / NUM_SEM_CLASSES
            tensor[2, 2, 2, 3] = 11 / NUM_SEM_CLASSES
            pos_x = 0
            pos_y = 0
            # rewards=[-1, 1, -1, 1, 0, 0,0,0,0,0,0,0,0,0,0]
            # rewards[2]=1
            # rewards[3] =-1
            # rewards[4] = 1

            episode = self.get_episode(cars, gamma, people, tensor,seq_len)

            episode.agent_size = np.array([0, 0, 0])

            pos, i , vel= episode.initial_position(None)
            episode.agent[0] =np.array( [1, 1.1, 1])
            episode.agent[1] = np.array( [1, 1, 1.2])
            episode.action[0]=4
            episode.action[1] = 4
            np.testing.assert_array_equal(episode.calculate_reward(0), 1)
            episode.agent[2] = np.array( [0, 0, 0.1])
            episode.action[2] = 4
            np.testing.assert_array_equal(episode.calculate_reward(1), 0)
            episode.agent[2] =np.array( [2, 2.2, 2])
            np.testing.assert_array_equal(episode.calculate_reward(1), -1/4.0)
            episode.action[0] = 3
            episode.agent[3] =np.array(  [0, 0.1, -.4])
            np.testing.assert_array_equal(episode.calculate_reward(2), 0)
            episode.agent[3] = np.array( [1, 0, 1.2])

            np.testing.assert_array_equal(episode.calculate_reward(2), 1+np.sqrt(1.25))

    # TO DO: Update reward check!
    def test_reward(self):
        cars, people, tensor, gamma = self.initialize()
        for i in [8]:
            tensor[1, 1, 1, 3] = i / NUM_SEM_CLASSES
            pos_x = 0
            pos_y = 0
            rewards = np.zeros(15)
            # rewards[2] = 1
            # rewards[3] = -1
            # rewards[4] = 1
            # rewards[5] = -1

            episode = self.get_episode(cars, gamma, people, tensor)
            Simple = episode.find_sidewalk(False)
            episode.agent_size=[0,0,0]
            np.testing.assert_array_equal(Simple[0], [1])
            np.testing.assert_array_equal(Simple[1], [1])
            np.testing.assert_array_equal(Simple[2], [1])
            pos, i , vel= episode.initial_position(None, initialization=8)
            np.testing.assert_array_equal(pos, [1, 1, 1])
            self.assertFalse(episode.out_of_axis(0))

            episode.agent[1] = np.array([0, 0, -.9])
            episode.action[0] = 4
            np.testing.assert_array_equal( episode.calculate_reward( 0), -1)


            episode.agent[2] = np.array([0, -.9, 0])
            episode.action[1] = 4
            self.assertTrue(episode.out_of_axis(1))
            np.testing.assert_array_equal(episode.calculate_reward( 1), -1)

            episode.agent[3] = np.array([0, -1.1, 0])
            episode.action[2] = 4
            self.assertTrue(episode.out_of_axis(2))
            np.testing.assert_array_equal(episode.calculate_reward(2),-1)


            for j in range(4,29):
                episode.agent[j]=np.array([0,0.1,0])
                episode.action[j-1] = 4
                self.assertFalse(episode.out_of_axis(j))
                np.testing.assert_array_equal(episode.calculate_reward(j-1), 0)
            import math
            episode.agent[29] = np.array([0, 0, 0.2])
            episode.action[28] = 4
            self.assertFalse(episode.out_of_axis(29))
            np.testing.assert_array_equal(episode.calculate_reward(28),0)
            episode.discounted_reward()
            # for i in range(26):
            #     np.testing.assert_array_almost_equal_nulp(episode.reward[28-i], 0, nulp=2 )
            #     np.testing.assert_array_almost_equal_nulp(episode.reward_d[28-i], 0, nulp=2)
            #
            # np.testing.assert_array_almost_equal_nulp(episode.reward_d[3], -1, nulp=2)
            # np.testing.assert_array_equal(episode.reward[3], -1)
            # np.testing.assert_array_almost_equal_nulp(episode.reward_d[2], -1.99, nulp=2)
            # np.testing.assert_array_equal(episode.reward[2], -1)
            # np.testing.assert_array_almost_equal_nulp(episode.reward_d[1], -1.99*.99-1, nulp=2)
            # np.testing.assert_array_equal(episode.reward[1], -1)
            # np.testing.assert_array_almost_equal_nulp(episode.reward_d[0], (-1.99 * .99 - 1)*.99+1, nulp=2)
            # np.testing.assert_array_equal(episode.reward[0], 1)

    # Test the neigbourhood that the agent sees when inside allowed axis.
    def test_neigbourhood_in_axis(self):
        cars, people, tensor, gamma = self.initialize()
        tensor = np.zeros((8, 4, 4, 6))
        tensor[0,:,:,4]=np.array([[1,2,3,4], [5,6,7,8],[9,10,11,12],[13,14,15,16]])
        pos_x = 0
        pos_y = 0
        episode = self.get_episode(cars, gamma, people, tensor)
        episode.agent[0]=[0,2.1,2.1]
        neigh,_,_,_,_=episode.get_agent_neighbourhood(episode.agent[0], [0, 1, 1],0)
        expected=np.array([[6,7,8],[10,11,12],[14,15,16]])
        np.testing.assert_array_equal(neigh[0,:,:,4],expected )
        episode.agent[1] = [0, 2.2, 3.1]

    # Test the neigbourhood that the agent sees when outside allowed axis.
    def test_neigbourhood_out_of_axis(self):
        cars, people, tensor, gamma = self.initialize()
        tensor = np.zeros((8, 4, 4, 6))
        tensor[0,:,:,4]=np.array([[1,2,3,4], [5,6,7,8],[9,10,11,12],[13,14,15,16]])
        pos_x = 0
        pos_y = 0
        episode = self.get_episode(cars, gamma, people, tensor)
        episode.agent[0]=[0,2.1,3.1]
        neigh,_,_,_,_=episode.get_agent_neighbourhood(episode.agent[0], [0, 1, 1],0)
        expected=np.array([[7,8,0],[11,12,0],[15,16,0]])
        np.testing.assert_array_equal(neigh[0,:,:,4],expected )
        episode.agent[1] = [0, 2.7, 3.1]
        neigh,_,_,_,_ = episode.get_agent_neighbourhood(episode.agent[1],[ 0, 1, 1], 1)
        expected = np.array([ [11, 12, 0], [15, 16, 0],[0,0,0]])
        np.testing.assert_array_equal(neigh[0, :, :, 4], expected)
        episode.agent[2] = [0, 4.1, 3.2]
        neigh,_,_,_,_ = episode.get_agent_neighbourhood(episode.agent[2], [0, 1, 1],2)
        expected = np.array([ [15, 16, 0], [0, 0, 0],[0,0,0]])
        np.testing.assert_array_equal(neigh[0, :, :, 4], expected)

    # Test the neigbourhood that the agent sees when on border to axis.
    def test_neigbourhood_out_of_axis_min(self):
        cars, people, tensor, gamma = self.initialize()
        tensor = np.zeros((8, 4, 4, 6))
        tensor[0,:,:,4]=np.array([[1,2,3,4], [5,6,7,8],[9,10,11,12],[13,14,15,16]])
        pos_x = 0
        pos_y = 0
        episode = self.get_episode(cars, gamma, people, tensor)
        episode.agent[0]=[0,1.1,0]
        neigh,_,_,_,_=episode.get_agent_neighbourhood(episode.agent[0], [0, 1, 1],0)
        expected=np.array([[0,1,2],[0,5,6],[0,9,10]])
        np.testing.assert_array_equal(neigh[0,:,:,4],expected )
        episode.agent[1] = [0, 0, 1.1]
        neigh,_,_,_,_ = episode.get_agent_neighbourhood(episode.agent[1], [0, 1, 1],1)
        expected = np.array([[0,0,0],[1, 2, 3], [5, 6, 7]])
        np.testing.assert_array_equal(neigh[0, :, :, 4], expected)
        episode.agent[2] = [0, 0.1, 0]
        neigh,_,_,_,_ = episode.get_agent_neighbourhood(episode.agent[2], [0, 1, 1],2)
        expected = np.array([[0, 0, 0], [0,1, 2], [0,5, 6]])
        np.testing.assert_array_equal(neigh[0, :, :, 4], expected)


    # # Test initialization when there is a pedestrian.
    # def test_one_person_NN_overlap(self):
    #     cars, people, tensor, gamma = self.initialize()
    #     people[29].append(np.zeros((3, 2)))
    #     mean = np.mean(people[29][0], axis=1)
    #     for i in [8]:
    #         tensor[0, 2, 0, 3] = i / 33.0
    #         tensor[0, 0, 0, 3] = i / 33.0
    #         pos_x = 0
    #         pos_y = 0
    #         episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, 30,
    #                                 (-1, 0, 1, -1, 1, -1, 0, -1, 0, 1, 0, 0, 0,0,0), agent_size=(0, 0, 0), defaultSettings=run_settings())
    #         episode.agent_size = [0, 0, 0]
    #         for nbr in range(29):
    #             episode.agent[nbr] = [0, 0, 0]
    #         #pos, i , vel= episode.initial_position(None, initialization=10)
    #         episode.agent[1] = [2.1, 2.1, 2.1]
    #         # self.assertEqual(i, 29)
    #         self.assertEqual(len(episode.intercept_person(0)), 1)
    #         episode.agent[1] = [2.1, 2.1, 2.1]
    #
    #         self.assertEqual(len(episode.intercept_person(1)), 0)
    #
    #
    # def test_one_person_NN_overlap2(self):
    #     cars, people, tensor, gamma = self.initialize()
    #     people[29].append(np.zeros((3, 2)))
    #     mean = np.mean(people[29][0], axis=1)
    #     for i in [8]:
    #         tensor[0, 2, 0, 3] = i / 33.0
    #         tensor[0, 0, 0, 3] = i / 33.0
    #         pos_x = 0
    #         pos_y = 0
    #         episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, 30,
    #                                 (-1, 0, 1, -1, 1, -1, 0, -1, 0, 1, 0, 0, 0,0,0), agent_size=(0, 0, 0), defaultSettings=run_settings())
    #         episode.agent_size = [0, 0, 0]
    #         for nbr in range(29):
    #             episode.agent[nbr] = [0, 0.2, -0.1]
    #         #pos, i , vel= episode.initial_position(None, initialization=10)
    #         #np.testing.assert_array_equal(pos, mean)
    #         # self.assertEqual(i, 29)
    #         self.assertEqual(len(episode.intercept_person(0)), 1)
    #         episode.agent[1] = [1, .9, 1.2]
    #
    #         self.assertEqual(len(episode.intercept_person(1)), 0)


    #
    # def test_one_person_NN_overlap3(self):
    #     cars, people, tensor, gamma = self.initialize()
    #     people[29].append(np.zeros((3, 2)))
    #     mean = np.mean(people[29][0], axis=1)
    #     for i in [8]:
    #         tensor[0, 2, 0, 3] = i / 33.0
    #         tensor[0, 0, 0, 3] = i / 33.0
    #         pos_x = 0
    #         pos_y = 0
    #         episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, 30,
    #                                 (-1, 0, 1, -1, 1, -1, 0, -1, 0, 1, 0, 0, 0,0,0), agent_size=(0, 0, 0), defaultSettings=run_settings())
    #         episode.agent_size = [0, 0, 0]
    #
    #         pos, i , vel= episode.initial_position(None, initialization=9)
    #         for nbr in range(29):
    #             episode.agent[nbr] =[2, 2.1, 1.9]
    #         np.testing.assert_array_equal(pos, mean)
    #         # self.assertEqual(i, 29)
    #
    #         episode.agent[1] = [2, 1.9, 2.1]
    #
    #
    #
    # def test_one_person_NN_overlap4(self):
    #     cars, people, tensor, gamma = self.initialize()
    #     people[29].append(np.zeros((3,2)))
    #     people[28].append(np.zeros((3, 2)))
    #     mean = np.mean(people[29][0], axis=1)
    #     for i in [8]:
    #         tensor[0, 2, 0, 3] = i / 33.0
    #         tensor[0, 0, 0, 3] = i / 33.0
    #         pos_x = 0
    #         pos_y = 0
    #         episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, 30,
    #                                 (-1, 0, 1, -1, 1, -1, 0, -1, 0, 1, 0, 0, 0,0,0), agent_size=(0, 0, 0), defaultSettings=run_settings())
    #         episode.agent_size = [0, 0, 0]
    #
    #         pos, i , vel= episode.initial_position(None, initialization=9)
    #         for nbr in range(29):
    #             episode.agent[nbr] = [0, 0.1, -0.1]
    #         np.testing.assert_array_equal(pos, mean)
    #         # self.assertEqual(i, 29)
    #
    #         episode.agent[1] = [2, -1.9, 2.2]

    #
    #
    # def test_one_person_NN_overlap5(self):
    #     cars, people, tensor, gamma = self.initialize()
    #     people[29].append(np.zeros((3,2)))
    #     people[28].append(np.ones((3,2)))
    #     mean = np.mean(people[29][0], axis=1)
    #     for i in [8]:
    #         tensor[0, 2, 0, 3] = i / 33.0
    #         tensor[0, 0, 0, 3] = i / 33.0
    #         pos_x = 0
    #         pos_y = 0
    #         episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, 30,
    #                                 (-1, 0, 1, -1, 1, -1, 0, -1, 0, 1, 0, 0, 0,0,0), agent_size=(0, 0, 0), defaultSettings=run_settings())
    #         episode.agent_size = [1, 1, 1]
    #
    #         pos, i , vel= episode.initial_position(None, initialization=9)
    #         for nbr in range(29):
    #             episode.agent[nbr] = [0, 0.1, -0.1]
    #
    #         # self.assertEqual(i, 29)
    #
    #         #episode.agent[1] = [2, 2, 2]
    #
    #
    # def test_one_person_NN_overlap6(self):
    #     cars, people, tensor, gamma = self.initialize()
    #     people[29].append(np.zeros((3,2)))
    #     people[29].append(np.ones((3, 2)))
    #     mean = np.mean(people[29][0], axis=1)
    #     for i in [8]:
    #         tensor[0, 2, 0, 3] = i / 33.0
    #         tensor[0, 0, 0, 3] = i / 33.0
    #         pos_x = 0
    #         pos_y = 0
    #         episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, 30,
    #                                 (-1, 0, 1, -1, 1, -1, 0, -1, 0, 1, 0, 0, 0,0,0), agent_size=(0, 0, 0), defaultSettings=run_settings())
    #         episode.agent_size = [1, 1, 1]
    #
    #         pos, i , vel= episode.initial_position(None, initialization=9)
    #         for nbr in range(29):
    #             episode.agent[nbr] = [0, 0.1, -0.2]
    #
    #         # self.assertEqual(i, 29)
    #
    #         # episode.agent[1] = [2, 2, 2]
    #
    #
    #
    # def test_one_person_NN_overlap7(self):
    #     cars, people, tensor, gamma = self.initialize()
    #     people[29].append(np.zeros((3, 2)))
    #     people[29].append(np.ones((3, 2)))
    #     mean = np.mean(people[29][0], axis=1)
    #     for i in [8]:
    #         tensor[0, 2, 0, 3] = i / 33.0
    #         tensor[0, 0, 0, 3] = i / 33.0
    #         pos_x = 0
    #         pos_y = 0
    #         episode = SimpleEpisode(tensor, people, cars, pos_x, pos_y, gamma, 30,
    #                                 (-1, 0, 1, -1, 1, -1, 0, -1, 0, 1, 0, 0, 0,0,0), agent_size=(0, 0, 0), defaultSettings=run_settings())
    #         episode.agent_size = [1, 1, 1]
    #
    #         pos, i , vel= episode.initial_position(None, initialization=9)
    #         for nbr in range(29):
    #             episode.agent[nbr] = [2, 2.1, 2.2]
    #
    #         # self.assertEqual(i, 29)
    #
    #         # episode.agent[1] = [2, 2, 2]


    def test_one_car_input(self):
        cars, people, tensor, gamma = self.initialize()
        for i in range (3):
            for j in range(3):
                cars[i*3+j].append([0,8,i,i,j,j])

        for i in [8]:
            tensor[0, 1, 1, 3] = i / NUM_SEM_CLASSES
            # tensor[0, 1, 1, 3] = i / NUM_SEM_CLASSES
            episode = self.get_episode(cars, gamma, people, tensor)
            episode.agent_size = [0, 0, 0]

            pos, i, vel = episode.initial_position(None, initialization=8)
            for i in range(3):
                for j in range(3):
                    episode.agent[i*3+j]=[0,i,j]

            #Agent standing still!

            for l in range(9):
                k = 0
                pos=episode.agent[l]
                for i in range(3):
                    for j in range(3):
                        #print "Position "+str(pos)+" "+str(cars[k])+" "+str(k)
                        if i-pos[1]!=0:
                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0],1.0/(i-pos[1]) )
                        else:
                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0], 2 )
                        if j - pos[2] != 0:

                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1], 1.0/(j - pos[2]))
                        else:
                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1], 2 )
                        k=k+1

    def test_two_cars_input(self):
        cars, people, tensor, gamma = self.initialize()
        for i in range (3):
            for j in range(3):
                cars[i*3+j].append([0,8,i,i,j,j])
                cars[i * 3 + j].append([0, 8, 2, 2, 2, 2])
        #print("Cars "+str(cars))
        for i in [8]:
            tensor[0, 1, 1, 3] = i / NUM_SEM_CLASSES
            # tensor[0, 1, 1, 3] = i / NUM_SEM_CLASSES
            episode = self.get_episode(cars, gamma, people, tensor)
            episode.agent_size = [0, 0, 0]

            pos, i, vel = episode.initial_position(None, initialization=8)
            for i in range(3):
                for j in range(3):
                    episode.agent[i*3+j]=[0,i,j]
            #print("Agent " + str(episode.agent))
            #Agent standing still!

            for l in range(2):
                k = 0
                pos=episode.agent[l]
                for i in range(3):
                    for j in range(3):

                        if i-pos[1]!=0 :
                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0],1.0/(i-pos[1]) )
                        else:
                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0],2)
                        if j - pos[2] != 0 :
                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1], 1.0/(j - pos[2]))
                        else:
                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1], 2)
                        k=k+1
            k = 0
            pos = episode.agent[8]
            print(" Position "+str(pos))
            for i in range(3):
                for j in range(3):
                    #print episode.get_input_cars_cont(pos, k)
                    self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0], 2.0)
                    self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1], 2)
                    k = k + 1

            k = 0 # equal distance
            pos = [0,1,0]
            for i in range(3):
                for j in range(3):
                    # print "Position "+str(pos)+" "+str(cars[k])+" "+str(k)
                    if i - pos[1] != 0:
                        self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0], 1.0 / (i - pos[1]))
                    else:
                        self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0], 2)
                    if j - pos[2] != 0:
                        self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1], 1.0 / (j - pos[2]))
                    else:
                        self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1], 2)
                    k = k + 1
            k = 0  # equal distance
            pos = [0, 0, 1]
            for i in range(3):
                for j in range(3):
                    # print "Position "+str(pos)+" "+str(cars[k])+" "+str(k)
                    if i - pos[1] != 0:
                        self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0], 1.0 / (i - pos[1]))
                    else:
                        self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0], 2)
                    if j - pos[2] != 0:
                        self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1], 1.0 / (j - pos[2]))
                    else:
                        self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1], 2)
                    k = k + 1
            k=0 # closest car as close
            pos = [0, 1, 1]
            for i in range(3):
                for j in range(3):
                    # print "Position "+str(pos)+" "+str(cars[k])+" "+str(k)
                    if i - pos[1] != 0:
                        self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0], 1.0 / (i - pos[1]))
                    else:
                        self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0], 2)
                    if j - pos[2] != 0:
                        self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1], 1.0 / (j - pos[2]))
                    else:
                        self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1], 2)
                    k = k + 1

            k = 0
            pos = [0, 0, 2]
            for i in range(3):
                for j in range(3):
                    #print "Position "+str(pos)+" "+str(cars[k])+" "+str(k)
                    if k==6 or k==3 or k==7:
                        self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0], 1/2.0)
                    else:
                        if i - pos[1] != 0:
                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0], 1.0 / (i - pos[1]))
                        else:
                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0], 2)
                        if j - pos[2] != 0:
                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1], 1.0 / (j - pos[2]))
                        else:
                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1], 2)
                    k = k + 1
            k = 0
            pos = [0, 1, 2]
            for i in range(3):
                for j in range(3):
                    # print "Position "+str(pos)+" "+str(cars[k])+" "+str(k)
                    if k == 5 or k ==  4or k == 8 or k == 2:
                        if i - pos[1] != 0:
                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0], 1.0 / (i - pos[1]))
                        else:
                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0], 2)
                        if j - pos[2] != 0:
                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1], 1.0 / (j - pos[2]))
                        else:
                            self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1], 2)
                    else:

                        self.assertEqual(episode.get_input_cars_cont(pos, k)[0,0],1 )
                        self.assertEqual(episode.get_input_cars_cont(pos, k)[0,1],2)

                    k = k + 1

    def test_goal_input(self):
        cars, people, tensor, gamma = self.initialize()
        goals=[]
        for i in range(3):
            for j in range(3):
                goals.append([0,i, j])

        for i in [8]:
            tensor[0, 1, 1, 3] = i / NUM_SEM_CLASSES
            # tensor[0, 1, 1, 3] = i / NUM_SEM_CLASSES
            episode = self.get_episode(cars, gamma, people, tensor)
            episode.agent_size = [0, 0, 0]

            pos, i, vel = episode.initial_position(None, initialization=8)
            for i in range(3):
                for j in range(3):
                    episode.agent[i * 3 + j] = [0, i, j]

            # Agent standing still!

            for l in range(9):
                k = 0
                episode.goal=goals[l]


                for i in range(3):
                    for j in range(3):
                        pos = episode.agent[k]
                        #print "Test Goal "+str(goals[l])+" pos "+str(pos)+" dist "+str(goals[l][2]-pos[2])

                        np.testing.assert_approx_equal(episode.get_goal_dir_cont(pos,goals[l])[0,0],(goals[l][1]-pos[1])/np.sqrt(episode.reconstruction.shape[1]**2+episode.reconstruction.shape[2]**2) )
                        np.testing.assert_approx_equal(episode.get_goal_dir_cont(pos, goals[l])[0,1],(goals[l][2]-pos[2])/np.sqrt(episode.reconstruction.shape[1]**2+episode.reconstruction.shape[2]**2))
                        k=k+1


if __name__ == '__main__':
    unittest.main()
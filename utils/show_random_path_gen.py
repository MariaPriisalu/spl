import unittest
import numpy as np
import sys
sys.path.append("RL/")
from RL.random_path_generator import random_path, random_walk
from RL.visualization import view_2D
import matplotlib.pyplot as plt
# Test methods in episode.

class TestRandomPath(unittest.TestCase):
    def test_one_path(self):
        tensor=np.zeros((8,128, 265,6), dtype=np.float32)
        width=5
        num_crossings=2
        # tensor, point_1, thetas, steps, cars
        tensor,point ,thetas, steps, cars=random_path(tensor, width, num_crossings)
        img_from_above=np.zeros((tensor.shape[1],tensor.shape[2],3),  dtype=np.uint8)
        img=view_2D(tensor, img_from_above, 0)

        fig = plt.figure
        plt.imshow(img)
        plt.show()

    def test_random_walk(self):
        tensor = np.zeros((8, 128, 265, 6), dtype=np.float32)
        width = 5
        num_steps=300
        tensor , pos, cars= random_walk(tensor, width, num_steps)
        img_from_above = np.zeros((tensor.shape[1], tensor.shape[2], 3), dtype=np.uint8)
        img = view_2D(tensor, img_from_above, 0)

        fig = plt.figure
        plt.imshow(img)
        plt.show()

    def test_random_walk_vert(self):
        tensor = np.zeros((8, 128, 265, 6), dtype=np.float32)
        width = 5
        num_steps=300
        tensor, pos, cars = random_walk(tensor, width, num_steps, horizontal=False)
        img_from_above = np.zeros((tensor.shape[1], tensor.shape[2], 3), dtype=np.uint8)
        img = view_2D(tensor, img_from_above, 0)
        fig = plt.figure
        plt.imshow(img)
        plt.show()

    def test_random_walk_diagonal(self):
        tensor = np.zeros((8, 128, 265, 6), dtype=np.float32)
        width = 5
        num_steps = 300
        tensor, pos, cars = random_walk(tensor, width, num_steps, diagonal=True)

        img_from_above = np.zeros((tensor.shape[1], tensor.shape[2], 3), dtype=np.uint8)
        img = view_2D(tensor, img_from_above, 0)

        fig = plt.figure
        plt.imshow(img)
        plt.show()

    def test_random_walk_diagonal_neg(self):
        tensor = np.zeros((8, 128, 265, 6), dtype=np.float32)
        width = 5
        num_steps = 300
        tensor, pos, cars= random_walk(tensor, width, num_steps, diagonal=True, dir=-1)

        img_from_above = np.zeros((tensor.shape[1], tensor.shape[2], 3), dtype=np.uint8)
        img = view_2D(tensor, img_from_above, 0)

        fig = plt.figure
        plt.imshow(img)
        plt.show()




if __name__ == '__main__':
    unittest.main()
import unittest
import numpy as np
from RL.extract_tensor import frame_reconstruction, objects_in_range


class TestExtractTensor(unittest.TestCase):

    def test_empty(self):
        cars_a, people_a, tensor = self.initialize()
        out, cars_predicted,people_predicted, reconstruction_2D=frame_reconstruction(tensor, cars_a, people_a)
        np.testing.assert_array_equal(out, tensor)

    def test_one_car(self):
        cars_a, people_a, tensor = self.initialize()
        expected=tensor.copy()
        expected[1,1,1,5]=0.1
        cars_a[0].append((1,1,1,1,1,1))
        out, cars_predicted,people_predicted, reconstruction_2D=frame_reconstruction(tensor, cars_a, people_a)
        np.testing.assert_array_equal(out, expected)

    def test_one_person(self):
        cars_a, people_a, tensor = self.initialize()
        expected=tensor.copy()
        expected[1,1,1,4]=0.1

        people_a[0].append(np.ones((3,2), dtype=np.int))
        out, cars_predicted,people_predicted, reconstruction_2D=frame_reconstruction(tensor, cars_a, people_a)
        np.testing.assert_array_equal(out, expected)

    def test_two_cars(self):
        cars_a, people_a, tensor = self.initialize()
        expected=tensor.copy()
        cars_a[0].append((0,0,0,0,0,0))
        cars_a[0].append((0,0, 0, 0, 2,2))
        cars_a[2].append((0,0, 0, 0, 0,1))
        expected[0, 0, 0, 5] = 0.2
        expected[0, 0, 1, 5] = 0.1
        expected[0, 0, 2, 5] = 0.1
        out, cars_predicted,people_predicted, reconstruction_2D=frame_reconstruction(tensor, cars_a, people_a)
        np.testing.assert_array_equal(out, expected)

    def test_many_cars(self):
        cars_a, people_a, tensor = self.initialize()
        expected=tensor.copy()
        indx=0
        for x in range(0,3):
            for y in range(0,3):
                cars_a[indx].append((0,0, x, x, y,y))
                indx+=1
        expected[0,:,:,5]=0.1*np.ones(expected.shape[1:3])
        out, cars_predicted,people_predicted, reconstruction_2D=frame_reconstruction(tensor, cars_a, people_a)
        np.testing.assert_array_equal(out, expected)
    #         expected[0, :, :, 5] = 0.1 * np.ones(expected.shape[1:3]) #
    #tensor, cars_a, people_a, no_dict=True, temporal=False, predict_future=False, run_2D=False, reconstruction_2D=[]
    # def test_many_car_dict(self):
    #     cars_a, people_a, tensor = self.initialize()
    #     expected = tensor.copy()
    #     indx = 0
    #     for x in range(0, 3):
    #         for y in range(0, 3):
    #             cars_a[indx].append((0, 0, x, x, y, y))
    #             indx += 1
    #     expected[0, :, :, 5] = 0.1 * np.ones(expected.shape[1:3])
    #     # reconstruction, cars_predicted,people_predicted, reconstruction_2D
    #     #frame_reconstruction(tensor, cars_a, people_a, no_dict=True, temporal=False, predict_future=False, run_2D=False, reconstruction_2D=[])
    #     out, cars_predicted, people_predicted, reconstruction_2D = frame_reconstruction(tensor, cars_a, people_a, no_dict=False,  temporal=True,  predict_future=True, run_2D=True, reconstruction_2D=[])
    #     np.testing.assert_array_equal(out, expected)

    def test_many_people(self):
        cars_a, people_a, tensor = self.initialize()
        expected=tensor.copy()
        indx=0
        #vals=[[0,1],[0,2],[1,3],[2,3]]
        for x in range(0,3):
            for y in range(0,3):
                people_a[indx].append(np.array([[0,0], [y,y],[x, x]], np.int32))
                indx+=1
        expected[0, :, :, 4] = 0.1 * np.ones(expected.shape[1:3])
        out, cars_predicted,people_predicted, reconstruction_2D=frame_reconstruction(tensor, cars_a, people_a)
        np.testing.assert_array_equal(out, expected)

    def initialize(self):
        tensor = np.zeros((3, 3, 3, 6))
        people_a = []
        cars_a = []
        for i in range(30):
            people_a.append([])
            cars_a.append([])

        return cars_a, people_a, tensor

    def test_obj_in_range_corner(self):
        cars_a, people_a, tensor = self.initialize()
        cars_a[0].append((2,3,2,3,2,3))
        out=objects_in_range(cars_a, 0, 0, 3,3)
        people_a[0].append((2,3,2,3,2,3))
        np.testing.assert_array_equal(out, people_a)
        people_a[0]=[]
        out = objects_in_range(cars_a, 3, 0, 3, 3)
        people_a[0].append((2,3, 2, 3, -1, 0))
        np.testing.assert_array_equal(out, people_a)
        people_a[0] = []
        out = objects_in_range(cars_a, 0, 3, 3, 3)
        people_a[0].append((2, 3,-1, 0, 2, 3))
        np.testing.assert_array_equal(out, people_a)
        people_a[0] = []
        out = objects_in_range(cars_a, 3, 3, 3, 3)
        people_a[0].append((2, 3, -1, 0, -1, 0))
        np.testing.assert_array_equal(out, people_a)

    def test_obj_in_range(self):
        cars_a, people_a, tensor = self.initialize()
        cars_a[0].append((2, 3, 3, 3, 2, 3))
        out = objects_in_range(cars_a, 0, 0, 3, 3)
        np.testing.assert_array_equal(out, people_a)

        out = objects_in_range(cars_a, 3, 0, 3, 3)
        np.testing.assert_array_equal(out, people_a)

        out = objects_in_range(cars_a, 0, 3, 3, 3)
        people_a[0].append((2, 3,0, 0, 2, 3))
        np.testing.assert_array_equal(out, people_a)

        people_a[0] = []
        out = objects_in_range(cars_a, 3, 3, 3, 3)
        people_a[0].append((2, 3, 0, 0, -1, 0))
        np.testing.assert_array_equal(out, people_a)

        cars_a[0]=[]
        people_a[0] = []
        cars_a[0].append((3, 5, 3, 5, 3, 5))
        out = objects_in_range(cars_a, 0, 0, 3, 3)
        np.testing.assert_array_equal(out, people_a)

        out = objects_in_range(cars_a, 0, 3, 3, 3)
        np.testing.assert_array_equal(out, people_a)

        out = objects_in_range(cars_a, 3, 0, 3, 3)
        np.testing.assert_array_equal(out, people_a)

        people_a[0].append((3,5,0,2,0,2))
        out = objects_in_range(cars_a, 3, 3, 3, 3)
        np.testing.assert_array_equal(out, people_a)

    def test_people_in_range_corner(self):
        cars_a, people_a, tensor = self.initialize()
        cars_a[0].append(np.array([[2, 3],[2, 3] ,[2, 3]]))
        out = objects_in_range(cars_a, 0, 0, 3, 3)
        people_a[0].append(np.array([[2, 3],[2, 3] ,[2, 3]]))
        np.testing.assert_array_equal(out[0], people_a[0])
        people_a[0] = []
        out = objects_in_range(cars_a, 3, 0, 3, 3)
        people_a[0].append(np.array([[2, 3],[2, 3] ,[-1, 0]]))
        np.testing.assert_array_equal(out[0], people_a[0])
        people_a[0] = []
        out = objects_in_range(cars_a, 0, 3, 3, 3)
        people_a[0].append(np.array([[2, 3],[-1, 0] ,[2, 3]]))
        np.testing.assert_array_equal(out[0], people_a[0])
        people_a[0] = []
        out = objects_in_range(cars_a, 3, 3, 3, 3)
        people_a[0].append(np.array([[2, 3],[-1, 0] ,[-1, 0]]))
        np.testing.assert_array_equal(out[0], people_a[0])

    def test_obj_in_range1(self):
        cars_a, people_a, tensor = self.initialize()
        cars_a[0].append(np.array([[2, 3],[3, 3] ,[2, 3]]))
        out = objects_in_range(cars_a, 0, 0, 3, 3)
        np.testing.assert_array_equal(out[0], people_a[0])

        out = objects_in_range(cars_a, 3, 0, 3, 3)
        np.testing.assert_array_equal(out[0], people_a[0])

        out = objects_in_range(cars_a, 0, 3, 3, 3)
        people_a[0].append(np.array([[2, 3],[0, 0] ,[2, 3]]))
        np.testing.assert_array_equal(out[0], people_a[0])

        people_a[0] = []
        out = objects_in_range(cars_a, 3, 3, 3, 3)
        people_a[0].append(np.array([[2, 3],[0, 0] ,[-1, 0]]))
        np.testing.assert_array_equal(out[0], people_a[0])

        cars_a[0] = []
        people_a[0] = []
        cars_a[0].append(np.array([[3, 5],[3, 5] ,[3, 5]]))
        out = objects_in_range(cars_a, 0, 0, 3, 3)
        np.testing.assert_array_equal(out[0], people_a[0])

        out = objects_in_range(cars_a, 0, 3, 3, 3)
        np.testing.assert_array_equal(out[0], people_a[0])

        out = objects_in_range(cars_a, 3, 0, 3, 3)
        np.testing.assert_array_equal(out[0], people_a[0])

        people_a[0].append(np.array([[3, 5],[0, 2] ,[0, 2]]))
        out = objects_in_range(cars_a, 3, 3, 3, 3)
        np.testing.assert_array_equal(out[0], people_a[0])

    def test_random_people_in_range(self):
        cars_a, people_a, tensor = self.initialize()
        people_a[0].append(np.random.randint(3, size=(3,14)))
        people_a[0].append(np.random.randint(3, size=(3, 14))+3*np.ones((3,14)))
        out = objects_in_range(people_a, 3, 3, 3, 3)
        np.testing.assert_array_equal(1, len(out[0]))
        out = objects_in_range(people_a, 0,0, 3, 3)
        np.testing.assert_array_equal(1, len(out[0]))
        out = objects_in_range(people_a, 0, 3, 3, 3)
        np.testing.assert_array_equal(0, len(out[0]))
        out = objects_in_range(people_a, 3, 0, 3, 3)
        np.testing.assert_array_equal(0, len(out[0]))


if __name__ == '__main__':
    unittest.main()
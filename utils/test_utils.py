import unittest
from utils.utils_functions import in_square, overlap
from triangulate_cityscapes.utils_3D import framepoints_to_collection, refine_voxel_values
import numpy as np
from RL.settings import NUM_SEM_CLASSES_ASINT


class TestUtils(unittest.TestCase):
    def test_overlap_voxel(self):
        cluster1=[1,1,1,1,1,1]
        self.assertFalse(overlap(cluster1, [0,0,0,0,0,0],1), "Overlap")
        self.assertFalse(overlap(cluster1, [0, 1, 0, 1, 0, 0], 1), "Overlap")
        self.assertFalse(overlap(cluster1, [0, 0, 0, 1, 0, 1], 1), "Overlap")
        self.assertFalse(overlap(cluster1, [0, 1, 0, 0, 0, 1], 1), "Overlap")
        self.assertTrue(overlap(cluster1, [0, 1, 0, 1, 0, 1], 1), "Overlap")

        self.assertFalse(overlap( [0, 0, 0, 0, 0, 0], cluster1,1), "Overlap")
        self.assertFalse(overlap( [0, 1, 0, 1, 0, 0],cluster1, 1), "Overlap")
        self.assertFalse(overlap( [0, 0, 0, 1, 0, 1],cluster1, 1), "Overlap")
        self.assertFalse(overlap( [0, 1, 0, 0, 0, 1],cluster1, 1), "Overlap")
        self.assertTrue(overlap( [0, 1, 0, 1, 0, 1],cluster1, 1), "Overlap")

        self.assertFalse(overlap(cluster1, [0, 0, 0, 0, 0, 0], 1), "Overlap")
        self.assertFalse(overlap(cluster1, [1, 1, 1, 1, 0, 0], 1), "Overlap")
        self.assertFalse(overlap(cluster1, [0, 0, 1, 1, 1, 1], 1), "Overlap")
        self.assertFalse(overlap(cluster1, [1, 1, 0, 0, 1, 1], 1), "Overlap")
        self.assertTrue(overlap(cluster1, [1, 1, 1, 1, 1, 1], 1), "Overlap")

        self.assertFalse(overlap([0, 0, 0, 0, 0, 0], cluster1,1), "Overlap")
        self.assertFalse(overlap([1, 1, 1, 1, 0, 0], cluster1, 1), "Overlap")
        self.assertFalse(overlap([0, 0, 1, 1, 1, 1],  cluster1,1), "Overlap")
        self.assertFalse(overlap( [1, 1, 0, 0, 1, 1],cluster1, 1), "Overlap")
        self.assertTrue(overlap( [1, 1, 1, 1, 1, 1],  cluster1,1), "Overlap")

        self.assertFalse(overlap(cluster1, [2, 2,2, 2, 2, 2], 1), "Overlap")
        self.assertFalse(overlap(cluster1, [1, 1, 1, 1,2, 2], 1), "Overlap")
        self.assertFalse(overlap(cluster1, [2, 2, 1, 1, 1, 1], 1), "Overlap")
        self.assertFalse(overlap(cluster1, [1, 1,2, 2, 1, 1], 1), "Overlap")
        self.assertTrue(overlap(cluster1, [1, 1, 1, 1, 1, 1], 1), "Overlap")

        self.assertFalse(overlap( [2, 2, 2, 2, 2, 2],  cluster1,1), "Overlap")
        self.assertFalse(overlap( [1, 1, 1, 1, 2, 2], cluster1,1), "Overlap")
        self.assertFalse(overlap( [2, 2, 1, 1, 1, 1], cluster1,1), "Overlap")
        self.assertFalse(overlap( [1, 1, 2, 2, 1, 1], cluster1,1), "Overlap")
        self.assertTrue(overlap( [1, 1, 1, 1, 1, 1], cluster1,1), "Overlap")

        self.assertFalse(overlap(cluster1, [1, 2, 0, 1, 0, 0], 1), "Overlap")
        self.assertFalse(overlap(cluster1, [0, 0, 1, 2, 0, 1], 1), "Overlap")
        self.assertFalse(overlap(cluster1, [0, 1, 0, 0, 1, 2], 1), "Overlap")
        self.assertTrue(overlap(cluster1, [0, 1, 0, 2, 0, 1], 1), "Overlap")

        self.assertFalse(overlap([1, 2, 0, 1, 0, 0],cluster1,1), "Overlap")
        self.assertFalse(overlap([0, 0, 1, 2, 0, 1],cluster1,1), "Overlap")
        self.assertFalse(overlap([0, 1, 0, 0, 1, 2],cluster1,1), "Overlap")
        self.assertTrue(overlap([0, 1, 0, 2, 0, 1], cluster1,1), "Overlap")

    def test_insquare(self):
        cluster1=[3,5,3,5]
        for x in range(0,9):
            for y in range(0,9):
                in_sq_b=in_square(cluster1, x,y,1, 1)
                in_sq=in_square(cluster1, x,y,0, 0)
                if (x <3 or y<3 ) or(x>5 or y>5):
                    self.assertFalse( in_sq, "Insquare gave true when should have given false, no border:"+str(x)+ " "+str(y))
                    if (x <2 or y<2 ) or(x>6 or y>6):
                        self.assertFalse( in_sq_b, "Insquare gave true when should have given false, border:"+str(x)+ " "+str(y))
                    else:
                        self.assertTrue( in_sq_b, "Insquare gave false when should have given true, border:"+str(x)+ " "+str(y))
                else:
                    self.assertTrue( in_sq, "Insquare gave false when should have given true, no border:"+str(x)+ " "+str(y))
                    self.assertTrue( in_sq_b, "Insquare gave false when should have given true, border:"+str(x)+ " "+str(y))


    def test_overlap_sq(self):
        cluster1=[3,5,3,5]
        cluster2=[0,2,0,2]
        for x in range(0,9):
            cluster2[0]=x
            cluster2[1]=x+2
            for y in range(0,9):
                cluster2[2]=y
                cluster2[3]=y+2
                ol_b= overlap( cluster1, cluster2, 0.4)or overlap( cluster2, cluster1, 0.4)
                ol=overlap( cluster1, cluster2, 0)or overlap( cluster2, cluster1, 0)
                if (x >6 or y>6):
                    self.assertFalse( ol, "Overlap gave true when should have given false, no border:"+str(x)+ " "+str(y))
                    self.assertFalse( ol_b, "Overlap gave true when should have given false, border:"+str(x)+ " "+str(y))
                else:
                    #self.assertTrue( ol_b, "Overlap gave false when should have given true, border:"+str(x)+ " "+str(y)+" cluster 1 "+str(cluster1)+" cluster 2 "+str(cluster2))
                    if (x>5 or y>5) or (x<1 or y<1):
                        self.assertFalse( ol, "Overlap gave true when should have given false, no border:"+str(x)+ " "+str(y))
                    else:
                        self.assertTrue( ol, "Overlap gave false when should have given true, no border:"+str(x)+ " "+str(y))

    def test_overlap(self):
        cluster1=[6,7,5,9]
        cluster2=[0,3,0,2]
        for x in range(0,12):
            cluster2[2]=x
            cluster2[3]=x+2
            for y in range(0,12):
                cluster2[0]=y
                cluster2[1]=y+3
                ol_b= overlap( cluster1, cluster2, 0.3)or overlap( cluster2, cluster1, 0.3)
                ol=overlap( cluster1, cluster2, 0)or overlap( cluster2, cluster1, 0)
                if (x >11 or y>8) or (x<1 or y<2):
                    self.assertFalse( ol, "Overlap gave true when should have given false, no border:"+str(x)+ " "+str(y))
                    self.assertFalse( ol_b, "Overlap gave true when should have given false, border:"+str(x)+ " "+str(y))
                else:
                    #self.assertTrue( ol_b, "Overlap gave false when should have given true, border:"+str(x)+ " "+str(y)+" cluster 1 "+str(cluster1)+" cluster 2 "+str(cluster2))
                    if (x<3 or y<3) or (x>9 or y>7):
                        self.assertFalse( ol, "Overlap gave true when should have given false, no border:"+str(x)+ " "+str(y))
                    else:
                        self.assertTrue( ol, "Overlap gave false when should have given true, no border:"+str(x)+ " "+str(y))


    def test_framepoints_to_collection_emptyPoints3D(self):
        # Setup
        out_colors=np.array([[255, 254, 253],
                            [0,1,2]])
        points=np.array([[0,0,0],
                         [0,0,1]]).T
        out_seg=np.array([[33],[32]])

        from collections import defaultdict
        points3D = defaultdict(list)

        points3D=framepoints_to_collection(out_colors, out_seg, points, points3D)
        self.assertEqual(len(points3D), 2, "Wrong length of dictionary")
        self.assertEqual(len(points3D[(0,0,0)]), 1, "wrong length of list")
        self.assertEqual(len(points3D[(0,0,5)]), 1, "wrong length of list")
        np.testing.assert_array_equal(np.array([255,254,253,33]), points3D[(0,0,0)][0], err_msg="Value is not equal!")
        np.testing.assert_array_equal(np.array([0, 1, 2, 32]), points3D[(0,0,5)][0], err_msg="Value is not equal!")
        out_colors = np.array([[255, 254, 253],
                               [0, 1, 2],
                               [0, 1, 3],
                               [1,1,1]])
        points = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 0, 1],
                           [0,0,2]]).T
        out_seg = np.array([[33], [32],[32],[0]])

        points3D = framepoints_to_collection(out_colors, out_seg, points, points3D)
        self.assertEqual(len(points3D), 3, "Wrong length of dictionary")
        self.assertEqual(len(points3D[(0, 0, 0)]), 2, "wrong length of list")
        self.assertEqual(len(points3D[(0, 0, 5)]), 3, "wrong length of list")
        self.assertEqual(len(points3D[(0, 0, 10)]), 1, "wrong length of list")
        np.testing.assert_array_equal(np.array([255, 254, 253, 33]), points3D[(0, 0, 0)][0],
                                      err_msg="Value is not equal!")
        np.testing.assert_array_equal(np.array([255, 254, 253, 33]), points3D[(0, 0, 0)][1],
                                      err_msg="Value is not equal!")
        np.testing.assert_array_equal(np.array([0, 1, 2, 32]), points3D[(0, 0, 5)][0], err_msg="Value is not equal!")
        np.testing.assert_array_equal(np.array([0, 1, 2, 32]), points3D[(0, 0, 5)][1], err_msg="Value is not equal!")
        np.testing.assert_array_equal(np.array([0, 1, 3, 32]), points3D[(0, 0, 5)][2], err_msg="Value is not equal!")
        np.testing.assert_array_equal(np.array([1, 1, 1, 0]), points3D[(0, 0, 10)][0], err_msg="Value is not equal!")




    def test_refine_voxel_values_one_value(self):
        # Setup
        pos=(0,0,0)
        value=np.array([255,254,253,4], dtype=np.int32)
        from collections import defaultdict
        points3D = defaultdict(list)
        points3D[pos].append(value)
        print("Refine 1")
        # Test
        tensor=refine_voxel_values(points3D)

        self.assertEqual(len(tensor), 1, "Wrong length of dictionary")
        self.assertEqual(len(points3D[pos]), 1, "wrong length of list")
        assert isinstance(points3D[pos][0], np.ndarray), 'Argument of wrong type!'
        assert isinstance(tensor[pos], list)
        np.testing.assert_array_equal(value, points3D[pos][0], err_msg="Value is not equal!")
        np.testing.assert_array_equal(value, tensor[pos], err_msg="Value is not equal!")

    def test_refine_voxel_values_two_values(self):
        # Setup
        pos1 = (0, 0, 0)
        value1 = np.array([255, 254, 253, 4], dtype=np.int32)
        pos2=(0,0,1)
        value2=np.array([0, 1, 2, 3], dtype=np.int32)
        from collections import defaultdict
        points3D = defaultdict(list)
        points3D[pos1].append(value1)
        points3D[pos2].append(value2)
        # Test
        print("Refine 2")
        tensor = refine_voxel_values(points3D)

        self.assertEqual(len(tensor), 2, "Wrong length of dictionary")
        self.assertEqual(len(points3D[pos1]), 1, "wrong length of list")
        self.assertEqual(len(points3D[pos2]), 1, "wrong length of list")

        assert isinstance(points3D[pos1][0], np.ndarray), 'Argument of wrong type!'
        np.testing.assert_array_equal(value1, points3D[pos1][0], err_msg="Value is not equal!")
        assert isinstance(tensor[pos1], list)
        np.testing.assert_array_equal(value1, tensor[pos1], err_msg="Value is not equal!")

        assert isinstance(points3D[pos2][0], np.ndarray), 'Argument of wrong type!'
        np.testing.assert_array_equal(value2, points3D[pos2][0], err_msg="Value is not equal!")
        assert isinstance(tensor[pos2], list)
        np.testing.assert_array_equal(value2, tensor[pos2], err_msg="Value is not equal!")

    def test_refine_voxel_values_equal_values(self):
        # Setup
        pos1 = (0, 0, 0)
        values1 = [np.array([255, 254, 253, 4], dtype=np.int32), np.array([255, 254, 253, 4], dtype=np.int32)]
        pos2=(0,0,1)
        value2=np.array([0, 1, 2, 3], dtype=np.int32)
        from collections import defaultdict
        points3D = defaultdict(list)
        points3D[pos1].append(values1[0])
        points3D[pos1].append(values1[1])
        points3D[pos2].append(value2)

        # Test
        print("Refine equal")
        tensor = refine_voxel_values(points3D)

        self.assertEqual(len(tensor), 2, "Wrong length of dictionary")
        self.assertEqual(len(points3D[pos1]), 2, "wrong length of list")
        self.assertEqual(len(points3D[pos2]), 1, "wrong length of list")
        assert isinstance(points3D[pos1][0], np.ndarray), 'Argument of wrong type!'
        np.testing.assert_array_equal(values1[0], points3D[pos1][0], err_msg="Value is not equal!")
        assert isinstance(tensor[pos1], list)
        np.testing.assert_array_equal(values1[0], tensor[pos1], err_msg="Value is not equal!")

        assert isinstance(points3D[pos2][0], np.ndarray), 'Argument of wrong type!'
        np.testing.assert_array_equal(value2, points3D[pos2][0], err_msg="Value is not equal!")
        assert isinstance(tensor[pos2], list)
        np.testing.assert_array_equal(value2, tensor[pos2], err_msg="Value is not equal!")

    def test_refine_voxel_values_no_mode(self):
        # Setup
        pos1 = (0, 0, 0)
        values1 = [np.array([0, 2, 2, 4], dtype=np.int32), np.array([1, 2, 4, 3], dtype=np.int32)]
        pos2 = (0, 0, 1)
        value2 = np.array([0, 1, 2, 3], dtype=np.int32)
        from collections import defaultdict
        points3D = defaultdict(list)
        points3D[pos1].append(values1[0])
        points3D[pos1].append(values1[1])
        points3D[pos2].append(value2)

        # Test
        print("Refine No mode")
        tensor = refine_voxel_values(points3D)

        self.assertEqual(len(tensor), 2, "Wrong length of dictionary")
        self.assertEqual(len(points3D[pos1]), 2, "wrong length of list")
        self.assertEqual(len(points3D[pos2]), 1, "wrong length of list")
        assert isinstance(points3D[pos1][0], np.ndarray), 'Argument of wrong type!'
        np.testing.assert_array_equal(values1[0], points3D[pos1][0], err_msg="Value is not equal!")
        assert isinstance(tensor[pos1], list)
        received= np.array(tensor[pos1], dtype=float)
        expected= np.array([0.5,2,3,3], dtype=float)
        np.testing.assert_array_almost_equal(received, expected, err_msg="Value is not equal!")

        assert isinstance(points3D[pos2][0], np.ndarray), 'Argument of wrong type!'
        np.testing.assert_array_equal(value2, points3D[pos2][0], err_msg="Value is not equal!")
        assert isinstance(tensor[pos2], list)
        np.testing.assert_array_equal(value2, tensor[pos2], err_msg="Value is not equal!")

    def test_refine_voxel_values_one_mode(self):
        # Setup
        pos1 = (0, 0, 0)
        values1 = [np.array([0, 2, 2, 4], dtype=np.int32),np.array([0, 2, 2, 4], dtype=np.int32),np.array([0, 2, 2, 3], dtype=np.int32)]
        pos2 = (0, 0, 1)
        value2 = np.array([0, 1, 2, 3], dtype=np.int32)
        from collections import defaultdict
        points3D = defaultdict(list)
        points3D[pos1].append(values1[0])
        points3D[pos1].append(values1[1])
        points3D[pos1].append(values1[2])
        points3D[pos2].append(value2)

        # Test
        print("Refine One mode")
        tensor = refine_voxel_values(points3D)

        self.assertEqual(len(tensor), 2, "Wrong length of dictionary")
        self.assertEqual(len(points3D[pos1]), 3, "wrong length of list")
        self.assertEqual(len(points3D[pos2]), 1, "wrong length of list")
        assert isinstance(points3D[pos1][0], np.ndarray), 'Argument of wrong type!'
        np.testing.assert_array_equal(values1[0], points3D[pos1][0], err_msg="Value is not equal!")
        assert isinstance(tensor[pos1], list)
        received = np.array(tensor[pos1], dtype=float)
        expected = np.array([0, 2, 2, 4], dtype=float)
        np.testing.assert_array_almost_equal(received, expected, err_msg="Value is not equal!")

        assert isinstance(points3D[pos2][0], np.ndarray), 'Argument of wrong type!'
        np.testing.assert_array_equal(value2, points3D[pos2][0], err_msg="Value is not equal!")
        assert isinstance(tensor[pos2], list)
        np.testing.assert_array_equal(value2, tensor[pos2], err_msg="Value is not equal!")

    def test_refine_voxel_values_two_mode(self):
        # Setup
        pos1 = (0, 0, 0)
        values1 = [np.array([0, 2, 2, 4], dtype=np.int32),np.array([0, 2, 2, 4], dtype=np.int32),np.array([0, 2, 2, 3], dtype=np.int32),np.array([0, 2, 2, 3], dtype=np.int32)]
        pos2 = (0, 0, 1)
        value2 = np.array([0, 1, 2, 3], dtype=np.int32)
        from collections import defaultdict
        points3D = defaultdict(list)
        points3D[pos1].append(values1[0])
        points3D[pos1].append(values1[1])
        points3D[pos1].append(values1[2])
        points3D[pos1].append(values1[3])
        points3D[pos2].append(value2)

        # Test
        print("Refine two modes ")
        tensor = refine_voxel_values(points3D)

        self.assertEqual(len(tensor), 2, "Wrong length of dictionary")
        self.assertEqual(len(points3D[pos1]), 4, "wrong length of list")
        self.assertEqual(len(points3D[pos2]), 1, "wrong length of list")
        assert isinstance(points3D[pos1][0], np.ndarray), 'Argument of wrong type!'
        np.testing.assert_array_equal(values1[0], points3D[pos1][0], err_msg="Value is not equal!")
        assert isinstance(tensor[pos1], list)
        received = np.array(tensor[pos1], dtype=float)
        expected = np.array([0, 2, 2, 3], dtype=float)
        np.testing.assert_array_almost_equal(received, expected, err_msg="Value is not equal!")

        assert isinstance(points3D[pos2][0], np.ndarray), 'Argument of wrong type!'
        np.testing.assert_array_equal(value2, points3D[pos2][0], err_msg="Value is not equal!")
        assert isinstance(tensor[pos2], list)
        np.testing.assert_array_equal(value2, tensor[pos2], err_msg="Value is not equal!")


    def test_framepoints_to_collection_and_refine_voxel_values(self):
        # Setup
        out_colors=np.array([[10, 20, 30],
                            [0,0,0]])
        points=np.array([[0,0,0],
                         [0,0,1]]).T
        out_seg=np.array([[33],[31]])

        from collections import defaultdict
        points3D = defaultdict(list)

        points3D=framepoints_to_collection(out_colors, out_seg, points, points3D)

        out_colors = np.array([[12, 22, 32],
                               [0, 3, 3],
                               [0, 3, 3],
                               [1,1,1],
                               [1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]])
        points = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 0, 1],
                           [0,0,2],
                           [0, 0, 2],
                           [0, 0, 2],
                           [0, 0, 2]]).T
        out_seg = np.array([[33], [32],[32],[0], [0], [2], [2]])

        points3D = framepoints_to_collection(out_colors, out_seg, points, points3D)

        tensor = refine_voxel_values(points3D)

        self.assertEqual(len(tensor), 3, "Wrong length of dictionary")
        self.assertEqual(len(points3D[(0,0,0)]), 2, "wrong length of list")
        self.assertEqual(len(points3D[(0,0,5)]), 3, "wrong length of list")
        self.assertEqual(len(points3D[(0, 0,10)]), 4, "wrong length of list")

        assert isinstance(tensor[(0,0,0)], list)
        received = np.array(tensor[(0,0,0)], dtype=float)
        expected = np.array([11, 21, 31, 33], dtype=float)
        np.testing.assert_array_almost_equal(received, expected, err_msg="Value is not equal!")

        assert isinstance(tensor[(0, 0, 5)], list)
        received = np.array(tensor[(0, 0, 5)], dtype=float)
        expected = np.array([0, 2, 2, 32], dtype=float)
        np.testing.assert_array_almost_equal(received, expected, err_msg="Value is not equal!")

        assert isinstance(tensor[(0, 0, 10)], list)
        received = np.array(tensor[(0, 0, 10)], dtype=float)
        expected = np.array([1, 1, 1, 0], dtype=float)
        np.testing.assert_array_almost_equal(received, expected, err_msg="Value is not equal!")


if __name__ == '__main__':
    unittest.main()

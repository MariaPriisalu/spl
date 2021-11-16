import unittest
import numpy as np
from colmap.reconstruct import get_colmap_camera_rotation_matrix

class TestEpisode(unittest.TestCase):

    # Test correct empty initialization.
    def test_get_colmap_camera_rotation_matrix_correct_len(self):
        path_seq="colmap"
        camera_rotation, translation =get_colmap_camera_rotation_matrix(path_seq)
        self.assertEqual(len(camera_rotation), 59, "Incorrect length of rotation matrix dictionary, should have been 59 was: "+str(len(camera_rotation)))
        self.assertEqual(len(translation), 59, "Incorrect length of translation vector dictionary, should have been 59 was: "+str(len(translation)))

        # Test correct empty initialization.

    def test_get_colmap_camera_rotation_matrix_correct_keys(self):
        path_seq = "colmap"
        camera_rotation, translation = get_colmap_camera_rotation_matrix(path_seq)
        for key in camera_rotation:
            parts=key.split("_")
            self.assertEqual(parts[0], "weimar", "Not correct key in camera roatation dict. does not contain city name: "+str(key))
            self.assertEqual(parts[1], "000141", "Not correct seq nbr in key key in camera roatation dict, should have been 000141: " + str(key))
            self.assertEqual(float(parts[2]), int(float(parts[2])), "Not correct frame_nbr: " + str(key))





if __name__ == '__main__':
    unittest.main()
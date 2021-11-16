import numpy as np
import os
from PIL import Image as Image

import cv2


def calcDepth(disp_path, baseline, inD):
    disparity = np.array(Image.open(disp_path))
    disparity[disparity != 0] = (disparity[disparity != 0].astype(np.float) - 1.0) / 256
    depth = np.divide(baseline * inD["fx"], disparity)
    depth[disparity == 0] = 0
    return depth, disparity


def find_disparity(city, constants, frame_nbr, left_img, right_img_path, seq_nbr):
    disparity = 0
    if constants.disparity_read:
        # Read from file.
        disp_file = city + "_" + seq_nbr + "_" + "0000%02d_disparity.png" % frame_nbr
        disp_path = os.path.join(
            "Datasets/cityscapes/disparity_sequence_trainvaltest/disparity_sequence/",
            constants.mode_name, city, disp_file)
        disparity = np.array(Image.open(disp_path))
        disparity[disparity != 0] = (disparity[disparity != 0].astype(np.float) - 1.0) / 256
    else:
        # Calculate disparity.
        right_img = cv2.imread(right_img_path)
        stereo = cv2.StereoBM_create(numDisparities=128, blockSize=9)
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(left_img, right_img)  # Left disparity

    return disp_path, disparity
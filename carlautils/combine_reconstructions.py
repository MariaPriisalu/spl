from multiprocessing import Pool
import os
import glob
from commonUtils.ReconstructionUtils import reconstruct3D_ply


if __name__ == '__main__':
    filespath = "Packages/CARLA_0.8.2/PythonClient/new_data-viz/"
    if not os.path.exists(filespath):
        filespath = "Datasets/carla-viz/"
    ending = "test_*"



    files=[]
    for filepath in glob.glob(filespath + ending):
        if os.path.isfile(os.path.join(filespath, filepath, 'cameras.p')) :
            files.append(os.path.join(filespath, filepath))


    p=Pool(10)
    p.map(reconstruct3D_ply, files)
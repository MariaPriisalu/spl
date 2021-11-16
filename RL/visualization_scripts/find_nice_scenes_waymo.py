import numpy as np
import tensorflow as tf
from RL.settings import RANDOM_SEED
tf.set_random_seed(RANDOM_SEED)
import os.path
external=True

import matplotlib.pyplot as plt

from RL.settings import run_settings
import os
import glob, subprocess
import traceback
import pickle, sys
import scipy.io

from commonUtils.ReconstructionUtils import reconstruct3D_ply,  CreateDefaultDatasetOptions_Waymo


from RL.episode import SimpleEpisode


import matplotlib.patches as patches

from utils.constants import Constants
from RL.settings import run_settings
from RL.environment_waymo import WaymoEnvironment
from RL.visualization import view_2D, view_pedestrians, view_cars, view_valid_pos,view_2D_rgb,view_pedestrians_nicer,view_cars_nicer,view_agent_nicer, view_valid_init_pos, view_valid_pedestrians, view_agent, view_agent_input
from RL.extract_tensor import  extract_tensor

from utils.utils_functions import in_square
import glob
import matplotlib.image as mpimg
from collections import Counter

from RL.net_sem_2d import Seg_2d

from RL.agent_net import NetAgent

######################################################################## Change values here!!!

# Save plots?
save_plots=True
save_regular_plots=True
toy_case=False
make_movie=False

car_people=False




timestamps=["2020-07-07-21-56-19.510945"]
ending=''

calc_dist=True
folder_name="/test_visualize_waymo/"
################################################################################3
settings_path="/some_results/"
settings_ending="settings.txt"

save_images_path="/visualization-results"


def trendline(data):
    trend_avg = np.zeros(len(data))
    trend_avg[0] = data[0]
    for i in range(1, len(data)):
        trend_avg[i] = (trend_avg[i - 1] * (i - 1.0) + data[i]) / i
    return trend_avg






actions_names = ['upL', 'up', 'upR', 'left', 'stand', 'right', 'downL', 'down', 'downR']
labels = ['human hist','car hist' ,'unlabeled' , 'ego vehicle', 'rectification border' ,'out of roi' , 'static' ,
          'dynamic','ground', 'road' , 'sidewalk', 'parking', 'rail track' , 'building' ,  'wall', 'fence' ,
          'guard rail', 'bridge' , 'tunnel' , 'pole'   , 'polegroup', 'traffic light' , 'traffic sign' ,
          'vegetation','terrain' , 'sky','person','rider', 'car'  ,'truck', 'bus',  'caravan', 'trailer',
          'train', 'motorcycle' ,'bicycle', 'license plate' ]

 # label_mapping[0] = 0 # None
 #    label_mapping[1] = 11 # Building
 #    label_mapping[2] = 13 # Fence
 #    label_mapping[3] = 4 # Other/Static
 #    label_mapping[4] = 24 # Pedestrian
 #    label_mapping[5] = 17 # Pole
 #    label_mapping[6] = 7 # RoadLines
 #    label_mapping[7] = 7 # Road
 #    label_mapping[8] = 8 # Sidewlk
 #    label_mapping[9] = 21 # Vegetation
 #    label_mapping[10] = 26 # Vehicles
 #    label_mapping[11] = 12 # Wall
 #    label_mapping[12] = 20 # Traffic sign

labels_indx = [11,13,4,17,7,8,21,12,20]
labels_mini=[]
labels_mini.append("R")
labels_mini.append("G")
labels_mini.append("B")
labels_mini.append('people-t')
labels_mini.append('cars-t')
if car_people:
    labels_mini.append('')
    labels_mini.append('cars and people')
else:
    labels_mini.append('people')
    labels_mini.append('cars')

for label in labels_indx:
    #print labels[label+2]
    labels_mini.append(labels[label+2])

epoch=3000# update_frequency*100


def get_cars_stats(filenames_itr_cars, sz_train):

    test_itr_car = []
    statistics_car = np.zeros((len(filenames_itr_cars) * sz_train[0], sz_train[1], sz_train[2]))
    statistics_test_car=[]
    indx_test = 0
    indx_same = 0
    prev_itr = 0
    test_points_car = []
    if len(test_files_cars) > 0:
        sz_test_car = np.load(test_files_cars[0][1]).shape
        statistics_test_car = np.zeros((len(test_files_cars) * sz_test_car[0], sz_test_car[1], sz_test_car[2]))
        for j, pair in enumerate(sorted(test_files_cars)):
            cur_statistics = np.load(pair[1])
            statistics_test_car[indx_test * sz_test_car[0]:(indx_test + 1) * sz_test_car[0], :, :] = cur_statistics
            test_itr_car.append(pair[0])
            if indx_same == pair[0]:
                prev_itr += 1
            else:
                prev_itr = 0
                test_points_car.append(j)
            indx_same = pair[0]

            test_itr_car[indx_test * sz_test_car[0]:(indx_test + 1) * sz_test_car[0]] = list(range(
                pair[0] + prev_itr * sz_test_car[0], pair[0] + (prev_itr + 1) * sz_test_car[0]))
            indx_test += 1
    indx_train_car = 0
    for i, pair in enumerate(sorted(filenames_itr_cars)):
        try:

            cur_statistics = np.load(pair[1])

            statistics_car[indx_train_car * sz_train[0]:(indx_train_car + 1) * sz_train[0], :, :] = cur_statistics[:,
                                                                                                    0: sz_train[1], :]
            indx_train_car += 1

        except IOError:
            print(("Could not load " + pair[1]))


    return indx_train_car, statistics_car, statistics_test_car, test_itr_car


def sort_files(files):

    filenames_itr_cars = []
    test_files_cars = []
    filenames_itr_people = []
    test_files_people = []
    filenames_itr = []
    test_files = []
    nbr_files = 0
    reconstructions = []
    reconstructions_test = []
    numbers = []
    iterations = {}
    numbers_cars = []
    special_cases=[]
    numbers_people = []
    iterations_cars=[]
    for agent_file in sorted(files):

        basename = os.path.basename(agent_file)

        nbrs = basename.strip()[:-len('.npy')]
        vals = nbrs.split('_')

        if vals[1] == 'weimar' or 'test' in vals[-4]:  # Testing data

            if "people" in vals[-4]:
                if not "reconstruction" in vals[-1]:
                    test_files_people.append((int(vals[-1]), agent_file))
                    test_files.append((int(vals[-1]), agent_file))
                    numbers_people.append(int(vals[-1]))

            elif "car" in vals[-4]:
                if not "reconstruction" in vals[-1]:
                    test_files_cars.append((int(vals[-1]), agent_file))
                    test_files.append((int(vals[-1]), agent_file))
                    numbers_people.append(int(vals[-1]))

            elif not "reconstruction" in vals[-1]:

                try:
                    test_files.append((int(vals[-1]), agent_file, int(vals[2])))
                    # numbers.append(int(vals[-1]))
                except ValueError:
                    test_files.append((int(vals[-1]), agent_file, int(vals[4])))
                    # numbers.append(int(vals[-1]))

            else:
                try:
                    reconstructions_test.append((int(vals[-1][:-len("reconstruction")]), agent_file, int(vals[2])))
                except ValueError:
                    reconstructions_test.append(
                        (int(vals[-1][:-len("reconstruction")]), agent_file, int(vals[4])))
            if  not "reconstruction" in vals[-1]:
                if int(vals[-1]) not in list(iterations.keys()):
                    iterations[int(vals[-1])]= False
                elif iterations[int(vals[-1])]:
                    special_cases.append(int(vals[-1]))
        else: # Training data
            if "people" in vals[-4]:

                if not "reconstruction" in vals[-1]:
                    filenames_itr_people.append((int(vals[-1]), agent_file))
                    filenames_itr.append((int(vals[-1]), agent_file))
                    numbers_people.append(int(vals[-1]))

            elif "car" in vals[-4]:

                if not "reconstruction" in vals[-1]:
                    filenames_itr_cars.append((int(vals[-1]), agent_file))
                    filenames_itr.append((int(vals[-1]), agent_file))
                    numbers_people.append(int(vals[-1]))

            elif not "reconstruction" in vals[-1]:

                try:
                    filenames_itr.append((int(vals[-1]), agent_file, int(vals[2])))
                    numbers.append(int(vals[-1]))
                except ValueError:
                    # numbers.append(int(vals[-1]))
                    filenames_itr.append((int(vals[-1]), agent_file, int(vals[3])))
            else:
                try:
                    reconstructions.append((int(vals[-1][:-len("reconstruction")]), agent_file, int(vals[2])))
                except ValueError:
                    reconstructions.append((int(vals[-1][:-len("reconstruction")]), agent_file, int(vals[3])))

            if  not "reconstruction" in vals[-1]:
                if "people" in vals[-4] or "car" in vals[-4]:

                    iterations_cars.append(int(vals[-1]))
                if int(vals[-1]) not in list(iterations.keys()):
                    iterations[int(vals[-1])] = True
                elif not iterations[int(vals[-1])] :

                    special_cases.append(int(vals[-1]))

            #special_cases.append(int(vals[-1]))

        nbr_files += 1
    return filenames_itr_cars, test_files_cars, filenames_itr_people,test_files_people,  filenames_itr, test_files, reconstructions_test, iterations, iterations_cars, special_cases

def sort_files_eval(files):

    test_files = []
    test_files_cars=[]
    test_files_people=[]
    reconstructions_test=[]
    nbr_files = 0

    special_cases=[]
    numbers_people = []

    for agent_file in sorted(files):

        basename = os.path.basename(agent_file)

        nbrs = basename.strip()[:-len('.npy')]

        vals = nbrs.split('_')

        try:
            pos=int(vals[-1])
        except ValueError:
            try:
                pos = int(vals[-1][:-len("reconstruction")])
            except ValueError:
                print(basename)
                pos=-1

        if pos>=0:

            if not "reconstruction" in vals[-1]:
                test_files.append((int(vals[-3]), agent_file, int(vals[-1])))
            else:

                reconstructions_test.append((int(vals[-3]), agent_file, int(vals[-1][:-len("reconstruction")])))

            nbr_files += 1
    return test_files, reconstructions_test


def read_settings_file():

    # Find settings file
    settings_file = glob.glob(settings_path + "*/" + timestamp + "*" + settings_ending)

    if len(settings_file) == 0:
        settings_file = glob.glob(settings_path + timestamp + "*" + settings_ending)
    if len(settings_file)==0:
        return [],[],[],[],[],[]
    labels_to_use = labels
    # Get movie name
    name_movie = os.path.basename(settings_file[0])[len(timestamp) + 1:-len("_settings.txt")]
    target_dir = settings_path + name_movie
    subprocess.call("cp " + settings_file[0] + " " + target_dir + "/", shell=True)
    semantic_channels_separated = False
    in_2D=False
    num_measures = 6
    with open(settings_file[0]) as s:
        for line in s:
            if "Semantic channels separated" in line:
                if line[len("Semantic channels separated: "):].strip()=="True":
                    semantic_channels_separated = True

            if "Minimal semantic channels : " in line:
                if line[len("Minimal semantic channels : "):].strip() == "True":
                    mini_labels = True
                    if mini_labels:
                        labels_to_use = labels_mini
            if "Number of measures" in line:
                num_measures = int(line[len("Number of measures: "):])
            if "2D input to network:"in line:
                if line[len("2D input to network: "):].strip()== "True":
                    in_2D=True
                print((line[len("2D input to network: "):].strip() +" "+str(bool(line[len("2D input to network: "):].strip()))+" "+str(in_2D)))

    return labels_to_use, name_movie, target_dir, num_measures, semantic_channels_separated, in_2D



init_names= {-1: "training", 0: "training", 1: "On pedestrian", 3: "On pavement", 2: "By car", 4: "Random",5: "car_env", 6: "Near pedestrian",7:"pedestrian environment", 9:"average"}




def get_scenes(test_files,test_points):

    scenes=[]

    stats_temp = []
    prev_test_pos = 0

    for j, pair in enumerate(sorted(test_files)):

        if prev_test_pos != test_points[pair[0]]:
            #print 'switch'
            for init_m, stats in enumerate(stats_temp):
                if len(stats)>0:
                    scenes=stats_temp
            stats_temp = []
        cur_stat = np.load(pair[1])

        for ep_nbr in range(cur_stat.shape[0]):
            #if int(cur_stat[ep_nbr, 0, 38 + 11])!=2:
            # agent_pos = cur_stat[ep_nbr, :, 0:3]
            # agent_probabilities = cur_stat[ep_nbr, :, 7:34]
            # agent_reward = cur_stat[ep_nbr, :, 34]
            # agent_measures = cur_stat[ep_nbr, :, 38:]
            # # car hit
            # out_of_axis = cur_stat[ep_nbr, -1, 0]
            # if out_of_axis==0:
            stats_temp.append([cur_stat[ep_nbr, :, :], pair[1], ep_nbr])
        prev_test_pos = test_points[pair[0]]

    scenes = stats_temp
    return stats_temp

################################################################### Main starts here!!!!!
make_vid=False
test_not_in_name=False
train_nbrs = [6, 7, 8, 9]  # , 0,2,4,5]
test_nbrs = [3, 0, 2, 4, 5]

init_names= {-1: "training", 0: "training", 1: "On pedestrian", 3: "On pavement", 2: "By car", 4: "Random",5: "car_env", 6: "Near pedestrian",7:"pedestrian environment", 9:"average", 10: "on ped traj"}



eval_path = "Results/statistics/val/"

for timestamp in timestamps:

    plt.close("all")

    # # find statistics files.
    # path=stat_path+timestamp+ending+'*'
    # match=path+"*.npy"
    #
    # files=glob.glob(match)
    print("Evaluation ------------------------------")
    path = eval_path + '*' + timestamp
    print(path)
    match = path + "*.npy"

    print(match)
    files = glob.glob(match)
    print(files)
    #test_files, reconstructions_test = sort_files_eval(files_eval)



    labels_to_use, name_movie, target_dir, num_measures, semantic_channels_separated, in_2D=read_settings_file()

        # Make movie

    #name_movie="random_agent"
    if len(files)>1:
        #filenames_itr_cars, test_files_cars, filenames_itr_people,test_files_people,  filenames_itr, test_files, reconstructions_test, iterations, iterations_cars, special_cases
        #filenames_itr_cars, test_files_cars, filenames_itr_people, test_files_people, filenames_itr, test_files, reconstructions_test, iterations, iterations_cars,special_cases=sort_files(files)

        test_files, reconstructions_test = sort_files_eval(files)
        #
        # train_itr={}
        # train_counter=0
        # test_points={}
        #
        # for itr in sorted(iterations.keys()):
        #
        #     training=iterations[itr]
        #     if training:
        #         train_itr[itr]=train_counter
        #         if itr in special_cases:
        #             test_points[itr] = train_counter
        #         if itr not in iterations_cars:
        #             train_counter=train_counter+1
        #     else:
        #         test_points[itr]= train_counter
        #         if itr in special_cases:
        #             train_itr[itr] = train_counter

        # print "----------------------------------------------------------------------------"
        #print train_itr
        # print "----------------------------------------------------------------------------"
        #print sorted(test_points)
        settings = run_settings()
        test_points = {0: 0}
        scenes=["15646511153936256674_1620_000_1640_000", "18311996733670569136_5880_000_5900_000"]
        filespath=settings.waymo_path

        epoch = 0
        filename_list = []
        # Get files to run on.
        ending_local = "test_*"
        for scene in scenes:
            filename_list.append(os.path.join(filespath, scene))
        saved_files_counter=0
        print(filename_list)

        file_nbrs =[0,1]

        ending = "/*"
        filespath = settings.waymo_path  # "Packages/CARLA_0.8.2/unrealistic_dataset/"

        epoch = 0
        pos = 0
        filename_list = []
        # Get files to run on.

        for filepath in glob.glob(filespath + ending):
            print(filepath)

            filename_list.append(filepath)

        file_nbrs = list(range(80,100))

        constants = Constants()
        settings.seq_len_train = 1

        env = WaymoEnvironment("", None, None, None, None,  settings)
        csv_rows = []
        print(("Settings: "+str(env.settings)+" "+str(settings)))


        if len(test_files)>0:
            scenes =get_scenes(test_files,test_points)
            prev_nbr = -1
            for scene in scenes:
                filepath_npy=scene[1]
                epoch=scene[2]
                print((filepath_npy.split('_')))

                nbr=filepath_npy.split('_')[-1]
                nbr=int(nbr[:-len('.npy')])#*4#file_nbrs[int(nbr[:-len('.npy')])]
                print(nbr)

                if nbr!=prev_nbr:
                    filepath=filename_list[int(nbr)]#*4
                    basename=os.path.basename(filepath)
                    if True:#"test_4" in basename:
                        metadata = None
                        with open(os.path.join(filepath, "centering.p")) as metadataFileHandle:
                            metadata = pickle.load(metadataFileHandle, encoding="latin1", fix_imports=True)

                        # Create the desired dataset options structure
                        options = CreateDefaultDatasetOptions_Waymo(metadata)

                        print(options)
                        # Set the parameters of the env
                        min_bbox = metadata["min_bbox"]
                        max_bbox = metadata["max_bbox"]
                        settings.depth = max_bbox[0] - min_bbox[0]
                        settings.height = max_bbox[2] - min_bbox[2]
                        settings.width = max_bbox[1] - min_bbox[1]

                        env.reconstruction, people_rec, cars_rec, scale, ped_dict, cars_2D, people_2D, valid_ids, cars_dict, init_frames, init_frames_cars = reconstruct3D_ply(
                            filepath,settings.scale_x, datasetOptions=options, recalculate=False)

                        # # final_reconstruction, people, cars, local_scale_x, ped_dict, cars_2D, people_2D, valid_ids, car_dict, init_frames
                        # env.reconstruction, people_rec, cars_rec, scale, ped_dict, cars_2D, people_2D, valid_ids, cars_dict, init_frames,init_frames_cars = reconstruct3D_ply(
                        #     filepath, settings.scale_x)  # , recalculate=True)

                        pos_x = 0
                        pos_y = -settings.width / 2

                        tensor, density = extract_tensor(pos_x, pos_y, env.reconstruction, settings.height,
                                                         settings.width, settings.depth)
                        filter = True
                        if filter:

                            for channel in range(4):
                                tensor[:, :, :, channel] = scipy.ndimage.filters.median_filter(tensor[:, :, :, channel],
                                                                                               size=(1, 4,
                                                                                                     4))  # (1,3,3), (2,3,3) or (1,4,4) which is more realistic?
                        frameRate, frameTime = settings.getFrameRateAndTime()
                        seq_len_pfnn = -1
                        if settings.pfnn:
                            seq_len_pfnn = seq_len * 60 / frameRate
                        # ep = env.set_up_episode(cars_rec, people_rec, pos_x, pos_y, tensor, True, people_dict=ped_dict,
                        #                         car_dict=cars_dict,
                        #                         init_frames=init_frames, init_frames_cars=init_frames_cars)
                        ep = env.set_up_episode(cars_rec, people_rec, pos_x, pos_y, tensor, True, people_dict=ped_dict, car_dict=cars_dict,
                                                      init_frames=init_frames, init_frames_cars=init_frames_cars,
                                                      seq_len_pfnn=seq_len_pfnn)

                        img_from_above = np.ones((ep.reconstruction.shape[1], ep.reconstruction.shape[2], 3),
                                                 dtype=np.uint8) * 255
                        img = view_2D(ep.reconstruction, img_from_above, 0)
                        print("View PEDESTRIANS")
                        img = view_pedestrians(0, ep.people, img, 0, trans=.15, )  # tensor=ep.reconstruction)

                        img = view_cars(img, ep.cars, img.shape[0], img.shape[1], 0, frame=0)  # ,tensor=ep.reconstruction)
                        img = view_cars(img, ep.cars, img.shape[0], img.shape[1], 0, frame=15)
                        # img = view_cars(img, ep2.cars, img.shape[0], img.shape[1], len(cars)/2, frame=0)  # ,tensor=ep.reconstruction)
                        img2 = view_valid_pos(ep, img, 0)

                        img3 = view_2D_rgb(ep.reconstruction, np.ones((ep.reconstruction.shape[1], ep.reconstruction.shape[2], 3),
                                                 dtype=np.uint8) * 255, 0,white_background=True)
                        print("View PEDESTRIANS")
                        img3 = view_pedestrians_nicer(0, ep.people, img3, trans=.3 )  # tensor=ep.reconstruction)

                        img3 = view_cars_nicer(img3,ep.cars, 0)  # ,tensor=ep.reconstruction)


                if True:#"test_4" in basename and epoch==9:
                    cur_stat=scene[0]
                    agent_pos = cur_stat[ :, 1:3]
                    agent_probabilities = cur_stat[ :, 7:34]
                    agent_reward = cur_stat[ :, 34]
                    agent_measures = cur_stat[ :, 38:]
                    agent_goals = cur_stat[ 3:5, -1]

                    seq_len=len(agent_pos)-1
                    print(("Goal "+str(cur_stat[ 3:5, -1])))
                    print(("Agent pos init"+str(cur_stat[ 0, :3])))

                    print(("View agent: "+str(seq_len)))
                    frame=0
                    while frame+4<seq_len:
                        print(("Frame "+str(frame)+" pos "+str(agent_pos[frame])+" Agent actions " + str(cur_stat[frame:frame+4, 6])+"Agent speed " + str(cur_stat[frame:frame+4, 37])))
                        frame=frame+4
                    # car hit
                    out_of_axis = cur_stat[ -1, 0]
                    img_from_above_loc = view_agent(agent_pos, img_from_above, 0, settings.agent_shape, settings.agent_shape_s,
                                      people=ep.people, cars=ep.cars,
                                      frame=0, goal=agent_goals)

                    img_from_above_loc_rgb= view_agent_nicer(agent_pos, img3, settings.agent_shape, settings.agent_shape_s,0)

                    for frame_nbr in range(agent_pos.shape[0]):
                        img_from_above_loc=view_agent(agent_pos, img_from_above_loc, 0, settings.agent_shape, [1,2,2],
                                   frame=frame_nbr, goal=agent_goals)

                    fig = plt.figure()
                    plt.imshow(img_from_above_loc)
                    plt.show()
                    plt.axis('off')
                    print((save_images_path+folder_name + basename+"_"+str(timestamp) + "_An" + str(epoch)+".png"))
                    fig.savefig(save_images_path+folder_name + basename+"_" +str(timestamp)+ "_new_long" + str(epoch)+".png", bbox_inches='tight')
                    plt.imshow(img_from_above_loc_rgb)
                    plt.show()
                    plt.axis('off')
                    print((save_images_path+folder_name + basename +"_"+str(timestamp)+"_rgb_new_long" + str(epoch) +".png"))
                    fig.savefig(save_images_path+folder_name + basename +"_"+str(timestamp)+ "_rgb_new" + str(epoch)+".png", bbox_inches='tight')

                    prev_nbr=nbr
                    print(prev_nbr)
                    #input = view_agent_input(img2, agent_pos[ frame_nbr, :], settings.agent_shape, settings.agent_shape_s, ep)


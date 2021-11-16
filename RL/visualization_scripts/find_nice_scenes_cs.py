import numpy as np
import os.path
external=True

import matplotlib.pyplot as plt
from RL.environment_test import TestEnvironment
from RL.settings import run_settings
import os
import glob, subprocess
import traceback
import pickle, sys
import scipy.io




from RL.episode import SimpleEpisode


import matplotlib.patches as patches

from utils.constants import Constants
from RL.settings import run_settings
from RL.environment import Environment
from RL.visualization import view_2D, view_pedestrians, view_cars, view_valid_pos,view_2D_rgb,view_agent_nicer, view_cars_nicer,view_pedestrians_nicer, view_valid_init_pos, view_valid_pedestrians, view_agent, view_agent_input
from RL.extract_tensor import  extract_tensor
from colmap.reconstruct import reconstruct3D_ply
from utils.utils_functions import in_square
import glob
import matplotlib.image as mpimg
from collections import Counter
import tensorflow as tf
from RL.net_sem_2d import Seg_2d

from RL.agent_net import NetAgent

######################################################################## Change values here!!!

# Save plots?
save_plots=True
save_regular_plots=True
toy_case=False
make_movie=False

car_people=False

timestamps=["2020-07-05-17-32-57.868782"]

folder="/visualize_tubingen_000112/"



folder_name="tubingen_000112"
ending=''

calc_dist=True
################################################################################3
settings_path="/some_results/"
settings_ending="settings.txt"
stat_path = "Results/statistics/train/"


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
def get_camera_path( filename):
    basename = os.path.basename(filename)
    parts = basename.split('_')
    city = parts[0]
    seq_nbr = parts[1]

    return basename
################################################################### Main starts here!!!!!
make_vid=False
test_not_in_name=False
train_nbrs = [6, 7, 8, 9]  # , 0,2,4,5]
test_nbrs = [3, 0, 2, 4, 5]

init_names= {-1: "training", 0: "training", 1: "On pedestrian", 3: "On pavement", 2: "By car", 4: "Random",5: "car_env", 6: "Near pedestrian",7:"pedestrian environment", 9:"average"}

save_images_path="/visualization-results"

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

    #test_files, reconstructions_test = sort_files_eval(files_eval)



    labels_to_use, name_movie, target_dir, num_measures, semantic_channels_separated, in_2D=read_settings_file()

        # Make movie
    print(("Files "+str(len(files))))
    #name_movie="random_agent"
    if len(files)>0:
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
        test_points = {0: 0}


        filespath =  "Datasets/colmap2/"
        if not os.path.exists(filespath):
            filespath = "Datasets/colmap/colmap/"
        filename_list=[]
        # for filepath in sorted(glob.glob(filespath + "/*")):
        #     filename_list.append(filepath)
        #
        folders = ["aachen_000042", "tubingen_000112", "tubingen_000136", "bremen_000028", "bremen_000160",
                   "munster_000039", "darmstadt_000019"]
        for folder in folders:
            for filepath in sorted(glob.glob(filespath + "/" + folder + "*")):
                filename_list.append(filepath)
        #filename_list = []
        # for filepath in sorted(glob.glob(filespath + "/stuttgart_000018*")):
        #     filename_list.append(filepath)
        # for filepath in sorted(glob.glob(filespath + "/stuttgart_000039*")):
        #     filename_list.append(filepath)
        # for filepath in sorted(glob.glob(filespath + "/bremen_000014*")):
        #     filename_list.append(filepath)
        # for filepath in sorted(glob.glob(filespath + "/bremen_000028*")):
        #     filename_list.append(filepath)
        # for filepath in sorted(glob.glob(filespath + "/munster_000*")):
        #     filename_list.append(filepath)
        # for filepath in sorted(glob.glob(filespath + "/munster_000146*")):
        #     filename_list.append(filepath)

        # saved_files_counter=0
        # train_set = range(100)
        # test_set = range(100, len(filename_list))
        # viz=True
        # if viz:
        #     images_path=filespath
        #     filename_list = []
        #     for filepath in sorted(glob.glob(images_path + "/stuttgart_000018*")):
        #         filename_list.append(filepath)
        #     for filepath in sorted(glob.glob(images_path + "/stuttgart_000039*")):
        #         filename_list.append(filepath)
        #
        #     for filepath in sorted(glob.glob(images_path + "/stuttgart_000101*")):
        #         filename_list.append(filepath)
        #     for filepath in sorted(glob.glob(images_path + "/bremen_000014*")):
        #         filename_list.append(filepath)
        #     for filepath in sorted(glob.glob(images_path + "/bremen_000028*")):
        #        filename_list.append(filepath)
        #        filename_list.append(filepath)
        #     for filepath in sorted(glob.glob(images_path + "/bremen_000026*")):
        #         filename_list.append(filepath)
        #     for filepath in sorted(glob.glob(images_path + "/jena_000063*")):
        #         filename_list.append(filepath)
        #     for filepath in sorted(glob.glob(images_path + "/jena_000018*")):
        #         filename_list.append(filepath)
        #     for filepath in sorted(glob.glob(images_path + "/munster_000146*")):
        #         filename_list.append(filepath)

        for folder in folders:
            for filepath in sorted(glob.glob(filespath + "/"+folder+"*")):
                filename_list.append(filepath)
            print("Filelist")
        print(filename_list)
        settings = run_settings()
        constants = Constants()
        settings.seq_len_train = 1
        #(self, path, sess, writer, gradBuffer, log,settings, net=None)
        env =  Environment(filespath, None, None, None, None, settings)
        csv_rows = []


        if len(test_files)>0:
            scenes =get_scenes(test_files,test_points)
            prev_nbr = -1
            for scene in scenes:
                filepath_npy=scene[1]
                epoch=scene[2]
                print((filepath_npy.split('_')))

                nbr=filepath_npy.split('_')[-1]
                nbr=nbr[:-len('.npy')]
                print(("Number "+str(nbr)))
                filepath = filename_list[int(nbr)]#150+int(nbr)]#[150 + int(nbr)]
                print((nbr + "              " + filepath))
                basename = os.path.basename(filepath)

                if True:
                    print((filepath_npy.split('_')))
                    if nbr!=prev_nbr:
                        filepath=filename_list[int(nbr)]#+int(nbr)]#[150+int(nbr)]
                        print(filepath)
                        basename=os.path.basename(filepath)
                        # final_reconstruction, people, cars, local_scale_x, ped_dict, cars_2D, people_2D, valid_ids, car_dict, init_frames
                        env.reconstruction, people_rec, cars_rec, scale,_,_ = reconstruct3D_ply(filepath,
                                                                                             settings,
                                                                                             False)

                        pos_x = 0
                        pos_y = -settings.width / 2

                        tensor, density = extract_tensor(pos_x, pos_y, env.reconstruction, settings.height, settings.width,
                                                         settings.depth)

                        print("Episode set up")
                        ep = env.set_up_episode(cars_rec, people_rec, pos_x, pos_y, tensor, False)

                        img_from_above = np.ones((ep.reconstruction.shape[1], ep.reconstruction.shape[2], 3),
                                                 dtype=np.uint8) * 255
                        img = view_2D(ep.reconstruction, img_from_above, 0)
                        print("View PEDESTRIANS")
                        img = view_pedestrians(0, ep.people, img, 0, trans=.15, )  # tensor=ep.reconstruction)

                        img = view_cars(img, ep.cars, img.shape[0], img.shape[1], 0, frame=0)  # ,tensor=ep.reconstruction)
                        img = view_cars(img, ep.cars, img.shape[0], img.shape[1], 0, frame=15)
                        # img = view_cars(img, ep2.cars, img.shape[0], img.shape[1], len(cars)/2, frame=0)  # ,tensor=ep.reconstruction)
                        img2 = view_valid_pos(ep, img, 0)

                        img3 = view_2D_rgb(ep.reconstruction,
                                           np.ones((ep.reconstruction.shape[1], ep.reconstruction.shape[2], 3),
                                                   dtype=np.uint8) * 255, 0, white_background=True)
                        print("View PEDESTRIANS")
                        img3 = view_pedestrians_nicer(0, ep.people, img3, trans=.15)  # tensor=ep.reconstruction)

                        img3 = view_cars_nicer(img3, ep.cars, 0)  # ,tensor=ep.reconstruction)

                    cur_stat=scene[0]
                    agent_pos = cur_stat[ :, 1:3]
                    agent_probabilities = cur_stat[ :, 7:34]
                    agent_reward = cur_stat[ :, 34]
                    agent_measures = cur_stat[ :, 38:]
                    print(filepath_npy)
                    #print agent_pos
                    # print "init " + str(agent_pos[0])
                    # print "View cars" + str(ep.cars[0])
                    # car hit
                    out_of_axis = cur_stat[ -1, 0]
                    img_from_above_loc = view_agent(agent_pos, img_from_above, 0, settings.agent_shape, settings.agent_shape_s,
                                      people=ep.people, cars=ep.cars,
                                      frame=0)
                    img_from_above_loc_rgb = view_agent_nicer(agent_pos, img3, settings.agent_shape,
                                                              settings.agent_shape_s, frame=0)
                    for frame_nbr in range(agent_pos.shape[0]):
                        img_from_above_loc=view_agent(agent_pos, img_from_above_loc, 0, settings.agent_shape, [1,2,2], people=ep.people,cars=ep.cars,
                                   frame=frame_nbr)
                        fig = plt.figure()
                    # plt.imshow(img_from_above_loc)
                    # plt.show()
                    # plt.axis('off')
                    # print save_images_path+folder + basename +"_"+str(timestamp)+"_new_long" + str(epoch) +".png"
                    # fig.savefig(save_images_path+folder + basename +"_"+str(timestamp)+"_new_long" + str(epoch) +".png", bbox_inches='tight')

                    if not os.path.exists(save_images_path+folder):
                        os.mkdir(save_images_path+folder)

                    plt.imshow(img_from_above_loc_rgb)
                    plt.show()
                    plt.axis('off')
                    print((save_images_path+folder + basename +"_"+str(timestamp)+"_rgb_new_long" + str(epoch) +".png"))
                    fig.savefig(save_images_path+folder + basename +"_"+str(timestamp)+"_rgb_new_long" + str(epoch) +".png", bbox_inches='tight')

                    prev_nbr=nbr
                    print(prev_nbr)
                    #input = view_agent_input(img2, agent_pos[ frame_nbr, :], settings.agent_shape, settings.agent_shape_s, ep)


import sys
sys.path.append("RL/") #

print (sys.path)
import numpy as np
import tensorflow as tf
from RL.settings import RANDOM_SEED
tf.set_random_seed(RANDOM_SEED)
import os.path
external=True

import matplotlib.pyplot as plt

import os
from RL.visualization_scripts.Visualize_evaluation import read_settings_file,sort_files_eval
from utils.constants import Constants
from RL.settings import run_settings
from RL.environment_waymo import CARLAEnvironment
from RL.visualization import view_2D, view_pedestrians, view_cars, view_valid_pos,view_2D_rgb,view_pedestrians_nicer,view_cars_nicer,view_agent_nicer, view_valid_init_pos, view_valid_pedestrians, view_agent, view_agent_input, plot_car, colours
from RL.extract_tensor import  extract_tensor
from commonUtils.ReconstructionUtils import reconstruct3D_ply
from RL.settings import run_settings,STATISTICS_INDX_MAP,STATISTICS_INDX_CAR
from RL.agent_car import CarAgent
import glob
from RL.RLmain import get_carla_prefix_eval, RL
from commonUtils.ReconstructionUtils import CreateDefaultDatasetOptions_CarlaRealTime
######################################################################## Change values here!!!

# Save plots?
save_plots=True
save_regular_plots=True
toy_case=False
make_movie=False
viz=True
car_people=False

timestamps=["2021-08-28-13-38-16.928006"]
ending=''

calc_dist=True
folder_name="/corl-supp-new/"
################################################################################3
settings_path="/some_results/"
settings_ending="settings.txt"
stat_path = "Results/statistics/train/"

save_images_path="/visualization-results"

actions_names = ['upL', 'up', 'upR', 'left', 'stand', 'right', 'downL', 'down', 'downR']
labels = ['human hist','car hist' ,'unlabeled' , 'ego vehicle', 'rectification border' ,'out of roi' , 'static' ,
          'dynamic','ground', 'road' , 'sidewalk', 'parking', 'rail track' , 'building' ,  'wall', 'fence' ,
          'guard rail', 'bridge' , 'tunnel' , 'pole'   , 'polegroup', 'traffic light' , 'traffic sign' ,
          'vegetation','terrain' , 'sky','person','rider', 'car'  ,'truck', 'bus',  'caravan', 'trailer',
          'train', 'motorcycle' ,'bicycle', 'license plate' ]


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


def default_seq_lengths( training, evaluate,settings):
    # Set the seq length
    if evaluate:
        seq_len = settings.seq_len_evaluate
    else:
        if training:
            seq_len = settings.seq_len_train
        else:
            seq_len = settings.seq_len_test

    frameRate, frameTime = settings.getFrameRateAndTime()
    seq_len_pfnn = -1
    if settings.pfnn:
        seq_len_pfnn = seq_len * 60 // frameRate

    print(("Environment seq Len {}. seq Len pfnn {}".format(seq_len, seq_len_pfnn)))
    return seq_len, seq_len_pfnn


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

show_heatmaps=True

eval_path = "Results/statistics/val/"

for timestamp in timestamps:
    plt.close("all")
    print("Evaluation ------------------------------")
    path = eval_path + '*' + timestamp
    print(path)
    match = path + "*.npy"
    files = glob.glob(match)

    labels_to_use, name_movie, target_dir, num_measures, semantic_channels_separated, in_2D, temporal_case, continous, num_measures_car, gaussian=read_settings_file(timestamp)
    test_files, reconstructions_test, learn_car, init_map_stats, init_maps, init_cars, test_files_poses= sort_files_eval(files)
    test_points = {0: 0}
    settings = run_settings()
    constants = Constants()



    settings = run_settings()

    val_data = True
    show_cars = True

    f = None

    rl = RL()
    counter, filecounter, images_path, init, net, saver, tmp_net, init_net, goal_net, car_net = rl.pre_run_setup(
        settings)

    if val_data:
        filespath = settings.carla_path_test
        if viz:
            filespath = settings.carla_path_viz
            ending = "test_*"
        ending = "test_*"
        epoch = 0
        filename_list = {}
        saved_files_counter = 0
        # Get files to run on.
        print (filespath + ending)
        for filepath in glob.glob(filespath + ending):
            parts = os.path.basename(filepath).split('_')
            pos = int(parts[-1])
            filename_list[pos] = filepath
    else:
        filespath = images_path
        epoch, filename_list, pos = rl.get_carla_files(settings, settings.realtime_carla_only)
        train_set, val_set, test_set = rl.get_train_test_fileIndices(filename_list, carla=True)
    print (filename_list)
    env, env_car, env_people = rl.get_environments_CARLA(None, filespath, None, None, settings, None)

    car = CarAgent(settings, car_net, None)
    fig = plt.figure()

    position = 174

    filepath = filename_list[position]

    pos_x, pos_y = env.default_camera_pos()
    seq_len, seq_len_pfnn = env.default_seq_lengths(training=True, evaluate=False)
    if settings.new_carla or settings.realtime_carla_only:
        datasetOptions = CreateDefaultDatasetOptions_CarlaRealTime(settings)
    else:
        datasetOptions = None

if len(test_files)>0:
        scenes =get_scenes(test_files,test_points)
        prev_nbr = -1
        for ep_itr, scene in enumerate(scenes):
            filepath_npy=scene[1]
            epoch=scene[2]
            print((filepath_npy.split('_')))

            nbr=filepath_npy.split('_')[-1]
            car_path=filepath_npy[:-len(".npy")] +"_learn_car.npy"
            map_path = filepath_npy[:-len(".npy")] + "_learn_car.npy"

            nbr=int(nbr[:-len('.npy')])*4#file_nbrs[int(nbr[:-len('.npy')])]
            print(nbr)


            if nbr!=prev_nbr or ep_itr>=len() :
                filepath=filename_list[position]#*4
                basename=os.path.basename(filepath)
                if True:
                    training = False
                    # Setup some options for the episode

                    if settings.new_carla or settings.realtime_carla:
                        if settings.new_carla:
                            file_name = filepath[0]
                        else:
                            file_name = filepath
                        datasetOptions = CreateDefaultDatasetOptions_CarlaRealTime(settings)
                        print("New dataset otions ")

                    else:
                        file_name = filepath
                        datasetOptions = None

                    seq_len, seq_len_pfnn=default_seq_lengths( False, True,settings)

                    pos_x=0
                    pos_y=-64
                    ep = env.set_up_episode(get_carla_prefix_eval( False, True,False, False), file_name, pos_x, pos_y, training, evaluate=True, useCaching=settings.useCaching, seq_len_pfnn=seq_len_pfnn,datasetOptions=datasetOptions, car=car)



                    img_from_above = np.ones((ep.reconstruction.shape[1], ep.reconstruction.shape[2], 3),
                                             dtype=np.uint8) * 255
                    img = view_2D(ep.reconstruction, img_from_above, 0)
                    print("View PEDESTRIANS")
                    img = view_pedestrians(0, ep.people, img, 0, trans=.15, )  # tensor=ep.reconstruction)
                    #( current_frame, cars, width, depth, seq_len, car, frame=-1, tensor=np.array((0)),  hit_by_car=-1 , transparency=0.1, x_min=0, y_min=0):
                    img = view_cars(img, ep.cars, img.shape[0], img.shape[1], 0,[], frame=0)  # ,tensor=ep.reconstruction)
                    #img = view_cars(img, ep.cars, img.shape[0], img.shape[1], 0,car, frame=15)
                    # img = view_cars(img, ep2.cars, img.shape[0], img.shape[1], len(cars)/2, frame=0)  # ,tensor=ep.reconstruction)
                    img2 = view_valid_pos(ep, img, 0)

                    img3 = view_2D_rgb(ep.reconstruction, np.ones((ep.reconstruction.shape[1], ep.reconstruction.shape[2], 3),
                                             dtype=np.uint8) * 255, 0,white_background=True)
                    print("View PEDESTRIANS")
                    img3 = view_pedestrians_nicer(0, ep.people, img3, trans=.3 )  # tensor=ep.reconstruction)
                    #current_frame, cars, frame,

                    img3 = view_cars_nicer(img3,ep.cars, 0)  # ,tensor=ep.reconstruction)


            if True:#"test_4" in basename and epoch==9:
                cur_stat=scene[0]
                agent_pos = cur_stat[ :, 1:3]
                agent_probabilities = cur_stat[ :, 7:34]
                agent_reward = cur_stat[ :, 34]
                agent_measures = cur_stat[ :, 38:]
                agent_goals = cur_stat[ 3:5, -1]
                car_stat = np.load(car_path)
                car_poses = car_stat[ep_itr, :, STATISTICS_INDX_CAR.bbox[0]:STATISTICS_INDX_CAR.bbox[1]]

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
                print(car_poses[frame,:])
                img_from_above_loc = view_agent(agent_pos, img_from_above, 0, settings.agent_shape, settings.agent_shape_s,car=car_poses,
                                  people=ep.people, cars=ep.cars,
                                  frame=0, goal=agent_goals)

                img_from_above_loc_rgb= view_agent_nicer(agent_pos, img3, settings.agent_shape, settings.agent_shape_s,0)

                for frame_nbr in range(agent_pos.shape[0]):
                    img_from_above_loc=view_agent(agent_pos, img_from_above_loc, 0, settings.agent_shape, [1,2,2],car=car_poses[frame],
                               frame=frame_nbr, goal=agent_goals)
                transparency = 0.6
                x_min = 0
                y_min = 0
                label = 25
                col = colours[label]
                col = (col[2], col[1], col[0])
                plot_car(car_poses[0], col, img_from_above_loc_rgb, 0, transparency, x_min, y_min)

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


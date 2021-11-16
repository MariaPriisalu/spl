import numpy as np
import os.path, math
import os
import copy

#from utils.constants import Constants
import glob
import pickle
import subprocess

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.insert(0,glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import logging
import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import queue as queue


def draw_image(surface, image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


# Save plots?
save_plots=True
save_regular_plots=True
toy_case=False
make_movie=False
car_people=False

timestamps=["2020-03-01-21-57-41.449517"]
ending=''
calc_dist=True
test_data=True
################################################################################3
settings_path="/some_results/"
settings_ending="settings.txt"
stat_path = "Results/statistics/train/" if USE_LOCAL_PATHS == 0 else "localUserData/Results/statistics/train/"


epoch=3000# update_frequency*100


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
    print("Setting file "+str(settings_file))
    if len(settings_file) == 0:
        settings_file = glob.glob(settings_path + timestamp + "*" + settings_ending)
    if len(settings_file)==0:
        return [],[],[],[],[],[]

    # Get movie name
    name_movie = os.path.basename(settings_file[0])[len(timestamp) + 1:-len("_settings.txt")]
    target_dir = settings_path + name_movie
    #subprocess.call("cp " + settings_file[0] + " " + target_dir + "/", shell=True)
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
            if "Number of measures" in line:
                num_measures = int(line[len("Number of measures: "):])
            if "2D input to network:"in line:
                if line[len("2D input to network: "):].strip()== "True":
                    in_2D=True
                print(line[len("2D input to network: "):].strip() +" "+str(bool(line[len("2D input to network: "):].strip()))+" "+str(in_2D))

    return {}, name_movie, target_dir, num_measures, semantic_channels_separated, in_2D



init_names= {-1: "training", 0: "training", 1: "On pedestrian", 3: "On pavement", 2: "By car", 4: "Random",5: "car_env", 6: "Near pedestrian",7:"pedestrian environment", 9:"average"}

def get_camera_matrix(filepath):
    cameras_path = os.path.join(filepath, 'cameras.p')
    cameras_dict = pickle.load(open(cameras_path, "rb"), encoding="latin1", fix_imports=True)
    frames=sorted(cameras_dict.keys())
    print(cameras_path)

    frame = frames[-1]#np.min(cameras_dict.keys())
    R_inv = cameras_dict[frame]['inverse_rotation'] # Transforms world to camera
    middle_n = R_inv[0:3, 3]
    R = np.transpose(R_inv[0:3, 0:3])
    middle_3 = -np.matmul(R, middle_n)
    #return R, middle, frames
    t=R_inv[0:3, 3]
    C= cameras_dict[frame]['camera_to_car']
    C2=np.matmul(C,R_inv )
    middle_w=-np.matmul(C2[0:3,0:3], C2[0:3,3])
    return R_inv[0:3, 0:3], t, frames, middle_3 ,C, middle_w # , car_pos


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
            #print "File " + str(pair[1]) + " " + str(ep_nbr)
            stats_temp.append([cur_stat[ep_nbr, :, :], pair[1], ep_nbr])
        prev_test_pos = test_points[pair[0]]

    scenes = stats_temp
    return stats_temp

def get_car_rotation_matrix(yaw):
    cy = math.cos(np.radians(yaw))
    sy = math.sin(np.radians(yaw))
    cr = 1
    sr = 0
    cp = 1
    sp = 0
    matrix=np.matrix(np.identity(3))
    matrix[0, 0] =cp * cy
    matrix[0, 1] =cy * sp * sr - sy * cr
    matrix[0, 2] = - (cy * sp * cr + sy * sr)
    matrix[1, 0] =  sy * cp
    matrix[1, 1] =sy * sp * sr + cy * cr
    matrix[1, 2] =cy * sr - sy * sp * cr
    matrix[2, 0] = sp
    matrix[2, 1] = -(cp * sr)
    matrix[2, 2] = cp * cr

    return matrix
P_inv = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) # camera to vehicle

def bbox_of_car(K, R,  car_id, find_poses, frame_cars, height, image_size, middle, scale):
    pos_car = frame_cars[car_id]['transform']

    bbox_car = np.squeeze(np.asarray(np.matmul(np.matmul(R, get_car_rotation_matrix(frame_cars[car_id]['yaw'])),
                                               frame_cars[car_id]['bounding_box'])))
    point_0 = np.squeeze(np.asarray(np.matmul(R, np.array(pos_car).reshape((3, 1))) + middle))
    boxes_2d=[]
    if find_poses:
        x_values = []
        y_values = []
        bbx = [np.array([point_0[0] - bbox_car[0], point_0[1] - bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] - bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] - bbox_car[1], point_0[2] + bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] + bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] + bbox_car[1], point_0[2] + bbox_car[2]]),
               np.array([point_0[0] - bbox_car[0], point_0[1] - bbox_car[1], point_0[2] + bbox_car[2]]),
               np.array([point_0[0] - bbox_car[0], point_0[1] + bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] - bbox_car[0], point_0[1] + bbox_car[1], point_0[2] + bbox_car[2]])]
        for point in bbx:
            point_2D = np.matmul(K, point.reshape((3, 1)))
            point_2D = point_2D / point_2D[2]
            x_2d = int(image_size[0] - point_2D[0])
            y_2d = int(image_size[1] - point_2D[1])
            x_values.append(x_2d)
            y_values.append(y_2d)
        boxes_2d.append([y_values, x_values])
    point_0 = np.squeeze(np.asarray(np.matmul(R, np.array(pos_car).reshape((3, 1))) + middle))
    bbox_car = bbox_car * scale
    point_0 = point_0 * scale
    bbox_car = bbox_car * [-1, -1, 1]  # normal camera coodrinates
    point_0 = point_0 * [-1, -1, 1]  # normal camera coodrinates
    point_0 = np.squeeze(np.asarray(np.matmul(P_inv, point_0)))
    point_0[2] = point_0[2] - height
    bbox_car = np.squeeze(np.asarray(np.matmul(P_inv, bbox_car)))
    car = np.column_stack((point_0 - np.abs(bbox_car), point_0 + np.abs(bbox_car)))
    return car, boxes_2d



################################################################### Main starts here!!!!!

P = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

init_names= {-1: "training", 0: "training", 1: "On pedestrian", 3: "On pavement", 2: "By car", 4: "Random",5: "car_env", 6: "Near pedestrian",7:"pedestrian environment", 9:"average"}

pos_y=-128 / 2
if test_data:

    eval_path = "Results/statistics/val/"
    filespath = "Packages/CARLA_0.8.2/PythonClient/new_data2/"
    if not os.path.exists(filespath):
        filespath = "Datasets/carla-sync/val/"
else:
    filespath = "Packages/CARLA_0.8.2/PythonClient/new_data/"  # "Packages/CARLA_0.8.2/unrealistic_dataset/"
    if not os.path.exists(filespath):
        filespath = "Datasets/carla-sync/train/"




timestamp =timestamps[0]
# find statistics files.
if test_data:
    path=eval_path+'*'+timestamp+"*"
else:
    path = stat_path + timestamp + ending + '*'
match=path+"*.npy"

files=glob.glob(match)

labels_to_use, name_movie, target_dir, num_measures, semantic_channels_separated, in_2D=read_settings_file()


if len(files)>0:

    #filenames_itr_cars, test_files_cars, filenames_itr_people,test_files_people,  filenames_itr, test_files, reconstructions_test, iterations, iterations_cars, special_cases
    #
    if test_data:
        test_files, reconstructions_test = sort_files_eval(files)
    else:
        filenames_itr_cars, test_files_cars, filenames_itr_people, test_files_people, filenames_itr, test_files, reconstructions_test, iterations, iterations_cars,special_cases=sort_files(files)

    print("Test  files: "+str(test_files))

    train_itr={}
    train_counter=0
    test_points={}

    if test_data:
        test_points = {0: 0}
    else:
        for itr in sorted(iterations.keys()):

            training=iterations[itr]
            if training:
                train_itr[itr]=train_counter
                if itr in special_cases:
                    test_points[itr] = train_counter
                if itr not in iterations_cars:
                    train_counter=train_counter+1
            else:
                test_points[itr]= train_counter
                if itr in special_cases:
                    train_itr[itr] = train_counter

    epoch = 0
    filename_list = {}
    # Get files to run on.
    ending_local = "test_*"
    for filepath in glob.glob(filespath + ending_local):
        parts = os.path.basename(filepath).split('_')
        pos = int(parts[-1])
        filename_list[pos] = filepath
    saved_files_counter = 0
    actor_perspective=True
    csv_rows = []

    if len(test_files)>0:
        scenes =get_scenes(test_files,test_points)
    results_path='Packages/CARLA_0.8.2/viz_results2'
    if not os.path.exists(results_path):
        results_path = "Results/viz_results2"
    prev_nbr=-1
    for scene in scenes:
        filepath_npy = scene[1]

        file_nbrs = [0, 8, 24, 36]
        if test_data:
            nbr = filepath_npy.split('_')[-1]
            nbr = nbr[:-len('.npy')]
            nbr=int(nbr)*4
            print(nbr)
        else:
            nbr = filepath_npy.split('_')[2]
        #print "Number "+str(nbr)

        epoch = scene[2]
        cur_stat = scene[0]
        agent_pos = cur_stat[:, 0:3]
        agent_action = cur_stat[ :, 6]
        agent_probabilities = cur_stat[:, 7:34]
        agent_reward = cur_stat[:, 34]
        agent_measures = cur_stat[:, 38:]
        agent_goals = cur_stat[3:5, -1]
        reached_goal = agent_measures[:,13]
        yaw_mapping=[-0.75, -0.5,-0.25,-1,0,0.25,0.75,0.5,0]

        # print filepath_npy
        # print str(nbr) +" "+str(epoch)
        if int(nbr)==32 and int(epoch) ==8:
            print(filepath_npy)
            print(epoch)
            filepath = filename_list[int(nbr)]
            basename = os.path.basename(filepath)
            print(filepath)

            cars_path = os.path.join(filepath, 'cars.p')
            cars_dict = pickle.load(open(cars_path, "rb"), encoding="latin1", fix_imports=True)
            print(cars_path)

            people_path = os.path.join(filepath, 'people.p')
            people_dict = pickle.load(open(people_path, "rb"), encoding="latin1", fix_imports=True)
            print(people_path)

            centering = pickle.load(open(os.path.join(filepath, "centering.p"), "rb"), encoding="latin1", fix_imports=True)
            print(os.path.join(filepath, "centering.p"))

            start_poses = pickle.load(open( "start_positions.p", "rb"), encoding="latin1", fix_imports=True)
            print("start_positions.p")

            R, middle, frames, middle_2,C, middle_w = get_camera_matrix(filepath)

            #grid_axis_x=[]
            # for y in range(128):
            #     p=[]



            agent_positions=[]
            agent_yaw=[]
            previous_pos=[0,0]
            p = np.array([0, agent_goals[0], agent_goals[1]])

            print("point goal " + str(p))
            p[1] = p[1] + pos_y

            print("point+y " + str(p))
            p[0] = p[0] + centering["height"]

            print("point+ height " + str(p))

            p = np.reshape(p[[2, 1, 0]], (3, 1)) * (1.0 / centering['scale'])

            print("point scaled " + str(p))
            p = np.matmul(P, p)

            print("point P" + str(p))
            p = p * np.reshape([-1, -1, 1], (3, 1))

            print("point p " + str(p))
            p = p - np.reshape(middle, (3, 1))

            print("point middle " + str(p))
            p = np.matmul(np.transpose(R), p)

            print("goal " + str(p))
            agent_goal = p
            directions=[]
            directions_angle=[]
            print(agent_pos)
            for frame in range(agent_pos.shape[0]):
                if reached_goal[max(frame-15,0)]==0:

                    p = agent_pos[frame, :]
                    if frame==0:
                        print("point " + str(p))
                    p[1] = p[1] + pos_y
                    if frame == 0:
                        print("point+y " + str(p))
                    p[0] = p[0] + centering["height"]
                    if frame == 0:
                        print("point+ height " + str(p))

                    p = np.reshape(p[[2, 1, 0]], (3, 1)) * (1.0 /centering['scale'])
                    if frame == 0:
                        print("point scaled " + str(p))
                    p=np.matmul(P,p)
                    if frame == 0:
                        print("point P" + str(p))
                    p=p* np.reshape([-1, -1, 1], (3,1))
                    if frame == 0:
                        print("point p " + str(p))
                    p=p-np.reshape(middle, (3, 1))
                    if frame == 0:
                        print("point middle " + str(p))
                    p=np.matmul(np.transpose(R),p)
                    if frame == 0:
                        print("point R " + str(p))

                    agent_positions.append(p.copy())
                    if frame==0:
                        agent_yaw.append(0)
                        agent_yaw.append(0)
                    else:

                        directions.append([p.item(0)-previous_pos[0],p.item(1)-previous_pos[1],0] )
                        directions_angle.append(math.atan2(p.item(1)-previous_pos[1],p.item(0)-previous_pos[0] )*180/math.pi)
                        if False:
                            #len(agent_yaw)>4 and np.sign(directions[-1])==np.sign(directions[-2]) and np.sign(directions[-2])==np.sign(directions[-3]) and np.sign(directions[-3])==np.sign(directions[-4]):
                            # dir=math.atan2(p[1]-previous_pos[1],p[0]-previous_pos[0] )*180/math.pi
                            # if len(agent_yaw)>2 and np.sign(dir)==np.sign(agent_yaw[-2]) and  np.sign(-dir)==np.sign(agent_yaw[-1]):
                            #     agent_yaw[-1]=np.mean([dir,agent_yaw[-2]]) # -90)

                            agent_yaw.append(math.atan2(p[1]-previous_pos[1],p[0]-previous_pos[0] )*180/math.pi)#math.atan2(p[1]-previous_pos[1],p[0]-previous_pos[0] )*180/math.pi)#-90)
                        else:
                            dir = math.atan2(agent_goal[1]-p[1], agent_goal[0]-p[0]) * 180 / math.pi
                            # if len(agent_yaw) > 2 and np.sign(dir) == np.sign(agent_yaw[-2]) and np.sign(
                            #         -dir) == np.sign(agent_yaw[-1]):
                            #     agent_yaw[-1] = np.mean([dir, agent_yaw[-2]])  # -90)

                            agent_yaw.append(math.atan2(agent_goal[1]-p[1], agent_goal[0]-p[0]) * 180 / math.pi)  # -90)

                        #print str(p[0]-previous_pos[0])+" "+str(p[1]-previous_pos[1])+" "+str(agent_yaw[-1])
                    previous_pos=[p.item(0),p.item(1)]
                #agent_yaw.append(yaw_mapping[int(agent_action[frame])]*180)#math.pi)


            camera_yaw=math.atan2(R[1,0],R[0,0])
            camera_pitch = math.atan2(-R[2,0] ,np.sqrt( R[2,1]**2+(R[2,2]**2)))
            camera_roll = math.atan2(R[2,1] , R[2,2])

            actor_list = []
            car_list={}
            pygame.init()

            client = carla.Client('localhost', 2000)
            client.set_timeout(2.0)

            world = client.get_world()
            print("initial speed: "+str(np.linalg.norm(directions)*10))
            control=carla.WalkerControl(speed=np.linalg.norm(directions)*10)



            print('enabling synchronous mode.')
            settings = world.get_settings()
            settings.synchronous_mode = True
            world.apply_settings(settings)


            cur_yaw=0
            name_movie=""
            name_movie_main=""
            try:
                m = world.get_map()
                print("starting ")
                poses= m.get_spawn_points()

                start_pose=poses[int(nbr)]

                print("Location")
                print(start_pose.location)
                print(middle_2)
                print(middle_w)
                print("Old Location")
                print(start_poses[int(nbr)])
                print("Centering")
                print(centering["middle"])
                init_pos = carla.Location(x=middle_w[0,0], y=middle_w[1,0], z=1.26)
                init_rot = carla.Rotation(yaw=0, pitch=0, roll=0)
                blueprint_library = world.get_blueprint_library()
                vehicles=blueprint_library.filter('vehicle.*')
                car_bp = [x for x in vehicles if int(x.get_attribute('number_of_wheels')) == 4]

                if not actor_perspective:
                    init_trans = carla.Transform(init_pos, init_rot)
                    spectator = world.get_spectator()
                    spectator.set_transform(init_trans)

                print("Position of actor")
                print(agent_positions[0])

                agent_init_pos = carla.Location(x=agent_positions[0].item(0), y=agent_positions[0].item(1), z=1.3)
                print(agent_init_pos)
                agent_init_rot = carla.Rotation(yaw=directions_angle[0], pitch=0, roll=0)
                print(agent_init_rot)
                bp = random.choice(blueprint_library.filter('walker.pedestrian.0002'))



                transform =  carla.Transform(agent_init_pos, agent_init_rot)
                print("Here")
                #if not actor_perspective:
                actor = world.try_spawn_actor(bp, transform)
                if actor is not None:
                    print("Initialized actor")
                else:
                    "Failed to initialize actor"
                # else:
                #     actor=None
                #     spectator = world.get_spectator()
                #     spectator.set_transform(transform)
                #static.prop.box03
                rotation = actor.get_transform().rotation
                #actor.set_velocity(carla.Vector3D(directions[0][0]/np.linalg.norm(disp),directions[0][1]/np.linalg.norm(disp),0.0 ))
                goal_box = None
                goal_based=True
                if goal_based:
                    bp_goal = random.choice(blueprint_library.filter('static.prop.box03'))
                    agent_goal_pos = carla.Location(x=agent_goal.item(0), y=agent_goal.item(1), z=0.3)
                    print(agent_goal_pos)
                    agent_goal_rot = carla.Rotation(yaw=0, pitch=0, roll=0)
                    print(agent_goal_rot)
                    transform_goal = carla.Transform(agent_goal_pos, agent_goal_rot)


                    goal_box = world.try_spawn_actor(bp_goal, transform_goal)

                vehicle_map={}
                pedestrian_map={}
                vehicle_vel_map = {}
                pedestrian_vel_map = {}

                prev_pos=[copy.copy(agent_init_pos.x), copy.copy(agent_init_pos.y), copy.copy(agent_init_pos.z)]
                camera_transform = carla.Transform(carla.Location(x=0.3,y=0.0, z=0),carla.Rotation(yaw=0, pitch=0, roll=0))
                if not actor_perspective:
                    camera = world.spawn_actor(
                        blueprint_library.find('sensor.camera.rgb'),
                        camera_transform,
                        attach_to=spectator)
                else:
                    camera = world.spawn_actor(
                        blueprint_library.find('sensor.camera.rgb'),
                        carla.Transform(carla.Location(x=agent_positions[0].item(0)-5.5, y=agent_positions[0].item(1), z=1+2.8), carla.Rotation(pitch=-15)))
                        #attach_to=actor)

                for car_key, car in cars_dict[50].items():
                    npc=None
                    tries=0
                    while npc is None and tries<3:
                        bp=random.choice(car_bp)#blueprint_library.filter('vehicle.*'))#z=1
                        transform=carla.Transform(carla.Location(x=car['transform'][0], y=car['transform'][1], z=0.35),carla.Rotation(yaw=car['yaw'],pitch=0, roll=0))
                        npc = world.try_spawn_actor(bp, transform)
                        tries=tries+1
                        if npc is not None:
                            vehicle_map[car_key]=npc
                            print(('created %s' % car_key))#npc.type_id)


                ped_controllers={}
                ped_rotations={}
                for car_key, car in people_dict[50].items():
                    npc = None
                    tries = 0
                    while npc is None and tries < 3:
                        bp = random.choice(blueprint_library.filter('walker.pedestrian.0001'))
                        print("Height "+str(car['bounding_box'][2]))
                        # print "Typical yaw"+str(car['yaw'])
                        transform = carla.Transform(carla.Location(x=car['transform'][0], y=car['transform'][1], z=car['bounding_box'][2]+0.3),
                                                    carla.Rotation(yaw=car['yaw'], pitch=0, roll=0))

                        npc = world.try_spawn_actor(bp, transform)
                        tries = tries + 1
                        if npc is not None:
                            pedestrian_map[car_key] = npc
                            ped_controllers[car_key] = carla.WalkerControl()
                            ped_rotations[car_key]=npc.get_transform().rotation
                            print(('created %s' % npc.type_id))



                # Make sync queue for sensor data.
                image_queue = queue.Queue()
                camera.listen(image_queue.put)

                frame = None
                my_frame=0

                display = pygame.display.set_mode(
                    (800, 600),
                    pygame.HWSURFACE | pygame.DOUBLEBUF)
                font = get_font()

                clock = pygame.time.Clock()

                while my_frame<len(agent_positions):
                    if should_quit():
                        break

                    clock.tick()
                    world.tick()
                    ts = world.wait_for_tick()

                    if frame is not None:
                        if ts.frame_count != frame + 1:
                            logging.warning('frame skip!')

                    frame = ts.frame_count

                    while True:
                        image = image_queue.get()
                        if image.frame_number == ts.frame_count:
                            break
                        logging.warning(
                            'wrong image time-stampstamp: frame=%d, image.frame=%d',
                            ts.frame_count,
                            image.frame_number)

                    for key, car in vehicle_map.items():
                        if my_frame + 50 < len(cars_dict):
                            if key in cars_dict[my_frame+50]:
                                pos=cars_dict[my_frame+50][key]
                                transform = carla.Transform(
                                    carla.Location(x=pos['transform'][0], y=pos['transform'][1], z=0.2),
                                    carla.Rotation(yaw=pos['yaw'], pitch=0, roll=0))
                                car.set_transform(transform)

                    for key, person in pedestrian_map.items():
                        if my_frame+50<len(people_dict):
                            if key in people_dict[my_frame+50]:
                                pos=people_dict[my_frame+50][key]
                                cur_pos = person.get_transform().location
                                disp = [pos['transform'][0] - cur_pos.x,
                                        pos['transform'][1] - cur_pos.y]  # , 1.3 - cur_pos.y]
                                ped_controllers[car_key].speed = np.linalg.norm(disp)  # directions[my_frame])
                                ped_rotations[car_key].yaw=pos['yaw']
                                ped_controllers[car_key].direction = ped_rotations[car_key].get_forward_vector()
                                person.apply_control(ped_controllers[car_key])
                                #angle = math.atan2(disp[1], disp[0]) * 180 / math.pi
                                # transform = carla.Transform(
                                #     carla.Location(x=pos['transform'][0], y=pos['transform'][1], z=pos['bounding_box'][2]+0.3),
                                #     carla.Rotation(yaw=pos['yaw'], pitch=0, roll=0))

                                #person.set_transform(transform)
                    if actor is not None:
                        if len(agent_positions)> my_frame:
                            agent_init_pos = carla.Location(x=agent_positions[my_frame].item(0), y=agent_positions[my_frame].item(1), z=1.3)
                            agent_init_rot=carla.Rotation(yaw=agent_yaw[my_frame], pitch=0, roll=0)
                            cur_yaw+=agent_yaw[my_frame]
                            cur_yaw=cur_yaw%360

                            # actor.set_transform(carla.Transform(agent_init_pos, agent_init_rot))
                            camera.set_transform(carla.Transform(carla.Location(x=agent_positions[my_frame].item(0) - 5.5, y=agent_positions[my_frame].item(1), z=1 + 2.8),carla.Rotation(pitch=-15)))



                            cur_pos = actor.get_transform().location
                            disp = [agent_positions[my_frame ].item(0) - cur_pos.x,
                                    agent_positions[my_frame ].item(1)- cur_pos.y]#, 1.3 - cur_pos.y]
                            angle = math.atan2(disp[1], disp[0]) * 180 / math.pi
                            print(disp)
                            #if my_frame<5:

                            control.speed =np.linalg.norm(disp)*1.5#*2.5#*4.5#directions[my_frame])


                            # rotation.yaw = round(rotation.yaw, 1)
                            print("Speed " + str(control.speed))

                            if my_frame < len(directions):
                                print("Direction " + str([directions[my_frame][0], directions[my_frame][1], 0.0]))
                                rotation.yaw =directions_angle[my_frame]
                                rotation.yaw = round(rotation.yaw, 1)
                                control.direction = rotation.get_forward_vector()
                                #control.direction = carla.Vector3D(directions[my_frame][0],directions[my_frame][1],0.0 )
                                print("Agent location "+str(actor.get_transform().location)+" wanted pos "+str(agent_init_pos))

                            # self._rotation.yaw = round(self._rotation.yaw, 1)
                            # self._control.direction = self._rotation.get_forward_vector()
                            actor.apply_control(control)
                            pos_n=actor.get_transform().location
                            new_pos=[prev_pos[0]-pos_n.x, prev_pos[1]-pos_n.y,prev_pos[2]-pos_n.z]
                            print("Actual speed: "+str(np.linalg.norm(new_pos)))
                            prev_pos = [copy.copy(cur_pos.x), copy.copy(cur_pos.y), copy.copy(cur_pos.z)]

                    # if actor_perspective:
                    #     if len(agent_positions) > my_frame:
                    #         agent_init_pos = carla.Location(x=agent_positions[my_frame].item(0),
                    #                                         y=agent_positions[my_frame].item(1), z=1.26)
                    #         agent_init_rot = carla.Rotation(yaw=cur_yaw, pitch=0, roll=0)
                    #         cur_yaw += agent_yaw[my_frame]
                    #         cur_yaw = cur_yaw % 360
                    #         print "Current yaw "+str( cur_yaw) +" "+str(agent_yaw[my_frame])
                    #         spectator.set_transform(carla.Transform(agent_init_pos, agent_init_rot))
                    draw_image(display, image)
                    my_frame=my_frame+1
                    if not actor_perspective:
                        image.save_to_disk('_out/test_%02d_%02d_%06d.jpg' % (int(nbr),int(epoch),my_frame))
                        name_movie_main='_out/test_%02d_%02d' % (int(nbr),int(epoch))
                        name_movie =name_movie_main+'_%06d.jpg'
                    else:
                        image.save_to_disk('_out/test_perpective_%02d_%02d_%06d.jpg' % (int(nbr), int(epoch), my_frame))
                        name_movie_main = '_out/test_perpective_%02d_%02d' % (int(nbr),int(epoch))
                        name_movie = name_movie_main + '_%06d.jpg'
                    text_surface = font.render('% 5d FPS' % clock.get_fps(), True, (255, 255, 255))
                    display.blit(text_surface, (8, 10))
                    pygame.display.flip()

            finally:
                print('\ndisabling synchronous mode.')
                settings = world.get_settings()
                settings.synchronous_mode = False
                world.apply_settings(settings)

                print('destroying actors.')

                for key, car in vehicle_map.items():
                    car.destroy()

                for key, car in pedestrian_map.items():
                    car.destroy()
                camera.destroy()
                if actor is not None:
                    actor.destroy()
                pygame.quit()
                if goal_box is not None:
                    goal_box.destroy()
                print('done.')
            if not actor_perspective:
                command = "ffmpeg -framerate 10 -i " +name_movie + ' -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -y ' +name_movie_main+ '.mp4'
            else:
                command = "ffmpeg -framerate 10 -i " + name_movie + ' -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -y ' + name_movie_main + '_perspective.mp4'

            print(command)
            #subprocess.call(command, shell=True)


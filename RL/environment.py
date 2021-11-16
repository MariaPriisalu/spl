import os
from datetime import datetime

import numpy as np
from visualization import make_movie,make_movie_eval
from episode import SimpleEpisode
from environment_abstract import AbstractEnvironment
from extract_tensor import objects_in_range, extract_tensor, objects_in_range_map
#from reconstruct3D.reconstruct3D import reconstruct3D
from colmap.reconstruct import reconstruct3D_ply, camera_to_cityscapes_coord
from commonUtils.ReconstructionUtils import CreateDefaultDatasetOptions_Cityscapes
import time, scipy
from settings import USE_LOCAL_PATHS

class Environment(AbstractEnvironment):

    def work(self, cachePrefix, file_name, agent, poses_db, epoch,saved_files_counter, training=True, road_width=5, conv_once=False,time_file=None, act=True):
        # Setup some options for the episode
        file_agent = self.get_file_agent_path(file_name)
        pos_x, pos_y = self.default_camera_pos()
        seq_len, seq_len_pfnn = self.default_seq_lengths(training, evaluate=False)

        # Create the desired dataset options structure
        options = CreateDefaultDatasetOptions_Cityscapes(None)
        options.LIMIT_FRAME_NUMBER = max(seq_len, seq_len_pfnn)
        options.envSettings = self.settings

        # Setup the episode
        start_setup = time.time()
        episode = self.set_up_episode(cachePrefix, file_name, pos_x, pos_y, training, useCaching=True, time_file=time_file, seq_len_pfnn=seq_len_pfnn, datasetOptions=options)
        setup_time = time.time() - start_setup
        print(("Setup time in s {:0.2f}s".format(setup_time)))

        # Initialize the agent
        run_episode = True
        test_init, succedeed, validRun = self.default_initialize_agent(training, agent, episode, None)
        if not succedeed:
            return self.counter, saved_files_counter

        # Act and learn for a repeated number of times
        number_of_runs_per_scene = 1
        if training:
            number_of_runs_per_scene = 1

        if act and (not test_init or run_episode):
            saved_files_counter,initializer_stats = self.default_doActAndGetStats(training, number_of_runs_per_scene, agent, episode, poses_db, file_agent, file_name, saved_files_counter)
        else:
            print("Not valid initialization")
            return self.counter, saved_files_counter

        return self.counter, saved_files_counter,initializer_stats

    def visualizeFromScript(self):


        with open("stats.pkl", "rb") as testFile:
            Objects_serialized = pickle.load(testFile, encoding="latin1", fix_imports=True)

        episode=Objects_serialized['episode']
        statistics_list = Objects_serialized['statistics']
        poses = Objects_serialized['poses']
        file_name = Objects_serialized['filename']
        seq_len = Objects_serialized['seq_len']
        episode = Objects_serialized['episode' ]
        training = Objects_serialized['training']
        jointsParents = Objects_serialized['jointParents']
        initialization_map=Objects_serialized['initialization_map']
        initialization_map=Objects_serialized['initialization_map']

        self.frame_counter = make_movie(episode.people,
                                        episode.reconstruction,
                                        statistics_list,
                                        self.settings.width,
                                        self.settings.depth,
                                        seq_len,
                                        self.frame_counter,
                                        self.settings.agent_shape,
                                        self.settings.agent_shape_s, episode,
                                        poses,
                                        initialization_map,
                                        initialization_car,
                                        training=training,
                                        name_movie=self.settings.name_movie,
                                        path_results=self.settings.statistics_dir + "/agent/",
                                        episode_name=os.path.basename(file_name),
                                        action_reorder=self.settings.reorder_actions,
                                        velocity_show=self.settings.velocity,
                                        velocity_sigma=self.settings.velocity_sigmoid,
                                        jointsParents=jointsParents,
                                        continous=self.settings.continous)


    def visualize(self, episode, file_name, statistics, training, poses, initialization_map,initialization_car,agent):
        seq_len=self.settings.seq_len_train
        if not training:
            seq_len =self.settings.seq_len_test
        jointsParents = None
        if self.settings.pfnn:
            jointsParents = agent.PFNN.getJointParents()


        HACK_ENABLED_SAVE = False
        if HACK_ENABLED_SAVE:
            # 'people' : episode.people, 'recontruction' : episode.reconstruction,
            Objects = {'statistics' : [statistics], 'poses' : poses, 'filename' : file_name
                       , 'seq_len' : seq_len, 'episode':episode, 'training' : training, 'jointParents' : jointsParents}

            with open("stats.pkl", "wb") as statsFile:
                Objects_serialized = pickle.dump(Objects, statsFile, protocol=pickle.HIGHEST_PROTOCOL)

            # Sanity check
            self.visualizeFromScript()
            exit(0)


        self.frame_counter = make_movie(episode.people,
                                        episode.reconstruction,
                                        [statistics],
                                        self.settings.width,
                                        self.settings.depth,
                                        seq_len,
                                        self.frame_counter,
                                        self.settings.agent_shape,
                                        self.settings.agent_shape_s, episode,
                                        poses,
                                        initialization_map,
                                        initialization_car,
                                        training=training,
                                        name_movie=self.settings.name_movie,
                                        path_results=self.settings.statistics_dir + "/agent/",
                                        episode_name=os.path.basename(file_name),
                                        action_reorder=self.settings.reorder_actions,
                                        velocity_show=self.settings.velocity,
                                        velocity_sigma=self.settings.velocity_sigmoid,
                                        jointsParents=jointsParents,
                                        continous=self.settings.continous)


    def evaluate(self, cachePrefix, file_name, agent, file_agent,saved_files_counter, viz=False, folder=""):
        training = False

        # Setup some options for the episode
        file_agent = self.get_file_agent_path(file_name)
        pos_x, pos_y = self.default_camera_pos()
        seq_len, seq_len_pfnn = self.default_seq_lengths(training=training, evaluate=True)
        self.settings.seq_len_test= seq_len # Override the test seq length because this is the one actually used, there is no different variable for evaluate in the core code

        options = CreateDefaultDatasetOptions_Cityscapes(None)
        options.LIMIT_FRAME_NUMBER = max(seq_len, seq_len_pfnn)
        options.envSettings = self.settings

        # Setup the episode
        start_setup = time.time()
        episode = self.set_up_episode(cachePrefix, file_name, pos_x, pos_y, training, useCaching=True,evaluate=True, seq_len_pfnn=seq_len_pfnn, datasetOptions=options)
        setup_time = time.time() - start_setup
        print(("Setup time in s {:0.2f}s".format(setup_time)))

        run_episode=True
        test_init, succedeed, validRun = self.default_initialize_agent(training, agent, episode, None)
        if not succedeed:
            return self.counter, saved_files_counter
        run_episode |= validRun

        # Act and get stats
        stats = []
        number_of_runs_per_scene = 1

        if (not test_init or run_episode):
            saved_files_counter,initializer_stats = self.default_doActAndGetStats(training,number_of_runs_per_scene, agent, episode, None, file_agent, file_name, saved_files_counter, outStatsGather=stats, evaluate=True)
        else:
            print ("Not valid initialization")
            return self.counter, saved_files_counter, None

        return stats, saved_files_counter,initializer_stats

    def visualize_aachen(self, camera_pos, middle, pos_x, pos_y, scale_vector, seq_len_pfnn, tensor):
        people_list = []
        cars_list = []
        cars_list_colmap = []
        for f in range(30):
            people_list.append([])
            cars_list.append([])
            p = camera_pos[f, :].copy()
            p = camera_to_cityscapes_coord(p, middle, scale_vector[0]).astype(int)
            p = p[[2, 1, 0]]
            p[1] = p[1] + (128 // 2)
            cars_list[-1].append(np.array([p[0] - 4, p[0] + 4, p[1] - 4, p[1] + 4, p[2] - 4, p[2] + 4]))
        cars_list[0] = cars_list[1]
        people_list[5] = [np.array([[1, 9], [75, 83], [95, 103]])]
        people_list[6] = [np.array([[1, 9], [75, 83], [95, 103]]), np.array([[1, 9], [73, 81], [91, 99]])]
        people_list[7] = [np.array([[1, 9], [75, 83], [95, 103]]), np.array([[1, 9], [73, 81], [91, 99]])]
        people_list[8] = [np.array([[1, 9], [74, 82], [94, 102]]), np.array([[1, 9], [72, 80], [90, 98]])]
        people_list[9] = [np.array([[1, 9], [74, 82], [93, 101]]), np.array([[1, 9], [72, 80], [90, 98]])]
        people_list[10] = [np.array([[1, 9], [74, 82], [93, 101]]), np.array([[1, 9], [72, 80], [90, 98]])]
        people_list[11] = [np.array([[1, 9], [74, 82], [93, 101]]), np.array([[1, 9], [72, 80], [90, 98]])]
        people_list[12] = [np.array([[1, 9], [74, 82], [93, 101]]), np.array([[1, 9], [72, 80], [90, 98]])]
        people_list[13] = [np.array([[1, 9], [74, 82], [92, 100]]), np.array([[1, 9], [72, 80], [90, 98]])]
        people_list[14] = [np.array([[1, 9], [73, 81], [90, 98]]), np.array([[1, 9], [71, 79], [86, 94]])]
        people_list[15] = [np.array([[1, 9], [72, 80], [89, 97]]), np.array([[1, 9], [70, 78], [85, 93]])]
        people_list[16] = [np.array([[1, 9], [72, 80], [86, 94]]), np.array([[1, 9], [70, 78], [82, 90]])]
        people_list[17] = [np.array([[1, 9], [72, 80], [86, 94]]), np.array([[1, 9], [70, 78], [82, 90]])]
        people_list[18] = [np.array([[1, 9], [72, 80], [86, 94]]), np.array([[1, 9], [70, 78], [82, 90]])]
        people_list[19] = [np.array([[1, 9], [71, 79], [86, 94]]), np.array([[1, 9], [69, 77], [82, 90]])]
        people_list[20] = [np.array([[1, 9], [71, 79], [86, 94]]), np.array([[1, 9], [69, 77], [82, 90]])]
        people_list[21] = [np.array([[1, 9], [70, 78], [86, 94]]), np.array([[1, 9], [68, 76], [82, 90]])]
        people_list[22] = [np.array([[1, 9], [70, 78], [86, 94]]), np.array([[1, 9], [68, 76], [82, 90]])]
        people_list[23] = [np.array([[1, 9], [70, 78], [86, 94]]), np.array([[1, 9], [68, 76], [82, 90]])]
        people_list[24] = [np.array([[1, 9], [70, 78], [82, 90]]), np.array([[0, 8], [68, 76], [79, 87]])]
        people_list[25] = [np.array([[1, 9], [70, 78], [82, 90]]), np.array([[0, 8], [68, 76], [79, 87]])]
        people_list[26] = [np.array([[1, 9], [69, 77], [82, 90]]), np.array([[0, 8], [67, 75], [79, 87]])]
        people_list[27] = [np.array([[1, 9], [68, 76], [82, 90]]), np.array([[0, 8], [66, 74], [79, 87]])]
        people_list[28] = [np.array([[1, 9], [68, 76], [82, 90]]), np.array([[0, 8], [66, 74], [79, 87]])]
        people_list[29] = [np.array([[1, 9], [67, 75], [82, 90]]), np.array([[0, 8], [65, 73], [79, 87]])]
        print("People list--- frame 5")
        print((people_list[5]))
        print("People list--- frame 10")
        print((people_list[10]))
        episode = self.set_up_episode(cars_list, people_list, pos_x, pos_y, tensor, False, evaluate=True, seq_len_pfnn=seq_len_pfnn)
        print("After episode set up ")
        print("People list--- frame 5")
        print((episode.people[5]))
        print("People list--- frame 10")
        print((episode.people[10]))
        return episode

    def agent_initialization(self, agent, episode, itr,  poses_db, training, init_m=-1, set_goal=""):
        init_key=itr
        # 1,6,2,4
        init_names = {1:"on ped", 6:"near ped", 2:"by car", 4:"random", -1 : "train"}
        #aachen_000024
        init_pairs={}
        init_pairs["aachen_000042"]=[([103, 68],[213,38]), ([103, 68],[254,45]), ([103, 68],[72,70]),
                    ([132, 42],[248,46]), ([132, 42],[60,23]), ([132, 42],[64, 32]), ([132, 42],[42,38]),
                    ([60, 25],[123,29]), ([60, 25],[121,2]), ([60, 25],[72,73]), ([60, 25],[141,70]),
                    ([60, 25],[252,49]) ]
        init_pairs["tubingen_000136"] = [([76, 16], [113, 62]), ([76, 16], [100, 83]), ([76, 16], [16, 62]), ([76, 16], [107,4]), ([76, 16], [169, 78]), ([76, 16], [150, 112])]
        #init_pairs["tubingen_000112"] = [([131, 2], [7, 78]), ([131, 2], [69, 104]),([131, 2], [100, 78]), ([131, 2], [74, 123]), ([131, 2], [152, 64])]
        init_pairs["tubingen_000112"] = [(np.array([ 5,  79,  99]), [85, 123]), (np.array([ 5,  79,  99]), [21, 2]), (np.array([ 5,  79,  99]), [6, 124])]

        init_pairs["bremen_000028"] = [([40, 71], [145, 40]), ([40, 71], [166, 54]),  ([12, 71], [157, 95]),  ([12, 71], [169, 47]),  ([12, 71], [86, 40])]
        init_pairs["bremen_000160"] = [([18, 42], [145, 82]), ([18, 42], [251, 57]),([18, 42], [144, 83]), ([18, 42], [79, 84]),
                                       ([58, 54], [145, 82]), ([58, 54], [251, 57]), ([58, 54], [144, 83]),
                                       ([58, 54], [79, 84])]
        init_pairs["munster_000039"] = [([23, 57], [118, 97]), ([23, 57], [190, 10]),([23, 57], [140, 42]),([23, 57], [127, 94])]
        init_pairs["darmstadt_000019"] = [([27, 52], [191, 18]),([27, 52], [80, 95]),([27, 52], [243, 89]),([27, 52], [59, 101]), ([59, 101], [191, 18]), ([59, 101], [243, 89])]

        if len(set_goal)>0:
            goal = init_pairs[set_goal][itr % len(init_pairs[set_goal])][1]

            if "tubingen_000112" in set_goal:
                episode.agent[0]=init_pairs[set_goal][itr%len(init_pairs[set_goal])][0]
                episode.goal = np.array([episode.agent[0][0], 128 - goal[1], goal[0]])
            else:
                pos_x=init_pairs[set_goal][itr%len(init_pairs[set_goal])][0]
                h = episode.get_height_init()
                episode.agent[0]=np.array([h, 128-pos_x[1], pos_x[0]])
                episode.goal=np.array([h, 128-goal[1], goal[0]])
            print(("Set pos "+str(episode.agent[0])+" "+str(goal)+" "+str(episode.goal)))
            return episode.agent[0], -1, episode.vel_init
        if itr >= len(episode.valid_keys) and len(episode.valid_keys)>0 :
            init_key=itr%len(episode.valid_keys)


        if init_m>0:
            print(("init_m- initialization method "+str(init_m)+" "+str(init_key)+" "+str(training)))
            pos, indx, vel_init = episode.initial_position(poses_db, training=training, init_key=init_key, initialization=init_m)
        elif len(episode.valid_keys) > 0:
            print(("valid_keys"+" "+str(init_key)+" "+str(training)))
            pos, indx, vel_init = episode.initial_position(poses_db, training=training, init_key=init_key)
        else:
            print(("training "+str(training)))
            pos, indx, vel_init = episode.initial_position(poses_db, training=training)
        return pos, indx, vel_init

    def __init__(self, path, sess, writer, gradBuffer, log,settings, net=None):
        super(Environment, self).__init__( path, sess, writer, gradBuffer, log,settings, net=None)
        self.init_methods = [ 8,  2, 4]  # [1,5]#[1, 3, 5, 9]#, 8, 6, 2, 4]#[1, 8, 6, 2, 4, 9,3,5, 10]#[1, 8, 6, 2, 4, 5, 9]#[8, 6, 2, 4] #<--- regular![1, 8, 6, 2, 4] [1]
        self.init_methods_train = [ 8,  2, 4]#9
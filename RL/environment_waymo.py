from carla_environment import CARLAEnvironment
from commonUtils.ReconstructionUtils import reconstruct3D_ply, cityscapes_colours, cityscapes_labels, CreateDefaultDatasetOptions_Waymo
from environment_abstract import AbstractEnvironment
import os, pickle
import time
import scipy
from extract_tensor import  extract_tensor, objects_in_range, extract_tensor, objects_in_range_map

LIMIT_FRAME_NUMBER = 199 # We put 2 here for debug purposes to make it load faster
PEDESTRIAN_SEG_LABEL = 26


class WaymoEnvironment(AbstractEnvironment):
    # Take steps, calculate reward and visualize actions of agent.



    def default_seq_lengths(self, training, evaluate=False):
        # Set the seq length
        frameRate, frameTime = self.settings.getFrameRateAndTime()
        if training and not evaluate:
            seq_len = self.settings.seq_len_train
        else:
            seq_len = self.settings.seq_len_test
        seq_len_pfnn = -1
        if self.settings.pfnn:
            seq_len_pfnn = seq_len * 60 // frameRate
        print(("Environment seq Len {}. seq Len pfnn {}".format(seq_len, seq_len_pfnn)))
        return seq_len, seq_len_pfnn

    # Take steps, calculate reward and visualize actions of agent.
    def work(self, cachePrefix, file_name, agent, poses_db, epoch, saved_files_counter, training=True, road_width=5, conv_once=False,
             time_file=None, act=True, supervised=False):

        # Read some metadata about the dataset and fill a structure containing dataset options specific to way,o
        metadata = None
        with open(os.path.join(file_name, "centering.p"), "rb") as metadataFileHandle:
            metadata = pickle.load(metadataFileHandle)

        # Create the desired dataset options structure
        options = CreateDefaultDatasetOptions_Waymo(metadata)

        # Set the parameters of the env
        min_bbox = metadata["min_bbox"]
        max_bbox = metadata["max_bbox"]

        self.settings.depth = max_bbox[0] - min_bbox[0]
        self.settings.height = max_bbox[2] - min_bbox[2]
        self.settings.width = max_bbox[1] - min_bbox[1]

        # Setup some options for the episode
        file_agent = self.get_file_agent_path(file_name)

        pos_x, pos_y = self.default_camera_pos()
        seq_len, seq_len_pfnn = self.default_seq_lengths(training, evaluate=False)

        # Setup the episode
        #start_setup = time.time()
        episode = self.set_up_episode(cachePrefix, file_name, pos_x, pos_y, training, useCaching=True,
                                      time_file=time_file, seq_len_pfnn=seq_len_pfnn, datasetOptions=options, supervised=False)
        #setup_time = time.time() - start_setup
        #print("Setup time in s {:0.2f}s".format(setup_time))

        # Initialize the agent
        run_episode = True
        test_init, succedeed, validRun = self.default_initialize_agent(training, agent, episode, None)
        if not succedeed:
            return self.counter, saved_files_counter


        # Act and learn for a repeated number of times
        number_of_runs_per_scene=1
        if training:
            number_of_runs_per_scene=1

        if act and (not test_init or run_episode):
            saved_files_counter,initializer_stats = self.default_doActAndGetStats(training, number_of_runs_per_scene, agent, episode, poses_db, file_agent, file_name, saved_files_counter)
        else:
            print("Not valid initialization")
            return self.counter, saved_files_counter, None
        return self.counter, saved_files_counter,initializer_stats





    def evaluate(self, cachePrefix, file_name, agent, fileagent, saved_files_counter, viz=False, folder=""):
        training = False

        # Read some metadata about the dataset and fill a structure containing dataset options specific to way,o
        metadata = None
        with open(os.path.join(file_name, "centering.p"), "rb") as metadataFileHandle:
            metadata = pickle.load(metadataFileHandle)
        options = CreateDefaultDatasetOptions_Waymo(metadata)

        # Setup some options for the episode
        file_agent = self.get_file_agent_path(file_name, eval_path=fileagent)
        pos_x, pos_y = self.default_camera_pos()
        seq_len, seq_len_pfnn = self.default_seq_lengths(training=training, evaluate=True)
        self.settings.seq_len_test= seq_len # Override the test seq length because this is the one actually used, there is no different variable for evaluate in the core code

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
            return self.counter, saved_files_counter,None

        return stats, saved_files_counter,initializer_stats


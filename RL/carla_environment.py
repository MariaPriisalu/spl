print ("In environment ")
from environment_abstract import AbstractEnvironment
from settings import PEDESTRIAN_INITIALIZATION_CODE

from commonUtils.ReconstructionUtils import CreateDefaultDatasetOptions_CarlaRealTime
import numpy as np
import os
import pickle
import time
if True: # Set this to False! or comment it out
    from visualization import make_movie, make_movie_eval

class CARLAEnvironment(AbstractEnvironment):

    def __init__(self, path,  sess, writer, gradBuffer, log, settings, net=None):
        super(CARLAEnvironment, self).__init__(path, sess, writer, gradBuffer, log, settings, net=net)
        self.viz_counter=0
        self.frame_counter=0
        self.scene_count=0
        self.viz_counter_test = 0
        self.scene_count_test = 0
        if settings.supervised_and_rl_car or settings.useRLToyCar  or settings.useHeroCar:
            self.init_methods = [PEDESTRIAN_INITIALIZATION_CODE.on_pavement,
                                 PEDESTRIAN_INITIALIZATION_CODE.randomly]  # [1,5]#[1, 3, 5, 9]#, 8, 6, 2, 4]#[1, 8, 6, 2, 4, 9,3,5, 10]#[1, 8, 6, 2, 4, 5, 9]#[8, 6, 2, 4] #<--- regular![1, 8, 6, 2, 4] [1]
            self.init_methods_train = [PEDESTRIAN_INITIALIZATION_CODE.on_pavement,
                                 PEDESTRIAN_INITIALIZATION_CODE.randomly]   # [1,5]#[1,  3, 5, 9]#, 8, 6, 2, 4]#[1, 8, 6, 2, 4, 9,3,5, 10]#[1, 8, 6, 2, 4, 5, 9]#[9, 8,6,2,4]#[8, 6, 2, 4,] # [1, 8,1, 6,1, 2,1, 4, 1] [1]

        elif settings.continous:
            self.init_methods = [PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                 PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian,
                                 PEDESTRIAN_INITIALIZATION_CODE.in_front_of_car,
                                 PEDESTRIAN_INITIALIZATION_CODE.near_obstacle]#, 8, 6, 2, 4]#[1, 8, 6, 2, 4, 9,3,5, 10]#[1, 8, 6, 2, 4, 5, 9]#[8, 6, 2, 4] #<--- regular![1, 8, 6, 2, 4] [1]
            self.init_methods_train = [PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                       PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian,
                                       PEDESTRIAN_INITIALIZATION_CODE.in_front_of_car,
                                       PEDESTRIAN_INITIALIZATION_CODE.near_obstacle]

        elif settings.goal_dir:
            self.init_methods = [PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                 PEDESTRIAN_INITIALIZATION_CODE.on_pavement,
                                 PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian,
                                 PEDESTRIAN_INITIALIZATION_CODE.by_car,
                                 PEDESTRIAN_INITIALIZATION_CODE.randomly]  # [1, 8, 6, 2, 4, 5, 9]#[8, 6, 2, 4] #<--- regular![1, 8, 6, 2, 4] [1]
            self.init_methods_train = [ PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                        PEDESTRIAN_INITIALIZATION_CODE.on_pavement,
                                        PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian,
                                        PEDESTRIAN_INITIALIZATION_CODE.by_car,
                                        PEDESTRIAN_INITIALIZATION_CODE.randomly,
                                        PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                        PEDESTRIAN_INITIALIZATION_CODE.near_obstacle]  # [1,
        else:
            self.init_methods =[PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                PEDESTRIAN_INITIALIZATION_CODE.on_pavement,
                                PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian,
                                PEDESTRIAN_INITIALIZATION_CODE.by_car,
                                PEDESTRIAN_INITIALIZATION_CODE.randomly]#[1,5]#[1, 3, 5, 9]#, 8, 6, 2, 4]#[1, 8, 6, 2, 4, 9,3,5, 10]#[1, 8, 6, 2, 4, 5, 9]#[8, 6, 2, 4] #<--- regular![1, 8, 6, 2, 4] [1]
            self.init_methods_train =[PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                      PEDESTRIAN_INITIALIZATION_CODE.on_pavement,
                                      PEDESTRIAN_INITIALIZATION_CODE.by_pedestrian,
                                      PEDESTRIAN_INITIALIZATION_CODE.by_car,
                                      PEDESTRIAN_INITIALIZATION_CODE.randomly,
                                      PEDESTRIAN_INITIALIZATION_CODE.on_pedestrian,
                                      PEDESTRIAN_INITIALIZATION_CODE.in_front_of_pedestrian,
                                      PEDESTRIAN_INITIALIZATION_CODE.in_front_of_car,
                                      PEDESTRIAN_INITIALIZATION_CODE.near_obstacle]#[1,5]#[1,  3, 5, 9]#, 8, 6, 2, 4]#[1, 8, 6, 2, 4, 9,3,5, 10]#[1, 8, 6, 2, 4, 5, 9]#[9, 8,6,2,4]#[8, 6, 2, 4,] # [1, 8,1, 6,1, 2,1, 4, 1] [1]


    # Take steps, calculate reward and visualize actions of agent
    def work(self, cachePrefix, filepath, agent, poses_db, epoch, saved_files_counter,car=None, training=True, road_width=5, conv_once=False, time_file=None, act=True, save_stats=True, iterative_training=None, realtime_carla=False):
        # Setup some options for the episode
        print(" File path " + str(filepath))
        if self.settings.new_carla :
            self.settings.depth = 1024
            self.settings.height = 250
            self.settings.width = 1024

        file_agent = self.get_file_agent_path(filepath)
        pos_x, pos_y = self.default_camera_pos()
        seq_len, seq_len_pfnn = self.default_seq_lengths(training=training, evaluate=False)
        if iterative_training!=None:
            print("In work train car:" + str(iterative_training.train_car) + "  train initializer " + str(
                iterative_training.train_initializer))
        #Setup the episode

        start_setup = time.time()
        if self.settings.new_carla or realtime_carla:
            if self.settings.new_carla:
                file_name=filepath[0]
            else:
                file_name = filepath
            datasetOptions=  CreateDefaultDatasetOptions_CarlaRealTime(self.settings)

        else:
            file_name = filepath
            datasetOptions=None
        episode = self.set_up_episode(cachePrefix, file_name, pos_x, pos_y, training, useCaching=self.settings.useCaching, datasetOptions= datasetOptions, time_file=time_file, seq_len_pfnn=seq_len_pfnn, car=car)
        setup_time = time.time() - start_setup
        print(("Setup time in s {:0.2f}s".format(setup_time)))
        # Act and learn
        number_of_runs_per_scene = 1
        if self.settings.learn_init or self.settings.useRLToyCar or self.settings.useHeroCar:
            if len(episode.init_cars):
                saved_files_counter,initializer_stats = self.default_doActAndGetStats(training, number_of_runs_per_scene, agent, episode,
                                                                    poses_db, file_agent, file_name,
                                                                    saved_files_counter, car=car, save_stats=save_stats, iterative_training=iterative_training)
            else:
                print ("Not valid initialization")
                return self.counter, saved_files_counter, None

        else:
            run_episode=True
            test_init, succedeed, validRun = self.default_initialize_agent(training, agent, episode, None)
            if not succedeed:
                return self.counter, saved_files_counter


            if act and (not test_init or run_episode):
                saved_files_counter,initializer_stats = self.default_doActAndGetStats(training, number_of_runs_per_scene, agent, episode, poses_db, file_agent, file_name, saved_files_counter, save_stats=save_stats)
            else:
                print ("Not valid initialization")
                return self.counter, saved_files_counter, None

        return self.counter, saved_files_counter, initializer_stats

    # Old flow, not used anymore - but good as an optimization idea still
    def episode_fast_setup(self, episode, file_name, read_3D, seq_len):
        if self.settings.old:
            if not os.path.exists(os.path.join(file_name, 'reconstruction_old.npy')):
                np.save(os.path.join(file_name, 'reconstruction_old.npy'), episode.reconstruction)
                print((os.path.join(file_name, 'reconstruction_old.npy')))
            if not os.path.exists(os.path.join(file_name, '2D_reconstruction_old.npy')) and len(
                    episode.reconstruction_2D) > 0:
                np.save(os.path.join(file_name, '2D_reconstruction_old.npy'), episode.reconstruction_2D)
                print((os.path.join(file_name, '2D_reconstruction_old.npy')))
            if not os.path.exists(os.path.join(file_name, 'pedestrian_heatmap.npy')):
                np.save(os.path.join(file_name, 'pedestrian_heatmap.npy'), episode.heatmap)
                print((os.path.join(file_name, 'pedestrian_heatmap.npy')))
            if not os.path.exists(os.path.join(file_name, 'valid_pavement.npy')):
                np.save(os.path.join(file_name, 'valid_pavement.npy'), episode.valid_pavement)
                print((os.path.join(file_name, 'valid_pavement.npy')))
            if not os.path.exists(os.path.join(file_name, 'valid_positions.npy')):
                np.save(os.path.join(file_name, 'valid_positions.npy'), episode.valid_positions)
                print((os.path.join(file_name, 'valid_positions.npy')))

            if not read_3D:
                episode.reconstruction = np.load(os.path.join(file_name, 'reconstruction.npy'))
                tmp = np.load(os.path.join(file_name, '2D_reconstruction.npy'))
                if len(tmp) > 0:
                    episode.reconstruction_2D = np.load(os.path.join(file_name, '2D_reconstruction.npy'))
                    print(("2D shape : " + str(episode.reconstruction_2D.shape)))

                episode.heatmap = np.load(os.path.join(file_name, 'pedestrian_heatmap.npy'))
                episode.valid_pavement = np.load(os.path.join(file_name, 'valid_pavement.npy'))
                episode.valid_positions = np.load(os.path.join(file_name, 'valid_positions.npy'))
                # if episode.people_predicted[0].shape[0]==32:
                #     holder=np.load(os.path.join(file_name, 'people_predicted.npy'))
                # else:
                #     holder = np.load(os.path.join(file_name, 'people_predicted_2d.npy'))
                # print "people predicted shape before : " + str(episode.people_predicted[0].shape)
                # episode.people_predicted = []
                # for pos in range(holder.shape[0]):
                #     episode.people_predicted.append(holder[pos,:,:,:])
                # print "people predicted shape : "+str(episode.people_predicted[0].shape)
                # #episode.car_predicted = np.load(os.path.join(file_name, 'cars_predicted.npy'))
                # if episode.people_predicted[0].shape[0]==32:
                #     holder_c=np.load(os.path.join(file_name, 'cars_predicted.npy'))
                # else:
                #     holder_c = np.load(os.path.join(file_name, 'cars_predicted_2d.npy'))
                # episode.car_predicted = []
                # for pos in range(holder_c.shape[0]):
                #     episode.car_predicted.append(holder_c[pos, :, :, :])
                episode.initialization_set_up_fast(seq_len, False)
        else:
            if not os.path.exists(os.path.join(file_name, 'reconstruction.npy')):
                np.save(os.path.join(file_name, 'reconstruction.npy'), episode.reconstruction)
                print((os.path.join(file_name, 'reconstruction.npy')))
            if not os.path.exists(os.path.join(file_name, '2D_reconstruction.npy')) and len(
                    episode.reconstruction_2D) > 0:
                np.save(os.path.join(file_name, '2D_reconstruction.npy'), episode.reconstruction_2D)
                print((os.path.join(file_name, '2D_reconstruction.npy')))
            if not os.path.exists(os.path.join(file_name, 'pedestrian_heatmap.npy')):
                np.save(os.path.join(file_name, 'pedestrian_heatmap.npy'), episode.heatmap)
                print((os.path.join(file_name, 'pedestrian_heatmap.npy')))
            if not os.path.exists(os.path.join(file_name, 'valid_pavement.npy')):
                np.save(os.path.join(file_name, 'valid_pavement.npy'), episode.valid_pavement)
                print((os.path.join(file_name, 'valid_pavement.npy')))
            if not os.path.exists(os.path.join(file_name, 'valid_positions.npy')):
                np.save(os.path.join(file_name, 'valid_positions.npy'), episode.valid_positions)
                print((os.path.join(file_name, 'valid_positions.npy')))
            if self.settings.temporal:
                if not os.path.exists(os.path.join(file_name, 'people_predicted.npy')) and \
                                episode.people_predicted[0].shape[0] == 32 and self.settings.temporal:
                    np.save(os.path.join(file_name, 'people_predicted.npy'), episode.people_predicted)
                    print((os.path.join(file_name, 'people_predicted.npy')))
                if not os.path.exists(os.path.join(file_name, 'people_predicted_2d.npy')) and \
                                episode.people_predicted[0].shape[0] != 32 and self.settings.temporal:
                    np.save(os.path.join(file_name, 'people_predicted_2d.npy'), episode.people_predicted)
                    print((os.path.join(file_name, 'people_predicted_2d.npy')))

                if not os.path.exists(os.path.join(file_name, 'cars_predicted.npy')) and \
                                episode.people_predicted[0].shape[0] == 32 and self.settings.temporal:
                    np.save(os.path.join(file_name, 'cars_predicted.npy'), episode.cars_predicted)
                    print((os.path.join(file_name, 'cars_predicted.npy')))
                if not os.path.exists(os.path.join(file_name, 'cars_predicted_2d.npy')) and \
                                episode.people_predicted[0].shape[0] != 32 and self.settings.temporal:
                    np.save(os.path.join(file_name, 'cars_predicted_2d.npy'), episode.cars_predicted)
                    print((os.path.join(file_name, 'cars_predicted_2d.npy')))
            if not read_3D:
                episode.reconstruction = np.load(os.path.join(file_name, 'reconstruction.npy'))
                tmp = np.load(os.path.join(file_name, '2D_reconstruction.npy'))
                if len(tmp) > 0:
                    episode.reconstruction_2D = np.load(os.path.join(file_name, '2D_reconstruction.npy'))
                    print(("2D shape : " + str(episode.reconstruction_2D.shape)))

                episode.heatmap = np.load(os.path.join(file_name, 'pedestrian_heatmap.npy'))
                episode.valid_pavement = np.load(os.path.join(file_name, 'valid_pavement.npy'))
                episode.valid_positions = np.load(os.path.join(file_name, 'valid_positions.npy'))

                if self.settings.temporal:
                    if episode.people_predicted[0].shape[0] == 32:
                        holder = np.load(os.path.join(file_name, 'people_predicted.npy'))
                    else:
                        holder = np.load(os.path.join(file_name, 'people_predicted_2d.npy'))
                    print(("people predicted shape before : " + str(episode.people_predicted[0].shape)))
                    episode.people_predicted = []
                    for pos in range(holder.shape[0]):
                        episode.people_predicted.append(holder[pos, :, :, :])
                    print(("people predicted shape : " + str(episode.people_predicted[0].shape)))
                    # episode.car_predicted = np.load(os.path.join(file_name, 'cars_predicted.npy'))
                    if episode.people_predicted[0].shape[0] == 32:
                        holder_c = np.load(os.path.join(file_name, 'cars_predicted.npy'))
                    else:
                        holder_c = np.load(os.path.join(file_name, 'cars_predicted_2d.npy'))
                    episode.cars_predicted = []
                    for pos in range(holder_c.shape[0]):
                        episode.cars_predicted.append(holder_c[pos, :, :, :])
                episode.initialization_set_up_fast(seq_len, False)

    def evaluate(self, cachePrefix, filepath, agent, file_path, saved_files_counter, viz=False, folder="",realtime_carla=False, car=None):
        training = False
        print(" Car in env.evaluate() " + str(car))
        # Setup some options for the episode
        file_agent = self.get_file_agent_path(filepath, eval_path=file_path)
        pos_x, pos_y = self.default_camera_pos()
        seq_len, seq_len_pfnn = self.default_seq_lengths(training=training, evaluate=True)
        self.settings.seq_len_test= seq_len # Override the test seq length because this is the one actually used, there is no different variable for evaluate in the core code


        # Setup the episode
        start_setup = time.time()
        if self.settings.new_carla or realtime_carla:
            if self.settings.new_carla:
                file_name = filepath[0]
            else:
                file_name = filepath
            datasetOptions = CreateDefaultDatasetOptions_CarlaRealTime(self.settings)
            print ("New dataset otions ")

        else:
            file_name = filepath
            datasetOptions = None


        episode = self.set_up_episode(cachePrefix, file_name, pos_x, pos_y, training,evaluate=True, useCaching=self.settings.useCaching, seq_len_pfnn=seq_len_pfnn,datasetOptions=datasetOptions,car=car)
        setup_time = time.time() - start_setup
        print(("Setup time in s {:0.2f}s".format(setup_time)))
        # Act and get stats
        stats = []
        initializer_stats=[]
        number_of_runs_per_scene = 1
        use_car = self.settings.learn_init or self.settings.useRLToyCar or self.settings.useHeroCar
        if use_car:
            if  len(episode.init_cars):
                saved_files_counter, initializer_stats = self.default_doActAndGetStats(training, number_of_runs_per_scene,
                                                                                       agent, episode, None, file_agent,
                                                                                       file_name, saved_files_counter,
                                                                                       car=car, outStatsGather=stats,
                                                                                       evaluate=True, save_stats=True, viz=viz)
            else:
                print ("No valid cars")
                return stats, saved_files_counter,initializer_stats
        else:
            run_episode=True
            test_init, succedeed, validRun = self.default_initialize_agent(training, agent, episode, None)
            if not succedeed:
                return stats, saved_files_counter,initializer_stats
            run_episode |= validRun



            if (not test_init or run_episode):
                use_car=self.settings.learn_init or self.settings.useRLToyCar or self.settings.useHeroCar
                if (use_car and len(episode.init_cars)) or not use_car:
                    saved_files_counter,initializer_stats = self.default_doActAndGetStats(training,number_of_runs_per_scene, agent, episode, None, file_agent, file_name, saved_files_counter,car=car, outStatsGather=stats, evaluate=True,save_stats=True, viz=viz)
            else:
                print ("Not valid initialization")
                return stats, saved_files_counter,initializer_stats

        return stats, saved_files_counter,initializer_stats



import os



# import matplotlib as mpl
# print(('current backend: ', mpl.get_backend()))
# backend = 'agg' #'cairo'  # 'agg'
# mpl.use(backend, force=True)
# print(('current backend: ', mpl.get_backend()))

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
from settings import RANDOM_SEED

tf.compat.v1.set_random_seed(RANDOM_SEED)
from tensorflow.python import pywrap_tensorflow


from agents_goal import GoalAgent, GoalCVAgent
from environment_test import TestEnvironment
from carla_environment import CARLAEnvironment
from environment import Environment
from agent_net import NetAgent, AgentNetPFNN, ContinousNetAgent, ContinousNetPFNNAgent
from agents_dummy import RandomAgent, PedestrianAgent
from agent_car import CarAgent
from net_sem_2d import Seg_2d, Seg_2d_min_cont, Seg_2d_min_softmax, Seg_2d_min_cont_angular,  ContinousMemNet_2D
from net_sem_3D import  Seg_3d_min, Seg_3d_RGB, Seg_3d_no_sem, Seg_3d_min_pop, Seg_3d_min_still_actions, Seg_3d_no_vis
from net_no_sem_2D import NoSem_2d
from net_simple_car import SimpleCarNet
from mini_net import Net_no_viz
from goal_net import GoalNet,GoalGaussianNet
from environment_waymo import WaymoEnvironment
from initializer_net import InitializerNet, InitializerGaussianNet
from semantic_net_architecture import InitializerArchNet
from environment_test_special_cases import CarEnvironment, PeopleEnvironment
import glob
import os
import numpy as np
import scipy.io as sio
from settings import run_settings, USE_LOCAL_PATHS, WAYMO_CACHE_PREFIX_EVALUATE, CARLA_CACHE_PREFIX_EVALUATE, \
    WAYMO_CACHE_PREFIX_EVALUATE_SUPERVISED, CARLA_CACHE_PREFIX_EVALUATE_SUPERVISED, WAYMO_CACHE_PREFIX_TRAIN, \
    CARLA_CACHE_PREFIX_TRAIN, WAYMO_CACHE_PREFIX_TRAIN_SUPERVISED, CARLA_CACHE_PREFIX_TRAIN_SUPERVISED, \
    CARLA_CACHE_PREFIX_TEST, WAYMO_CACHE_PREFIX_TEST_SUPERVISED, CARLA_CACHE_PREFIX_TEST_SUPERVISED, \
    WAYMO_CACHE_PREFIX_TEST, CARLA_CACHE_PREFIX_TEST_TOY, CARLA_CACHE_PREFIX_EVALUATE_TOY, CARLA_CACHE_PREFIX_TRAIN_TOY,\
    CARLA_CACHE_PREFIX_TEST_NEW, CARLA_CACHE_PREFIX_EVALUATE_NEW, CARLA_CACHE_PREFIX_TRAIN_NEW, CAR_REWARD_INDX, \
    CARLA_CACHE_PREFIX_EVALUATE_REALTIME, CARLA_CACHE_PREFIX_TRAIN_REALTIME, CARLA_CACHE_PREFIX_TEST_REALTIME


def get_carla_prefix_eval(supervised, useToyEnv, new_carla, realtime_carla):
    if realtime_carla:
        return CARLA_CACHE_PREFIX_EVALUATE_REALTIME
    if new_carla:
        return CARLA_CACHE_PREFIX_EVALUATE_NEW
    if useToyEnv:
        return CARLA_CACHE_PREFIX_EVALUATE_TOY
    return (CARLA_CACHE_PREFIX_EVALUATE if not supervised else CARLA_CACHE_PREFIX_EVALUATE_SUPERVISED)


def get_carla_prefix_train(supervised, useToyEnv, new_carla, realtime_carla):
    if realtime_carla:
        return CARLA_CACHE_PREFIX_TRAIN_REALTIME
    if new_carla:
        return CARLA_CACHE_PREFIX_TRAIN_NEW
    if useToyEnv:
        return CARLA_CACHE_PREFIX_TRAIN_TOY
    return (CARLA_CACHE_PREFIX_TRAIN if not supervised else CARLA_CACHE_PREFIX_TRAIN_SUPERVISED)


def get_carla_prefix_test( supervised, useToyEnv, new_carla, realtime_carla):
    if realtime_carla:
        return CARLA_CACHE_PREFIX_TRAIN_REALTIME
    if new_carla:
        return CARLA_CACHE_PREFIX_TEST_NEW
    if useToyEnv:
        return CARLA_CACHE_PREFIX_TEST_TOY
    return (CARLA_CACHE_PREFIX_TEST if not supervised else CARLA_CACHE_PREFIX_TEST_SUPERVISED)


def get_waymo_prefix_eval(supervised):
    return (WAYMO_CACHE_PREFIX_EVALUATE if not supervised else WAYMO_CACHE_PREFIX_EVALUATE_SUPERVISED)


def get_waymo_prefix_train(supervised):
    return (WAYMO_CACHE_PREFIX_TRAIN if not supervised else WAYMO_CACHE_PREFIX_TRAIN_SUPERVISED)


def get_waymo_prefix_test(supervised):
    return (WAYMO_CACHE_PREFIX_TEST if not supervised else WAYMO_CACHE_PREFIX_TEST_SUPERVISED)
from dotmap import DotMap
import time

import logging

from supervised_environment import SupervisedWaymoEnvironment, SupervisedCARLAEnvironment


##
# Main script- train an RL agent on real or toy case.
#

import sys
import traceback

class TracePrints(object):
  def __init__(self):
    self.stdout = sys.stdout
  def write(self, s):
    self.stdout.write("Writing %r\n" % s)
    traceback.print_stack(file=self.stdout)




class RL(object):
    def __init__(self):
        # Load H3.6M database of valid poses.
        self.poses_path = "Datasets/cityscapes/poses/" if USE_LOCAL_PATHS == 0 else "localUserData/Datasets/cityscapes/poses/"
        # Log path tensorboard
        self.log_path = "Datasets/cityscapes/logs/" if USE_LOCAL_PATHS == 0 else "localUserData/Datasets/cityscapes/logs/"
        # Log path
        self.save_model_path="Models/rl_model/" if USE_LOCAL_PATHS == 0 else "localUserData/Models/rl_model/"

        self.test_counter=-1
        self.car_test_counter=0
        self.real_counter=0
        self.save_counter=0

        print("Poses path exists? "+str(os.path.exists(self.poses_path)))


        if not os.path.exists(self.poses_path):
            self.poses_path = "Documents/Datasets/Human3.6/Release-v1.1/" if USE_LOCAL_PATHS == 0 else "localUserData/Datasets/Human3.6/Release-v1.1/"

            self.log_path = "Results/agent/" if USE_LOCAL_PATHS == 0 else "localUserData/Results/agent/"
            self.save_model_path="Results/agent/" if USE_LOCAL_PATHS == 0 else "localUserData/Results/agent/"

        if os.path.exists(self.poses_path):
            self.poses_db = sio.loadmat(self.poses_path + "poses_hips.mat")
            self.poses_db = np.delete(self.poses_db["poses"], [24, 25, 26], 1)
        else:
            self.poses_db=None
            print("No Poses db!")

    def eraseCachedEpisodes(self, folderPath):
        import os, shutil
        for filename in os.listdir(folderPath):
            file_path = os.path.join(folderPath, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(('Failed to delete %s. Reason: %s' % (file_path, e)))

    # Tensorflow session.
    def run(self,  settings, time_file=None, supervised=False):
        #print("Supervised? "+supervised)
        # Check if episodes cached data should be deleted
        if settings.deleteCacheAtEachRun:
            self.eraseCachedEpisodes(settings.target_episodesCache_Path)
            print("I have deleted the cache folder !")

        counter, filecounter, images_path, init, net, saver,  tmp_net, init_net,goal_net,car_net = self.pre_run_setup(settings)
        tf.compat.v1.set_random_seed(settings.random_seed_tf)
        #from tensorflow.python import debug as tf_debug
        with tf.Session(graph=tf.get_default_graph()) as sess:
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            tf.compat.v1.set_random_seed(settings.random_seed_tf)
            sess.run(init)



            grad_buffer, grad_buffer_init,grad_buffer_goal,grad_buffer_car = self.create_gradient_buffer(sess)

            print((settings.load_weights))
            if "model.ckpt" in settings.load_weights:
                self.load_net_weights(saver, sess, settings, latest=False)#, exact=True)
            else:
                self.load_net_weights(saver, sess, settings, latest=True)
            self.create_setting_folder(settings)
            # variables_policy = []
            # variables_init = []
            # variables_goal = []
            # variables_car = []
            # for v in tf.compat.v1.trainable_variables():
            #     if "goal" in v.name:
            #         variables_goal.append(v)
            #     elif "init" in v.name:
            #         variables_init.append(v)
            #     elif "car" in v.name:
            #         variables_car.append(v)
            #     else:
            #         variables_policy.append(v)
            # [car_vars] = sess.run([variables_car])
            # for var in car_vars:
            #     print("Car var " + str(var))
            sess.graph.finalize()
            agent = self.get_agent(grad_buffer, net, grad_buffer_init, init_net,grad_buffer_goal,goal_net, settings)
            if settings.useRLToyCar:
                car=CarAgent(settings, car_net, grad_buffer_car)
            else:
                car=None
            # if settings.refine_follow:
            #     agent = SupervisedAgent(net, grad_buffer)
            # if settings.confined_actions:
            #     agent=DirectionalAgent( settings, net, grad_buffer)
            if not settings.cv_agent:
                net.set_session(sess)
            if settings.learn_init:
                init_net.set_session(sess)
            if settings.separate_goal_net:
                goal_net.set_session(sess)
            if settings.useRLToyCar:
                car_net.set_session(sess)

            writer=None
            log_file=None

            self.start_run(agent, car, counter, filecounter, grad_buffer, images_path, log_file, saver, sess, settings,
                           supervised, tmp_net, writer)

    def start_run(self, agent, car, counter, filecounter, grad_buffer, images_path, log_file, saver, sess, settings,
                  supervised, tmp_net, writer):
        saved_files_counter = 0
        print(("Before run seq len " + str(settings.seq_len_train) + " " + str(settings.seq_len_train_final)))
        if settings.carla:

            counter = self.train_on_carla(agent, counter, filecounter, grad_buffer, images_path,
                                          log_file, saved_files_counter, saver, sess, settings, writer, supervised, car)

        elif settings.waymo:

            counter = self.train_on_waymo(agent, counter, filecounter, grad_buffer, images_path,
                                          log_file, saved_files_counter, saver, sess, settings, writer, supervised)

        elif settings.toy_case:

            counter = self.train_on_test_set(agent, counter, grad_buffer, images_path, log_file, saver,
                                             sess, settings, tmp_net, writer)
        elif settings.temp_case:

            counter = self.train_on_temp_case(agent, counter, filecounter, grad_buffer, images_path,
                                              log_file, saved_files_counter, saver, sess, settings, writer)
        else:
            counter = self.train_on_cs(agent, counter, filecounter, grad_buffer, images_path,
                                       log_file, saved_files_counter, saver, sess, settings, writer)
        saver.save(sess,
                   self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt",
                   global_step=counter)

    def train_on_temp_case(self, agent,counter, filecounter, grad_buffer, images_path,
                                              log_file, saved_files_counter, saver, sess, settings, writer):
        if settings.old:
            settings.run_2D = False
        epoch, filename_list, pos = self.get_carla_files(settings)
        train_set, val_set, test_set = self.get_train_test_fileIndices(filename_list)
        num_epochs = 20  # 10  # 23
        if settings.overfit:
            train_set = [0, 4, 8]
            val_set = [100, 104]  # [100]
            num_epochs = 200

        filepath = filename_list[0]

        # print "Hej "
        env = self.get_environments_CARLA(grad_buffer, images_path, log_file,
                                          sess, settings, writer)

        num_epochs = 500  # 10  # 23

        for epoch in range(0, num_epochs):
            # print "Epoch "+str(epoch)
            self.adapt_variables(epoch, settings)

            # car_counter, saved_files_counter = env_car.work(filepath, agent, self.poses_db,epoch,saved_files_counter, training=True)
            # people_counter, saved_files_counter = env_people.work(filepath, agent,self.poses_db, epoch,saved_files_counter,training=True)
            if epoch % 100 == 0:
                # car_counter, saved_files_counter = env_car.work(filepath, agent, self.poses_db, epoch,
                #                                                saved_files_counter, training=False)
                # people_counter, saved_files_counter = env_people.work(filepath, agent, self.poses_db,
                #                                                      epoch,
                #                                                      saved_files_counter,
                #                                                      training=False)
                self.test_counter = self.test_counter + 1

                saver.save(sess,
                           self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt",
                           global_step=epoch)
                print(("Save model : " + self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt-" + str(
                    epoch)))
                filecounter += 1
                print(("filecounter " + str(filecounter)))

            self.test_counter = self.test_counter + 1
            saver.save(sess, self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt",
                       global_step=epoch)

            return counter





    def train_on_test_set(self, agent, counter, grad_buffer, images_path, log_file, saver, sess, settings, tmp_net, writer):
        env = TestEnvironment(images_path, sess, writer, grad_buffer, log_file, settings, net=tmp_net)

        counter = env.work("", agent, self.poses_db)
        saver.save(sess, self.save_model_path + settings.name_movie, global_step=counter)
        return counter

    def train_on_carla(self, agent,  counter, filecounter, grad_buffer, images_path, log_file,
                       saved_files_counter, saver, sess, settings, writer, supervised, car):
        print("CARLA ")
        if settings.old:
            settings.run_2D=False
        env, env_car, env_people = self.get_environments_CARLA( grad_buffer, images_path, log_file,
                                                               sess, settings, writer)
        epoch, filename_list, pos = self.get_carla_files(settings, False)

        train_set, val_set, test_set = self.get_train_test_fileIndices(filename_list, carla=True, new_carla=settings.new_carla, realtime_carla=settings.realtime_carla_only)
        print (" train set ")
        print(train_set)
        num_epochs = 20 #10#10  # 23

        num_epochs, train_set, val_set = self.adjust_training_set(num_epochs, settings, train_set, val_set)
        if settings.realtime_carla:
            epoch_rt, filename_list_realtime, pos_rt = self.get_carla_files(settings, settings.realtime_carla)


        if settings.overfit:
            save_stats=False
        else:
            save_stats=True

        car_epoch=0
        init_epoch=0
        train_car=settings.useHeroCar
        train_initializer=settings.learn_init or settings.learn_goal
        success_rate_car=0
        collision_rate_initializer=0
        if len(settings.train_car_and_initialize)>0 and (settings.learn_goal or  settings.learn_init or settings.useRLToyCar) and settings.useHeroCar:
            iterative_training=DotMap()
        else:
            iterative_training=None

        print (" Initially: iterative training "+str(iterative_training)+" train_car "+str(train_car)+" train initializer "+str(train_initializer))
        for epoch in range(0, num_epochs):
            print(("$$$$$$$$$$ Epoch "+str(epoch)))

            if settings.overfit and num_epochs%settings.save_frequency==0:
                save_stats=True
            if  iterative_training!=None:
                print ("Switch training ")
                train_car, train_initializer, car_epoch, init_epoch=self.switch_training(epoch, car_epoch, init_epoch, settings, train_car, train_initializer, success_rate_car, collision_rate_initializer)
                iterative_training.train_car=train_car
                iterative_training.train_initializer = train_initializer
                print("Done switch training train car:"+str(iterative_training.train_car)+"  train initializer "+str(iterative_training.train_initializer))


            for posIdx, pos in enumerate(train_set):
                print(("-------- Epoch {0}, train environment index {1}/{2}".format(epoch, posIdx, len(train_set))))
                self.adapt_variables(epoch, settings, success_rate_car)
                print (" pos "+str(pos)+" pos indx "+str(posIdx) )
                filepath = filename_list[pos]


                try:

                    counter, saved_files_counter,initializer_stats = env.work(get_carla_prefix_train(supervised,  settings.useRLToyCar or settings.useHeroCar , settings.new_carla, settings.realtime_carla_only), filepath, agent, self.poses_db, epoch, saved_files_counter, car=car,training=True, save_stats=save_stats, iterative_training=iterative_training, realtime_carla=settings.realtime_carla_only)

                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    logging.exception('Fatal error in main loop ' + settings.timestamp)
                    print("Unexpected error:", sys.exc_info()[0])
                    traceback.print_exception(*sys.exc_info())
                    if settings.stop_for_errors:
                        raise # Always raise exception otherwise it will be masked and apperantly it would continue...

                if settings.realtime_carla  and filecounter > settings.realtime_freq * (self.real_counter + 1):
                    for pos_rt in range(4):
                        filepath = filename_list_realtime[pos_rt]
                        print (" Filepath "+str(filepath))
                        try:
                            counter, saved_files_counter, initializer_stats = env.work(
                                get_carla_prefix_train(supervised, settings.useRLToyCar or settings.useHeroCar, settings.new_carla,
                                                            settings.realtime_carla), filepath, agent, self.poses_db, epoch,
                                saved_files_counter, car=car, training=True, save_stats=save_stats,
                                iterative_training=iterative_training, realtime_carla=True)

                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except:
                            logging.exception('Fatal error in main loop ' + settings.timestamp)
                            print("Unexpected error:", sys.exc_info()[0])
                            traceback.print_exception(*sys.exc_info())
                            if settings.stop_for_errors:
                                raise  # Always raise exception otherwise it will be masked and apperantly it would continue...
                    self.real_counter = self.real_counter + 1
                # print ("Before running on car env "+str(env_car))
                # print (filecounter > settings.car_test_frq * self.car_test_counter)
                # print ( not settings.overfit or settings.learn_init)
                # print ( not settings.pfnn)
                if (env_car and filecounter > settings.car_test_frq * self.car_test_counter and not (settings.overfit or settings.learn_init or settings.useHeroCar or settings.useRLToyCar)): #and not settings.pfnn:
                    print ("Car env ")
                    car_counter, saved_files_counter = env_car.work("test_car", filepath, agent, self.poses_db, epoch,
                                                                    saved_files_counter, road_width=pos,car=car,
                                                                    training=True, save_stats=save_stats)
                    people_counter, saved_files_counter = env_people.work("test_pedestrian", filepath, agent, self.poses_db, epoch,
                                                                          saved_files_counter, car=car,training=True, save_stats=save_stats)
                    self.car_test_counter = self.car_test_counter + 1

                #print("Filecounter "+str(filecounter)+" test freq "+str(settings.test_freq)+" test_counter "+str(self.test_counter + 1) +" testing? "+str(filecounter > settings.test_freq * (self.test_counter + 1)))
                if filecounter > settings.test_freq * (self.test_counter + 1): #and save_stats:
                    success_rates_car = []
                    collision_rates_initializer = []
                    #print("In testing: val set "+str(val_set))
                    for test_pos in val_set:
                        if test_pos % settings.VAL_JUMP_STEP_CARLA == 0:

                            filepath = filename_list[test_pos]
                            #print("Filepath at " + str(test_pos) + " file " + str(filepath))
                            try:
                                # print("Carla prefix "+str(get_carla_prefix_test(supervised,  settings.useRLToyCar or settings.useHeroCar, settings.new_carla, settings.realtime_carla_only)))
                                _, saved_files_counter,initializer_stats = env.work(get_carla_prefix_test(supervised,  settings.useRLToyCar or settings.useHeroCar, settings.new_carla, settings.realtime_carla_only), filepath, agent, self.poses_db, epoch,
                                                                  saved_files_counter,car=car, training=False, realtime_carla=settings.realtime_carla_only)
                                if initializer_stats:
                                    print (" From initializers stats get success rate "+str(initializer_stats.success_rate_car)+" collision rate "+str(initializer_stats.collision_rate_initializer))
                                    success_rates_car.append(initializer_stats.success_rate_car)
                                    collision_rates_initializer.append(initializer_stats.collision_rate_initializer)
                            except (KeyboardInterrupt, SystemExit):
                                raise
                            except:
                                logging.exception('Fatal error in main loop ' + settings.timestamp)
                                print("Exception")

                                if settings.stop_for_errors:
                                    raise
                    if settings.realtime_carla:

                        for pos_rt in range(4,6):
                            print(" Filepath " + str(filepath))
                            filepath = filename_list_realtime[pos_rt]

                            try:

                                _, saved_files_counter, initializer_stats = env.work(
                                    get_carla_prefix_test(supervised, settings.useRLToyCar or settings.useHeroCar, settings.new_carla,
                                                               settings.realtime_carla), filepath, agent, self.poses_db,
                                    epoch,
                                    saved_files_counter, car=car, training=False, realtime_carla=True)

                            except (KeyboardInterrupt, SystemExit):
                                raise
                            except:
                                logging.exception('Fatal error in main loop ' + settings.timestamp)
                                print("Unexpected error:", sys.exc_info()[0])
                                traceback.print_exception(*sys.exc_info())
                                if settings.stop_for_errors:
                                    raise  # Always raise exception otherwise it will be masked and apperantly it would continue...
                    success_rate_car=np.mean(success_rates_car)
                    collision_rate_initializer=np.mean(collision_rates_initializer)

                    print(" Update success rate "+str(success_rate_car)+" Update collision rate "+str(collision_rate_initializer))


                    if np.isfinite(settings.car_test_frq) and not (settings.overfit or settings.learn_init or settings.useHeroCar or settings.useRLToyCar):
                        car_counter, saved_files_counter = env_car.work("test_car",filepath, agent, self.poses_db, epoch,
                                                                        saved_files_counter,car=car, training=False)
                        people_counter, saved_files_counter = env_people.work("test_pedestrian", filepath, agent, self.poses_db, epoch,
                                                                              saved_files_counter,car=car, training=False)


                    self.test_counter = self.test_counter + 1
                    if not settings.overfit or epoch%100==0:
                        saver.save(sess,
                                   self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt",
                                   global_step=epoch)
                    print(("Save model : " + self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt-" + str(
                        epoch)))
                    self.save_counter = self.save_counter + 1
                filecounter += 1
                print(("filecounter " + str(filecounter)))

        print ("Final testing ")
        for test_pos in val_set:
            if test_pos % settings.VAL_JUMP_STEP_CARLA == 0:
                filepath = filename_list[test_pos]
                try:
                    _, saved_files_counter,initializer_stats = env.work(get_carla_prefix_test(supervised,  settings.useRLToyCar or settings.useHeroCar, settings.new_carla,settings.realtime_carla_only), filepath, agent, self.poses_db, epoch,
                                                      saved_files_counter,car=car, training=False)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    logging.exception('Fatal error in main loop ' + settings.timestamp)
                    print("Exception")
                    if settings.stop_for_errors:
                        raise

        saver.save(sess, self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt",
                   global_step=epoch)

        self.test_counter = self.test_counter + 1

        if not settings.overfit:
            self.evaluate_method_carla(agent,  sess, settings, car=car)
        return counter

    def adjust_training_set(self, num_epochs, settings, train_set, val_set):

        # if settings.fastTrainEvalDebug is True:
        #     num_epochs = 2
        #     if settings.new_carla:
        #         return num_epochs, train_set, val_set
        #     train_set = [0, 1, 2]
        #     val_set = [100, 101]
        if settings.overfit:
            print("Overfit! ")
            num_epochs = 300
            if settings.new_carla:
                return num_epochs, train_set, val_set
            if settings.realtime_carla:
                train_set = [ 1, 0]
                val_set = []  # [100]
                return num_epochs, train_set, val_set
            train_set = [0, 1, 2]
            val_set = [101]  # [100]
            if settings.learn_init or settings.useHeroCar or settings.useRLToyCar:
                train_set = [0, 1, 2, 19]
                val_set = [104, 101]  # [100]
                num_epochs = 300#300
                if len(settings.train_car_and_initialize)>0:
                    num_epochs = 1000  # 300
                # train_set = [ 0, 4,8]
                # test_set =[100,104]# [100]
                # num_epochs = 200
        return num_epochs, train_set, val_set

    def train_on_waymo(self, agent,  counter, filecounter, grad_buffer, images_path, log_file,
                       saved_files_counter, saver, sess, settings, writer, supervised=False):

        env, env_car, env_people = self.get_environments_Waymo(grad_buffer, images_path, log_file,
                                          sess, settings, writer)
        epoch, filename_list, pos = self.get_waymo_files(settings, evaluate=False)

        print(filename_list)

        train_set, val_set, test_set = self.get_train_test_fileIndices(filename_list, waymo=True)

        print(train_set)
        num_epochs =10  # 23
        if settings.overfit:
            print("Overfit! ")
            train_set = list(range(5))
            val_set = [5, 6]  # [100]

            num_epochs = 300
            # train_set = [ 0, 4,8]
            # test_set =[100,104]# [100]
            # num_epochs = 200

        for epoch in range(0, num_epochs):
            print(("$$$$$$$$$$ Epoch " + str(epoch)))


            for posIdx, pos in enumerate(train_set):
                print(("-------- Epoch {0}, train environment index {1}/{2}".format(epoch, posIdx, len(train_set))))

                self.adapt_variables(epoch, settings)
                filepath = filename_list[pos]

                try:
                    counter, saved_files_counter = env.work(get_waymo_prefix_train( supervised), filepath, agent, self.poses_db, epoch,
                                                            saved_files_counter,
                                                            training=True, supervised=supervised)


                    if env_car and filecounter > settings.car_test_frq * self.car_test_counter :#and not settings.pfnn:
                        car_counter, saved_files_counter = env_car.work("test_car",filepath, agent, self.poses_db, epoch,
                                                                        saved_files_counter, road_width=pos,
                                                                        training=True)
                        people_counter, saved_files_counter = env_people.work("test_pedestrian",filepath, agent, self.poses_db, epoch,
                                                                              saved_files_counter, training=True)
                        self.car_test_counter = self.car_test_counter + 1
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    logging.exception('Fatal error in main loop ' + settings.timestamp)
                    print("Exception")
                    if settings.stop_for_errors:
                        raise

                if filecounter > settings.test_freq * (self.test_counter + 1):
                    for test_pos in val_set:
                        if test_pos % settings.VAL_JUMP_STEP_WAYMO == 0:
                            filepath = filename_list[test_pos]
                            try:
                                _, saved_files_counter = env.work(get_waymo_prefix_test( supervised), filepath, agent, self.poses_db, epoch,
                                                                  saved_files_counter, training=False)
                            except (KeyboardInterrupt, SystemExit):
                                raise
                            except:
                                logging.exception('Fatal error in main loop ' + settings.timestamp)
                                print("Exception")
                                if settings.stop_for_errors:
                                    raise

                    print("Testing ")

                    self.test_counter = self.test_counter + 1
                    try:
                        if  env_car :#and not settings.pfnn:
                            car_counter, saved_files_counter = env_car.work("test_car",filepath, agent, self.poses_db, epoch,
                                                                            saved_files_counter, training=False)
                            people_counter, saved_files_counter = env_people.work("test_pedestrian",filepath, agent, self.poses_db, epoch,
                                                                         saved_files_counter, training=False)

                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except:
                        logging.exception('Fatal error in main loop ' + settings.timestamp)
                        print( "Exception")
                        if settings.stop_for_errors:
                            raise
                    if not settings.overfit:
                        saver.save(sess,
                                   self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt",
                                   global_step=epoch)
                    print(("Save model : " + self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt-" + str(
                        epoch)))
                    self.save_counter = self.save_counter + 1
                filecounter += 1
                print(("filecounter " + str(filecounter)))


        for test_pos in val_set:
            if test_pos % settings.VAL_JUMP_STEP_WAYMO == 0:
                filepath = filename_list[test_pos]
                try:
                    _, saved_files_counter = env.work(get_waymo_prefix_test( supervised), filepath, agent, self.poses_db, epoch,
                                                      saved_files_counter, training=False)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    logging.exception('Fatal error in main loop ' + settings.timestamp)
                    print("Exception")
                    if settings.stop_for_errors:
                        raise
        saver.save(sess, self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt",
                   global_step=epoch)
        self.test_counter = self.test_counter + 1

        self.evaluate_method_waymo(agent,  sess, settings, supervised)
        return counter

    def train_on_cs(self, agent,  counter, filecounter, grad_buffer, images_path, log_file,
                       saved_files_counter, saver, sess, settings, writer):
        env, env_car, env_people = self.extract_enviornments_cityscapes(grad_buffer, images_path,
                                                                        log_file, sess, settings, writer)
        HACK_ONLY_VIS_STATS = False
        if HACK_ONLY_VIS_STATS:
            env.visualizeFromScript()
            exit(0)
        nbr_files = 0
        nbr_dense = 0
        epochs = 20
        print(images_path)
        filename_list = []
        for filepath in sorted(glob.glob(images_path + "/*")):
            filename_list.append(filepath)
        if settings.old:
            settings.run_2D = False

        print(len(filename_list))
        train_set, val_set, test_set = self.get_train_test_fileIndices(filename_list)


        for epoch in range(0, epochs):
            print(("$$$$$$$$$$ Epoch " + str(epoch)))

            for posIdx, pos in enumerate(train_set):
                print(("-------- Epoch {0}, train environment index {1}/{2}".format(epoch, posIdx, len(train_set))))

                filename = filename_list[pos]

                try:
                    basename, path_camera = self.get_camera_path(filename, settings)
                    print(filename)
                    if '.npy' not in filename and '.json' not in filename and os.path.exists(path_camera):
                        print("Working")
                        counter, saved_files_counter = env.work("cityscapes_train_", os.path.join(images_path, basename), agent, self.poses_db, epoch, saved_files_counter, training=True)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:

                    print("Exception")
                    if settings.stop_for_errors:
                        raise
                    else:
                        logging.exception('Fatal error in main loop ' + settings.timestamp)
                if filecounter > settings.car_test_frq * self.car_test_counter:
                    car_counter, saved_files_counter = env_car.work("test_car",filepath, agent, self.poses_db, epoch,
                                                                    saved_files_counter, road_width=pos,
                                                                    training=True)
                    people_counter, saved_files_counter = env_people.work("test_pedestrian",filepath, agent, self.poses_db, epoch,
                                                                          saved_files_counter, training=True)
                    self.car_test_counter = self.car_test_counter + 1

            # Test?
            for test_pos in val_set:
                try:
                    filename_test = filename_list[test_pos]
                    if '.npy' not in filename_test and '.json' not in filename_test:
                        counter, saved_files_counter = env.work("cityscapes_test_", filename_test, agent, self.poses_db, epoch, saved_files_counter, training=False)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    print("Exception")
                    if settings.stop_for_errors:
                        raise
                    else:
                        logging.exception('Fatal error in main loop ' + settings.timestamp)
            self.test_counter = self.test_counter + 1

            car_counter, saved_files_counter = env_car.work("test_car",filepath, agent, self.poses_db, epoch,
                                                            saved_files_counter, training=False)
            people_counter, saved_files_counter = env_people.work("test_pedestrian",filepath, agent, self.poses_db, epoch,
                                                                  saved_files_counter, training=False)

            saver.save(sess, self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt",
                       global_step=epoch)
            print(("Save model : " + self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/"))
            self.save_counter = self.save_counter + 1
            filecounter += 1
        self.evaluate_methods_cs(agent, sess, settings, viz=False)
        return counter

    def switch_training(self, epoch, car_epoch, init_epoch, settings, train_car, train_initializer, success_rate_car, collision_rate_initializer):
        print (" In switch trainig ")
        if "simultaneously" in settings.train_car_and_initialize :
            print(" Simultaneous")
            train_car=True
            train_initializer=True
            car_epoch=car_epoch+1
            init_epoch=init_epoch+1
        elif "alternatively" in settings.train_car_and_initialize:
            print(" Alternative")
            if car_epoch< (epoch//(settings.num_car_epochs+settings.num_init_epochs)+1)*settings.num_car_epochs:
                print("Set train car don't train initializer")
                train_car = True
                train_initializer = False
                car_epoch = car_epoch + 1
            elif init_epoch< (epoch//(settings.num_car_epochs+settings.num_init_epochs)+1)*settings.num_init_epochs:
                print("Set train initializer don't train car")
                train_car = False
                train_initializer = True
                init_epoch = init_epoch + 1
        elif "according_to_stats":
            print(" According to statistics")
            print ("Train car "+str(train_car)+" and reached success rate? "+str(success_rate_car>settings.car_success_rate_threshold)+" rate "+str(success_rate_car)+" threshold "+str(settings.car_success_rate_threshold))
            print ("Train initializer "+str(train_initializer)+" and reached success rate? " + str(collision_rate_initializer>settings.initializer_collision_rate_threshold) + " rate " + str(
                collision_rate_initializer) + " threshold " + str(settings.initializer_collision_rate_threshold))

            if train_initializer and collision_rate_initializer>=settings.initializer_collision_rate_threshold:
                train_car = True
                train_initializer = False
                car_epoch = car_epoch + 1
                print("Set train initializer don't train car")
            elif train_car and success_rate_car>=settings.car_success_rate_threshold:
                train_car = False
                train_initializer = True
                print("Set train car don't train car")
                init_epoch = init_epoch + 1
            elif success_rate_car==0 and collision_rate_initializer==0:
                print("Set train initializer don't train car- init")
                train_car = False
                train_initializer = True
                init_epoch = init_epoch + 1
        return train_car, train_initializer, car_epoch, init_epoch

    def adapt_variables(self, epoch, settings, success_rate_car):

        if settings.car_input == "difference" and settings.useRLToyCar:
            if epoch>150:
                settings.reward_weights_car[CAR_REWARD_INDX.distance_travelled_towards_goal] =0.1
                settings.reward_weights_car[CAR_REWARD_INDX.reached_goal] =2
            if success_rate_car>0.9 and settings.overfit and settings.seq_len_train!=settings.seq_len_train_final and settings.reward_weights_car[CAR_REWARD_INDX.reached_goal] ==2:
                print("Change variables")
                settings.seq_len_train = settings.seq_len_train_final
                settings.seq_len_test = settings.seq_len_test_final
        if settings.useRLToyCar  and (settings.train_car_and_initialize =="alternatively" or (settings.train_car_and_initialize =="according_to_stats" and settings.initializer_collision_rate_threshold<1.0)) and epoch>=4:
            settings.use_occlusion =True

        if settings.goal_dir and (settings.learn_goal or  settings.learn_init or settings.useHeroCar or settings.useRLToyCar):
            if epoch/2==0:
                settings.init_std=max(settings.init_std-0.016, 0.1)
                settings.goal_std = max(settings.goal_std - 0.016, 0.1)
        if settings.waymo and (settings.pfnn or settings.goal_dir):

            if epoch==1:
                settings.seq_len_train = settings.seq_len_train_final
                settings.seq_len_test = settings.seq_len_test_final

        if settings.pfnn:
            if epoch / 2 == 0:  # 5

                settings.sigma_vel = max(settings.sigma_vel / 2.0, 0.1)

        if settings.overfit or settings.temp_case:
            if epoch%3==0:
                settings.sigma_vel=max(settings.sigma_vel-0.016, 0.1)#-0.005

        else:
            #settings.sigma_vel = max(settings.sigma_vel - 0.0008, 0.1)  # -0.005#- 0.0014
            print(("Adapt variables epochs: "+str(epoch)+" seq len "+str(settings.seq_len_train)))
            settings.sigma_vel = max(settings.sigma_vel - 0.00533, 0.1)

            # if epoch>5 and settings.goal_dir:
            #     print("Change variables")
            #     settings.seq_len_train = settings.seq_len_train_final
            #     settings.seq_len_test = settings.seq_len_test_final
            #
            #     if epoch>15:
            #         if settings.carla:
            #             settings.seq_len_train = 450
            #             settings.seq_len_test = 450
            #         if settings.waymo:
            #             settings.seq_len_train = 200
            #             settings.seq_len_test = 200
            # if epoch>6 and not settings.goal_dir:
            #     if settings.carla:
            #         settings.seq_len_train = 450
            #         settings.seq_len_test = 450
            #     if settings.waymo:
            #         settings.seq_len_train = 200
            #         settings.seq_len_test = 200
        print("Sigma: "+str(settings.sigma_vel))


    def get_waymo_files(self, settings, evaluate):
        ending = "/*"

        filespath = settings.waymo_path if evaluate is False else settings.waymo_path_test # "Packages/CARLA_0.8.2/unrealistic_dataset/"

        epoch = 0
        pos=0
        filename_list = []
        # Get files to run on.
        print ("Path "+str(filespath))
        for filepath in glob.glob(filespath + ending):
            filename_list.append(filepath)
        return epoch, filename_list, pos


    def get_carla_files(self, settings, realtime=False):
        if settings.new_carla:
            ending = "*/*"
        else:
            ending = "test_*"
        if realtime:
            filespath = settings.carla_path_realtime
        else:
            filespath = settings.carla_path

        epoch = 0
        filename_local_list = {}
        # Get files to run on.
        print(filespath + ending)
        for filepath in glob.glob(filespath + ending):
            parts = os.path.basename(filepath).split('_')
            pos = int(parts[-1])
            if settings.new_carla:
                if pos not in filename_local_list:
                    filename_local_list[pos]=[]
                filename_local_list[pos].append(filepath)
            else:
                filename_local_list[pos] = filepath
            print (" Pos "+str(pos)+" path "+str(filepath)+" "+str(parts))
        return epoch, filename_local_list, pos

    def get_agent(self, grad_buffer, net, grad_buffer_init, init_net, grad_buffer_goal,goal_net, settings):
        if settings.cv_agent:
            return GoalCVAgent(settings, net, grad_buffer, init_net, grad_buffer_init,goal_net,grad_buffer_goal )
        if settings.carla_agent:
            return PedestrianAgent(settings)

        if settings.angular:
            if settings.pfnn:
                return ContinousNetPFNNAgent(settings, net, grad_buffer, init_net, grad_buffer_init)
            agent = ContinousNetAgent(settings, net, grad_buffer, init_net, grad_buffer_init)
            return agent
        if settings.pfnn:
            agent = AgentNetPFNN(settings, net, grad_buffer, init_net, grad_buffer_init,goal_net,grad_buffer_goal)
            return agent
        agent = NetAgent(settings, net, grad_buffer, init_net, grad_buffer_init,goal_net,grad_buffer_goal )
        return agent

    def extract_enviornments_cityscapes(self,  grad_buffer, images_path, log_file, sess, settings, writer):
        env = Environment(images_path, sess, writer, grad_buffer, log_file, settings)
        ##path,  sess, writer, gradBuffer, log,settings, net=None):
        env_car = CarEnvironment(images_path, sess, writer, grad_buffer, log_file, settings)
        env_people = PeopleEnvironment(images_path,  sess, writer, grad_buffer, log_file, settings)
        return env, env_car, env_people

    def get_environments_CARLA(self,  grad_buffer, images_path, log_file, sess, settings, writer):
        env = CARLAEnvironment(images_path,  sess, writer, grad_buffer, log_file, settings)

        env_car = CarEnvironment(images_path,  sess, writer, grad_buffer, log_file, settings)
        env_people = PeopleEnvironment(images_path,  sess, writer, grad_buffer, log_file, settings)
        return env, env_car, env_people

    def get_environments_Waymo(self,  grad_buffer, images_path, log_file, sess, settings, writer):
        env = WaymoEnvironment(images_path,  sess, writer, grad_buffer, log_file, settings)

        env_car = CarEnvironment(images_path,  sess, writer, grad_buffer, log_file, settings)
        env_people = PeopleEnvironment(images_path,  sess, writer, grad_buffer, log_file, settings)
        return env, env_car, env_people


    def pre_run_setup(self, settings):
        car_net, goal_net, init, init_net, net, saver, tmp_net = self.get_models_and_tf_saver(settings)
        images_path = settings.colmap_path
        counter = 0
        car_counter = 0
        people_counter = 0
        filecounter = 0
        settings.model_path = self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt"
        settings.save_settings("")
        logging.basicConfig(filename=os.path.join(settings.path_settings_file, settings.timestamp + '.log'))
        return  counter, filecounter, images_path, init, net, saver, tmp_net, init_net, goal_net, car_net

    def get_models_and_tf_saver(self, settings):
        net, tmp_net, init_net, init, goal_net, car_net = self.get_model(
            settings)  # Initialize various constants for 3D reconstruction.
        # Tensorboard saver.
        saver = tf.compat.v1.train.Saver(max_to_keep=50)
        return car_net, goal_net, init, init_net, net, saver, tmp_net

    def create_setting_folder(self, settings):
        if not os.path.exists(self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/"):
            os.makedirs(self.save_model_path + settings.name_movie + "_" + settings.timestamp)

    def load_net_weights(self, saver, sess, settings, latest=False, exact=False):
        print(latest)
        ignore_cars=False
        if len(settings.load_weights_car) > 0:
            ignore_cars = True
            print(" Ignore cars ")
        if len(settings.load_weights) > 0:
            if latest:
                latest_model = tf.train.latest_checkpoint(settings.load_weights)
            else:
                latest_model=settings.load_weights
            if exact:
                saver.restore(sess, latest_model)
                print("Exact")
            else:
                print(" Load weights, ignore car? "+str(ignore_cars)+" "+str(latest_model))
                restored_vars = self.get_tensors_in_checkpoint_file(file_name=latest_model, ignore_cars=ignore_cars)
                tensors_to_load = self.build_tensors_in_checkpoint_file(restored_vars)
                loader = tf.compat.v1.train.Saver(tensors_to_load)
                loader.restore(sess, settings.load_weights)
            print((settings.load_weights))

        print("Restored weights")
        if len(settings.load_weights_car) > 0:
            if 'model.ckpt' in settings.load_weights_car:
                latest_car=settings.load_weights_car
            else:
                latest_car = tf.train.latest_checkpoint(settings.load_weights_car)
            print("Load weights car "+str(settings.load_weights_car) )
            loader_car = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='car'))
            print (" Car variables")
            print(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='car'))
            if exact:
                loader_car.restore(sess, latest_car)
                ignore_cars=True
                print("Restored car weights")
            else:
                restored_vars = self.get_tensors_in_checkpoint_file(file_name=latest_car, ignore_cars=False, ignore_model=True)
                tensors_to_load = self.build_tensors_in_checkpoint_file(restored_vars)
                loader = tf.compat.v1.train.Saver(tensors_to_load)
                loader.restore(sess, settings.load_weights_car)
            print((settings.load_weights))


    def get_tensors_in_checkpoint_file(self,file_name, all_tensors=True, tensor_name=None, ignore_cars=True, ignore_model=False):
        varlist = []

        var_value = []
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                if not ignore_model:
                    if ignore_cars and "car" in key:
                        print ("Key "+str(key))
                    else:
                        varlist.append(key)
                        var_value.append(reader.get_tensor(key))
                else:
                    if "car" in key:
                        varlist.append(key)
                        var_value.append(reader.get_tensor(key))
        else:
            varlist.append(tensor_name)
            var_value.append(reader.get_tensor(tensor_name))
        return (varlist, var_value)

    def build_tensors_in_checkpoint_file(self,loaded_tensors):
        full_var_list = list()

        # Loop all loaded tensors
        for i, tensor_name in enumerate(loaded_tensors[0]):
            # Extract tensor
            #if "fully_connected" not in tensor_name:

            if True:#not "vel" in tensor_name:
                try:
                    print(("recover " + tensor_name))
                    tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name + ":0")
                    full_var_list.append(tensor_aux)
                except:
                    print(('Not found: ' + tensor_name))
            else:
                print(("Don't recover " + tensor_name))


        return full_var_list

    def create_gradient_buffer(self, sess):
        variables_policy = []
        variables_init = []
        variables_goal = []
        variables_car = []
        for v in tf.compat.v1.trainable_variables():
            if "goal" in v.name:
                variables_goal.append(v)
            elif "init" in v.name:
                variables_init.append(v)
            elif "car" in v.name:
                variables_car.append(v)
            else:
                variables_policy.append(v)
        [car_vars]=sess.run([variables_car])
        for var in car_vars:
            print ("Car var "+str(var))

        # print ("Create gradient buffer for policy ")
        grad_buffer = sess.run(variables_policy)
        for ix, grad in enumerate(grad_buffer):
            grad_buffer[ix] = grad * 0
            # print ("Buffer indx "+str(ix) +" name "+str(variables_policy[ix].name))

        # print ("Create gradient buffer for init ")
        grad_buffer_init = sess.run(variables_init)
        for ix, grad in enumerate(grad_buffer_init):
            grad_buffer_init[ix] = grad * 0
            # print ("Buffer indx " + str(ix) + " name " + str(variables_init[ix].name))
            # print ("Create gradient buffer for init ")
        grad_buffer_goal = sess.run(variables_goal)
        for ix, grad in enumerate(grad_buffer_goal):
            grad_buffer_goal[ix] = grad * 0
            # print ("Buffer indx " + str(ix) + " name " + str(variables_init[ix].name))
        grad_buffer_car = sess.run(variables_car)
        for ix, grad in enumerate(grad_buffer_car):
            grad_buffer_car[ix] = grad * 0
            # print ("Buffer indx " + str(ix) + " name " + str(variables_init[ix].name))

        return grad_buffer, grad_buffer_init,grad_buffer_goal, grad_buffer_car

    def get_model(self, settings):
        # tf.reset_default_graph()
        # tf.set_random_seed(settings.random_seed_tf)

        print("Set seed")
        tmp_net = None
        init_net=None
        goal_net = None
        car_net=None
        net=None

        if not settings.cv_agent:
            if settings.mininet:
                net=Net_no_viz(settings)

            elif settings.run_2D:

                if settings.sem == 0:
                    net = NoSem_2d(settings)
                else:

                    if settings.min_seg:
                        if settings.continous:
                            if settings.angular:
                                net= Seg_2d_min_cont_angular(settings)
                            elif settings.mem_2d:
                                net=ContinousMemNet_2D(settings)
                            else:
                                net = Seg_2d_min_cont(settings)
                        else:
                            net = Seg_2d_min_softmax(settings)
                    else:
                        net = Seg_2d(settings)

            else:

                if settings.sem == 0:
                    net = Seg_3d_no_sem(settings)
                else:
                    if settings.no_seg:
                        net=Seg_3d_no_vis(settings)
                        init = tf.global_variables_initializer()

                        return net, tmp_net, init

                    if settings.min_seg:
                        if settings.pop:
                            net=Seg_3d_min_pop(settings)
                        elif settings.confined_actions:
                            net=Seg_3d_min_still_actions(settings)
                        else:
                            net = Seg_3d_min(settings)
                    else:
                        net = Seg_3d_RGB(settings)
        print("Set seed")
        if settings.learn_init:
            if settings.sem_arch_init:
                init_net = InitializerArchNet(settings)
            elif settings.gaussian_init_net and not settings.separate_goal_net:
                init_net = InitializerGaussianNet(settings)
            else:
                init_net=InitializerNet(settings)# InitializerGaussianNet(settings)
        if settings.separate_goal_net:
            if settings.goal_gaussian:
                goal_net = GoalGaussianNet(settings)
            else:
                goal_net=GoalNet(settings)

        if settings.useRLToyCar:
            car_net=SimpleCarNet(settings)

        init = tf.compat.v1.global_variables_initializer()

        return net, tmp_net,init_net, init, goal_net, car_net

    def get_camera_path(self, filename, settings):
        basename = os.path.basename(filename)
        parts = basename.split('_')
        if len(parts)>1:
            city = parts[0]
            seq_nbr = parts[1]
            camera_file = city + "_" + seq_nbr + "_000019_camera.json"
            path_camera = os.path.join(settings.camera_path, settings.mode_name,city, camera_file)
            return basename, path_camera
        else:
            return "", ""

    def evaluate(self, settings, time_file=None, viz=False):
        if viz:
            settings.seq_len_evaluate=300#1200
            settings.update_frequency_test=100
            settings.threshold_dist = 100
            settings.deterministic=True
            if not settings.carla:
                settings.update_frequency_test = 6#10#30#10

        if not settings.random_agent:
            net, tmp_net, init_net, init,goal_net, car_net = self.get_model(settings)# Initialize various constants for 3D reconstruction.
            saver = tf.train.Saver()

        settings.model_path = self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt"
        settings.save_settings("")

        # Tensorboard saver.
        #saver = tf.train.Saver()

        latest=True
        if "model.ckpt" in settings.load_weights:
            latest=False

        sess=None
        if not settings.random_agent and not settings.goal_agent  and not settings.social_lstm:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess=tf.Session(graph=tf.get_default_graph(), config=config)

        if settings.goal_agent:

            agent = GoalAgent(settings)
        elif settings.pedestrian_agent:

            agent = PedestrianAgent(settings)
            # elif settings.cv_agent:
            #
            #     agent = GoalCVAgent(settings)
        elif settings.random_agent:

            agent = RandomAgent(settings)
        else:
            sess.run(init)
            self.load_net_weights(saver, sess, settings, latest=latest, exact=True)

            sess.graph.finalize() # grad_buffer, net, grad_buffer_init, init_net, settings

            agent = self.get_agent(None, net, None, init_net,None,goal_net, settings)
            if settings.useRLToyCar:
                car=CarAgent(settings, car_net, None)
            else:
                car=None
            if not settings.cv_agent:
                net.set_session(sess)

            if settings.learn_init:
                init_net.set_session(sess)
            if settings.separate_goal_net:
                goal_net.set_session(sess)
            if settings.useRLToyCar:
                car_net.set_session(sess)

        if settings.old:
            settings.run_2D = False

        if settings.carla:
            print( "CARLA")
            print (" Car in RL.evaluate() "+str(car))
            self.evaluate_method_carla(agent, sess, settings, viz, car=car)

        elif settings.waymo:
            self.evaluate_method_waymo(agent, sess, settings, viz)
        else:
            print( "Cityscapes")
            self.evaluate_methods_cs(agent,sess, settings, viz)



    def evaluate_methods_cs(self, agent,  sess, settings, viz=False):
        images_path =settings.colmap_path
        statics = []
        counter = 0


        env = Environment(images_path, sess, None, None, None, settings)
        if viz:
            env.init_methods=[8]
        nbr_files = 0
        nbr_dense = 0
        filename_list = []
        saved_files_counter=0

        for filepath in sorted(glob.glob(images_path + "/*")):
            filename_list.append(filepath)

        folder=""
        if viz:
            folders=["aachen_000042","tubingen_000112", "tubingen_000136", "bremen_000028", "bremen_000160", "munster_000039", "darmstadt_000019"]


            for folder in folders:
                poses=list(range(150, 200))
                filename_list = []
                for filepath in sorted(glob.glob(images_path + "/" + folder + "*")):
                    filename_list.append(filepath)

                if viz:
                    poses=list(range(len(filename_list)))


                for pos in poses:
                    filename = filename_list[pos]
                    basename, path_camera = self.get_camera_path(filename, settings)
                    try:
                        stats,saved_files_counter = env.evaluate(os.path.join(images_path, basename), agent, settings.evaluation_path,saved_files_counter, viz, folder)
                        if len(stats) > 0:
                            statics.append(stats)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except:
                        logging.exception('Fatal error in main loop ' + settings.timestamp)
                        print("Exception")
                        if not viz:
                            raise

                np.save(settings.evaluation_path, statics)
        else:

            poses = list(range(150, 200))
            if max(poses) > len(filename_list):
                print("YOU DON'T HAVE THE EVAL DATASET COMPLETE, which might be normal. fallback range")
                poses = list(range(len(filename_list))) # Note should be the last 50 environments ! range(150, 200)

            if viz:
                poses=list(range(len(filename_list)))

            for pos in poses:
                filename = filename_list[pos]
                basename, path_camera = self.get_camera_path(filename, settings)
                try:
                    stats,saved_files_counter = env.evaluate("cityscapes_eval_", os.path.join(images_path, basename), agent, settings.evaluation_path,saved_files_counter, viz, folder)
                    if len(stats) > 0:
                        statics.append(stats)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    logging.exception('Fatal error in main loop ' + settings.timestamp)
                    print("Exception")

                np.save(settings.evaluation_path, statics)

    def get_train_test_fileIndices(self, datasetFilenamesList, overfit = False, waymo=False, carla=False, new_carla=False, realtime_carla=False):
        all_dataset_size = len(datasetFilenamesList)
        spawn_points=list(datasetFilenamesList.keys())
        if realtime_carla:
            train_size = 25
            train_set = spawn_points[0:15]
            val_set = spawn_points[15:]
            print (datasetFilenamesList)
            print(" train set "+str(train_set))
            print(" val set " + str(val_set))
            test_set = []
        elif new_carla :
            train_size=25
            train_set = spawn_points[0:train_size]
            val_set = spawn_points[train_size:]
            test_set = []
        elif carla:
            train_size = 100#int(all_dataset_size * 0.667)
            train_set = list(range(train_size))
            val_set = list(range(train_size, all_dataset_size))
            test_set=list(range(0, 150))
        elif waymo:
            train_size = 70#int(all_dataset_size * 0.667)
            train_set = list(range(train_size))
            val_set = list(range(train_size, 80))
            test_set=list(range(80, all_dataset_size))
        else:
            train_size = 100  # int(all_dataset_size * 0.667)
            train_set = list(range(train_size))
            val_set = list(range(train_size, 150))
            test_set = list(range(150, all_dataset_size))
        return train_set, val_set, test_set


    def evaluate_method_carla(self, agent, sess, settings, viz=False, supervised=False, car=None):
        if not os.path.exists(settings.evaluation_path):
            os.makedirs(settings.evaluation_path)

        filespath = settings.carla_path_test
        if viz:
            filespath = settings.carla_path_viz
            ending = "test_*"
        statics = []
        print("CARLA ")

        if not supervised:
            env = CARLAEnvironment(filespath, sess, None, None, None, settings)
        else:
            env = SupervisedCARLAEnvironment(filespath, sess, None, None, None, settings)

        #env_car = CarEnvironment(filespath, sess, None, None, None, settings)
        #env_people = PeopleEnvironment(filespath,  sess, None, None, None, settings)
        ending = "test_*"

        # if viz:
        #     env.init_methods=[2]

        epoch = 0
        filename_list = {}
        saved_files_counter=0
        # Get files to run on.
        for filepath in glob.glob(filespath + ending):
            parts = os.path.basename(filepath).split('_')
            pos = int(parts[-1])
            filename_list[pos] = filepath
        print (" Files ")
        print (filename_list)
        #file_nbrs=list(filename_list.keys())
        # if viz:
        #     file_nbrs=[0,8,24,36]
        for  pos in sorted(filename_list.keys()):
            filepath = filename_list[pos]
            if pos % settings.VAL_JUMP_STEP_CARLA == 0 or viz:
                print(f"## Evaluating file {pos} out of 150")
                stats, saved_files_counter, initializer_stats= env.evaluate(get_carla_prefix_eval(supervised,  settings.useRLToyCar,settings.new_carla, settings.realtime_carla),filepath, agent, settings.evaluation_path, saved_files_counter,realtime_carla=settings.realtime_carla_only,car=car,viz=viz)
                if len(stats) > 0:
                    statics.extend(stats)
        np.save(settings.evaluation_path, statics)




    def evaluate_method_waymo(self, agent, sess, settings, viz=False, supervised=False):
        if not os.path.exists(settings.evaluation_path):
            os.makedirs(settings.evaluation_path)

        filespath = settings.waymo_path_test
        if viz:
            filespath = settings.waymo_path_viz
            ending = "/*"
        statics = []
        print("WAYMO ")

        if not supervised:
            env = WaymoEnvironment(filespath, sess, None, None, None, settings)
        else:
            env = SupervisedWaymoEnvironment(filespath, sess, None, None, None, settings)

        epoch, filename_list, pos = self.get_waymo_files(settings, evaluate=True)

        if viz:
            env.init_methods = [1]

        saved_files_counter=0


        if viz:
            val_set = ["15646511153936256674_1620_000_1640_000", "18311996733670569136_5880_000_5900_000"]
            for test_pos in val_set:
                filepath = os.path.join(settings.waymo_path, test_pos)
                stats, saved_files_counter = env.evaluate(get_waymo_prefix_eval( supervised),
                                                          filepath, agent, settings.evaluation_path, saved_files_counter)
        else:
            for keyIndex, filepath in enumerate(filename_list):
                if keyIndex % settings.VAL_JUMP_STEP_WAYMO == 0:
                    print(f"## Evaluating key {keyIndex} out of {len(filename_list)}")
                    stats, saved_files_counter =env.evaluate(get_waymo_prefix_eval( supervised),
                                                             filepath, agent, settings.evaluation_path, saved_files_counter)

                    if len(stats) > 0:
                        statics.extend(stats)
            np.save(settings.evaluation_path, statics)

        self.test_counter = self.test_counter + 1


    def viz_decision(self, settings, time_file=None):


        if not settings.random_agent and not settings.goal_agent and not settings.cv_agent:
            net, tmp_net, init_net, init,goal_net,car_net = self.get_model(settings)# Initialize various constants for 3D reconstruction.
            saver = tf.train.Saver()

        # Tensorboard saver.
        #saver = tf.train.Saver()
        images_path=settings.colmap_path
        statics=[]
        sess=None

        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.gpu_options.allow_growth = True
        sess=tf.Session(graph=tf.get_default_graph(), config=config)

        sess.run(init)
        self.load_net_weights(saver, sess, settings, latest=True)
        # Load settings folder!

        sess.graph.finalize()
        agent = self.get_agent(None, net, settings)
        net.set_session(sess)
        init_net.set_session(sess)



        print("CARLA ")
        env = CARLAEnvironment(images_path, sess, None, None, None, settings)
        ending="test_*"
        filespath = settings.carla_path_test

        epoch = 19
        filename_list={}
        # Get files to run on.
        for filepath in glob.glob(filespath+ending):
            parts=os.path.basename( filepath).split('_')
            pos=int(parts[-1])
            filename_list[pos]=filepath

        pos=19
        filepath=filename_list[pos]
        stats = env.evaluate(filepath, agent,settings.evaluation_path)

    def set_seq_len(self, settings):
        pass


def main(setting):
    f=None
    if setting.timing:
        f=open('times.txt', 'w')
        start = time.time()
    rl=RL()

    if not f is None:
        end=time.time()
        f.write(str(end-start)+ " Setting up RL\n")
    rl.run(setting, time_file=f)

#import carlaEnv

if __name__ == "__main__":
    setup=run_settings()

    np.random.seed(setup.random_seed_np)

    tf.compat.v1.set_random_seed(setup.random_seed_tf)

    if setup.profile:
        print("Run profiler----------------------------------------------------------------------")
        import cProfile, pstats
        profiler = cProfile.Profile()
        profiler.enable()
        main(setup)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
        print(" Save to "+str(os.path.join(setup.profiler_file_path, setup.name_movie+".psats")))
        stats.dump_stats(os.path.join(setup.profiler_file_path, setup.name_movie+".psats"))
    else:
        main(setup)


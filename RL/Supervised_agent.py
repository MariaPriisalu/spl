
import tensorflow as tf
from settings. import RANDOM_SEED
tf.compat.v1.set_random_seed(RANDOM_SEED)

from supervised_environment import SupervisedEnvironment, SupervisedWaymoEnvironment, SupervisedCARLAEnvironment
from agent_net import SupervisedAgent, PedestrianLikelihoodAgent, PedestrianLikelihoodPFNNNetAgent, PedestrianLikelihoodPFNNAgent, SupervisedPFNNAgent
from agents_dummy import PedestrianAgent,RandomAgent
from agents_goal import  CVMAgent

from environment_test_special_cases import CarEnvironmentSupervised,PeopleEnvironmentSupervised
from environment import Environment
import glob

from mini_net import Net_no_viz_supervised
import os
from datetime import datetime
from utils.constants import Constants
import numpy as np
import scipy.io as sio

from settings import run_settings, USE_LOCAL_PATHS, CARLA_CACHE_PREFIX_EVALUATE_SUPERVISED, WAYMO_CACHE_PREFIX_EVALUATE_SUPERVISED
import time
import logging
from RL import RLmain
from net_sem_2d import  SupervisedNet_2D


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



class RL_S(RLmain):
    def __init__(self, agentType):

        self.poses_path = "Datasets/cityscapes/poses/" if USE_LOCAL_PATHS == 0 else "localUserData/Datasets/cityscapes/poses/"
        # Log path tensorboard
        self.log_path = "Datasets/cityscapes/logs/" if USE_LOCAL_PATHS == 0 else "localUserData/Datasets/cityscapes/logs/"
        # Log path
        self.save_model_path = "Models/rl_model/" if USE_LOCAL_PATHS == 0 else "localUserData/Models/rl_model/"

        self.test_counter = -1
        self.car_test_counter = 0
        self.save_counter = 0

        print(("Poses path exists? " + str(os.path.exists(self.poses_path))))


        if not os.path.exists(self.poses_path):
            self.poses_path = "Documents/Datasets/Human3.6/Release-v1.1/" if USE_LOCAL_PATHS == 0 else "localUserData/Datasets/Human3.6/Release-v1.1/"

            self.log_path = "Results/agent/" if USE_LOCAL_PATHS == 0 else "localUserData/Results/agent/"
            self.save_model_path = "Results/agent/" if USE_LOCAL_PATHS == 0 else "localUserData/Results/agent/"

        if os.path.exists(self.poses_path):
            self.poses_db = sio.loadmat(self.poses_path + "poses_hips.mat")
            self.poses_db = np.delete(self.poses_db["poses"], [24, 25, 26], 1)
        else:
            self.poses_db = None
            print("No Poses db!")

        self.test_counter = -1

        self.agentType = agentType


    def run_tests(self,  settings, time_file=None, mem_agent=False):
        counter, filecounter, images_path, init, net, saver, tmp_net, init_net = self.pre_run_setup(settings)

        if not settings.evaluateExternalModel:
            with tf.Session() as sess:
                sess.run(init)
                grad_buffer,grad_buffer_init = self.create_gradient_buffer(sess)
                self.create_setting_folder(settings)

                epoch, filename_list, pos =self.get_carla_files()

                saver.restore(sess, settings.load_weights)  # latest)#ckpt.model_checkpoint_path)

                sess.graph.finalize()
                agent = SupervisedAgent(net, grad_buffer)
                net.set_session(sess)

                self.evaluate_ped(agent, sess, settings)
        else:
                agent = SupervisedAgent(None, None)
                self.evaluate_ped(agent, None, settings)

    # Tensorflow session.
    def run(self,  settings, time_file=None, supervised=True):
        counter, filecounter, images_path, init, net, saver, tmp_net, init_net = self.pre_run_setup(settings)


        self.set_seq_len(settings)
        settings.car_test_frq=np.inf
        settings.seq_len_train=450
        settings.seq_len_test=450
        if settings.waymo:
            settings.car_test_frq = np.inf
            settings.seq_len_train = 200
            settings.seq_len_test = 200

        #from tensorflow.python import debug as tf_debug
        with tf.Session() as sess:
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            sess.run(init)

            grad_buffer,grad_buffer_init = self.create_gradient_buffer(sess)
            print((settings.load_weights))
            if "model.ckpt" in settings.load_weights:
                self.load_net_weights(saver, sess, settings, latest=False)#, exact=True)
            else:
                self.load_net_weights(saver, sess, settings, latest=True)
            self.create_setting_folder(settings)


            sess.graph.finalize()
            agent = self.get_agent(grad_buffer, net, settings)
            net.set_session(sess)
            writer=None
            log_file=None

            saved_files_counter=0
            settings.car_test_frq=np.inf

            if settings.carla:

                counter = self.train_on_carla(agent,  counter, filecounter, grad_buffer, images_path,
                                              log_file, saved_files_counter, saver, sess, settings, writer, supervised=True)
            elif settings.waymo:
                counter = self.train_on_waymo(agent, counter, filecounter, grad_buffer, images_path, log_file,
                                              saved_files_counter, saver, sess, settings, writer, supervised=True)

            else:
                print("Cityscapes ")

                images_path =settings.colmap_path
                statics = []
                counter = 0
                env = Environment(images_path,  sess, None, None, None, settings)
                nbr_files = 0
                nbr_dense = 0
                filename_list = []

                for filepath in sorted(glob.glob(images_path + "/*")):
                    filename_list.append(filepath)
                for pos in range(150, 200):
                    filename = filename_list[pos]

                    basename, path_camera = self.get_camera_path(filename, settings)
                    print((os.path.join(images_path, basename)))

                    stats,saved_files_counter = env.evaluate(os.path.join(images_path, basename), agent, settings.evaluation_path,saved_files_counter)
                    if len(stats) > 0:
                        statics.append(stats)
                np.save(settings.evaluation_path, statics)

    def set_seq_len(self, settings):
        settings.seq_len_train = settings.seq_len_train_final
        settings.seq_len_test = settings.seq_len_test_final

    """
    def evaluate_method_carla(self, agent, sess, settings, viz=False):
        filespath = settings.carla_path_test
        statics = []
        print "CARLA "
        env = SupervisedCARLAEnvironment(filespath,  sess, None, None, None, settings)
        ending = "test_*"

        epoch = 0
        filename_list = {}
        saved_files_counter=0
        # Get files to run on.
        for filepath in glob.glob(filespath + ending):
            parts = os.path.basename(filepath).split('_')
            pos = int(parts[-1])
            filename_list[pos] = filepath
        for keyIndex, keyName in enumerate(filename_list.keys()):
            filepath = filename_list[keyName]
            if keyIndex % 4 == 0:
                print filepath
                stats, saved_files_counter = env.evaluate(CARLA_CACHE_PREFIX_EVALUATE_SUPERVISED, filepath, agent, settings.evaluation_path, saved_files_counter)
                if len(stats) > 0:
                    statics.extend(stats)
        np.save(settings.evaluation_path, statics)
    """

    def evaluate_ped(self, agent,  sess, settings):
        filespath = settings.carla_path_test
        statics = []
        print("CARLA ")
        env = SupervisedCARLAEnvironment(filespath, sess, None, None, None, settings)
        ending = "test_*"

        epoch = 0
        filename_list = {}
        saved_files_counter=0
        # Get files to run on.
        for filepath in glob.glob(filespath + ending):
            parts = os.path.basename(filepath).split('_')
            pos = int(parts[-1])
            filename_list[pos] = filepath
        for pos in range(len(filename_list)):
            filepath = filename_list[pos]
            #if pos % 4 == 0:
            print(filepath)
            #  cachePrefix, file_name, agent, fileagent, saved_files_counter

            stats, saved_files_counter = env.evaluate(CARLA_CACHE_PREFIX_EVALUATE_SUPERVISED, filepath, agent, settings.evaluation_path, saved_files_counter)

            if len(stats) > 0:
                statics.extend(stats)
        np.save(settings.evaluation_path, statics)

    def get_environments_CARLA(self, grad_buffer, images_path, log_file, sess, settings, writer):
        env = SupervisedCARLAEnvironment(images_path, sess, writer, grad_buffer, log_file, settings)
        return env, None, None

    def get_environments_Waymo(self,  grad_buffer, images_path, log_file, sess, settings, writer):
        env = SupervisedWaymoEnvironment(images_path,  sess, writer, grad_buffer, log_file, settings)
        return env , None, None



    def get_model(self, settings):
        tmp_net = None
        if settings.likelihood:
            return super(RL_S, self).get_model(settings)

        tf.reset_default_graph()
        if settings.mininet:
            net = Net_no_viz_supervised(settings)
            # elif settings.likelihood:
            #     return super(RL_S, self).get_model(settings)
        else:
            net=SupervisedNet_2D(settings)

        init = tf.global_variables_initializer()
        return net,tmp_net, init

    def get_agent(self, grad_buffer, net, settings):
        print("Get agent ")
        if settings.likelihood:

            print ("Likelihood agent")

            if settings.pfnn:
                if settings.evaluateExternalModel is None:
                    return PedestrianLikelihoodPFNNNetAgent(settings,net, grad_buffer)
                else:
                    return PedestrianLikelihoodPFNNAgent(settings, net, grad_buffer)
            return PedestrianLikelihoodAgent(settings,net, grad_buffer)
        if settings.pedestrian_agent:
            print("Pedestrian agent")
            return PedestrianAgent(settings,net, grad_buffer)
        elif settings.cv_agent:
            print("Constant vel agent")
            return CVMAgent(settings,net, grad_buffer)
        elif settings.random_agent:
            print("Random agent")
            return RandomAgent(settings,net, grad_buffer)
        else:
            print("Supervised agent")
            if settings.pfnn:

                # print "PFNN"
                # agent=AgentNetPFNN(settings,net, grad_buffer)#SupervisedPFNNAgent( settings,net, grad_buffer)

                print("PFNN")
                agent=SupervisedPFNNAgent( settings,net, grad_buffer)

            else:
                print("No PFNN")
                agent = SupervisedAgent( settings,net, grad_buffer)
        return agent

    def adapt_variables(self, epoch, settings):
        pass


    def evaluate(self,  settings, time_file=None, viz=False):

        settings.model_path = self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt"
        settings.save_settings("")
        sess = None

        if settings.evaluateExternalModel is None:
            net, _, init = self.get_model(settings)
            saver = tf.train.Saver()
        else:
            net = None
            init = None
            saver = None
        agent = self.get_agent(None, net, settings)

        config = tf.ConfigProto()
        if not settings.pedestrian_agent and not settings.random_agent and not settings.cv_agent and settings.evaluateExternalModel is None:
            sess = tf.Session(config=config)
            sess.run(init)
            self.load_net_weights(saver, sess, settings, exact=True)
            sess.graph.finalize()
            net.set_session(sess)
        else:
            sess = None

        if settings.carla:
            self.evaluate_method_carla(agent, sess, settings, supervised = True)
        elif settings.waymo:
            self.evaluate_method_waymo(agent, sess, settings, supervised = True)
        #self.evaluate_ped(agent,  sess, settings)




def main(setting, agentType):
    f=None
    if setting.timing:
        f=open('times.txt', 'w')
        start = time.time()


    rl=RL_S(agentType=agentType)
    #rl.run(setting)

    rl.evaluate(setting)
    # if not f is None:
    #     end=time.time()
    #     f.write(str(end-start)+ " Setting up RL\n")
    # rl.run(setting, time_file=f, mem_agent=False)


if __name__ == "__main__":
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    train_supervised=False

    if train_supervised:
        setup=run_settings(evaluate=False)
        tf.compat.v1.set_random_seed(setup.random_seed_tf)
        np.random.seed(setup.random_seed_np)
        main(setup, "")
    else:

        evaluationModels = ["STGCNN", "SGAN", "STGAT"] # our model is [""]
        datasets = ["carla", "waymo"] # ["carla", "waymo", "cityscapes"]
        likelihoods = [True]
        pfnnOptions = [True] #[True, False]

        for evaluatedModel in evaluationModels:
            for dataset in datasets:
                for pfnnOption in pfnnOptions:
                    for likelihood in likelihoods:
                        print(("========== RUNNING SUPERVISED MODEL: ", evaluatedModel, " on dataset: ", dataset))
                        setup = run_settings(evaluate=True, likelihood=likelihood, evaluatedModel=evaluatedModel, datasetToUse=dataset, pfnn=pfnnOption)
                        np.random.seed(setup.random_seed_np)
                        tf.compat.v1.set_random_seed(setup.random_seed_tf)
                        main(setup, evaluatedModel)



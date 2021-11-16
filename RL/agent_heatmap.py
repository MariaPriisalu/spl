
import os
external=True

from settings import RANDOM_SEED
import tensorflow as tf
tf.compat.v1.set_random_seed(RANDOM_SEED)
#import matplotlib.pyplot as plt
from .carla_environment import CARLAEnvironment, SupervisedEnvironment
from .environment import Environment
from .environment_test import TestEnvironment
from .agent_net import NetAgent, RandomAgent, SupervisedAgent
from .net_2d import Net_2d
from .net_2d_memory import Net_2d_mem, Seg_2d_mem
from .net_sem_2d import Seg_2d, Seg_2d_min
from .net_sem_3D import Seg_3d, Seg_3d_min, Seg_3d_RGB, Seg_3d_no_sem
from .net_no_sem_2D import NoSem_2d, NetCars_2d, NetPedestrians_2d, NetRGB_2d, Only_Sem_2d

from .environment_test_special_cases import CarEnvironment, PeopleEnvironment

import glob


import os
from datetime import datetime
from utils.constants import Constants
import numpy as np
import scipy.io as sio
from .nets_abstract import SimpleSoftMaxNet
from .settings import run_settings
import time
from tensorflow.python import pywrap_tensorflow
import logging
import scipy.io as sio


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
        self.poses_path = "Datasets/cityscapes/poses/"
        # Log path tensorboard
        self.log_path = "Datasets/cityscapes/logs/"
        # Log path
        self.save_model_path="Models/rl_model/"

        self.test_counter=-1
        self.car_test_counter=0
        self.save_counter=0


        if not os.path.exists(self.poses_path):
            self.poses_path = "Documents/Datasets/Human3.6/Release-v1.1/"

            self.log_path = "Results/agent/"
            self.save_model_path="Results/agent/"

        if os.path.exists(self.poses_path):
            self.poses_db = sio.loadmat(self.poses_path + "poses_hips.mat")
            self.poses_db = np.delete(self.poses_db["poses"], [24, 25, 26], 1)
        else:
            self.poses_db=None
            print("No Poses db!")


    # Tensorflow session.
    def run(self,  settings, time_file=None):


        if settings.timing:
            start=time.time()

        constants = Constants()
        net, tmp_net, init = self.get_model(settings)# Initialize various constants for 3D reconstruction.


        # Tensorboard saver.
        saver = tf.train.Saver()

        images_path="Datasets/colmap2/"
        if not os.path.exists(images_path):
            images_path = "Datasets/colmap/colmap/"
        counter = 0
        car_counter=0
        people_counter = 0
        filecounter=0
        settings.model_path= self.save_model_path + settings.name_movie+"_"+settings.timestamp+"/model.ckpt"
        settings.save_settings("")
        logging.basicConfig(filename=os.path.join(settings.path_settings_file,settings.timestamp+'.log'))

        results_path = 'Packages/CARLA_0.8.2/viz_results'
        if not os.path.exists(results_path):
            results_path = "Results/viz_results"

        #from tensorflow.python import debug as tf_debug
        with tf.Session() as sess:
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            sess.run(init)

            grad_buffer = self.create_gradient_buffer(sess)
            print((settings.load_weights))
            self.load_net_weights(saver, sess, settings, latest=True)
            self.create_setting_folder(settings)

            sess.graph.finalize()
            agent = NetAgent(net, grad_buffer)
            if settings.refine_follow:
                agent = SupervisedAgent(net, grad_buffer)
            net.set_session(sess)
            writer=None
            log_file=None
            run_heatmap=True

            saved_files_counter=0

            if settings.carla:

                print("CARLA ")
                env = CARLAEnvironment(images_path, constants, sess, writer, grad_buffer, log_file, settings)

                env_car = CarEnvironment(images_path, constants, sess, writer, grad_buffer, log_file, settings)
                env_people = PeopleEnvironment(images_path, constants, sess, writer, grad_buffer, log_file, settings)
                ending="test_*"
                filespath = "Packages/CARLA_0.8.2/PythonClient/new_data/"#"Packages/CARLA_0.8.2/unrealistic_dataset/"
                if not os.path.exists(filespath):
                    filespath="Datasets/carla-sync/train/"
                    #ending = "standing_still_*"

                epoch = 0
                filename_list={}
                # Get files to run on.
                for filepath in glob.glob(filespath+ending):
                    parts=os.path.basename( filepath).split('_')
                    pos=int(parts[-1])
                    filename_list[pos]=filepath
                if run_heatmap:
                    filepath = filename_list[144]
                    init_pos = [5, 42, 55]
                    env.get_heatmap_runs( filepath, agent, init_pos[2],init_pos[1],os.path.join(results_path, settings.name_movie+"_heatmap_traj"))
                else:
                    filepath = filename_list[100]

                    heatmap, maxdir= env.get_heatmap(filepath, agent, self.poses_db, epoch, saved_files_counter, training=True)
                    # car_counter,saved_files_counter=env_car.get_heatmap(filepath, agent, self.poses_db, epoch,saved_files_counter, road_width=pos, training=True)
                    #heatmap, maxdir = env_people.get_heatmap(filepath, agent, self.poses_db, epoch, saved_files_counter, training=True)
                    sio.savemat(os.path.join(results_path, settings.name_movie+"_heatmap_prob"),{'heatmap':heatmap,'maxdir': maxdir})





    def create_setting_folder(self, settings):
        if not os.path.exists(self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/"):
            os.makedirs(self.save_model_path + settings.name_movie + "_" + settings.timestamp)

    def load_net_weights(self, saver, sess, settings, latest=False):
        print((settings.load_weights))
        if len(settings.load_weights) > 0:
            if latest:
                #from https://stackoverflow.com/questions/41621071/restore-subset-of-variables-in-tensorflow
                latest = tf.train.latest_checkpoint(settings.load_weights)
                # restored_vars = self.get_tensors_in_checkpoint_file(file_name=latest)
                # tensors_to_load = self.build_tensors_in_checkpoint_file(restored_vars)
                # loader = tf.train.Saver(tensors_to_load)
                # loader.restore(sess, latest)
                #
                # print latest
                saver.restore(sess,latest)# latest)#ckpt.model_checkpoint_path)
                print("Restored weights")
            else:
                restored_vars = self.get_tensors_in_checkpoint_file(file_name=settings.load_weights)
                tensors_to_load = self.build_tensors_in_checkpoint_file(restored_vars)
                loader = tf.train.Saver(tensors_to_load)
                loader.restore(sess, settings.load_weights)
                #saver.restore(sess, settings.load_weights)  # latest)#ckpt.model_checkpoint_path)
                print(("Restored weights "+settings.load_weights))

    def get_tensors_in_checkpoint_file(self,file_name, all_tensors=True, tensor_name=None):
        varlist = []

        var_value = []
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
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
            try:
                tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name + ":0")
            except:
                print(('Not found: ' + tensor_name))
            full_var_list.append(tensor_aux)
        return full_var_list

    def create_gradient_buffer(self, sess):
        grad_buffer = sess.run(tf.trainable_variables())
        for ix, grad in enumerate(grad_buffer):
            grad_buffer[ix] = grad * 0
        return grad_buffer

    def get_model(self, settings):
        tf.reset_default_graph()
        tmp_net = None
        if settings.run_2D:
            if settings.sem == 0:
                net = NoSem_2d(settings)
            else:
                if settings.min_seg:
                    net = Seg_2d_min(settings)
                else:
                    net = Seg_2d(settings)

        else:
            if settings.sem == 0:
                net = Seg_3d_no_sem(settings)
            else:
                if settings.min_seg:
                    net = Seg_3d_min(settings)
                else:
                    net = Seg_3d_RGB(settings)
        init = tf.global_variables_initializer()

        return net, tmp_net, init

    def get_camera_path(self, filename):
        basename = os.path.basename(filename)
        parts = basename.split('_')
        city = parts[0]
        seq_nbr = parts[1]
        camera_file = city + "_" + seq_nbr + "_000019_camera.json"
        path_camera = os.path.join(
            "Datasets/cityscapes/camera_trainvaltest/camera", "train",
            city, camera_file)
        return basename, path_camera

    def evaluate(self, settings, time_file=None):

        constants = Constants()
        if not settings.random_agent:
            net, tmp_net, init = self.get_model(settings)# Initialize various constants for 3D reconstruction.
            saver = tf.train.Saver()



        # Tensorboard saver.
        #saver = tf.train.Saver()

        sess=None
        if not settings.random_agent:
            config = tf.ConfigProto()

            sess=tf.Session(config=config)

        if settings.random_agent:
            agent = RandomAgent()
        else:
            sess.run(init)
            self.load_net_weights(saver, sess, settings, latest=False)


            sess.graph.finalize()
            agent = NetAgent(net, None)
            net.set_session(sess)


        if settings.carla:
            self.evaluate_method_carla(agent, constants, sess, settings)

        else:
            self.evaluate_methods_cs(agent, constants,  sess, settings)



    def evaluate_methods_cs(self, agent, constants,  sess, settings):
        images_path =settings.colmap_path
        statics = []
        counter = 0
        env = Environment(images_path, constants, sess, None, None, None, settings)
        nbr_files = 0
        nbr_dense = 0
        filename_list = []
        saved_files_counter=0

        for filepath in sorted(glob.glob(images_path + "/*")):
            filename_list.append(filepath)
        for pos in range(150, 200):
            filename = filename_list[pos]

            basename, path_camera = self.get_camera_path(filename)
            stats,saved_files_counter = env.evaluate(os.path.join(images_path, basename), agent, settings.evaluation_path,saved_files_counter)
            if len(stats) > 0:
                statics.append(stats)
        np.save(settings.evaluation_path, statics)

    def evaluate_method_carla(self, agent, constants, sess, settings):
        filespath = "Packages/CARLA_0.8.2/PythonClient/new_data2/"
        if not os.path.exists(filespath):
            filespath = "Datasets/carla-sync/val/"
        statics = []
        print("CARLA ")
        env = CARLAEnvironment(filespath, constants, sess, None, None, None, settings)
        env_car = CarEnvironment(filespath, constants, sess, None, None, None, settings)
        env_people = PeopleEnvironment(filespath, constants, sess, None, None, None, settings)
        ending = "test_*"

        # if settings.random_agent:
        #     filespath = "Packages/CARLA_0.8.2/PythonClient/new_data/"  # "Packages/CARLA_0.8.2/unrealistic_dataset/"
        #     if not os.path.exists(filespath):
        #         filespath = "Datasets/carla-sync/train/"
        #     epoch = 0
        #     filename_list = {}
        #     # Get files to run on.
        #     for filepath in glob.glob(filespath + ending):
        #         parts = os.path.basename(filepath).split('_')
        #         pos = int(parts[-1])
        #         filename_list[pos] = filepath
        #     saved_files_counter=0
        #     train_set = range(100)
        #     test_set = range(100, len(filename_list))
        #     for test_pos in test_set:
        #         if test_pos % 4 == 0:
        #             filepath = filename_list[test_pos]
        #             try:
        #                 _, saved_files_counter = env.work(filepath, agent, self.poses_db, epoch, saved_files_counter,
        #                                                   training=False)
        #             except (KeyboardInterrupt, SystemExit):
        #                 raise
        #             except:
        #                 logging.exception('Fatal error in main loop ' + settings.timestamp)
        #                 print "Exception"
        #                 if settings.stop_for_errors:
        #                     raise
        #     print "Testing "
        #     car_counter, saved_files_counter = env_car.work(filepath, agent, self.poses_db, epoch, saved_files_counter,
        #                                                     training=False)
        #     people_counter, saved_files_counter = env_people.work(filepath, agent, self.poses_db, epoch,
        #                                                           saved_files_counter, training=False)
        #     self.test_counter = self.test_counter + 1
        #
        # else:
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
            if pos % 4 == 0:
                print(filepath)
                stats, saved_files_counter = env.evaluate(filepath, agent, settings.evaluation_path, saved_files_counter)
                if len(stats) > 0:
                    statics.extend(stats)
        np.save(settings.evaluation_path, statics)

    def viz_decision(self, settings, time_file=None):

        constants = Constants()
        if not settings.random_agent:
            net, tmp_net, init = self.get_model(settings)# Initialize various constants for 3D reconstruction.
            saver = tf.train.Saver()

        # Tensorboard saver.
        #saver = tf.train.Saver()
        images_path="Datasets/colmap2/"
        statics=[]
        sess=None

        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.2
        #config.gpu_options.allow_growth = True
        sess=tf.Session(config=config)

        sess.run(init)
        self.load_net_weights(saver, sess, settings)
        # Load settings folder!

        sess.graph.finalize()
        agent = NetAgent(net, None)
        net.set_session(sess)



        print("CARLA ")
        env = CARLAEnvironment(images_path, constants, sess, None, None, None, settings)
        ending="test_*"
        filespath = "Packages/CARLA_0.8.2/PythonClient/val/"
        if not os.path.exists(filespath):
            filespath="Datasets/carla_val/"
            #ending = "standing_still_*"

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

if __name__ == "__main__":
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    setup=run_settings()
    tf.compat.v1.set_random_seed(setup.random_seed_tf)
    np.random.seed(setup.random_seed_np)
    main(setup)

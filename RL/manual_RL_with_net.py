
import tensorflow as tf
from setting import RANDOM_SEED
tf.compat.v1.set_random_seed(RANDOM_SEED)
external=False
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from manual_environment import ManualCARLAEnvironment, ManualEnvironment, ManualWaymoEnvironment

from agent_manual import  ManualAgentContinousActions# PFNNManualAgent, ManualActionsContinousAgent,

from environment_test_special_cases import CarEnvironment, PeopleEnvironment

import numpy as np

from settings import run_settings
import time

from RL.RLmain import RL


np.set_printoptions(precision=5)
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




class RL_manual(RL):
    def __init__(self):
        super(RL_manual, self).__init__()

    def run(self,  settings, time_file=None, supervised=False):
        settings.update_frequency = 5
        settings.seq_len_train = 10#30#2000#12
        settings.seq_len_test = 10
        super(RL_manual, self).run( settings, time_file, supervised)

    def get_agent(self, grad_buffer, net, grad_buffer_init, init_net, grad_buffer_goal, goal_net, settings):
        agent = ManualAgentContinousActions(settings, net, grad_buffer, init_net, grad_buffer_init, goal_net, grad_buffer_goal)
        return agent

    def get_environments_CARLA(self, grad_buffer, images_path, log_file, sess, settings, writer):
        env = ManualCARLAEnvironment(images_path, sess, None, grad_buffer, None, settings)
        env_car = CarEnvironment(images_path, sess, None, None, None, settings)
        env_people = PeopleEnvironment(images_path,  sess, None, None, None, settings)
        return env, env_car, env_people

    # def adjust_training_set(self, num_epochs, settings, train_set, val_set):
    #
    #     train_set = [4]
    #     val_set = [104]  # [100]
    #     num_epochs = 1
    #     return num_epochs, train_set, val_set

    def extract_enviornments_cityscapes(self, grad_buffer, images_path, log_file, sess, settings, writer):
        env = ManualEnvironment(images_path, sess, writer, grad_buffer, log_file, settings)
        env_car = CarEnvironment(images_path, sess, writer, grad_buffer, log_file, settings)
        env_people = PeopleEnvironment(images_path, sess, writer, grad_buffer, log_file, settings)
        return env, env_car, env_people


    def get_environments_Waymo(self, grad_buffer, images_path, log_file, sess, settings, writer):
        env = ManualWaymoEnvironment(images_path, sess, writer, grad_buffer, log_file, settings)
        env_car = CarEnvironment(images_path, sess, writer, grad_buffer, log_file, settings)
        env_people = PeopleEnvironment(images_path, sess, writer, grad_buffer, log_file, settings)
        return env, env_car, env_people




def main(setting):
    f=None
    if setting.timing:
        f=open('times.txt', 'w')
        start = time.time()
    rl=RL_manual()

    if not f is None:
        end=time.time()
        f.write(str(end-start)+ " Setting up RL\n")
    rl.run(setting, time_file=f)

if __name__ == "__main__":
    # import matplotlib as mpl
    # mpl.use('Agg')
    # import matplotlib.pyplot as plt

    setup=run_settings()

    np.random.seed(setup.random_seed_np)
    main(setup)

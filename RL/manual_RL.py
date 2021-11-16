
external=False
import matplotlib
print((matplotlib.__version__))
print((matplotlib.__file__))
print((matplotlib.is_interactive()))
print((matplotlib.get_backend()))

from agent_manual_simplified import ManualAgent

import numpy as np

from settings import run_settings
import time

from manual_RL_with_net import RL_manual
from agent_car import CarAgent

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




class RL_manual_no_tf(RL_manual):
    def __init__(self):

        super(RL_manual_no_tf, self).__init__()

    def get_agent(self, grad_buffer, net, grad_buffer_init, init_net, grad_buffer_goal, goal_net, settings):
        agent = ManualAgent(settings)

        return agent

    def run(self,settings, time_file=None, supervised=False):
        # print("Supervised? "+supervised)
        # Check if episodes cached data should be deleted
        sess=None
        if settings.deleteCacheAtEachRun:
            self.eraseCachedEpisodes(settings.target_episodesCache_Path)
            print("I have deleted the cache folder !")

        counter, filecounter, images_path, init, net, saver, tmp_net, init_net, goal_net, car_net = self.pre_run_setup(
            settings)

        # from tensorflow.python import debug as tf_debug

        grad_buffer, grad_buffer_init, grad_buffer_goal, grad_buffer_car = self.create_gradient_buffer(sess)

        print((settings.load_weights))
        if "model.ckpt" in settings.load_weights:
            self.load_net_weights(saver, sess, settings, latest=False)  # , exact=True)
        else:
            self.load_net_weights(saver, sess, settings, latest=True)
        self.create_setting_folder(settings)


        agent = self.get_agent(grad_buffer, net, grad_buffer_init, init_net, grad_buffer_goal, goal_net, settings)
        if settings.useRLToyCar:
            car = CarAgent(settings, car_net, grad_buffer_car)
        else:
            car = None


        writer = None
        log_file = None

        self.start_run(agent, car, counter, filecounter, grad_buffer, images_path, log_file, saver, sess, settings,
                       supervised, tmp_net, writer)



def main(setting):
    f=None
    if setting.timing:
        f=open('times.txt', 'w')
        start = time.time()
    rl=RL_manual_no_tf()

    if not f is None:
        end=time.time()
        f.write(str(end-start)+ " Setting up RL\n")
    rl.run(setting, time_file=f)

if __name__ == "__main__":

    setup=run_settings()

    np.random.seed(setup.random_seed_np)
    main(setup)

import sys

import numpy as np

#from RL.agent import SimplifiedAgent

from RL.agent_net import NetAgent

class ManualAgent(NetAgent):
    def __init__(self, settings, net=None, grad_buffer=None ,init_net=None, init_grad_buffer=None, goal_net=None, grad_buffer_goal=None):
        super(ManualAgent, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer, goal_net, grad_buffer_goal)
        self.position=np.zeros(3, dtype=np.float)
        self.frame=0

    def init_agent(self, episode_in, training=True,  viz=False):
        return self.init_net.feed_forward(episode_in, self.current_init_frame + self.frame, training, 0)

    def next_action(self,episode_in, training=True,viz=False):
        value = self.get_action(episode_in)

        speed = float(eval(input("Next speed:")))

        self.update_episode(episode_in, speed, value)

        return episode_in.velocity[self.frame]

    def update_episode(self, episode_in, speed, value,velocity=None, probabilities=None):
        episode_in.probabilities[self.frame, value] = 1
        episode_in.velocity[self.frame] = np.array(episode_in.actions[value])
        episode_in.action[self.frame] = value
        if self.settings.velocity:
            if np.linalg.norm(episode_in.velocity[self.frame]) != 1 and np.linalg.norm(
                    episode_in.velocity[self.frame]) > 0.1:
                episode_in.velocity[self.frame] = episode_in.velocity[self.frame] / np.linalg.norm(
                    episode_in.velocity[self.frame])
            if not self.settings.acceleration:
                print("Velocity "+str(episode_in.velocity[self.frame])+" speed: "+str(speed))
                episode_in.velocity[self.frame] = episode_in.velocity[self.frame] * speed * 5 / episode_in.frame_rate
        elif self.settings.controller_len > 0:
            episode_in.velocity[self.frame] = np.array(episode_in.actions[value]) * (speed + 1)

    def get_action(self, episode_in):
        if np.linalg.norm(episode_in.actions[0]) == 0:
            value = eval(input(
                "Next action: [1:'stand',2:'down',3:'downL'4:'left',5:'upL',6:'up', 7:'upR', 8:'right', 9:'downR', 0: stop excecution] "))
        else:
            value = eval(input(
                "Next action: [ 1:'downL', 2:'down', 3:'downR', 4:'left', 5:'stand', 6:'right',7:'upL', 8:'up', 9:'upR', 0: stop excecution] "))
        value = int(value) - 1
        if value == -1:
            sys.exit(0)
        return value

    def train(self,ep_itr,  statistics,episode, filename, filename_weights, poses):
        return statistics

    def init_net_train(self,ep_itr,  statistics, episode, filename, filename_weights, poses,priors, initialization_car):
        return statistics

    def set_session(self, session):
        self.net.sess = session

    def evaluate(self,ep_itr,  statistics, episode, poses):
        return statistics
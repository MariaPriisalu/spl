from .agent import SimplifiedAgent
from .episode import SimpleEpisode
import numpy as np

class TestAgent(SimplifiedAgent):
    def __init__(self):
        super(SimplifiedAgent, self).__init()

    def next_action(self, state_in):
        return [0,0,1]

    def train(self,ep_itr,statistics, episode):
        pass

class TestEpisode(SimpleEpisode):
    # def __init__(self, tensor, people_e, cars_e, pos_x, pos_y, gamma, init_on_pavement):
    #     super(TestEpisode, self).__init__(tensor, people_e, cars_e, pos_x, pos_y,gamma, init_on_pavement)

    def initial_position(self, poses_db):
        self.agent[0]=np.array([0,0,0])


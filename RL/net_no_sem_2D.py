from net_2d import SimpleSoftMaxNet_2D
import numpy as np
from settings import CHANNELS
from dotmap import DotMap

class NetRGB_2d(SimpleSoftMaxNet_2D):

    def __init__(self, settings):
        self.settings = settings
        self.set_nbr_channels()
        super(NetRGB_2d, self).__init__(settings)

    def set_nbr_channels(self):
        self.nbr_channels = CHANNELS.rgb[1]

    def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):
        tensor, people, cars,people_cv, cars_cv = episode_in.get_agent_neighbourhood(agent_pos_cur, [self.settings.agent_s[1], self.settings.agent_s[2]], frame_in)
        sem = np.zeros((self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))
        for x in range(sem.shape[0]):
            for y in range(sem.shape[1]):

                sem[x, y, CHANNELS.rgb[0]] = tensor[x, y, CHANNELS.rgb[0]]
                sem[x, y,CHANNELS.rgb[0]+1] = tensor[x, y, CHANNELS.rgb[0]+1]
                sem[x, y, CHANNELS.rgb[0]+2] = tensor[x, y, CHANNELS.rgb[0]+2]
        return np.expand_dims(sem, axis=0)


    def reset_mem(self):
        pass

class NoSem_2d(SimpleSoftMaxNet_2D):
    def __init__(self, settings):
        self.settings = settings
        self.set_nbr_channels()
        super(NoSem_2d, self).__init__(settings)
        self.channels = DotMap()
        self.channels.rgb = [0, 3]
        self.channels.pedestrian_trajectory = 3
        self.channels.cars_trajectory = 4
        self.channels.pedestrians = 5
        self.channels.cars = 6
        self.channels.semantic = 7

    def set_nbr_channels(self):
        self.nbr_channels = 7

    def get_input(self, episode_in, agent_pos_cur, frame_in=-1,training=True):
        tensor, people, cars, people_cv, cars_cv = episode_in.get_agent_neighbourhood(agent_pos_cur, [self.settings.agent_s[1], self.settings.agent_s[2]], frame_in)
        sem = np.zeros((self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))
        if self.settings.old:
            for x in range(sem.shape[0]):
                for y in range(sem.shape[1]):
                    # z = self.return_highest_col(tensor[:, x, y, :])
                    sem[x, y, self.channels.rgb[0]] = np.max(tensor[:, x, y, CHANNELS.rgb[0]])
                    sem[x, y, self.channels.rgb[0]+1] = np.max(tensor[:, x, y,CHANNELS.rgb[0]+ 1])
                    sem[x, y, self.channels.rgb[0]+2] = np.max(tensor[:, x, y, CHANNELS.rgb[0]+2])
                    if self.settings.predict_future:
                        if training:
                            sem[x, y, self.channels.pedestrian_trajectory] = np.max(tensor[:,x, y, CHANNELS.pedestrian_trajectory])
                            sem[x, y, self.channels.cars_trajectory] = np.max(tensor[:,x, y,CHANNELS.cars_trajectory])
                        else:

                            sem[x, y, self.channels.pedestrian_trajectory] = np.max(people_cv[:,x, y, 0])
                            sem[x, y, self.channels.cars_trajectory] = np.max(cars_cv[:,x, y, 0])
                    elif training or self.settings.temporal:
                        sem[x, y, self.channels.pedestrian_trajectory] = self.traj_forget_rate * np.max(tensor[:,x, y, CHANNELS.pedestrian_trajectory])
                        sem[x, y, self.channels.cars_trajectory] = self.traj_forget_rate * np.max(tensor[:,x, y, CHANNELS.cars_trajectory])
                    sem[x, y, self.channels.pedestrians] = np.max(people[:,x, y, 0])
                    sem[x, y, self.channels.cars] = np.max(cars[:,x, y, 0])
                    # sem[x, y, 5] = np.max(people[:, x, y, 0])
                    # sem[x, y, 6] = np.max(cars[:, x, y, 0])
        else:
            for x in range(sem.shape[0]):
                for y in range(sem.shape[1]):
                    #z = self.return_highest_col(tensor[:, x, y, :])
                    sem[x, y, self.channels.rgb[0]] = tensor[ x, y, CHANNELS.rgb[0]]
                    sem[x, y, self.channels.rgb[0]+1] = tensor[ x, y, CHANNELS.rgb[0]+1]
                    sem[x, y, self.channels.rgb[0]+2] = tensor[ x, y, CHANNELS.rgb[0]+2]
                    if self.settings.predict_future:
                        if training:
                            sem[x, y, self.channels.pedestrian_trajectory] =tensor[ x, y, CHANNELS.pedestrian_trajectory]
                            sem[x, y, self.channels.cars_trajectory] =tensor[  x, y, CHANNELS.cars_trajectory]
                        else:

                            sem[x, y, self.channels.pedestrian_trajectory] = people_cv[ x, y, 0]
                            sem[x, y, self.channels.cars_trajectory] = cars_cv[ x, y,0]
                    elif training or self.settings.temporal:
                        sem[x, y, self.channels.pedestrian_trajectory] = self.traj_forget_rate * tensor[x, y, CHANNELS.pedestrian_trajectory]
                        sem[x, y, self.channels.cars_trajectory] = self.traj_forget_rate * tensor[ x, y, CHANNELS.cars_trajectory]
                    sem[x, y, self.channels.pedestrians] = people[x, y, 0]
                    sem[x, y, self.channels.cars] = cars[x, y, 0]
                    # sem[x, y, 5] = np.max(people[:, x, y, 0])
                    # sem[x, y, 6] = np.max(cars[:, x, y, 0])
        return np.expand_dims(sem, axis=0)


    def reset_mem(self):
        pass

class Only_Sem_2d(SimpleSoftMaxNet_2D):
    def __init__(self, settings):
        self.settings = settings
        self.set_nbr_channels()
        super(Only_Sem_2d, self).__init__(settings)

    def set_nbr_channels(self):
        self.nbr_channels = 31

    def get_input(self, episode_in, agent_pos_cur, frame_in=-1,training=True):
        tensor, _,_, people_cv, cars_cv= episode_in.get_agent_neighbourhood(agent_pos_cur, [self.settings.agent_s[1], self.settings.agent_s[2]], frame_in)
        sem = np.zeros((self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))
        segmentation = (tensor[ :, :, CHANNELS.semantic] * 33.0).astype(np.int)
        for x in range(sem.shape[0]):
            for y in range(sem.shape[1]):
                sem[x, y, segmentation[ x, y]] = 1
        return np.expand_dims(sem, axis=0)


    def reset_mem(self):
        pass
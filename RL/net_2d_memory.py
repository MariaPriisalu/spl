from net_2d import SimpleSoftMaxNet_2D
import numpy as np
from settings import NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, CHANNELS


class Net_2d_mem(SimpleSoftMaxNet_2D):
    def __init__(self, settings):
        self.settings = settings
        self.set_nbr_channels()
        super(Net_2d_mem, self).__init__(settings)
        self.memory=np.zeros((self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))

    def set_nbr_channels(self):
        self.nbr_channels_t = 6
        self.nbr_channels = self.nbr_channels_t * self.settings.nbr_timesteps

    # get_input(self, episode_in, agent_pos_cur):
    def get_input(self, episode_in, agent_pos_cur, frame_in=-1,training=True):
        img_from_above = super(Net_2d_mem, self).get_input( episode_in, agent_pos_cur, frame_in,training)[0,:,:,:]
        self.memory[:,:,0:-self.nbr_channels_t]=np.copy(self.memory[:,:,self.nbr_channels_t:self.nbr_channels ])
        self.memory[:,:,-self.nbr_channels_t:]=img_from_above[:,:, 0:self.nbr_channels_t]
        return np.expand_dims(self.memory, axis=0)

    def reset_mem(self):
        self.memory = np.zeros((self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))



from net_sem_2d import  Seg_2d
class Seg_2d_mem(Seg_2d):
    def __init__(self, settings):
        self.settings = settings
        self.set_nbr_channels()
        super(Seg_2d_mem, self).__init__(settings)
        self.memory = np.zeros((self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))

    def set_nbr_channels(self):
        self.nbr_channels_t = NUM_SEM_CLASSES_ASINT + 2
        self.nbr_channels = self.nbr_channels_t * self.settings.nbr_timesteps


    def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):
        img_from_above = super(Seg_2d_mem, self).get_input(episode_in, agent_pos_cur, frame_in,training)[0,:,:,:]
        self.memory[:, :, 0:-self.nbr_channels_t] = np.copy(self.memory[:, :, self.nbr_channels_t:self.nbr_channels])
        self.memory[:, :, -self.nbr_channels_t:] = np.copy(img_from_above[:,:, 0:self.nbr_channels_t])
        return np.expand_dims(self.memory, axis=0)

    def reset_mem(self):
        self.memory = np.zeros((self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))


from net_2d import SimpleSoftMaxNet_2D
import numpy as np

class Seg_mem_2d(SimpleSoftMaxNet_2D):
    def __init__(self, settings):
        self.settings=settings
        #super(Seg_mem_2d, self).__init__(settings)
        self.set_nbr_channels()
        super(Seg_mem_2d, self).__init__(settings)
        self.memory = np.zeros((self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))



    def set_nbr_channels(self):
        self.nbr_channels = (NUM_SEM_CLASSES_ASINT + 2)* self.settings.nbr_timesteps

    # get_input(self, episode_in, agent_pos_cur):
    def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):
        tensor ,_,_,_,_= episode_in.get_agent_neighbourhood(agent_pos_cur, self.settings.agent_agent_s, frame_in=frame_in)

        sem = np.zeros((self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels/self.settings.nbr_timesteps))


        segmentation = (tensor[:, :, :, CHANNELS.semantic] * NUM_SEM_CLASSES).astype(np.int) + 2
        for x in range(sem.shape[0]):
            for y in range(sem.shape[1]):
                if training:
                    sem[x, y, 0] = np.max(tensor[:, x, y, CHANNELS.pedestrian_trajectory])* self.traj_forget_rate
                    sem[x, y, 1] = np.max(tensor[:, x, y, CHANNELS.cars_trajectory])* self.traj_forget_rate
                sem[x, y, segmentation[:, x, y]] = 1

        # if frame_in>0:
        #     cars_intercept=episode_in.get_intercepting_cars(agent_pos_cur, self.settings.net_size, frame_in)
        #     for car in cars_intercept:
        #         sem[car[2]:car[3],car[4]:car[5],28]=np.ones(sem[car[2]:car[3],car[4]:car[5],28].shape)

        self.memory[:, :, 0:-self.nbr_channels] = np.copy(self.memory[:, :, self.nbr_channels:self.settings.nbr_timesteps * self.nbr_channels])
        self.memory[:, :, -self.nbr_channels:] = sem

        return np.expand_dims(self.memory, axis=0)




    def reset_mem(self):
        self.memory = np.zeros((self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))

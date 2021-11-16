from net_2d import SimpleSoftMaxNet_2D, Net_2d, ContinousNet_2D, ContinousNet_2D_angular
import numpy as np
from supervised_net import SupervisedNet
from nets_abstract import SoftMaxNet
from net_continous import ContinousMemNet
from commonUtils.ReconstructionUtils import LAST_CITYSCAPES_SEMLABEL, cityscapes_labels_dict
from settings import NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, CHANNELS

from dotmap import DotMap
class Seg_2d(Net_2d):
    def __init__(self, settings):
        self.channels=DotMap()
        self.channels.rgb=[0,3]
        self.channels.pedestrian_trajectory = 3
        self.channels.cars_trajectory = 4
        self.channels.pedestrians = 5
        self.channels.cars = 6
        self.channels.semantic=7
        super(Seg_2d, self).__init__(settings)

    def set_nbr_channels(self):
        self.nbr_channels = NUM_SEM_CLASSES_ASINT+self.channels.semantic

    def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):

        tensor, people, cars, people_cv, cars_cv = episode_in.get_agent_neighbourhood(agent_pos_cur, [self.settings.agent_s[1], self.settings.agent_s[2]],frame_in)
        sem = np.zeros((self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))
        segmentation = (tensor[ :, :, CHANNELS.semantic] * NUM_SEM_CLASSES).astype(np.int)
        for x in range(sem.shape[0]):
            for y in range(sem.shape[1]):

                sem[x, y, self.channels.rgb[0]] = tensor[ x, y, CHANNELS.rgb[0]]
                sem[x, y, self.channels.rgb[0]+1] = tensor[ x, y, CHANNELS.rgb[0]+1]
                sem[x, y, self.channels.rgb[0]+2] = tensor[ x, y, CHANNELS.rgb[0]+2]
                if self.settings.predict_future:
                    # if training:
                    #     sem[x, y, 3] =tensor[self.return_highest_object(tensor[:,  x, y, 4]) ,x, y, 4]
                    #     sem[x, y, 4] =tensor[self.return_highest_object(tensor[:,  x, y, 5]) ,x, y, 5]
                    # else:

                    sem[x, y, self.channels.pedestrian_trajectory] = people_cv[ x, y, 0]
                    sem[x, y, self.channels.cars_trajectory] = cars_cv[ x, y,0]
                elif training or self.settings.temporal:
                    sem[x, y, self.channels.pedestrian_trajectory] = self.traj_forget_rate * tensor[x, y, CHANNELS.pedestrian_trajectory]
                    sem[x, y, self.channels.cars_trajectory] = self.traj_forget_rate * tensor[x, y, CHANNELS.cars_trajectory]

                sem[x, y, self.channels.pedestrians] = np.max(people[ x, y, 0])
                sem[x, y,self.channels.cars] = np.max(cars[ x, y, 0])

                if segmentation[x, y]>0:
                    sem[x, y, segmentation[x, y] + self.channels.semantic] = 1


        return np.expand_dims(sem, axis=0)

    # def return_highest_object(self, tens):
    #     z = 0
    #     while (np.linalg.norm(tens[z]) and z < tens.shape[0]) - 1:
    #         z = z + 1
    #     return z
    # def return_highest_col(self, tens):
    #     z = 0
    #     while (np.linalg.norm(tens[z,:]) and z < tens.shape[0]) - 1:
    #         z = z + 1
    #     return z

    def reset_mem(self):
        pass
class Seg_2d_softmax(SoftMaxNet, Seg_2d):
    def __init__(self, settings):
        super(Seg_2d_softmax, self).__init__(settings)

class Seg_2d_cont(ContinousNet_2D, Seg_2d):
    def __init__(self, settings):
        super(Seg_2d_cont, self).__init__(settings)
class Seg_2d_cont_angular(ContinousNet_2D_angular, Seg_2d):
    def __init__(self, settings):
        super(Seg_2d_cont_angular, self).__init__(settings)

# Network with fewer semantic classes
 # label_mapping[0] = 0 # None
 #    label_mapping[1] = 11 # Building
 #    label_mapping[2] = 13 # Fence
 #    label_mapping[3] = 4 # Other/Static
 #    label_mapping[4] = 24 # Pedestrian
 #    label_mapping[5] = 17 # Pole
 #    label_mapping[6] = 7 # RoadLines
 #    label_mapping[7] = 7 # Road
 #    label_mapping[8] = 8 # Sidewlk
 #    label_mapping[9] = 21 # Vegetation
 #    label_mapping[10] = 26 # Vehicles
 #    label_mapping[11] = 12 # Wall
 #    label_mapping[12] = 20 # Traffic sign
class Seg_2d_min(Net_2d):
    def __init__(self, settings):
        self.channels = DotMap()
        self.channels.rgb = [0, 3]
        self.channels.pedestrian_trajectory = 3
        self.channels.cars_trajectory = 4
        self.channels.pedestrians = 5
        self.channels.cars = 6
        self.channels.semantic = 7

        self.semantic_channels = DotMap()
        self.labels_indx={}
        # Local sem mapping
        if settings.new_carla_net:
            self.semantic_channels.building = 0
            self.semantic_channels.fence = 1
            self.semantic_channels.static = 2
            self.semantic_channels.vegetation = 3
            self.semantic_channels.traffic_sign = 4
            self.semantic_channels.traffic_light = 5
            self.semantic_channels.road = 6
            self.semantic_channels.terrain = 7
            self.semantic_channels.crosswalk = 8
            self.semantic_channels.sidewalk = 9
            self.semantic_channels.sidewalk_extra = 10

            #
            # Buildings and large structures
            self.labels_indx[cityscapes_labels_dict['building']] = self.semantic_channels.building
            self.labels_indx[cityscapes_labels_dict['bridge']] = self.semantic_channels.building
            self.labels_indx[cityscapes_labels_dict['tunnel']] = self.semantic_channels.building

            # Fences and guard and individual walls
            self.labels_indx[cityscapes_labels_dict['fence']] = self.semantic_channels.fence
            self.labels_indx[cityscapes_labels_dict['guard rail']] = self.semantic_channels.fence
            self.labels_indx[cityscapes_labels_dict['wall']] = self.semantic_channels.fence

            # All non-moving obstacles
            self.labels_indx[cityscapes_labels_dict['static']] = self.semantic_channels.static
            self.labels_indx[cityscapes_labels_dict['dynamic']] = self.semantic_channels.static
            self.labels_indx[cityscapes_labels_dict['pole']] = self.semantic_channels.static
            self.labels_indx[cityscapes_labels_dict['polegroup']] = self.semantic_channels.static

            # Vegetation
            self.labels_indx[cityscapes_labels_dict['vegetation']] = self.semantic_channels.vegetation

            # Traffic sign
            self.labels_indx[cityscapes_labels_dict['traffic sign']] = self.semantic_channels.traffic_sign

            # Traffic light
            self.labels_indx[cityscapes_labels_dict['traffic light']] = self.semantic_channels.traffic_light

            # Road, areas where car drives
            self.labels_indx[cityscapes_labels_dict['ground']] = self.semantic_channels.road
            self.labels_indx[cityscapes_labels_dict['road']] = self.semantic_channels.road
            self.labels_indx[cityscapes_labels_dict['parking']] = self.semantic_channels.road

            # Terrain & rail track
            self.labels_indx[cityscapes_labels_dict['terrain']] = self.semantic_channels.terrain
            self.labels_indx[cityscapes_labels_dict['rail track']] = self.semantic_channels.terrain

            # Crosswalk
            self.labels_indx[cityscapes_labels_dict['crosswalk']] = self.semantic_channels.crosswalk

            # Sidewalk
            self.labels_indx[cityscapes_labels_dict['sidewalk']] = self.semantic_channels.sidewalk

            # Sidewalk extra
            self.labels_indx[cityscapes_labels_dict['sidewalk_extra']] = self.semantic_channels.sidewalk_extra


        else:
            self.semantic_channels.building = 0
            self.semantic_channels.fence = 1
            self.semantic_channels.static = 2
            self.semantic_channels.pole = 3
            self.semantic_channels.road = 4
            self.semantic_channels.sidewalk = 5
            self.semantic_channels.vegetation = 6
            self.semantic_channels.wall = 7
            self.semantic_channels.traffic_sign = 8

            self.labels_indx[cityscapes_labels_dict['building'] ]=self.semantic_channels.building
            self.labels_indx[cityscapes_labels_dict['fence']] = self.semantic_channels.fence
            self.labels_indx[cityscapes_labels_dict['static']] = self.semantic_channels.static
            self.labels_indx[cityscapes_labels_dict['bridge']] = self.semantic_channels.static
            self.labels_indx[cityscapes_labels_dict['tunnel']] = self.semantic_channels.static
            self.labels_indx[cityscapes_labels_dict['dynamic']] = self.semantic_channels.static
            self.labels_indx[cityscapes_labels_dict['guard rail']] = self.semantic_channels.static
            self.labels_indx[cityscapes_labels_dict['pole']] = self.semantic_channels.pole
            self.labels_indx[cityscapes_labels_dict['polegroup']] = self.semantic_channels.pole
            self.labels_indx[cityscapes_labels_dict['ground']] = self.semantic_channels.road
            self.labels_indx[cityscapes_labels_dict['road']] = self.semantic_channels.road
            self.labels_indx[cityscapes_labels_dict['parking']] = self.semantic_channels.road
            self.labels_indx[cityscapes_labels_dict['rail track']] = self.semantic_channels.road
            self.labels_indx[cityscapes_labels_dict['sidewalk']] = self.semantic_channels.sidewalk
            self.labels_indx[cityscapes_labels_dict['vegetation']] = self.semantic_channels.vegetation
            self.labels_indx[cityscapes_labels_dict['terrain']] = self.semantic_channels.vegetation
            self.labels_indx[cityscapes_labels_dict['wall']] = self.semantic_channels.wall
            self.labels_indx[cityscapes_labels_dict['traffic light']] = self.semantic_channels.traffic_sign
            self.labels_indx[cityscapes_labels_dict['traffic sign']] = self.semantic_channels.traffic_sign

        self.carla=settings.carla

        self.channels = DotMap()
        self.channels.rgb = [0, 3]
        self.channels.pedestrian_trajectory = 3
        self.channels.cars_trajectory = 4
        self.channels.pedestrians = 5
        self.channels.cars = 6
        self.channels.semantic = 7
        self.set_nbr_channels()
        super(Seg_2d_min, self).__init__(settings)

    def set_nbr_channels(self):

        self.nbr_channels = max(self.labels_indx.values())+1+self.channels.semantic # 10+7 with crosswalks

    def get_input(self, episode_in, agent_pos_cur, frame_in=-1, training=True):

        tensor, people, cars, people_cv, cars_cv  = episode_in.get_agent_neighbourhood(agent_pos_cur, self.settings.agent_s, frame_in)
        #print ("Model input cars "+str(np.sum(abs(cars_cv)))+" people "+str(np.sum(abs(people_cv))))
        #print "Forgetting traj "+str(self.traj_forget_rate )
        if self.settings.old:

            sem = np.zeros((self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))

            segmentation = (tensor[:, :, :, CHANNELS.semantic] * NUM_SEM_CLASSES).astype(np.int)
            for x in range(sem.shape[0]):
                for y in range(sem.shape[1]):
                    sem[x, y, self.channels.rgb[0]] = np.max(tensor[:, x, y, CHANNELS.rgb[0]])
                    sem[x, y, self.channels.rgb[0]+1] = np.max(tensor[:, x, y, CHANNELS.rgb[0]+1])
                    sem[x, y, self.channels.rgb[0]+2] = np.max(tensor[:, x, y, CHANNELS.rgb[0]+2])
                    if training:
                        sem[x, y, self.channels.pedestrian_trajectory] = self.traj_forget_rate * np.max(tensor[:, x, y, CHANNELS.pedestrian_trajectory])
                        sem[x, y, self.channels.cars_trajectory] = self.traj_forget_rate * np.max(tensor[:, x, y, CHANNELS.pedestrian_trajectory])

                    sem[x, y, self.channels.pedestrians] = np.max(people[:, x, y, 0])
                    sem[x, y, self.channels.cars] = np.max(cars[:, x, y, 0])
                    for label in segmentation[:, x, y]:
                        if segmentation[ x, y]>cityscapes_labels_dict['unlabeled'] and segmentation[ x, y]!=cityscapes_labels_dict['sky']:
                            sem[x, y, self.labels_indx[label] + self.channels.semantic] = 1
        else:
            sem = np.zeros(( self.settings.net_size[1], self.settings.net_size[2], self.nbr_channels))
            segmentation = (tensor[ :, :, CHANNELS.semantic] * NUM_SEM_CLASSES).astype(np.int)
            #print (" Input shape "+str(sem.shape)+" people_cv shape "+str(people_cv.shape)+" tensor shape "+str(tensor.shape)+" people shape "+str(people.shape))
            for x in range(sem.shape[0]):
                for y in range(sem.shape[1]):

                    sem[x, y, self.channels.rgb[0]] = tensor[ x, y, CHANNELS.rgb[0]]
                    sem[x, y, self.channels.rgb[0]+1] = tensor[ x, y, CHANNELS.rgb[0]+1]
                    sem[x, y, self.channels.rgb[0]+2] = tensor[ x, y, CHANNELS.rgb[0]+2]
                    if self.settings.predict_future:
                        sem[x, y, self.channels.pedestrian_trajectory] = people_cv[ x, y,0]
                        sem[x, y, self.channels.cars_trajectory] = cars_cv[ x, y,0]

                    elif training or self.settings.temporal:

                        sem[x, y, self.channels.pedestrian_trajectory] = self.traj_forget_rate *tensor[x, y, CHANNELS.pedestrian_trajectory]
                        sem[x, y, self.channels.cars_trajectory] = self.traj_forget_rate *tensor[ x, y, CHANNELS.pedestrian_trajectory]

                    sem[x, y, self.channels.pedestrians] = people[ x, y, 0]
                    sem[x, y,self.channels.cars] = cars[ x, y, 0]

                    if segmentation[ x, y]>cityscapes_labels_dict['unlabeled'] and segmentation[ x, y]!=cityscapes_labels_dict['sky']:
                        if segmentation[ x, y] in self.labels_indx:
                            sem[x, y,self.labels_indx[segmentation[ x, y]] + self.channels.semantic] = 1
        #print (" Channel " + str(self.channels.cars_trajectory)+" sum "+str(np.sum(sem[:, :, self.channels.cars_trajectory])))

        return np.expand_dims(sem, axis=0)


class Seg_2d_min_softmax(SimpleSoftMaxNet_2D, Seg_2d_min):
    def __init__(self, settings):
        super(Seg_2d_min_softmax, self).__init__(settings)

class Seg_2d_min_cont(ContinousNet_2D, Seg_2d_min):
    def __init__(self, settings):
        super(Seg_2d_min_cont, self).__init__(settings)

class Seg_2d_min_cont_angular(ContinousNet_2D_angular, Seg_2d_min):
    def __init__(self, settings):
        super(ContinousNet_2D_angular, self).__init__(settings)

class SupervisedNet_2D(SupervisedNet, Seg_2d_min):
    def __init__(self, settings):
        super(SupervisedNet_2D, self).__init__(settings)

class ContinousMemNet_2D(ContinousMemNet,Seg_2d_min):
    def __init__(self, settings):
        super(ContinousMemNet_2D, self).__init__(settings)
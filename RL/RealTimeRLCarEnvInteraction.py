from  commonUtils.RealTimeEnv.RealTimeNullEnvInteraction import NullRealTimeEnv
from settings import  CAR_MEASURES_INDX,CAR_REWARD_INDX,NUM_SEM_CLASSES

from dotmap import DotMap
import numpy as np
import copy
from commonUtils.ReconstructionUtils import ROAD_LABELS,SIDEWALK_LABELS, OBSTACLE_LABELS_NEW, OBSTACLE_LABELS, MOVING_OBSTACLE_LABELS

# Interaction test class for one paramtere toy car. Might include import to tensorflow so under RL to not mess with imports-
class RLCarRealTimeEnv(NullRealTimeEnv):
    def __init__(self, car, cars_dict_sample, cars_sample, init_frames, init_frames_cars, people_dict_sample,
                 people_sample, reconstruction, seq_len, car_dim, max_speed, car_input, set_up_done, car_goal_closer):
        #car,cars_dict_sample, cars_sample, init_frames, init_frames_cars, people_dict_sample, people_sample, tensor)

        # super(RLCarRealTimeEnv, self).__init__()
        self.car = car
        self.cars_dict = cars_dict_sample
        self.cars = cars_sample
        self.init_frames = init_frames
        self.people_dict = people_dict_sample
        self.people = people_sample
        self.init_frames_cars = init_frames_cars
        self.reconstruction=reconstruction
        if not set_up_done:
            self.first_frame = 50
        else:
            self.first_frame = 0

        self.people = self.people[self.first_frame:]

        self.cars = self.cars[self.first_frame:]

        self.seq_len=seq_len
        self.valid_positions=np.ones(self.reconstruction.shape[1:3]) # VALID POSITOONS FOR FIRST FRAME
        self.car_dim=car_dim
        self.car_input=car_input
        self.max_speed=max_speed
        self.car_goal_closer=car_goal_closer
        self.car_id=-1

        self.car_keys=[]
        self.valid_car_keys = []
        # for car_key in self.cars_dict.keys():
        #     print (" Key " + str(car_key) + " len " + str(len(self.cars_dict[car_key])))

        for pedestrian_key in self.people_dict.keys():
            if not set_up_done:
                diff = self.init_frames[pedestrian_key] - self.first_frame
                if diff < 0:
                    self.people_dict[pedestrian_key] = self.people_dict[pedestrian_key][-diff:]
                    self.init_frames[pedestrian_key] = 0
                else:
                    self.init_frames[pedestrian_key] = diff
            else:
                diff= self.init_frames[pedestrian_key]
            if diff<=0 and len(self.people_dict[pedestrian_key])>abs(diff):
                person_flat=self.people_dict[pedestrian_key][0].flatten()
                person_bbox = [0,0,0,0]
                for pos in range(2,6):
                    person_bbox[pos-2]=int(round(max(min(person_flat[pos], self.valid_positions.shape[(pos-2)//2]),0)))
                self.valid_positions[person_bbox[0]:person_bbox[1], person_bbox[2]:person_bbox[3]]=np.zeros_like(self.valid_positions[person_bbox[0]:person_bbox[1], person_bbox[2]:person_bbox[3]])


        for car_key in self.cars_dict.keys():
            if not set_up_done:
                diff = self.init_frames_cars[car_key] - self.first_frame
                if diff < 0:
                    self.cars_dict[car_key] = self.cars_dict[car_key][-diff:]
                    self.init_frames_cars[car_key] = 0
                else:
                    self.init_frames_cars[car_key] = diff
            else:
                diff=self.init_frames_cars[car_key]

            if diff<=0 and len(self.cars_dict[car_key])>abs(diff):
                person_flat=self.cars_dict[car_key][0]
                person_bbox = [0,0,0,0]
                for pos in range(2,6):
                    person_bbox[pos-2]=int(round(max(min(person_flat[pos], self.valid_positions.shape[(pos-2)//2]),0)))
                self.valid_positions[person_bbox[0]:person_bbox[1], person_bbox[2]:person_bbox[3]]=np.zeros_like(self.valid_positions[person_bbox[0]:person_bbox[1], person_bbox[2]:person_bbox[3]])
                if diff==0:
                    # print (" Append valid car "+str(car_key)+" diff "+str(diff )+" pso "+str(person_bbox))
                    self.valid_car_keys.append(car_key)
        self.car_vel_dict = {}
        for car_key in self.cars_dict.keys():
            # print (" Key "+str(car_key)+" len "+str(len(self.cars_dict[car_key])))
            if len(self.cars_dict[car_key])>=2:
                self.car_keys.append(car_key)
                car_current= self.cars_dict[car_key][0]
                car_next=self.cars_dict[car_key][1]
                diff=[[],[],[]]
                for i in range (len(car_current)):
                    diff[int(i/2)].append(car_next[i]-car_current[i])
                self.car_vel_dict[car_key] = np.mean(np.array(diff), axis=1)


        self.agent = []
        self.agent_velocity = []
        # print (" Before initialization : " + str(self.cars))
        reconstruction_semantic = (self.reconstruction[:, :, :, 3] * NUM_SEM_CLASSES).astype(np.int)
        self.road=np.zeros(self.reconstruction.shape[1:3])
        for label in ROAD_LABELS:
            self.road = np.logical_or(np.any(reconstruction_semantic == label, axis=0), self.road)
        # print(" Road " + str(np.where(self.road )))


    # This is indeed the get observation function
    def get_cars_and_people(self, frame):
        # episode.update_pedestrians_and_cars(observation.frame, observation.heroCarPos, observation.heroCarVel,
        #                                     observation.people_dict, observation.cars_dict, observation.people_vel_dict,
        #                                     observation.car_vel_dict, observation.measures, observation.reward,
        #                                     observation.probabilities)
        realTimeEnvObservation = DotMap()
        # print (" Get cars and people frame " + str(frame))
        if frame <0:
            frame=0
            realTimeEnvObservation.car_init_dir=self.car.init_dir
        else:
            realTimeEnvObservation.car_init_dir =[]
            realTimeEnvObservation.heroCarVel = copy.copy(self.car.velocities[frame - 1])
            realTimeEnvObservation.heroCarAction=copy.copy(self.car.action[frame - 1])
            realTimeEnvObservation.heroCarAngle = copy.copy(self.car.angle[frame - 1])
            # print(" Save in real time obs angle: " + str(realTimeEnvObservation.heroCarAngle) + " frame " + str(frame))
            #print(" Transfer " + str(self.car.action[frame - 1]))
        # print (" Get cars and people frame "+str(frame))
        # print (" Car init dir "+str(realTimeEnvObservation.car_init_dir))
        if self.car.frame!=frame:
            print ("Frames not in sync. Car frame "+str(self.car.frame)+ " frame here "+str(frame))

        realTimeEnvObservation.frame = copy.copy(frame)
        realTimeEnvObservation.heroCarPos = copy.copy(self.car.car[frame])
        realTimeEnvObservation.heroCarGoal = copy.copy(self.car.goal)



        realTimeEnvObservation.heroCarBBox=copy.copy(self.car.bbox[frame])
        # print(" Update car measures in REALTime frame"+str(max(frame - 1,0))+" distracted: "+str(self.car.measures[max(frame - 1,0), CAR_MEASURES_INDX.distracted]))
        realTimeEnvObservation.measures = copy.copy(self.car.measures[max(frame - 1,0), :])

        if self.car.reward[max(frame - 1,0)]==None:
            realTimeEnvObservation.reward =0
        else:
            realTimeEnvObservation.reward =copy.copy(self.car.reward[max(frame - 1,0)])
            # print(" Copy reward to RLRealTimeEnv "+str(realTimeEnvObservation.reward)+" frame "+str(max(frame - 1,0)))
        realTimeEnvObservation.probabilities = copy.copy(self.car.probabilities[max(frame - 1,0)])
        # print (" Get probabilities "+str(realTimeEnvObservation.probabilities)+" frame "+str(max(frame - 1,0)))
        # print ("Saved initial position ? " + str(self.car.car[0]) +" saved "+str(realTimeEnvObservation.heroCarPos))
        # print(" Goal car " + str(self.car.goal) +" saved "+str(realTimeEnvObservation.heroCarGoal))
        # print (" Car dir " + str(self.car.init_dir)+ " saved "+str(realTimeEnvObservation.heroCarVel))

        # Get cars ane people dict

        pedestrian_dict = {}
        pedestrian_vel_dict = {}
        for pedestrian_key in self.people_dict.keys():
            if frame < len(self.people_dict[pedestrian_key]):
                local_frame =frame - self.init_frames[pedestrian_key]
                if self.init_frames[pedestrian_key] <= frame and local_frame< len(self.people_dict[pedestrian_key]):

                    value=self.people_dict[pedestrian_key][local_frame]
                    pedestrian_dict[pedestrian_key] = value
                    pedestrian_vel_dict[pedestrian_key] = np.mean(
                        self.people_dict[pedestrian_key][min(local_frame+1,len(self.people_dict[pedestrian_key])-1)] -
                        self.people_dict[pedestrian_key][local_frame], axis=1)

        realTimeEnvObservation.people_dict = pedestrian_dict

        realTimeEnvObservation.pedestrian_vel_dict = pedestrian_vel_dict
        # print (" pedestrian_dict " + str(realTimeEnvObservation.pedestrian_vel_dict ))

        car_dict = {}
        car_vel_dict = {}
        for car_key in self.cars_dict.keys():
            if frame < len(self.cars_dict[car_key]):
                local_frame = frame - self.init_frames_cars[car_key]
                if self.init_frames_cars[car_key] <= frame and local_frame< len(self.cars_dict[car_key]):

                    car_current = self.cars_dict[car_key][local_frame]

                    car_next = self.cars_dict[car_key][min(local_frame+1, len(self.cars_dict[car_key])-1)]
                    diff = [[], [], []]
                    for i in range(len(car_current)):
                        diff[int(i / 2)].append(car_next[i] - car_current[i])

                    car_dict[car_key] = car_current
                    car_vel_dict[car_key] = np.mean(np.array(diff), axis=1)

        realTimeEnvObservation.cars_dict = car_dict
        realTimeEnvObservation.car_vel_dict = car_vel_dict



        return realTimeEnvObservation

    def reset(self, initDict):
        car, goal_car, car_dir=self.init_car(initDict.on_car)
        # print ("Car initial position "+str(car)+" goal car "+str(goal_car)+" car dir "+str(car_dir))
        self.car.initial_position(car, goal_car,  self.seq_len, init_dir=car_dir, car_id=self.car_id)
        # print ("Saved initial position ? " + str(self.car.car[0]) + " goal car " + str(self.car.goal) + " car dir " + str(self.car.init_dir))
        return self.get_cars_and_people(-1)

    def init_car(self, on_car=False):
        # Find positions where cars could be initialized. This should be prefferable done in CARLA.
        # Just choosing a  future car location as an initial spot
        # for key, value in self.cars_dict.items():
        #     print (" Key "+str(key)+" car "+str(value[0])+" vel "+str(self.car_vel_dict[key]))
        # print (" On car? RL Interaction Env "+str(on_car)+" valid cars? "+str(self.valid_car_keys))
        if len(self.car_keys)>0:
            if on_car and len(self.valid_car_keys)>0:
                if len(self.valid_car_keys)>1:
                    car_key=self.valid_car_keys[np.random.randint(len(self.valid_car_keys))]
                else:
                    car_key = self.valid_car_keys[0]
                self.car_id=car_key
                # print(" Choose car id  " + str(self.car_id) )
            else:
                self.car_id =-1
                all_keys= self.car_keys

                car_key = all_keys[np.random.randint(len(all_keys))]
            # while  len(self.cars_dict[car_key]) < 2:
            #     car_key = all_keys[np.random.randint(len(all_keys))]
            # print ("Chose key " + str(car_key))
            car=self.cars_dict[car_key][0]

            car_init_pos=np.array([np.mean(car[0:2]),np.mean(car[2:4]), np.mean(car[4:])])
            # print ("Car pos " + str(car_init_pos))
            car_vel=self.car_vel_dict[car_key]#[1]-self.cars_dict[car_key][0]
            if on_car and len(self.valid_car_keys) > 0:
                car_dir=car_vel
                goal_car=self.cars_dict[car_key][min(len(self.cars_dict[car_key])-1, self.seq_len+3)]
                car_goal_pos = np.array([np.mean(goal_car[0:2]), np.mean(goal_car[2:4]), np.mean(goal_car[4:])])
                print(" Init on car " + str(car_init_pos) + " goal " + str(goal_car)+" vel "+str(car_dir)+ " car "+str(car))
                # print (" Goal is frame "+str(min(len(self.cars_dict[car_key])-1, self.seq_len))+" pos: "+str(self.cars_dict[car_key][:min(len(self.cars_dict[car_key])-1, self.seq_len)+1]))
                return car_init_pos, car_goal_pos, car_dir

            min_speed=5*5/(17*3.6)#5km/h
            # print ("Car velocity " + str(car_vel) + " speed " + str(np.linalg.norm(car_vel[1:])))
            if np.linalg.norm(car_vel[1:])<min_speed**2:# 5km/h
                # print ("Readjust speed")
                max_speed=self.max_speed#*5/(17*3.6)# 60 km/h
                car_speed=np.random.rand(1)*(max_speed-min_speed)+min_speed
                if np.linalg.norm(car_vel[1:])< 1e-5:
                    car_vel =np.zeros_like(car_vel)
                    car_speed=0
                else:
                    car_vel =car_vel*(car_speed)/np.linalg.norm(car_vel)
            # print ("Car velocity "+str(car_vel)+" speed "+str(np.linalg.norm(car_vel[1:])))


            t=[]
            for i in range(2):
                if car_vel[i+1]>0:
                    #if abs(car_vel[i+1])>1e-5:
                    t.append((self.reconstruction.shape[i+1]-car_init_pos[i+1])/car_vel[i+1])
                    # print (" Append t " +str(t[-1])+ " diff "+str((self.reconstruction.shape[i+1]-car_init_pos[i+1]))+" vel "+str(car_vel[i+1]))
                else:
                    #if abs(car_vel[i+1])>1e-5:
                    t.append((-car_init_pos[i+1])/car_vel[i+1])
                    # print (" Append t " + str(t[-1]) + " diff " + str(-car_init_pos[i+1]) + " vel " + str(car_vel[i + 1]))

            t_max=min(t)
            t_min=np.max(self.car_dim[1:])*2/min_speed
            #print ("Car init: t_min "+str(t_min)+" t_max "+str(t_max))
            if t_max>t_min:
                time=np.random.rand()*(t_max-t_min)+t_min
                car=car_init_pos+ (time*car_vel)
                t_goal=t_max-time
                if self.car_goal_closer: #or self.car_input =="distance_to_car_and_pavement_intersection" :
                    t_goal=min(t_max-time,(np.random.rand()*0.75+0.25)*(self.seq_len-2) )
                    print (" Adapt car goal time, draw random number between "+str(0.25*(self.seq_len-2))+" and "+str((self.seq_len-2))+" "+str(t_goal))
                goal_car = car + (t_goal * car_vel)
                print (" Car goal "+str(goal_car)+" dist to car "+str(np.linalg.norm(goal_car[1:]-car[1:]))+" time "+str(t_goal)+" car pos "+str(car[1:])+" car dir "+str(car_vel))
                car_dir = car_vel
                return car, goal_car, car_dir
            else:
                print ("Issue! car temp is out!")

            # print (" Before initialization : " + str(self.cars))

            # print(" Road " + str(np.where(reconstruction_semantic== 6))+" Non-zero "+str(np.sum(reconstruction_semantic>0)))
            # print (reconstruction_semantic[reconstruction_semantic>0])
            valid_road=np.logical_and(self.valid_positions,self.road )
            positions = np.where(valid_road)
            indx = np.random.randint(len(positions[0]))
            car = np.array([0, positions[0][indx],
                   positions[1][indx]])

            dist = int(self.seq_len * 0.75*self.max_speed)
            # print (" Maximal distance to goal "+str(dist))
            limits = [int(max(car[1] - dist, 0)),
                      int(min(car[1] + dist, self.reconstruction.shape[1])),
                      int(max(car[2] - dist, 0)),
                      int(min(car[2] + dist, self.reconstruction.shape[2]))]
            # print ("limits "+str(limits))
            positions_goal = np.where(valid_road[limits[0]:limits[1], limits[2]:limits[3]])
            # print (" Number of possible goals "+str(len(positions_goal[0])))
            goal_indx= np.random.randint(len(positions_goal[0]))
            goal_car = np.array([0, limits[0] + positions_goal[0][goal_indx], limits[2] + positions_goal[1][goal_indx]])
            # print (" Car goal " + str(goal_car) + " dist to car " + str(np.linalg.norm(goal_car[1:] - car[1:]))+" less than? "+str(self.seq_len * 0.25*self.max_speed))
            i=0
            while np.linalg.norm(goal_car[1:] - car[1:])< self.seq_len * 0.25*self.max_speed and i<5:
                goal_indx = np.random.randint(len(positions_goal[0]))
                goal_car = [0, limits[0] + positions_goal[0][goal_indx], limits[2] + positions_goal[1][goal_indx]]
                i=i+1
                # print (" Car goal " + str(goal_car) + " dist to car " + str(np.linalg.norm(goal_car[1:] - car[1:]))+" less than? "+str(self.seq_len * 0.25*self.max_speed))

            car_dir=np.array(goal_car)-np.array(car)
            car_dir= car_dir*(1/self.seq_len)
            print (" Car goal " + str(goal_car) + " dist to car " + str(np.linalg.norm(goal_car[1:] - car[1:]))+" car pos "+str(car[1:])+" car dir "+str(car_dir))
        return car, goal_car, car_dir

    def action(self, actionDict, training):

        # self.car.measures[self.car.frame, CAR_MEASURES_INDX.inverse_dist_to_closest_car]=actionDict.inverse_dist_to_car
        # print (" Car observes distance to pedestrian "+str(self.car.measures[self.car.frame, CAR_MEASURES_INDX.inverse_dist_to_closest_car]))
        realTimeEnvObservation = DotMap()
        # print (" Update real time environment frame "+str(actionDict.frame)+" value "+str(actionDict.agentPos))
        if len(self.agent) <= actionDict.frame:
            self.agent.append(actionDict.agentPos)
        else:
            self.agent[actionDict.frame] = actionDict.agentPos

        if len(self.agent_velocity) <= actionDict.frame:
            self.agent_velocity.append(actionDict.agentVel)
        else:
            self.agent_velocity[actionDict.frame] = actionDict.agentVel
       # print("IN RL env after update pos "+str(self.agent)+" vel "+str(self.agent_velocity))
        self.init_dir=actionDict.init_dir
            # episode_in.agent, episode_in.car, episode_in.velocity, episode_in.velocity_car, agent_frame
        fake_episode = DotMap()
        fake_episode.agent = self.agent
        fake_episode.velocity = self.agent_velocity
        fake_episode.init_dir=self.init_dir
        fake_episode.car_goal=self.car.goal
        fake_episode.reconstruction = self.reconstruction
        fake_episode.people = self.people[0:self.car.frame + 2]
        fake_episode.cars = self.cars[0:self.car.frame + 2]
        if self.car_id>0 and len(self.cars_dict[self.car_id])>self.car.frame+1:
            # print ("Add all cars except for id car "+str(len(self.cars_dict[self.car_id]))+" "+str(self.car_id))
            fake_episode.cars =[]
            for frame in range(self.car.frame +2):
                fake_episode.cars.append([])
            for key in self.car_keys:
                init_frame=self.init_frames_cars[key]
                if key!=self.car_id and init_frame<self.car.frame +2:
                    # print("Add id "+str(key)+" in range "+str((init_frame, min(self.car.frame +2,len(self.cars_dict[key])))))
                    for frame in range(init_frame, min(self.car.frame +2,len(self.cars_dict[key]))):
                        fake_episode.cars[frame].append(self.cars_dict[key][frame-init_frame])
            # print("Cars in episode " + str(fake_episode.cars))
            # print("Compare to  " + str(self.cars[0:self.car.frame + 2]))
            fake_episode.supervised_car_vel=np.array(self.cars_dict[self.car_id][self.car.frame+1])-np.array(self.cars_dict[self.car_id][self.car.frame])
            fake_episode.supervised_car_vel=np.array([np.mean(fake_episode.supervised_car_vel[0:2]),np.mean(fake_episode.supervised_car_vel[2:4]),np.mean(fake_episode.supervised_car_vel[4:])])
            # print (" Car vel "+str(fake_episode.supervised_car_vel))
            if self.car.frame >0:
                fake_episode.supervised_car_vel_prev = np.array(self.cars_dict[self.car_id][self.car.frame]) - \
                                                  np.array(self.cars_dict[self.car_id][self.car.frame-1])
                fake_episode.supervised_car_vel_prev = np.array(
                    [np.mean(fake_episode.supervised_car_vel_prev[0:2]), np.mean(fake_episode.supervised_car_vel_prev[2:4]),
                     np.mean(fake_episode.supervised_car_vel_prev[4:])])

        else:
            # print (" use regular cars")
            fake_episode.supervised_car_vel =[]
        self.car.next_action(fake_episode, training=training)


        #print (" Before taking car action len of cars : "+str(len(fake_episode.cars)))

        # Now take action and evaluate measures
        # print (" Car velocity "+str(self.car.velocities[self.car.frame])+" frame "+str(self.car.frame)+" vector "+str(self.car.velocities))
        self.car.perform_action(self.car.velocities[self.car.frame], fake_episode)
        # print(" After performing action "+str(self.car.velocity)+" car position "+str(self.car.car[self.car.frame])+" frame "+str(self.car.frame))

        # Evaluate reward
        self.car.evaluate_car_reward(max(self.car.frame - 1, 0))

        # print("Content of car episode after take action ")
        # for key, value in fake_episode.items():
        #     if "reconstruction" in key:
        #         print ("reconstruction ")logica
        #         print (value.shape)
        #         # print ("Input cars " + str(np.sum(value[0, :, :, 6])) + " people " + str(
        #         #     np.sum(value[0, :, :, 5])) + " Input cars traj" + str(
        #         #     np.sum(value[0, :, :, 4])) + " people traj " + str(
        #         #     np.sum(value[0, :, :, 3])))
        #         # print ("Input building " + str(np.sum(value[0, :, :, 7])) + " fence " + str(
        #         #     np.sum(value[0, :, :, 7 + 1])) + " static " + str(np.sum(value[0, :, :, 7 + 2])) + " pole " + str(
        #         #     np.sum(value[0, :, :, 7 + 3])))
        #         # print ("Input sidewalk " + str(np.sum(value[0, :, :, 7 + 5])) + " road " + str(
        #         #     np.sum(value[0, :, :, 7 + 4])) + " veg. " + str(
        #         #     np.sum(value[0, :, :, 7 + 6])) + " wall " + str(np.sum(value[0, :, :, 7 + 7])) + " sign " + str(
        #         #     np.sum(value[0, :, :, 7 + 8])))
        #     elif "people" in key or "cars " in key:
        #         print (" Key " + str(key) + " value " + str(len(value)))
        #     else:
        #         print (" Key " + str(key) + " value " + str(value))
        return self.get_cars_and_people(self.car.frame )



#statistics, episode_in, filename, filename_weights,poses,priors,statistics_car, seq_len=-1):
    def train(self, ep_itr, statistics, episode, filename, filename_weights, poses, priors, statistics_car):
        self.car.train( ep_itr, statistics, episode, filename, filename_weights, poses, priors,
                        statistics_car)  # Shell of the other class

    def evaluate(self, ep_itr, statistics, episode, poses, priors, statistics_car):
        self.car.evaluate(ep_itr, statistics, episode, poses, priors, statistics_car)  # Shell of the other class
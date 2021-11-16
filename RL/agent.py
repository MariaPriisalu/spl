
import numpy as np
from settings import RANDOM_SEED
np.random.seed(RANDOM_SEED)
import copy
from settings import PEDESTRIAN_MEASURES_INDX

# Basic agent behaviour. Implement next_action to decide the velocity of the next action.
# train and evaluate are for neural-net evaluations.
class SimplifiedAgent(object):
    def __init__(self, settings,net=None, grad_buffer=None, init_net=None, init_grad_buffer=None,goal_net=None, grad_buffer_goal=None):
        self.position=np.zeros(3, dtype=np.float)
        self.frame=0 # current frame
        self.current_init_frame=0 # initialize from this frame
        self.pos_exact = np.zeros(3, dtype=float) # exact position of agent
        self.velocity=0
        self.acceleration=settings.acceleration # Is agent taking acceleration steps?
        self.settings=settings
        self.goal=[]
        self.PFNN = None
        self.distracted=False
        self.distracted_len=-1

    # Initializes agent on position pos, with a goal
    def initial_position(self, pos,goal, current_frame=0, vel=0, init_dir=[]):
        self.position=np.array(pos).astype(int)
        self.frame = 0
        self.current_init_frame = current_frame
        self.pos_exact=np.array(pos).astype(np.float)
        self.velocity=vel
        self.goal=goal
        self.distracted = False
        self.distracted_len = -1
        print(("Agent initial position "+str(self.position)))

    def is_distracted(self):
        add_noise=False
        # print (" Is agent distracted? " + str(self.distracted) + " distracted len " + str(self.distracted_len))
        if not self.distracted and self.settings.distracted_pedestrian:
            add_noise = np.random.binomial(1, self.settings.pedestrian_noise_probability)
            # print (" Add noise to car input? Draw random variable with prob " + str(
            #     self.settings.car_noise_probability) + " value " + str(add_noise))

            if add_noise:
                self.distracted=True
                self.distracted_len=np.random.poisson(self.settings.avg_distraction_length)
                # print (" Set agent distracted for "+str(self.distracted_len))
        elif self.distracted_len>0 and self.distracted:
            # print (" Agent is still distracted ")
            return True
        else:
            # print (" Agent is not distracted ")
            self.distracted=False
            return False

        return add_noise

    # Implement this!
    def next_action(self,  state_in, training):

        raise NotImplementedError("Please Implement this method")

    # Implement this! - for neural net, this is where gradient update is made
    def train(self, ep_itr, statistics, episode, filename, filename_weights, poses):
        raise NotImplementedError("Please Implement this method")

    # Implement this! - for neural net, this is where loss is evaluated
    def evaluate(self,ep_itr, statistics, episode, poses, priors):
        raise NotImplementedError("Please Implement this method")

    # Make agent stand still if next step results in collision otherwise take step.
    def perform_action(self, vel, episode, prob=0):

        exact = True
        # print ("vel before get planned " + str(vel))
        next_pos, step = self.get_planned_position(episode, exact, vel)

        hit_static_object, valid_movement = self.is_next_pos_valid(episode, next_pos)


        if valid_movement:
            next_pos = self.take_step(next_pos, step, episode)
            self.mark_valid_move_in_episode(episode)
        else:

            if not hit_static_object:
                self.mark_agent_hit_obstacle_in_episode(episode)
            next_pos, step = self.stand_still(next_pos, step, episode)
            self.mark_agent_dead_in_episode(episode)

        self.carry_out_step(next_pos, step, episode)
        self.frame +=1

        if self.distracted:

            self.distracted_len=self.distracted_len-1
            # print(" update distracted len " + str(self.distracted_len))

        self.update_agent_pos_in_episode(episode)

    def update_agent_pos_in_episode(self, episode):

        episode.agent[self.frame] = self.pos_exact

        episode.measures[self.frame-1, PEDESTRIAN_MEASURES_INDX.distracted] = self.distracted
        # print(" Episode save distracted "+str(episode.measures[self.frame-1, PEDESTRIAN_MEASURES_INDX.distracted])+" frame "+str(self.frame-1))
        episode.end_of_episode_measures(max(self.frame - 1, 0))

    def mark_agent_hit_obstacle_in_episode(self, episode):

        episode.measures[self.frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles] = 1

    def mark_agent_dead_in_episode(self, episode):

        episode.measures[self.frame, PEDESTRIAN_MEASURES_INDX.agent_dead] = 1

    def mark_valid_move_in_episode(self, episode):
        episode.measures[self.frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles] = 0

    # What is the next planned position given the velocity from next_action
    def get_planned_position(self, episode, exact, vel):

        next_pos = self.pos_exact + np.array(vel)

        step = np.array(vel)
        #print ("Planned pos"+str(next_pos)+" "+str(self.frame)+" previous "+str(self.pos_exact)+" diff "+str(next_pos-self.pos_exact)+" vel "+str(vel[1:]))
        if not exact:
            step=np.array(round(vel).astype(int))
            next_pos = self.pos_exact + np.array(round(vel).astype(int))

        return next_pos, step

    # Is the planned position a valid position, or will aent collide?
    def is_next_pos_valid(self, episode, next_pos):

        valid = episode.valid_position(next_pos)

        alive=self.is_agent_alive(episode)
        valid_case = valid and alive
        # print (" Is next pos valid agent " + str(valid)+" is alive "+str(alive)+" valid step "+str(valid_case)+" next pos "+str(next_pos))
        return valid, valid_case

    # Check if agent is alive
    def is_agent_alive(self, episode):
        # print("Hit by car "+str(PEDESTRIAN_MEASURES_INDX.hit_by_car))
        # print (" Hit by car "+str(episode.measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_by_car])+" reached goal: "+str(episode.measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.goal_reached])+" agent dead "+str(episode.measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.agent_dead] ))
        if self.settings.end_on_bit_by_pedestrians:
            if self.settings.stop_on_goal:
                return episode.measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_by_car] <= 0 and episode.measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.goal_reached] == 0 and \
                       episode.measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_pedestrians] <= 0
            return episode.measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_by_car] <= 0  and episode.measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_pedestrians] <= 0
        else:
            if self.settings.stop_on_goal:
                return episode.measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_by_car] <= 0 and episode.measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.goal_reached] == 0
            return episode.measures[max(0, self.frame - 1), PEDESTRIAN_MEASURES_INDX.hit_by_car] <= 0

    # How should agent stand still when walking into onjects.
    def stand_still(self,next_pos, vel, episode):
        # print ("Stand still")
        vel=np.zeros_like(vel)
        return next_pos, vel

    # result of taking a step
    def take_step(self, next_pos, vel, episode):
        #print ("Take step "+str(next_pos))
        self.pos_exact = next_pos
        self.position = np.round(self.pos_exact).astype(int)
        #print ("Pos exact  " + str(self.pos_exact))
        return next_pos

    # Pfnn agent does sometjing here.
    def carry_out_step(self,next_pos, step, episode):
        #print "Pass"
        pass

    # tf session
    def set_session(self, session):
        pass


# An agent that rotates the coordinate system at each step
class ContinousAgent(SimplifiedAgent):
    def __init__(self, settings, net=None, grad_buffer=None):
        super(ContinousAgent, self).__init__(settings, net, grad_buffer)
        self.velocity=np.zeros(3, dtype=np.float)
        self.direction=np.zeros(3, dtype=np.float)
        self.angle=0
        #self.rotation_matrix=np.identity(2)
        self.rotation_matrix_prev = np.identity(3)
        print("Continous agent--------------------------->")


    def initial_position(self, pos,goal, current_frame=0, vel=0, init_dir=[]):
        super(ContinousAgent, self).initial_position( pos,goal, current_frame, vel)
        self.rotation_matrix_prev = np.identity(3)
        if len(init_dir)>0:
            #print "Initial direction "+str(init_dir)
            self.direction=init_dir
            self.get_rotation_matrix(init_dir)
        else:
            self.angle = 0
            #self.rotation_matrix = np.identity(2)
            self.rotation_matrix_prev = np.identity(3)

    def get_rotation_matrix(self, init_dir, angle=0):
        #print "Angular input "+str(angle)+" dir "+str(init_dir)
        if self.frame % self.settings.action_freq==0:
            if angle==0 and np.linalg.norm(init_dir[1:])>1e-5:
                # d = np.sqrt(init_dir[1] ** 2 + init_dir[2] ** 2)
                # self.rotation_matrix_prev = np.identity(3)
                # self.rotation_matrix_prev[1, 1] = init_dir[1] / d
                # self.rotation_matrix_prev[1, 2] = -init_dir[2] / d
                # self.rotation_matrix_prev[2, 2] = init_dir[1] / d
                # self.rotation_matrix_prev[2, 1] = init_dir[2] / d
                self.angle=np.arctan2(init_dir[1] ,init_dir[2] )-(np.pi/2)
                #print " Arctan dir "+str(self.angle)
                #print np.arctan2(init_dir[1] ,init_dir[2] )/np.pi
            else:
                #print "Agent angle "+str(self.angle)+" "+str(angle)
                self.angle=self.angle+angle
                #print "After addition angle " + str(self.angle) + " " + str(angle)

            self.normalize_angle()

            self.rotation_matrix_prev[1, 1] = np.cos(self.angle)
            self.rotation_matrix_prev[1, 2] = np.sin(self.angle)
            self.rotation_matrix_prev[2, 2] = np.cos(self.angle)
            self.rotation_matrix_prev[2, 1] = -np.sin(self.angle)#


        # print  self.rotation_matrix_prev
        # print "Cos "+str(np.cos(self.angle))
        # print "Sin "+str(np.sin(self.angle))

        # print "Rotation around: "+str(init_dir)
        # print "Rotation matrix "
        # print str(self.rotation_matrix_prev)
        # print "Direction: "+str(np.matmul(self.rotation_matrix_prev, np.array([0,1,0])))

    def normalize_angle(self):
        if self.angle <= -np.pi:
            # print "Before addition " + str(self.angle) + " " + str(self.angle / np.pi)
            self.angle = 2 * np.pi + self.angle
            # print "After Rotation " + str(self.angle) + " " + str(self.angle / np.pi)
        if self.angle > np.pi:
            # print "Before minus " + str(self.angle) + " " + str(self.angle / np.pi)
            self.angle = self.angle - (2 * np.pi)
            # print "After Rotation " + str(self.angle)+" "+str(self.angle/np.pi)

    def get_planned_position(self, episode, exact, vel):

        vel_rotated=np.matmul(self.rotation_matrix_prev, np.array(vel))
        #print "Rotated direction step= "+str(vel_rotated)
        return super(ContinousAgent, self).get_planned_position( episode, exact, vel_rotated)

    def take_step(self, next_pos, vel, episode):
        next_pos=super(ContinousAgent, self).take_step(next_pos, vel, episode)
        self.get_rotation_matrix(vel, episode.action[self.frame])  # np.matmul(rotation_matrix,self.rotation_matrix_prev)
        if self.frame+1<len(episode.angle):
            episode.angle[self.frame+1] = copy.copy(self.angle)
        #print "Episode angle "+str(episode.angle[self.frame+1])+" "+str(self.frame+1)+" "+str(episode.angle[self.frame+1]/np.pi)
        return next_pos

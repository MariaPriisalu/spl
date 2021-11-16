from agent import SimplifiedAgent

import numpy as np

# An agent that always returns 1
class DummyAgent(SimplifiedAgent):

    def next_action(self, state_in, training):
        return 1

    def train(self,ep_itr,statistics, episode, filename, filename_weights, poses):
        return statistics

    def set_session(self, session):
        pass

    def evaluate(self,ep_itr, statistics, episod, poses):
        return statistics

    def cleanup(self):
        pass

# Agent taking Random actions
class RandomAgent(DummyAgent):
    def __init__(self, settings, net=None, grad_buffer=None):
        super(RandomAgent, self).__init__(settings, net, grad_buffer)

    def next_action(self,episode_in,goal, training=True):
        #print "Random action"
        value=np.random.randint(9)
        episode_in.probabilities[self.frame, 0:9] = 1/9.0*np.ones_like(episode_in.probabilities[self.frame, 0:9])
        episode_in.velocity[self.frame] = episode_in.actions[value]
        episode_in.action[self.frame] = value
        episode_in.speed[self.frame] = 1
        return episode_in.velocity[self.frame]


class PedestrianAgent(DummyAgent):
    def __init__(self,  settings, net=None, grad_buffer=None):
        super(PedestrianAgent, self).__init__( settings, net, grad_buffer)
        self.current_init_frame = 0
        self.pos_exact = np.zeros(2, dtype=float)

    def initial_position(self, pos,goal, current_frame=0,vel=0, init_dir=[]):
        self.position=np.array(np.array(pos)).astype(int)
        self.frame = 0
        self.current_init_frame =current_frame
        self.pos_exact=np.array(np.array(pos))#pos_exact
        #print "Agent init frame"+str(self.frame)+"  current frame "+str(self.current_frame)

    def perform_action(self, vel, episode, prob=0, person_id=-1, training=False):
        super(PedestrianAgent, self).perform_action( vel, episode, prob=0)
        #print "Agent frame "+str(self.frame)
        if self.frame<len(episode.valid_people_tracks[episode.goal_person_id_val]):
            self.pos_exact=np.mean(episode.valid_people_tracks[episode.goal_person_id_val][self.frame], axis=1)
        else:
            self.pos_exact = np.mean(episode.valid_people_tracks[episode.goal_person_id_val][-1], axis=1)
        #episode.get_valid_vel_n(agent_frame, episode_in.goal_person_id)



    def next_action(self, episode_in, training=True):
        vel = self.pos_exact - self.pos_exact
        if self.frame+1<len(episode_in.valid_people_tracks[episode_in.goal_person_id_val]):
            vel= np.mean(episode_in.valid_people_tracks[episode_in.goal_person_id_val][self.frame+1], axis=1)-self.pos_exact


        value = episode_in.find_action_to_direction([0,vel[1],vel[2]], np.sqrt(vel[1]**2+vel[1]**2))
        #print value
        episode_in.probabilities[self.frame, 0:9] =np.zeros(9)
        episode_in.probabilities[self.frame, int(value)] = 1
        episode_in.velocity[self.frame] = vel
        episode_in.action[self.frame] = value
        episode_in.speed[self.frame] = 1
        return vel
        #print "next action agent " + str(self.frame) + "  current frame " + str(self.current_frame+self.frame)
        #return self.net.feed_forward(episode_in, self.current_init_frame + self.frame, training, agent_frame)

   

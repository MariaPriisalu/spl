import numpy as np
from agent import SimplifiedAgent, ContinousAgent
from agents_dummy import DummyAgent
from net import Net
from agent_pfnn import AgentPFNN




import numpy as np

class NetAgent(SimplifiedAgent):

    def __init__(self, settings, net, grad_buffer, init_net=None, init_grad_buffer=None,goal_net=None,grad_buffer_goal=None):
        super(NetAgent, self).__init__(settings)
        self.net=net
        if init_net:
            self.init_net=init_net
        if goal_net:
            self.goal_net=goal_net

        if grad_buffer:
            net.set_grad_buffer(grad_buffer)
        if self.settings.learn_init and init_grad_buffer:
            self.init_net.set_grad_buffer(init_grad_buffer)
        if self.settings.separate_goal_net and grad_buffer_goal:
            self.goal_net.set_grad_buffer(grad_buffer_goal)
        self.current_init_frame = 0

    def init_agent(self, episode_in, training=True,  viz=False):
        init_value=self.init_net.feed_forward(episode_in, self.current_init_frame + self.frame, training, 0)
        if self.settings.separate_goal_net:
            self.goal_net.feed_forward(episode_in, self.current_init_frame + self.frame, training, 0)
        return init_value

    def next_action(self, episode_in, training=True,  viz=False):
        #print "Net agent next action"
        #super(NetAgent, self).next_action(episode_in, training)
        self.is_distracted()
        if training:
            agent_frame=self.frame
        else:
            agent_frame = -1
        if self.frame % self.settings.action_freq==0 or  episode_in.goal_person_id >= 0:
            return self.net.feed_forward(episode_in, self.current_init_frame + self.frame, training, agent_frame, distracted=self.distracted)#,  viz=viz)
        else:
            self.update_episode(episode_in, episode_in.speed[self.frame-1], episode_in.action[self.frame-1], episode_in.velocity[self.frame-1],  episode_in.probabilities[self.frame-1] )
            return episode_in.velocity[self.frame-1]


    def update_episode(self, episode_in, speed, value, velocity, probabilities):
        episode_in.velocity[self.frame] = velocity
        episode_in.action[self.frame] = value
        episode_in.probabilities[self.frame] = probabilities
        episode_in.speed[self.frame] = speed

    def train(self, ep_itr, statistics,episode, filename, filename_weights, poses):

        return self.net.train( ep_itr, statistics,episode, filename, filename_weights, poses, [], [])

    def init_net_train(self, ep_itr, statistics, episode, filename, filename_weights, poses,priors, initialization_car):
        if self.settings.separate_goal_net:
            return self.goal_net.train(ep_itr, statistics, episode, filename, filename_weights, poses, priors, initialization_car,[])
        return self.init_net.train(ep_itr, statistics, episode, filename, filename_weights, poses, priors, initialization_car,[])

    def set_session(self, session):
        self.net.sess = session
        self.init_net.sess = session
        self.goal_net.sess = session

    def evaluate(self, ep_itr, statistics, episode, poses, priors):
        return self.net.evaluate( ep_itr, statistics,episode, poses, priors, [])

    def init_net_evaluate(self,ep_itr, statistics, episode, poses, priors,initialization_car):
        return self.init_net.evaluate(ep_itr,  statistics,episode, poses, priors,initialization_car, [])

class AgentNetPFNN(AgentPFNN, NetAgent):
    def __init__(self, settings, net, grad_buffer, init_net, init_grad_buffer,goal_net=None,grad_buffer_goal=None):

        super(AgentNetPFNN, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer,goal_net,grad_buffer_goal)

    def initial_position(self, pos,goal, current_frame=0, vel=0, init_dir=[]):
        super(AgentNetPFNN, self).initial_position(pos, goal, current_frame=current_frame, vel=vel)


class ContinousNetAgent(ContinousAgent, NetAgent):
    def __init__(self, settings, net, grad_buffer, init_net, init_grad_buffer):

        super(ContinousNetAgent, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer)

    def initial_position(self, pos,goal, current_frame=0, vel=0, init_dir=[]):
        super(ContinousNetAgent, self).initial_position(pos, goal, current_frame=current_frame, vel=vel, init_dir=init_dir)

class ContinousNetPFNNAgent(ContinousAgent, AgentNetPFNN):
    def __init__(self, settings, net, grad_buffer, init_net, init_grad_buffer):

        super(ContinousNetPFNNAgent, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer)

    def initial_position(self, pos,goal, current_frame=0, vel=0, init_dir=[]):
        super(ContinousNetPFNNAgent, self).initial_position(pos, goal, current_frame=current_frame, vel=vel, init_dir=init_dir)



class SupervisedAgent(NetAgent):
    def __init__(self,  settings, net, grad_buffer, init_net, init_grad_buffer):
        super(SupervisedAgent, self).__init__( settings, net, grad_buffer, init_net, init_grad_buffer)
        self.current_init_frame = 0
        self.pos_exact = np.zeros(2, dtype=float)

    def initial_position(self, pos,goal, current_frame=0,vel=0, init_dir=[]):
        # self.position=np.array(np.array(pos)).astype(int)
        # self.frame = 0
        # self.current_init_frame =current_frame
        # self.pos_exact=np.array(np.array(pos))#pos_exact
        #print "Agent init frame"+str(self.frame)+"  current frame "+str(self.current_frame)
        super(SupervisedAgent, self).initial_position( pos,goal, current_frame,vel, init_dir)

    def perform_action(self, vel, episode, prob=0, person_id=-1, training=False):
        super(SupervisedAgent, self).perform_action( vel, episode, prob=0)
        #print "Agent frame "+str(self.frame)
        self.pos_exact=np.mean(episode.valid_people_tracks[episode.key_map[person_id]][self.frame], axis=1)
        #"Agent pos updated "+str(self.pos_exact)
        #episode.get_valid_vel_n(agent_frame, episode_in.goal_person_id)



    def next_action(self, episode_in, training=True, filename="", filename_weights=""):
        #super(SupervisedAgent, self).next_action( episode_in, training)
        if training:
            agent_frame=self.frame

        else:
            agent_frame = -1
        #print "next action agent " + str(self.frame) + "  current frame " + str(self.current_frame+self.frame)
        return self.net.feed_forward(episode_in, self.current_init_frame + self.frame, training, agent_frame, filename="", filename_weights="")

    def train(self,ep_itr, statistics, episode, filename, filename_weights , poses):
        return self.net.train( ep_itr, statistics,episode, filename, filename_weights)

    def set_session(self, session):
        self.net.sess = session

    def evaluate(self, ep_itr, statistics, episode,  poses, priors):
        return self.net.evaluate(statistics, episode, priors)

class SupervisedPFNNAgent(SupervisedAgent, AgentNetPFNN):
    def __init__(self, settings, net, grad_buffer, init_net, init_grad_buffer):
        super(SupervisedPFNNAgent, self).__init__(settings, net, grad_buffer, init_net, init_grad_buffer)

class PedestrianAbstractLikelihoodAgent(SimplifiedAgent):
    def __init__(self, settings, net=None, grad_buffer=None):
        super(PedestrianAbstractLikelihoodAgent, self).__init__(settings, net, grad_buffer)
        self.current_init_frame = 0
        self.delegatedExternalAgent = None
        self.delegatedExternalAgent_address = None

        # Delegate the actions to an external agent than ours ?
        if settings.evaluateExternalModel is not None:
            # Find the address where to send / get data
            self.delegatedExternalAgent_address, _, _ =None



    np.set_printoptions(precision=5)

    # This is likelihood evaluation!
    def next_action(self, episode_in, training=True):
        #print "Pedestrian action"
        if self.delegatedExternalAgent is None:
            super(PedestrianAbstractLikelihoodAgent, self).next_action(episode_in, training, viz=True)

        agent_action = np.argmax(episode_in.probabilities[self.frame, :9])

        agent_vel=episode_in.actions[agent_action]

        pos = episode_in.get_valid_pos(self.frame + 1, person_id=episode_in.goal_person_id_val)# - self.pos_exact

        dir = episode_in.get_goal_dir(episode_in.agent[self.frame], pos)
        value = np.nonzero(dir[0, :len(dir[0]) - 1])
        value = value[0][0]
        #print "Prob vel: "+str(episode_in.probabilities[self.frame, 0:9][value])+" vel:  "+str(pos-self.pos_exact)+" dir "+str(value)+" "+str(episode_in.actions[value])

        #print "Probabilities: "+str(episode_in.probabilities[self.frame, value])
        episode_in.probabilities[self.frame, 15] = episode_in.probabilities[self.frame, value]*1.25
        #episode_in.velocity[self.frame] = pos-self.pos_exact#episode_in.actions[value]

        episode_in.action[self.frame] = value
        episode_in.speed[self.frame] = np.linalg.norm(episode_in.velocity[self.frame][1:]) * episode_in.frame_rate / 5.0
        episode_in.probabilities[self.frame, 16] = (episode_in.speed[self.frame]-(episode_in.probabilities[self.frame, 9]))**2*50
        episode_in.probabilities[self.frame, 17]=np.log(episode_in.probabilities[self.frame, 15])-episode_in.probabilities[self.frame, 16]-np.log(2*np.pi*.01)
        episode_in.probabilities[self.frame, 18] = np.linalg.norm(self.pos_exact+episode_in.velocity[self.frame]-pos)
        #print "Agent takes action "+str(agent_vel)+" true: "+str(pos-self.pos_exact)
        #print "True speed: " + str(episode_in.speed[self.frame]) + " net out:  " + str(episode_in.probabilities[self.frame, 9]) +" z-value "+ str(episode_in.probabilities[self.frame, 16])+" log likelihood:  "+str(episode_in.probabilities[self.frame, 17])

        return episode_in.velocity[self.frame]




class PedestrianAgent(DummyAgent, PedestrianAbstractLikelihoodAgent):
    def __init__(self, settings, net=None, grad_buffer=None):
        super(PedestrianAgent, self).__init__(settings , net, grad_buffer)
        self.current_init_frame = 0



class PedestrianLikelihoodAgent(PedestrianAbstractLikelihoodAgent, NetAgent):
    def __init__(self, settings, net=None, grad_buffer=None, init_net=None, init_grad_buffer=None):
        #print "Pedestrian Likelihood agent"
        super(PedestrianLikelihoodAgent, self).__init__(settings , net, grad_buffer, init_net, init_grad_buffer)
        self.current_init_frame = 0

    def next_action(self, episode_in, training=True, viz=False):
        #print "Likelihood"
        return PedestrianAbstractLikelihoodAgent.next_action(self, episode_in, training)

    def perform_action(self, vel, episode, prob=0, person_id=-1, training=False):
        super(PedestrianLikelihoodAgent, self).perform_action( vel, episode, prob=0)
        self.pos_exact=np.mean(episode.valid_people_tracks[episode.key_map[person_id]][self.frame], axis=1)
        #episode.get_valid_vel_n(agent_frame, episode_in.goal_person_id)

class PedestrianLikelihoodPFNNAgent(PedestrianAbstractLikelihoodAgent, AgentPFNN):
    def __init__(self, settings, net=None, grad_buffer=None):
        #print "Pedestrian Likelihood agent"
        super(PedestrianLikelihoodPFNNAgent, self).__init__(settings , net, grad_buffer)
        self.current_init_frame = 0

    def next_action(self, episode_in, training=True, viz=False):
        #print "Likelihood"
        return PedestrianAbstractLikelihoodAgent.next_action(self, episode_in, training)

    def perform_action(self, vel, episode, prob=0, person_id=-1, training=False):
        super(PedestrianLikelihoodPFNNAgent, self).perform_action( vel, episode, prob=0)
        self.pos_exact=np.mean(episode.valid_people_tracks[episode.key_map[person_id]][self.frame], axis=1)

class PedestrianLikelihoodPFNNNetAgent(PedestrianAbstractLikelihoodAgent, AgentNetPFNN):
    def __init__(self, settings, net=None, grad_buffer=None, init_net=None, init_grad_buffer=None):
        #print "Pedestrian Likelihood agent"
        super(PedestrianLikelihoodPFNNNetAgent, self).__init__(settings , net, grad_buffer)
        self.current_init_frame = 0

    def next_action(self, episode_in, training=True, viz=False):
        #print "Likelihood"
        return PedestrianAbstractLikelihoodAgent.next_action(self, episode_in, training)

    def perform_action(self, vel, episode, prob=0, person_id=-1, training=False):
        super(PedestrianLikelihoodPFNNNetAgent, self).perform_action( vel, episode, prob=0)
        self.pos_exact=np.mean(episode.valid_people_tracks[episode.key_map[person_id]][self.frame], axis=1)
        #episode.get_valid_vel_n(agent_frame, episode_in.goal_person_id)


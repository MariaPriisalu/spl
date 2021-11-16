
import numpy as np
from environment import Environment
from episode import SimpleEpisode
from visualization import make_movie, make_movie_paralell, plot
from random_path_generator import random_path, random_walk
from utils.utils_functions import overlap
import time
from settings import NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Queue as PQueue
import queue




class TestEnvironment(Environment):
    
    def __init__(self, path,    sess, writer, gradBuffer, log,settings, net=None):
        super(TestEnvironment,self).__init__(path,   sess, writer, gradBuffer, log,settings, net=net)
        self.counter=0
        self.settings=settings
        self.pedestrians = True
        self.cars = True
        self.obstacles = True

        if self.settings.reward_weights[0]==0:
            self.cars = False
        if self.settings.reward_weights[1]==0:
            self.pedestrians = False
        if self.settings.reward_weights[3]==0:
            self.obstacles= False
        self.start=0
        if self.settings.paralell:
            self.my_jobs=[]
            self.my_q=PQueue()




# Take steps, calculate reward and visualize actions of agent.
    def work(self, file_name, agent, poses_db, training=True, road_width=5, conv_once=False, time_file=None):
        if not time_file is None:
            self.start = time.time()
        paralell=False


        path= self.settings.statistics_dir + "/stats_t/toy/"

        if conv_once:
            path+="conv/"
        else:
            path+="test/"
        path+=self.settings.timestamp
        if self.settings.paralell:
            pool = Pool(processes=8)
        else:
            pool=None

        repetitions = 1
        if training:
            repetitions = self.settings.repeat
        indx=0
        stats_all=np.zeros((repetitions,))

        rand_nbrs_train_indx=np.random.randint(len(self.settings.train_nbrs), size=repetitions)
        rand_nbrs_width_indx = np.random.randint(len(self.settings.widths), size=repetitions)


        for itr in range(repetitions):
            train_nbr = np.copy(self.settings.train_nbrs[rand_nbrs_train_indx[itr]])
            road_width = np.copy(self.settings.widths[rand_nbrs_width_indx[itr]])
            if train_nbr in[3,6,7]:
                road_width=max(4,road_width)

            if training:
                seq_len = self.settings.seq_len_train
                # To do: put this into settings!
                if self.settings.curriculum_reward:
                    if itr > 10 and not self.pedestrians:
                        self.settings.reward_weights[1] = 1
                        self.pedestrians = True
                    if itr > 20 and not self.obstacles:
                        self.settings.reward_weights[3] = 1
                        self.obstacles = True
                if self.settings.curriculum_seq_len:
                    if itr > repetitions / 4 and not self.settings.seq_len_train == 20:
                        self.settings.seq_len_train = 20
                    if itr > repetitions / 2 and not self.settings.seq_len_train == 40:
                        self.settings.seq_len_train = 40
                    if itr > repetitions * 3 / 4 and not self.settings.seq_len_train == 80:
                        self.settings.seq_len_train = 80
            else:
                seq_len = self.settings.seq_len_test

            if training:

                if not time_file is None:
                    self.start = time.time()



                print((path+"_"+ str(train_nbr)+"_"+str(itr)))
                if not self.settings.deterministic:
                    if  self.settings.prob<1:
                        curriculum_prob = self.settings.prob
                    else:
                        curriculum_prob = 1 - (itr * 1.0 / repetitions)
                else:
                    curriculum_prob = 1
                # act_and_learn(self, agent, agent_file, episode, poses_db, training, road_width=0, curriculum_prob=0,
                #               time_file=None):

                episode = self.set_up_episode([], [], train_nbr, 0, None, training, road_width,
                                              time_file=time_file)
                pos, _,_ = episode.initial_position(poses_db)

                statistics=self.act_and_learn(agent, path+"_"+ str(train_nbr)+"_"+str(itr),episode, poses_db, True,
                                              road_width,curriculum_prob=curriculum_prob)
                if not time_file is None:
                    end=time.time()
                    time_file.write(
                        str(end - self.start) + " Work on env " + str(self.settings.train_nbrs[rand_nbrs_train_indx[itr]]) +
                        " road width " + str(self.settings.widths[rand_nbrs_width_indx[itr]]) + " seq_len " + str(
                            self.settings.seq_len_train) + "\n")
                    self.start = time.time()

            if indx%100==0 or not training:
                for train_nbr in self.settings.train_nbrs:
                    if train_nbr in [3, 6, 7]:
                        road_width = max(4, road_width)
                    if not time_file is None:
                        self.start = time.time()
                    episode = self.set_up_episode([], [], train_nbr, 0, None, False, road_width,
                                                  time_file=time_file)
                    statistics = self.act_and_learn(agent,path + "_" + str(train_nbr) + "_" + str(itr), episode, poses_db, False,
                                                     road_width, time_file=time_file)

                    if not time_file is None:
                        end = time.time()
                        time_file.write(str(end - self.start) + " Work on env " + str(
                            train_nbr) + " road width " + str(
                            self.settings.widths[rand_nbrs_width_indx[itr]])+ " seq_len "+str(self.settings.seq_len_train)  + "\n")
                        self.start = time.time()
                    if self.settings.paralell:
                        self.counter, viz, pool=make_movie_paralell(episode.people, episode.reconstruction, [statistics],
                                                self.settings.width, self.settings.depth, self.settings.seq_len_train, self.counter,
                                                self.settings.agent_shape, self.settings.agent_shape_s, episode,
                                                training=True, name_movie= self.settings.name_movie, path_results=self.settings.img_dir)
                        self.my_jobs += viz
                        if not time_file is None:
                            end = time.time()
                            time_file.write(str(end - self.start) + " Visualization prep env" + str(
                                train_nbr) + " road width " + str(
                                self.settings.widths[rand_nbrs_width_indx[itr]]) + " seq_len " + str(
                                self.settings.seq_len_train) + "\n")
                            self.start = time.time()
                    else:
                        jointsParents = None
                        if self.settings.pfnn:
                            jointsParents = agent.PFNN.getJointParents()
                        self.counter=make_movie(episode.people,
                                                  episode.reconstruction,
                                                  [statistics],
                                                  self.settings.width,
                                                  self.settings.depth,
                                                  self.settings.seq_len_train,
                                                  self.counter,
                                                  self.settings.agent_shape,
                                                  self.settings.agent_shape_s, episode,
                                                  training=training,
                                                  name_movie=self.settings.name_movie,
                                                  nbr_episodes_to_vis=1,
                                                  path_results=self.statistics_dir.statistics_dir + "/agent/",
                                                  episode_name=file_name,
                                                  jointsParents=jointsParents)
                        # self.counter = make_movie(episode.people,
                        #                           episode.reconstruction,
                        #                           [statistics],
                        #                           self.settings.width,
                        #                           self.settings.depth,
                        #                           self.settings.seq_len_train,
                        #                           self.counter,
                        #                           self.settings.agent_shape,
                        #                           self.settings.agent_shape_s, episode,
                        #                           training=True,
                        #                           name_movie=self.settings.name_movie,
                        #                           path_results=self.statistics_dir + "/agent/")



                for train_nbr in self.settings.test_nbrs:
                    if train_nbr in [3, 6, 7]:
                        road_width = max(4, road_width)
                    if not time_file is None:
                        self.start = time.time()
                    path_here=path + "_test_" + str(train_nbr) + "_" + str(itr)
                    episode = self.set_up_episode([], [], train_nbr, 0, None, False, road_width,
                                                  time_file=time_file)
                    statistics= self.act_and_learn(agent, path_here, episode, poses_db, False,road_width=4)
                    if not time_file is None:
                        end = time.time()
                        time_file.write(str(end - self.start) + " Work on env " + str(train_nbr) + " road width " + str(
                            self.settings.widths[rand_nbrs_width_indx[itr]]) + " seq_len "+str(self.settings.seq_len_test)+ "\n")
                        self.start = time.time()

                    if self.settings.paralell:
                        # self, agent, cars, file_agent, people, pos_x, pos_y, poses_db, tensor, training, road_width
                        self.counter, viz, pool=make_movie_paralell(episode.people,
                                                                    episode.reconstruction,
                                                                    [statistics],
                                                                    self.settings.width,
                                                                    self.settings.depth,
                                                                    self.settings.seq_len_test,
                                                                    self.counter,
                                                                    self.settings.agent_shape,
                                                                    self.settings.agent_shape_s,
                                                                    episode,
                                                                    training=False,
                                                                    nbr_episodes_to_vis=5,
                                                                    name_movie=self.settings.name_movie,
                                                                    path_results=self.settings.img_dir)
                        self.my_jobs += viz

                    else:
                        self.counter = make_movie(episode.people,
                                                  episode.reconstruction,
                                                  [statistics],
                                                  self.settings.width,
                                                  self.settings.depth,
                                                  self.settings.seq_len_train,
                                                  self.counter,
                                                  self.settings.agent_shape,
                                                  self.settings.agent_shape_s, episode,
                                                  training=False,
                                                  nbr_episodes_to_vis=5,
                                                  name_movie=self.settings.name_movie,
                                                  path_results=self.settings.statistics_dir + "/agent/")


                    if not time_file is None:
                        end = time.time()
                        time_file.write(str(end - self.start) + " Visualization prep test env " + str(train_nbr) +  " road width " + str(
                            self.settings.widths[rand_nbrs_width_indx[itr]]) + " seq_len "+str(self.settings.seq_len_test)+ "\n")
                        self.start = time.time()


            indx = indx + 1
        if self.settings.paralell:
            if not time_file is None:
                self.start = time.time()

            print((pool.map(plot, [(self.my_q, entry) for entry in self.my_jobs])))

            its = []
            while True:

                # If we go more than 30 seconds without something, die
                try:
                    print("Waiting for item from queue for up to 2 min")
                    i = self.my_q.get(True, 120)
                    its.append(i)
                except queue.Empty:
                    print("Caught queue empty exception, done")
                    break
            print(("processed %d items, completion successful" % len(its)))

            pool.close()
            pool.join()
            if not time_file is None:
                end = time.time()
                time_file.write(
                    str(end - self.start) + " Visualization paralell ")
                self.start = time.time()
            print(("Saved!+ frames+ " + str(self.counter)))


    # Return file name as to where to save statistics.
    def statistics_file_name(self, file_agent, pos_x, pos_y, training):
        return file_agent

    # Set up toy episodes.
    def set_up_episode(self, cars, people, pos_x, pos_y, tensor, training, road_width=0, time_file=None):

        if not time_file is None:
            start = time.time()
        if training:
            seq_len=self.settings.seq_len_train
        else:
            seq_len = self.settings.seq_len_test

        self.reconstruction = np.zeros((self.settings.height, self.settings.width, self.settings.depth, 6))
        cars = []
        people = []
        if pos_x==0: # horizontal line
            for frame in range(seq_len):
                cars.append([])
                people.append([])

                #cars[frame].append([0, 8, self.settings.width / 2 - 5, self.settings.width / 2 + 5, self.settings.depth / 2 - 5, self.settings.depth / 2 + 5])
                # [self.settings.width / 2 - 2, self.settings.width / 2 + 2], [frame, frame + 2]]
            if self.pedestrians:
                for p in range(self.settings.depth/2):
                    people[0].append(self.create_person(self.settings.width/2-10, p,2))
                    people[2].append(self.create_person(self.settings.width / 2 + 10,self.settings.depth/2+ p, 2))
                for p in range(self.settings.width/2-10,self.settings.width/2+10):
                    people[1].append(self.create_person(p,self.settings.depth/2, 2))

            pos_x = 0
            pos_y = 0

            self.set_values([0,2, self.settings.width / 2 +road_width*2- road_width,self.settings.width / 2 +(road_width*2)+road_width,0, self.settings.depth], 8/NUM_)
            self.set_values([0, 2, self.settings.width / 2 -(road_width*2)- road_width,self.settings.width / 2 -(road_width*2)+ road_width, 0, self.settings.depth],
                            8 / NUM_SEM_CLASSES)
            if self.obstacles:
                # Buildings
                self.set_values([0, 8, self.settings.width / 2 + (road_width*2) + road_width, self.settings.width, 0, self.settings.depth],
                                11 / NUM_SEM_CLASSES)
                self.set_values([0, 8, 0,self.settings.width / 2 -(road_width*2) - road_width, 0, self.settings.depth],
                                11 / NUM_SEM_CLASSES)

                # Post
                self.set_values([0, 8,self.settings.width/2-(road_width*2)+road_width/2,self.settings.width/2-(road_width*2)+road_width,self.settings.depth/2 +2,self.settings.depth/2+2+road_width/2],
                                11 / NUM_SEM_CLASSES)
                # self.set_values(
                #     [0, 8,  self.settings.width / 2 + (road_width*2) - road_width ,self.settings.width / 2 + (road_width*2) - road_width/2,
                # self.settings.depth / 2 - 4,self.settings.depth / 2 - 4 + road_width / 2 ],
                #     11 / NUM_SEM_CLASSES)
            if self.cars:
                for p in range(self.settings.depth ):
                    cars[p%(seq_len-1)].append([0, 8, self.settings.width / 2 - road_width/2, self.settings.width / 2 + road_width/2,  p-road_width/2, p+ road_width/2])



        elif pos_x==1: # One dot in the middle
            for frame in range(seq_len):
                cars.append([])
                people.append([])
            if self.pedestrians:
                people[0].append(self.create_person(self.settings.width/2, self.settings.depth/2 ,1))
            pos_x = 0
            pos_y = 0
            self.set_values([0,2, self.settings.width / 2 -road_width,self.settings.width / 2 +road_width,  self.settings.depth/2-road_width,self.settings.depth/2+road_width], 8/NUM_SEM_CLASSES)

            # Buildings
            if self.obstacles:
                self.set_values([0,8, 0,self.settings.width, 0,self.settings.depth / 2 - 1 - road_width], 11 / NUM_SEM_CLASSES)

                self.set_values([0, 8, 0, self.settings.width, self.settings.depth / 2 + 1 + road_width,self.settings.depth], 11 / NUM_SEM_CLASSES)

            if self.cars:
                for p in range(self.settings.width):
                    cars[p % (seq_len - 1)].append(
                        [0, 8, p , p + road_width,  self.settings.depth / 2  + road_width-1, self.settings.depth / 2 +  road_width+1])

        elif pos_x == 2: # diagonal

            for frame in range(seq_len):
                cars.append([])
                people.append([])
            i=0
            while i+road_width< self.settings.width and  i+road_width<self.settings.depth:
                self.reconstruction[0:2, i,i,3] = 8 / NUM_SEM_CLASSES*np.ones(2)
                if self.cars:
                    cars[i%seq_len].append( [0, 8, i, i+2,int(i+road_width*2), int(i+road_width*2)+2])
                if self.pedestrians and (i< self.settings.width/2-(road_width*np.sqrt(2)) or i> self.settings.width/2+(road_width*np.sqrt(2))):
                    people[min(i,seq_len-2)].append(self.create_person(i, i,1))#np.array([[0,8], [i-1,i+1], [i-1,i+1]]))
                tmp=self.reconstruction[0:2, i-road_width:i+road_width, i-road_width:i+road_width, 3]
                self.reconstruction[0:2, i-road_width:i+road_width, i-road_width:i+road_width, 3] = 8 / NUM_SEM_CLASSES * np.ones(tmp.shape)
                i+=1
            if self.pedestrians:
                for i in range(self.settings.width/2-int(road_width*np.sqrt(2)),self.settings.width/2+int(road_width*np.sqrt(2))):
                    people[min(i, seq_len - 2)].append(self.create_person(self.settings.width/2-(road_width*np.sqrt(2)), i, 1))
                    people[min(i, seq_len - 2)].append(self.create_person(i,self.settings.width/2+(road_width*np.sqrt(2)), 1))
            pos_x = 0
            pos_y = 0

            # Buildings
            if self.obstacles:
                self.set_values([0,8, self.settings.width / 2 - road_width,self.settings.width / 2 + road_width, self.settings.width / 2 - road_width,self.settings.width / 2 + road_width], 11/NUM_SEM_CLASSES)



        elif pos_x == 3:
            for frame in range(seq_len): # Fork
                cars.append([])
                people.append([])

            # middle vertical line
            self.set_values([0,2, 0,self.settings.width/2,self.settings.depth/2,self.settings.depth/2+road_width], 8/NUM_SEM_CLASSES)
            if self.pedestrians:
                for i in range(3):
                    for y in range(self.settings.width/2+road_width/2):
                        people[i].append(self.create_person(y,self.settings.depth/2+road_width/2,1))#np.array([[0,8],[y,y], [self.settings.depth/2,self.settings.depth/2]]))

            # horizontal line
            self.set_values([0,2, self.settings.width / 2,self.settings.width / 2+road_width,self.settings.depth / 4,self.settings.depth*3 / 4], 8 / NUM_SEM_CLASSES)

            if self.pedestrians:
                for i in range(2):
                    for x in range(self.settings.depth / 4,self.settings.depth /2):
                        people[i].append(self.create_person(self.settings.width/2+road_width/2,x+road_width/2,1))#np.array([[0,8],[self.settings.width/2,self.settings.width/2], [x,x]]))

                for x in range(self.settings.depth /2, self.settings.depth*3/4):
                    people[2].append(self.create_person(self.settings.width/2+road_width/2,x+road_width/2,1))#[[0,8],[self.settings.width/2,self.settings.width/2], [x,x]]))

            # left line
            self.set_values([0, 2, self.settings.width / 2,self.settings.width, self.settings.depth / 4,self.settings.depth / 4+road_width],
                            8 / NUM_SEM_CLASSES)
            if self.pedestrians:
                for i in range(2):
                    for y in range(self.settings.width/2,self.settings.width ):
                        people[i].append(self.create_person(y+road_width/2,self.settings.depth/4+road_width/2,1))#[[0,8],[y,y], [self.settings.depth/4,self.settings.depth/4]]))

            # right line
            self.set_values([0, 2, self.settings.width / 2,self.settings.width, self.settings.depth*3 / 4,self.settings.depth*3 / 4+road_width],
                            8 / NUM_SEM_CLASSES)
            if self.pedestrians:
                for y in range(self.settings.width/2,self.settings.width ):
                    people[2].append(self.create_person(y+road_width/2,self.settings.depth*3/4+road_width/2,1))#np.array([[0,8],[y,y], [self.settings.depth*3/4,self.settings.depth*3/4]]))

            if self.obstacles:
                # Buildings - under halva
                self.set_values([0,8, self.settings.width / 2+road_width,self.settings.width, self.settings.depth / 4+road_width,self.settings.depth * 3 / 4],
                                11 / NUM_SEM_CLASSES)

                # upper quarter
                self.set_values([0, 8, 0,self.settings.width/2-2,0,self.settings.depth/2-2],11 / NUM_SEM_CLASSES)


                # vertical line
                self.set_values([0, 8, 0, self.settings.width ,self.settings.depth *3/ 4+road_width+2,self.settings.depth], 11 / NUM_SEM_CLASSES)

            if self.cars:
                for p in range(self.settings.depth):
                    cars[p % (seq_len - 1)].append(
                        [0, 8, self.settings.width / 2 - road_width/2 , self.settings.width / 2 , p - road_width / 2,
                         p + road_width / 2])

            pos_x = 0
            pos_y = 0

        elif pos_x == 4: # vertical
            for frame in range(seq_len):
                cars.append([])
                people.append([])
            #     cars[frame].append([0, 8, self.settings.width / 2 - 5, self.settings.width / 2 + 5, self.settings.depth / 2 - 5, self.settings.depth / 2 + 5])
                #people[frame].append(self.create_person(frame, self.settings.depth/2, 2))

                    #np.array([[0, 8], [frame, frame + 2],[self.settings.depth / 2 - 2, self.settings.depth / 2 + 2]]).reshape(3, 2))
            pos_x = 0
            pos_y = 0
            self.set_values([0, 2, 0, self.settings.width, self.settings.depth / 2 - road_width,self.settings.depth / 2 + road_width], 8 / NUM_SEM_CLASSES)
            #self.reconstruction[0:2, :,self.settings.depth / 2 - road_width:self.settings.depth / 2 + road_width, 3] = 8 / NUM_SEM_CLASSES * np.ones((2, self.settings.width,road_width))

            # Poles
            if self.obstacles:
                self.set_values([0, 8, self.settings.width/10-road_width/2, self.settings.width/10+road_width/2+1, self.settings.depth / 2 -road_width,self.settings.depth /2], 11 / NUM_SEM_CLASSES)
            # self.reconstruction[0:8, self.settings.width/10-road_width/2: self.settings.width/10+road_width/2+1, self.settings.depth / 2 -road_width:self.settings.depth /2,3] = 11 / NUM_SEM_CLASSES * np.ones(
            #     (8, road_width, road_width))

            if self.pedestrians:
                for p in range(self.settings.width/5):
                    people[0].append(self.create_person(p, self.settings.depth / 2+road_width/2, 2))

                for q in range(self.settings.depth / 2 - road_width / 2, self.settings.depth / 2 + road_width / 2):
                    people[0].append(self.create_person(self.settings.width * 1 / 5, q, 2))
            if self.obstacles:
                self.set_values([0, 8, 3*self.settings.width / 10 - road_width / 2, 3*self.settings.width / 10 + road_width / 2 + 1,
                self.settings.depth / 2 ,self.settings.depth / 2+road_width], 11 / NUM_SEM_CLASSES)

            if self.pedestrians:
                for p in range(self.settings.width/5,self.settings.width*2/5):
                    people[0].append(self.create_person(p, self.settings.depth / 2-road_width/2, 2))

                for q in range(self.settings.depth / 2 - road_width / 2, self.settings.depth / 2 + road_width ):
                    people[0].append(self.create_person(self.settings.width * 2 / 5, q, 2))
            if self.obstacles:
                self.set_values([0, 8, 7*self.settings.width / 10 - road_width / 2, 7*self.settings.width / 10 + road_width / 2 + 1,
                                 self.settings.depth / 2 - road_width,self.settings.depth / 2], 11 / NUM_SEM_CLASSES)

            if self.pedestrians:
                for p in range(self.settings.width*3/5, self.settings.width*4/5):
                    people[0].append(self.create_person(p, self.settings.depth / 2+road_width/2, 2))

                for q in range(self.settings.depth / 2 + road_width / 2, self.settings.depth / 2 + road_width ):
                    people[0].append(self.create_person(self.settings.width * 3 / 5, q, 2))

            if self.obstacles:
                self.set_values([0, 8,  9 * self.settings.width / 10 - road_width / 2, 9 * self.settings.width / 10 + road_width / 2 + 1,
                self.settings.depth / 2,self.settings.depth / 2 + road_width], 11 / NUM_SEM_CLASSES)

            if self.pedestrians:
                for p in range(self.settings.width*4/5, self.settings.width):
                    people[0].append(self.create_person(p, self.settings.depth / 2 - road_width / 2, 2))

                for q in range(self.settings.depth / 2 - road_width / 2, self.settings.depth / 2 + road_width / 2):
                    people[0].append(self.create_person(self.settings.width * 4 / 5, q, 2))

            # middle point
            if self.obstacles:
                self.set_values([0, 8, self.settings.width / 2 - road_width / 2, self.settings.width / 2 + road_width / 2 + 1,
                self.settings.depth / 2- road_width/2,self.settings.depth / 2 + road_width/2+1], 11 / NUM_SEM_CLASSES)


            if self.pedestrians:
                for p in range(self.settings.width * 2 / 5, self.settings.width * 3 / 5):
                    people[0].append(self.create_person(p, self.settings.depth / 2 + road_width , 2))

            if self.cars:
                for p in range(self.settings.width):
                    cars[p % (seq_len - 1)].append(
                        [0, 8, p , p + road_width,  self.settings.depth / 2  + road_width-1, self.settings.depth / 2 +  road_width+1])


        elif pos_x == 5: # diagonal. Other way around.

            for frame in range(seq_len):
                cars.append([])
                people.append([])

            i=0
            margin=road_width*np.sqrt(2) #
            upper_bound = self.settings.width * 5 / 8-margin/2
            lower_bound = self.settings.width * 3 / 8
            while i + road_width < self.settings.width and i + road_width < self.settings.depth:
                j=self.settings.width-i-1-road_width
                self.reconstruction[0:2, j, i, 3] = 8 / NUM_SEM_CLASSES * np.ones(2)
                if self.cars:
                    cars[i % seq_len].append(
                        [0, 8, j, j + 2, int(i + road_width * 2), int(i + road_width * 2) + 2])
                    # cars[i % self.seq_len].append(
                    #     [0, 8, j, j + 2, int(i - road_width * 2), int(i - road_width * 2) + 2])

                if self.pedestrians:
                    if i< lower_bound or i> upper_bound:
                        if  i>margin:
                            people[min(i, seq_len - 2)].append(self.create_person(j, i-margin, 1))
                    else:
                        people[min(i, seq_len - 2)].append(self.create_person(j, i + margin, 1))
                tmp = self.reconstruction[0:2, j - road_width:j+ road_width, i - road_width:i + road_width, 3]
                self.reconstruction[0:2,j - road_width:j + road_width, i - road_width:i + road_width,
                3] = 8 / NUM_SEM_CLASSES * np.ones(tmp.shape)
                i += 1
            if self.pedestrians:
                for i in range(-int(margin), int(margin)):
                    j = self.settings.width - i - 1 - road_width
                    people[min(i,seq_len - 2)].append(self.create_person(self.settings.width-lower_bound-1-road_width,lower_bound+i, 1))
                    people[min(i,seq_len - 2)].append(self.create_person(self.settings.width - upper_bound - 1 - road_width, upper_bound + i, 1))
            if self.obstacles:
                self.set_values([0, 8, self.settings.width *1/ 4 - road_width , self.settings.width *1/ 4 + road_width ,
                                 self.settings.width * 3 / 4 - road_width , self.settings.width * 3 / 4 + road_width ], 11 / NUM_SEM_CLASSES)

                self.set_values([0, 8, self.settings.width * 3 / 4 - road_width, self.settings.width * 3 / 4 + road_width,
                                 self.settings.width * 1 / 4 - road_width, self.settings.width * 1 / 4 + road_width], 11 / NUM_SEM_CLASSES)

                self.set_values([0, 8, self.settings.width * 1 / 2 - 2*road_width, self.settings.width * 1 / 2-road_width/2 ,
                                 self.settings.width * 1 / 2 -2*road_width, self.settings.width * 1 / 2- road_width/2], 11 / NUM_SEM_CLASSES)
        elif pos_x == 6:
            for frame in range(seq_len):
                cars.append([])
                people.append([])
            num_crossings = 4
            if self.obstacles:
                self.reconstruction[0:2,:,:,3]=11 / NUM_SEM_CLASSES*np.ones((2,  self.settings.width, self.settings.depth))
            if self.cars:
                self.reconstruction, point, thetas, steps, cars= random_path(self.reconstruction, road_width, num_crossings, cars)
            else:
                self.reconstruction, point, thetas, steps, _= random_path(self.reconstruction, road_width,
                                                                              num_crossings)
            if self.pedestrians:
                people[0].append(self.create_person(point[0],point[1],1))
                for p in steps:
                    people[0].append(self.create_person(p[0], p[1], 1))
            #np.array([[0, 8], [point[0]-1,point[0]-1 ], [point[1]-1,point[1]-1]]))
            if not time_file is None:
                end = time.time()
                time_file.write(
                    str(end - start) + " Create env " + str(pos_x) + "road_width" + str(road_width) + "\n")
            return SimpleEpisode(self.reconstruction, people, cars, pos_x, pos_y, self.settings.gamma, seq_len,
                                 reward_weights=self.settings.reward_weights, defaultSettings=self.settings)

        elif pos_x == 7:
            for frame in range(seq_len):
                cars.append([])
                people.append([])

            num_crossings = 2
            if self.obstacles:
                self.reconstruction[0:2, :, :, 3] = 11 / NUM_SEM_CLASSES * np.ones((2, self.settings.width, self.settings.depth))
            if self.cars:
                self.reconstruction, point, thetas, steps, cars= random_path(self.reconstruction, road_width, num_crossings, cars)
            else:
                self.reconstruction, point, thetas, steps, _ = random_path(self.reconstruction, road_width,
                                                                           num_crossings)
            if self.pedestrians:
                for p in steps:
                    people[0].append(self.create_person(p[0], p[1], 1))
                people[0].append(self.create_person(point[0], point[1],1))
            #np.array([[0, 8], [point[0] - 1, point[0] - 1], [point[1] - 1, point[1] - 1]]))
            if not time_file is None:
                end = time.time()
                time_file.write(
                    str(end - start) + " Create env " + str(pos_x) + "road_width" + str(road_width) + "\n")
            return SimpleEpisode(self.reconstruction, people, cars, pos_x, pos_y, self.settings.gamma, seq_len,
                                 reward_weights=self.settings.reward_weights, defaultSettings=self.settings)
        elif pos_x == 8:
            for frame in range(seq_len):
                cars.append([])
                people.append([])
            width = road_width
            num_steps=300
            if self.obstacles:
                self.reconstruction[0:2, :, :, 3] = 11 / NUM_SEM_CLASSES * np.ones((2, self.settings.width, self.settings.depth))
            #people[0].append(np.array([[0, 8], [self.settings.width/2,self.settings.width/2], [self.settings.depth/2,self.settings.depth/2]]))
            self.reconstruction, pos_1 , _= random_walk(self.reconstruction, width, num_steps, horizontal=True) # tensor, width, num_steps, horizontal=True, diagonal=False, dir=1
            self.reconstruction , pos_2, _= random_walk(self.reconstruction, width, num_steps, horizontal=False)
            first_p = []
            second_p = []
            previous = None

            c=0
            for p in pos_1+pos_2:
                if p[0] < self.reconstruction.shape[1] and p[1] < self.reconstruction.shape[2] and p[0] > 0 and p[
                    1] > 0:
                    if len(people[0])==0:
                        first_p=[p[0],p[1]]
                    if c>=len(pos_1) and len(first_p)==2:
                        first_p.append(p[0])
                        first_p.append(p[1])
                        second_p.append(previous[0])
                        second_p.append(previous[1])
                    if self.pedestrians:
                        people[0].append(self.create_person(p[0],p[1],1))
                    previous=[p[0],p[1]]
                c+=1
            second_p.append(previous[0])
            second_p.append( previous[1])
            if self.cars:


                dir=[ second_p[0]-first_p[0], second_p[1]-first_p[1] ,second_p[2] - first_p[2], second_p[3] - first_p[3]]
                for t in range(100):
                    cars[t/len(cars)].append([0,2,dir[0]*t/100+ first_p[0],dir[0]*t/100+ first_p[0]+1, dir[1]*t/100+ first_p[1], dir[1]*t/100+ first_p[1]+1 ])
                    cars[t / len(cars)].append([0, 2, dir[2] * t / 100 + first_p[2], dir[2] * t / 100 + first_p[2] + 1,
                                                dir[3] * t / 100 + first_p[3], dir[3] * t / 100 + first_p[3] + 1])
            if not time_file is None:
                end = time.time()
                time_file.write(
                    str(end - start) + " Create env " + str(pos_x) + "road_width" + str(road_width) + "\n")
            return SimpleEpisode(self.reconstruction, people, cars, pos_x, pos_y, self.settings.gamma, seq_len,
                                 reward_weights=self.settings.reward_weights, defaultSettings=self.settings)
        elif pos_x == 9:
            for frame in range(seq_len):
                cars.append([])
                people.append([])
            width = road_width
            num_steps = 300
            #self.reconstruction[0:2, :, :, 3] = 11 / NUM_SEM_CLASSES * np.ones((2, self.settings.width, self.settings.depth))
            if self.cars:
                self.reconstruction, pos_1, cars= random_walk(self.reconstruction, width, num_steps, cars=cars,diagonal=True)
                self.reconstruction, pos_2 , cars= random_walk(self.reconstruction, width, num_steps, cars=cars,diagonal=True, dir=-1)
            else:
                self.reconstruction, pos_1, _ = random_walk(self.reconstruction, width, num_steps,
                                                               diagonal=True)
                self.reconstruction, pos_2, _ = random_walk(self.reconstruction, width, num_steps,
                                                               diagonal=True, dir=-1)
            positions=pos_1+pos_2
            if self.pedestrians:
                for p in positions:
                    if p[0]<self.reconstruction.shape[1] and p[1]<self.reconstruction.shape[2] and p[0]>0 and p[1]>0:
                        people[0].append(self.create_person(p[0],p[1],1))
            if self.obstacles:
                obstacles=np.random.randint(0, len(positions)-1, size=10)
                for obs in obstacles:
                    middle=(positions[obs]+positions[obs+1])/2
                    bbox=[0,8,int(middle[0]-road_width/2),int(middle[0]+road_width/2+1), int(middle[1]-road_width/2),int(middle[1]+road_width/2+1)]

                    people_overlapped = []
                    for indx, person in enumerate(people[0]):
                        x_pers = [min(person[0, :]), max(person[0, :]), min(person[1, :]), max(person[1, :]),
                                  min(person[2, :]), max(person[2, :])]
                        if overlap(x_pers, bbox, 1) or overlap(bbox,x_pers,  1) :
                            people_overlapped.append(indx)
                    for indx in sorted(people_overlapped, reverse=True):
                        del people[0][indx]

                    self.set_values(bbox, 11/NUM_SEM_CLASSES)
            if not time_file is None:
                end = time.time()
                time_file.write(
                    str(end - start) + " Create env " + str(pos_x) + "road_width" + str(road_width) + "\n")
            return SimpleEpisode(self.reconstruction, people, cars, pos_x, pos_y, self.settings.gamma, seq_len,
                                 reward_weights=self.settings.reward_weights, defaultSettings=self.settings)
        elif pos_x == 10:
            for frame in range(seq_len):
                cars.append([])
                people.append([])
            width = road_width
            num_steps = 300
            # self.reconstruction[0:2, :, :, 3] = 11 / NUM_SEM_CLASSES * np.ones((2, self.settings.width, self.settings.depth))
            self.reconstruction, pos_1 = random_walk(self.reconstruction, width, num_steps, diagonal=True)
            self.reconstruction, pos_2 = random_walk(self.reconstruction, width, num_steps, diagonal=True, dir=-1)
            positions = pos_1 + pos_2
            if self.pedestrians:
                for p in positions:
                    if p[0] < self.reconstruction.shape[1] and p[1] < self.reconstruction.shape[2] and p[0] > 0 and p[
                        1] > 0:
                        people[0].append(self.create_person(p[0], p[1], 1))

            obstacles = np.random.randint(0, len(positions) - 1, size=10)
            for obs in obstacles:
                middle = (positions[obs] + positions[obs + 1]) / 2
                bbox = [0, 8, int(middle[0] - road_width / 2), int(middle[0] + road_width / 2 + 1),
                        int(middle[1] - road_width / 2), int(middle[1] + road_width / 2 + 1)]

                people_overlapped = []
                for indx, person in enumerate(people[0]):
                    x_pers = [min(person[0, :]), max(person[0, :]), min(person[1, :]), max(person[1, :]),
                              min(person[2, :]), max(person[2, :])]
                    if overlap(x_pers, bbox, 1) or overlap(bbox, x_pers, 1):
                        people_overlapped.append(indx)
                for indx in sorted(people_overlapped, reverse=True):
                    del people[0][indx]

                self.set_values(bbox, 0)
            if not time_file is None:
                end = time.time()
                time_file.write(
                    str(end - start) + " Create env " + str(pos_x) + "road_width" + str(road_width) + "\n")
            return SimpleEpisode(self.reconstruction, people, cars, pos_x, pos_y, self.settings.gamma, seq_len,
                                 reward_weights=self.settings.reward_weights, defaultSettings=self.settings)
        if not time_file is None:
            end = time.time()
            time_file.write(str(end - start) + " Create env " + str(pos_x) +"road_width"+str(road_width)+ "\n")
        return SimpleEpisode(self.reconstruction, people, cars, pos_x, pos_y, self.settings.gamma, seq_len,
                             reward_weights=self.settings.reward_weights, defaultSettings=self.settings)

    def create_person(self,y_p,x_p,w_p):
        #[self.settings.width / 2 - 2, self.settings.width / 2 + 2], [frame, frame + 2]]
        return np.array([[0, 8],[y_p-w_p,y_p+w_p], [x_p-w_p, x_p+w_p]] ).reshape(3, 2)

    def set_values(self, indx, val):
        tmp=self.reconstruction[indx[0]:indx[1], indx[2]:indx[3], indx[4]:indx[5],3].shape
        self.reconstruction[indx[0]:indx[1], indx[2]:indx[3], indx[4]:indx[5],3]=val*np.ones(tmp)
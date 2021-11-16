
import numpy as np
from RL.environment import Environment
from RL.episode import SimpleEpisode
from RL.visualization import make_movie
from RL.random_path_generator import random_path, random_walk
from utils.utils_functions import overlap
from RL.settings import NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT

class TestEnvironment(Environment):
    # __init__(self, path,constants,  sess, writer, gradBuffer, log,settings, net=None):
    def __init__(self, path,constants,  sess, writer, gradBuffer, log,settings, net=None):
        super(TestEnvironment,self).__init__(path,constants,  sess, writer, gradBuffer, log,settings, net=None)
        self.counter=0

        self.pedestrians = True
        self.cars = True
        self.obstacles = True
        self.deterministic=True



# Take steps, calculate reward and visualize actions of agent.
    def work(self,cachePrefix, file_name, agent, poses_db, epoch,saved_files_counter, training=True, road_width=5, conv_once=False,time_file=None, act=True):
        #work(self, file_name, agent, poses_db, training=True, road_width=5, conv_once=False):
        path= self.statistics_dir + "/stats_t/toy/"

        if conv_once:
            path+="conv/"
        else:
            path+="test/"


        widths = [ 6, 7, 10]
        repeat = len(widths)*len(self.train_nbrs)*50  # How many epochs.
        indx=0
        stats_all=np.zeros((repeat,))

        rand_nbrs_train_indx=np.random.randint(len(self.train_nbrs), size=repeat)
        rand_nbrs_width_indx = np.random.randint(len(widths), size=repeat)
        for itr in range(repeat):
            train_nbr=np.copy(self.train_nbrs[rand_nbrs_train_indx[itr]])
            road_width=np.copy(widths[rand_nbrs_width_indx[itr]])
            print (path+"_"+ str(train_nbr)+"_"+str(itr))
            if not self.deterministic:
                curriculum_prob = 1 - (itr * 1.0 / repeat)
            else:
                curriculum_prob = 1
            statistics,  people, cars, recon, episode, poses=self.act_and_learn(agent, None, path+"_"+ str(train_nbr)+"_"+str(itr),
                                                                        None,train_nbr, indx, poses_db, None, True, road_width, curriculum_prob=curriculum_prob)

            if indx%100==0:
                for train_nbr in self.train_nbrs:
                    statistics, people, cars, recon, episode = self.act_and_learn(agent, None, path + "_" + str(train_nbr) + "_" + str(itr),
                                                                                  None, train_nbr, indx, poses_db, None,
                                                                                  False, road_width)
                    self.counter=make_movie(people, recon, [statistics], self.width, self.depth, self.seq_len_train, self.counter,
                                            self.agent_shape, self.agent_sight, episode,poses, training=True, name_movie= self.name_movie, path_results=self.img_dir)
                for train_nbr in self.test_nbrs:
                    path_here=path + "_test_" + str(train_nbr) + "_" + str(itr)

                    statistics, people, cars, recon, episode = self.act_and_learn(agent, None,path_here, None, train_nbr,indx, poses_db, None, False, 7)

                    # self, agent, cars, file_agent, people, pos_x, pos_y, poses_db, tensor, training, road_width
                    self.counter=make_movie(people, recon, [statistics], self.width, self.depth, self.seq_len_test, self.counter,
                                            self.agent_shape, self.agent_sight, episode, poses, training=False, nbr_episodes_to_vis=5,
                                            name_movie=self.name_movie, path_results=self.img_dir)

            indx = indx + 1
        print ("Saved!+ frames+ " + str(self.counter))

    # Return file name as to where to save statistics.
    def statistics_file_name(self, file_agent, pos_x, pos_y):
        return file_agent

    # Set up toy episodes.
    def set_up_episode(self, cars, people, pos_x, pos_y, tensor, training, road_width=0):
        if training:
            seq_len=self.seq_len_train
        else:
            seq_len = self.seq_len_test

        self.reconstruction = np.zeros((self.height, self.width, self.depth, 6))
        cars = []
        people = []
        if pos_x==0: # horizontal line
            for frame in range(seq_len):
                cars.append([])
                people.append([])

                #cars[frame].append([0, 8, self.width / 2 - 5, self.width / 2 + 5, self.depth / 2 - 5, self.depth / 2 + 5])
                # [self.width / 2 - 2, self.width / 2 + 2], [frame, frame + 2]]
            if self.pedestrians:
                for p in range(self.depth/2):
                    people[0].append(self.create_person(self.width/2-10, p,2))
                    people[2].append(self.create_person(self.width / 2 + 10,self.depth/2+ p, 2))
                for p in range(self.width/2-10,self.width/2+10):
                    people[1].append(self.create_person(p,self.depth/2, 2))

            pos_x = 0
            pos_y = 0

            self.set_values([0,2, self.width / 2 +road_width*2- road_width,self.width / 2 +(road_width*2)+road_width,0, self.depth], 8/NUM_SEM_CLASSES)
            self.set_values([0, 2, self.width / 2 -(road_width*2)- road_width,self.width / 2 -(road_width*2)+ road_width, 0, self.depth],
                            8 / NUM_SEM_CLASSES)
            if self.obstacles:
                # Buildings
                self.set_values([0, 8, self.width / 2 + (road_width*2) + road_width, self.width, 0, self.depth],
                                11 / NUM_SEM_CLASSES)
                self.set_values([0, 8, 0,self.width / 2 -(road_width*2) - road_width, 0, self.depth],
                                11 / NUM_SEM_CLASSES)

                # Post
                self.set_values([0, 8,self.width/2-(road_width*2)+road_width/2,self.width/2-(road_width*2)+road_width,self.depth/2 +2,self.depth/2+2+road_width/2],
                                11 / NUM_SEM_CLASSES)
                # self.set_values(
                #     [0, 8,  self.width / 2 + (road_width*2) - road_width ,self.width / 2 + (road_width*2) - road_width/2,
                # self.depth / 2 - 4,self.depth / 2 - 4 + road_width / 2 ],
                #     11 / NUM_SEM_CLASSES)
            if self.cars:
                for p in range(self.depth ):
                    cars[p%(seq_len-1)].append([0, 8, self.width / 2 - road_width/2, self.width / 2 + road_width/2,  p-road_width/2, p+ road_width/2])



        elif pos_x==1: # One dot in the middle
            for frame in range(seq_len):
                cars.append([])
                people.append([])
            if self.pedestrians:
                people[0].append(self.create_person(self.width/2, self.depth/2 ,1))
            pos_x = 0
            pos_y = 0
            self.set_values([0,2, self.width / 2 -road_width,self.width / 2 +road_width,  self.depth/2-road_width,self.depth/2+road_width], 8/NUM_SEM_CLASSES)

            # Buildings
            if self.obstacles:
                self.set_values([0,8, 0,self.width, 0,self.depth / 2 - 1 - road_width], 11 / NUM_SEM_CLASSES)

                self.set_values([0, 8, 0, self.width, self.depth / 2 + 1 + road_width,self.depth], 11 / NUM_SEM_CLASSES)

            if self.cars:
                for p in range(self.width):
                    cars[p % (seq_len - 1)].append(
                        [0, 8, p , p + road_width,  self.depth / 2  + road_width-1, self.depth / 2 +  road_width+1])

        elif pos_x == 2: # diagonal

            for frame in range(seq_len):
                cars.append([])
                people.append([])
            i=0
            while i+road_width< self.width and  i+road_width<self.depth:
                self.reconstruction[0:2, i,i,3] = 8 / NUM_SEM_CLASSES*np.ones(2)
                if self.cars:
                    cars[i%seq_len].append( [0, 8, i, i+2,int(i+road_width*2), int(i+road_width*2)+2])
                if self.pedestrians and (i< self.width/2-(road_width*np.sqrt(2)) or i> self.width/2+(road_width*np.sqrt(2))):
                    people[min(i,seq_len-2)].append(self.create_person(i, i,1))#np.array([[0,8], [i-1,i+1], [i-1,i+1]]))
                tmp=self.reconstruction[0:2, i-road_width:i+road_width, i-road_width:i+road_width, 3]
                self.reconstruction[0:2, i-road_width:i+road_width, i-road_width:i+road_width, 3] = 8 / NUM_SEM_CLASSES * np.ones(tmp.shape)
                i+=1
            if self.pedestrians:
                for i in range(self.width/2-int(road_width*np.sqrt(2)),self.width/2+int(road_width*np.sqrt(2))):
                    people[min(i, seq_len - 2)].append(self.create_person(self.width/2-(road_width*np.sqrt(2)), i, 1))
                    people[min(i, seq_len - 2)].append(self.create_person(i,self.width/2+(road_width*np.sqrt(2)), 1))
            pos_x = 0
            pos_y = 0

            # Buildings
            if self.obstacles:
                self.set_values([0,8, self.width / 2 - road_width,self.width / 2 + road_width, self.width / 2 - road_width,self.width / 2 + road_width], 11/NUM_SEM_CLASSES)



        elif pos_x == 3:
            for frame in range(seq_len): # Fork
                cars.append([])
                people.append([])

            # middle vertical line
            self.set_values([0,2, 0,self.width/2,self.depth/2,self.depth/2+road_width], 8/NUM_SEM_CLASSES)
            if self.pedestrians:
                for i in range(3):
                    for y in range(self.width/2+road_width/2):
                        people[i].append(self.create_person(y,self.depth/2+road_width/2,1))#np.array([[0,8],[y,y], [self.depth/2,self.depth/2]]))

            # horizontal line
            self.set_values([0,2, self.width / 2,self.width / 2+road_width,self.depth / 4,self.depth*3 / 4], 8 / NUM_SEM_CLASSES)

            if self.pedestrians:
                for i in range(2):
                    for x in range(self.depth / 4,self.depth /2):
                        people[i].append(self.create_person(self.width/2+road_width/2,x+road_width/2,1))#np.array([[0,8],[self.width/2,self.width/2], [x,x]]))

                for x in range(self.depth /2, self.depth*3/4):
                    people[2].append(self.create_person(self.width/2+road_width/2,x+road_width/2,1))#[[0,8],[self.width/2,self.width/2], [x,x]]))

            # left line
            self.set_values([0, 2, self.width / 2,self.width, self.depth / 4,self.depth / 4+road_width],
                            8 / NUM_SEM_CLASSES)
            if self.pedestrians:
                for i in range(2):
                    for y in range(self.width/2,self.width ):
                        people[i].append(self.create_person(y+road_width/2,self.depth/4+road_width/2,1))#[[0,8],[y,y], [self.depth/4,self.depth/4]]))

            # right line
            self.set_values([0, 2, self.width / 2,self.width, self.depth*3 / 4,self.depth*3 / 4+road_width],
                            8 / NUM_SEM_CLASSES)
            if self.pedestrians:
                for y in range(self.width/2,self.width ):
                    people[2].append(self.create_person(y+road_width/2,self.depth*3/4+road_width/2,1))#np.array([[0,8],[y,y], [self.depth*3/4,self.depth*3/4]]))

            if self.obstacles:
                # Buildings - under halva
                self.set_values([0,8, self.width / 2+road_width,self.width, self.depth / 4+road_width,self.depth * 3 / 4],
                                11 / NUM_SEM_CLASSES)

                # upper quarter
                self.set_values([0, 8, 0,self.width/2-2,0,self.depth/2-2],11 / NUM_SEM_CLASSES)


                # vertical line
                self.set_values([0, 8, 0, self.width ,self.depth *3/ 4+road_width+2,self.depth], 11 / NUM_SEM_CLASSES)

            if self.cars:
                for p in range(self.depth):
                    cars[p % (seq_len - 1)].append(
                        [0, 8, self.width / 2 - road_width/2 , self.width / 2 , p - road_width / 2,
                         p + road_width / 2])

            pos_x = 0
            pos_y = 0

        elif pos_x == 4: # vertical
            for frame in range(seq_len):
                cars.append([])
                people.append([])
            #     cars[frame].append([0, 8, self.width / 2 - 5, self.width / 2 + 5, self.depth / 2 - 5, self.depth / 2 + 5])
                #people[frame].append(self.create_person(frame, self.depth/2, 2))

                    #np.array([[0, 8], [frame, frame + 2],[self.depth / 2 - 2, self.depth / 2 + 2]]).reshape(3, 2))
            pos_x = 0
            pos_y = 0
            self.set_values([0, 2, 0, self.width, self.depth / 2 - road_width,self.depth / 2 + road_width], 8 / NUM_SEM_CLASSES)
            #self.reconstruction[0:2, :,self.depth / 2 - road_width:self.depth / 2 + road_width, 3] = 8 / NUM_SEM_CLASSES * np.ones((2, self.width,road_width))

            # Poles
            if self.obstacles:
                self.set_values([0, 8, self.width/10-road_width/2, self.width/10+road_width/2+1, self.depth / 2 -road_width,self.depth /2], 11 / NUM_SEM_CLASSES)
            # self.reconstruction[0:8, self.width/10-road_width/2: self.width/10+road_width/2+1, self.depth / 2 -road_width:self.depth /2,3] = 11 / NUM_SEM_CLASSES * np.ones(
            #     (8, road_width, road_width))

            if self.pedestrians:
                for p in range(self.width/5):
                    people[0].append(self.create_person(p, self.depth / 2+road_width/2, 2))

                for q in range(self.depth / 2 - road_width / 2, self.depth / 2 + road_width / 2):
                    people[0].append(self.create_person(self.width * 1 / 5, q, 2))
            if self.obstacles:
                self.set_values([0, 8, 3*self.width / 10 - road_width / 2, 3*self.width / 10 + road_width / 2 + 1,
                self.depth / 2 ,self.depth / 2+road_width], 11 / NUM_SEM_CLASSES)

            if self.pedestrians:
                for p in range(self.width/5,self.width*2/5):
                    people[0].append(self.create_person(p, self.depth / 2-road_width/2, 2))

                for q in range(self.depth / 2 - road_width / 2, self.depth / 2 + road_width ):
                    people[0].append(self.create_person(self.width * 2 / 5, q, 2))
            if self.obstacles:
                self.set_values([0, 8, 7*self.width / 10 - road_width / 2, 7*self.width / 10 + road_width / 2 + 1,
                                 self.depth / 2 - road_width,self.depth / 2], 11 / NUM_SEM_CLASSES)

            if self.pedestrians:
                for p in range(self.width*3/5, self.width*4/5):
                    people[0].append(self.create_person(p, self.depth / 2+road_width/2, 2))

                for q in range(self.depth / 2 + road_width / 2, self.depth / 2 + road_width ):
                    people[0].append(self.create_person(self.width * 3 / 5, q, 2))

            if self.obstacles:
                self.set_values([0, 8,  9 * self.width / 10 - road_width / 2, 9 * self.width / 10 + road_width / 2 + 1,
                self.depth / 2,self.depth / 2 + road_width], 11 / NUM_SEM_CLASSES)

            if self.pedestrians:
                for p in range(self.width*4/5, self.width):
                    people[0].append(self.create_person(p, self.depth / 2 - road_width / 2, 2))

                for q in range(self.depth / 2 - road_width / 2, self.depth / 2 + road_width / 2):
                    people[0].append(self.create_person(self.width * 4 / 5, q, 2))

            # middle point
            if self.obstacles:
                self.set_values([0, 8, self.width / 2 - road_width / 2, self.width / 2 + road_width / 2 + 1,
                self.depth / 2- road_width/2,self.depth / 2 + road_width/2+1], 11 / NUM_SEM_CLASSES)


            if self.pedestrians:
                for p in range(self.width * 2 / 5, self.width * 3 / 5):
                    people[0].append(self.create_person(p, self.depth / 2 + road_width , 2))

            if self.cars:
                for p in range(self.width):
                    cars[p % (seq_len - 1)].append(
                        [0, 8, p , p + road_width,  self.depth / 2  + road_width-1, self.depth / 2 +  road_width+1])


        elif pos_x == 5: # diagonal. Other way around.

            for frame in range(seq_len):
                cars.append([])
                people.append([])

            i=0
            margin=road_width*np.sqrt(2) #
            upper_bound = self.width * 5 / 8-margin/2
            lower_bound = self.width * 3 / 8
            while i + road_width < self.width and i + road_width < self.depth:
                j=self.width-i-1-road_width
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
                    j = self.width - i - 1 - road_width
                    people[min(i,seq_len - 2)].append(self.create_person(self.width-lower_bound-1-road_width,lower_bound+i, 1))
                    people[min(i,seq_len - 2)].append(self.create_person(self.width - upper_bound - 1 - road_width, upper_bound + i, 1))
            if self.obstacles:
                self.set_values([0, 8, self.width *1/ 4 - road_width , self.width *1/ 4 + road_width ,
                                 self.width * 3 / 4 - road_width , self.width * 3 / 4 + road_width ], 11 / NUM_SEM_CLASSES)

                self.set_values([0, 8, self.width * 3 / 4 - road_width, self.width * 3 / 4 + road_width,
                                 self.width * 1 / 4 - road_width, self.width * 1 / 4 + road_width], 11 / NUM_SEM_CLASSES)

                self.set_values([0, 8, self.width * 1 / 2 - 2*road_width, self.width * 1 / 2-road_width/2 ,
                                 self.width * 1 / 2 -2*road_width, self.width * 1 / 2- road_width/2], 11 / NUM_SEM_CLASSES)
        elif pos_x == 6:
            for frame in range(seq_len):
                cars.append([])
                people.append([])
            num_crossings = 4
            if self.obstacles:
                self.reconstruction[0:2,:,:,3]=11 / NUM_SEM_CLASSES*np.ones((2,  self.width, self.depth))
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
            return SimpleEpisode(self.reconstruction, people, cars, pos_x, pos_y, self.gamma, seq_len,
                                 reward_weights=self.r_weights, defaultSettings=self.settings)

        elif pos_x == 7:
            for frame in range(seq_len):
                cars.append([])
                people.append([])

            num_crossings = 2
            if self.obstacles:
                self.reconstruction[0:2, :, :, 3] = 11 / NUM_SEM_CLASSES * np.ones((2, self.width, self.depth))
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
            return SimpleEpisode(self.reconstruction, people, cars, pos_x, pos_y, self.gamma, seq_len,
                                 reward_weights=self.r_weights, defaultSettings=self.settings)
        elif pos_x == 8:
            for frame in range(seq_len):
                cars.append([])
                people.append([])
            width = road_width
            num_steps=300
            if self.obstacles:
                self.reconstruction[0:2, :, :, 3] = 11 / NUM_SEM_CLASSES * np.ones((2, self.width, self.depth))
            #people[0].append(np.array([[0, 8], [self.width/2,self.width/2], [self.depth/2,self.depth/2]]))
            self.reconstruction, pos_1 , _= random_walk(self.reconstruction, width, num_steps, horizontal=True) # tensor, width, num_steps, horizontal=True, diagonal=False, dir=1
            self.reconstruction , pos_2, _= random_walk(self.reconstruction, width, num_steps, horizontal=False)
            first_p = []
            second_p = []
            previous = None


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
            return SimpleEpisode(self.reconstruction, people, cars, pos_x, pos_y, self.gamma, seq_len,
                                 reward_weights=self.r_weights, defaultSettings=self.settings)
        elif pos_x == 9:
            for frame in range(seq_len):
                cars.append([])
                people.append([])
            width = road_width
            num_steps = 300
            #self.reconstruction[0:2, :, :, 3] = 11 / NUM_SEM_CLASSES * np.ones((2, self.width, self.depth))
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
            return SimpleEpisode(self.reconstruction, people, cars, pos_x, pos_y, self.gamma, seq_len,
                                 reward_weights=self.r_weights, defaultSettings=self.settings)
        elif pos_x == 10:
            for frame in range(seq_len):
                cars.append([])
                people.append([])
            width = road_width
            num_steps = 300
            # self.reconstruction[0:2, :, :, 3] = 11 / NUM_SEM_CLASSES * np.ones((2, self.width, self.depth))
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
            return SimpleEpisode(self.reconstruction, people, cars, pos_x, pos_y, self.gamma, seq_len,
                                 reward_weights=self.r_weights, defaultSettings=self.settings)
        return SimpleEpisode(self.reconstruction, people, cars, pos_x, pos_y, self.gamma, seq_len,
                             reward_weights=self.r_weights, defaultSettings=self.settings)

    def create_person(self,y_p,x_p,w_p):
        #[self.width / 2 - 2, self.width / 2 + 2], [frame, frame + 2]]
        return np.array([[0, 8],[y_p-w_p,y_p+w_p], [x_p-w_p, x_p+w_p]] ).reshape(3, 2)

    def set_values(self, indx, val):
        tmp=self.reconstruction[indx[0]:indx[1], indx[2]:indx[3], indx[4]:indx[5],3].shape
        self.reconstruction[indx[0]:indx[1], indx[2]:indx[3], indx[4]:indx[5],3]=val*np.ones(tmp)
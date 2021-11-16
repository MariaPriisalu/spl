import sys
sys.path.append("PFNNBaseCodeAngular2")
from pfnncharacter import PFNNCharacter as Ch
from agent import SimplifiedAgent
import copy
import numpy as np
from settings import PEDESTRIAN_MEASURES_INDX

class AgentPFNN(SimplifiedAgent):

    def __init__(self, settings,  net=None, grad_buffer=None, init_net=None, init_grad_buffer=None,goal_net=None,grad_buffer_goal=None):
        super(AgentPFNN, self).__init__(settings, net, grad_buffer,init_net, init_grad_buffer,goal_net,grad_buffer_goal)
        self.PFNN = Ch("Character")
        print ("Load ")
        print(settings.pfnn_path)
        self.PFNN.init( settings.pfnn_path)  # Parameter is the assets folder. Since it is a subfolder, let it blank for this project
        self.init_pos = []
        self.detect_turning_points=self.settings.detect_turning_points
        self.distanceReachedThreshold = 0
        self.maxSpeedReached = 0
        self.deltaTime_at60fps = 0.0167  # THis is the 'normal' update time to give you 60 fps
        self.deltaTime_minSafe = self.deltaTime_at60fps * 0.99 # We detected cases where having very high FPS results in character still moving but not at the corect speed and created numerical instability
        self.skeleton_scaling_pfnn_to_carla= 1.0 #1.7 / 1.5
        self.Time_at60fps = 0
        self.Time_inMySimulation = 0  # Let's say you have 30 fps in your simulation
        print ("Time in my simulation init "+str(self.Time_inMySimulation))
        self.itr_counter = 0
        self.pos_pfnn_x = 0
        self.pos_pfnn_z = 0
        self.pos_pfnn_x_init = 0
        self.pos_pfnn_z_init = 0
        self.poseData = 0
        self.rotation_matrix = np.zeros((2, 2), np.float)
        self.inverse_rotation_matrix = np.zeros((2, 2), np.float)
        self.previous_goal = copy.deepcopy(self.pos_exact)
        self.previous_action = np.array([0, 0, 0])
        self.max_frames_to_init_pfnnvelocity = settings.max_frames_to_init_pfnnvelocity

    def initial_position(self, pos, goal, current_frame=0, vel=0, init_dir=[]):
        super(AgentPFNN, self).initial_position(pos, goal, current_frame=0, vel=0)
        # First we reset the position of the agent, set him a target position and desired speed
        # Remember: You can always change these values at runtime !
        # Remember: You can always change these values at runtime !
        dir_x = 0 if len(init_dir) < 1 else init_dir[0]
        dir_z = 0 if len(init_dir) < 2 else init_dir[1]

        # self.PFNN.resetPosAndOrientation(0, 0, 0, dir_x, dir_z)  # x,y, z (y is up !)
        # print ("Initialize position "+str(self.init_pos))

        self.init_pos = copy.deepcopy(self.pos_exact)
        self.distanceReachedThreshold = self.PFNN.getTargetReachedThreshold()
        self.maxSpeedReached = 0
        self.poseData = []
        self.Time_at60fps = 0
        self.Time_inMySimulation = 0  # Let's say you have 30 fps in your simulation
        self.rotation_matrix = np.zeros((2, 2), np.float)
        self.inverse_rotation_matrix = np.zeros((2, 2), np.float)
        self.previous_goal = copy.deepcopy(self.pos_exact)
        self.previous_action = np.array([0, 0, 0])
        self.pos_pfnn_x = 0
        self.pos_pfnn_z = 0
        self.poseData = self.PFNN.getCurrentPose()
        self.itr_counter = 0
        # print "Agent position in agent class: "+str(self.position)+" position "+str(pos)+" "+str(goal)

    def set_init_matrix(self, next_pos, episode):

        self.deltaTime_inMySimulation = episode.frame_time

        if np.linalg.norm(self.rotation_matrix) == 0:
            print ("Set init matrix rotation matrix is 0")
            #if np.linalg.norm(episode.vel_init[1:])*20>120:
            dir_y = episode.vel_init[1] / self.deltaTime_inMySimulation #(next_pos[1] - self.init_pos[1])
            dir_z = episode.vel_init[2] / self.deltaTime_inMySimulation #(next_pos[2] - self.init_pos[2])
            # else:
            #     dir_y = (next_pos[1] - self.init_pos[1]) / self.deltaTime_inMySimulation  # (next_pos[1] - self.init_pos[1])
            #     dir_z = (next_pos[2] - self.init_pos[2]) / self.deltaTime_inMySimulation  # (next_pos[2] - self.init_pos[2])

            d = np.sqrt(dir_y ** 2 + dir_z ** 2)
            x=0
            z=0

            """
            dirX_in_pfnn = dir_y
            dirZ_in_pfnn = dir_z
            currentPFNN_X, currentPFNN_Z = self.PFNN.getCurrent2DPos()
            assert currentPFNN_X == 0.0 and currentPFNN_Z == 0.0, "You are overwritting the position too which is wrong here"
            self.PFNN.resetPosAndOrientation(0, 0, 0, dirX_in_pfnn, dirZ_in_pfnn)  # x,y, z (y is up !)
            """



            speed=copy.copy(d)
            #print "Set PFNN direction distance " + str(d) + " step " + str(dir_y) + " " + str(dir_z)
            if d > 0:
                self.rotation_matrix[0, 0] = dir_y / d
                self.rotation_matrix[0, 1] = dir_z / d
                self.rotation_matrix[1, 1] = dir_y / d
                self.rotation_matrix[1, 0] = -dir_z / d
                self.inverse_rotation_matrix[0, 0] = dir_y / d
                self.inverse_rotation_matrix[0, 1] = -dir_z / d
                self.inverse_rotation_matrix[1, 1] = dir_y / d
                self.inverse_rotation_matrix[1, 0] = dir_z / d
            else:
                self.rotation_matrix[0, 0] = 1
                self.rotation_matrix[0, 1] = 0
                self.rotation_matrix[1, 1] = 1
                self.rotation_matrix[1, 0] = 0
                self.inverse_rotation_matrix[0, 0] = 1
                self.inverse_rotation_matrix[0, 1] = 0
                self.inverse_rotation_matrix[1, 1] = 1
                self.inverse_rotation_matrix[1, 0] = 0

            distInCm = (d*20)


            if  distInCm > 122: # compare in cm / second
                step=np.array([0,dir_y,dir_z])
                #print "Carry out step " + str([0,dir_y,dir_z])
                targetPos2D_x, targetPos2D_z, true_targetPos2D_x, true_targetPos2D_z = self.get_pfnn_target([0,dir_y,dir_z])

                currentPFNN_X, currentPFNN_Z = self.PFNN.getCurrent2DPos()
                #assert currentPFNN_X == 0.0 and currentPFNN_Z == 0.0, "You are overwritting the position too which is wrong here"

                self.PFNN.resetPosAndOrientation(0, 0, 0, true_targetPos2D_x, true_targetPos2D_z)  # x,y, z (y is up !)

                desiredSpeed = min(np.linalg.norm( step * 20) / self.skeleton_scaling_pfnn_to_carla, 280) # maximum 300 m/s.

                #print "PFNN desired speed(cm): " + str(desiredSpeed)

                target_rotated = np.matmul(self.inverse_rotation_matrix,
                                           np.array([targetPos2D_z-self.pos_pfnn_z_init, targetPos2D_x-self.pos_pfnn_x_init]).transpose())

                self.PFNN.setTargetPosition(targetPos2D_x, targetPos2D_z)
                # print "Target " + str(target_rotated[0]*self.skeleton_scaling_pfnn_to_carla / 20.0 + self.init_pos[1]) + " " + str(
                #     target_rotated[1] *self.skeleton_scaling_pfnn_to_carla/ 20.0 + self.init_pos[2]) + " orginal " + str(targetPos2D_x) + " " + str(
                #     targetPos2D_z)

                self.previous_action = copy.deepcopy(step) * (1 / np.linalg.norm(step))
                self.previous_goal = copy.deepcopy(next_pos)


                print(f"Setting a desired speed of {desiredSpeed} in cm/s to the agent")

                self.PFNN.setDesiredSpeed(desiredSpeed)

                # Update one tick for you environment. Your frame could be split in many small parts to update the simulation correctly
                current_speed = 0
                previous_x=0
                previous_z=0

                vel_x=0
                vel_z=0


                speedInThisTick = 0
                # print "Time in simulation "+str(self.Time_inMySimulation )+" time at 60fps"+str(self.Time_at60fps)
                switch_dir = False
                counter=0
                carryOn = 0
                while True:
                    velocityDistance = np.sqrt((vel_x-true_targetPos2D_x)**2+(vel_z-true_targetPos2D_z)**2)
                    if velocityDistance < .5 or (counter >= 20 and velocityDistance < 10) or counter>self.max_frames_to_init_pfnnvelocity: # velocity threshold is now in cm per second that's why it is increased, but let the system start up a bit in the beggining anyway
                        break


                    remainingDeltaTime = self.deltaTime_inMySimulation + carryOn
                    carryOn = 0


                    while remainingDeltaTime > 0:
                        deltaTime = min(self.deltaTime_at60fps, remainingDeltaTime)
                        remainingDeltaTime -= deltaTime

                        # It is not safe to run at a too high FPS ? Then add to carry on and move on to the next frame
                        if deltaTime < self.deltaTime_minSafe:
                            carryOn += deltaTime
                            # print(f"CarryOn {carryOn}")
                            continue

                        # Update and post update a simulation frame with the given time
                        self.PFNN.updateAnim(deltaTime)
                        self.PFNN.postUpdateAnim(deltaTime)

                        # Update the maximum speed obtained so far
                        speed = self.PFNN.getCurrentAvgSpeed()

                        # Since there can be multiple speeds in a single simulation tick, take the maximum one
                        if speedInThisTick < speed:
                            speedInThisTick = speed

                        x, z = self.PFNN.getCurrent2DPos()
                        self.pos_pfnn_x = x
                        self.pos_pfnn_z = z
                        #print "Agent pos in PFNN: x "+str(self.pos_pfnn_x)+" z "+str(self.pos_pfnn_x)

                        vel_x = (self.pos_pfnn_x -previous_x) / deltaTime
                        vel_z = (self.pos_pfnn_z-previous_z) / deltaTime

                        previous_x = copy.copy(self.pos_pfnn_x)
                        previous_z = copy.copy(self.pos_pfnn_z)

                        #print(f"Agent speed {speed},pos {x:.2f}, {z:.2f} vel {vel_x:.2f}, {vel_z:.2f} ")


                    counter=counter+1
                    # if counter > self.max_frames_to_init_pfnnvelocity:
                    #
                    #     print(f"Stopped agent vel initialization at frame {counter}. Diff vel: {velocityDistance} current vel: {vel_x,vel_z} target {targetPos2D_x, targetPos2D_z}")
                    #     break
                    # elif counter % 5 == 0:
                    #     print(f"Frame {counter} : Diff vel: {velocityDistance} current vel: {vel_x,vel_z} target {true_targetPos2D_x, true_targetPos2D_z}")
            print(f"initialized agent vel initialization at frame {self.itr_counter}")

            episode.agent_high_frq_pos[self.itr_counter, 0] = copy.copy(x)
            episode.agent_high_frq_pos[self.itr_counter, 1] = copy.copy(z)
            episode.avg_speed[self.itr_counter] = copy.copy(speed)

            self.pos_pfnn_x_init = self.pos_pfnn_x
            self.pos_pfnn_z_init = self.pos_pfnn_z

            episode.agent_pose[self.frame, :] = self.PFNN.getCurrentPose()*self.skeleton_scaling_pfnn_to_carla
            episode.agent_pose_hidden[self.frame] = self.PFNN.getPrevFrame_SecondHiddenLayerOutput()
            episode.agent_pose_hidden[self.frame][np.isnan(episode.agent_pose_hidden[self.frame])]=0
            #self.itr_counter = self.itr_counter + 1


    def stand_still(self,next_pos, vel, episode):

        #print "Stand still"
        self.set_init_matrix(next_pos, episode)
        step=np.array(vel)
        step[0] = 0
        step[1] = 0
        step[2] = 0
        return next_pos, step



    def take_step(self, next_pos, vel, episode):
        #print "Take step "+str(next_pos)
        self.set_init_matrix(next_pos, episode)
        self.pos_exact = next_pos
        self.position = np.round(self.pos_exact).astype(int)
        return next_pos

    # def not_valid_move(self, episode, next_pos, step, valid):
    #     if not self.is_agent_alive(episode):
    #         print "Agent dead"
    #         if not valid:
    #             episode.measures[self.frame, 3] = 1
    #         next_pos, step = self.stand_still(next_pos, step, episode)
    #         episode.measures[self.frame, 14] = 1
    #     else:
    #         print "Agent alive"
    #         next_pos = self.take_step(next_pos, step, episode)
    #         episode.measures[self.frame, 3] = 0
    #     return next_pos, step


    def get_pfnn_target(self, step):
        #print "Step "+str(step)
        true_targetPos2D_x = step[1] * 20/self.skeleton_scaling_pfnn_to_carla  # (next_pos[1]-self.init_pos[1])*20#0*3
        true_targetPos2D_z = step[2] * 20/self.skeleton_scaling_pfnn_to_carla  # (next_pos[2]-self.init_pos[2])*20#0*3
        target = np.array([true_targetPos2D_x, true_targetPos2D_z]).transpose()
        #print "Scaled step "+str(target)
        target_rotated = np.matmul(self.rotation_matrix, target)
        #print "Target rotated " + str(target_rotated)
        true_targetPos2D_z = target_rotated[0]
        true_targetPos2D_x = target_rotated[1]
        #print "PFNN target from currect pos x "+str(true_targetPos2D_x+ self.pos_pfnn_x)+" z: "+str(true_targetPos2D_z+ self.pos_pfnn_z)
        targetPos2D_x = true_targetPos2D_x * 1000 + self.pos_pfnn_x  # /dist*10000
        targetPos2D_z = true_targetPos2D_z * 1000 + self.pos_pfnn_z  # / dist * 10000
        #print "PFNN target pos x " + str(targetPos2D_x) + " z: " + str(targetPos2D_z)
        return targetPos2D_x, targetPos2D_z, true_targetPos2D_x+ self.pos_pfnn_x, true_targetPos2D_z+ self.pos_pfnn_z

    def carry_out_step(self,next_pos, step, episode):
        #print "Carry out step "+str(step)+" equivalent to pos "+str(next_pos)
        targetPos2D_x, targetPos2D_z,true_targetPos2D_x, true_targetPos2D_z=self.get_pfnn_target(step)


        desiredSpeed = min(np.linalg.norm(step) * 20 /self.deltaTime_inMySimulation/self.skeleton_scaling_pfnn_to_carla, 280)
        #print "PFNN desired speed(cm): "+str(desiredSpeed)


        target_in_carla=self.pfnn_to_carla(true_targetPos2D_x, true_targetPos2D_z)
        #print "True PFNN target in CARLA: "+str(target_in_carla)
        #print( "Time in simulation "+str(self.Time_inMySimulation) +" delta "+str(self.deltaTime_inMySimulation))
        self.Time_inMySimulation = self.Time_inMySimulation + self.deltaTime_inMySimulation

        self.PFNN.setTargetPosition(targetPos2D_x, targetPos2D_z)
        self.previous_goal = copy.deepcopy(next_pos)

        self.PFNN.setDesiredSpeed(desiredSpeed)

        # Update one tick for you environment. Your frame could be split in many small parts to update the simulation correctly
        maxSpeedReached = 0

        speedInThisTick = 0
        #print "Time in simulation "+str(self.Time_inMySimulation )+" time at 60fps"+str(self.Time_at60fps)
        switch_dir=False


        self.pos_pfnn_x, self.pos_pfnn_z = self.PFNN.getCurrent2DPos()
        previous_x = copy.copy(self.pos_pfnn_x)
        previous_z = copy.copy(self.pos_pfnn_z)


        while self.Time_inMySimulation > self.Time_at60fps:

            deltaTime = min(self.deltaTime_at60fps, self.Time_inMySimulation - self.Time_at60fps)

            if deltaTime < self.deltaTime_minSafe:
                break
            # Update and post update a simulation frame with the given time
            self.PFNN.updateAnim(deltaTime)
            self.PFNN.postUpdateAnim(deltaTime)

            self.Time_at60fps = self.Time_at60fps + self.deltaTime_at60fps

            # Update the maximum speed obtained so far
            speed = self.PFNN.getCurrentAvgSpeed()
            if maxSpeedReached < speed:
                maxSpeedReached = speed

            # Since there can be multiple speeds in a single simulation tick, take the maximum one
            if speedInThisTick < speed:
                speedInThisTick = speed

            x, z = self.PFNN.getCurrent2DPos()
            self.pos_pfnn_x = x
            self.pos_pfnn_z = z

            # In CARLA coordinates
            self.pfnn_to_carla_pos_exact(x, z)
            #print "PFNN agent pos " + str(x) + " "+str(z)+" "+str(self.pos_exact[1])+" "+str(self.pos_exact[2])
            vel_x = (self.pos_pfnn_x - previous_x) / deltaTime
            vel_z = (self.pos_pfnn_z - previous_z) / deltaTime
            realSpeed = np.sqrt((vel_x ** 2) + (vel_z ** 2))


            vel_x = (self.pos_pfnn_x - previous_x) / deltaTime
            vel_z = (self.pos_pfnn_z - previous_z) / deltaTime
            realSpeed = np.sqrt((vel_x ** 2) + (vel_z ** 2))

            previous_x = copy.copy(self.pos_pfnn_x)
            previous_z = copy.copy(self.pos_pfnn_z)


            valid, valid_case=self.is_next_pos_valid(episode, self.pos_exact)
            if not valid:

                episode.measures[self.frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles] = 1
            if np.linalg.norm(step)>0 and not valid_case and not switch_dir:
                #print ("Stop")
                next_pos, step = self.stand_still(next_pos, step, episode)
                episode.measures[self.frame, PEDESTRIAN_MEASURES_INDX.agent_dead] = 1
                targetPos2D_x, targetPos2D_z, true_targetPos2D_x, true_targetPos2D_z = self.get_pfnn_target(step)
                self.PFNN.setTargetPosition(targetPos2D_x, targetPos2D_z)

                desiredSpeed = min(np.linalg.norm(step) * 20 /self.deltaTime_inMySimulation/self.skeleton_scaling_pfnn_to_carla, 280)

                self.PFNN.setDesiredSpeed(desiredSpeed)
                switch_dir=True
                #print ("PFNN agent collides changes direction: "+str(targetPos2D_x)+" "+str( targetPos2D_z)+" Current pos "+str(x)+" "+str(z))


            self.itr_counter = self.itr_counter + 1

            #print ("Frame "+str(self.itr_counter)+" time in simulation "+str(self.Time_inMySimulation)+" time in PFNN "+str(self.Time_at60fps))
            episode.agent_high_frq_pos[self.itr_counter, 0] = copy.copy(x)
            episode.agent_high_frq_pos[self.itr_counter, 1] = copy.copy(z)
            episode.avg_speed[self.itr_counter] = copy.copy(realSpeed)
            episode.agent_pose_hidden[self.itr_counter] = self.PFNN.getPrevFrame_SecondHiddenLayerOutput()
            episode.agent_pose[self.itr_counter] = self.PFNN.getCurrentPose()
            #self.itr_counter = self.itr_counter + 1


        #print "Done with loop. Time in my sim "+str(self.Time_inMySimulation)+" time in FPNN "+str(self.Time_at60fps)+" "+str(self.itr_counter)
        # Get Current position in 2D space

        episode.agent_pose_frames[self.frame + 1] = copy.copy(self.frame+1)
        episode.measures[self.frame,PEDESTRIAN_MEASURES_INDX.change_in_pose]=np.mean(self.PFNN.getCurrentPose_Stats_avgYaw()[1:10])
        # print self.PFNN.getCurrentPose_Stats_avgYaw()[1:10]
        # print "Agent frame "+str(self.frame)+" yaw "+str(episode.measures[self.frame,16])+" step: "+str(step)+" speed: "+str(desiredSpeed)
        #
        # print "PFNN agent state "+str(episode.agent_pose_hidden[self.itr_counter-1, :5])
        # print "Frame "+str(self.frame+1)+" "+str(self.itr_counter - 1)

        # sprint episode.agent_pose_frames
        x, z = self.PFNN.getCurrent2DPos()
        # print "Done with loop " + str(self.frame + 1) + " agent itr_counter: " + str(
        #     self.itr_counter) + " pos: " + str(x) + " pos: " + str(z) + " pose: " + str(sum(poseData))
        self.pos_pfnn_x = x
        self.pos_pfnn_z = z

        target_rotated = self.pfnn_to_carla_pos_exact(x, z)

        self.position = np.round(self.pos_exact).astype(int)

        #print "Agent exact position after PFNN update: " + str(self.pos_exact) + " " + str(self.position) + " vel:" + str(step)+" "+str(np.linalg.norm(step))

    def pfnn_to_carla_pos_exact(self, x, z):
        current_pos = np.array([z-self.pos_pfnn_z_init, x-self.pos_pfnn_x_init]).transpose()
        target_rotated = np.matmul(self.inverse_rotation_matrix, current_pos)
        self.pos_exact[1] = target_rotated[0] *self.skeleton_scaling_pfnn_to_carla / 20.0 + self.init_pos[1]
        self.pos_exact[2] = target_rotated[1] *self.skeleton_scaling_pfnn_to_carla / 20.0 + self.init_pos[2]
        return target_rotated

    def pfnn_to_carla(self, x, z):
        current_pos = np.array([z-self.pos_pfnn_z_init, x-self.pos_pfnn_x_init]).transpose()
        #print "Target minus init reverted "+str(current_pos)
        target_rotated = np.matmul(self.inverse_rotation_matrix, current_pos)
        # print "Target rotated " + str(target_rotated)
        # print "Init pos "+str(self.init_pos)
        x_out = target_rotated[0] *self.skeleton_scaling_pfnn_to_carla / 20.0 + self.init_pos[1]
        y_out = target_rotated[1] *self.skeleton_scaling_pfnn_to_carla / 20.0 + self.init_pos[2]
        #print "Target scaled " + str(target_rotated[0] *self.skeleton_scaling_pfnn_to_carla / 20.0)+" "+str(target_rotated[1] *self.skeleton_scaling_pfnn_to_carla / 20.0)
        return x_out, y_out

    def carla_to_pfnn(self, pos):
        pos_1 = (pos[1]-self.init_pos[1]) * 20 / self.skeleton_scaling_pfnn_to_carla  # (next_pos[1]-self.init_pos[1])*20#0*3
        pos_2 = (pos[2]-self.init_pos[1]) * 20 / self.skeleton_scaling_pfnn_to_carla  # (next_pos[2]-self.init_pos[2])*20#0*3
        position = np.array([pos_1, pos_2]).transpose()
        position_rotated = np.matmul(self.rotation_matrix, position)
        z=position_rotated[1]+ self.pos_pfnn_z_init
        x=position_rotated[0]+ self.pos_pfnn_x_init
        return x,z
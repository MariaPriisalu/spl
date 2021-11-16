from pfnncharacter import PFNNCharacter as Ch
import numpy as np 
import math
import copy

DO_VISUALIZATION = True
if DO_VISUALIZATION:
	import PFNNPythonLiveVisualizer as PFNNViz

def getBonePos(poseData, boneId):
	x = poseData[boneId*3 + 0]
	y = poseData[boneId*3 + 1]
	z = poseData[boneId*3 + 2]
	return x,y,z

def distance2D(x0, z0, x1, z1):
	return math.sqrt((x0-x1)**2 + (z0-z1)**2)

def prettyPrintBonePos(poseData, boneID):
	print("Bone {0} has position: ({1:.2f},{2:.2f},{3:.2f})".format(boneID, *getBonePos(poseData, boneID)))
	
def run_demo():
	agentInst = Ch("a")
	agentInst.init("") # Parameter is the assets folder. Since it is a subfolder, let it blank for this project
	print("Agent name: ", agentInst.getName())
	print("Agent has: ", agentInst.getNumJoints(), " joints")
	poseData = agentInst.getCurrentPose() # Returns a numpy representing the pose 
	jointsParents = agentInst.getJointParents(); # Returns the parent of each of the numjoints bones (same numpy)
	print("Printing entire initial pose numpy array: ", poseData)
	print("Joint parents ", jointsParents)

	if DO_VISUALIZATION:
		# Init the visualization engine if needed
		useTkAGG = False # Use this on True only if there are compatibility issues. For example in Pycharm you need it
		# Unfortunately it will show down rendering and processing a lot. The only option in case you need it with True
		# would be to render at a slower fps
		agentVisual = PFNNViz.PFNNLiveVisualizer(jointsParents, poseData, getBonePos, useTkAGG)

	boneID = 0

	print("====== Beginning simulation ========")
	# First we reset the position of the agent, set him a target position and desired speed
	# Remember: You can always change these values at runtime !
	agentInst.resetPosAndOrientation(0, 0, 0, 1, 1) # x,y, z (y is up !)
	targetPos2D_x = 1.11022302463e-13
	targetPos2D_z = 10165.3372132
	desiredSpeed = 270 #172.810732671
	agentInst.setTargetPosition(targetPos2D_x, targetPos2D_z)
	agentInst.setDesiredSpeed(desiredSpeed)
	enableVerboseLogging = True

	print("Speed converted internally after desired speed set ", agentInst.getInternalSystemSpeed())

	# This is the default distance to target goal that we considered and reached and stop.
	# You can overwrite it with "setTarget.." but it was tuned and default seems to be the best so be carefull
	distanceReachedThreshold = agentInst.getTargetReachedThreshold()
	maxSpeedReached = 0

	# This is the time in seconds for update time.
	# For example this value represents ~60 frames per second.
	# It can be variable in your simulation of course!
	# But since training data was captured at 60 frames per second, you'll have to split your delta time updates
	# in small units to fit on delta times that gives 60 gps.
	# This code provides example of how you need to use it.
	deltaTime_at60fps = 0.0167 # THis is the 'normal' update time to give you 60 fps
	deltaTime_minSafe = deltaTime_at60fps * 0.99 # We detected cases where having very high FPS results in character still moving but not at the corect speed and created numerical instability
	assert	deltaTime_minSafe <= deltaTime_at60fps, "Whatever you do this should be smaller than original"
	deltaTime_inMySimulation = 1/17 # Let's say you have 17 fps in your simulation

	secondHiddenLayerSize = agentInst.getSecondHiddenSizeLayerOutput()
	#print("Second hidden layer size ", secondHiddenLayerSize, "neurons.");

	previous_x=0
	previous_z=0

	frameCounter = 0 
	carryOn = 0
	while True:		
		# Update one tick for you environment. Your frame could be split in many small parts to update the simulation correctly
		remainingDeltaTime = deltaTime_inMySimulation + carryOn
		carryOn = 0

		speedInThisTick = 0
		while remainingDeltaTime > 0:
			deltaTime = min(deltaTime_at60fps, remainingDeltaTime)
			remainingDeltaTime -= deltaTime
			
			# It is not safe to run at a too high FPS ? Then add to carry on and move on to the next frame
			if deltaTime < deltaTime_minSafe:
				carryOn += deltaTime
				#print(f"CarryOn {carryOn}")
				continue

			# Update and post update a simulation frame with the given time
			agentInst.updateAnim(deltaTime)
			agentInst.postUpdateAnim(deltaTime)

			if enableVerboseLogging:
				#print("Prev phase={0:.2f}, New phase={1:.2f}".format(agentInst.getPrevFrame_Phase(), agentInst.getCurrentFrame_Phase()))
				hiddenOut = agentInst.getPrevFrame_SecondHiddenLayerOutput()
				#print("First 5 values from second hidden layer: ", hiddenOut[:5]) # secondHiddenLayerSize is total size

			# Update the maximum speed obtained so far
			speed = agentInst.getCurrentAvgSpeed()
			if maxSpeedReached < speed:
				maxSpeedReached = speed

			# Since there can be multiple speeds in a single simulation tick, take the maximum one
			if speedInThisTick < speed:
				speedInThisTick = speed

			# Get Current position in 2D space
			x, z = agentInst.getCurrent2DPos()
			#print("Agent current pos x", x, " z ", z)

			# Get Current position in 2D space
			x, z = agentInst.getCurrent2DPos()
			#print("Final agent current pos x", x, " z ", z, " previous x", previous_x, "previous z ", previous_z)
			vel_x=(x-previous_x) / deltaTime
			vel_z=(z-previous_z) / deltaTime
			
			previous_x = x
			previous_z = z
			
			#print(f"Agent speed {speed},pos {x:.2f}, {z:.2f} vel {vel_x:.2f}, {vel_z:.2f}. Dt used {deltaTime}, safe dt {deltaTime_minSafe} ")

		print("Final vel x", vel_x, " z ", vel_z)
		# Printing one of the bones. You have access to pose on each frame ! Proving just for a single bone
		poseData = agentInst.getCurrentPose()
		# if enableVerboseLogging:
		# 	prettyPrintBonePos(poseData, boneID)
		frameCounter += 1

		#if enableVerboseLogging:
			# print("------ \nFrame: {0}. Deltatime given: {1:.2f}".format(frameCounter, deltaTime_inMySimulation))
			# print("New agent pos in 2D: x={0:.2f}, z={1:.2f}. AvgSpeed={2:.2f}".format(x, z, speedInThisTick))


			# Showing the new pose stats API
			#---------------------------------
			# Some global pre-computed stats:
			# All these are  shown are computed for bones 1:10 which are really the ones interested in since they
			# affect the motion of legs:
			# JOINT_ROOT_L = 1,JOINT_HIP_L = 2, JOINT_KNEE_L = 3, JOINT_HEEL_L = 4, JOINT_TOE_L = 5, \
			# JOINT_ROOT_R = 6, JOINT_HIP_R = 7, JOINT_KNEE_R = 8,JOINT_HEEL_R = 9, JOINT_TOE_R = 10

			# print("Max of max yaws: ", agentInst.getCurrentPose_Stats_maxOfMaxYaws())
			# print("Max of avg yaws: ", agentInst.getCurrentPose_Stats_maxOfAvgYaws())
			# print("Avg of max yaws: ", agentInst.getCurrentPose_Stats_avgOfMaxYaws())
			# print("Avg of avg yaws: ", agentInst.getCurrentPose_Stats_avgOfAvgYaws())
            #
			# print("Avg angular vel: bones 1-10 ", agentInst.getCurrentPose_Stats_avgYaw()[1:10])  # Angular velocities avg and max
			# print("Max angular vel: bones 1-10 ", agentInst.getCurrentPose_Stats_maxYaw()[1:10])
			# print("Max velocities: bones 1-10 ", agentInst.getCurrentPose_Stats_maxVel()[1:10]) # Bone velocities
			# print("Avg velocities: bones 1-10 ", agentInst.getCurrentPose_Stats_avgVel()[1:10])
			# print("Min velocities: bones 1-10 ", agentInst.getCurrentPose_Stats_minVel()[1:10])

		# targetPos2D_x = float(input("Target pos x"))
		# targetPos2D_z = float(input("Target pos z"))
		# desiredSpeed= float(input("Desired speed"))
		# agentInst.setTargetPosition(targetPos2D_x, targetPos2D_z)
		# agentInst.setDesiredSpeed(desiredSpeed)
		if DO_VISUALIZATION:
			# We want to have character in the middle of the graph always. So we translate to the estimated character
			# position by his relative position on the ground
			agentVisual.updateSkeletonGraph(x, z, poseData)

		# Check if reached goal velocity:
		if abs(vel_z-10.1653372132)<0.1:
			print("Reached the goal vel after {0} frames set new goal that is the same".format(frameCounter))
			#print("Max speed obtained: ", maxSpeedReached)
			#targetPos2D_z=targetPos2D_z+1000
			# agentInst.setTargetPosition(targetPos2D_x, targetPos2D_z)
			# agentInst.setDesiredSpeed(desiredSpeed)
			#break

		previous_x=copy.copy(x)
		previous_z = copy.copy(z)
		# Check if we have reached the destination considering the threshold
		currDistToGoal = distance2D(x,z, targetPos2D_x, targetPos2D_z)
		if currDistToGoal < distanceReachedThreshold:
			print("Reached the goal after {0} frames".format(frameCounter))
			print("Max speed obtained: ", maxSpeedReached)
			break

		

if __name__ == "__main__":
	run_demo()












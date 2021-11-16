# These are agents running externally, deployind on Flask server instances.
# Check the documentation (Readme) to see where and how to clone and use these repositories

from agents_dummy import DummyAgent
import numpy as np

from flask import Flask, jsonify, request
import json
import io
import requests
import os
import ast
import subprocess
import signal
import logging
from commonUtils.ReconstructionUtils import FILENAME_PEOPLE_TRAJECTORIES

logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


VOXEL_TO_METER = 0.2
METER_TO_VOXEL = 5.0
USE_CORRECT = True

# Returns x,y,z in meters from a pos given as input in voxels
def convertPosInVoxelSpaceToMeters(posInVoxelsSpace):  # [z, y, x]
    return (posInVoxelsSpace[2] * VOXEL_TO_METER, posInVoxelsSpace[1] * VOXEL_TO_METER, posInVoxelsSpace[0] * VOXEL_TO_METER)

def convert2DPosInMetersToVoxelsSpace(posInMeters): # [x,y]
    return (0, posInMeters[1] * METER_TO_VOXEL, posInMeters[0] * METER_TO_VOXEL)  # z, y, x

# Agent SGAN model
class Agent_SGAN(DummyAgent):
    def __init__(self, settings, net=None, grad_buffer=None):
        super(Agent_SGAN, self).__init__(settings, net, grad_buffer)
        self.delegatedExternalAgent_address, _, _ = settings.getExternalModelDeploymentAddressAndPaths()
        self.NUM_FRAMES_TO_OBSERVE = 8

    def next_action(self, episode_in, goal, training=True):
        return self.next_action_external(episode_in, goal, self.pos_exact, self.frame, training, self.delegatedExternalAgent_address)

    @staticmethod
    def next_action_external(episode_in, goal, pos_exact, frame, training, deploymentAddress):
        assert(frame >= 0)
        # Step 1: Fill the agents observed on this frame:
        agentPositions = {}
        # Main agent first
        mainagent_x_meters, mainagent_y_meters, _ = convertPosInVoxelSpaceToMeters(pos_exact) # pos or pos_exact ???
        agentPositions['main'] = [mainagent_x_meters, mainagent_y_meters]

        # Find the people in the scene by their names
        people, people_names = episode_in.get_people_in_frame(frame, needPeopleNames=True)
        for i,person in enumerate(people):
            x = person[0] * VOXEL_TO_METER
            y = person[1] * VOXEL_TO_METER
            v_x =  person[2] * VOXEL_TO_METER
            v_y =  person[3] * VOXEL_TO_METER # Get person with correct id
            radius =  person[4] * VOXEL_TO_METER

            agentPositions[people_names[i]] = [x, y]

        # Step 2: Serialize the message and send to the deployed model on Flask
        dataToSend = {'agentsPosThisFrame': agentPositions}
        # Uee dataToSend = {'agentsPosThisFrame' : agentPositions, 'agentsForcedHistoryPos' : initialAgentPositions} to send history positions !
        dataToSend = json.dumps(dataToSend)  # Dumps to a json
        resp = requests.post(deploymentAddress, data=dataToSend)

        # Step 3: Get back results containing the new main agent pos
        dataResponse = resp.json()
        newAgentPos_meters = dataResponse['newMainAgentPos']
        newAgentPos_meters = ast.literal_eval(newAgentPos_meters)

        newAgent_x_meters = newAgentPos_meters[0]
        newAgent_y_meters = newAgentPos_meters[1]

        vel_x_meters = newAgent_x_meters - mainagent_x_meters
        vel_y_meters = newAgent_y_meters - mainagent_y_meters
        move = episode_in.find_action_to_direction([0, vel_y_meters, vel_x_meters], np.sqrt(vel_x_meters ** 2 + vel_y_meters ** 2))

        episode_in.action[frame] = move
        episode_in.speed[frame] = np.sqrt(vel_x_meters ** 2 + vel_y_meters ** 2)
        episode_in.velocity[frame] = np.array([0, vel_y_meters, vel_x_meters])#episode_in.actions[move] # Or should we put velocities here ???


        fixed_prob_forDecision = 0.98
        other_nonzero_probs = (1.0 - fixed_prob_forDecision) / 7.0  # because there are 8 decisions..
        episode_in.probabilities[frame, 0:9] = other_nonzero_probs
        episode_in.probabilities[frame, move] = fixed_prob_forDecision
        episode_in.loss[frame] = 0  # ??

        return episode_in.velocity[frame]

# Agent SGAN model
class Agent_STGAT(DummyAgent):
    def __init__(self, settings, net=None, grad_buffer=None):
        super(Agent_STGAT, self).__init__(settings, net, grad_buffer)
        self.delegatedExternalAgent_address, _, _ = settings.getExternalModelDeploymentAddressAndPaths()
        self.NUM_FRAMES_TO_OBSERVE = 8

    def next_action(self, episode_in, goal, training=True):
        return self.next_action_external(episode_in, goal, self.pos_exact, self.frame, training, self.delegatedExternalAgent_address)

    @staticmethod
    def next_action_external(episode_in, goal, pos_exact, frame, training, deploymentAddress):
        assert(frame >= 0)
        # Step 1: Fill the agents observed on this frame:
        agentPositions = {}
        # Main agent first
        mainagent_x_meters, mainagent_y_meters, _ = convertPosInVoxelSpaceToMeters(pos_exact) # pos or pos_exact ???
        agentPositions['main'] = [mainagent_x_meters, mainagent_y_meters]

        # Find the people in the scene by their names
        people, people_names = episode_in.get_people_in_frame(frame, needPeopleNames=True)
        for i,person in enumerate(people):
            x = person[0] * VOXEL_TO_METER
            y = person[1] * VOXEL_TO_METER
            v_x =  person[2] * VOXEL_TO_METER
            v_y =  person[3] * VOXEL_TO_METER # Get person with correct id
            radius =  person[4] * VOXEL_TO_METER

            agentPositions[people_names[i]] = [x, y]

        # Step 2: Serialize the message and send to the deployed model on Flask
        dataToSend = {'agentsPosThisFrame': agentPositions}
        # Uee dataToSend = {'agentsPosThisFrame' : agentPositions, 'agentsForcedHistoryPos' : initialAgentPositions} to send history positions !
        dataToSend = json.dumps(dataToSend)  # Dumps to a json
        resp = requests.post(deploymentAddress, data=dataToSend)

        # Step 3: Get back results containing the new main agent pos
        dataResponse = resp.json()
        newAgentPos_meters = dataResponse['newMainAgentPos']
        newAgentPos_meters = ast.literal_eval(newAgentPos_meters)

        newAgent_x_meters = newAgentPos_meters[0]
        newAgent_y_meters = newAgentPos_meters[1]

        vel_x_meters = newAgent_x_meters - mainagent_x_meters
        vel_y_meters = newAgent_y_meters - mainagent_y_meters
        move = episode_in.find_action_to_direction([0, vel_y_meters, vel_x_meters], np.sqrt(vel_x_meters ** 2 + vel_y_meters ** 2))

        episode_in.action[frame] = move
        episode_in.speed[frame] = np.sqrt(vel_x_meters ** 2 + vel_y_meters ** 2)#1 # ???????
        episode_in.velocity[frame] = np.array([0, vel_y_meters, vel_x_meters])#episode_in.actions[move] # Or should we put velocities here ???


        fixed_prob_forDecision = 0.98
        other_nonzero_probs = (1.0 - fixed_prob_forDecision) / 7.0 # because there are 8 decisions..
        episode_in.probabilities[frame, 0:9] = other_nonzero_probs
        episode_in.probabilities[frame, move] = fixed_prob_forDecision
        episode_in.loss[frame] = 0 # ??

        return episode_in.velocity[frame]

# Agent SGAN model
class Agent_SOCIALSTGCNN(DummyAgent):
    def __init__(self, settings, net=None, grad_buffer=None):
        super(Agent_SOCIALSTGCNN, self).__init__(settings, net, grad_buffer)
        self.delegatedExternalAgent_address, _, _ = settings.getExternalModelDeploymentAddressAndPaths()
        self.NUM_FRAMES_TO_OBSERVE = 8

    def next_action(self, episode_in, goal, training=True):
        return self.next_action_external(episode_in, goal, self.pos_exact, self.frame, training, self.delegatedExternalAgent_address)

    @staticmethod
    def next_action_external(episode_in, goal, pos_exact, frame, training, deploymentAddress):
        assert(frame >= 0)
        # Step 1: Fill the agents observed on this frame:
        agentPositions = {}
        # Main agent first
        mainagent_x_meters, mainagent_y_meters, _ = convertPosInVoxelSpaceToMeters(pos_exact) # pos or pos_exact ???
        agentPositions['main'] = [mainagent_x_meters, mainagent_y_meters]

        # Find the people in the scene by their names
        people, people_names = episode_in.get_people_in_frame(frame, needPeopleNames=True)
        for i,person in enumerate(people):
            x = person[0] * VOXEL_TO_METER
            y = person[1] * VOXEL_TO_METER
            v_x =  person[2] * VOXEL_TO_METER
            v_y =  person[3] * VOXEL_TO_METER # Get person with correct id
            radius =  person[4] * VOXEL_TO_METER

            agentPositions[people_names[i]] = [x, y]

        # Step 2: Serialize the message and send to the deployed model on Flask
        dataToSend = {'agentsPosThisFrame': agentPositions}
        # Uee dataToSend = {'agentsPosThisFrame' : agentPositions, 'agentsForcedHistoryPos' : initialAgentPositions} to send history positions !
        dataToSend = json.dumps(dataToSend)  # Dumps to a json
        resp = requests.post(deploymentAddress, data=dataToSend)

        # Step 3: Get back results containing the new main agent pos
        dataResponse = resp.json()
        newAgentPos_meters = dataResponse['newMainAgentPos']
        newAgentPos_meters = ast.literal_eval(newAgentPos_meters)

        newAgent_x_meters = newAgentPos_meters[0]
        newAgent_y_meters = newAgentPos_meters[1]

        vel_x_meters = newAgent_x_meters - mainagent_x_meters
        vel_y_meters = newAgent_y_meters - mainagent_y_meters
        move = episode_in.find_action_to_direction([0, vel_y_meters, vel_x_meters], np.sqrt(vel_x_meters ** 2 + vel_y_meters ** 2))

        episode_in.action[frame] = move
        episode_in.speed[frame] = np.sqrt(vel_x_meters ** 2 + vel_y_meters ** 2)#1 # ???????
        episode_in.velocity[frame] = np.array([0, vel_y_meters, vel_x_meters])#episode_in.actions[move] # Or should we put velocities here ???

        fixed_prob_forDecision = 0.98
        other_nonzero_probs = (1.0 - fixed_prob_forDecision) / 7.0 # because there are 8 decisions..
        episode_in.probabilities[frame, 0:9] = other_nonzero_probs
        episode_in.probabilities[frame, move] = fixed_prob_forDecision
        episode_in.loss[frame] = 0 # ??

        return episode_in.velocity[frame]

# Given a path with folders containing people.p file, generate in output Path the data needed for training the SGAN and STGAN models
# The outputPath will contain three folders: test, train, val split in 66 % train, 14 % test, 20 % eval
# how many  frames to skip between outputing data (warning: no interpolation)
import shutil
import pickle
def generateSGANData(inputPath, outputPath, framesSkipRate):
    assert os.path.exists(inputPath)
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
    os.makedirs(outputPath)

    # Step 1: Establish the output data split and create the folders
    input_dirs = [os.path.join(inputPath, item) for item in os.listdir(inputPath) if os.path.isdir(os.path.join(inputPath, item))]
    #print(input_dirs)

    #input_dirs = input_dirs[:10] # hack for testing
    sizeOfData = len(input_dirs)

    train_index = 0
    train_size = int(sizeOfData * 0.66)
    train_dirs = input_dirs[train_index:train_size]

    test_index = train_size
    test_size = int(sizeOfData * 0.14)
    test_dirs = input_dirs[test_index : test_index + test_size]

    eval_index = test_index + test_size
    eval_size = sizeOfData - eval_index
    eval_dirs = input_dirs[eval_index : ]

    train_folder_path = os.path.join(outputPath, "train")
    test_folder_path = os.path.join(outputPath, "test")
    eval_folder_path = os.path.join(outputPath, "val")
    os.makedirs(train_folder_path)
    os.makedirs(test_folder_path)
    os.makedirs(eval_folder_path)

    # Step 2: iterate over each of the directory, process the data and put the output data into the right folder

    globalAgentUniqueIds = 0
    agentNameToId = {} # agentIdToNumber['nameOfAgent'] = unique number as float in the order of appearance inside the simulation frames

    for index, inputFolderPath in enumerate(input_dirs):
        print(("Processing input folder ", inputFolderPath))
        inputFolderName = os.path.basename(inputFolderPath)
        # 2.1 Set the target output dir to write the processed output from the input dir
        targetOutputDir = train_folder_path
        if index >= test_index and index < test_index + test_size:
            targetOutputDir = test_folder_path
        elif index >= eval_index and index < eval_index + eval_size:
            targetOutputDir = eval_folder_path

        # 2.2 Process people.p file from the input folder
        people_path = os.path.join(inputFolderPath, FILENAME_PEOPLE_TRAJECTORIES)
        people_dict = None
        with open(people_path, 'rb') as handle:
            people_dict = pickle.load(handle, encoding="latin1", fix_imports=True)

        if people_dict == None:
            print(("Couldn't process input folder ", inputFolderPath))

        outputFile = open(os.path.join(targetOutputDir, inputFolderName + str(".txt")), "w")

        allFrameKeys = sorted(people_dict.keys())
        allFramesSize = len(allFrameKeys)
        #assert allFramesSize > 0 and isinstance(allFrameKeys[0], int)

        outputFrameCounter = -1
        for frameIndex in range(0, allFramesSize, framesSkipRate):
            frameKey = allFrameKeys[frameIndex]
            outputFrameCounter  += 1
            #print(frameKey)

            for entityName, entityData in list(people_dict[frameKey].items()):

                if entityName not in agentNameToId:
                    globalAgentUniqueIds += 1
                    agentNameToId[entityName] = globalAgentUniqueIds

                entityId = agentNameToId[entityName]

                pos = None
                if 'transform' in entityData:
                    pos = entityData['transform']
                elif 'BBMinMax' in entityData:
                    pos =(entityData['BBMinMax'][0] + entityData['BBMinMax'][1]) * 0.5

                dataToWrite = (float(outputFrameCounter), float(entityId), float(pos[0]), float(pos[1])) # just x,y value...
                strToWrite = "{0:0.1f}\t{1:0.1f}\t{2:0.2f}\t{3:0.2f}\n".format(dataToWrite[0], dataToWrite[1], dataToWrite[2], dataToWrite[3])
                outputFile.write(strToWrite)


    assert input_dirs > 5, "Not enough data !"



# Just a test for data extraction to SGAN and STGAN format and our datasets
if __name__ == "__main__":
    inCarlaPath = "localUserData/Datasets/SGAN/Carla"
    inWaymoPath = "localUserData/Datasets/SGAN/Waymo"

    outCarlaPath = "localUserData/Datasets/SGAN/Carla"
    outWaymoPath = "localUserData/Datasets/SGAN/Waymo"

    #generateSGANData(inputPath=inWaymoPath, outputPath=outWaymoPath, framesSkipRate=1)
    generateSGANData(inputPath=inCarlaPath, outputPath=outCarlaPath, framesSkipRate=1)


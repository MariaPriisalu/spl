
import os
external=True

from carla_environment import CARLAEnvironment
from environment_waymo import WaymoEnvironment
#from environment import Environment # This is for cityscapes
from agents_dummy import RandomAgent



import glob
import os
import numpy as np
from settings import run_settings
from settings import USE_LOCAL_PATHS, WAYMO_CACHE_PREFIX_EVALUATE, CARLA_CACHE_PREFIX_EVALUATE
from enum import Enum


class RL(object):
    def __init__(self, agentType):

        # Log path- where to save log file
        self.save_model_path="Models/rl_model/" if USE_LOCAL_PATHS == 0 else "localUserData/Models/rl_model/"

        self.test_counter=-1

        self.agentType = agentType

    def evaluate(self, settings):

        settings.model_path = self.save_model_path + settings.name_movie + "_" + settings.timestamp + "/model.ckpt"
        settings.save_settings("")

        sess = None

        # Start any flask instances or dependencies needed
        settings.startExternalAgent()

        if self.agentType == "random":
            agent = RandomAgent(settings)
        else:
            raise NotImplementedError

        if settings.carla:
            self.evaluate_method_carla(agent, sess, settings)
        elif settings.waymo:
            self.evaluate_method_waymo(agent, sess, settings)
        else:
            raise NotImplementedError

        # Very important to call this to kill the used resources such as flask instances
        settings.cleanExternalAgent()


    # Evaluation on CARLA
    def evaluate_method_carla(self, agent, sess, settings):
        # Path to carla test set
        filespath = settings.carla_path_test

        # Where to save statistics files.
        eval_path=settings.evaluation_path
        if os.path.exists(eval_path) == False:
            os.makedirs(eval_path)

        statics = []

        # Environment class does all of the work
        env = CARLAEnvironment(filespath, sess, None, None, None, settings)

        # Sort files by test position
        ending = "test_*" # Folder pattern of data
        filename_list = {}
        for filepath in glob.glob(filespath + ending):
            parts = os.path.basename(filepath).split('_')
            pos = int(parts[-1])
            filename_list[pos] = filepath

        """
        if settings.fastTrainEvalDebug:
            keys = list(filename_list.keys())[0:2]
            filename_list = { k : filename_list[k] for k in keys}
        """

        # Go through the validation set
        saved_files_counter = 0
        for keyIndex, keyData in enumerate(filename_list.keys()):
            filepath = filename_list[keyData]
            if keyData % settings.VAL_JUMP_STEP_CARLA == 0: # jump over every 4th file to speed up
                print(f"## Evaluating key {keyIndex} out of {len(filename_list)}")
                stats, saved_files_counter = env.evaluate(CARLA_CACHE_PREFIX_EVALUATE, filepath, agent, eval_path, saved_files_counter)
                if len(stats) > 0:
                    statics.extend(stats)
        np.save(eval_path, statics)

    def get_waymo_files(self, settings, evaluate):
        ending = "/*"

        filespath = settings.waymo_path if evaluate is False else settings.waymo_path_test # "Packages/CARLA_0.8.2/unrealistic_dataset/"

        epoch = 0
        pos=0
        filename_list = []
        # Get files to run on.

        for filepath in glob.glob(filespath + ending):
            filename_list.append(filepath)
        return epoch, filename_list, pos

    # Evaluation on WAYMO
    def evaluate_method_waymo(self, agent, sess, settings):
        # Path to Waymo test set
        filespath = settings.waymo_path_test

        # Where to save statistics files.
        eval_path = settings.evaluation_path
        if os.path.exists(eval_path) == False:
            os.makedirs(eval_path)

        statics = []

        # Environment class does all of the work
        env = WaymoEnvironment(filespath, sess, None, None, None, settings)

        # Sort files by test position
        epoch, filename_list, pos = self.get_waymo_files(settings, evaluate=True)

        # Go through the validation set
        saved_files_counter = 0
        for keyIndex, filepath in enumerate(filename_list):
            if keyIndex % settings.VAL_JUMP_STEP_WAYMO == 0:
                print(f"## Evaluating key {keyIndex} out of {len(filename_list)}")
                stats, saved_files_counter = env.evaluate(WAYMO_CACHE_PREFIX_EVALUATE, filepath, agent, eval_path, saved_files_counter)
                if len(stats) > 0:
                    statics.extend(stats)
        np.save(eval_path, statics)



def main(setting, modelName): # TODO: unify with the  external name somehow
    rl = RL(agentType=modelName)
    rl.evaluate(setting)

if __name__ == "__main__":

    evaluationModels = ["STGCNN", "SGAN", "STGAT"] # our model is [""]
    datasets = ["waymo", "carla"] #["carla", "waymo"] # ["carla", "waymo", "cityscapes"]
    pfnnOptions = [True]
    for evaluatedModel in evaluationModels:
        for dataset in datasets:
            for pfnnOption in pfnnOptions:
                print(("========== RUNNING MODEL: ", evaluatedModel, " on dataset: ", dataset))
                setup = run_settings(evaluate=True, likelihood=False, evaluatedModel=evaluatedModel, datasetToUse=dataset, pfnn=pfnnOption)
                np.random.seed(setup.random_seed_np)
                main(setup, evaluatedModel)




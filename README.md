# Official Repository of Semantic Synthesis of Pedestrian Locomotion and Generating Scenarios with Diverse Pedestrian Behaviors for Autonomous Vehicle Testing #

Code for the papers _Semantic Synthesis of Pedestrian Locomotion_ published at Asian Conference on Computer Vision (ACCV) 2020 
and _Generating Scenarios with Diverse Pedestrian Behaviors for Autonomous Vehicle Testing_ to be published at Conference on Robot Learning (CoRL) 2021. 

You can find the paper _Semantic Synthesis of Pedestrian Locomotion_ [here](https://openaccess.thecvf.com/content/ACCV2020/html/Priisalu_Semantic_Synthesis_of_Pedestrian_Locomotion_ACCV_2020_paper.html), a spotlight video [here](https://youtu.be/xRdbkPtF7SU), and a presentation of the paper [here](https://accv2020.github.io/miniconf/poster_246.html).

**Authors:** [Maria Priisalu](http://www.maths.lth.se/sminchisescu/research/profile/7/maria-priisalu), [Ciprian Paduraru](https://scholar.google.com/citations?user=EaAekU4AAAAJ&hl=en), [Aleksis Pirinen](https://aleksispi.github.io/) and [Cristian Sminchisescu](http://www.maths.lth.se/sminchisescu/)

You can find the paper _Generating Scenarios with Diverse Pedestrian Behaviors for Autonomous Vehicle Testing_ [here](https://openreview.net/forum?id=HTfApPeT4DZ) with the ![poster](https://github.com/MariaPriisalu/spl/blob/master/Corl-poster-final.png?raw=true).

**Authors:** [Maria Priisalu](http://www.maths.lth.se/sminchisescu/research/profile/7/maria-priisalu), [Aleksis Pirinen](https://aleksispi.github.io/), [Ciprian Paduraru](https://scholar.google.com/citations?user=EaAekU4AAAAJ&hl=en) and [Cristian Sminchisescu](http://www.maths.lth.se/sminchisescu/)


### Overview
This repository contains code for training an RL-agent on the 3D point-clouds of cityscapes.
The RL-logic and agent are in the folder `RL`.
The policy gradient network with two 3D convolutional layers can be found in the class `net.py`.
The agent's logic (moving after an action) can be found in the abstract class `agent.py`.


The class `Episode` is a container class. It contains the agent's actions, positions, and reward for the length of an episode.
The class `Environment` goes through different episode set-ups (different environments) and applies the agent in the method `work`.
Currently this class also contains the visualization of the agent.

The class `test_episode.py` contains unit tests for the initialization of an episode, and the different parts of the reward.
The class `test_tensor` contains unit tests for the 3D reconstruction needed for environment.

Other folders:
 `reconstruct 3D`: Reconstruction of the 3D RGB and Segmentation in cityscapes.
 `utils`: various utility functions

### Running the code

To train a new model: from the main directory, type:
```
python RL/RLmain.py
```
To evaluate a model: from the main directory, type:
```
python RL/evaluate.py
```

To visualize results insert the timestamp of the run into RL/visualization_scripts/Visualize_evaluation.py
```
timestamps=[ "2021-10-27-21-14-35.842985"]
```
and run the file with 
```
python RL/visualization_scripts/Visualize_evaluation.py
```
from spl/.

### Licence
This work utilizes [CARLA](https://github.com/carla-simulator/carla) (MIT Licence), [Waymo](https://github.com/waymo-research/waymo-open-dataset) (Apache Licence), [PFNN](https://github.com/sreyafrancis/PFNN) (free for academic use), [COLMAP](https://colmap.github.io/license.html) (new BSD licence), [Cityscapes](https://github.com/mcordts/cityscapesScripts) (the Cityscapes licence - free for non-profit use), [GRFP](https://github.com/D-Nilsson/GRFP) (free for academic use) as well as the list of libraries in the yml file. 
The requirements of the licences that the work builds upon apply. Note that different licences may apply to different directories. We wish to allow the use of our repository in academia freely or as much as allowed by the licences of our dependencies. We provide no warranties on the code.


If you use this code in your academic work please cite one of the works,
```
@inproceedings{Priisalu_2020_ACCV,
    author    = {Priisalu, Maria and Paduraru, Ciprian and Pirinen, Aleksis and Sminchisescu, Cristian},
    title     = {Semantic Synthesis of Pedestrian Locomotion},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {November},
    year      = {2020}
}
```
```
@inproceedings{Priisalu_2021_CoRL,
    author    = {Priisalu, Maria and Pirinen, Aleksis and Paduraru, Ciprian and Sminchisescu, Cristian},
    title     = {Generating Scenarios with Diverse PedestrianBehaviors for Autonomous Vehicle Testing},
    booktitle = {PMLR: Proceedings of CoRL 2021},
    month     = {November},
    year      = {2021}
}
```

### Prerequisites
The models have been trained on Ubuntu 16.04, Tensorflow 1.15.04 (see official prerequists), python 3.7.7 on Nvidia Titan Xp.
Please see the yml file for the exact anaconda environment. 
Note that pfnncharacter is a special library and must be installed by hand. See below.

### Installation
Install the conda environment according to environment.yml
To install pfnncharacter enter commonUtils/PFNNBaseCode and follow the README.

# Datasets
## CARLA
In CARLA 0.8.2 or 0.9.4 you can gather the equivalent dataset with the scripts in CARLA_simulation_client.

## Cityscapes
To create the Cityscapes dataset you need to have downloaded the cityscapes dataset, and you need to attain the semantic segmentation and bounding boxes of each image. After this the scripts in colmap can be used to create 3d reconstructions of the dataset.  

## Waymo dataset
Please check the documentation of the fork here to understand how to use the pipeline to get out the dataset needed for training/visualization [here](https://github.com/AGAPIA/waymo-open-dataset) .Also it currently depends on the segmentation from here: https://github.com/AGAPIA/semantic-segmentation-pytorch 

Use script RL/visualization_scripts/show_real_reconstruction_small.py for visualization purposes and an example on how to initialize correctly data data for setting up the environment.
### Caching mechanism
The caching of episodes data happens in environment_abstract.py, inside set_up_episode function. This means that the first time around may be slow, but once the dataset is cached everything will run faster.

```
for each epoch E:
   for each scenerio in dataset S:
      episode = set_up_episode(S)
      .....
```
# Models 
## Semantic Pedestrian Locomotion Models
Models mentioned in the ACCV paper can be found in localData/Models/SemanticPedestrian.

There are two CARLA agents: a goal free generative/forecasting model (train_pfnn_pose_penalized_2020-06-30-22-25-50.394548) and a goal-driven generative model(train_carla_goal_new_reward_pfnn_new_penalty_2020-07-04-23-24-16.893992).

On the Waymo dataset there is one agent: a goal free generative/forecasting model (agent_waymo_goal_free_pfnn_2020-07-05-15-15-49.541581).

On the Cityscapes dataset there is one agent: goal-driven generative model (train_cityscapes_pfnn_angular_old_penalty_2020-07-04-17-52-32.011443).


Settings in RL/setting.py
Include the folder name of the weights you want to use in model. The model is searched for th path in settings.img_dir.
Remember to set the goal_dir=True in RL/setting.py for goal reaching agents and to goal_dir=False for goal free agents.
To just run the SPL pedestrian agent make sure the following settings are False: 
```
    self.learn_init = False # Do we need to learn initialization
    self.learn_goal = False # Do we need to learn where to place the goal of the pedestrian at initialization
    self.learn_time = False
    ...
    self.useRealTimeEnv = False
    self.useHeroCar = False
    self.useRLToyCar = False
```
When running on carla set carla=True and waymo=False. 
When running on waymo set carla=False and waymo=True. 
When running on cityscapes set carla=False and waymo=False. 

## Adversarial Test Synthesizer Models
Models mentioned in the CoRL paper can be found in localData/Models/BehaviorVariedTestCaseGeneration.


To instead train the Adversarial Test Synthesizer make sure that the following settings are True:
```    
    self.learn_init =True # Do we need to learn initialization
    self.learn_goal=False # Do we need to learn where to place the goal of the pedestrian at initialization
    self.learn_time=False
    ...
    self.useRealTimeEnv = True
    self.useHeroCar = True
    self.useRLToyCar = False
```



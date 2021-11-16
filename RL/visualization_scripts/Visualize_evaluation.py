import numpy as np
import os.path
external=True
import sys
sys.path.append("RL/")
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
print(sys.path)

import matplotlib.pyplot as plt
from RL.environment_test import TestEnvironment
from RL.settings import run_settings, USE_LOCAL_PATHS, STATISTICS_INDX,PEDESTRIAN_MEASURES_INDX,STATISTICS_INDX_CAR,CAR_MEASURES_INDX,PEDESTRIAN_INITIALIZATION_CODE, STATISTICS_INDX_MAP, STATISTICS_INDX_CAR_INIT, STATISTICS_INDX_MAP_STAT
import os
import glob, subprocess
import traceback
import pickle, sys
import scipy.io
import scipy.stats as stats
np.set_printoptions(precision=2)
######################################################################## Change values here!!!
general_id = 0 # id for average initialization

# Save plots?
save_plots=True # save gradient plots?
save_regular_plots=True # save all plots?
toy_case=False # Toy environment: if waymo, carla or cityscapes then F

make_movie=False # Save movie?
save_mat=False # Save mat file with pedestrian prediction error?
car_people=False # Plot cars and people together
supervised=False # Is this a supervised agent i.e. trained with supervised loss
print_values=False

# insert timestamp here

timestamps=[ "2021-10-27-21-14-35.842985"]
ending=''

# Should distance moved be re-calculated?
calc_dist=True
#################

# Set some paths
settings=run_settings()
settings_path=settings.path_settings_file
settings_ending="settings.txt"

if USE_LOCAL_PATHS == 0:
    stat_path = "Results/statistics/train/"
else:
    stat_path = "localUserData/statistics/train"

# Pedestrian actions
actions = []
v = [-1, 0, 1]
j=0
for y in range(3):
    for x in range(3):

        actions.append(np.array([0, v[y],v[x]]))
        j += 1

actions_names = ['upL', 'up', 'upR', 'left', 'stand', 'right', 'downL', 'down', 'downR']
# Semantic labels
labels = ['human hist','car hist' ,'unlabeled' , 'ego vehicle', 'rectification border' ,'out of roi' , 'static' ,
          'dynamic','ground', 'road' , 'sidewalk', 'parking', 'rail track' , 'building' ,  'wall', 'fence' ,
          'guard rail', 'bridge' , 'tunnel' , 'pole'   , 'polegroup', 'traffic light' , 'traffic sign' ,
          'vegetation','terrain' , 'sky','person','rider', 'car'  ,'truck', 'bus',  'caravan', 'trailer',
          'train', 'motorcycle' ,'bicycle', 'license plate' ]


labels_indx = [11,13,4,17,7,8,21,12,20]
labels_mini=[]
labels_mini.append("R")
labels_mini.append("G")
labels_mini.append("B")
labels_mini.append('people-t')
labels_mini.append('cars-t')
if car_people:
    labels_mini.append('')
    labels_mini.append('cars and people')
else:
    labels_mini.append('people')
    labels_mini.append('cars')

for label in labels_indx:
    #print labels[label+2]
    labels_mini.append(labels[label+2])
labels_mini.append("prior")
epoch=3000# update_frequency*100
make_vid=False
test_not_in_name=False
train_nbrs = [6, 7, 8, 9]  # , 0,2,4,5]
test_nbrs = [3, 0, 2, 4, 5]

init_names= {-1: "training", 0: "average", 1: "On ped.",2: "By car",3:"In front of ped.",4: "Random",5: "In front of car", 6: "Near ped.",7:"ped. env.", 8: "On pavement", 9: "On ped.", 10:"Near obstacles",11:"average"}


################################################################################

# General function calculating a moving average
def trendline(data):
    trend_avg = np.zeros(len(data))
    trend_avg[0] = data[0]
    for i in range(1, len(data)):
        trend_avg[i] = (trend_avg[i - 1] * (i - 1.0) + data[i]) / i
    return trend_avg

# function that gather various statistics from data
def evaluate(statistics,continous, rewards, hit_objs, ped_traj, people_hit, entropy,dist_left, dist_travelled, car_hit, pavement,
             people_heatmap, successes, norm_dist_travelled ,nbr_direction_switches, losses, one_step_error, one_step_errors ,
             prob_of_max, most_freq_action, freq_most_freq_action, variance, speed_mean, speed_var, likelihood_actions,
             likelihood_full, prediction_error, dist_to_car,time_to_collision, init_dist_to_goal, goal_locations,
             goal_times, init_locations, goal_speed, init_dist_to_car,speed_to_dist_correlation,collision_between_car_and_agent,
              car_case=False):

    stat_indx = STATISTICS_INDX
    if car_case:
        #print(" car case")
        stat_indx = STATISTICS_INDX_CAR
    agent_pos = statistics[:, :, stat_indx.agent_pos[0]:stat_indx.agent_pos[1]]
    # print "Agent positions "
    # print agent_pos[0,:]
    # print "Velocities "
    # print statistics[0, :, STATISTICS_INDX.velocity[0]:STATISTICS_INDX.velocity[1]]
    if len(rewards)==0:
        print(str(agent_pos.shape[1]))
    if car_case:
        agent_probabilities = statistics[:, :, stat_indx.probabilities]
    else:
        agent_probabilities = statistics[:, :, stat_indx.probabilities[0]:stat_indx.probabilities[1]]
    agent_reward = statistics[:, :, stat_indx.reward]
    agent_loss = statistics[:, :, stat_indx.loss]
    agent_action = statistics[:, :, stat_indx.action]
    agent_speed=statistics[:, :, stat_indx.speed]

    if not car_case:
        agent_likelihood=agent_probabilities[:,:,15]
        agent_likelihood_full = agent_probabilities[:,:,17]
        prediction_errors =agent_probabilities[:,:,18]*0.2

    goal_locations.append(statistics[:, 3:5, stat_indx.goal])
    init_locations.append(agent_pos[:,0,1:])
    # print ("Add goal locations "+str(goal_locations[-1].shape))
    # print ("Add init locations " + str(init_locations[-1].shape))
    goal_time=statistics[:, 5, stat_indx.goal]
    goal_times.append([np.mean(goal_time, axis=0), np.std(goal_time, axis=0)])
    #print ("Add goal times " + str(goal_times[-1].shape))

    # for frame in range(agent_pos.shape[1]):
    #     print "Speed: "+str(statistics[0, frame, 37])+" vel: "+str(statistics[0, frame, 3:6])+" pos: "+str(agent_pos[0, frame,:])+" frame "+str(frame)




    agent_reward_d = statistics[:, :, stat_indx.reward_d]

    # Mask out dead or successful agents.
    agent_measures = statistics[:, :, stat_indx.measures[0]:stat_indx.measures[1]]
    mask=np.ones_like(agent_reward_d)
    mask_2 = np.ones_like(agent_reward_d)

    #divisor = np.sum(mask, axis=1)
    for ep_itr in range(mask.shape[0]):
        for frame in range(mask.shape[1]):
            if (agent_measures[ep_itr, frame, PEDESTRIAN_MEASURES_INDX.goal_reached]==1 or agent_measures[ep_itr, frame, PEDESTRIAN_MEASURES_INDX.hit_by_car]>0) and agent_reward[ep_itr, frame]==0:#(agent_measures[ep_itr, frame,PEDESTRIAN_MEASURES_INDX.agent_dead]>0) and agent_reward[ep_itr, frame]==0:
                mask[ep_itr, frame]=0
                if agent_measures[ep_itr, frame, 0]>0:
                    mask_2[ep_itr, frame]=0
    base_divisor=np.sum(mask, axis=1)
    divisor = np.maximum(base_divisor, np.ones_like(base_divisor))

    losses.append([np.mean(agent_loss), np.std(agent_loss)])

    one_error=agent_measures[:,:,PEDESTRIAN_MEASURES_INDX.one_step_prediction_error]#*mask
    one_step_error.append([np.mean(one_error*0.2), np.std(one_error*.2)])
    one_step_errors.append(one_error*0.2)

    likelihoods=[]
    likelihoods_full=[]
    prediction_error_local=[]
    cum_reward=[]
    if not car_case:
        for ep_itr in range(mask.shape[0]):
            likelihoods.append(0)
            likelihoods_full.append(0)
            cum_reward.append(0)

            error_list=[]
            for frame in range(mask.shape[1]):

                if agent_likelihood[ep_itr, frame]>0:
                    likelihoods[-1]-=np.log(agent_likelihood[ep_itr, frame])
                    likelihoods_full[-1]-=agent_likelihood_full[ep_itr, frame]
                if mask[ep_itr, frame]:
                    error_list.append(prediction_errors[ep_itr, frame])
                    max_action=np.argmax(agent_probabilities[ep_itr, frame, :9])
                    # if frame +1< mask.shape[1]:
                    #     if np.linalg.norm(agent_pos[ep_itr, frame])>0:
                    #         if frame>0:
                    #             speed=np.linalg.norm(agent_pos[ep_itr, frame-1]-agent_pos[ep_itr, frame])
                    #         else:
                    #             speed=1
                    #         vel=actions[max_action]
                    #
                    #         vel=vel*speed
                    #
                    #         error_list.append(0.2*np.linalg.norm(agent_pos[ep_itr, frame]+vel-agent_pos[ep_itr, frame+1]))
                    cum_reward[-1]+=((agent_measures[ep_itr, frame, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]+agent_measures[ep_itr, frame, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory])*0.01)-(2*agent_measures[ep_itr, frame, PEDESTRIAN_MEASURES_INDX.hit_by_car])-(0.02*agent_measures[ep_itr, frame, PEDESTRIAN_MEASURES_INDX.hit_obstacles])-(0.1*agent_measures[ep_itr, frame, PEDESTRIAN_MEASURES_INDX.hit_pedestrians])
            prediction_error_local.append(np.mean(error_list))
        likelihood_actions.append([np.mean(agent_likelihood), np.std(agent_likelihood)])
        likelihood_full.append([np.mean(likelihoods_full), np.std(agent_likelihood_full)])
        prediction_error.append([np.mean(prediction_error_local), np.std(agent_likelihood_full)])





    clipped_measures = agent_measures[:, 0, PEDESTRIAN_MEASURES_INDX.dist_to_final_pos]
    mask_d = clipped_measures > 0
    #mask_d= np.logical_and(mask_d,mask)
    #val = dist_left_l[mask_d]

    dist_global = np.zeros((agent_measures.shape[0]))
    dist_tot = np.zeros((agent_measures.shape[0]))
    seq_lens = np.zeros((agent_measures.shape[0]))
    dist_left_l=np.zeros((agent_measures.shape[0]))
    dist_to_goal=np.zeros((agent_measures.shape[0]))
    for i in range(agent_measures.shape[0]):
        dist_global[i] = 0
        seq_len = 0
        while seq_len + 1 < mask.shape[1] and mask[i, seq_len + 1] == 1:
            seq_len = seq_len + 1
        seq_lens[i] = seq_len
        # print mask[i, :]
        # print seq_len
        dist_global[i] = np.linalg.norm(agent_pos[i, seq_len, 1:] - agent_pos[i, 0, 1:]) * 0.2
        # if dist_global[i]==0:
        #     print seq_lenF
        dist_tot[i] = agent_measures[i, seq_len, PEDESTRIAN_MEASURES_INDX.total_distance_travelled_from_init]
        dist_left_l[i]=np.min(agent_measures[i, :seq_len+1, PEDESTRIAN_MEASURES_INDX.dist_to_goal])
        dist_to_goal[i]=agent_measures[i, 0, PEDESTRIAN_MEASURES_INDX.dist_to_goal]

        # print "dist glob "+str(dist_global[i])
        # print "dist tot " + str(dist_tot[i])
        # print "dist left " + str(dist_left_l[i])

    val = dist_left_l[mask_d]
    if agent_measures.shape[2] >= 13:
        suc = np.max(agent_measures[:, :, PEDESTRIAN_MEASURES_INDX.goal_reached]*mask, axis=1)
        successes.append(np.sum(suc) * 1.0 / len(suc))
    else:
        successes.append(np.sum(val < np.sqrt(2)) * 1.0 / len(val))

    dist_left.append([np.mean(val * 0.2), np.std(val * 0.2)])
    init_dist_to_goal.append([np.mean(dist_to_goal* 0.2), np.std(dist_to_goal* 0.2)])
    local_speed=dist_to_goal/goal_time
    goal_speed.append([np.mean(local_speed), np.std(local_speed)])

    avg_reward = np.mean(agent_reward, axis=1)

    std_reward = np.std(agent_reward, axis=1)

    # Reward
    rew_sum = np.sum(agent_reward*mask, axis=1)
    rewards.append([np.mean(rew_sum), np.std(rew_sum)])
    if not continous:

        most_freq_action_m = stats.mode(agent_action, axis=None)
        most_freq_action.append(most_freq_action_m[0])
        freq_most_freq_action.append(most_freq_action_m[1] *1.0/ len(agent_action))
        #print most_freq_action_m
        # most_freq_action[1]
        if not car_case:
            prob_of_max.append(np.mean(np.max(agent_probabilities, axis=2)))
        else:
            prob_of_max.append(np.mean(agent_probabilities))
    else:
        mean_x=agent_probabilities[:,:,0]
        mean_y=agent_probabilities[:,:,1]
        var=agent_probabilities[:,:,3]
        freq_most_freq_action.append(np.std(mean_x))
        most_freq_action.append(np.std(mean_y))
        entropy.append(np.mean(mean_x))
        prob_of_max.append(np.mean(mean_y))
        variance.append(np.mean(var))

    # Hit objects
    dist_global_temp=dist_global
    dist_global_temp[dist_global_temp==0]=1
    tmp = np.sum(agent_measures[:, :, PEDESTRIAN_MEASURES_INDX.hit_obstacles]*mask, axis=1)*1.0/(dist_global_temp*divisor)#*np.sum(mask, axis=1))

    hit_objs.append([np.mean(tmp), np.std(tmp)])
    if car_case:
        # Entropy
        entropy.append([np.mean(agent_probabilities), np.std(agent_probabilities)])
    else:
        hit_by_hero_car=np.sum(agent_measures[:,:, PEDESTRIAN_MEASURES_INDX.hit_by_hero_car]*mask, axis=1)>0
        not_hit_by_hero_car=hit_by_hero_car==0
        calculate_entropy_of=agent_probabilities
        episode_mask=np.sum(mask, axis=1)
        collision_mask=episode_mask*hit_by_hero_car
        non_collison_mask=episode_mask*not_hit_by_hero_car
        action_len = 9
        import scipy.stats
        #car_dist = agent_measures[:, 0, CAR_MEASURES_INDX.dist_to_agent]
        entropy_t = np.zeros(calculate_entropy_of.shape[0])
        for i, episode in enumerate(calculate_entropy_of):
            for f,frame in enumerate(episode):
                if mask[i,f]==1:
                    entropy_t[i] += scipy.stats.entropy(frame[0:action_len])
            # print "Final entropy "+str(entropy_t)
        if print_values:
            print(" Entropy "+str(np.histogram(entropy_t)))
            print("Entropy mean "+str(np.mean(entropy_t/np.sum(mask)))+" std "+str(np.std(entropy_t/np.sum(mask)))+ " previous mean "+str(np.mean(entropy_t/calculate_entropy_of.shape[1]))+" std "+str(np.std(entropy_t/calculate_entropy_of.shape[1])))
        entropy.append([np.mean(entropy_t/calculate_entropy_of.shape[1]), np.std(entropy_t/calculate_entropy_of.shape[1])])
        collision_entropy=entropy_t[hit_by_hero_car]
        if print_values:
            print(" Collision  Entropy " + str(np.histogram(collision_entropy)))
            print("Collision  Entropy mean " + str(np.mean(collision_entropy / np.sum(collision_mask))) + " std " + str(
                np.std(collision_entropy / np.sum(collision_mask))) )
        #entropy.append([np.mean(entropy_t), np.std(entropy_t)])
        non_collision_entropy=entropy_t[not_hit_by_hero_car]
        if print_values:
            print("Non Collision  Entropy " + str(np.histogram(non_collision_entropy)))
            print("Non Collision  Entropy mean " + str(np.mean(non_collision_entropy / np.sum(non_collison_mask))) + " std " + str(np.std(non_collision_entropy / np.sum(non_collison_mask))))

    import math
    # Dist travelled
    for epoch in range(agent_measures.shape[0]):
        for frame in range(1,agent_measures.shape[1]):
            dist = np.linalg.norm(agent_pos[epoch, frame, 1:] - agent_pos[epoch, frame-1,1:])
            # if dist>math.sqrt(2):
            #     print "Larger dist "+str(frame)+" "+str(dist)+" "+str(agent_pos[epoch, frame-1, 1:])+" "+str(agent_pos[epoch, frame, 1:])
    tmp = agent_measures[:, -1, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init] * 0.2

    if car_case:
        tmp2 = np.sum(agent_measures[:, :, PEDESTRIAN_MEASURES_INDX.hit_pedestrians] * mask, axis=1) > 0
        # print(" Distance travelled when car hit ")
        # print( agent_measures[np.argmax(tmp2), :, PEDESTRIAN_MEASURES_INDX.distance_travelled_from_init])
        if print_values:
            print("Distance travelled " + str(np.histogram(tmp)))
            print(" Distance travelled mean "+str(np.mean(tmp))+" std "+str(np.std(tmp)))
        dist_travelled_by_collision=tmp[tmp2]
        if print_values:
            print("Distance travelled collisions: " + str(np.histogram(dist_travelled_by_collision)))
            print(" Distance travelled collisions mean " + str(np.mean(dist_travelled_by_collision)) + " std " + str(np.std(dist_travelled_by_collision)))
        dist_travelled_not_by_collision=tmp[tmp2==0]
        if print_values:
            print("Distance travelled no collisions: " + str(np.histogram(dist_travelled_not_by_collision)))
            print(" Distance travelled no collisions mean " + str(np.mean(dist_travelled_not_by_collision)) + " std " + str(
                np.std(dist_travelled_not_by_collision)))

    #dist_global=np.linalg.norm(agent_pos[:,-1,1:]-agent_pos[:,0,1:], axis=1)*0.2

    dist_travelled.append([np.mean(dist_global), np.std(dist_global)])

    # On pedestrian trajectory
    tmp = np.sum(agent_measures[:, :, PEDESTRIAN_MEASURES_INDX.frequency_on_pedestrian_trajectory]*mask, axis=1)/divisor#np.sum(mask, axis=1)
    ped_traj.append([np.mean(tmp), np.std(tmp)])

    # Number of people hit
    tmp = np.sum(agent_measures[:, :, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]*mask, axis=1)*1.0/divisor#np.sum(mask, axis=1)
    if car_case:
        tmp2=np.sum(agent_measures[:, :, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]*mask, axis=1)
        non_zeros=tmp2>0
        only_non_zeros=tmp2[non_zeros]
        if print_values:
            print("Number of people hit "+str(sorted(tmp[non_zeros]*divisor[non_zeros])))
            print("Previous: Mean value "+str(np.mean(tmp))+" std "+str(np.std(tmp))+" New: non-zeros "+str(np.sum(non_zeros))+" of "+str(len(tmp))+": "+str(np.sum(non_zeros)/len(tmp))+" std of non-zeros "+str(np.std(only_non_zeros))+"  mean: "+str(np.mean(only_non_zeros)))
        #print(agent_measures[:, :, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]*mask)
        non_zeros_new = (agent_measures[:, :, PEDESTRIAN_MEASURES_INDX.hit_pedestrians]*mask) > 0
        earliest_collisons=non_zeros_new.argmax(axis=1)
        earliest_collisons_only=earliest_collisons[earliest_collisons>0]
        car_dist = agent_measures[:, 0, CAR_MEASURES_INDX.dist_to_agent]
        car_dist_collision=car_dist[non_zeros]
        if print_values:
            print ("Earliest collision: mean "+str(np.mean(earliest_collisons_only))+" std "+str(np.std(earliest_collisons_only)))
            print(" values "+str(sorted(earliest_collisons_only)))
            print(" Initial distance at collision: mean " + str(np.mean(car_dist_collision)) + " std " + str(np.std(car_dist_collision)) )
            print(" Initial distance at collision " + str(sorted(car_dist_collision)))

    people_hit.append([np.mean(tmp), np.std(tmp)])

    if car_case:
        tmp = np.sum(agent_measures[:, :, CAR_MEASURES_INDX.hit_by_agent]*mask, axis=1)*1.0/divisor
        collision_between_car_and_agent.append([np.mean(tmp), np.std(tmp)])

    # Distance to left to goal

    #for frame in range(agent_measures.shape[1]):

    #data_holder[-1].append(np.mean(val*0.2))
    # dist_left[clipped_measures>seq_len_train]=dist_left[clipped_measures>seq_len_train]- (clipped_measures[clipped_measures > seq_len_train]-seq_len_train)
    # clipped_measures[clipped_measures > seq_len_train] = seq_len_train
    # clipped_measures[clipped_measures ==0]=seq_len_train

    # people_hit = dist_left/ clipped_measures
    # dist_left_t = agent_measures[:, 0, 7]
    #val = dist_left_t[mask]



    # Normalized distance
    if agent_measures.shape[2]>=13:
        mask_d=dist_global>0

        norm_dist=dist_global
        norm_dist[mask_d]=np.divide(dist_tot[mask_d],dist_global[mask_d]*5)
        #print norm_dist
        norm_dist_travelled.append([np.nanmean(norm_dist), np.nanstd(norm_dist)])

    # Actions
    if agent_measures.shape[2] >= 12:
        num_switches = np.sum(agent_measures[:, :, PEDESTRIAN_MEASURES_INDX.change_in_direction]*mask, axis=1)*1/divisor#np.sum(mask, axis=1)
        nbr_direction_switches.append([np.mean(num_switches), np.std(num_switches)])

    # car hit
    car_hits = agent_measures[:, :, PEDESTRIAN_MEASURES_INDX.hit_by_car]*mask
    car_hits[car_hits>0]=1
    timestep_to_collision=np.nonzero(car_hits)[1]
    time_to_collision.append([np.mean(timestep_to_collision), np.std(timestep_to_collision)])
    #init_dist_to_goal

    # print (car_hits)
    # print ("Sum "+str(np.sum(car_hits))+" divide by "+str(mask.shape[0]))
    car_hit.append(np.sum(car_hits)*1.0/mask.shape[0])
    if not car_case:
        car_dist=np.reciprocal(agent_measures[:, :, PEDESTRIAN_MEASURES_INDX.inverse_dist_to_closest_car])*mask
    else:
        car_dist=agent_measures[:, :, CAR_MEASURES_INDX.dist_to_agent]*mask


        cov=np.cov(np.concatenate([car_dist.reshape([1,-1]),agent_speed.reshape([1,-1])], axis=0))
        speed_to_dist_correlation.append(cov[1,0])
    dist_to_car.append([np.mean(car_dist), np.std(car_dist)])
    #min_dist_to_car.append([np.min(car_dist), np.max(car_dist)])
    init_dist_to_car.append([np.mean(car_dist[:,0]), np.std(car_dist[:,0])])


    # Pavement
    pavement_t = agent_measures[:, :, PEDESTRIAN_MEASURES_INDX.iou_pavement]*mask
    pavement_t[pavement_t > 0] = 1
    tmp = np.sum(pavement_t, axis=1)*1.0/divisor#np.sum(mask, axis=1)

    pavement.append([np.mean(tmp), np.std(tmp)])

    # Heatmap
    people_heatmap_local = agent_measures[:, :, PEDESTRIAN_MEASURES_INDX.pedestrian_heatmap]*mask

    tmp = np.sum(people_heatmap_local, axis=1)*1.0/divisor#np.sum(mask, axis=1)

    people_heatmap.append([np.mean(tmp), np.std(tmp)])

    # Average speed
    tmp = np.mean(agent_speed * mask, axis=1) * 1.0 /divisor# np.sum(mask, axis=0)
    speed_mean.append([np.mean(tmp), np.std(tmp)])

    # Per trajectory std speed
    tmp = np.std(agent_speed * mask, axis=1) * 1.0 /divisor# np.sum(mask, axis=0)
    #print(" Std of speed shape "+str(tmp.shape)+" "+str(agent_speed.shape))
    speed_var.append([np.mean(tmp), np.std(tmp)])

    # # Per trajectory std speed
    # entropy = np.std(agent_speed * mask, axis=1) * 1.0 / divisor  # np.sum(mask, axis=1)
    # speed_var.append([np.mean(tmp), np.std(tmp)])



# Get statistics of car environment
def get_cars_stats(filenames_itr_cars, sz_train):

    test_itr_car = []
    statistics_car = np.zeros((len(filenames_itr_cars) * sz_train[0], sz_train[1], sz_train[2]))
    statistics_test_car=[]
    indx_test = 0
    indx_same = 0
    prev_itr = 0
    test_points_car = []
    if len(test_files_cars) > 0:
        sz_test_car = np.load(test_files_cars[0][1]).shape
        statistics_test_car = np.zeros((len(test_files_cars) * sz_test_car[0], sz_test_car[1], sz_test_car[2]))
        for j, pair in enumerate(sorted(test_files_cars)):
            cur_statistics = np.load(pair[1])
            statistics_test_car[indx_test * sz_test_car[0]:(indx_test + 1) * sz_test_car[0], :, :] = cur_statistics
            test_itr_car.append(pair[0])
            if indx_same == pair[0]:
                prev_itr += 1
            else:
                prev_itr = 0
                test_points_car.append(j)
            indx_same = pair[0]

            test_itr_car[indx_test * sz_test_car[0]:(indx_test + 1) * sz_test_car[0]] = list(range(
                pair[0] + prev_itr * sz_test_car[0], pair[0] + (prev_itr + 1) * sz_test_car[0]))
            indx_test += 1
    indx_train_car = 0
    for i, pair in enumerate(sorted(filenames_itr_cars)):
        try:

            cur_statistics = np.load(pair[1])

            statistics_car[indx_train_car * sz_train[0]:(indx_train_car + 1) * sz_train[0], :, :] = cur_statistics[:,
                                                                                                    0: sz_train[1], :]
            indx_train_car += 1

        except IOError:
            print("Could not load " + pair[1])


    return indx_train_car, statistics_car, statistics_test_car, test_itr_car

# Sort statistics files
def sort_files(files):

    filenames_itr_cars = []
    test_files_cars = []
    filenames_itr_people = []
    test_files_people = []
    filenames_itr = []
    test_files = []
    nbr_files = 0
    reconstructions = []
    reconstructions_test = []
    init_maps = []
    init_map_stats = []
    init_cars = []
    init_maps_test = []
    init_map_stats_test = []
    init_cars_test = []
    numbers = []
    iterations = {}
    numbers_cars = []
    special_cases=[]
    numbers_people = []
    iterations_cars=[]
    learn_car_test=[]
    learn_car=[]

    for agent_file in sorted(files):

        basename = os.path.basename(agent_file)
        #print (basename)
        if not "pose" in basename:
            nbrs = basename.strip()[:-len('.npy')]
            vals = nbrs.split('_')

            #print (str(vals)+" Check "+str(vals[-4])+" for test "+str(vals[-6]))
            if vals[1] == 'weimar' or 'test' in vals[-4] or 'test' in vals[-6] or 'test' in vals[3]:  # Testing data
                #print (" weimar ? "+str(vals[1] == 'weimar')+" test in 4 "+str('test' in vals[-4] )+" test in 6 "+str('test' in vals[-6])+" test in 7 "+str(( len(vals)>7 and 'test' in vals[-7])))
                if "learn" in vals[-2]:
                    #print(" add file to car list "+str(agent_file))
                    learn_car_test.append((int(vals[-3]), agent_file, int(vals[-5])))
                elif "init" in vals[-3]:

                    init_map_stats_test.append((int(vals[-4]), agent_file, int(vals[-8])))
                elif "init" in vals[-2]:
                    if "car" in vals[-1]:
                        #print("Save as init test file as car ")
                        init_cars_test.append((int(vals[-3]), agent_file, int(vals[-7])))
                    else:
                        init_maps_test.append((int(vals[-3]), agent_file, int(vals[-7])))
                elif "people" in vals[-4]:
                    #print "people test"
                    if not "reconstruction" in vals[-1]:
                        test_files_people.append((int(vals[-1]), agent_file))
                        #test_files.append((int(vals[-1]), agent_file))
                        numbers_people.append(int(vals[-1]))

                elif "car" in vals[-4]:
                    #print "car test"
                    if not "reconstruction" in vals[-1]:
                        test_files_cars.append((int(vals[-1]), agent_file))
                        #test_files.append((int(vals[-1]), agent_file))
                        numbers_people.append(int(vals[-1]))

                elif not "reconstruction" in vals[-1]:

                    try:
                        test_files.append((int(vals[-1]), agent_file, int(vals[2])))
                        # numbers.append(int(vals[-1]))
                    except ValueError:
                        test_files.append((int(vals[-1]), agent_file, int(vals[4])))
                        # numbers.append(int(vals[-1]))
                    #print test_files[-1]
                else:
                    try:
                        reconstructions_test.append((int(vals[-1][:-len("reconstruction")]), agent_file, int(vals[2])))
                    except ValueError:
                        reconstructions_test.append(
                            (int(vals[-1][:-len("reconstruction")]), agent_file, int(vals[4])))
                if  not "reconstruction" in vals[-1] and not  "init" in vals and not  "learn" in vals:
                    if int(vals[-1]) not in list(iterations.keys()):
                        iterations[int(vals[-1])]= False
                    elif iterations[int(vals[-1])]:
                        special_cases.append(int(vals[-1]))
            else: # Training data
                #print (vals)
                if "learn" in vals[-2]:
                    # print(" add file to car list " + str(agent_file))
                    learn_car.append((int(vals[-3]), agent_file, int(vals[-6])))
                elif "init" in vals[-3]:
                    #print (" ADD " + str(vals[-4]) + " and " + str(vals[-7]))
                    init_map_stats.append((int(vals[-4]), agent_file, int(vals[-7])))
                elif "init" in vals[-2]:
                    if "car" in vals[-1]:
                        #print("Save as init train file as car ")
                        init_cars.append((int(vals[-3]), agent_file, int(vals[-6])))
                    else:
                        #print("Save as init train file as map ")
                        init_maps.append((int(vals[-3]), agent_file, int(vals[-6])))
                elif "people" in vals[-4]:
                    #print "People"
                    if not "reconstruction" in vals[-1]:
                        filenames_itr_people.append((int(vals[-1]), agent_file))
                        #filenames_itr.append((int(vals[-1]), agent_file))
                        numbers_people.append(int(vals[-1]))

                elif "car" in vals[-4]:
                    #print "Car"
                    if not "reconstruction" in vals[-1]:
                        filenames_itr_cars.append((int(vals[-1]), agent_file))
                        #filenames_itr.append((int(vals[-1]), agent_file))
                        numbers_people.append(int(vals[-1]))

                elif not "reconstruction" in vals[-1]:
                    #print vals[-4]
                    try:
                        filenames_itr.append((int(vals[-1]), agent_file, int(vals[2])))
                        numbers.append(int(vals[-1]))
                    except ValueError:
                        # numbers.append(int(vals[-1]))
                        filenames_itr.append((int(vals[-1]), agent_file, int(vals[3])))
                else:
                    try:
                        reconstructions.append((int(vals[-1][:-len("reconstruction")]), agent_file, int(vals[2])))
                    except ValueError:
                        reconstructions.append((int(vals[-1][:-len("reconstruction")]), agent_file, int(vals[3])))

                if  not "reconstruction" in vals[-1] and not  "init" in vals and not  "learn" in vals:
                    if "people" in vals[-4] or "car" in vals[-4]:

                        iterations_cars.append(int(vals[-1]))
                    if int(vals[-1]) not in list(iterations.keys()):
                        iterations[int(vals[-1])] = True
                    elif not iterations[int(vals[-1])] :

                        special_cases.append(int(vals[-1]))

                #special_cases.append(int(vals[-1]))

            nbr_files += 1
    filenames_itr.sort(key=lambda x: x[2])
    if filenames_itr[-1][2]==24: # Realtime dataset
        train_set=[21, 20, 13, 16, 6, 18, 15, 14, 24, 19, 23, 12, 8, 9, 10]
        val_set=[7, 17, 25, 22, 11]
        for tuple in filenames_itr:
            print (tuple[2])
            if tuple[2] in val_set:
                test_files.append(tuple)
        filenames_itr_new = [x for x in filenames_itr if not x in val_set]
        filenames_itr=filenames_itr_new
        print ("Realtime dataset")
        print(" Training files")
        print(filenames_itr)
        print(" Test files ")
        print(test_files)

    else:
        filenames_itr.sort(key=lambda x: x[0])
    return filenames_itr_cars, test_files_cars, filenames_itr_people,test_files_people,  filenames_itr, test_files, reconstructions_test, iterations, iterations_cars, special_cases, init_maps, init_cars, init_maps_test, init_cars_test, learn_car, learn_car_test, init_map_stats, init_map_stats_test


# Sort evaluation statistics files
def sort_files_eval(files):
    test_files_poses=[]
    test_files = []
    test_files_cars=[]
    test_files_people=[]
    reconstructions_test=[]
    nbr_files = 0

    special_cases=[]
    numbers_people = []
    init_maps = []
    init_map_stats = []
    init_cars = []
    learn_car = []

    for agent_file in sorted(files):

        basename = os.path.basename(agent_file)

        nbrs = basename.strip()[:-len('.npy')]

        vals = nbrs.split('_')


        if "learn" in vals[-2]:
            # print (vals)
            # print ((int(vals[-5]), agent_file, int(vals[-3])))
            learn_car.append((int(vals[-5]), agent_file, int(vals[-3])))
        elif "init" in vals[-3]:
            # print(vals)
            # print((int(vals[-6]), agent_file, int(vals[-4])))
            init_map_stats.append((int(vals[-4]), agent_file, int(vals[-8])))
        elif "init" in vals[-2]:
            if "car" in vals[-1]:
                # print("Save as init test file as car ")
                # print(vals)
                # print((int(vals[-5]), agent_file, int(vals[-3])))
                init_cars.append((int(vals[-5]), agent_file, int(vals[-3])))
            else:
                # print(vals)
                # print((int(vals[-3]), agent_file, int(vals[-7])))
                init_maps.append((int(vals[-3]), agent_file, int(vals[-7])))

        elif "poses" in vals[-1]:
            #agent', 'test', '0', 'test', '0', '-64', '1', 'poses']

            # print(vals)
            # print((int(vals[-4]), agent_file, int(vals[-2])))
            test_files_poses.append((int(vals[-2]), agent_file, int(vals[-6])))
        else:
            try:
                pos=int(vals[-1])
            except ValueError:
                try:
                    pos = int(vals[-1][:-len("reconstruction")])
                except ValueError:
                    print(basename)
                    pos=-1

            if pos>=0:

                if not "reconstruction" in vals[-1]:
                    print(vals)
                    print((int(vals[-3]), agent_file, int(vals[-1])))
                    test_files.append((int(vals[-3]), agent_file, int(vals[-1])))
                else:
                    if int(vals[-1][:-len("reconstruction")]) <42:
                        reconstructions_test.append((int(vals[-3]), agent_file, int(vals[-1][:-len("reconstruction")])))

                nbr_files += 1
    return test_files, reconstructions_test, learn_car, init_map_stats, init_maps, init_cars, test_files_poses



# Gather images into a movie
def make_movies(movie_path, name_movies_eval,name_movie,target_dir,timestamp):
    command = "ffmpeg -framerate 25 -i " + movie_path + name_movie + "frame_%06d.jpg -c:v libx264  -pix_fmt yuv420p -y " + target_dir + "/" + timestamp + "_" + name_movie + ".mp4"
    if save_regular_plots and make_movie:
        print(command)
        subprocess.call(command, shell=True)
        try:
            command = "ffmpeg -framerate 10 -i " + movie_path + name_movie + '_cars_frame_%06d.jpg -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -y ' + target_dir + "/" + timestamp + "_" + name_movie + '_car.mp4'
            print(command)
            subprocess.call(command, shell=True)
        except IOError:
            traceback.print_exc()

        print(command)
        subprocess.call(command, shell=True)
        try:
            command = "ffmpeg -framerate 10 -i " + movie_path + name_movie + '_people_frame_%06d.jpg -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -y ' + target_dir + "/" + timestamp + "_" + name_movie + '_people.mp4'
            print(command)
            subprocess.call(command, shell=True)
        except IOError:
            traceback.print_exc()
        try:
            command = "ffmpeg -framerate 10 -i " + movie_path + name_movie + '_train__cars_frame_%06d.jpg -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -y ' + target_dir + "/" + timestamp + "_" + name_movie + '_train_car.mp4'
            print(command)
            subprocess.call(command, shell=True)
        except IOError:
            traceback.print_exc()

        print(command)
        subprocess.call(command, shell=True)
        try:
            command = "ffmpeg -framerate 10 -i " + name_movies_eval + name_movie + 'frame_%06d.jpg -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -y ' + target_dir + "/" + timestamp + "_" + name_movie + '_eval.mp4'
            print(command)
            subprocess.call(command, shell=True)
        except IOError:
            traceback.print_exc()

        print(command)
        subprocess.call(command, shell=True)
        try:
            command = "ffmpeg -framerate 10 -i " + movie_path + name_movie + '_train__people_frame_%06d.jpg -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -y ' + target_dir + "/" + timestamp + "_" + name_movie + '_train_people.mp4'
            print(command)
            subprocess.call(command, shell=True)
        except IOError:
            traceback.print_exc()

# Read settings file to get necessary settings
def read_settings_file(timestamp):

    # Find settings file
    settings_file = glob.glob(settings_path + "*/" + timestamp + "*" + settings_ending)

    if len(settings_file) == 0:
        settings_file = glob.glob(settings_path + timestamp + "*" + settings_ending)
    if len(settings_file)==0:
        return [],[],[],[],[],[],[], [],[], []
    labels_to_use = labels
    # Get movie name
    name_movie = os.path.basename(settings_file[0])[len(timestamp) + 1:-len("_settings.txt")]
    target_dir = settings_path + name_movie
    subprocess.call("cp " + settings_file[0] + " " + target_dir + "/", shell=True)
    semantic_channels_separated = False
    in_2D=False
    temporal_case=False
    carla=False
    num_measures = 6
    num_measures_car=0
    continous=False
    gaussian=False
    with open(settings_file[0]) as s:
        for line in s:
            if "Semantic channels separated" in line:
                if line[len("Semantic channels separated: "):].strip()=="True":
                    semantic_channels_separated = True

            if "Minimal semantic channels : " in line:
                if line[len("Minimal semantic channels : "):].strip() == "True":
                    mini_labels = True
                    if mini_labels:
                        labels_to_use = labels_mini
            if "Number of measures" in line:
                if " car" in line:
                    num_measures_car = int(line[len("Number of measures car : "):])
                else:
                    num_measures = int(line[len("Number of measures: "):])
            if "2D input to network:"in line:
                if line[len("2D input to network: "):].strip()== "True":
                    in_2D=True
                print(line[len("2D input to network: "):].strip() +" "+str(bool(line[len("2D input to network: "):].strip()))+" "+str(in_2D))
            if "Temporal case : " in line :
                if line[len("Temporal case : " ):].strip()== "True":
                    temporal_case=True
            if "CARLA : "  in line and line[len("CARLA : " ):].strip()== "True":
                carla=True
            if "Continous: " in line and line[len("Continous: " ):].strip()== "True":
                continous=True

            if "Gaussian Initializer : " in line and line[len("Gaussian Initializer : " ):].strip()== "True":
                gaussian=True
    if carla:
        temporal_case=False

    return labels_to_use, name_movie, target_dir, num_measures, semantic_channels_separated, in_2D, temporal_case, continous, num_measures_car, gaussian

####################################################### Gathering data about gradients and weights
# ACtion names

def find_gradients(path):

    match = path + "*.pkl"
    weights = []
    weights_car=[]
    gradients = []
    gradients_car=[]
    files = glob.glob(match)
    #print files
    for file in files:
        basename = os.path.basename(file)

        nbrs = basename.strip()[len(timestamp):-len('.pkl')]
        #print(basename+" "+str(nbrs))
        if toy_case:

            vals = nbrs.split('_')

            if len(vals) == 3:
                weights.append((int(vals[1]), file, int(vals[0])))

            else:
                gradients.append((int(vals[1]), file, int(vals[0])))
        else:
            vals = basename.split('_')
            if "weights" in basename:
                if "car" in vals[-2]:
                    weights_car.append((int(vals[-4]), file, int(vals[2])))
                elif "init" in vals[-2]:
                    try:
                        # print ("Values")
                        # print (vals[-4])
                        #
                        # print (file)
                        # print (vals[2])
                        weights.append((int(vals[-4]), file, int(vals[2])))
                        #print ("Appended: "+str(weights[-1]))
                    except ValueError:
                        #print ("Value Error")
                        # print ("Values")
                        # print (vals[-4])
                        #
                        # print (file)
                        # print (vals[3])
                        weights.append((int(vals[-4]), file, int(vals[3])))
                        #print ("Appended: " + str(weights[-1]))
                else:
                    try:

                        weights.append((int(vals[-2]), file, int(vals[2])))
                    except ValueError:
                        weights.append((int(vals[-2]), file, int(vals[3])))
            else:
                if "car" in vals[-2]:
                    gradients_car.append((int(vals[-3]), file, int(vals[2])))
                elif "init" in vals[-2]:
                    try:
                        #print ("Values")
                        # print (vals[-4])
                        #
                        # print (file)
                        # print (vals[2])
                        gradients.append((int(vals[-3]), file, int(vals[2])))
                        #print ("Appended: "+str(gradients[-1]))
                    except ValueError:
                        #print ("Value Error")
                        # print ("Values")
                        # print (vals[-4])
                        #
                        # print (file)
                        # print (vals[3])
                        gradients.append((int(vals[-3]), file, int(vals[3])))
                        #print ("Appended: " + str(gradients[-1]))
                else:
                    try:
                        gradients.append((int(vals[-2]), file, int(vals[2])))
                    except ValueError:
                        gradients.append((int(vals[-2]), file, int(vals[3])))
    return weights, gradients, files, weights_car, gradients_car


def read_gradients(gradients):
    itr = 0
    grads = []
    if len(gradients) > 0:
        with open(gradients[0][1], 'rb') as handle:
            b = pickle.load(handle, encoding="latin1", fix_imports=True)

        grads = []
        K = 0
        for var in b:
            # print var.shape
            shape = [len(gradients)]
            for dim in var.shape:
                shape.append(dim)
            grads.append(np.zeros(shape))
            print (str(K) + " " + str(shape))


        for pair in sorted(gradients):

            with open(pair[1], 'rb') as handle:
                b = pickle.load(handle, encoding="latin1", fix_imports=True)
                for indx, var in enumerate(b):
                    grads[indx][itr, :] = var.copy()
                    #print ("Gradients "+str(indx)+" " + pair[1] + " " + str(np.sum(np.absolute(var[:])))+" "+str(np.sum(np.absolute(grads[indx][itr, :])))+" "+str(np.sum(np.absolute(grads[indx][ :]))))

            itr += 1
    return grads, itr
 #######################################
def get_init_stats(init_maps_test, init_map_stats, init_cars_test, itr_points):
    print ("Get init stats!")
    entropy_maps_test = []
    entropy_prior_maps_test = []
    kl_to_prior_maps_test = []
    diff_to_prior_maps_test = []
    entropy_maps_test_goal = []
    entropy_prior_maps_test_goal = []
    kl_to_prior_maps_test_goal = []
    diff_to_prior_maps_test_goal = []
    goal_position_mode=[]
    goal_prior_mode = []
    init_position_mode = []
    init_prior_mode = []
    goal_time_mode=[]

    cars_vel_test_x=[]
    cars_vel_test_y = []
    cars_vel_test_speed = []
    cars_pos_test_x = []
    cars_pos_test_y = []

    iterations=[]
    init_maps_test=sorted(init_maps_test)


    init_cars_test = sorted(init_cars_test)


    for j, pair in enumerate(init_cars_test):
        iterations.append(itr_points[pair[0]])
        car_test = np.load(pair[1])

        cars_vel_test_x.append((np.mean(car_test[:,STATISTICS_INDX_CAR_INIT.car_vel[0]]), np.std(STATISTICS_INDX_CAR_INIT.car_vel[0])))
        cars_vel_test_y.append((np.mean(car_test[:, STATISTICS_INDX_CAR_INIT.car_vel[1]-1]), np.std(car_test[:, STATISTICS_INDX_CAR_INIT.car_vel[1]-1])))
        speed=np.linalg.norm(car_test[:, STATISTICS_INDX_CAR_INIT.car_vel[0]:STATISTICS_INDX_CAR_INIT.car_vel[1]], axis=1)*.2

        cars_vel_test_speed.append((np.mean(speed), np.std(speed)))
        cars_pos_test_x.append((np.mean(car_test[:,STATISTICS_INDX_CAR_INIT.car_pos[0]]), np.std(car_test[:,STATISTICS_INDX_CAR_INIT.car_pos[1]])))
        cars_pos_test_y.append((np.mean(car_test[:, STATISTICS_INDX_CAR_INIT.car_pos[1]-1]), np.std(car_test[:, STATISTICS_INDX_CAR_INIT.car_pos[1]-1])))
        manual_goal=car_test[:, STATISTICS_INDX_CAR_INIT.manual_goal[0]:STATISTICS_INDX_CAR_INIT.manual_goal[1]]
        #for ep_itr in range(car_test.shape[0]):
        goal_prior_mode.append(np.mean(manual_goal, axis=0))
        #print ("Appended avg pos " + str(cars_pos_test[-1]))

    gaussian=False# Read this from stat file instead

    interation_list=init_maps_test
    use_stats = False
    if len(init_map_stats)>0:
        use_stats=True
        interation_list=sorted(init_map_stats)

    for j, pair in enumerate(interation_list):
        if not use_stats:
            print (" Do not use stats")
            map_test = np.load(pair[1])
            distr=map_test[:,:,STATISTICS_INDX_MAP.init_distribution]
            prior = map_test[:, :, STATISTICS_INDX_MAP.prior]
            distr_goal = map_test[:, :, STATISTICS_INDX_MAP.goal_distribution]
            prior_goal = map_test[:, :, STATISTICS_INDX_MAP.goal_prior]
            entropys=[]
            entropys_prior = []
            kls=[]
            entropys_goal = []
            entropys_prior_goal = []
            kls_goal = []
            if gaussian:

                goal_position_mode.append(np.mean(distr_goal[:, :2], axis=0))
                #print ("Goal position shape " + str(distr_goal[:, :2].shape)+" added shape "+str(goal_position_mode[-1].shape))
                #goal_prior_mode.append(np.mean(distr[:, :2], axis=0))
            else:
                product_goal = distr_goal * prior_goal
                goal_mode=[]
                goal_prior=[]
                for ep_itr in range(distr.shape[0]):
                    # print ("distribution")
                    # print (min(distr[ep_itr,:]))
                    # print (max (distr[ep_itr,:]))

                    entropys.append(scipy.stats.entropy(distr[ep_itr,:]))

                    entropys_prior.append(scipy.stats.entropy(prior[ep_itr, :]))

                    entropys_goal.append(scipy.stats.entropy(distr_goal[ep_itr, :]))

                    entropys_prior_goal.append(scipy.stats.entropy(distr_goal[ep_itr, :]))
                    # print ("prior")
                    # print (prior[ep_itr, :])
                    kls.append(scipy.stats.entropy(distr[ep_itr,:],qk=prior[ep_itr,:] ))
                    kls_goal.append(scipy.stats.entropy(distr_goal[ep_itr, :], qk=distr_goal[ep_itr, :]))
                    goal_mode.append(np.unravel_index(np.argmax(product_goal[ep_itr, :]), settings.env_shape[1:]))
                    goal_prior.append(np.unravel_index(np.argmax(prior_goal[ep_itr, :]), settings.env_shape[1:]))

                goal_position_mode.append(np.mean(goal_mode, axis=0))  # To Do: reshape to matrix index here!
                # goal_prior_mode.append(np.mean(goal_prior, axis=0))



            product=distr*prior
            init_mode = []
            init_prior = []
            for ep_itr in range(product.shape[0]):
                init_mode.append(np.unravel_index(np.argmax(product[ep_itr, :]), settings.env_shape[1:]))
                init_prior.append(np.unravel_index(np.argmax(prior[ep_itr, :]), settings.env_shape[1:]))

            diff_to_prior = np.sum(np.abs(distr - prior), axis=1)
            diff_to_prior_goal = np.sum(np.abs(distr_goal - prior_goal), axis=1)

        else:
            initialization_map_stats=np.load(pair[1])
            init_mode=initialization_map_stats[:,STATISTICS_INDX_MAP_STAT.init_position_mode[0]:STATISTICS_INDX_MAP_STAT.init_position_mode[1]]
            init_prior=initialization_map_stats[:,STATISTICS_INDX_MAP_STAT.init_prior_mode[0]:STATISTICS_INDX_MAP_STAT.init_prior_mode[1]]

            entropys=initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.entropy]
            entropys_prior=initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.entropy_prior]
            kls=initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.kullback_leibler_divergence_init_and_prior]
            diff_to_prior=initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.prior_init_difference]
            entropys_goal=initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.entropy_goal]
            entropys_prior_goal=initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.entropy_prior_goal]
            kls_goal=initialization_map_stats[:,STATISTICS_INDX_MAP_STAT.kullback_leibler_divergence_init_and_prior_goal]
            diff_to_prior_goal=initialization_map_stats[:, STATISTICS_INDX_MAP_STAT.prior_init_difference_goal]
            #print(" Pos " + str(initialization_map_stats[:,STATISTICS_INDX_MAP_STAT.goal_position_mode[0]:STATISTICS_INDX_MAP_STAT.goal_position_mode[1]].shape) + " " + str(initialization_map_stats[:,STATISTICS_INDX_MAP_STAT.goal_prior_mode[0]:STATISTICS_INDX_MAP_STAT.goal_prior_mode[1]].shape))
            goal_position_mode.append(np.mean(initialization_map_stats[:,STATISTICS_INDX_MAP_STAT.goal_position_mode[0]:STATISTICS_INDX_MAP_STAT.goal_position_mode[1]], axis=0))
            #print(" Mode shape "+str(len(goal_position_mode))+" "+str(goal_position_mode[-1].shape))
            #goal_prior_mode.append(np.mean(initialization_map_stats[:,STATISTICS_INDX_MAP_STAT.goal_prior_mode[0]:STATISTICS_INDX_MAP_STAT.goal_prior_mode[1]], axis=0))
            #print(" Prior shape " + str(len(goal_prior_mode)) + " " + str(goal_prior_mode[-1].shape))
        #print (init_mode)

        init_position_mode.append(np.mean(init_mode, axis=0))  # To Do: reshape to matrix index here!
        #print ("Average "+str(init_position_mode))

        init_prior_mode.append(np.mean(init_prior, axis=0))
        #print("file " + str(j) + " init pos mode len " + str(len(init_position_mode)) + " init prior mode len " + str(len(init_prior_mode)))

        entropy_maps_test.append((np.mean(entropys), np.std(entropys)))
        # print ("Final entropy: "+str(entropy_maps_test[-1]))
        # print (entropys)
        entropy_prior_maps_test.append((np.mean(entropys_prior), np.std(entropys_prior)))

        # print ("Prior's entropy: " + str(entropy_prior_maps_test[-1]))
        # print (entropys_prior)
        kl_to_prior_maps_test.append((np.mean(kls), np.std(kls)))
        # print("file " + str(j) + " entropy " + str(len(entropy_maps_test)) + " init prior mode len " + str(
        #     len(entropy_prior_maps_test))+" kl  "+str(len(kl_to_prior_maps_test)))

        entropy_maps_test_goal.append((np.mean(entropys_goal), np.std(entropys_goal)))
        # print ("Final entropy: "+str(entropy_maps_test[-1]))
        # print (entropys)
        entropy_prior_maps_test_goal.append((np.mean(entropys_prior_goal), np.std(entropys_prior_goal)))
        # print ("Prior's entropy: " + str(entropy_prior_maps_test[-1]))
        # print (entropys_prior)
        kl_to_prior_maps_test_goal.append((np.mean(kls_goal), np.std(kls_goal)))
        # print("file " + str(j) + " entropy " + str(len(entropy_maps_test_goal)) + " init prior mode len " + str(
        #     len(entropy_prior_maps_test_goal)) + " kl  " + str(len(kl_to_prior_maps_test_goal)))

        diff_to_prior_maps_test.append((np.mean(diff_to_prior), np.std(diff_to_prior)))
        diff_to_prior_maps_test_goal.append((np.mean(diff_to_prior_goal), np.std(diff_to_prior_goal)))
        # print("file " + str(j) + " diff to prior " + str(len(diff_to_prior_maps_test)) + " diff to prior goal " + str(
        #     len(diff_to_prior_maps_test_goal)) )
        # print ("Final diff: " + str(diff_to_prior))

    return entropy_maps_test, entropy_prior_maps_test, kl_to_prior_maps_test, diff_to_prior_maps_test, cars_vel_test_x,cars_vel_test_y,cars_vel_test_speed, cars_pos_test_x,cars_pos_test_y, iterations,entropy_maps_test_goal, entropy_prior_maps_test_goal, kl_to_prior_maps_test_goal,diff_to_prior_maps_test_goal, goal_position_mode,goal_prior_mode, gaussian, init_position_mode, init_prior_mode





# Separates data according to inilatizationa and gathers statictics
def get_stats(test_files,test_points,continous, temp_case=False, car_case=False):

    stats_temp = []
    rewards = []
    hit_objs = []
    ped_traj = []
    people_hit = []
    entropy = []
    dist_left = []
    dist_travelled=[]
    car_hit = []
    pavement = []
    plot_iterations_test=[]
    people_heatmap=[]
    successes = []
    norm_dist_travelled = []
    nbr_direction_switches = []
    nbr_inits=[]
    losses = []
    one_step_error = []
    one_step_errors=[]
    one_step_errors_g=[]
    prob_of_max_action=[]
    most_freq_action=[]
    freq_most_freq_action=[]
    variance=[]
    speed_mean=[]
    speed_var=[]
    likelihood_actions = []
    likelihood_full=[]
    prediction_error = []
    dist_to_car = []
    time_to_collision=[]
    init_dist_to_goal=[]
    goal_locations=[]
    init_locations=[]
    goal_times=[]
    goal_speed=[]
    init_dist_to_car=[]
    speed_to_dist_correlation=[]
    collision_between_car_and_agent=[]


    for init_m in range(len(PEDESTRIAN_INITIALIZATION_CODE)+2):
        stats_temp.append([])
        rewards.append([])
        hit_objs.append([])
        ped_traj.append([])
        people_hit.append([])
        entropy.append([])
        dist_left.append([])
        dist_travelled.append([])
        car_hit.append([])
        pavement.append([])
        plot_iterations_test.append([])
        people_heatmap.append([])
        successes.append([])
        norm_dist_travelled.append([])
        nbr_direction_switches.append([])
        losses.append([])
        one_step_error.append([])
        prob_of_max_action.append([])
        most_freq_action.append([])
        freq_most_freq_action.append([])
        variance.append([])
        speed_mean.append([])
        speed_var.append([])
        likelihood_actions.append([])
        likelihood_full.append([])
        prediction_error.append([])
        dist_to_car.append([])
        time_to_collision.append([])
        init_dist_to_goal.append([])
        goal_locations.append([])
        init_locations.append([])
        goal_times.append([])
        goal_speed.append([])
        init_dist_to_car.append([])
        speed_to_dist_correlation.append([])
        collision_between_car_and_agent.append([])


    prev_test_pos = 0
    general_stats=[]

    for j, pair in enumerate(sorted(test_files)):
        # if prev_test_pos>300:

        # print (str(pair)+" prev test pos "+str(prev_test_pos))
        if prev_test_pos != test_points[pair[0]] or (temp_case and j%2==0):
            #print 'switch'
            for init_m, stats in enumerate(stats_temp):
                if len(stats)>0:
                    statistics = np.concatenate(stats, axis=0)
                    evaluate(statistics,continous, rewards[init_m], hit_objs[init_m], ped_traj[init_m], people_hit[init_m], entropy[init_m],
                             dist_left[init_m],dist_travelled[init_m], car_hit[init_m], pavement[init_m], people_heatmap[init_m],
                             successes[init_m], norm_dist_travelled[init_m], nbr_direction_switches[init_m], losses[init_m],
                             one_step_error[init_m], one_step_errors, prob_of_max_action[init_m],most_freq_action[init_m] ,
                             freq_most_freq_action[init_m], variance[init_m], speed_mean[init_m],speed_var[init_m],
                             likelihood_actions[init_m],likelihood_full[init_m], prediction_error[init_m], dist_to_car[init_m],
                             time_to_collision[init_m], init_dist_to_goal[init_m], goal_locations[init_m], goal_times[init_m],
                             init_locations[init_m], goal_speed[init_m],init_dist_to_car[init_m],
                             speed_to_dist_correlation[init_m],collision_between_car_and_agent[init_m],car_case)
                    if temp_case:
                        plot_iterations_test[init_m].append(j/2)
                        #print "Temp case "+str(plot_iterations_test[init_m][-1])
                    else:
                        plot_iterations_test[init_m].append(prev_test_pos)
                        #print "Test " + str(plot_iterations_test[init_m][-1])
            if len(general_stats)>0:
                if len(general_stats)>1:
                    statistics = np.concatenate(general_stats, axis=0)
                else:
                    statistics=general_stats[0]

                evaluate(statistics,continous, rewards[general_id], hit_objs[general_id], ped_traj[general_id], people_hit[general_id],
                         entropy[general_id],dist_left[general_id],dist_travelled[general_id],
                         car_hit[general_id], pavement[general_id],people_heatmap[general_id], successes[general_id],
                         norm_dist_travelled[general_id], nbr_direction_switches[general_id], losses[general_id],
                         one_step_error[general_id], one_step_errors_g, prob_of_max_action[general_id],
                         most_freq_action[general_id] , freq_most_freq_action[general_id], variance[general_id],
                         speed_mean[general_id],speed_var[general_id],likelihood_actions[general_id],
                         likelihood_full[general_id], prediction_error[general_id], dist_to_car[general_id],
                         time_to_collision[general_id], init_dist_to_goal[general_id], goal_locations[general_id],
                         goal_times[general_id],init_locations[general_id], goal_speed[general_id],init_dist_to_car[general_id],
                         speed_to_dist_correlation[general_id],collision_between_car_and_agent[general_id],car_case)
                if temp_case:
                    plot_iterations_test[general_id].append(j / 2)
                else:
                    plot_iterations_test[general_id].append(prev_test_pos)
            stats_temp = []
            for init_m in range(11):
                stats_temp.append([])
            general_stats = []
        cur_stat = np.load(pair[1])
        #print ("file "+str(pair[1])+" "+str(cur_stat.shape)+" (" +str(pair[0])+") "+" "+str(test_points[pair[0]])+" "+str(prev_test_pos))
        for ep_nbr in range(cur_stat.shape[0]):
            nbr_inits.append(cur_stat[ep_nbr, 0, -1])
            stats_temp[np.int(cur_stat[ep_nbr, 0, -1])].append(np.expand_dims(cur_stat[ep_nbr, :300, :], axis=0))

            #if np.int(cur_stat[ep_nbr, 0, -1])not in{5,7}:
            general_stats.append(np.expand_dims(cur_stat[ep_nbr, :300, :], axis=0))
        prev_test_pos = test_points[pair[0]]


    for init_m, stats in enumerate(stats_temp):
        if len(stats) > 0:
            # if len(stats) > 0:
            #     prev_val_shape=(0,0,0)
            #     for i,val in enumerate(stats):
            #         if prev_val_shape!=val.shape:
            #             prev_val_shape=val.shape
            #             print str(val.shape) +" "+str(i)
            statistics = np.concatenate(stats, axis=0)
            evaluate(statistics,continous, rewards[init_m], hit_objs[init_m], ped_traj[init_m], people_hit[init_m], entropy[init_m],
                     dist_left[init_m], dist_travelled[init_m], car_hit[init_m], pavement[init_m], people_heatmap[init_m],
                     successes[init_m],norm_dist_travelled[init_m], nbr_direction_switches[init_m], losses[init_m],
                     one_step_error[init_m], one_step_errors,  prob_of_max_action[init_m],most_freq_action[init_m] ,
                     freq_most_freq_action[init_m],variance[init_m], speed_mean[init_m],speed_var[init_m],
                     likelihood_actions[init_m],likelihood_full[init_m], prediction_error[init_m], dist_to_car[init_m],
                     time_to_collision[init_m], init_dist_to_goal[init_m], goal_locations[init_m], goal_times[init_m],
                     init_locations[init_m], goal_speed[init_m],init_dist_to_car[init_m],speed_to_dist_correlation[init_m],
                     collision_between_car_and_agent[init_m], car_case)
            if temp_case:
                plot_iterations_test[init_m].append(len(test_files) / 2)
            else:
                plot_iterations_test[init_m].append(prev_test_pos)
    if len(general_stats) > 0:
        if len(general_stats) > 1:
            statistics = np.concatenate(general_stats, axis=0)
        else:
            statistics = general_stats[0]
        evaluate(statistics,continous, rewards[general_id], hit_objs[general_id], ped_traj[general_id], people_hit[general_id],
                 entropy[general_id], dist_left[general_id],dist_travelled[general_id], car_hit[general_id],
                 pavement[general_id],people_heatmap[general_id], successes[general_id],
                 norm_dist_travelled[general_id], nbr_direction_switches[general_id], losses[general_id],
                 one_step_error[general_id], one_step_errors_g,prob_of_max_action[general_id],most_freq_action[general_id] ,
                 freq_most_freq_action[general_id], variance[general_id], speed_mean[general_id],speed_var[general_id] ,
                 likelihood_actions[general_id],likelihood_full[general_id], prediction_error[general_id],
                 dist_to_car[general_id], time_to_collision[general_id], init_dist_to_goal[general_id],
                 goal_locations[general_id], goal_times[general_id],init_locations[general_id], goal_speed[general_id],
                 init_dist_to_car[general_id],speed_to_dist_correlation[general_id],collision_between_car_and_agent[general_id],
                 car_case)
        if temp_case:
            plot_iterations_test[general_id].append(len(test_files) / 2)
        else:
            plot_iterations_test[general_id].append(prev_test_pos)
    return rewards, hit_objs, ped_traj, people_hit, entropy, dist_left,dist_travelled, car_hit, pavement, plot_iterations_test, people_heatmap, nbr_inits, successes, norm_dist_travelled, nbr_direction_switches, losses, one_step_error, one_step_errors, prob_of_max_action, most_freq_action, freq_most_freq_action, variance, speed_mean, speed_var, likelihood_actions, likelihood_full, prediction_error, dist_to_car, time_to_collision, init_dist_to_goal, goal_locations, goal_times,init_locations, goal_speed, init_dist_to_car,speed_to_dist_correlation,collision_between_car_and_agent

######################################################################## Plotting functions
def plot(train_itr, avg_reward, test_itr, avg_reward_test , title, filename):



    data=[]
    names=[]
    fig = plt.figure()
    ax = plt.subplot(111)
    min_y= sys.maxsize
    max_y=-sys.maxsize - 1
    for init_m, reward_init_m in enumerate(avg_reward):
        if len(reward_init_m)>1:
            stacked_reward=np.stack(reward_init_m,axis=0)
            if len(stacked_reward.shape)==2 and stacked_reward.shape[1]==2:
                stacked_reward=stacked_reward[:,0]
            if (min(stacked_reward)<min_y ):
                min_y=min(stacked_reward)
            if (max(stacked_reward)>max_y ):
                max_y=max(stacked_reward)
            data1, = ax.plot(train_itr[init_m], stacked_reward, label='training data')
            data3, = ax.plot(train_itr[init_m], trendline(stacked_reward), label='training avg')
            data.append(data1)
            data.append(data3)
            names.append(init_names[init_m])#'train data '+str(init_m))
            names.append( init_names[init_m]+' avg')#'train avg.'+str(init_m))
        if len(avg_reward_test[init_m]) > 1:

            stacked_reward = np.stack(avg_reward_test[init_m], axis=0)
            if len(stacked_reward.shape) == 2 and stacked_reward.shape[1] == 2:
                stacked_reward = stacked_reward[:, 0]
            if (min(stacked_reward) < min_y):
                min_y = min(stacked_reward)
            if (max(stacked_reward)>max_y ):
                max_y=max(stacked_reward)
            data2, = ax.plot(test_itr[init_m], stacked_reward, label='test data')
            data4, = ax.plot(test_itr[init_m], trendline(stacked_reward), label='test avg')
            data.append(data2)
            data.append(data4)
            names.append("test "+init_names[init_m])#'test data ' + str(init_m))
            names.append("test "+init_names[init_m]+' avg')#'test avg.' + str(init_m))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.legend(data, names, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6,
          fancybox=True, shadow=True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)

    if save_plots and save_regular_plots:
        print("Saved  " + timestamp + filename)
        fig.savefig(os.path.join(target_dir, timestamp + filename))
    else:
        plt.show()

def plot_separately(train_itr, avg_reward, test_itr, avg_reward_test, title, filename, plot_train=False, axis=0):

    data_outer = []
    names_outer = []

    indx=1

    # print "Plot iterations: "
    # print str(train_itr)
    # print "Plot iterations test : "
    # print str(test_itr)

    number_non_zero_inits=0

    for init_m in range(10):
        if len(avg_reward[init_m]) > 1:
            number_non_zero_inits=number_non_zero_inits+1

    number_of_plots=1
    allowed_inits=[{general_id, 1,3, 5, 6,2,4,9, 10}]
    if number_non_zero_inits>4:
        number_of_plots=2
        allowed_inits = [{2,3,6,5}, {1,8,9,general_id}]

    for plot_number in range(number_of_plots):
        fig = plt.figure(figsize=(8.0, 5.0))
        number_of_rows = 2
        number_of_columns = 2
        inits_in_plot = 0
        for init_m in allowed_inits[plot_number]:
            if len(avg_reward[init_m]) > 1:
                inits_in_plot = inits_in_plot + 1
        if inits_in_plot<4:
            number_of_rows = 1
            number_of_columns = inits_in_plot
            # number_of_rows = 1
            # number_of_columns = number_non_zero_inits-4

        for init_m, reward_init_m in enumerate(avg_reward_test):
            data = []

            min_y = sys.maxsize
            max_y = -sys.maxsize - 1

            #print ("Inits  " +str(allowed_inits[plot_number]))

            if len(avg_reward[init_m]) > 1  and (init_m in allowed_inits[plot_number] or temporal_case) and indx<5:#10}
                if plot_train and len(avg_reward[init_m])>0:
                    ax = plt.subplot(number_of_rows, number_of_columns, indx)
                    stacked_reward_train = np.stack(avg_reward[init_m], axis=0)
                    if len(stacked_reward_train.shape) == 2 and stacked_reward_train.shape[1] == 2:
                        stacked_reward_train = stacked_reward_train[:, axis]
                    if (min(stacked_reward_train) < min_y):
                        min_y = min(stacked_reward_train)
                    if (max(stacked_reward_train) > max_y):
                        max_y = max(stacked_reward_train)
                    data1= ax.scatter(np.array(train_itr[init_m])*0.01, stacked_reward_train, alpha=0.4, label='training data')
                    data3, = ax.plot(np.array(train_itr[init_m])*0.01, trendline(stacked_reward_train), label='training avg')
                    data.append(data1)
                    data.append(data3)
                if len(avg_reward_test[init_m]) > 1:

                    stacked_reward = np.stack(avg_reward_test[init_m], axis=0)

                    if len(stacked_reward.shape) == 2 and stacked_reward.shape[1] == 2:
                        stacked_reward = stacked_reward[:, axis]
                    if (min(stacked_reward) < min_y):
                        min_y = min(stacked_reward)
                    if (max(stacked_reward) > max_y):
                        max_y = max(stacked_reward)
                    data2, = ax.plot(np.array(test_itr[init_m])*0.01, stacked_reward,marker="x", label='test data')
                    data4, = ax.plot(np.array(test_itr[init_m])*0.01, trendline(stacked_reward), label='test avg')

                    data.append(data2)
                    data.append(data4)


                ax.set_title(init_names[init_m])
                if len(data_outer)==0:
                    data_outer.append(data1)
                    data_outer.append(data3)
                    names_outer.append("train")  # 'test data ' + str(init_m))
                    names_outer.append("train avg")  # 'test avg.' + str(init_m))

                if len(data_outer)==2 and plot_train and len(avg_reward_test[init_m])>1:
                    data_outer.append(data2)
                    data_outer.append(data4)
                    names_outer.append("test")  # 'test data ' + str(init_m))
                    names_outer.append("test avg")  # 'test avg.' + str(init_m))
                indx=indx+1
                if number_of_plots ==2 and indx==5:
                    indx=1
        plt.subplots_adjust(left=0.15, bottom=0.125, right=0.9, top=0.9, wspace=None, hspace=0.25)


        fig.text(0.5, 0.04, 'Epoch', ha='center', fontdict={'size':'large'})
        fig.text(0.04, 0.5, title, va='center', rotation='vertical', fontdict={'size':'large'})
        fig.legend(data_outer, names_outer, loc='lower right', shadow=True, ncol=2)

        if save_plots and save_regular_plots:
            print( plot_number)
            print( "Saved  " + timestamp + filename)
            fig.savefig(os.path.join(target_dir, timestamp+"_"+str(plot_number)+"_"+ filename))
            plt.close('all')
        else:
            plt.show()
        #plot_separately_cars(train_itr, avg_reward, test_itr, avg_reward_test, title, "cars_"+filename, plot_train=plot_train)


def plot_goal(train_itr, goal_locations, test_itr, goal_locations_test,avg_goal_location_test, prior_goal_location_test,avg_goal_location, prior_goal_location, title, filename, plot_train=False, axis=0, gaussian=False):
    #print ("Plot goal! ")
    data_outer = []
    names_outer = []

    indx=1

    # print "Plot iterations: "
    # print str(train_itr)
    # print "Plot iterations test : "
    # print str(test_itr)


    fig = plt.figure(figsize=(8.0, 5.0))
    number_of_rows = 1
    number_of_columns =2
    inits_in_plot = 0

    # if inits_in_plot<4:
    #     number_of_rows = 1
    #     number_of_columns = inits_in_plot
    #     # number_of_rows = 1
    #     # number_of_columns = number_non_zero_inits-4

    data = []

    min_y = sys.maxsize
    max_y = -sys.maxsize - 1
    indx=0
    if len(goal_locations[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization]) > 0:
        #print ("Data shape " + str(goal_locations[7][-1].shape))
        nbr_train_itrs=goal_locations[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization][-1].shape[0]
        stacked_reward_train = np.stack(goal_locations[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization], axis=0)
        #print ("Data shape " + str(stacked_reward_train.shape))
        stacked_reward_train = np.reshape(stacked_reward_train, (-1, 2))

        avg_goal_location = np.stack(avg_goal_location, axis=0)

        prior_goal_location=np.stack(prior_goal_location, axis=0)
        # print ("Prior shape " + str(prior_goal_location.shape))
        if gaussian:
            avg_goal_location =avg_goal_location +prior_goal_location
    if len(goal_locations_test[7]) > 1:

        stacked_reward = np.vstack(goal_locations_test[PEDESTRIAN_INITIALIZATION_CODE.learn_initialization])#, axis=0)
        print (" Stacked reward shape "+str(stacked_reward.shape))
        stacked_reward = np.reshape(stacked_reward, (-1, 2))


        nbr_test_itrs = stacked_reward.shape[0] / len(avg_goal_location_test)

        #print ("Goal location number of elements "+str(len(avg_goal_location_test))+" size of element "+str(avg_goal_location_test[-1].shape))
        avg_goal_location_test = np.stack(avg_goal_location_test, axis=0)

        # print ("Prior Goal location number of elements " + str(len(prior_goal_location_test)) + " size of element " + str(
        #     prior_goal_location_test[-1].shape))

        prior_goal_location_test = np.stack(prior_goal_location_test, axis=0)
        if gaussian:
            avg_goal_location_test = avg_goal_location_test + prior_goal_location_test



    #print ("Inits  " +str(allowed_inits[plot_number]))
    for indx in range(2):
        if len(goal_locations[7]) > 1:
            ax = plt.subplot(number_of_rows, number_of_columns, indx+1)
            if  len(goal_locations[7])>0:

                # min_y = min([min(stacked_reward_train[:,indx]),min(avg_goal_location[:,indx]),min(prior_goal_location[:,indx])] )
                # max_y = max([max(stacked_reward_train[:,indx]),max(avg_goal_location[:,indx]),max(prior_goal_location[:,indx])] )

                data1= ax.scatter(np.repeat(np.array(train_itr[7])*0.01,nbr_train_itrs), stacked_reward_train[:,indx], alpha=0.4, label='goal locations')

                data3= ax.scatter(np.array(train_itr[7])*0.01, avg_goal_location[:,indx], alpha=0.4, label='average goal')
                # print ("Itr size "+str(len(train_itr[7]))+" prior len "+str(prior_goal_location[:,indx].shape))
                data5= ax.scatter(np.array(train_itr[7]) * 0.01, prior_goal_location[:,indx], alpha=0.4, label='prior')
                data.append(data1)
                data.append(data3)
                data.append(data5)
            if len(goal_locations_test[7]) > 1:
                # min_y = min([min(stacked_reward[:, indx]), min(avg_goal_location_test[:, indx]),
                #              min(prior_goal_location_test[:, indx])])
                # max_y =max([max(stacked_reward[:, indx]), max(avg_goal_location_test[:, indx]),
                #              max(prior_goal_location_test[:, indx])] )
                #print ("test itr " + str(np.array(test_itr[7]).shape) + " avg_goal " + str(stacked_reward[:, indx].shape))
                # print (np.repeat(np.array(test_itr)*0.01, nbr_test_itrs))
                # print (stacked_reward[:,indx])
                #print("Shape itr "+str(np.repeat(np.array(test_itr),nbr_test_itrs).shape)+" shape data "+str(stacked_reward[:,indx].shape) )
                data2= ax.scatter(np.repeat(np.array(test_itr)*0.01, nbr_test_itrs), stacked_reward[:,indx], alpha=0.4, label='test data')

                data4, = ax.plot(np.array(test_itr) * 0.01, avg_goal_location_test[:,indx], label='test average goal')
                data6= ax.scatter(np.array(test_itr) * 0.01, prior_goal_location_test[:,indx], alpha=0.4, label='test prior')
                data.append(data2)
                data.append(data4)
                data.append(data6)


            if len(data_outer)==0:
                data_outer.append(data1)
                data_outer.append(data3)
                data_outer.append(data5)
                names_outer.append("train")  # 'test data ' + str(init_m))
                names_outer.append("train mean")  # 'test avg.' + str(init_m))
                names_outer.append("train prior")  # 'test avg.' + str(init_m))

            if len(data_outer)==3 and len(goal_locations_test[7]) > 1:
                #print ("Added tag")
                data_outer.append(data2)
                data_outer.append(data4)
                data_outer.append(data6)
                names_outer.append("test")  # 'test data ' + str(init_m))
                names_outer.append("test mean")  # 'test avg.' + str(init_m))
                names_outer.append("test prior")  # 'test avg.' + str(init_m))
    plt.subplots_adjust(left=0.15, bottom=0.125, right=0.9, top=0.9, wspace=None, hspace=0.25)


    fig.text(0.5, 0.04, 'Epoch', ha='center', fontdict={'size':'large'})
    fig.text(0.04, 0.5, title, va='center', rotation='vertical', fontdict={'size':'large'})
    fig.legend(data_outer, names_outer, loc='lower right', shadow=True, ncol=2)

    if save_plots and save_regular_plots:

        print( "Saved  " + timestamp + filename)
        fig.savefig(os.path.join(target_dir, timestamp+"_"+ filename))
    else:
        plt.show()
    #plot_separately_cars(train_itr, avg_reward, test_itr, avg_reward_test, title, "cars_"+filename, plot_train=plot_train)


def plot_loss(train_itr, avg_reward, test_itr, avg_reward_test, title, filename, plot_train=False):
    #print ("Plotting!")
    data_outer = []
    names_outer = []
    fig = plt.figure(figsize=(8.0, 5.0))
    indx=1


    data = []

    min_y = sys.maxsize
    max_y = -sys.maxsize - 1

    if len(avg_reward_test) > 1:

        stacked_reward = np.stack(avg_reward_test, axis=0)
        if len(stacked_reward.shape) == 2 and stacked_reward.shape[1] == 2:
            stacked_reward = stacked_reward[:, 0]
        if (min(stacked_reward) < min_y):
            min_y = min(stacked_reward)
        if (max(stacked_reward) > max_y):
            max_y = max(stacked_reward)
        data2, = plt.plot(np.array(test_itr), stacked_reward, label='test data')
        data4, = plt.plot(np.array(test_itr), trendline(stacked_reward), label='test avg')


        data.append(data2)
        data.append(data4)
        if plot_train and len(avg_reward)>0:
            stacked_reward_train = np.stack(avg_reward, axis=0)
            if len(stacked_reward_train.shape) == 2 and stacked_reward_train.shape[1] == 2:
                stacked_reward_train = stacked_reward_train[:, 0]
            if (min(stacked_reward_train) < min_y):
                min_y = min(stacked_reward_train)
            if (max(stacked_reward_train) > max_y):
                max_y = max(stacked_reward_train)
            data1= plt.scatter(np.array(train_itr), stacked_reward_train, alpha=0.4, label='training data')
            data3, = plt.plot(np.array(train_itr), trendline(stacked_reward_train), label='training avg')
            data.append(data1)
            data.append(data3)


        if len(data_outer)==0:
            data_outer.append(data2)
            data_outer.append(data4)
            names_outer.append("test" )  # 'test data ' + str(init_m))
            names_outer.append("test avg")  # 'test avg.' + str(init_m))

        if len(data_outer)==2 and plot_train and len(avg_reward)>0:
                data_outer.append(data1)
                data_outer.append(data3)
                names_outer.append("train")  # 'test data ' + str(init_m))
                names_outer.append("train avg")  # 'test avg.' + str(init_m))



        plt.subplots_adjust(left=0.15, bottom=0.125, right=0.9, top=0.9, wspace=None, hspace=0.25)

        fig.text(0.5, 0.04, 'Epoch', ha='center', fontdict={'size':'large'})
        fig.text(0.04, 0.5, title, va='center', rotation='vertical', fontdict={'size':'large'})
        fig.legend(data_outer, names_outer, loc='lower right', shadow=True, ncol=2)

        if save_plots and save_regular_plots:
            print("Saved  " + timestamp + filename)


            fig.savefig(os.path.join(target_dir, timestamp + filename))
        else:
            plt.show()





def plot_separately_cars(train_itr, avg_reward, test_itr, avg_reward_test, title, filename,plot_train):

    data_outer = []
    names_outer = []
    fig = plt.figure(figsize=(8.0, 5.0))
    indx=1


    for init_m, reward_init_m in enumerate(avg_reward_test):
        data = []

        min_y = sys.maxsize
        max_y = -sys.maxsize - 1
        # if init_m in init_names:
        #     print str(init_m)+" "+init_names[init_m] +" "+str(len(avg_reward_test[init_m]))
        if len(avg_reward_test[init_m]) > 1 and init_m in {5,7, -1, 9}:
            ax = plt.subplot(2 ,2,indx )
            stacked_reward = np.stack(avg_reward_test[init_m], axis=0)
            if len(stacked_reward.shape) == 2 and stacked_reward.shape[1] == 2:
                stacked_reward = stacked_reward[:, 0]
            if (min(stacked_reward) < min_y):
                min_y = min(stacked_reward)
            if (max(stacked_reward) > max_y):
                max_y = max(stacked_reward)
            data2, = ax.plot(np.array(test_itr[init_m]) * 0.01, stacked_reward, label='test data')
            data4, = ax.plot(np.array(test_itr[init_m]) * 0.01, trendline(stacked_reward), label='test avg')

            data.append(data2)
            data.append(data4)

            if plot_train and len(avg_reward[init_m])>1:

                stacked_reward_train = np.stack(avg_reward[init_m], axis=0)

                if len(stacked_reward_train.shape) == 2 and stacked_reward_train.shape[1] == 2:
                    stacked_reward_train = stacked_reward_train[:, 0]
                if (min(stacked_reward_train) < min_y):
                    min_y = min(stacked_reward_train)
                if (max(stacked_reward_train) > max_y):
                    max_y = max(stacked_reward_train)
                data1= ax.scatter(np.array(train_itr[init_m]) * 0.01, stacked_reward_train,  alpha=0.4,label='training data')
                data3, = ax.plot(np.array(train_itr[init_m]) * 0.01, trendline(stacked_reward_train), label='training avg')
                data.append(data1)
                data.append(data3)


            ax.set_title(init_names[init_m])
            if len(data_outer) == 0:
                data_outer.append(data2)
                data_outer.append(data4)
                names_outer.append("test")  # 'test data ' + str(init_m))
                names_outer.append("test avg")  # 'test avg.' + str(init_m))
                if plot_train and len(avg_reward[init_m])>1:
                    data_outer.append(data1)
                    data_outer.append(data3)
                    names_outer.append("train")  # 'test data ' + str(init_m))
                    names_outer.append("train avg")  # 'test avg.' + str(init_m))
            indx = indx + 1
    plt.subplots_adjust(left=0.15, bottom=0.125, right=0.9, top=0.9, wspace=None, hspace=0.25)

    fig.text(0.5, 0.04, 'Epoch', ha='center', fontdict={'size':'large'})
    fig.text(0.04, 0.5, title, va='center', rotation='vertical', fontdict={'size':'large'})
    fig.legend(data_outer, names_outer, loc='lower right', shadow=True, ncol=2)

    if save_plots and save_regular_plots:
        #print "Saved  " + timestamp + filename


        fig.savefig(os.path.join(target_dir, timestamp + filename))
    else:
        plt.show()


def plot_gradient_by_sem_channel(in_2D):
    non_zero_channels = []
    if len(gradients) > 0:
        sz = grads[0].shape
        print("gradient shape "+str(sz))
        if len(sz) > 5:
            in_2D = False
        if len(sz)>3:
            for channel in range(sz[-2]):
                print("Grads "+str(np.sum(np.abs(grads[0][:, :, :, channel])))+" "+str(grads[0][:, :, :, channel].shape))
                # print np.sum(np.abs(grads[0][:, :, :, channel,0]))
                # print np.sum(np.abs(grads[1][:, :, :, channel,0]))
                # print np.sum(np.abs(grads[-1][:, :, :, channel]))
                if not in_2D:
                    if np.any(grads[0][:, :, :, :, channel]) > 0:
                        non_zero_channels.append(channel)
                        print("NON_zero channel" + str(channel))
                else:
                    if np.any(grads[0][:, :, :, channel]) > 0:
                        non_zero_channels.append(channel)
                        print("NON_zero channel" + str(channel))
            fig1, axarr = plt.subplots(len(non_zero_channels) // 3 + 1, 3)

            plt.suptitle("Gradients conv, channel wise")

            for channel in range(len(non_zero_channels)):
                #print "Channel "+str(channel)

                if not in_2D:
                    if sz[-1] > 1:
                        local = grads[0][:, :, :, :, non_zero_channels[channel]].reshape(sz[0], sz[1] * sz[2] * sz[3]*sz[-1])
                    else:
                        local = grads[0][:, :, :, :, non_zero_channels[channel]].reshape(sz[0], sz[1] * sz[2] * sz[3])
                else:
                    if sz[-1]>1:
                        local = grads[0][:, :, :, non_zero_channels[channel]].reshape(sz[0], sz[1] * sz[2]*sz[-1])
                    else:
                        local = grads[0][:, :, :, non_zero_channels[channel]].reshape(sz[0], sz[1] * sz[2])

                local = np.mean(np.absolute(local), axis=1)
                if len(non_zero_channels) // 3==0:
                    axarr[channel % 3].plot(np.squeeze(local))
                else:
                    axarr[channel // 3, channel % 3].plot(np.squeeze(local))

                if non_zero_channels[channel] >= len(labels_to_use):
                    axarr[channel // 3, channel % 3].set_title(
                        labels_to_use[non_zero_channels[channel] % len(labels_to_use)] + " t=" + str(
                            non_zero_channels[channel] // len(labels_to_use)))
                else:
                    if len(non_zero_channels) // 3 == 0:
                        axarr[ channel % 3].set_title(labels_to_use[non_zero_channels[channel]])
                    else:
                        axarr[channel // 3, channel % 3].set_title(labels_to_use[non_zero_channels[channel]])
                if len(non_zero_channels) // 3 == 0:
                    axarr[ channel % 3].set_xlabel('train_itr')
                else:
                    axarr[channel // 3, channel % 3].set_xlabel('train_itr')
            if save_plots:
                print("Saved  " + timestamp + 'gradient_conv2.png')
                fig1.savefig(os.path.join(target_dir, timestamp + 'gradient_conv2.png'))
            else:
                plt.show()
    return non_zero_channels

def plot_gradient_softmax():
    if len(gradients) > 0:
        fig1, axarr = plt.subplots(3, 3)
        #for channel in grads[-1]:
        itrs=list(range(grads[-1].shape[0]))
        # fig=plt.figure()
        action_label = [ 'downL', 'down', 'downR', 'left', 'stand', 'right','upL', 'up', 'upR']
        #[0:'downL', 1:'down', 2:'downR', 3:'left', 4:'stand', 5:'right',6:'upL', 7:'up', 8:'upR']

        #f=np.isnan(grads[-1][:,0])

        #plt.plot(itrs, np.absolute(grads[-1][:,0]), label='NW')
        for channel in range(len(grads[-1][0,:])):

            local = np.absolute(grads[-1][:, channel])
            #print "Local: "+str( local.shape)
            axarr[channel // 3, channel % 3].plot(np.squeeze(local))
            if non_zero_channels[channel] >= len(action_label):
                axarr[channel // 3, channel % 3].set_title(
                    action_label[non_zero_channels[channel] % len(action_label)] + " t=" + str(
                        non_zero_channels[channel] // len(action_label)))
            else:
                axarr[channel // 3, channel % 3].set_title(action_label[non_zero_channels[channel]])
            axarr[channel // 3, channel % 3].set_xlabel('train_itr')

        if save_plots:
            print("Saved  " + timestamp + 'gradient_biases_fc2.png')
            fig1.savefig(os.path.join(target_dir, timestamp + 'gradient_biases_fc2.png'))
        else:
            plt.show()
        for counter, grad in enumerate(grads):
            print(str(grad.shape) + " " + str(counter))



def plot_gradients_fully_connected():
    fig1, axarr = plt.subplots(3, 3)
    plt.suptitle("Gradients fc")
    for action in range(grads[-1].shape[1]):
        #print("Gradients: "+str(np.sum(np.absolute(grads[-2][:, action]), axis=1).shape))
        axarr[action // 3, action % 3].plot(np.sum(np.abs(grads[-2][:, action]), axis=1))
        axarr[action // 3, action % 3].set_title(actions_names[action])
        axarr[action // 3, action % 3].set_xlabel('train_itr')
    if save_plots:
        print("Saved  " + timestamp + 'gradient_fc2.png')
        fig1.savefig(os.path.join(target_dir, timestamp + 'gradient_fc2.png'))
    else:
        plt.show()


def plot_gradients_softmax(gradients):
    fig1, axarr = plt.subplots(3, 3)
    plt.suptitle("Gradients softmax")
    for action in range(gradients[-1].shape[1]):
        axarr[action // 3, action % 3].plot(np.sum(np.abs(grads[-1][:, action]), axis=1))
        axarr[action // 3, action % 3].set_title(actions_names[gradients])
        axarr[action // 3, action % 3].set_xlabel('train_itr')
        axarr[action // 3, action % 3].set_ylabel('Gradient Softmax')
    if save_plots:
        print("Saved  " + timestamp + 'gradient_softmax2.png')
        fig1.savefig(os.path.join(target_dir, timestamp + 'gradient_softmax2.png'))
    else:
        plt.show()

def plot_gradients_car(gradients,timestamp):
    fig1, axarr = plt.subplots(len(gradients),1)
    plt.suptitle("Gradients car")
    i=0
    for grad in gradients:
        # print (" size "+str(grad.shape))
        for j in range(grad.shape[1]):
            if len(gradients)==1:
                axarr.plot(grad[:,j].flatten(), label=str(j))
                axarr.set_title("Weight " +str(i))
                axarr.set_xlabel('train_itr')
                axarr.set_ylabel('Gradient car')
            else:

                axarr[i].plot(grad[:,j].flatten(), label=str(j))
                axarr[i].set_title("Weight " +str(i))
                axarr[i].set_xlabel('train_itr')
                axarr[i].set_ylabel('Gradient car')
        axarr[i].legend(loc='upper right',ncol=np.int(np.ceil(grad.shape[1]/3)))
        i=i+1

    if save_plots:
        print("Saved  " + timestamp + 'gradient_car.png')
        fig1.savefig(os.path.join(target_dir, timestamp + 'gradient_car.png'))
    else:
        plt.show()

def plot_weights_conv(non_zero_channels, in_2D):

    import pickle

    with open(weights[0][1], 'rb') as handle:
        b = pickle.load(handle, encoding="latin1", fix_imports=True)
    weights_holder = []
    for indx, var in enumerate(b):
        shape = [len(weights)]

        for dim in var.shape:
            shape.append(dim)
        # print(str(indx)+" "+str(var.shape))
        weights_holder.append(np.zeros(shape))
    itr = 0
    for pair in sorted(weights):
        with open(pair[1], 'rb') as handle:
            b = pickle.load(handle, encoding="latin1", fix_imports=True)
            for indx, var in enumerate(b):
                weights_holder[indx][itr, :] = var
            #print
        itr += 1
    sz = weights_holder[0].shape
    # print("Weights holder : "+str(sz))
    fig1, axarr = plt.subplots(len(non_zero_channels) // 3 + 1, 3)
    plt.suptitle("Weights conv, channel wise")
    weights_abs_value = []
    names_labels = []

    for channel in range(len(non_zero_channels)):
        #print channel
        if (car_people and channel!=5) or car_people==False:
            if not in_2D:

                local = weights_holder[0][:, :, :, :, non_zero_channels[channel]].reshape(sz[0], sz[1] * sz[2] * sz[3]*sz[-1])
            else:

                local = weights_holder[0][:, :, :, non_zero_channels[channel]].reshape(sz[0], sz[1] * sz[2]*sz[-1])

            weights_abs_value.append(np.mean(np.abs(np.squeeze(local))))
            local = np.mean(local, axis=1)
            if len(non_zero_channels) // 3 == 0:
                axarr[channel % 3].plot(np.squeeze(local))
            else:
                axarr[channel // 3, channel % 3].plot(np.squeeze(local))
            names_labels.append(labels_to_use[non_zero_channels[channel] % len(labels_to_use)])
            if non_zero_channels[channel] >= len(labels_to_use):
                axarr[channel // 3, channel % 3].set_title(
                    labels_to_use[non_zero_channels[channel] % len(labels_to_use)] + " t=" + str(
                        non_zero_channels[channel] // len(labels_to_use)))
            else:
                if len(non_zero_channels) // 3 == 0:
                    axarr[ channel % 3].set_title(labels_to_use[non_zero_channels[channel]])
                else:
                    axarr[channel // 3, channel % 3].set_title(labels_to_use[non_zero_channels[channel]])
            if len(non_zero_channels) // 3 == 0:
                axarr[ channel % 3].set_xlabel('train_itr')
            else:
                axarr[channel // 3, channel % 3].set_xlabel('train_itr')
    if save_plots:
        print("Saved  " + timestamp + 'weights_conv2.png')
        fig1.savefig(os.path.join(target_dir, timestamp + 'weights_conv2.png'))
    else:
        plt.show()
    # for channel in range(len(grads[-1][0, :])):
    #
    #     local = grads[-1][:, channel]
    #     axarr[channel / 3, channel % 3].plot(np.squeeze(local))
    #     if non_zero_channels[channel] >= len(action_label):
    #         axarr[channel / 3, channel % 3].set_title(
    #             action_label[non_zero_channels[channel] % len(action_label)] + " t=" + str(
    #                 non_zero_channels[channel] / len(action_label)))
    #     else:
    #         axarr[channel / 3, channel % 3].set_title(action_label[non_zero_channels[channel]])
    #     axarr[channel / 3, channel % 3].set_xlabel('train_itr')
    #
    # if save_plots:
    #     print "Saved  " + timestamp + 'gradient_fc.png'
    #     fig1.savefig(os.path.join(target_dir, timestamp + 'gradient_fc.png'))
    # else:
    #     plt.show()
    for pair in sorted(zip(weights_abs_value, names_labels), key=lambda tup: tup[0], reverse=True):
        print("%s & %.3e \\" % (pair[1], pair[0]))
    return weights_holder

def plot_weights_direction(weights_holder):
    global fig1, axarr
    fig1, axarr = plt.subplots(2, 3)
    plt.suptitle("Segmentation weights conv, direction wise")
    if in_2D:
        local = weights_holder[0][:, :, :, 3]
    else:
        local = weights_holder[0][:, :, :, :, 3]
    for direction in range(3):
        cur = np.mean(local, axis=direction + 1)
        for direction2 in range(2):
            axarr[direction2, direction].plot(np.squeeze(np.mean(cur, axis=direction2 + 1)))
            axarr[direction2, direction].set_title("direction " + str(direction + 1))
            axarr[direction2, direction].set_xlabel('train_itr')
    if save_plots:
        print("Saved  " + timestamp + 'weights_seg_conv2.png')
        fig1.savefig(os.path.join(target_dir, timestamp + 'weights_seg_conv2.png'))
    else:
        plt.show()


def plot_weights_fully_connected(weights_holder):
    global fig1, axarr, action
    fig1, axarr = plt.subplots(3, 3)
    plt.suptitle("Weights fc")
    for action in range(weights_holder[-1].shape[1]):
        axarr[action // 3, action % 3].plot(weights_holder[-2][:, action])
        axarr[action // 3, action % 3].set_title(actions_names[action])
        axarr[action // 3, action % 3].set_xlabel('train_itr')
    if save_plots:
        print("Saved  " + timestamp + 'weights_fc2.png')
        fig1.savefig(os.path.join(target_dir, timestamp + 'weights_fc2.png'))
    else:
        plt.show()


def plot_weights_fc(weights_holder):
    global fig1, axarr, action
    fig1, axarr = plt.subplots(3, 3)
    plt.suptitle("Weights fc")
    for action in range(weights_holder[-1].shape[1]):
        axarr[action // 3, action % 3].plot(np.mean(weights_holder[-2][:, action], axis=1))
        axarr[action // 3, action % 3].set_title(actions_names[action])
        axarr[action // 3, action % 3].set_xlabel('train_itr')
    if save_plots:
        print("Saved  " + timestamp + 'weights_fc2.png')
        fig1.savefig(os.path.join(target_dir, timestamp + 'weights_fc2.png'))
    else:
        plt.show()


def plot_weights_softmax(weights_holder):
    global fig1, axarr, action
    fig1, axarr = plt.subplots(3, 3)
    plt.suptitle("Weights biases fc")
    print(weights_holder[-1])
    for action in range(weights_holder[-1].shape[1]):
        axarr[action // 3, action % 3].plot(weights_holder[-1][:, action])
        axarr[action // 3, action % 3].set_title(actions_names[action])
        axarr[action // 3, action % 3].set_xlabel('train_itr')
    if save_plots:
        print("Saved  " + timestamp + 'weights_biases_fc2.png')
        fig1.savefig(os.path.join(target_dir, timestamp + 'weights_biases2.png'))
    else:
        plt.show()

def plot_weights_car(weights_holder,timestamp):

    fig1, axarr = plt.subplots(len(weights_holder))
    plt.suptitle("Weights car")
    i=0
    max_j=0
    for weight in weights_holder:
        for j in range(weight.shape[1]):
            max_j=max(max_j, j)
            if len(weights_holder)==1:
                axarr.plot(weight[:,j].flatten(), label=str(j))
                axarr.set_title("Weight" +str(i))
                axarr.set_xlabel('train_itr')

            else:
                axarr[i].plot(weight[:,j].flatten(), label=str(j))
                axarr[i].set_title("Weight" +str(i))
                axarr[i].set_xlabel('train_itr')
        axarr[i].legend(loc='upper right',ncol=np.int(np.ceil(weight.shape[1]/3)))
        i=i+1
    # labels = []
    # for j in range(max_j):
    #     labels.append(str(j))
    # fig1.legend(labels, loc='upper right',ncol=np.int(np.ceil(weight.shape[1]/3)))
    if save_plots:
        print("Saved  " + timestamp + 'weights_car.png')
        fig1.savefig(os.path.join(target_dir, timestamp + 'weights_car.png'))
    else:
        plt.show()

################################################################# Make Table

#init_names= {-1: "training", 0: "training", 1: "On pedestrian", 3: "On pavement",
# 2: "By car", 4: "Random",5: "car_env", 6: "Near pedestrian",7:"pedestrian environment", 9:"average"}

def make_table(rewards, hit_objs, ped_traj, people_hit, entropy, dist_travelled, car_hit, pavement,dist_left, sucesses,
               plot_iterations_test, people_heatmap, nbr_direction_switches , norm_dist_travelled,likelihood_actions,
               likelihood_full,prediction_error,speed_mean, speed_var,continous,collision_between_car_and_agent_test=[], eval=False):
    if not eval:
        best_itr=-1
        best_reward=-100
        values=[]
        for itr, pair in enumerate(rewards[general_id]):
            if len(pair)>0:
                values.append(pair[0])
        print ("Best iter reward "+str(np.argmax(values))+ " len of file "+str(len(rewards[general_id]))+" values "+str(values))
        best_itr = -1
        best_reward = -100
        values=[]
        jump_rate=1
        if len(collision_between_car_and_agent_test)>0:
            if len(collision_between_car_and_agent_test[general_id])==2*len(rewards[general_id]):
                jump_rate=2
            for itr, pair in enumerate(collision_between_car_and_agent_test[general_id]):
                if len(pair) > 0 and itr%jump_rate==0:
                    values.append(pair[0])
            print("Best iter collisisons " + str(np.argmax(values))+ " len of file "+str(len(collision_between_car_and_agent_test[general_id]))+" len values "+str(len(values))+" values "+str(values))
    else:
        best_itr=0
    final=False
    non_valid_keys={}#{-1,5,7}
    if final:
        non_valid_keys={-1,5,7,8,0}
    print("All keys ")
    for key, val in enumerate(rewards):
        if len(val) > 0:

           print(key)


    #print(non_valid_keys)
    print("Rewards: " +str(rewards))
    print("Collisions with pedestrian : " + str(people_hit))
    non_zero_nbrs=[]
    init = "Metric"
    for key, val in enumerate(rewards):
        if len(val)>0 and key not in non_valid_keys:

            non_zero_nbrs.append(key)
            init = init + " & " + init_names[key]

    init += " \\\\ \hline"
    print(init)
    init = "Cum. reward"
    for key in non_zero_nbrs:
        if best_itr< len(rewards[key]):
            rew=rewards[key][best_itr]

            val1= '%s' % float('%.2g' % rew[0])
            val2 = '%s' % float('%.2g' % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)

    init = "Hit objs"
    for key in non_zero_nbrs:
        if best_itr < len(hit_objs[key]):
            rew = hit_objs[key][best_itr]
            val1 = '%s' % float('%.2g' % rew[0])
            val2 = '%s' % float('%.2g' % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)
    if not continous:
        init = "Entropy"
        for key in non_zero_nbrs:
            if best_itr < len(entropy[key]):
                rew = entropy[key][best_itr]
                val1 = '%s' % float('%.2g' % rew[0])
                val2 = '%s' % float('%.2g' % rew[1])
                if final:
                    init = init + " & " + str(val1)
                else:
                    init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
            else:
                init = init + " & -"
        init += " \\\\ \hline"
    print(init)
    init = "Distance travelled (m)"
    for key in non_zero_nbrs:
        if best_itr < len(dist_travelled[key]):
            rew = dist_travelled[key][best_itr]
            val1 = '%s' % float('%.2g' % rew[0])
            val2 = '%s' % float('%.2g' % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)
    init = "Ped. Trajectory"
    for key in non_zero_nbrs:
        if best_itr < len(ped_traj[key]):
            rew = ped_traj[key][best_itr]
            val1 = '%s' % float('%.2g' % rew[0])
            val2 = '%s' % float('%.2g' % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)
    init = "People hit"
    for key in non_zero_nbrs:
        if best_itr < len(people_hit[key]):
            rew = people_hit[key][best_itr]
            val1 = '%s' % float('%.2g' % rew[0])
            val2 = '%s' % float('%.2g' % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)
    # init = "Prediction error (m)"
    # for key in non_zero_nbrs:
    #     rew = people_hit[key][best_itr]
    #     init = init + " & " + str(round(rew[0], 1)) + " & " + str(round(rew[1], 1))
    # init += " \\ \hline"
    # print init
    init = "Hit by car"
    for key in non_zero_nbrs:
        if best_itr < len(car_hit[key]):
            rew = car_hit[key][best_itr]
            val1 = '%s' % float('%.2g' % rew)
            init = init + " & " + str(val1)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)
    init = "Pavement"
    for key in non_zero_nbrs:
        if best_itr < len(pavement[key]):
            rew = pavement[key][best_itr]
            val1 = '%s' % float('%.2g' % rew[0])
            val2 = '%s' % float('%.2g' % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)
    init = "Distance to goal"
    for key in non_zero_nbrs:
        if best_itr < len(dist_left[key]):
            rew = dist_left[key][best_itr]
            val1 = '%s' % float('%.2g' % rew[0])
            val2 = '%s' % float('%.2g' % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)
    init = "No of sucesses"
    for key in non_zero_nbrs:
        if best_itr < len(successes[key]):
            rew = successes[key][best_itr]
            val1 = '%s' % float('%.2g' % rew)
            init = init + " & " + str(val1)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)

    init = "Heatmap"
    for key in non_zero_nbrs:
        if best_itr < len(people_heatmap[key]):
            rew = people_heatmap[key][best_itr]
            val1 = '%s' % float('%.2g' % rew[0])
            val2 = '%s' % float('%.2g' % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)


    init = "Nbr dir. switches"
    for key in non_zero_nbrs:
        if best_itr < len(nbr_direction_switches[key]):
            rew = nbr_direction_switches[key][best_itr]
            val1 = '%s' % float('%.2g' % rew[0])
            val2 = '%s' % float('%.2g' % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)

    init = "Norm. distance"
    for key in non_zero_nbrs:
        if best_itr < len(norm_dist_travelled[key]):
            rew = norm_dist_travelled[key][best_itr]
            val1 = '%s' % float('%.2g' % rew[0])
            val2 = '%s' % float('%.2g' % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)

    init = "NLL actions"
    for key in non_zero_nbrs:
        if best_itr < len(likelihood_actions[key])and len(likelihood_actions[key])>0:
            print (len(likelihood_actions[key]))
            print (best_itr)
            rew = likelihood_actions[key][best_itr]
            val1 = '%s' % float('%.2g' % rew[0])
            val2 = '%s' % float('%.2g' % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)

    init = "NLL vel"
    for key in non_zero_nbrs:
        if best_itr < len(likelihood_full[key])and len(likelihood_full[key])>0:
            rew = likelihood_full[key][best_itr]
            val1 = '%s' % float('%.2g' % rew[0])
            val2 = '%s' % float('%.2g' % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)

    init = "ADE"
    for key in non_zero_nbrs:
        if best_itr < len(prediction_error[key])and len(prediction_error[key])>0:
            rew = prediction_error[key][best_itr]
            val1 = '%s' % float('%.2g' % rew[0])
            val2 = '%s' % float('%.2g' % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)

    init = "Mean speed"
    for key in non_zero_nbrs:
        if best_itr < len(speed_mean[key]) and len(speed_mean[key]) > 0:
            rew = speed_mean[key][best_itr]
            val1 = '%s' % float('%.2g' % rew[0])
            val2 = '%s' % float('%.2g' % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)

    init = "STD speed"
    for key in non_zero_nbrs:
        if best_itr < len(speed_var[key]) and len(speed_var[key]) > 0:
            rew = speed_var[key][best_itr]
            val1 = '%s' % float('%.2g' % rew[0])
            val2 = '%s' % float('%.2g' % rew[1])
            if final:
                init = init + " & " + str(val1)
            else:
                init = init + " & " + str(val1) + " $\mypm$ " + str(val2)
        else:
            init = init + " & -"
    init += " \\\\ \hline"
    print(init)
################################################################### Main starts here!!!!!


def read_files_and_plot(make_movie=False):
    global  test_files_cars, init_cars, keys, goal_position_mode, goal_prior_mode, init_position_mode, init_prior_mode, entropy, successes, pos, values, weights, gradients, grads, non_zero_channels, loss, frame_rate, frame_time, temporal_case, timestamp, target_dir, labels_to_use
    eval_path = settings.evaluation_path
    print("Evaluation path " + str(eval_path))
    from datetime import datetime
    for timestamp in timestamps:

        plt.close("all")

        # For old files do not update plots!
        if "2018-" in timestamp or "2017-" in timestamp or "2019-01" in timestamp or "2019-02" in timestamp:
            make_movie = False
        dt_object = datetime.strptime(timestamp, '%Y-%m-%d-%H-%M-%S.%f')
        if dt_object < datetime.strptime("2021-04-08-00-00-00.421640", '%Y-%m-%d-%H-%M-%S.%f'):
            make_movie = False
        if dt_object > datetime.strptime("2020-11-20-13-55-01.877475",
                                         '%Y-%m-%d-%H-%M-%S.%f') and dt_object < datetime.strptime(
                "2021-04-08-00-00-01.877475", '%Y-%m-%d-%H-%M-%S.%f'):
            break

        # find statistics files.
        path = stat_path + timestamp + ending + '*'
        match = path + "/*.npy"
        match2 = path + "*.npy"
        print("Pattern " + str(match))
        files = glob.glob(match)
        print("Pattern " + str(match2))
        files = files + glob.glob(match2)
        print("Number of files: "+str(len(files)))

        # Read settings file
        labels_to_use, name_movie, target_dir, num_measures, semantic_channels_separated, in_2D, temporal_case, continous, num_measures_car, gaussian = read_settings_file(timestamp)
        if len(labels_to_use) > 0 or supervised:

            # Set paths
            movie_path = settings.statistics_dir + "/agent/"  # os.path.join(settings.img_dir[0], "statistics/")
            name_movie_eval = settings.name_movie_eval
            # print (settings.name_movie_eval)

            # Make movie
            make_movies(movie_path, name_movie_eval,name_movie,target_dir,timestamp)

            if len(files) > 0 or supervised:
                # Plot car / people enviornment results
                filenames_itr_cars, test_files_cars, filenames_itr_people, test_files_people, filenames_itr, test_files, reconstructions_test, iterations, iterations_cars, special_cases, init_maps, init_cars, init_maps_test, init_cars_test, learn_car, learn_car_test, init_map_stats, init_map_stats_test = sort_files(
                    files)


                train_itr = {}
                train_counter = 0
                test_points = {}
                keys = list(iterations.keys())

                for itr in sorted(keys):
                    training = iterations[itr]
                    if training:
                        train_itr[itr] = train_counter
                        if itr in special_cases:
                            test_points[itr] = train_counter
                        if itr not in iterations_cars:  # or toy_case:
                            train_counter = train_counter + 1
                    else:
                        test_points[itr] = train_counter
                        if itr in special_cases:
                            train_itr[itr] = train_counter
                # #print ("train itr "+str(train_itr))
                # print ("Test itr")
                # print(sorted(test_points))
                # print("Number of test itr "+str(len(filenames_itr)))

                # Get statistics on validation set
                if len(filenames_itr) >= 1:
                    # if not temporal_case:
                    rewards_train, hit_objs_train, ped_traj_train, people_hit_train, entropy_train, dist_left_train, dist_travelled_train, car_hit_train, pavement_train, plot_iterations, people_heatmap_train, nbr_inits_train, successes_train, norm_dist_travelled_train, nbr_direction_switches_train, loss_train, one_step_train, one_steps, prob_of_max_action, most_freq_action, freq_most_freq_action, variance, speed_mean, speed_var, likelihood_actions, likelihood_full, prediction_error, dist_to_car, time_to_collision, init_dist_to_goal, goal_locations, goal_times, init_locations, goal_speed, init_dist_to_car, speed_to_dist_correlation, _ = get_stats(
                        filenames_itr,
                        train_itr, continous, temporal_case)
                    # print(init_locations)
                # print ("Init maps test len "+str(sorted(init_maps_test)))
                # print ("Init maps len " + str(len(init_maps)))
                # print (" train: "+str(init_map_stats))
                if len(init_maps) >= 1 or len(init_map_stats) > 0:
                    # print (" Training data")
                    entropy_maps, entropy_prior_maps, kl_to_prior_maps, diff_to_prior_maps, cars_vel_x, cars_vel_y, cars_vel_speed, cars_pos_x, cars_pos_y, iterations_init, entropy_maps_goal, entropy_prior_maps_goal, kl_to_prior_maps_goal, diff_to_prior_maps_goal, goal_position_mode, goal_prior_mode, gaussian, init_position_mode, init_prior_mode = get_init_stats(
                        init_maps, init_map_stats, init_cars, train_itr)

                if len(learn_car) > 0:
                    rewards_train_car, hit_objs_train_car, ped_traj_train_car, people_hit_train_car, entropy_train_car, dist_left_train_car, dist_travelled_train_car, car_hit_train_car, pavement_train_car, plot_iterations_car, people_heatmap_train_car, nbr_inits_train_car, successes_train_car, norm_dist_travelled_train_car, nbr_direction_switches_train_car, loss_train_car, one_step_train_car, one_steps_car, prob_of_max_action_car, most_freq_action_car, freq_most_freq_action_car, variance_car, speed_mean_car, speed_var_car, likelihood_actions_car, likelihood_full_car, prediction_error_car, dist_to_car_car, time_to_collision_car, init_dist_to_goal_car, goal_locations_car, goal_times_car, init_locations_car, goal_speed_car, init_dist_to_car_car, speed_to_dist_correlation_car, collision_between_car_and_agent = get_stats(
                        learn_car,
                        train_itr, continous, temporal_case, car_case=True)

                if len(test_files) > 0:
                    # print("Number of test files " + str(len(test_files)))
                    rewards, hit_objs, ped_traj, people_hit, entropy, dist_left, dist_travelled, car_hit, pavement, plot_iterations_test, people_heatmap, nbr_inits, successes, norm_dist_travelled, nbr_direction_switches, loss_test, one_step_test, one_steps_test, prob_of_max_action_test, most_freq_action_test, freq_most_freq_action_test, variance_test, speed_mean_test, speed_var_test, likelihood_actions_test, likelihood_full_test, prediction_error_test, dist_to_car_test, time_to_collision_test, init_dist_to_goal_test, goal_locations_test, goal_times_test, init_locations_test, goal_speed_test, init_dist_to_car_test, speed_to_dist_correlation_test, _ = get_stats(
                        test_files, test_points, continous, temporal_case)


                else:

                    car_hit, dist_left, dist_to_car, dist_to_car_test, dist_travelled, freq_most_freq_action_test, goal_locations_test, goal_times_test, hit_objs, init_dist_to_goal_test, loss_test, most_freq_action_test, nbr_direction_switches, nbr_inits, norm_dist_travelled, pavement, ped_traj, people_heatmap, people_hit, plot_iterations_test, prob_of_max_action_test, rewards, speed_mean_test, speed_var_test, time_to_collision_test, variance_test = initialize_empty(
                        car_hit, dist_left, dist_to_car, dist_to_car_test, dist_travelled, freq_most_freq_action_test,
                        goal_locations_test, goal_times_test, hit_objs, init_dist_to_goal_test, loss_test,
                        most_freq_action_test, nbr_direction_switches, nbr_inits, norm_dist_travelled, pavement,
                        ped_traj, people_heatmap, people_hit, plot_iterations_test, prob_of_max_action_test, rewards,
                        speed_mean_test, speed_var_test, time_to_collision_test, variance_test)

                if len(init_maps_test) >= 1 or len(init_map_stats_test) >= 1:
                    print(" Get test data")
                    entropy_maps_test, entropy_prior_maps_test, kl_to_prior_maps_test, diff_to_prior_maps_test, cars_vel_test_x, cars_vel_test_y, cars_vel_test_speed, cars_pos_test_x, cars_pos_test_y, iterations_test_init, entropy_maps_test_goal, entropy_prior_maps_test_goal, kl_to_prior_maps_test_goal, diff_to_prior_maps_test_goal, goal_position_mode_test, goal_prior_mode_test, gaussian_test, init_position_mode_test, init_prior_mode_test = get_init_stats(
                        init_maps_test, init_map_stats_test, init_cars_test, test_points)

                if len(learn_car_test) > 0:
                    rewards_car, hit_objs_car, ped_traj_car, people_hit_car, entropy_car, dist_left_car, dist_travelled_car, car_hit_car, pavement_car, plot_iterations_test_car, people_heatmap_car, nbr_inits_car, successes_car, norm_dist_travelled_car, nbr_direction_switches_car, loss_test_car, one_step_test_car, one_steps_test_car, prob_of_max_action_test_car, most_freq_action_test_car, freq_most_freq_action_test_car, variance_test_car, speed_mean_test_car, speed_var_test_car, likelihood_actions_test_car, likelihood_full_test_car, prediction_error_test_car, dist_to_car_test_car, time_to_collision_test_car, init_dist_to_goal_test_car, goal_locations_test_car, goal_times_test_car, init_locations_test_car, goal_speed_test_car, init_dist_to_car_test_car, speed_to_dist_correlation_car_test, collision_between_car_and_agent_test = get_stats(
                        learn_car_test, test_points, continous, temporal_case, car_case=True)
                    # print(rewards)'
                    # print(rewards)'

                # Plot cars/ people stats
                if len(filenames_itr_cars) > 0 and len(filenames_itr_cars[0]) > 0 and False:
                    sz_train = np.load(filenames_itr_cars[0][1]).shape
                    # print sz_train
                    indx_train_car, statistics_car, statistics_test_car, test_itr_car = get_cars_stats(
                        filenames_itr_cars, sz_train)
                    sz_train = np.load(filenames_itr_people[0][1]).shape
                    # print sz_train
                    indx_train_people, statistics_people, statistics_test_people, test_itr_people = get_cars_stats(
                        filenames_itr_people,
                        sz_train)

                ################################## Plot
                if save_regular_plots and len(filenames_itr) > 1:

                    do_plotting(car_hit, car_hit_car, car_hit_train, car_hit_train_car, collision_between_car_and_agent,
                                collision_between_car_and_agent_test, continous, diff_to_prior_maps,
                                diff_to_prior_maps_goal, diff_to_prior_maps_test, diff_to_prior_maps_test_goal,
                                dist_left, dist_left_car, dist_left_train, dist_left_train_car, dist_to_car,
                                dist_to_car_test, dist_travelled, dist_travelled_car, dist_travelled_train,
                                dist_travelled_train_car, entropy, entropy_car, entropy_maps, entropy_maps_goal,
                                entropy_maps_test, entropy_maps_test_goal, entropy_prior_maps, entropy_prior_maps_goal,
                                entropy_prior_maps_test, entropy_prior_maps_test_goal, entropy_train, entropy_train_car,
                                freq_most_freq_action, freq_most_freq_action_test, gaussian, goal_locations,
                                goal_locations_test, goal_position_mode, goal_position_mode_test, goal_prior_mode,
                                goal_prior_mode_test, goal_speed, goal_speed_test, goal_times, goal_times_test,
                                hit_objs, hit_objs_car, hit_objs_train, hit_objs_train_car, init_dist_to_car,
                                init_dist_to_car_test, init_dist_to_goal, init_dist_to_goal_test, init_locations,
                                init_locations_test, init_map_stats_test, init_maps_test, init_position_mode,
                                init_position_mode_test, init_prior_mode, init_prior_mode_test, iterations_init,
                                iterations_test_init, learn_car_test, loss_test, loss_test_car, loss_train,
                                loss_train_car, most_freq_action, most_freq_action_car, most_freq_action_test,
                                most_freq_action_test_car, nbr_direction_switches, nbr_direction_switches_train,
                                norm_dist_travelled, norm_dist_travelled_car, norm_dist_travelled_train,
                                norm_dist_travelled_train_car, ped_traj, ped_traj_car, ped_traj_train,
                                ped_traj_train_car, people_heatmap, people_heatmap_car, people_heatmap_train,
                                people_heatmap_train_car, people_hit, people_hit_car, people_hit_train,
                                people_hit_train_car, plot_iterations, plot_iterations_car, plot_iterations_test,
                                plot_iterations_test_car, prob_of_max_action, prob_of_max_action_car,
                                prob_of_max_action_test, prob_of_max_action_test_car, rewards, rewards_car,
                                rewards_train, rewards_train_car, speed_mean, speed_mean_test,
                                speed_to_dist_correlation_car, speed_to_dist_correlation_car_test, speed_var,
                                speed_var_test, successes, successes_car, successes_train, successes_train_car,
                                test_points, time_to_collision, time_to_collision_test, variance, variance_test)

                    get_mat_plot_files(entropy, name_movie,
                       nbr_inits, nbr_inits_train, people_hit_car,
                       plot_iterations_test, rewards, target_dir, dist_travelled_car,rewards_car, timestamp)
                        # (car_hit, dist_left, dist_travelled, entropy, hit_objs, name_movie,
                        #                nbr_direction_switches, nbr_inits, nbr_inits_train, norm_dist_travelled,
                        #                ped_traj, people_heatmap, people_hit, plot_iterations_test, rewards, successes,
                        #                target_dir, timestamp)

                    ################################### Go through gradients and Plot!
                    if not supervised and save_regular_plots:
                        weights, gradients, files, weights_car, gradients_car = find_gradients(path)
                        try:
                            grads, itr = read_gradients(gradients)
                            non_zero_channels = plot_gradient_by_sem_channel(in_2D)
                            # plot_gradient_softmax()
                            # plot_gradients_fully_connected()

                            if len(weights) > 0:
                                weights_holder = plot_weights_conv(non_zero_channels, in_2D)  #
                            # plot_weights_softmax(weights_holder)
                            # plot_weights_fc(weights_holder)
                        except ValueError as IndexError:
                            print("Value Error or Index error")
                        # try:

                        grads_car, itr = read_gradients(gradients_car)
                        # print (" Gradients "+str(grads_car))
                        plot_gradients_car(grads_car, timestamp)

                        if len(weights_car) > 0:
                            weights_c, itr = read_gradients(weights_car)
                            weights_holder = plot_weights_car(weights_c, timestamp)  #
                            # plot_weights_softmax(weights_holder)
                            # plot_weights_fc(weights_holder)
                        # except ValueError as IndexError:
                        #     print("Value Error or Index error")
                    # plot_gradients_fully_connected()#
                    # plot_gradients_softmax()#
                    # plot_weights_direction()
                    # plot_weights_fully_connected()
                    # plot_weights_fc()
                    # plot_weights_softmax()#

                # Make table with results
                if len(filenames_itr) >= 1:
                    make_table(rewards, hit_objs, ped_traj, people_hit, entropy, dist_travelled, car_hit,
                               pavement, dist_left, successes, plot_iterations_test, people_heatmap,
                               nbr_direction_switches,
                               norm_dist_travelled, likelihood_actions, likelihood_full, prediction_error, speed_mean,
                               speed_var,continous,collision_between_car_and_agent_test)

                if len(filenames_itr_cars) >= 1:
                    make_table(rewards_car, hit_objs_car, ped_traj_car, people_hit_car, entropy_car, dist_travelled_car,
                               car_hit_car,
                               pavement_car, dist_left_car, successes_car, plot_iterations_test_car, people_heatmap_car,
                               nbr_direction_switches_car,
                               norm_dist_travelled_car, likelihood_actions_car, likelihood_full_car,
                               prediction_error_car, speed_mean_car, speed_var_car,continous,collision_between_car_and_agent_test)

        ############################################ Evaluate test set
        # find statistics files on test set.
        print("Evaluation ------------------------------")
        path = eval_path + '*' + timestamp

        match = path + "*/*.npy"
        match2 = path + "*.npy"
        # print(path+ "*/*.npy")
        # print(match2)
        files_eval = glob.glob(match)
        files_eval = files_eval + glob.glob(match2)

        # TO DO: ADD car to evaluation, then update this visualization script as well-
        test_files, reconstructions_test, learn_car, init_map_stats, init_maps, init_cars, test_files_poses = sort_files_eval(
            files_eval)
        print("Test files  ------------------------------")
        # print(test_files)
        # print(learn_car)

        # print("Number of files " + str(test_files))
        if len(test_files) > 0:
            train_itr = {}
            train_counter = 0
            test_points = {0: 0}
        if len(test_files) > 0:

            rewards, hit_objs, ped_traj, people_hit, entropy, dist_left, dist_travelled, car_hit, pavement, plot_iterations_test, people_heatmap, nbr_inits, successes, norm_dist_travelled, nbr_direction_switches, loss, one_step, one_steps, prob_of_max_action, most_freq_action, freq_most_freq_action, variance, speed_mean, speed_var, likelihood_actions, likelihood_full, prediction_error, dist_to_car, time_to_collision, init_dist_to_goal, goal_locations, goal_times, init_locations, goal_speed, init_dist_to_car, speed_to_dist_correlation, collision_between_car_and_agent = get_stats(
                test_files, test_points, continous)

            if len(init_maps) >= 1 or len(init_map_stats) >= 1:
                print(" Get test data")
                entropy_maps_test, entropy_prior_maps_test, kl_to_prior_maps_test, diff_to_prior_maps_test, cars_vel_test_x, cars_vel_test_y, cars_vel_test_speed, cars_pos_test_x, cars_pos_test_y, iterations_test_init, entropy_maps_test_goal, entropy_prior_maps_test_goal, kl_to_prior_maps_test_goal, diff_to_prior_maps_test_goal, goal_position_mode_test, goal_prior_mode_test, gaussian_test, init_position_mode_test, init_prior_mode_test = get_init_stats(
                    init_maps, init_map_stats, init_cars, test_points)

            if len(learn_car) > 0:
                rewards_car, hit_objs_car, ped_traj_car, people_hit_car, entropy_car, dist_left_car, dist_travelled_car, car_hit_car, pavement_car, plot_iterations_test_car, people_heatmap_car, nbr_inits_car, successes_car, norm_dist_travelled_car, nbr_direction_switches_car, loss_test_car, one_step_test_car, one_steps_test_car, prob_of_max_action_test_car, most_freq_action_test_car, freq_most_freq_action_test_car, variance_test_car, speed_mean_test_car, speed_var_test_car, likelihood_actions_car, likelihood_full_car, prediction_error_car, dist_to_car_car, time_to_collision_car, init_dist_to_goal_car, goal_locations_car, goal_times_car, init_locations_car, goal_speed_car, init_dist_to_car_car, speed_to_dist_correlation_car, collision_between_car_and_agent = get_stats(
                    learn_car, test_points, continous, temporal_case, car_case=True)
                # test_files, test_points,num_measures

            # Make table
            make_table(rewards, hit_objs, ped_traj, people_hit, entropy, dist_travelled, car_hit, pavement,
                       dist_left, successes, plot_iterations_test, people_heatmap, nbr_direction_switches
                       , norm_dist_travelled, likelihood_actions, likelihood_full, prediction_error, speed_mean,
                       speed_var,continous)

            print("Car  ------------------------------")
            make_table(rewards_car, hit_objs_car, ped_traj_car, people_hit_car, entropy_car, dist_travelled_car,
                       car_hit_car, pavement_car,
                       dist_left_car, successes_car, plot_iterations_test_car, people_heatmap_car,
                       nbr_direction_switches_car
                       , norm_dist_travelled_car, likelihood_actions_car, likelihood_full_car, prediction_error_car,
                       speed_mean_test_car, speed_var_test_car,continous)
            # print("Shape "+ str(one_steps[0].shape)+" "+str(len(one_steps)))

            pred_error = np.mean(one_steps[0], axis=0)

            # if len(pred_error)>1:
            #     scipy.io.savemat(os.path.join(target_dir, timestamp + '_pred_error'), {'prediction_error':pred_error.tolist()})

            frame_rate, frame_time = settings.getFrameRateAndTime()

            # Plot prediction error. Somewhat outdated
            #
            # fig=plt.figure()
            # plt.plot(np.arange(len(pred_error))*frame_time,pred_error)
            # plt.title("Prediction error (m)")
            # plt.show()
            # fig.savefig(os.path.join(target_dir, timestamp + '_pred_error2.png'))


def initialize_empty(car_hit, dist_left, dist_to_car, dist_to_car_test, dist_travelled, freq_most_freq_action_test,
                     goal_locations_test, goal_times_test, hit_objs, init_dist_to_goal_test, loss_test,
                     most_freq_action_test, nbr_direction_switches, nbr_inits, norm_dist_travelled, pavement, ped_traj,
                     people_heatmap, people_hit, plot_iterations_test, prob_of_max_action_test, rewards,
                     speed_mean_test, speed_var_test, time_to_collision_test, variance_test):
    global entropy, successes
    rewards = []
    hit_objs = []
    ped_traj = []
    people_hit = []
    entropy = []
    dist_left = []
    dist_travelled = []
    car_hit = []
    pavement = []
    plot_iterations_test = []
    people_heatmap = []
    dist_to_car = []
    nbr_inits = []
    successes = []
    norm_dist_travelled = []
    nbr_direction_switches = []
    loss_test = []
    one_step_test = []
    one_steps_test = []
    prob_of_max_action_test = []
    most_freq_action_test = []
    freq_most_freq_action_test = []
    variance_test = []
    speed_mean_test = []
    speed_var_test = []
    likelihood_actions_test = []
    likelihood_full_test = []
    init_dist_to_goal_test = []
    time_to_collision_test = []
    dist_to_car_test = []
    goal_locations_test = []
    goal_times_test = []
    return car_hit, dist_left, dist_to_car, dist_to_car_test, dist_travelled, freq_most_freq_action_test, goal_locations_test, goal_times_test, hit_objs, init_dist_to_goal_test, loss_test, most_freq_action_test, nbr_direction_switches, nbr_inits, norm_dist_travelled, pavement, ped_traj, people_heatmap, people_hit, plot_iterations_test, prob_of_max_action_test, rewards, speed_mean_test, speed_var_test, time_to_collision_test, variance_test


def get_mat_plot_files(entropy, name_movie,
                       nbr_inits, nbr_inits_train, people_hit_car,
                       plot_iterations_test, rewards, target_dir, dist_travelled_car,rewards_car, timestamp):
    global pos, values
    bins = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5,13.5,14.5, 15.5 ]
    fig = plt.figure()
    plt.hist([nbr_inits_train, nbr_inits], bins, label=['train', 'test'])
    # print np.histogram(nbr_inits, bins=bins)
    plt.legend(loc='upper right')
    plt.xticks([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    fig.savefig(os.path.join(target_dir, timestamp + "_init_hist2.png"))
    # Save various statistics to mat file sto compare
    if save_mat:
        rewards_mat = np.zeros((16))

        for pos, values in enumerate(rewards):
            if pos == 7:

                for i, val in enumerate(values):

                    rewards_mat[i] = val[0]

        rewards_car_mat = np.zeros(( 16))
        for pos, values in enumerate(rewards_car):
            if pos == 7:
                for i, val in enumerate(values):
                    rewards_car_mat[ i] = val[0]

        people_hit_car_mat = np.zeros(( 16))
        for pos, values in enumerate(people_hit_car):
            if pos == 7:
                for i, val in enumerate(values):
                    people_hit_car_mat[ i] = val[0]

        entropy_mat = np.zeros(( 16))
        for pos, values in enumerate(entropy):
            if pos == 7:
                for i, val in enumerate(values):
                    entropy_mat[ i] = val[0]

        dist_travelled_car_mat = np.zeros((16))
        for pos, values in enumerate(dist_travelled_car):
            if pos == 7:
                for i, val in enumerate(values):
                    dist_travelled_car_mat[i] = val[0]

        itrs_mat = np.zeros(( 16))
        for pos, values in enumerate(plot_iterations_test):
            print(len(values))
            if pos == 7:
                itrs_mat = values[:16]



        mat_dict = {name_movie + '_reward': rewards_mat,
                    name_movie + '_reward_car': rewards_car_mat,
                    name_movie + '_people_hit_car': people_hit_car_mat,
                    name_movie + '_itrs': itrs_mat,
                    name_movie +'_dist_travelled_car':dist_travelled_car_mat
                    }
        print(mat_dict)
        scipy.io.savemat(timestamp+".mat", mat_dict)


def do_plotting(car_hit, car_hit_car, car_hit_train, car_hit_train_car, collision_between_car_and_agent,
                collision_between_car_and_agent_test, continous, diff_to_prior_maps, diff_to_prior_maps_goal,
                diff_to_prior_maps_test, diff_to_prior_maps_test_goal, dist_left, dist_left_car, dist_left_train,
                dist_left_train_car, dist_to_car, dist_to_car_test, dist_travelled, dist_travelled_car,
                dist_travelled_train, dist_travelled_train_car, entropy, entropy_car, entropy_maps, entropy_maps_goal,
                entropy_maps_test, entropy_maps_test_goal, entropy_prior_maps, entropy_prior_maps_goal,
                entropy_prior_maps_test, entropy_prior_maps_test_goal, entropy_train, entropy_train_car,
                freq_most_freq_action, freq_most_freq_action_test, gaussian, goal_locations, goal_locations_test,
                goal_position_mode, goal_position_mode_test, goal_prior_mode, goal_prior_mode_test, goal_speed,
                goal_speed_test, goal_times, goal_times_test, hit_objs, hit_objs_car, hit_objs_train,
                hit_objs_train_car, init_dist_to_car, init_dist_to_car_test, init_dist_to_goal, init_dist_to_goal_test,
                init_locations, init_locations_test, init_map_stats_test, init_maps_test, init_position_mode,
                init_position_mode_test, init_prior_mode, init_prior_mode_test, iterations_init, iterations_test_init,
                learn_car_test, loss_test, loss_test_car, loss_train, loss_train_car, most_freq_action,
                most_freq_action_car, most_freq_action_test, most_freq_action_test_car, nbr_direction_switches,
                nbr_direction_switches_train, norm_dist_travelled, norm_dist_travelled_car, norm_dist_travelled_train,
                norm_dist_travelled_train_car, ped_traj, ped_traj_car, ped_traj_train, ped_traj_train_car,
                people_heatmap, people_heatmap_car, people_heatmap_train, people_heatmap_train_car, people_hit,
                people_hit_car, people_hit_train, people_hit_train_car, plot_iterations, plot_iterations_car,
                plot_iterations_test, plot_iterations_test_car, prob_of_max_action, prob_of_max_action_car,
                prob_of_max_action_test, prob_of_max_action_test_car, rewards, rewards_car, rewards_train,
                rewards_train_car, speed_mean, speed_mean_test, speed_to_dist_correlation_car,
                speed_to_dist_correlation_car_test, speed_var, speed_var_test, successes, successes_car,
                successes_train, successes_train_car, test_points, time_to_collision, time_to_collision_test, variance,
                variance_test):
    print("Plot")
    plot_trains = True
    plot_separately(plot_iterations, rewards_train, plot_iterations_test, rewards, "Average reward",
                    "_avg_reward.png", plot_train=plot_trains)
    #
    plot_separately(plot_iterations, hit_objs_train, plot_iterations_test, hit_objs,
                    "Number of hit objects", "_hit_objs.png", plot_train=plot_trains)
    #
    plot_separately(plot_iterations, ped_traj_train, plot_iterations_test, ped_traj,
                    "Frequency on ped traj", "_ped_traj.png", plot_train=plot_trains)
    #
    plot_separately(plot_iterations, people_hit_train, plot_iterations_test, people_hit,
                    "Number of people hit", "_people_hit.png", plot_train=plot_trains)
    plot_separately(plot_iterations, dist_travelled_train, plot_iterations_test, dist_travelled,
                    "Distance travelled", "_dist_travelled.png", plot_train=plot_trains)
    plot_separately(plot_iterations, car_hit_train, plot_iterations_test, car_hit,
                    "Number of times hit by car", "_car_hit.png", plot_train=plot_trains)
    plot_separately(plot_iterations, people_heatmap_train, plot_iterations_test, people_heatmap,
                    "Average sum from people heatmap",
                    "_avg_heatmap.png", plot_train=plot_trains)
    plot_separately(plot_iterations, dist_left_train, plot_iterations_test, dist_left,
                    "Average min distance left",
                    "_dist_left.png", plot_train=plot_trains)
    plot_separately(plot_iterations, successes_train, plot_iterations_test, successes,
                    "Number of successes",
                    "_success.png", plot_train=plot_trains)
    plot_separately(plot_iterations, norm_dist_travelled_train, plot_iterations_test,
                    norm_dist_travelled,
                    "Normalized distance travelled",
                    "_norm_dist.png", plot_train=plot_trains)
    plot_separately(plot_iterations, nbr_direction_switches_train, plot_iterations_test,
                    nbr_direction_switches,
                    "Number of direction switches",
                    "_dir_switches.png", plot_train=plot_trains)
    plot_separately(plot_iterations, loss_train, plot_iterations_test,
                    loss_test,
                    "Loss",
                    "_loss.png", plot_train=plot_trains)
    if len(learn_car_test):
        plot_separately(plot_iterations_car, rewards_train_car, plot_iterations_test_car, rewards_car,
                        "Car Average reward",
                        "_avg_reward_car.png", plot_train=plot_trains)
        #
        plot_separately(plot_iterations_car, hit_objs_train_car, plot_iterations_test_car, hit_objs_car,
                        "Car Number of hit objects", "_hit_objs_car.png", plot_train=plot_trains)
        #
        plot_separately(plot_iterations_car, ped_traj_train_car, plot_iterations_test_car, ped_traj_car,
                        "Car Frequency on ped traj", "_ped_traj_car.png", plot_train=plot_trains)
        #
        plot_separately(plot_iterations_car, people_hit_train_car, plot_iterations_test_car,
                        people_hit_car,
                        "Car Number of people hit", "_people_hit_car.png", plot_train=plot_trains)
        plot_separately(plot_iterations_car, dist_travelled_train_car, plot_iterations_test_car,
                        dist_travelled_car,
                        "Car Distance travelled", "_dist_travelled_car.png", plot_train=plot_trains)
        plot_separately(plot_iterations_car, car_hit_train_car, plot_iterations_test_car, car_hit_car,
                        "Car Number of times hit by car", "_car_hit_car.png", plot_train=plot_trains)
        plot_separately(plot_iterations_car, people_heatmap_train_car, plot_iterations_test_car,
                        people_heatmap_car,
                        "Car Average sum from people heatmap",
                        "_avg_heatmap_car.png", plot_train=plot_trains)

        plot_separately(plot_iterations_car, dist_left_train_car, plot_iterations_test_car,
                        dist_left_car,
                        "Car Average min distance left",
                        "_dist_left_car.png", plot_train=plot_trains)
        plot_separately(plot_iterations_car, successes_train_car, plot_iterations_test_car,
                        successes_car,
                        "Car Number of successes",
                        "_success_car.png", plot_train=plot_trains)
        plot_separately(plot_iterations_car, norm_dist_travelled_train_car, plot_iterations_test_car,
                        norm_dist_travelled_car,
                        "Car Normalized distance travelled",
                        "_norm_dist_car.png", plot_train=plot_trains)
        # plot_separately(plot_iterations_car, nbr_direction_switches_train_car, plot_iterations_test_car,
        #                 nbr_direction_switches_car,
        #                 "Car Number of direction switches",
        #                 "_dir_switches_car.png", plot_train=plot_trains)

        plot_separately(plot_iterations_car, loss_train_car, plot_iterations_test_car,
                        loss_test_car,
                        "Car loss",
                        "_loss_car.png", plot_train=plot_trains)
        plot_separately(plot_iterations_car, entropy_train_car, plot_iterations_test_car, entropy_car,
                        "Car entropy",
                        "_entropy_car.png", plot_train=plot_trains)

        plot_separately(plot_iterations_car, prob_of_max_action_car, plot_iterations_test_car,
                        prob_of_max_action_test_car,
                        "Car highest probability of an action",
                        "_max_prob_car.png", plot_train=plot_trains)

        plot_separately(plot_iterations_car, most_freq_action_car, plot_iterations_test_car,
                        most_freq_action_test_car,
                        "Car mode action",
                        "_action_car.png", plot_train=plot_trains)
        plot_separately(plot_iterations_car, speed_to_dist_correlation_car, plot_iterations_test_car,
                        speed_to_dist_correlation_car_test,
                        "Correlation between distance to pedestrian and car's speed",
                        "_dist_to_pedestrian_correlation_to_car_speed.png", plot_train=plot_trains)

        plot_separately(plot_iterations_car, collision_between_car_and_agent, plot_iterations_test_car,
                        collision_between_car_and_agent_test,
                        "Collisions_with_agent",
                        "_collision_car_with_agent.png", plot_train=plot_trains)
    if continous:
        plot_separately(plot_iterations, entropy_train, plot_iterations_test, entropy,
                        "Agent mean step in x direction",
                        "_mean_x.png", plot_train=plot_trains)

        plot_separately(plot_iterations, prob_of_max_action, plot_iterations_test,
                        prob_of_max_action_test,
                        "Mean of agent output y",
                        "_mean_y.png", plot_train=plot_trains)

        plot_separately(plot_iterations, most_freq_action, plot_iterations_test,
                        most_freq_action_test,
                        "Variance of agent output y",
                        "_var_y.png", plot_train=plot_trains)
        plot_separately(plot_iterations, freq_most_freq_action, plot_iterations_test,
                        freq_most_freq_action_test,
                        "Variance of agent output x",
                        "_var_x.png", plot_train=plot_trains)
        plot_separately(plot_iterations, variance, plot_iterations_test,
                        variance_test,
                        "Variance",
                        "_var_model.png", plot_train=plot_trains)
    else:
        plot_separately(plot_iterations, entropy_train, plot_iterations_test, entropy, "Entropy",
                        "_entropy.png", plot_train=plot_trains)

        plot_separately(plot_iterations, prob_of_max_action, plot_iterations_test,
                        prob_of_max_action_test,
                        "Highest probability of an action",
                        "_max_prob.png", plot_train=plot_trains)

        plot_separately(plot_iterations, most_freq_action, plot_iterations_test,
                        most_freq_action_test,
                        "Mode action",
                        "_action.png", plot_train=plot_trains)
    plot_separately(plot_iterations, speed_mean, plot_iterations_test,
                    speed_mean_test, "Mean Speed",
                    "_speed_mean.png", plot_train=plot_trains)
    plot_separately(plot_iterations, speed_mean, plot_iterations_test,
                    speed_mean_test, "Standard deviation of average speeds exhibited in one batch",
                    "_std_speed_batch.png", plot_train=plot_trains, axis=1)
    plot_separately(plot_iterations, speed_var, plot_iterations_test,
                    speed_var_test,
                    "Average standard deviation of the speed exhibited during a trajectory, should be low",
                    "_std_speed_traj.png", plot_train=plot_trains)
    plot_separately(plot_iterations, dist_to_car, plot_iterations_test,
                    dist_to_car_test,
                    "Minimal distance to car",
                    "_dist_to_car.png", plot_train=plot_trains)
    plot_separately(plot_iterations, init_dist_to_car, plot_iterations_test,
                    init_dist_to_car_test,
                    "init distance to car",
                    "_init_to_car.png", plot_train=plot_trains)
    plot_separately(plot_iterations, init_dist_to_goal, plot_iterations_test,
                    init_dist_to_goal_test,
                    "initial distance to goal",
                    "_init_dist_to_goal.png", plot_train=plot_trains)
    plot_separately(plot_iterations, time_to_collision, plot_iterations_test,
                    time_to_collision_test,
                    "Time to collision",
                    "_time_to_collision.png", plot_train=plot_trains)
    print("Init maps test len " + str(len(init_maps_test)))
    if len(init_maps_test) >= 1 or len(init_map_stats_test) >= 1:
        if gaussian:
            # print ("Gaussian")
            # train_itr, goal_locations, test_itr, goal_locations_test,avg_goal_location_test, prior_goal_location_test,avg_goal_location, prior_goal_location, title, filename
            if len(goal_locations) > 0:
                plot_goal(plot_iterations, goal_locations, np.array(list(test_points.values())),
                          goal_locations_test, goal_position_mode_test,
                          goal_prior_mode_test, goal_position_mode, goal_prior_mode,
                          "Goal location distribution", "goal_location_distribution.png",
                          plot_train=False, axis=0, gaussian=True)
            plot_goal(plot_iterations, init_locations, np.array(list(test_points.values())),
                      init_locations_test, init_position_mode_test,
                      init_prior_mode_test, init_position_mode, init_prior_mode,
                      "Init location distribution", "init_location_distribution.png",
                      plot_train=False, axis=0)
            # plot_goal(plot_iterations, goal_locations, np.array(list(test_points.values())),
            #           init_locations_test, init_position_mode_test,
            #           init_prior_mode_test, goal_position_mode, goal_prior_mode,
            #           "Goal time location distribution", "goal_time_location_distribution.png",
            #           plot_train=False, axis=0)
        else:
            # print (" Not Gaussian")
            #
            # print (" goal locations len  " + str(len(goal_locations[7])) + " test len " + str(
            #     len(goal_locations_test[7])))
            # print (" goal mode len  " + str(len(goal_position_mode)) + " test len " + str(
            #     len(goal_position_mode_test)))
            # print (" goal prior mode len  " + str(len(goal_prior_mode)) + " test len " + str(
            #     len(goal_prior_mode_test)))
            plot_goal(plot_iterations, goal_locations, np.array(list(test_points.values())),
                      goal_locations_test, goal_position_mode_test,
                      goal_prior_mode_test, goal_position_mode, goal_prior_mode,
                      "Goal location distribution", "goal_location_distribution.png",
                      plot_train=False, axis=0, gaussian=False)
            # print (" init locations len  "+str(len(init_locations)) +" test len "+str(len(init_locations_test)))
            # print (" init prior mode len  " + str(len(init_prior_mode)) + " test len " + str(
            #     len(init_prior_mode_test)))
            # print (" init pos mode len  " + str(len(init_position_mode)) + " test len " + str(
            #     len(init_position_mode_test)))
            plot_goal(plot_iterations, init_locations, np.array(list(test_points.values())),
                      init_locations_test, init_position_mode_test,
                      init_prior_mode_test, init_position_mode, init_prior_mode,
                      "Init location distribution", "init_location_distribution.png",
                      plot_train=False, axis=0)
            # print ("Not Gaussian")
            # entropy_maps_test, entropy_prior_maps_test, kl_to_prior_maps_test, diff_to_prior_maps_test, cars_vel_test, cars_pos_test
            plot_loss(iterations_init, entropy_maps, iterations_test_init, entropy_maps_test,
                      "Entropy of initialization distribution", "_init_entropy.png", plot_train=False)
            # plot_loss(iterations_init, entropy_prior_maps, iterations_test_init, entropy_prior_maps_test,
            #           "Entropy of initialization distribution", "_init_entropy.png", plot_train=False)
            plot_loss(iterations_init, entropy_prior_maps, iterations_test_init,
                      entropy_prior_maps_test,
                      "Entropy of prior distribution", "_init_prior_entropy.png", plot_train=False)
            plot_loss(iterations_init, diff_to_prior_maps, iterations_test_init,
                      diff_to_prior_maps_test,
                      "Difference to prior", "diff_yp_prior.png", plot_train=False)
            # Make scatter plot between car speed and number of collisions

            # entropy_maps_test, entropy_prior_maps_test, kl_to_prior_maps_test, diff_to_prior_maps_test, cars_vel_test, cars_pos_test
            plot_loss(iterations_init, entropy_maps_goal, iterations_test_init, entropy_maps_test_goal,
                      "Entropy of initialization distribution", "_goal_entropy.png", plot_train=False)
            # plot_loss(iterations_init, entropy_prior_maps, iterations_test_init, entropy_prior_maps_test,
            #           "Entropy of initialization distribution", "_init_entropy.png", plot_train=False)
            plot_loss(iterations_init, entropy_prior_maps_goal, iterations_test_init,
                      entropy_prior_maps_test_goal,
                      "Entropy of prior distribution", "_goal_prior_entropy.png", plot_train=False)
            plot_loss(iterations_init, diff_to_prior_maps_goal, iterations_test_init,
                      diff_to_prior_maps_test_goal,
                      "Difference to prior", "diff_yp_prior_goal.png", plot_train=False)
            # Make scatter plot between car speed and number of collisions

        #print (len(plot_iterations)+" goal times "+str(len(goal_times))+" test: "+len(plot_iterations_test)+" goal times test"+str(len(goal_times_test)))
        plot_separately(plot_iterations, goal_times, plot_iterations_test,
                        goal_times_test,
                        "Goal times",
                        "_goal_times.png", plot_train=plot_trains)
        plot_separately(plot_iterations, goal_speed, plot_iterations_test,
                        goal_speed_test,
                        "Average speed",
                        "_goal_speed.png", plot_train=plot_trains)


read_files_and_plot(make_movie)

# #
#
# #
# # if __name__ == "__main__":
# #     main()

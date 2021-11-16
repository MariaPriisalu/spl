import numpy as np
import copy, sys, os

from settings import AGENT_MAX_HEIGHT_VOXELS, NUM_SEM_CLASSES, NUM_SEM_CLASSES_ASINT, CHANNELS

from commonUtils.ReconstructionUtils import isLabelIgnoredInReconstruction


def objects_in_range(objects, pos_x, pos_y, depth, width, to_int=False, carla=True):
    # Finds object in range!
    # reorder objects to match tensor.
    objs_sample = []

    all_objs=[]
    all_objs_list=[]
    print (" Get people in frame ")
    for indx, objs_frame in enumerate(objects):
        objs_sample.append([])
        for obj in objs_frame:
            if len(obj) == 6 or len(obj) == 7 :
                x_min = min(obj[0],obj[1])
                x_max = max(obj[0],obj[1])
                y_min = min(obj[2],obj[3])
                y_max = max(obj[2],obj[3])
            else:
                x_min = min(obj[0, :])
                x_max = max(obj[0, :])
                y_min = min(obj[1, :])
                y_max = max(obj[1, :])
                # if indx == 0:
                #     print(" Pedestrian: "+str(x_min)+" " +str(x_max)+" in range "+str(pos_x)+" "+str(pos_x + depth) )
                #     print(" Pedestrian: " + str(y_min) + " " + str(y_max) + " in range " + str(pos_y) + " " + str(
                #         pos_y + width))
            if ((pos_x <= x_min and x_min <pos_x + depth) or (pos_x <= x_max and x_max < pos_x + depth)) \
                    and ((pos_y <= y_min and  y_min <pos_y + width ) or ( pos_y <= y_max and y_max < pos_y + width)):
                if len(obj) == 6 :
                    objs_sample[indx].append(
                        (obj[4], obj[5],  obj[2] - pos_y, obj[3] - pos_y,obj[0] - pos_x, obj[1] - pos_x))
                elif len(obj) == 7:
                    objs_sample[indx].append(
                        (obj[4], obj[5], obj[2] - pos_y, obj[3] - pos_y, obj[0] - pos_x, obj[1] - pos_x, obj[6]))
                else:
                    # if indx==0:
                    #     print(" Append pedestrian")
                    objs_sample[indx].append(
                        obj[[2, 1, 0], :] - np.tile(np.array([0,pos_y, pos_x]).reshape((3, 1)), (1, obj.shape[1])))
                if to_int:
                    for i,index in enumerate(objs_sample[indx]):
                        objs_sample[indx][i]=int(index)
        all_objs=all_objs+objs_sample[indx]


    if not carla:
        for indx, objs_frame in enumerate(objects):
            all_objs_list.append(copy.deepcopy(all_objs))
        return all_objs_list
            # else:
            #     print "New Object"
            #     if  x_max < pos_x:
            #         print "Limit x too small: "+str(x_max) +" minimum allowed: "+str(pos_x)
            #     if  y_max < pos_y:
            #         print "Limit y too small: "+str(y_max)+" minimum allowed: "+str(pos_y)
            #     if  x_min > pos_x + depth:
            #         print "Limit x too large: "+str(x_min)+" maximum allowed: "+str(pos_x+depth)
            #     if  y_min > pos_y + width:
            #         print "Limit y too large: "+str(y_min)+" maximum allowed: "+str(pos_y+width)
    return objs_sample

# Assumes cityscapes coordinate system
def objects_in_range_map(objs_dict, pos_x, pos_y, depth, width, to_int=False, init_frames= {}):
    # Finds object in range!
    # reorder objects to match tensor.
    objs_in_range_dictionary = {}

    # print ("Objects dictionary ")
    # print (objs_dict)

    for key, objs_frame in list(objs_dict.items()):

        for i, obj in enumerate(objs_frame):
            if len(obj) == 6 or len(obj) == 7 :
                x_min = min(obj[0],obj[1])
                x_max = max(obj[0],obj[1])
                y_min = min(obj[2],obj[3])
                y_max = max(obj[2],obj[3])
            else:
                x_min = min(obj[0, :])
                x_max = max(obj[0, :])
                y_min = min(obj[1, :])
                y_max = max(obj[1, :])


            if ((pos_x <= x_min and x_min <pos_x + depth) or (pos_x <= x_max and x_max < pos_x + depth)) \
                    and ((pos_y <= y_min and  y_min <pos_y + width ) or ( pos_y <= y_max and y_max < pos_y + width)):

                if key not in objs_in_range_dictionary:
                    objs_in_range_dictionary[key] = []
                    if key in init_frames:
                        init_frames[key]=init_frames[key]+i

                if len(obj) == 6 :
                    objs_in_range_dictionary[key].append(
                        (obj[4], obj[5],  obj[2] - pos_y, obj[3] - pos_y,obj[0] - pos_x, obj[1] - pos_x))
                elif len(obj) == 7:
                    objs_in_range_dictionary[key].append(
                        (obj[4], obj[5], obj[2] - pos_y, obj[3] - pos_y, obj[0] - pos_x, obj[1] - pos_x, obj[6]))
                else:
                    objs_in_range_dictionary[key].append(
                        obj[[2, 1, 0], :] - np.tile(np.array([0,pos_y, pos_x]).reshape((3, 1)), (1, obj.shape[1])))

                # if to_int:
                #     for i,index in enumerate(objs_sample[key]):
                #         objs_sample[key][i]=int(index)
            # else:
            #
            #     if  x_max < pos_x:
            #         print "Limit x too small: "+str(x_max) +" minimum allowed: "+str(pos_x)
            #     if  y_max < pos_y:
            #         print "Limit y too small: "+str(y_max)+" minimum allowed: "+str(pos_y)
            #     if  x_min > pos_x + depth:
            #         print "Limit x too large: "+str(x_min)+" maximum allowed: "+str(pos_x+depth)
            #     if  y_min > pos_y + width:
            #         print "Limit y too large: "+str(y_min)+" maximum allowed: "+str(pos_y+width)

    return objs_in_range_dictionary, init_frames



# Takes coordinates in cityscapes coordinate system (in voxels). Returns in tensor [z, y, x]
def extract_tensor(pos_x, pos_y, reconstruction, height, width, depth):
    height_b = height
    begin_z = 0
    tensor = np.zeros((height, width, depth, 6))
    density = 0
    max_x=0
    max_y=0
    max_z=0
    for (x, y, z) in reconstruction:
        max_z=max(int(z - begin_z), max_z)
        max_y =max(int(y - pos_y), max_y)
        max_x=max(int(x - pos_x), max_x)
    # print (" Maximal dimension "+str(max_x)+" max y "+str(max_y)+" max z "+str(max_z))
    # print(" Maximal allowed dimension " + str(depth) + " max y " + str(width) + " max z " + str(height_b))
    #x,y,z in cityscapes
    for z in range(begin_z, height_b + begin_z):# x
        for y in range(pos_y, pos_y + width):#y
            for x in range(pos_x, pos_x + depth):  # z
                if (x,y,z) in reconstruction:
                    tensor[int(z - begin_z), int(y - pos_y),int(x - pos_x) , CHANNELS.rgb[0]:CHANNELS.rgb[1]] = [
                        reconstruction[(x,y,z)][CHANNELS.rgb[0]] * (1 / 255.0),
                        reconstruction[(x,y,z)][CHANNELS.rgb[0]+1] * (1 / 255.0),
                        reconstruction[(x,y,z)][CHANNELS.rgb[0]+2] * (1 / 255.0)]

                    labelInRec = reconstruction[(x,y,z)][3]
                    if not isLabelIgnoredInReconstruction(labelInRec, isCarlaLabel=False): # The input data is in cityscapes !
                        tensor[int(z - begin_z), int(y - pos_y),int(x - pos_x),  CHANNELS.semantic] = \
                            reconstruction[(x,y,z)][CHANNELS.semantic] * (1 / NUM_SEM_CLASSES)
                    density += 1

    return tensor, density
def return_highest_object( tens):
    z = 0
    while z < tens.shape[0] - 1 and np.linalg.norm(tens[z]) == 0:
        z = z + 1
    return z

def return_highest_col(tens):
    z = 0
    while z < AGENT_MAX_HEIGHT_VOXELS and  z < tens.shape[0] - 1 and np.linalg.norm(tens[z, :]) == 0:
        z = z + 1
    return z
#





# This updates the tensor created previously with the function above to add 0.1 density to each voxel occupied by an agent or car
# reconstruction_2D is a 2D projection of the 3D space given as input in tensor
def frame_reconstruction(tensor, cars_a, people_a, no_dict=True, temporal=False, predict_future=False, run_2D=False, reconstruction_2D=[], number_of_frames_eval=-1):
    reconstruction, reconstruction_2D = reconstruct_static_objects(tensor, run_2D)


    if number_of_frames_eval>=0:
        number_of_frames=number_of_frames_eval
    else:
        number_of_frames = len(cars_a)


    #print(("Frame reconstruction "+str(temporal)))
    cars_predicted=[]
    people_predicted=[]
    for frame in range(number_of_frames):
        cars_map = np.zeros(tensor.shape[:3])#.shape[1:3])
        if run_2D:
            cars_map = np.zeros(tensor.shape[1:3])

        for car in cars_a[frame]:
            car_a = []

            car_middle_exact = np.array([np.mean(car[0:1]), np.mean(car[2:3]), np.mean(car[4:5])])
            for count in range(3):
                p = car[count*2]
                car_a.append(max(p, 0))
                car_a.append(min(car[count*2+1]+1, reconstruction.shape[count]))
            car_middle=np.array([np.mean(car_a[0:1]),np.mean(car_a[2:3]), np.mean(car_a[4:5])])
            if predict_future:
                closest_car=[]
                cc_voxel=[0,0,0]
                vel=[0,0,0]
                min_dist=50
                if frame>0 and no_dict:
                    for car_2 in cars_a[frame-1]:
                        car_middle2 = np.array([np.mean(car_2[0:1]), np.mean(car_2[2:3]), np.mean(car_2[4:5])])
                        dist=np.linalg.norm(car_middle_exact[1:]-car_middle2[1:])
                        if dist< min_dist:
                            closest_car = car_middle2
                            cc_voxel=np.array(car_2)
                            vel=car_middle_exact-car_middle2
                            min_dist=dist.copy()

                    vel[0]=0

                    if np.linalg.norm(vel)>0:

                        loc_frame=1
                        while np.all(closest_car[1:]>1) and np.all(reconstruction.shape[1:3]-closest_car[1:]>1):
                            cc_voxel=cc_voxel+np.array([0,0, vel[1], vel[1], vel[2], vel[2]])
                            if run_2D:
                                cars_map[
                                max(cc_voxel[2].astype(int), 0):min(cc_voxel[3].astype(int), reconstruction.shape[1]),
                                max(cc_voxel[4].astype(int), 0):min(cc_voxel[5].astype(int),
                                                                    reconstruction.shape[2])] = cars_map[
                                                                                                max(cc_voxel[2].astype(
                                                                                                    int), 0):min(
                                                                                                    cc_voxel[3].astype(
                                                                                                        int),
                                                                                                    reconstruction.shape[
                                                                                                        1]),
                                                                                                max(cc_voxel[4].astype(
                                                                                                    int), 0):min(
                                                                                                    cc_voxel[5].astype(
                                                                                                        int),
                                                                                                    reconstruction.shape[
                                                                                                        2])] + 0.1
                            else:
                                cars_map[max(cc_voxel[0].astype(int),0):min(cc_voxel[1].astype(int),reconstruction.shape[0]),
                                max(cc_voxel[2].astype(int),0):min(cc_voxel[3].astype(int),reconstruction.shape[1]),
                                max(cc_voxel[4].astype(int),0):min(cc_voxel[5].astype(int),reconstruction.shape[2]) ] = cars_map[max(cc_voxel[0].astype(int),0):min(cc_voxel[1].astype(int),reconstruction.shape[0]),
                                max(cc_voxel[2].astype(int),0):min(cc_voxel[3].astype(int),reconstruction.shape[1]),
                                max(cc_voxel[4].astype(int),0):min(cc_voxel[5].astype(int),reconstruction.shape[2]) ]+0.1
                            if temporal:
                                if run_2D:
                                    cars_map[max(cc_voxel[2].astype(int), 0):min(cc_voxel[3].astype(int), reconstruction.shape[1]),
                                    max(cc_voxel[4].astype(int), 0):min(cc_voxel[5].astype(int),
                                                                        reconstruction.shape[2])] = loc_frame*np.ones_like(cars_map[
                                                                                                                 max(cc_voxel[
                                                                                                                         2].astype(
                                                                                                                     int),
                                                                                                                     0):min(
                                                                                                                     cc_voxel[
                                                                                                                         3].astype(
                                                                                                                         int),
                                                                                                                     reconstruction.shape[
                                                                                                                         1]),
                                                                                                                 max(cc_voxel[
                                                                                                                         4].astype(
                                                                                                                     int),
                                                                                                                     0):min(
                                                                                                                     cc_voxel[
                                                                                                                         5].astype(
                                                                                                                         int),
                                                                                                                     reconstruction.shape[
                                                                                                                         2])])
                                else:
                                    cars_map[
                                    max(cc_voxel[0].astype(int), 0):min(cc_voxel[1].astype(int), reconstruction.shape[0]),
                                    max(cc_voxel[2].astype(int), 0):min(cc_voxel[3].astype(int), reconstruction.shape[1]),
                                    max(cc_voxel[4].astype(int), 0):min(cc_voxel[5].astype(int),
                                                                        reconstruction.shape[
                                                                            2])] = loc_frame * np.ones_like(cars_map[max(
                                        cc_voxel[0].astype(int), 0):min(cc_voxel[1].astype(int), reconstruction.shape[0]),
                                                                                                            max(cc_voxel[
                                                                                                                2].astype(
                                                                                                                int),
                                                                                                                0):min(
                                                                                                                cc_voxel[
                                                                                                                    3].astype(
                                                                                                                    int),
                                                                                                                reconstruction.shape[
                                                                                                                    1]),
                                                                                                            max(cc_voxel[
                                                                                                                4].astype(
                                                                                                                int),
                                                                                                                0):min(
                                                                                                                cc_voxel[
                                                                                                                    5].astype(
                                                                                                                    int),
                                                                                                                reconstruction.shape[
                                                                                                                    2])])

                            loc_frame=loc_frame+1
                            closest_car=closest_car+vel


            reconstruction[car_a[0]:car_a[1], car_a[2]:car_a[3], car_a[4]:car_a[5], CHANNELS.cars_trajectory] =reconstruction[car_a[0]:car_a[1], car_a[2]:car_a[3], car_a[4]:car_a[5], CHANNELS.cars_trajectory]+0.1
            if run_2D:
                reconstruction_2D[ car_a[2]:car_a[3], car_a[4]:car_a[5], CHANNELS.cars_trajectory] = reconstruction_2D[car_a[2]:car_a[3],car_a[4]:car_a[5], CHANNELS.cars_trajectory] + 0.1

            if temporal and predict_future:

                reconstruction[car_a[0]:car_a[1], car_a[2]:car_a[3], car_a[4]:car_a[5], CHANNELS.cars_trajectory] =frame* np.ones_like(reconstruction[
                                                                                             car_a[0]:car_a[1],
                                                                                             car_a[2]:car_a[3],
                                                                                             car_a[4]:car_a[5], CHANNELS.cars_trajectory] )
                if run_2D:
                    reconstruction_2D[ car_a[2]:car_a[3], car_a[4]:car_a[5], CHANNELS.cars_trajectory] = frame * np.ones_like(
                        reconstruction_2D[car_a[2]:car_a[3],car_a[4]:car_a[5], CHANNELS.cars_trajectory])
        #if no_dict:
        if run_2D:
            cars_predicted.append(reconstruction_2D[ :, :, CHANNELS.cars_trajectory].copy())
        else:
            cars_predicted.append(reconstruction[:,:,:,CHANNELS.cars_trajectory].copy())

        if predict_future:
            if no_dict:
                if temporal:
                    temp_array =cars_predicted[-1]>0
                    cars_predicted[-1] =cars_predicted[-1]-(frame* temp_array)
                    temp_2 = cars_map > 0
                    cars_predicted[-1][temp_2] = cars_map[temp_2].copy()

                else:
                    temp_2 = cars_map > 0
                    cars_predicted[-1][temp_2] =cars_predicted[-1][temp_2]+ cars_map[temp_2].copy()
            else:
                if temporal:
                    temp_array = cars_predicted[-1] > 0
                    cars_predicted[-1] = cars_predicted[-1] - (frame * temp_array)

        # print "Predicted "+ str(np.sum(cars_predicted[-1][:]))
    for frame in range(len(people_a)):
        if run_2D:
            people_map = np.zeros(tensor.shape[1:3])
        else:
            people_map = np.zeros(tensor.shape[:3])  # .shape[1:3])
        for person in people_a[frame]:
            x_pers=[]
            person_middle_exact=np.array([np.mean(person[0]), np.mean(person[1]), np.mean(person[2])])
            for count in range(3):
                p=person[count, :].copy()
                if len(p)>0:
                    x_pers.append((max(min(p),0), min(max(p)+1,reconstruction.shape[count])))
            if len(x_pers)==3:
                if run_2D:
                    reconstruction_2D[ int(x_pers[1][0]):int( x_pers[1][1]),
                    int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory] =reconstruction_2D[int(x_pers[1][0]):int( x_pers[1][1]),
                    int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory] +0.1

                reconstruction[int(x_pers[0][0]):int(x_pers[0][1]), int(x_pers[1][0]):int(x_pers[1][1]),
                int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory] = reconstruction[int(x_pers[0][0]):int(x_pers[0][1]),
                                                              int(x_pers[1][0]):int(x_pers[1][1]),
                                                              int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory] + 0.1
            if temporal and predict_future:
                if run_2D:
                    reconstruction_2D[ int(x_pers[1][0]):int(x_pers[1][1]),
                    int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory] = frame * np.ones_like(
                        reconstruction_2D[
                        int(x_pers[1][0]):int(x_pers[1][1]),
                        int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory])

                reconstruction[int(x_pers[0][0]):int(x_pers[0][1]), int(x_pers[1][0]):int(x_pers[1][1]),
                int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory] = frame*np.ones_like(reconstruction[int(x_pers[0][0]):int(x_pers[0][1]),
                                                              int(x_pers[1][0]):int(x_pers[1][1]),
                                                              int(x_pers[2][0]):int(x_pers[2][1]), CHANNELS.pedestrian_trajectory] )
            if predict_future:

                closest_person = []
                cc_voxel = [0, 0, 0]
                vel = [0, 0, 0]
                min_dist = 50
                if frame > 0 and no_dict:
                    for person_2 in people_a[frame - 1]:
                        person_middle2 = np.array([np.mean(person_2[0]), np.mean(person_2[1]), np.mean(person_2[2])])
                        dist = np.linalg.norm(person_middle_exact[1:] - person_middle2[1:])
                        if dist < min_dist:
                            closest_person = person_middle2
                            cc_voxel = np.array(person_2)
                            vel = person_middle_exact - person_middle2
                            min_dist = dist.copy()


                    vel[0] = 0

                    if np.linalg.norm(vel) > 0:
                        loc_frame=1
                        while np.all(closest_person[1:] > 1) and np.all(reconstruction.shape[1:3] - closest_person[1:] > 1):

                            cc_voxel = cc_voxel + np.tile(vel,[2,1]).T

                            if run_2D:
                                people_map[
                                max(cc_voxel[1][0].astype(int), 0):min(cc_voxel[1][1].astype(int),
                                                                       reconstruction.shape[1]),
                                max(cc_voxel[2][0].astype(int), 0):min(cc_voxel[2][1].astype(int),
                                                                       reconstruction.shape[2])] = people_map[max(cc_voxel[1][
                                                                                                           0].astype(
                                                                                                       int), 0):min(
                                                                                                       cc_voxel[1][
                                                                                                           1].astype(
                                                                                                           int),
                                                                                                       reconstruction.shape[
                                                                                                           1]),
                                                                                                   max(cc_voxel[2][
                                                                                                           0].astype(
                                                                                                       int), 0):min(
                                                                                                       cc_voxel[2][
                                                                                                           1].astype(
                                                                                                           int),
                                                                                                       reconstruction.shape[
                                                                                                           2])] + 0.1
                            else:

                                people_map[max(cc_voxel[0][0].astype(int), 0):min(cc_voxel[0][1].astype(int), reconstruction.shape[0]),
                                max(cc_voxel[1][0].astype(int), 0):min(cc_voxel[1][1].astype(int), reconstruction.shape[1]),
                                max(cc_voxel[2][0].astype(int), 0):min(cc_voxel[2][1].astype(int),
                                                                    reconstruction.shape[2])] =  people_map[max(cc_voxel[0][0].astype(int), 0):min(cc_voxel[0][1].astype(int), reconstruction.shape[0]),
                                max(cc_voxel[1][0].astype(int), 0):min(cc_voxel[1][1].astype(int), reconstruction.shape[1]),
                                max(cc_voxel[2][0].astype(int), 0):min(cc_voxel[2][1].astype(int),
                                                                    reconstruction.shape[2])] +0.1
                            if temporal:
                                if run_2D:
                                    people_map[
                                    max(cc_voxel[1][0].astype(int), 0):min(cc_voxel[1][1].astype(int), reconstruction.shape[1]),
                                    max(cc_voxel[2][0].astype(int), 0):min(cc_voxel[2][1].astype(int),
                                                                           reconstruction.shape[2])] = loc_frame*np.ones_like(people_map[
                                                                                                                    max(
                                                                                                                        cc_voxel[
                                                                                                                            1][
                                                                                                                            0].astype(
                                                                                                                            int),
                                                                                                                        0):min(
                                                                                                                        cc_voxel[
                                                                                                                            1][
                                                                                                                            1].astype(
                                                                                                                            int),
                                                                                                                        reconstruction.shape[
                                                                                                                            1]),
                                                                                                                    max(
                                                                                                                        cc_voxel[
                                                                                                                            2][
                                                                                                                            0].astype(
                                                                                                                            int),
                                                                                                                        0):min(
                                                                                                                        cc_voxel[
                                                                                                                            2][
                                                                                                                            1].astype(
                                                                                                                            int),
                                                                                                                        reconstruction.shape[
                                                                                                                            2])])
                                else:
                                    people_map[
                                    max(cc_voxel[0][0].astype(int), 0):min(cc_voxel[0][1].astype(int),
                                                                           reconstruction.shape[0]),
                                    max(cc_voxel[1][0].astype(int), 0):min(cc_voxel[1][1].astype(int),
                                                                           reconstruction.shape[1]),
                                    max(cc_voxel[2][0].astype(int), 0):min(cc_voxel[2][1].astype(int),
                                                                           reconstruction.shape[
                                                                               2])] = loc_frame * np.ones_like(
                                        people_map[
                                        max(
                                            cc_voxel[
                                                0][
                                                0].astype(
                                                int),
                                            0):min(
                                            cc_voxel[
                                                0][
                                                1].astype(
                                                int),
                                            reconstruction.shape[
                                                0]),
                                        max(
                                            cc_voxel[
                                                1][
                                                0].astype(
                                                int),
                                            0):min(
                                            cc_voxel[
                                                1][
                                                1].astype(
                                                int),
                                            reconstruction.shape[
                                                1]),
                                        max(
                                            cc_voxel[
                                                2][
                                                0].astype(
                                                int),
                                            0):min(
                                            cc_voxel[
                                                2][
                                                1].astype(
                                                int),
                                            reconstruction.shape[
                                                2])])

                            loc_frame=loc_frame+1
                            closest_person = closest_person + vel
        if run_2D:
            people_predicted.append(reconstruction_2D[ :, :, CHANNELS.pedestrian_trajectory].copy())
        else:
            people_predicted.append(reconstruction[:, :, :, CHANNELS.pedestrian_trajectory].copy())

        if no_dict:
            if temporal and predict_future:
                temp_array=people_predicted[-1]>0
                temp_2=people_map>0

                people_predicted[-1] = people_predicted[-1]-(frame*temp_array)
                people_predicted[-1][temp_2]=people_map[temp_2].copy()

            else:
                temp_2 = people_map > 0
                people_predicted[-1][temp_2] =people_predicted[-1][temp_2]+ people_map[temp_2].copy()
        else:
            if temporal:
                temp_array = people_predicted[-1] > 0
                people_predicted[-1] = people_predicted[-1] - (frame * temp_array)

    return reconstruction, cars_predicted,people_predicted, reconstruction_2D


def reconstruct_static_objects( tensor, run_2D):
    reconstruction = tensor.copy()
    if run_2D:
        reconstruction_2D = np.zeros(reconstruction.shape[1:])
        for x in range(reconstruction_2D.shape[0]):
            for y in range(reconstruction_2D.shape[1]):
                z = return_highest_col(tensor[:, x, y, :3])
                reconstruction_2D[x, y, CHANNELS.rgb[0]] = tensor[z, x, y, CHANNELS.rgb[0]]
                reconstruction_2D[x, y, CHANNELS.rgb[0]+1] = tensor[z, x, y, CHANNELS.rgb[0]+1]
                reconstruction_2D[x, y, CHANNELS.rgb[0]+2] = tensor[z, x, y, CHANNELS.rgb[0]+2]
                z1 = return_highest_object(tensor[:, x, y, CHANNELS.semantic])
                reconstruction_2D[x, y, CHANNELS.semantic] = tensor[z1, x, y, CHANNELS.semantic]
    else:
        reconstruction_2D=[]
    return reconstruction, reconstruction_2D
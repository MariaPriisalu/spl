#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""

from __future__ import print_function

import argparse
import logging
import random
import time
import copy

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from carla.image_converter import depth_to_local_point_cloud, to_rgb_array,labels_to_array
from carla.transform import Transform

from collections import Counter

import numpy as np
import pickle, os
from PIL import Image


episode_nbr=0
def run_carla_client( args):
    # Here we will run 3 episodes with 300 frames each.
    number_of_episodes = 150
    frames_per_episode = 1500

    # for start_i in range(150):
    #     if start_i%4==0:
    #         output_folder = 'Packages/CARLA_0.8.2/PythonClient/new_data-viz/test_' + str(start_i)
    #         if not os.path.exists(output_folder):
    #             os.mkdir(output_folder)
    #             print( "make "+str(output_folder))
    # ./CarlaUE4.sh -carla-server  -benchmark -fps=17 -windowed
    # carla-server "/usr/local/carla/Unreal/CarlaUE4/CarlaUE4.uproject" /usr/local/carla/Maps/Town03 -benchmark -fps=10 -windowed


    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:

        print('CarlaClient connected')
        global episode_nbr
        print (episode_nbr)
        for episode in range(0,150):
            if episode % 4 == 0:
                output_folder = 'Packages/CARLA_0.8.2/PythonClient/new_data-viz/test_' + str(episode)
                if not os.path.exists(output_folder+"/cameras.p"):
                    # Start a new episode.
                    episode_nbr=episode
                    frame_step = 1  # Save one image every 100 frames
                    pointcloud_step=50
                    image_size = [800, 600]
                    camera_local_pos = [0.3, 0.0, 1.3]  # [X, Y, Z]
                    camera_local_rotation = [0, 0, 0]  # [pitch(Y), yaw(Z), roll(X)]
                    fov = 70
                    # Create a CarlaSettings object. This object is a wrapper around
                    # the CarlaSettings.ini file. Here we set the configuration we
                    # want for the new episode.
                    settings = CarlaSettings()
                    settings.set(
                        SynchronousMode=True,
                        SendNonPlayerAgentsInfo=True,
                        NumberOfVehicles=50,
                        NumberOfPedestrians=200,
                        WeatherId=random.choice([1, 3, 7, 8, 14]),
                        QualityLevel=args.quality_level)
                    settings.randomize_seeds()

                    # Now we want to add a couple of cameras to the player vehicle.
                    # We will collect the images produced by these cameras every
                    # frame.


                    camera1 = Camera('CameraDepth', PostProcessing='Depth', FOV=fov)
                    camera1.set_image_size(*image_size)
                    camera1.set_position(*camera_local_pos)
                    camera1.set_rotation(*camera_local_rotation)
                    settings.add_sensor(camera1)

                    camera2 = Camera('CameraRGB', PostProcessing='SceneFinal', FOV=fov)
                    camera2.set_image_size(*image_size)
                    camera2.set_position(*camera_local_pos)
                    camera2.set_rotation(*camera_local_rotation)
                    settings.add_sensor(camera2)

                    camera3 = Camera('CameraSeg', PostProcessing='SemanticSegmentation', FOV=fov)
                    camera3.set_image_size(*image_size)
                    camera3.set_position(*camera_local_pos)
                    camera3.set_rotation(*camera_local_rotation)
                    settings.add_sensor(camera3)



                    # Now we load these settings into the server. The server replies
                    # with a scene description containing the available start spots for
                    # the player. Here we can provide a CarlaSettings object or a
                    # CarlaSettings.ini file as string.
                    scene = client.load_settings(settings)

                    # Choose one player start at random.
                    number_of_player_starts = len(scene.player_start_spots)
                    player_start = episode#random.randint(0, max(0, number_of_player_starts - 1))

                    output_folder = 'Packages/CARLA_0.8.2/PythonClient/new_data-viz/test_' + str(episode)
                    # Notify the server that we want to start the episode at the
                    # player_start index. This function blocks until the server is ready
                    # to start the episode.
                    print('Starting new episode...')
                    client.start_episode(player_start)


                    cameras_dict = {}
                    pedestrians_dict = {}
                    cars_dict = {}
                    # Compute the camera transform matrix
                    camera_to_car_transform = camera2.get_unreal_transform()
                    # (Intrinsic) (3, 3) K Matrix
                    K = np.identity(3)
                    K[0, 2] = image_size[0] / 2.0
                    K[1, 2] = image_size[1] / 2.0
                    K[0, 0] = K[1, 1] = image_size[0] / (2.0 * np.tan(fov * np.pi / 360.0))
                    with open(output_folder + '/camera_intrinsics.p', 'w') as camfile:
                        pickle.dump(K, camfile)


                    # Iterate every frame in the episode.
                    for frame in range(0, frames_per_episode):

                        # Read the data produced by the server this frame.
                        measurements, sensor_data = client.read_data()

                        # Print some of the measurements.
                        #print_measurements(measurements)
                        if not frame % frame_step:
                            # Save the images to disk if requested.

                            for name, measurement in sensor_data.items():
                                filename = args.out_filename_format.format(episode, name, frame)
                                print (filename)
                                measurement.save_to_disk(filename)

                            # We can access the encoded data of a given image as numpy
                            # array using its "data" property. For instance, to get the
                            # depth value (normalized) at pixel X, Y
                            #
                            #     depth_array = sensor_data['CameraDepth'].data
                            #     value_at_pixel = depth_array[Y, X]
                            #

                            # Now we have to send the instructions to control the vehicle.
                            # If we are in synchronous mode the server will pause the
                            # simulation until we send this control.

                            # RGB image [[[r,g,b],..[r,g,b]],..[[r,g,b],..[r,g,b]]]
                            image_RGB = to_rgb_array(sensor_data['CameraRGB'])

                            labels=labels_to_array(sensor_data['CameraSeg'])[:,:,np.newaxis]

                            image_seg = np.tile(labels, (1, 1, 3))
                            depth_array = sensor_data['CameraDepth'].data*1000


                            # 2d to (camera) local 3d
                            # We use the image_RGB to colorize each 3D point, this is optional.
                            # "max_depth" is used to keep only the points that are near to the
                            # camera, meaning 1.0 the farest points (sky)
                            if not frame % pointcloud_step:
                                point_cloud = depth_to_local_point_cloud(
                                    sensor_data['CameraDepth'],
                                    image_RGB,
                                    max_depth=args.far
                                )

                                point_cloud_seg = depth_to_local_point_cloud(
                                    sensor_data['CameraDepth'],
                                    segmentation=image_seg,
                                    max_depth=args.far
                                )

                            # (Camera) local 3d to world 3d.
                            # Get the transform from the player protobuf transformation.
                            world_transform = Transform(
                                measurements.player_measurements.transform
                            )

                            # Compute the final transformation matrix.
                            car_to_world_transform = world_transform * camera_to_car_transform

                            # Car to World transformation given the 3D points and the
                            # transformation matrix.
                            point_cloud.apply_transform(car_to_world_transform)
                            point_cloud_seg.apply_transform(car_to_world_transform)

                            Rt = car_to_world_transform.matrix
                            Rt_inv = car_to_world_transform.inverse().matrix
                            # R_inv=world_transform.inverse().matrix
                            cameras_dict[frame] = {}
                            cameras_dict[frame]['inverse_rotation'] = Rt_inv[:]
                            cameras_dict[frame]['rotation'] = Rt[:]
                            cameras_dict[frame]['translation'] = Rt_inv[0:3, 3]
                            cameras_dict[frame]['camera_to_car'] = camera_to_car_transform.matrix

                            # Get non-player info
                            vehicles = {}
                            pedestrians = {}
                            for agent in measurements.non_player_agents:
                                # check if the agent is a vehicle.
                                if agent.HasField('vehicle'):
                                    pos = agent.vehicle.transform.location
                                    pos_vector = np.array([[pos.x], [pos.y], [pos.z], [1.0]])

                                    trnasformed_3d_pos = np.dot(Rt_inv, pos_vector)
                                    pos2d = np.dot(K, trnasformed_3d_pos[:3])

                                    # Normalize the point
                                    norm_pos2d = np.array([
                                        pos2d[0] / pos2d[2],
                                        pos2d[1] / pos2d[2],
                                        pos2d[2]])

                                    # Now, pos2d contains the [x, y, d] values of the image in pixels (where d is the depth)
                                    # You can use the depth to know the points that are in front of the camera (positive depth).

                                    x_2d = image_size[0] - norm_pos2d[0]
                                    y_2d = image_size[1] - norm_pos2d[1]
                                    vehicles[agent.id] = {}
                                    vehicles[agent.id]['transform'] = [agent.vehicle.transform.location.x,
                                                                       agent.vehicle.transform.location.y,
                                                                       agent.vehicle.transform.location.z]
                                    vehicles[agent.id][
                                        'bounding_box.transform'] = agent.vehicle.bounding_box.transform.location.z

                                    vehicles[agent.id]['yaw'] = agent.vehicle.transform.rotation.yaw
                                    vehicles[agent.id]['bounding_box'] = [agent.vehicle.bounding_box.extent.x,
                                                                          agent.vehicle.bounding_box.extent.y,
                                                                          agent.vehicle.bounding_box.extent.z]
                                    vehicle_transform = Transform(agent.vehicle.bounding_box.transform)
                                    pos = agent.vehicle.transform.location

                                    bbox3d = agent.vehicle.bounding_box.extent

                                    # Compute the 3D bounding boxes
                                    # f contains the 4 points that corresponds to the bottom
                                    f = np.array([[pos.x + bbox3d.x, pos.y - bbox3d.y,
                                                   pos.z - bbox3d.z + agent.vehicle.bounding_box.transform.location.z],
                                                  [pos.x + bbox3d.x, pos.y + bbox3d.y,
                                                   pos.z - bbox3d.z + agent.vehicle.bounding_box.transform.location.z],
                                                  [pos.x - bbox3d.x, pos.y + bbox3d.y,
                                                   pos.z - bbox3d.z + agent.vehicle.bounding_box.transform.location.z],
                                                  [pos.x - bbox3d.x, pos.y - bbox3d.y,
                                                   pos.z - bbox3d.z + agent.vehicle.bounding_box.transform.location.z],
                                                  [pos.x + bbox3d.x, pos.y - bbox3d.y,
                                                   pos.z + bbox3d.z + agent.vehicle.bounding_box.transform.location.z],
                                                  [pos.x + bbox3d.x, pos.y + bbox3d.y,
                                                   pos.z + bbox3d.z + agent.vehicle.bounding_box.transform.location.z],
                                                  [pos.x - bbox3d.x, pos.y + bbox3d.y,
                                                   pos.z + bbox3d.z + agent.vehicle.bounding_box.transform.location.z],
                                                  [pos.x - bbox3d.x, pos.y - bbox3d.y,
                                                   pos.z + bbox3d.z + agent.vehicle.bounding_box.transform.location.z]])

                                    f_rotated = vehicle_transform.transform_points(f)
                                    f_2D_rotated = []
                                    vehicles[agent.id]['bounding_box_coord'] = f_rotated

                                    for i in range(f.shape[0]):
                                        point = np.array([[f_rotated[i, 0]], [f_rotated[i, 1]], [f_rotated[i, 2]], [1]])
                                        transformed_2d_pos = np.dot(Rt_inv, point)
                                        pos2d = np.dot(K, transformed_2d_pos[:3])
                                        norm_pos2d = np.array([
                                            pos2d[0] / pos2d[2],
                                            pos2d[1] / pos2d[2],
                                            pos2d[2]])
                                        # print([image_size[0] - (pos2d[0] / pos2d[2]), image_size[1] - (pos2d[1] / pos2d[2])])
                                        f_2D_rotated.append(
                                            np.array([image_size[0] - norm_pos2d[0], image_size[1] - norm_pos2d[1]]))
                                    vehicles[agent.id]['bounding_box_2D'] = f_2D_rotated


                                elif agent.HasField('pedestrian'):
                                    pedestrians[agent.id] = {}
                                    pedestrians[agent.id]['transform'] = [agent.pedestrian.transform.location.x,
                                                                          agent.pedestrian.transform.location.y,
                                                                          agent.pedestrian.transform.location.z]
                                    pedestrians[agent.id]['yaw'] = agent.pedestrian.transform.rotation.yaw
                                    pedestrians[agent.id]['bounding_box'] = [agent.pedestrian.bounding_box.extent.x,
                                                                             agent.pedestrian.bounding_box.extent.y,
                                                                             agent.pedestrian.bounding_box.extent.z]
                                    # get the needed transformations
                                    # remember to explicitly make it Transform() so you can use transform_points()
                                    pedestrian_transform = Transform(agent.pedestrian.transform)
                                    bbox_transform = Transform(agent.pedestrian.bounding_box.transform)

                                    # get the box extent
                                    ext = agent.pedestrian.bounding_box.extent
                                    # 8 bounding box vertices relative to (0,0,0)
                                    bbox = np.array([
                                        [  ext.x,   ext.y,   ext.z],
                                        [- ext.x,   ext.y,   ext.z],
                                        [  ext.x, - ext.y,   ext.z],
                                        [- ext.x, - ext.y,   ext.z],
                                        [  ext.x,   ext.y, - ext.z],
                                        [- ext.x,   ext.y, - ext.z],
                                        [  ext.x, - ext.y, - ext.z],
                                        [- ext.x, - ext.y, - ext.z]
                                    ])

                                    # transform the vertices respect to the bounding box transform
                                    bbox = bbox_transform.transform_points(bbox)

                                    # the bounding box transform is respect to the pedestrian transform
                                    # so let's transform the points relative to it's transform
                                    bbox = pedestrian_transform.transform_points(bbox)

                                    # pedestrian's transform is relative to the world, so now,
                                    # bbox contains the 3D bounding box vertices relative to the world
                                    pedestrians[agent.id]['bounding_box_coord'] =copy.deepcopy(bbox)

                                    # Additionally, you can print these vertices to check that is working
                                    f_2D_rotated=[]
                                    ys=[]
                                    xs=[]
                                    zs=[]
                                    for vertex in bbox:
                                        pos_vector = np.array([
                                            [vertex[0,0]],  # [[X,
                                            [vertex[0,1]],  #   Y,
                                            [vertex[0,2]],  #   Z,
                                            [1.0]           #   1.0]]
                                        ])

                                        # transform the points to camera
                                        transformed_3d_pos =np.dot(Rt_inv, pos_vector)# np.dot(inv(self._extrinsic.matrix), pos_vector)
                                        zs.append(transformed_3d_pos[2])
                                        # transform the points to 2D
                                        pos2d =np.dot(K, transformed_3d_pos[:3]) #np.dot(self._intrinsic, transformed_3d_pos[:3])

                                        # normalize the 2D points
                                        pos2d = np.array([
                                            pos2d[0] / pos2d[2],
                                            pos2d[1] / pos2d[2],
                                            pos2d[2]
                                        ])

                                        # print the points in the screen
                                        if pos2d[2] > 0: # if the point is in front of the camera
                                            x_2d = image_size[0]-pos2d[0]#WINDOW_WIDTH - pos2d[0]
                                            y_2d = image_size[1]-pos2d[1]#WINDOW_HEIGHT - pos2d[1]
                                            ys.append(y_2d)
                                            xs.append(x_2d)
                                            f_2D_rotated.append( (y_2d, x_2d))
                                    if len(xs)>1:
                                        bbox=[[int(min(xs)), int(max(xs))],[int(min(ys)), int(max(ys))]]
                                        clipped_seg=labels[bbox[1][0]:bbox[1][1],bbox[0][0]:bbox[0][1]]
                                        recounted = Counter(clipped_seg.flatten())


                                        if 4 in recounted.keys() and recounted[4]>0.1*len(clipped_seg.flatten()):
                                            clipped_depth=depth_array[bbox[1][0]:bbox[1][1],bbox[0][0]:bbox[0][1]]
                                            #print (clipped_depth.shape)
                                            people_indx=np.where(clipped_seg==4)
                                            masked_depth=[]
                                            for people in zip(people_indx[0],people_indx[1] ):
                                                #print(people)
                                                masked_depth.append(clipped_depth[people])
                                            #masked_depth=clipped_depth[np.where(clipped_seg==4)]
                                            #print (masked_depth)
                                            #print ("Depth "+ str(min(zs))+" "+ str(max(zs)))
                                            #recounted = Counter(masked_depth)
                                            #print(recounted)
                                            avg_depth=np.mean(masked_depth)
                                            if avg_depth<700 and avg_depth>=min(zs)-10 and avg_depth<= max(zs)+10:
                                                #print("Correct depth")
                                                pedestrians[agent.id]['bounding_box_2D'] = f_2D_rotated
                                                pedestrians[agent.id]['bounding_box_2D_size']=recounted[4]
                                                pedestrians[agent.id]['bounding_box_2D_avg_depth']=avg_depth
                                                pedestrians[agent.id]['bounding_box_2D_depths']=zs
                                                #print ( pedestrians[agent.id].keys())
                                            #else:
                                                # print(recounted)
                                                # print ("Depth "+ str(min(zs))+" "+ str(max(zs)))


                                    #if sum(norm(depth_array-np.mean(zs))<1.0):


                                    # pedestrians[agent.id] = {}
                                    # pedestrians[agent.id]['transform'] = [agent.pedestrian.transform.location.x,
                                    #                                       agent.pedestrian.transform.location.y,
                                    #                                       agent.pedestrian.transform.location.z]
                                    # pedestrians[agent.id]['yaw'] = agent.pedestrian.transform.rotation.yaw
                                    # pedestrians[agent.id]['bounding_box'] = [agent.pedestrian.bounding_box.extent.x,
                                    #                                          agent.pedestrian.bounding_box.extent.y,
                                    #                                          agent.pedestrian.bounding_box.extent.z]
                                    # vehicle_transform = Transform(agent.pedestrian.bounding_box.transform)
                                    # pos = agent.pedestrian.transform.location
                                    #
                                    # bbox3d = agent.pedestrian.bounding_box.extent
                                    #
                                    # # Compute the 3D bounding boxes
                                    # # f contains the 4 points that corresponds to the bottom
                                    # f = np.array([[pos.x + bbox3d.x, pos.y - bbox3d.y, pos.z- bbox3d.z ],
                                    #               [pos.x + bbox3d.x, pos.y + bbox3d.y, pos.z- bbox3d.z ],
                                    #               [pos.x - bbox3d.x, pos.y + bbox3d.y, pos.z- bbox3d.z ],
                                    #               [pos.x - bbox3d.x, pos.y - bbox3d.y, pos.z- bbox3d.z ],
                                    #               [pos.x + bbox3d.x, pos.y - bbox3d.y, pos.z + bbox3d.z],
                                    #               [pos.x + bbox3d.x, pos.y + bbox3d.y, pos.z + bbox3d.z],
                                    #               [pos.x - bbox3d.x, pos.y + bbox3d.y, pos.z + bbox3d.z],
                                    #               [pos.x - bbox3d.x, pos.y - bbox3d.y, pos.z + bbox3d.z]])
                                    #
                                    # f_rotated = pedestrian_transform.transform_points(f)
                                    # pedestrians[agent.id]['bounding_box_coord'] = f_rotated
                                    # f_2D_rotated = []
                                    #
                                    # for i in range(f.shape[0]):
                                    #     point = np.array([[f_rotated[i, 0]], [f_rotated[i, 1]], [f_rotated[i, 2]], [1]])
                                    #     transformed_2d_pos = np.dot(Rt_inv, point)
                                    #     pos2d = np.dot(K, transformed_2d_pos[:3])
                                    #     norm_pos2d = np.array([
                                    #         pos2d[0] / pos2d[2],
                                    #         pos2d[1] / pos2d[2],
                                    #         pos2d[2]])
                                    #     f_2D_rotated.append([image_size[0] - norm_pos2d[0], image_size[1] - norm_pos2d[1]])
                                    # pedestrians[agent.id]['bounding_box_2D'] = f_2D_rotated

                            cars_dict[frame] = vehicles

                            pedestrians_dict[frame] = pedestrians
                            #print("End of Episode")
                            #print(len(pedestrians_dict[frame]))

                            # Save PLY to disk
                            # This generates the PLY string with the 3D points and the RGB colors
                            # for each row of the file.
                            if not frame % pointcloud_step:
                                point_cloud.save_to_disk(os.path.join(
                                    output_folder, '{:0>5}.ply'.format(frame))
                                )
                                point_cloud_seg.save_to_disk(os.path.join(
                                    output_folder, '{:0>5}_seg.ply'.format(frame))
                                )

                        if not args.autopilot:

                            client.send_control(
                                hand_brake=True)

                        else:

                            # Together with the measurements, the server has sent the
                            # control that the in-game autopilot would do this frame. We
                            # can enable autopilot by sending back this control to the
                            # server. We can modify it if wanted, here for instance we
                            # will add some noise to the steer.

                            control = measurements.player_measurements.autopilot_control
                            control.steer += random.uniform(-0.1, 0.1)
                            client.send_control(control)
                    print ("Start pickle save")
                    with open(output_folder + '/cameras.p', 'w') as camerafile:
                        pickle.dump(cameras_dict, camerafile)
                    with open(output_folder + '/people.p', 'w') as peoplefile:
                        pickle.dump(pedestrians_dict, peoplefile)
                    with open(output_folder + '/cars.p', 'w') as carfile:
                        pickle.dump(cars_dict, carfile)
                    print ("Episode done")



def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.0f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)

def check_far(value):
    fvalue = float(value)
    if fvalue < 0.0 or fvalue > 1.0:
        raise argparse.ArgumentTypeError(
            "{} must be a float between 0.0 and 1.0")
    return fvalue


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-f', '--far',
        default=0.2,
        type=check_far,
        help='The maximum save distance of camera-point '
             '[0.0 (near), 1.0 (far)] (default: 0.2)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')


    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = 'new_data-viz/test_{:d}/{:s}/{:0>6d}'

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(5)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

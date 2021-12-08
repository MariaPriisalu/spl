#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB), and the INTEL Visual Computing Lab.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client to generate point cloud in PLY format that you
   can visualize with MeshLab (meshlab.net) for instance. Please
   refer to client_example.py for a simpler and more documented example."""

# Used to gather visualization dataset

from __future__ import print_function

import argparse
import logging
import os
import random
import time

from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line, StopWatch
from carla.image_converter import depth_to_local_point_cloud, to_rgb_array,labels_to_array
from carla.transform import Transform
from carla.client import VehicleControl
import numpy as np
import pickle


def run_carla_client(host, port, far):
    # Here we will run a single episode with 300 frames.
    number_of_frames = 500
    frame_step = 50  # Save one image every 100 frames

    image_size = [800, 600]
    camera_local_pos = [0.3, 0.0, 1.3] # [X, Y, Z]
    camera_local_rotation = [0, 0, 0]  # [pitch(Y), yaw(Z), roll(X)]
    fov = 70
    autopilot=False
    control = VehicleControl()
    for start_i in range(150):
        output_folder = 'Packages/CARLA_0.8.2/PythonClient/_out/test_' + str(start_i)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
            print( "make "+str(output_folder))

    # Connect with the server
    with make_carla_client(host, port) as client:
        print('CarlaClient connected')
        for start_i in range(150):
            output_folder = 'Packages/CARLA_0.8.2/PythonClient/_out/test_' + str(start_i)
            print(output_folder)

            # Here we load the settings.
            settings = CarlaSettings()
            settings.set(
                SynchronousMode=True,
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=100,
                NumberOfPedestrians=500,
                WeatherId=random.choice([1, 3, 7, 8, 14]))
            settings.randomize_seeds()

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

            camera3 = Camera('CameraSeg', PostProcessing='SemanticSegmentation')
            camera3.set_image_size(*image_size)
            camera3.set_position(*camera_local_pos)
            camera3.set_rotation(*camera_local_rotation)
            settings.add_sensor(camera3)

            client.load_settings(settings)

            # Start at location index id '0'
            client.start_episode(start_i)

            cameras_dict = {}
            pedestrians_dict = {}
            cars_dict = {}
            # Compute the camera transform matrix
            camera_to_car_transform = camera2.get_unreal_transform()
            # (Intrinsic) (3, 3) K Matrix
            K = np.identity(3)
            K[0, 2] = image_size[0]/ 2.0
            K[1, 2] = image_size[1] / 2.0
            K[0, 0] = K[1, 1] = image_size[0] / (2.0 * np.tan(fov * np.pi / 360.0))
            with open(output_folder + '/camera_intrinsics.p', 'w') as camfile:
                pickle.dump(K, camfile)


            # Iterate every frame in the episode except for the first one.
            for frame in range(1, number_of_frames):
                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()

                # Save one image every 'frame_step' frames
                if not frame % frame_step:
                    for name, measurement in sensor_data.items():
                        filename ='{:s}/{:0>6d}'.format( name, frame)
                        measurement.save_to_disk(os.path.join(output_folder, filename))
                    # Start transformations time mesure.
                    timer = StopWatch()

                    # RGB image [[[r,g,b],..[r,g,b]],..[[r,g,b],..[r,g,b]]]
                    image_RGB = to_rgb_array(sensor_data['CameraRGB'])
                    image_seg = np.tile(labels_to_array(sensor_data['CameraSeg']), (1, 1, 3))

                    # 2d to (camera) local 3d
                    # We use the image_RGB to colorize each 3D point, this is optional.
                    # "max_depth" is used to keep only the points that are near to the
                    # camera, meaning 1.0 the farest points (sky)
                    point_cloud = depth_to_local_point_cloud(
                        sensor_data['CameraDepth'],
                        image_RGB,
                        max_depth=far
                    )

                    point_cloud_seg = depth_to_local_point_cloud(
                        sensor_data['CameraDepth'],
                        image_seg,
                        max_depth=far
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
                    Rt_inv=car_to_world_transform.inverse().matrix
                    #R_inv=world_transform.inverse().matrix
                    cameras_dict[frame] = {}
                    cameras_dict[frame]['inverse_rotation'] = Rt_inv[:]
                    cameras_dict[frame]['rotation']=Rt[:]
                    cameras_dict[frame]['translation']=Rt_inv[0:3,3]
                    cameras_dict[frame]['camera_to_car'] =camera_to_car_transform.matrix



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
                                                               agent.vehicle.transform.location.z ]
                            vehicles[agent.id]['bounding_box.transform'] =agent.vehicle.bounding_box.transform.location.z

                            vehicles[agent.id]['yaw'] = agent.vehicle.transform.rotation.yaw
                            vehicles[agent.id]['bounding_box'] = [agent.vehicle.bounding_box.extent.x,
                                                                  agent.vehicle.bounding_box.extent.y,
                                                                  agent.vehicle.bounding_box.extent.z]
                            vehicle_transform = Transform(agent.vehicle.bounding_box.transform)
                            pos = agent.vehicle.transform.location

                            bbox3d = agent.vehicle.bounding_box.extent

                            # Compute the 3D bounding boxes
                            # f contains the 4 points that corresponds to the bottom
                            f = np.array([[pos.x + bbox3d.x, pos.y - bbox3d.y, pos.z - bbox3d.z+agent.vehicle.bounding_box.transform.location.z],
                                          [pos.x + bbox3d.x, pos.y + bbox3d.y, pos.z - bbox3d.z+agent.vehicle.bounding_box.transform.location.z],
                                          [pos.x - bbox3d.x, pos.y + bbox3d.y, pos.z - bbox3d.z+agent.vehicle.bounding_box.transform.location.z],
                                          [pos.x - bbox3d.x, pos.y - bbox3d.y, pos.z - bbox3d.z+agent.vehicle.bounding_box.transform.location.z],
                                          [pos.x + bbox3d.x, pos.y - bbox3d.y, pos.z + bbox3d.z+agent.vehicle.bounding_box.transform.location.z],
                                          [pos.x + bbox3d.x, pos.y + bbox3d.y, pos.z + bbox3d.z+agent.vehicle.bounding_box.transform.location.z],
                                          [pos.x - bbox3d.x, pos.y + bbox3d.y, pos.z + bbox3d.z+agent.vehicle.bounding_box.transform.location.z],
                                          [pos.x - bbox3d.x, pos.y - bbox3d.y, pos.z + bbox3d.z+agent.vehicle.bounding_box.transform.location.z]])

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
                                #print([image_size[0] - (pos2d[0] / pos2d[2]), image_size[1] - (pos2d[1] / pos2d[2])])
                                f_2D_rotated.append(np.array([image_size[0] - norm_pos2d[0], image_size[1] - norm_pos2d[1]]))
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
                            vehicle_transform = Transform(agent.pedestrian.bounding_box.transform)
                            pos = agent.pedestrian.transform.location

                            bbox3d = agent.pedestrian.bounding_box.extent

                            # Compute the 3D bounding boxes
                            # f contains the 4 points that corresponds to the bottom
                            f = np.array([[pos.x + bbox3d.x, pos.y - bbox3d.y, pos.z],
                                          [pos.x + bbox3d.x, pos.y + bbox3d.y, pos.z],
                                          [pos.x - bbox3d.x, pos.y + bbox3d.y, pos.z],
                                          [pos.x - bbox3d.x, pos.y - bbox3d.y, pos.z],
                                          [pos.x + bbox3d.x, pos.y - bbox3d.y, pos.z + bbox3d.z],
                                          [pos.x + bbox3d.x, pos.y + bbox3d.y, pos.z + bbox3d.z],
                                          [pos.x - bbox3d.x, pos.y + bbox3d.y, pos.z + bbox3d.z],
                                          [pos.x - bbox3d.x, pos.y - bbox3d.y, pos.z + bbox3d.z]])

                            f_rotated=vehicle_transform.transform_points(f)
                            pedestrians[agent.id]['bounding_box_coord'] = f_rotated
                            f_2D_rotated= []


                            for i in range(f.shape[0]):
                                point=np.array([[f_rotated[i,0]],[f_rotated[i,1]],[f_rotated[i,2]],[1]])
                                transformed_2d_pos=np.dot(Rt_inv, point)
                                pos2d=np.dot(K, transformed_2d_pos[:3])
                                norm_pos2d = np.array([
                                    pos2d[0] / pos2d[2],
                                    pos2d[1] / pos2d[2],
                                    pos2d[2]])
                                f_2D_rotated.append([image_size[0] - norm_pos2d[0], image_size[1] - norm_pos2d[1]])
                            pedestrians[agent.id]['bounding_box_2D'] = f_2D_rotated


                    cars_dict[frame] = vehicles
                    pedestrians_dict[frame] = pedestrians

                    # End transformations time mesure.
                    timer.stop()

                    # Save PLY to disk
                    # This generates the PLY string with the 3D points and the RGB colors
                    # for each row of the file.
                    point_cloud.save_to_disk(os.path.join(
                        output_folder, '{:0>5}.ply'.format(frame))
                    )
                    point_cloud_seg.save_to_disk(os.path.join(
                         output_folder,'{:0>5}_seg.ply'.format(frame))
                    )

                    print_message(timer.milliseconds(), len(point_cloud), frame)

                if autopilot:
                    client.send_control(
                        measurements.player_measurements.autopilot_control
                    )
                else:
                    control.hand_brake = True
                    client.send_control(control)
            with open(output_folder + '/cameras.p', 'w') as camerafile:
                pickle.dump(cameras_dict, camerafile)
                print(output_folder +"cameras.p")
            with open(output_folder + '/people.p', 'w') as peoplefile:
                pickle.dump(pedestrians_dict, peoplefile)
            with open(output_folder + '/cars.p', 'w') as carfile:
                pickle.dump(cars_dict, carfile)


def print_message(elapsed_time, point_n, frame):
    message = ' '.join([
        'Transformations took {:>3.0f} ms.',
        'Saved {:>6} points to "{:0>5}.ply".'
    ]).format(elapsed_time, point_n, frame)
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

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    while True:
        try:
            run_carla_client(host=args.host, port=args.port, far=args.far)
            print('\nDone!')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nClient stoped by user.')

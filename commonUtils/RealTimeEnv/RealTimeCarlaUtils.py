import glob
import os
import sys
import random
import logging
import time
import argparse
import re
import math
from enum import Enum
from typing import Dict, List, Tuple
import pathlib

############# SOME USFEULL GLOBAL PARAMS ##########
ENABLE_LOCAL_SCENE_RENDERING = True # If true, a window will be created on the RLAgent simulator side to show graphical outputs real time
if ENABLE_LOCAL_SCENE_RENDERING:
    import pygame


# TODO put this in an env variable
CARLA_INSTALL_PATH = "CARLA"#os.getenv('CARLA_INSTALL_PATH') #= 'SSD/carla'
assert CARLA_INSTALL_PATH, "Define CARLA_INSTALL_PATH. Inside it should contain the PythonAPI folder directly "
# carla-0.9.11-py3.7-linux-x86_64.egg

try:
    carlaModuleFilter = os.path.join(CARLA_INSTALL_PATH, 'PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))
    print (carlaModuleFilter)
    carlaModulePath = glob.glob(carlaModuleFilter)[0]

    assert os.path.exists(carlaModulePath), "Can't find carla egg"
    sys.path.append(carlaModulePath)
except IndexError:
    assert False, "Path to carla egg not found !"


sys.path.append(os.path.join(CARLA_INSTALL_PATH, 'PythonAPI/carla'))


# This is CARLA agent
from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import is_within_distance_ahead, compute_magnitude_angle

import carla
print(dir(carla))
print(carla.__file__)

from carla import Client as Cli


import copy
import numpy as np
import pickle
import traceback
import shutil
from enum import Enum
import math

try:
    import queue
except ImportError:
    import Queue as queue

def check_far(value):
    fvalue = float(value)
    if fvalue < 0.0 or fvalue > 1.0:
        raise argparse.ArgumentTypeError(
            "{} must be a float between 0.0 and 1.0")
    return fvalue

SYNC_TIME = 0.1
SYNC_TIME_PLUS = 3.0 # When destroying the environment

##################################### BEGIN BASIC FUNCTIONS ########################################
def is_within_distance(target_location, current_location, orientation, max_distance, d_angle_th_up, d_angle_th_low=0):
    """
    Check if a target object is within a certain distance from a reference object.
    A vehicle in front would be something around 0 deg, while one behind around 180 deg.
        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :param max_distance: maximum allowed distance
        :param d_angle_th_up: upper thereshold for angle
        :param d_angle_th_low: low thereshold for angle (optional, default is 0)
        :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle_th_low < d_angle < d_angle_th_up


def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D po-0.427844-0.427844ints
        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm

def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))

def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2

        :param location_1, location_2: carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]


def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points

        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm

def dot3D(v1, v2):
    return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z)

def dot2D(v1, v2):
    return (v1.x * v2.x + v1.y * v2.y)

    # Given a carla transform, create a rotation matrix and return it
def get_matrix(transform):
    """
    Creates matrix from carla transform.
    """
    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix

# Get the bbox of an actor in world space
def getActorWorldBBox(actor: carla.Actor):
    actorBBox = actor.bounding_box
    cords = np.zeros((8, 4))

    # Get the box extent
    extent = actorBBox.extent
    cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
    cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
    cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
    cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
    cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
    cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
    cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
    cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])

    bb_transform = carla.Transform(actorBBox.location)
    bb_matrix = get_matrix(bb_transform)
    actor_world_matrix = get_matrix(actor.get_transform())
    bb_world_matrix = np.dot(actor_world_matrix, bb_matrix)
    world_cords = np.dot(bb_world_matrix, np.transpose(cords))
    return world_cords

def convertMToCM(loc : carla.Location):
    newLoc = carla.Location(loc.x * 100.0, loc.y * 100.0, loc.z * 100.0)
    return newLoc

##################################### END BASIC FUNCTIONS ########################################


##################################### BEGIN LOCAL RENDERING FUNCTIONS ########################################
class RenderUtils(object):
    class EventType(Enum):
        EV_NONE = 0
        EV_QUIT = 1
        EV_SWITCH_TO_DEPTH = 2
        EV_SWITCH_TO_RGBANDSEG = 3
        EV_SIMPLIFIED_TOPVIEW = 4

    @staticmethod
    def draw_image(surface, image, blend=False):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if blend:
            image_surface.set_alpha(100)
        surface.blit(image_surface, (0, 0))

    @staticmethod
    def get_font():
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = 'ubuntumono'
        font = default_font if default_font in fonts else fonts[0]
        font = pygame.font.match_font(font)
        return pygame.font.Font(font, 14)

    @staticmethod
    def get_input_event():
        for event in pygame.event.get():
            # See QUIT first then swap
            if event.type == pygame.QUIT:
                return RenderUtils.EventType.EV_QUIT
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return RenderUtils.EventType.EV_QUIT
                if event.key == pygame.K_d:
                    return RenderUtils.EventType.EV_SWITCH_TO_DEPTH
                if event.key == pygame.K_s:
                    return RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG
                if event.key == pygame.K_t:
                    return RenderUtils.EventType.EV_SIMPLIFIED_TOPVIEW
        return RenderUtils.EventType.EV_NONE

# Manager to synchronize output from different sensors.

##################################### END LOCAL RENDERING FUNCTIONS ########################################


##################################### BEGIN Traffic Management FUNCTIONS ########################################
class CustomGlobalRoutePlanner(GlobalRoutePlanner):
    def __init__(self, dao):
        super(CustomGlobalRoutePlanner, self).__init__(dao=dao)

    def compute_direction_velocities(self, origin, velocity, destination):
        node_list = super(CustomGlobalRoutePlanner, self)._path_search(origin=origin, destination=destination)

        origin_xy = np.array([origin.x, origin.y])
        velocity_xy = np.array([velocity.x, velocity.y])
        first_node_xy = self._graph.nodes[node_list[0]]['vertex']
        first_node_xy = np.array([first_node_xy[0], first_node_xy[1]])
        target_direction_vector = first_node_xy - origin_xy
        target_unit_vector = np.array(target_direction_vector) / np.linalg.norm(target_direction_vector)

        vel_s = np.dot(velocity_xy, target_unit_vector)

        unit_velocity = velocity_xy / (np.linalg.norm(velocity_xy) + 1e-8)
        angle = np.arccos(np.clip(np.dot(unit_velocity, target_unit_vector), -1.0, 1.0))
        vel_perp = np.linalg.norm(velocity_xy) * np.sin(angle)
        return vel_s, vel_perp

    def compute_distance(self, origin, destination):
        node_list = super(CustomGlobalRoutePlanner, self)._path_search(origin=origin, destination=destination)
        # print('Node list:', node_list)
        first_node_xy = self._graph.nodes[node_list[1]]['vertex']
        # print('Diff:', origin, first_node_xy)

        # distance = 0.0
        distances = []
        distances.append(np.linalg.norm(np.array([origin.x, origin.y, 0.0]) - np.array(first_node_xy)))

        for idx in range(len(node_list) - 1):
            distances.append(super(CustomGlobalRoutePlanner, self)._distance_heuristic(node_list[idx], node_list[idx + 1]))
        # print('Distances:', distances)
        # import pdb; pdb.set_trace()
        return np.sum(distances)


######################### BEGIN: Weather and scene methods ####################
class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        min_alt, max_alt = [20, 90]
        self.altitude = 0.5 * (max_alt + min_alt) + 0.5 * (max_alt - min_alt) * math.cos(self._t)

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 60.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    def __init__(self, world, changing_weather_speed):
        self.world = world
        self.reset()
        self.weather = world.get_weather()
        self.changing_weather_speed = changing_weather_speed
        self._sun = Sun(self.weather.sun_azimuth_angle, self.weather.sun_altitude_angle)
        self._storm = Storm(self.weather.precipitation)

    def reset(self):
        weather_params = carla.WeatherParameters(sun_altitude_angle=90.)
        self.world.set_weather(weather_params)

    def tick(self):
        self._sun.tick(self.changing_weather_speed)
        self._storm.tick(self.changing_weather_speed)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude
        self.world.set_weather(self.weather)

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)
######################### END: Weather and scene methods ####################


######################### BEGIN: Sync methods ####################


class SensorsDataManagement(object):
    def __init__(self, world, fps, sensorsDict):
        self.world = world
        self.sensors = sensorsDict  # [name->obj instance]
        self.frame = None

        self.delta_seconds = 1.0 / fps
        self._queues = {}  # Data queues for each sensor given + update messages
        self.RenderAsDepth = RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG

        def make_queue(register_event, sensorname):
            q = queue.Queue()
            register_event(q.put)
            assert self._queues.get(sensorname) is None
            self._queues[sensorname] = q

        make_queue(self.world.on_tick, "worldSnapshot") # The ontick register event
        for name, inst in self.sensors.items():
            make_queue(inst.listen, name)

    # Tick the manager using a timeout to get data and a targetFrame that the parent is looking to receive data for
    def tick(self, targetFrame, timeout):
        #logging.log(logging.INFO, ("Ticking manager to get data for targetFrame {0}").format(targetFrame))

        """
        print("--Debug print the queue status")
        for name,inst in self._queues.items():
             print(f"--queue name {name} has size {inst.qsize()}")
        """

        data = {name: self._retrieve_data(targetFrame, q, timeout) for name, q in self._queues.items()}
        assert all(inst.frame == targetFrame for key,inst in data.items())  # Need to get only data for the target frame requested
        #logging.log(logging.INFO, ("Got data for frame {0}").format(targetFrame))

        return data

    # Gets the target frame from a sensor queue
    def _retrieve_data(self, targetFrame,  sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            # assert data.frame <= targetFrame, ("You are requesting an old frame which was already processed !. data %d, target %d" % (data.frame, targetFrame))
            if data.frame == targetFrame:
                return data

######################### END: Sync methods ####################


# This is used to understand the parameters - input/output of an online or offline data gathering process
# Here we put static things about the environment, simulation , params etc.
class DataGatherParams:
    # First, some fixed parameters for data capture. Maybe some of them must be expxorted as well
    #---------------------------------------------------------------
    frame_step = 1  # Save one image every 10 frames

    # How to find the episode LENGTH
    # Explanation: In one second you have fixedFPS. We record a measure at each frame_step frames. If fixedFPS=30
    # then in one second you have 3 captures.
    # If number_of_frames_to_capture = 200 => An episode length is 100/3 seconds

    # Camera specifications, size, local pos and rotation to car
    image_size = [800, 600]
    # camera_local_pos = carla.Location(x=0.3, y=0.0, z=1.3)  # [X, Y, Z] of local camera
    camera_front_transform = carla.Transform(carla.Location(x=0.8, z=1.65))
    fov = 70
    isAutopilotForPlayerVehicle = True


    # Environment settings
    OUTPUT_DATA_PREFIX = "out/%s/episode_%d_%d"  # out/MapName/episode_index_spawnPointIndex
    OUTPUT_SEG = "CameraSeg"
    OUTPUT_SEGCITIES = "CameraSegCities"
    OUTPUT_DEPTH = "CameraDepth"
    OUTPUT_DEPTHLOG = "CameraDepthProc"
    OUTPUT_RGB = "CameraRGB"
    ENABLED_SAVING = False  # If activated, output saving data is enabled
    CLEAN_PREVIOUS_DATA = False  # If activated, previous data folder is deleted

    # If true, then car stays and waits for pedestrians and other cars to move around. If
    # If false, then on each frame it moves to a random proximity waypoint
    STATIC_CAR = True

    def prepareEpisodeOutputFolders(self, episodeDataPath):
        if self.CLEAN_PREVIOUS_DATA:
            if os.path.exists(episodeDataPath):
                shutil.rmtree(episodeDataPath)

        if True: #not os.path.exists(episodeDataPath):
            #os.makedirs(episodeDataPath, exist_ok=True)
            os.makedirs(self.getOutputFolder_depth(episodeDataPath), exist_ok=True)
            os.makedirs(self.getOutputFolder_depthLog(episodeDataPath), exist_ok=True)
            os.makedirs(self.getOutputFolder_rgb(episodeDataPath), exist_ok=True)
            os.makedirs(self.getOutputFolder_seg(episodeDataPath), exist_ok=True)
            os.makedirs(self.getOutputFolder_segcities(episodeDataPath), exist_ok=True)

    def prepareSceneOutputFolders(self, sceneDataPath):
        if self.CLEAN_PREVIOUS_DATA:
            if os.path.exists(sceneDataPath):
                shutil.rmtree(sceneDataPath)

        if True: #not os.path.exists(sceneDataPath):
            os.makedirs(sceneDataPath, exist_ok=True)
    #----------------------------------------------------------------

    @staticmethod
    def getOutputFolder_seg(baseFolder):
        return os.path.join(baseFolder, DataGatherParams.OUTPUT_SEG)

    @staticmethod
    def getOutputFolder_segcities(baseFolder):
        return os.path.join(baseFolder, DataGatherParams.OUTPUT_SEGCITIES)

    @staticmethod
    def getOutputFolder_rgb(baseFolder):
        return os.path.join(baseFolder, DataGatherParams.OUTPUT_RGB)

    @staticmethod
    def getOutputFolder_depth(baseFolder):
        return os.path.join(baseFolder, DataGatherParams.OUTPUT_DEPTH)

    @staticmethod
    def getOutputFolder_depthLog(baseFolder):
        return os.path.join(baseFolder, DataGatherParams.OUTPUT_DEPTHLOG)

    @staticmethod
    def configureCameraBlueprint(cameraBp):
        cameraBp.set_attribute('image_size_x', str(DataGatherParams.image_size[0]))
        cameraBp.set_attribute('image_size_y', str(DataGatherParams.image_size[1]))
        cameraBp.set_attribute('fov', str(DataGatherParams.fov))
        # Set the time in seconds between sensor captures
        # cameraBp.set_attribute('sensor_tick', '1.0')

    # outputEpisodeDataPath will be used to put the scene data + simulation data such that you can read that folder and load it in the RLAgent directly after as a dataset example
    # Num frames is for how many number of frames the simulation should happen. -1 means infinite
    # Use copyScenePaths to copy the ply files from base data path to each sampled scene
    def __init__(self,  outputEpisodeDataPath,
                        sceneName=None,
                        useHeroCarPerspective=False,
                        episodeIndex = 0,
                        numFrames=-1,
                        maxNumberOfEpisodes=1,
                        mapsToTest = ["Town03"],
                        copyScenePaths=False,
                        host="localhost",
                        port=2000,
                        args=None):  # If true, the RGB/semantic images will be gathered from a hero car perspective instead of top view

        self.outputEpisodesBasePath = outputEpisodeDataPath
        self.sceneName = sceneName
        self.outputEpisodesBasePath_currentSceneData = os.path.join(self.outputEpisodesBasePath, sceneName)
        os.makedirs(self.outputEpisodesBasePath_currentSceneData, exist_ok=True)
        self.outputCurrentEpisodePath = None
        self.useHeroCarPerspective = useHeroCarPerspective
        self.episodeIndex = episodeIndex
        self.numFrames = numFrames
        self.maxNumberOfEpisodes = maxNumberOfEpisodes
        self.mapsToTest = mapsToTest
        self.host = host
        self.port = port
        self.collectSceneData = False
        self.copyScenePaths = copyScenePaths

        # Check if we need to rewrite the point cloud scene files
        needSceneReconstruct = False
        if args.forceSceneReconstruction:
            needSceneReconstruct = True
        else:
            # Check if the ply files are there
            plyFilesAreThere = False
            src_files = os.listdir(self.outputEpisodesBasePath_currentSceneData)
            for file_name in src_files:
                if "ply" in pathlib.Path(file_name).suffix:
                    plyFilesAreThere = True
                    break
            needSceneReconstruct = plyFilesAreThere == False

        self.rewritePointCloud = needSceneReconstruct # TODO fix from params

        # Cache the folders for storing the outputs
        if self.useHeroCarPerspective and self.outputEpisodesBasePath:
            self.outputFolder_depth  = self.getOutputFolder_depth(self.outputEpisodesBasePath)
            self.outputFolder_depthLog = self.getOutputFolder_depthLog(self.outputEpisodesBasePath)
            self.outputFolder_rgb    = self.getOutputFolder_rgb(self.outputEpisodesBasePath)
            self.outputFolder_seg    = self.getOutputFolder_seg(self.outputEpisodesBasePath)
            self.output_segcities    = self.getOutputFolder_segcities(self.outputEpisodesBasePath)


    def saveHeroCarPerspectiveData(self, syncData, frameId):
        if self.useHeroCarPerspective and self.outputEpisodesBasePath:
            worldSnapshot = syncData['worldSnapshot']
            image_seg = syncData["seg"]
            image_rgb = syncData["rgb"]
            image_depth = syncData["depth"]

            # Save output from sensors to disk if requested
            if self.ENABLED_SAVING and (frameId % self.frame_step == 0):
                fileName = ("%06d.png" % frameId)
                # Save RGB
                image_rgb.save_to_disk(os.path.join(self.outputFolder_rgb, fileName))

                # Save seg
                image_seg.save_to_disk(os.path.join(self.outputFolder_seg, fileName))
                image_seg.convert(carla.ColorConverter.CityScapesPalette)
                image_seg.save_to_disk(os.path.join(self.output_segcities, fileName))

                # Save depth
                image_depth.save_to_disk(os.path.join(self.outputFolder_depth, fileName))
                image_depth.convert(carla.ColorConverter.Depth)
                image_depth.save_to_disk(os.path.join(self.outputFolder_depthLog, fileName))
            else:
                image_seg.convert(carla.ColorConverter.CityScapesPalette)
                image_depth.convert(carla.ColorConverter.Depth)

    # Save some simulation data on the episode's output path
    def saveSimulatedData(self, vehiclesData, pedestriansData):
        if True: #self.useHeroCarPerspective and self.outputCurrentEpisodePath:
            filepathsAndDictionaries = {'pedestrians': (pedestriansData, os.path.join(self.outputCurrentEpisodePath, "people.p")),
                                        'cars': (vehiclesData, os.path.join(self.outputCurrentEpisodePath, "cars.p"))
                                        }
            for key, value in filepathsAndDictionaries.items():
                dataObj = value[0]
                filePath = value[1]
                with open(filePath, mode="wb") as fileObj:
                    pickle.dump(dataObj, fileObj, protocol=2)  # Protocol 2 because seems to be compatible between 2.x and 3.x !


class RenderType(Enum):
    RENDER_NONE = 1 # No local rendering
    RENDER_COLORED=2 # Full RGB/ semantic rendering of the scene
    RENDER_SIMPLIFIED=3  # Simplified dots/lines rendering

# How is RLAgent controlling the pedestrian
class ControlledPedestrianType:
    CONTROLLED_PEDESTRIAN_AUTHORITIVE= 1, # RLAgents sets the position and orientation, no control from Engine
    CONTROLLED_PEDESTRIAN_AUTHORITIVE_WITH_POSE = 2 # Same as above, RLAgents also sets the POSE
    CONTROLLED_PEDESTRIAN_ENGINE_CONTROLLED = 3 # Engine controls the agent, RLAgent that sends the velocity

class RenderOptions:
    def __init__(self, renderType : RenderType, topViewResX, topViewResY): # Res of the output vis
        if ENABLE_LOCAL_SCENE_RENDERING == False:
            assert renderType != RenderType.RENDER_NONE

        self.sceneRenderType = renderType
        self.topViewResX = topViewResX
        self.topViewResY = topViewResY

class ControlledCarSpawnParams:
    def __init__(self, position : carla.Location, isPositionInVoxels, yaw : float):
        self.position = position
        self.yaw = yaw
        self.isPositionsInVoxels = isPositionInVoxels

class ControlledPedestrianSpawnParams:
    def __init__(self, position : carla.Location, isPositionInVoxels, yaw : float, type : ControlledPedestrianType):
        self.position = position
        self.yaw = yaw
        self.isPositionsInVoxels = isPositionInVoxels
        self.type = type

class EnvSetupParams:
    # Some fixed set of params first
    #-------------------------------
    # Sim params
    fixedFPS = 30  # FPS for recording and simulation
    tm_port = 8000
    speedLimitExceedPercent = -30 # Exceed the speed limit by 30%
    distanceBetweenVehiclesCenters = 2.0 # Normal distance to keep between one distance vehicle to the one in its front
    synchronousComm = True
    useOnlySpawnPointsNearCrossWalks = False


    vehicles_filter_str = "vehicle.*"
    walkers_filter_str = "walker.pedestrian.*"

    # To promote having agents around the player spawn position, we randomly select F * numPedestrians locations as start/destination points
    PedestriansSpawnPointsFactor = 100
    PedestriansDistanceBetweenSpawnpoints = 1  # m
    #-------------------------------

    def __init__(self, controlledCarsParams : List[ControlledCarSpawnParams], controlledPedestriansParams : List[ControlledPedestrianSpawnParams], # Parameters for what cars and pedestrian to spawn in the environment
                 NumberOfCarlaVehicles, NumberOfCarlaPedestrians,
                 observerSpawnTransform : carla.Transform = None,
                 observerVoxelSize = 5.0,
                 observerNumVoxelsX = 2048,
                 observerNumVoxelsY = 1024,
                 observerNumVoxelsZ = 1024,
                 forceExistingRaycastActor = False,
                 mapToUse = ["Town3"],
                 sceneName=None):

        self.observerSpawnTransform = observerSpawnTransform
        self.observerVoxelSize = observerVoxelSize
        self.observerNumVoxelsX = observerNumVoxelsX
        self.observerNumVoxelsY = observerNumVoxelsY
        self.observerNumVoxelsZ = observerNumVoxelsZ
        self.forceExistingRaycastActor = forceExistingRaycastActor
        self.sceneName=sceneName

        self.NumberOfCarlaVehicles = NumberOfCarlaVehicles
        self.NumberOfCarlaPedestrians = NumberOfCarlaPedestrians
        self.controlledCarsParams = controlledCarsParams
        self.controlledPedestriansParams = controlledPedestriansParams
        self.mapToUse = mapToUse
        self.episodeIndex = 99999




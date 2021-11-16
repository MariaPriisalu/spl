from RealTimeCarlaUtils import *
from CarlaServerConnection import *
import carla

# This handles the management of actors, blueprints etc in the scene
class EnvironmentActors:
    def __init__(self, parent, carlaConnectionManager : CarlaServerConnection):
        self.parent = parent
        self.s_weather_presets = EnvironmentActors.find_weather_presets()
        self.s_heroCaptureCarPerspective = []  # The list of all actors currently spawned for hero perspective car capture (his car, sensor cameras ,etc)
        self.s_vehicles_list = []  # The list of all vehicle
        self.s_all_pedestrian_ids = []  # controller,walker pairs
        self.all_pedestrian_actors = []  # controller, walker pairs
        self.s_heroCaptureCarPerspective_sensors = dict()  # THe list of sensors (part of s_players_actor_list
        self.carlaConnectionManager = carlaConnectionManager
        self.dataGatherParams = None


        # Vehicles and pedestrians data as dicts of [FrameId][EntityId]['BBoxMinMax'], each with a 3x2 describing the bounding box as min value on column 0 and max on column 1
        # And 'velocity'
        self.vehicles_data = {}
        self.pedestrians_data = {}



    # Get the weather presets lists
    @staticmethod
    def find_weather_presets():
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


    # Promote uniform sampling around map + spawn in front of walkers
    @staticmethod
    def uniformSampleSpawnPoints(allSpawnpoints, numToSelect):
        availablePoints = [(index, transform) for index, transform in allSpawnpoints]
        selectedPointsAndIndices = [] #[None]*numToSelect

        for selIndex in range(numToSelect):
            # Select the one available that is furthest from existing ones
            bestPoint = None
            bestDist = -1
            for x in availablePoints:
                target_index = x[0]
                target_transform = x[1].location

                # Find the closest selected point to x
                closestDist = math.inf
                closestSelPoint = None
                for y in selectedPointsAndIndices:
                    selPointLocation = y[1].location
                    d = compute_distance(target_transform, selPointLocation)
                    if d < closestDist:
                        closestDist     = d
                        closestSelPoint = y

                if closestSelPoint == None or bestDist < closestDist:
                    bestDist = closestDist
                    bestPoint = x

            if  bestPoint != None:
                availablePoints.remove(bestPoint)
                selectedPointsAndIndices.append(bestPoint)

        return selectedPointsAndIndices

    def createWalkersBlueprintLibrary(self):
        blueprints = self.blueprint_library.filter(EnvSetupParams.walkers_filter_str)
        return blueprints

    def createVehiclesBlueprintLibrary(self):
        # Filter some vehicles library
        blueprints = self.blueprint_library.filter(EnvSetupParams.vehicles_filter_str)
        #blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        return blueprints

    def destroy_current_environment(self, client):
        if len(self.s_vehicles_list) == 0 and len(self.s_heroCaptureCarPerspective) == 0 and len(self.s_all_pedestrian_ids) == 0:
            logging.log(logging.INFO, 'Environment already distroyed')
            return

        logging.log(logging.INFO, 'Destroying %d vehicles' % len(self.s_vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in self.s_vehicles_list])
        self.s_vehicles_list = []

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        logging.log(logging.INFO,"Stopping the walker controllers")
        for i in range(0, len(self.all_pedestrian_actors), 2):
            self.all_pedestrian_actors[i].stop()

        logging.log(logging.INFO, f'Destroying all {len(self.s_all_pedestrian_ids)/2} walkers actors spawned')
        client.apply_batch([carla.command.DestroyActor(x) for x in self.s_all_pedestrian_ids])
        self.s_all_pedestrian_ids = []

        if len(self.s_heroCaptureCarPerspective) > 0:
            logging.log(logging.INFO, f"Destroying all {len(self.s_heroCaptureCarPerspective) / 2} player\'s actors")
            client.apply_batch([carla.command.DestroyActor(x) for x in self.s_heroCaptureCarPerspective])
            self.s_heroCaptureCarPerspective = []

        time.sleep(SYNC_TIME_PLUS)
        self.world.tick()

        logging.log(logging.INFO, "===End destroying the environment...")

        # Creates the point cloud or copies it if needed. Check the comment above DataGatheringParams class

    def solveSceneData(self):
        if self.dataGatherParams.rewritePointCloud:
            print("WRITING the point cloud files...please wait")
            self.capturePointCloud(self.dataGatherParams.outputEpisodesBasePath_currentSceneData)
            time.sleep(SYNC_TIME)
            self.world.tick()
            time.sleep(SYNC_TIME)
            self.dataGatherParams.rewritePointCloud = False

        # Copy all files from outputEpisodesBasePath to the outputCurrentEpisodePath to have data for the corresponding episode
        if self.dataGatherParams.copyScenePaths:
            os.makedirs(self.dataGatherParams.outputEpisodesBasePath_currentSceneData, exist_ok=True)
            src_files = os.listdir(self.dataGatherParams.outputEpisodesBasePath_currentSceneData)
            for file_name in src_files:
                fullFileName = os.path.join(self.dataGatherParams.outputEpisodesBasePath_currentSceneData, file_name)
                if os.path.isfile(fullFileName):
                    if not os.path.exists(self.dataGatherParams.outputCurrentEpisodePath):
                        os.makedirs(self.dataGatherParams.outputCurrentEpisodePath)
                    shutil.copy(fullFileName, self.dataGatherParams.outputCurrentEpisodePath)

    def capturePointCloud(self, outScenePath):
        self.world.capture_raycastActor(outpath=outScenePath, synchronous=True)

    def spawnActors(self, envSetup : EnvSetupParams, dataGatherParams : DataGatherParams):
        self.dataGatherParams = dataGatherParams
        self.envSetup = envSetup

        # reset some stats - see constructor to understand these
        self.vehicles_data = {}
        self.pedestrians_data = {}

        # Step 0: First reload the world
        self.carlaConnectionManager.reloadWorld(envSetup)
        self.world = self.carlaConnectionManager.world
        self.map = self.carlaConnectionManager.world.get_map()

        # Step 1: Get feedback from the created world
        #print(dir(self.carlaConnectionManager.world))
        self.spectator = self.carlaConnectionManager.world.get_spectator()
        self.raycastActor = self.carlaConnectionManager.world.get_raycastActor()

        isThereAnExistingRaycastActor = envSetup.forceExistingRaycastActor == True and self.raycastActor != None and self.raycastActor.id != 0
        assert isThereAnExistingRaycastActor or envSetup.observerSpawnTransform is not None, "Either you provide a raycasting actor or you specify the transform to spawn one at !"
        if isThereAnExistingRaycastActor is False or envSetup.observerSpawnTransform is not None:
            obsPosList : carla.Transform = envSetup.observerSpawnTransform
            loc = obsPosList.location  #carla.Location(-2920.0, 13740.0, 1140.0)
            dir = obsPosList.rotation
            dir = carla.Vector3D(dir.pitch, dir.yaw, dir.roll) #obsPosList.rotation
            loc_forUnreal = convertMToCM(loc)
            self.carlaConnectionManager.world.spawn_raycastActor(location=loc_forUnreal, direction=dir,
                                                                voxelsize=envSetup.observerVoxelSize, numvoxelsX=envSetup.observerNumVoxelsX,
                                                                 numvoxelsY=envSetup.observerNumVoxelsY, numvoxelsZ=envSetup.observerNumVoxelsZ)

            self.raycastActor = self.carlaConnectionManager.world.get_raycastActor()
            raycastActorTransform = carla.Transform(loc, carla.Rotation(dir.x, dir.y, dir.z))
            self.raycastActor.set_transform(raycastActorTransform)

            time.sleep(SYNC_TIME)
            self.world.tick()  # Be sure that player's vehicle is spawned

            assert self.raycastActor, "Can't create raycast actor !!!!"


        crosswalks = self.map.get_crosswalks()
        landmarks = self.map.get_all_landmarks()

        self.blueprint_library = self.carlaConnectionManager.world.get_blueprint_library()

        # These are the spawnpoints for the vehicles in the map
        self.vehicles_spawn_points = self.map.get_spawn_points()

        self.player_spawn_pointsAndIndices = None
        if EnvSetupParams.useOnlySpawnPointsNearCrossWalks:
            # These are the spawnpoints for the player vehicle
            # Get one for each episode indeed, sorted by importance
            # We try to spawn the player close and with view to crosswalks
            self.spawn_points_nearcrosswalks = self.map.get_spawn_points_nearcrosswalks()
            self.player_spawn_pointsAndIndices = self.uniformSampleSpawnPoints(self.spawn_points_nearcrosswalks,
                                                                               dataGatherParams.maxNumberOfEpisodes)
        else:
            self.player_spawn_pointsAndIndices = [(i, transform) for i, transform in enumerate(self.vehicles_spawn_points)]


        if len(self.player_spawn_pointsAndIndices) <= 0:
            "There are no interesting spawn points on this map. Remove map or lower requirements from the server side"
            self.carlaConnectionManager.releaseServerConnection()
            raise Exception()

        dataGatherParams.maxNumberOfEpisodes = min(len(self.player_spawn_pointsAndIndices), dataGatherParams.maxNumberOfEpisodes)
        logging.log(logging.INFO, 'I will simulate %d episodes' % dataGatherParams.maxNumberOfEpisodes)
        logging.log(logging.INFO, "There are %d interesting spawn points on the map" % len(self.player_spawn_pointsAndIndices))

        # Step 2: Spawn the requested entities
        #---------------------------------------------------------------------
        if dataGatherParams.useHeroCarPerspective:
            #spawnPointIter = dataGatherParams.episodeIndex % len(self.player_spawn_pointsAndIndices)
            #envSetup.observerSpawnTransform = self.player_spawn_pointsAndIndices[spawnPointIter][1]  # The location where to spawn
            #playerSpawnIndex = self.player_spawn_pointsAndIndices[spawnPointIter][0]  # The index from the original set of spawn points where to spawn
            envSetup.observerSpawnTransform = self.raycastActor.get_transform()
        else:
            #playerSpawnIndex = 9999
            envSetup.observerSpawnTransform = self.raycastActor.get_transform()

        # Save the spawn point index that is closest to our observer
        self.closestSpawnPointIndex = None
        self.closestSpawnPointDist = None
        self.closestSpawnPointTransform = None
        for spawnPointIndex, spawnPointTransform in self.player_spawn_pointsAndIndices:
            distToThisSpawnPoint = compute_distance(spawnPointTransform.location, envSetup.observerSpawnTransform.location)
            if self.closestSpawnPointIndex is None or distToThisSpawnPoint < self.closestSpawnPointDist:
                self.closestSpawnPointDist = distToThisSpawnPoint
                self.closestSpawnPointIndex = spawnPointIndex
                self.closestSpawnPointTransform = spawnPointTransform

        print(f"$$$ For scene name {dataGatherParams.sceneName} we found that the closest spawn point index is {self.closestSpawnPointIndex}")

        self.dataGatherParams.outputCurrentEpisodePath = os.path.join(dataGatherParams.outputEpisodesBasePath, str(self.envSetup.mapToUse), str(self.envSetup.episodeIndex), str(self.closestSpawnPointIndex))
        self.solveSceneData()
        if dataGatherParams.useHeroCarPerspective:
            envSetup.observerSpawnTransform = self.closestSpawnPointTransform

        logging.log(logging.INFO, "Starting to create the environment...")

        # Spawn the player's vehicle at the given location
        if dataGatherParams.useHeroCarPerspective:
            observerSpawnLocation = envSetup.observerSpawnTransform.location
            self.currWaypoint = self.map.get_waypoint(envSetup.observerSpawnTransform.location)  # This is its first waypoint
            vehiclesLib = self.blueprint_library.filter('vehicle.audi.a*')
            vehicleToSpawn = random.choice(vehiclesLib)
            self.observerVehicle = self.world.spawn_actor(vehicleToSpawn, envSetup.observerSpawnTransform)
            self.s_heroCaptureCarPerspective.append(self.observerVehicle)
            self.observerVehicle.set_simulate_physics(False)
            self.observerVehicle.set_autopilot(True)
        else:
            observerSpawnLocation = envSetup.observerSpawnTransform.location
            self.observerVehicle = self.raycastActor

        # Set the spectator pos and rot
        spectator_loc = observerSpawnLocation
        spectator_rot = carla.Rotation(yaw=0, pitch=0, roll=0)
        spectator_transform = carla.Transform(spectator_loc, spectator_rot)
        self.spectator.set_transform(spectator_transform)

        # Spawn the camera sensors/actors for the car perspective
        #------------------------------------------------------
        if dataGatherParams.useHeroCarPerspective:
            logging.log(logging.INFO, 'Spawning sensors...')
            # Create sensors blueprints
            cameraRgbBlueprint = self.blueprint_library.find('sensor.camera.rgb')
            cameraDepthBlueprint = self.blueprint_library.find('sensor.camera.depth')
            cameraSegBlueprint = self.blueprint_library.find('sensor.camera.semantic_segmentation')
            cameraBlueprints = [cameraRgbBlueprint, cameraDepthBlueprint, cameraSegBlueprint]
            for cam in cameraBlueprints:
                dataGatherParams.configureCameraBlueprint(cam)
                camera_rgb = self.world.spawn_actor(cameraRgbBlueprint, dataGatherParams.camera_front_transform, attach_to=self.observerVehicle)
                self.s_heroCaptureCarPerspective.append(camera_rgb)
                camera_depth = self.world.spawn_actor(cameraDepthBlueprint, dataGatherParams.camera_front_transform, attach_to=self.observerVehicle)
                self.s_heroCaptureCarPerspective.append(camera_depth)
                camera_semseg = self.world.spawn_actor(cameraSegBlueprint, dataGatherParams.camera_front_transform, attach_to=self.observerVehicle)
                self.s_heroCaptureCarPerspective.append(camera_semseg)
                self.s_heroCaptureCarPerspective_sensors = {'rgb' : camera_rgb, 'depth' : camera_depth, 'seg' : camera_semseg}
        #------------------------------------------------------

        self.dataManager = SensorsDataManagement(self.world, self.envSetup.fixedFPS, self.s_heroCaptureCarPerspective_sensors)
        SpawnActorFunctor = carla.command.SpawnActor

        # some settings
        percentagePedestriansRunning = 0.3  # how many pedestrians will run
        percentagePedestriansCrossing = 0.6  # how many pedestrians will walk through the road

        time.sleep(SYNC_TIME)
        self.world.tick() # Be sure that player's vehicle is spawned

        blueprints_walkers  = self.createWalkersBlueprintLibrary()
        blueprints_vehicles = self.createVehiclesBlueprintLibrary()

        # --------------
        # Spawn vehicles
        # --------------
        logging.log(logging.INFO, 'Spawning vehicles')
        self.vehicles_spawn_points = sorted(self.vehicles_spawn_points, key = lambda transform : compute_distance(transform.location, self.envSetup.observerSpawnTransform.location))
        for n, transform in enumerate(self.vehicles_spawn_points):
            if n >= self.envSetup.NumberOfCarlaVehicles:
                break
            blueprint = random.choice(blueprints_vehicles)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                self.s_vehicles_list.append(vehicle)

        spawnAndDestinationPoints = []

        # -------------
        # Spawn Walkers
        # -------------
        playerSpawnForward = self.envSetup.observerSpawnTransform.rotation.get_forward_vector()

        # Tests if a position is in front of an observer knowing its position and forward
        def isPosInFaceOfObserverPos(observerForward, observerPos, targetPos):
            observeToPos = targetPos - observerPos
            return dot2D(observeToPos, observerForward) > 0


        logging.log(logging.INFO, 'Spawning walkers...')
        walkers_list = []
        # 1. take all the random locations to spawn
        spawnAndDestinationPoints_extended = []
        # To promote having agents around the player spawn position, we randomly select F * numPedestrians locations,
        # on the navmesh, then select the closest ones to the spawn position
        numSpawnPointsToGenerate = self.envSetup.PedestriansSpawnPointsFactor * self.envSetup.NumberOfCarlaPedestrians
        for i in range(numSpawnPointsToGenerate):
            loc1 = self.world.get_random_location_from_navigation()
            loc2 = self.world.get_random_location_from_navigation()
            if (loc1 != None and loc2 != None):

                # Check if both of them are in front of the car
                isLoc1InFront = isPosInFaceOfObserverPos(playerSpawnForward, observerSpawnLocation, loc1)
                isLoc2InFront = isPosInFaceOfObserverPos(playerSpawnForward, observerSpawnLocation, loc2)
                if isLoc1InFront and isLoc2InFront:
                    # Swap spawn with destination maybe position
                    #if isLoc1InFront == False:
                    #    loc2, loc1 = loc1, loc2
                    spawn_point = carla.Transform()
                    spawn_point.location = loc1
                    destination_point = carla.Transform()
                    destination_point.location = loc2
                    distance = compute_distance(spawn_point.location, self.envSetup.observerSpawnTransform.location)
                    spawnAndDestinationPoints_extended.append((spawn_point, destination_point, distance))

        # Sort the points depending on their distance to playerSpawnTransform
        spawnAndDestinationPoints_extended = sorted(spawnAndDestinationPoints_extended, key = lambda SpawnAndDestTransform : SpawnAndDestTransform[2])

        if len(spawnAndDestinationPoints_extended) > 0:
            # Now select points that are Xm depart from each other
            spawnAndDestinationPoints = [spawnAndDestinationPoints_extended[0]]
            unselected_points = []
            for pIndex in range(1, len(spawnAndDestinationPoints_extended)):
                potential_point = spawnAndDestinationPoints_extended[pIndex]
                shortedDistToAnySelected = math.inf
                for selectedPoint, destPoint, distToObserverpos in spawnAndDestinationPoints:
                    distToThisSelPoint = compute_distance(potential_point[0].location, selectedPoint.location)
                    if distToThisSelPoint < shortedDistToAnySelected:
                        shortedDistToAnySelected = distToThisSelPoint

                if shortedDistToAnySelected > self.envSetup.PedestriansDistanceBetweenSpawnpoints:
                    spawnAndDestinationPoints.append(potential_point)
                else:
                    unselected_points.append(potential_point)

                # Selecting enough, so leaving
                if len(spawnAndDestinationPoints) >= self.envSetup.NumberOfCarlaPedestrians:
                    break

            # Didn't complete the list with the filter above ? just chose some random points
            diffNeeded = self.envSetup.NumberOfCarlaPedestrians - len(spawnAndDestinationPoints)
            if diffNeeded > 0:
                random.shuffle(unselected_points)
                numPointsToAppendExtra = min(diffNeeded, len(unselected_points))
                if numPointsToAppendExtra > 0:
                    spawnAndDestinationPoints.extend(unselected_points[:numPointsToAppendExtra])

            spawnAndDestinationPoints = spawnAndDestinationPoints[:self.envSetup.NumberOfCarlaPedestrians]

            # Destination points are from the same set, but we shuffle them
            #destination_points = spawnAndDestinationPoints
            #random.shuffle(destination_points)


        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        target_points = []
        for spawn_point, target_point, distToObserverpos in spawnAndDestinationPoints:
            walker_bp = random.choice(blueprints_walkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                maxRunningSpeed = float(walker_bp.get_attribute('speed').recommended_values[2])
                maxWalkingSpeed = float(walker_bp.get_attribute('speed').recommended_values[1])
                minRunningSpeed = maxWalkingSpeed
                minWalkingSpeed = max(1.2, maxWalkingSpeed * 0.5)

                outSpeed = maxWalkingSpeed
                if random.random() > percentagePedestriansRunning:
                    # walking
                    outSpeed = minWalkingSpeed + np.random.rand() * (maxWalkingSpeed - minWalkingSpeed)
                else:
                    # running
                    outSpeed = minRunningSpeed + np.random.rand() * (maxRunningSpeed - minRunningSpeed)

                walker_speed.append(outSpeed)
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            target_points.append(target_point)
            batch.append(SpawnActorFunctor(walker_bp, spawn_point))
        results = self.carlaConnectionManager.client.apply_batch_sync(batch, True)

        # Store from walker speeds and target points only those that succeeded
        walker_speed2 = []
        target_points2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
                target_points2.append(target_points[i])
        walker_speed = walker_speed2
        target_point = target_points2

        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActorFunctor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = self.carlaConnectionManager.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            self.s_all_pedestrian_ids.append(walkers_list[i]["con"])
            self.s_all_pedestrian_ids.append(walkers_list[i]["id"])
        all_pedestrian_actors = self.world.get_actors(self.s_all_pedestrian_ids)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.s_all_pedestrian_ids), 2):
            # start walker
            all_pedestrian_actors[i].start()
            # set walk to random point
            all_pedestrian_actors[i].go_to_location(target_points[int(i/2)].location)
            # max speed
            all_pedestrian_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        logging.log(logging.INFO, 'Spawned %d vehicles and %d walkers', len(self.s_vehicles_list), len(walkers_list))

        # Wait to have all things spawned on server side
        time.sleep(SYNC_TIME)
        self.world.tick()

        # Set auto pilot for vehicles spawned
        for v in self.s_vehicles_list:
            v.set_autopilot(True)

        logging.log(logging.INFO, 'Setting some random weather and traffic management...')
        # Now set the weather
        weather_id = np.random.choice(len(self.s_weather_presets))
        preset = self.s_weather_presets[weather_id]
        self.world.set_weather(preset[0])

        self.carlaConnectionManager.setupTrafficManager()

        time.sleep(SYNC_TIME)
        self.world.tick()


    # Some operation that are done after updating the environment and data gathering stuff
    def doPostUpdate(self):
        # Choose the next waypoint and update the car location if car perspective is used
        if self.dataGatherParams.useHeroCarPerspective:
            if not DataGatherParams.STATIC_CAR:
                self.currWaypoint = random.choice(self.currWaypoint.next(1.5))
                self.observerVehicle.set_transform(self.currWaypoint.transform)
            else:
                # Apply brake
                self.observerVehicle.apply_control(carla.VehicleControl(hand_brake=True))

    # Given a list of actors, write a dictionary for each frame and actor id, the BBoxMinMax and velocity
    def addFrameData_internal(self, listOfActors, outputDS):
        for actor in listOfActors:
            assert isinstance(actor, carla.Walker) or isinstance(actor, carla.Vehicle)
            actorId = actor.id
            actorTransform = actor.get_transform()
            actorLocation = actor.get_location()
            actorVelocity = actor.get_velocity()
            actorVelocity = np.array([actorVelocity.x, actorVelocity.y, actorVelocity.z])

            # Returns as [4 x 8], x,y,z1 for each of the 8 points. So all X are on row 0, Y on row 1, Z on row 2
            actorWorldBBox = getActorWorldBBox(actor)

            xMin = np.min(actorWorldBBox[0, :])
            xMax = np.max(actorWorldBBox[0, :])
            yMin = np.min(actorWorldBBox[1, :])
            yMax = np.max(actorWorldBBox[1, :])
            zMin = np.min(actorWorldBBox[2, :])
            zMax = np.max(actorWorldBBox[2, :])
            bboxMinMax = np.array([[xMin, xMax], [yMin, yMax], [zMin, zMax]])

            assert actorId not in outputDS
            # Fill the data for this actor
            actorData = {'BBMinMax' : bboxMinMax, 'Velocity':actorVelocity}
            outputDS[actorId] = actorData

    # Given world data and where to write output for a single frame, do a snapshot of the world there
    def addFrameData(self, frameId, worldFrame, out_vehicles_data, out_pedestrians_data):
        assert frameId not in out_pedestrians_data
        assert frameId not in out_vehicles_data

        out_vehicles_data[frameId] = {}
        out_pedestrians_data[frameId] = {}

        # Iterate over walkers and get their
        # DO NOT CACHE THESE BECAUSE THEY CAN MODIFY AT RUNTIME
        allWalkerActorsIds = [self.s_all_pedestrian_ids[walkerId] for walkerId in range(1, len(self.s_all_pedestrian_ids), 2)]
        allVehicleActors = [vehicle for vehicle in self.s_vehicles_list]
        allWalkerActors = self.world.get_actors(allWalkerActorsIds)

        self.addFrameData_internal(allWalkerActors, out_pedestrians_data[frameId])
        self.addFrameData_internal(allVehicleActors, out_vehicles_data[frameId])

    # Returns the vehicles and pedestrians data for the given frame
    def getFrameData(self, frameId):
        return self.vehicles_data[frameId], self.pedestrians_data[frameId]

    # Simulate an environment frame and returns a tuple of A. actors frame data B. syncData dictionary containing registered sensors values
    def simulateFrame(self, simFrame):
        # Output statistics to see where we are
        tenthNumFrames = (self.dataGatherParams.numFrames / 10) if self.dataGatherParams.numFrames > 0 else None
        if tenthNumFrames and simFrame % tenthNumFrames == 0:
            print(f"{(simFrame * 10.0) / tenthNumFrames}%...")

        # Tick the  world
        worldFrame = self.world.tick()

        # Now take the actors and update the data and add the date for this frame
        self.addFrameData(simFrame, worldFrame, self.vehicles_data, self.pedestrians_data)

        # Advance the simulation and wait for the data.
        # logging.log(logging.INFO, f"Getting data for frame {worldFrame}")
        syncData = self.dataManager.tick(targetFrame=worldFrame, timeout=None)  # self.EnvSettings.TIMEOUT_VALUE * 100.0) # Because sometimes you forget to put the focus on server and BOOM
        # logging.log(logging.INFO, f"Data retrieved for frame {worldFrame}")

        return self.getFrameData(simFrame), syncData

    def saveSimulatedData(self):
        self.dataGatherParams.saveSimulatedData(self.vehicles_data, self.pedestrians_data)

    """
    def onConnectionSolved(self):
        self.world = self.carlaConnectionManager.world
    """

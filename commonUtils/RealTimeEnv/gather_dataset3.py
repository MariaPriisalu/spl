from __future__ import print_function
from RealTimeCarlaUtils import *
from CarlaWorldManagement import *
import json

class DataCollector(object):
    def __init__(self,):
        pass

    def collectData(self, host, port,
                    numCarlaVehicles, numCarlaPedestrians,
                    spawnHeroCarPerspective, numEpisodesPerMap,
                    outEpisodesBasePath, framesPerEpisodes, scenesConfigFile,
                    args=None):

        scenesToCapture = []
        # Parse the observers transforms
        assert os.path.exists(scenesConfigFile), "The file with observer transform positions doesn' exists !"
        with open(scenesConfigFile, 'r') as scenesConfigStream:
            data = json.load(scenesConfigStream)
            for sceneName, sceneData in data['scenes'].items():
                mapName = sceneData['map']
                x=sceneData['X']
                y=sceneData['X']
                yaw=sceneData['yaw']*np.pi/180.0
                x=x+(20*np.cos(yaw))
                y=y+(20*np.sin(yaw))
                print (" Previous location x: "+str(sceneData['X'])+" y: "+str(sceneData['Y'])+" yaw: "+str(sceneData['yaw']))
                print (" New position x:"+str(x)+" y: "+str(y)+" diff x"+str((20*np.cos(yaw)))+" y: "+str((20*np.sin(yaw))))
                location = carla.Location(x, y, sceneData['Z'])
                rotation = carla.Rotation(sceneData['pitch'], 0, sceneData['roll'])
                observerSpawnTransform = carla.Transform(location, rotation)
                voxelRes = sceneData['voxelRes']
                voxelsX = sceneData['voxelsX']
                voxelsY = sceneData['voxelsY']
                voxelsZ = sceneData['voxelsZ']

                parsedSceneData = {'sceneName' : sceneName,
                                    'map' : mapName, 'observerSpawnTransform' : observerSpawnTransform,
                                    'voxelRes' : voxelRes, 'voxelsX' : voxelsX, 'voxelsY' : voxelsY, 'voxelsZ' : voxelsZ}

                scenesToCapture.append(parsedSceneData)

        # For each scene position, do a data capture
        for sceneData in scenesToCapture:
            mapToTestName=sceneData['map']
            sceneName = sceneData['sceneName']
            print("@@@@@@@ =========== Capturing Scene: ", sceneData, "\n\n")
            envParams = EnvSetupParams(controlledCarsParams=[],
                                       controlledPedestriansParams=[],
                                       NumberOfCarlaVehicles=numCarlaVehicles,
                                       NumberOfCarlaPedestrians=numCarlaPedestrians,
                                       observerSpawnTransform=sceneData['observerSpawnTransform'],
                                        observerVoxelSize=sceneData['voxelRes'],
                                       observerNumVoxelsX=sceneData['voxelsX'],
                                       observerNumVoxelsY=sceneData['voxelsY'],
                                       observerNumVoxelsZ=sceneData['voxelsZ'],
                                       forceExistingRaycastActor = args.forceExistingRaycastActor,
                                       mapToUse=[mapToTestName],
                                       sceneName=sceneName)
            logging.log(logging.INFO, "Create data gathering parameters")
            dataGatheringParams = DataGatherParams(outputEpisodeDataPath=outEpisodesBasePath,
                                                   sceneName=sceneName,
                                                   useHeroCarPerspective=spawnHeroCarPerspective,
                                                   episodeIndex=-1,
                                                   numFrames=framesPerEpisodes,
                                                   maxNumberOfEpisodes=numEpisodesPerMap,
                                                   mapsToTest=[mapToTestName],
                                                   copyScenePaths=True, # DO NOT USE TRUE for real time env !!
                                                   host=host,
                                                   port=port,
                                                   args=args)
            logging.log(logging.INFO, "Set Render options")
            renderParams = RenderOptions(RenderType.RENDER_SIMPLIFIED if args.no_client_rendering == False else RenderType.RENDER_NONE, topViewResX=1024, topViewResY=1024)
            logging.log(logging.INFO, "Environment management")
            envManagement = EnvironmentManagement()

            # Sanity checks
            envParams.mapToUse = mapToTestName

            #dataGatheringParams.prepareSceneOutputFolders(sceneDataPath=outputScenePath)

            # Do the episodes. We cycle through the spawn points if not enough
            for episodeIndex in range(numEpisodesPerMap):
                envParams.episodeIndex = episodeIndex
                logging.log(logging.INFO, f"Preparing episode {episodeIndex} on map {mapToTestName}\n=================")
                logging.log(logging.INFO, "Spawning world")

                # Collect scene raycast/pointcloud data only on the first episode
                dataGatheringParams.collectSceneData = 1 if episodeIndex == 0 else 0

                envManagement.SpawnWorld(dataGatherSetup=dataGatheringParams, envSetup=envParams, renderOptions=renderParams, args=args)
                logging.log(logging.INFO, "Donw qith spawn world")
                try:
                    #dataGatheringParams.prepareEpisodeOutputFolders(dataGatheringParams.outputEpisodeDataPath) # The parameter will be set at spawn time inside SpawnWorld since it depends on the spawn index

                    # Simulate frame by frame
                    for frameId in range(framesPerEpisodes):
                        envManagement.SimulateFrame()
                    logging.log(logging.INFO, "Save simulated")
                    envManagement.SaveSimulatedData()
                    logging.log(logging.INFO, "Despawn world")
                    envManagement.DespawnWorld()
                    logging.log(logging.INFO, "Done")
                except:
                    envManagement.DespawnWorld()

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
        '-pedcount', '--numCarlaPedestrians',
        metavar='P',
        default=50,
        type=int,
        help='NUmber of carla pedestrians on the sim episode')

    argparser.add_argument(
        '-vehcount', '--numCarlaVehicles',
        metavar='P',
        default=20,
        type=int,
        help='NUmber of carla vehicles on the sim episode')

    argparser.add_argument(
        '-numEpisodesPerMap', '--numEpisodesPerMap',
        metavar='P',
        default=3,
        type=int,

        help='Number of episodes to gather for each map')#required=True,

    argparser.add_argument(
        '-numFramesPerEpisode', '--numFramesPerEpisode',
        metavar='P',
        default=2000,
        type=int,

        help='Number of frames per episode')#required=True,

    argparser.add_argument(
        '--topviewRes',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        type=str,
        help='window resolution (default: 1280x720)')

    argparser.add_argument(
        '-outputDataBasePath', '--outputDataBasePath',
        metavar='P',
        default='Dataset/carla/testing',
        type=str,

        help='Number of frames per episode')#required=True,

    """
    argparser.add_argument(
        '-listOfMapsToTest', '--listOfMapsToTest',
        metavar='P',
        type=str,
        required=True,
        help='Number of frames per episode')
    """

    argparser.add_argument(
        '-heroPerspectiveView', '--heroPerspectiveView',
        metavar='P',
        default=0,
        type=int,
        help='Should spawn hero car perspective data gathering ?')

    """
    argparser.add_argument(
        '-capturePointCloud', '--capturePointCloud',
        metavar='P',
        type=int,
        help='Should we capture the point cloud ?'
    )
    """

    argparser.add_argument(
        '-scenesConfigFile', '--scenesConfigFile',
        metavar='spawnObserverTransformsFile',
        type=str,
        default=str(None),
        help='Json config file for scenes'

    )
    argparser.add_argument(
        '-no_server_rendering', '--no_server_rendering',
        metavar='no_rendering',
        type=int,
        default=1,
        help="1 if you don't want rendering on the server side "
    )

    argparser.add_argument(
        '-no_client_rendering', '--no_client_rendering',
        metavar='no_client_rendering',
        type=int,
        default=1,
        help="1 if you don't want rendering on the server side "
    )

    argparser.add_argument(
        '-forceSceneReconstruction', '--forceSceneReconstruction',
        metavar='forceSceneReconstruction',
        type=int,
        default=0,
        help="1 if you don't want rendering on the server side "
    )

    argparser.add_argument(
        '-forceExistingRaycastActor', '--forceExistingRaycastActor',
        metavar='forceExistingRaycastActor',
        type=int,
        default=0,
        help="1 if you don't want rendering on the server side "
    )

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.topviewRes.split('x')]
    args.no_rendering = True if args.no_server_rendering == 1 else False
    args.no_client_rendering = True if args.no_client_rendering == 1 else False
    args.forceSceneReconstruction = int(args.forceSceneReconstruction)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    dc = DataCollector()
    dc.collectData(host=args.host, port=args.port,
                   numCarlaVehicles=args.numCarlaVehicles,
                   numCarlaPedestrians=args.numCarlaPedestrians,
                   spawnHeroCarPerspective=True if args.heroPerspectiveView == 1 else False,
                   numEpisodesPerMap=args.numEpisodesPerMap,
                   outEpisodesBasePath=args.outputDataBasePath,
                   framesPerEpisodes=args.numFramesPerEpisode,
                   scenesConfigFile=args.scenesConfigFile,
                   args=args)

    print('\nDone!')


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nClient stoped by user.')

from RealTimeCarlaUtils import *
import carla
import sys

# This is a manager for keeping conectivity tracking with Carla Server
class CarlaServerConnection:
    def __init__(self, parent, args):
        self.parent = parent
        self.actorsManager = None
        self.map = None
        self.world = None
        self.client = None
        self.traffic_manager = None
        self.TIMEOUT_VALUE = 1000000.0
        self.args = args

    # Connect to the carla server
    def connectToServer(self, host, port):
        # Connect with the server
        self.client = carla.Client(host, port)
        self.client.set_timeout(self.TIMEOUT_VALUE)
        self.availableMaps = self.client.get_available_maps()
        logging.log(logging.INFO, ("Available maps are: {0}").format(self.availableMaps))
        self.orig_settings = self.client.get_world().get_settings()

    def reloadWorld(self, envParams : EnvSetupParams):
        self.world =  self.client.get_world()
        if self.world is None or self.world.get_map() is None or self.world.get_map().name != envParams.mapToUse:
            self.client.load_world(envParams.mapToUse)
        else:
            self.client.reload_world()

        self.envParams = envParams
        # Set settings for this episode and reload the world
        settings = carla.WorldSettings(
            no_rendering_mode=self.args.no_rendering,
            synchronous_mode=envParams.synchronousComm,
            fixed_delta_seconds=1.0 / envParams.fixedFPS)
        # settings.randomize_seeds()
        self.world.apply_settings(settings)
        self.map = self.world.get_map()

    def releaseServerConnection(self):
        # Deactivate sync mode
        if self.world == None:
            return

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

    def disconnectFromServer(self, withError: bool):
        if withError:
            print("Unexpected error:", sys.exc_info()[0])
            tb = traceback.format_exc()
            print(tb)
        self.actorsManager.destroy_current_environment(self.client)
        logging.log(logging.INFO, 'Destroying the environment')
        self.releaseServerConnection()
        self.parent.renderContext.quit()
        if withError is True:
            sys.exit()

    def setupTrafficManager(self):
        # Set the traffic management stuff
        # NOTE: the issue in the past with traffic manager was that cars were not moving after the second episode
        # To that end why i did was to:
        # - increase the timeout value to 10s and check the outputs from TM
        # - destroy the client each time between episodes (i.e. having a script that handles data gathering and
        # connects each time with a new client.)
        self.traffic_manager = self.client.get_trafficmanager(EnvSetupParams.tm_port)
        self.traffic_manager.set_global_distance_to_leading_vehicle(EnvSetupParams.distanceBetweenVehiclesCenters)
        self.traffic_manager.set_synchronous_mode(self.envParams.synchronousComm)
        self.traffic_manager.global_percentage_speed_difference(EnvSetupParams.speedLimitExceedPercent)



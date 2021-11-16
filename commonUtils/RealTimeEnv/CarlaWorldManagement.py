# Utils for spawning a world sequence with different parameters

from RealTimeCarlaUtils import *
from CarlaServerConnection import *
from CarlaEnvironmentActors import  *
from CarlaEnvironmentRendering import *
import carla
import shutil


class EnvironmentManagement:
    # Spawns a new world with settings, either for the purpose of online/offline data gathering or real time simulation
    def __init__(self):
        self.dataGatherParams = None
        self.renderOptions = None
        self.envOptions = None

        # Context members for actors spawned
        self.actorsContext = None

        # Carla server connection context
        self.carlaConnection = None

        # Render context
        self.renderContext = None

        self.simFrame = None

    # Basically this function will be parametrized and the entry point for real time simulation and data gathering
    def SpawnWorld(self, dataGatherSetup : DataGatherParams, envSetup : EnvSetupParams, renderOptions: RenderOptions, args):
        # Step 0: set options and sanity checks
        self.dataGatherParams = dataGatherSetup
        self.renderOptions = renderOptions
        self.envSetup = envSetup

        # Step 1: create contexts, solve scene data gathering
        # Context members for actors spawned
        self.carlaConnection = CarlaServerConnection(self, args)
        try:
            self.actorsContext = EnvironmentActors(self, self.carlaConnection)
            self.carlaConnection.actorsManager = self.actorsContext

            self.renderContext = EnvironmentRendering(self.renderOptions, dataGatherSetup, args)

            # Step 2: create connection to server
            self.carlaConnection.connectToServer(self.dataGatherParams.host, self.dataGatherParams.port)

            # Step 3: spawn world as requested
            self.actorsContext.spawnActors(self.envSetup, dataGatherSetup)

            # Step 4: fix the 2D rendering if used
            loc = envSetup.observerSpawnTransform.location
            self.renderContext.setFixedObserverPos([loc.x, loc.y, loc.z])

        except:
            self.DespawnWorld(withError=True)

        self.simFrame = -1

    def SimulateFrame(self):
        lastFrameVehiclesData = None
        lastFramePedestriansData = None

        try:
            self.simFrame += 1
            #logging.log(logging.INFO, f"Simulating environment frame {self.simFrame}")
            frameActorsData, syncData = self.actorsContext.simulateFrame(self.simFrame)
            lastFrameVehiclesData, lastFramePedestriansData = frameActorsData

            # Take the date from world and send them to render side
            self.renderContext.tick(syncData)

            # Save the needed stuff from sensors
            self.dataGatherParams.saveHeroCarPerspectiveData(syncData, self.simFrame)

            self.actorsContext.doPostUpdate()
        except:
            self.DespawnWorld(withError=True)

        return (lastFrameVehiclesData, lastFramePedestriansData)

    # Save the actors positions etc during simulation of the environment within the episode output folder
    def SaveSimulatedData(self):
        self.vehicles_data = {}
        self.pedestrians_data = {}
        self.actorsContext.saveSimulatedData()

    def DespawnWorld(self, withError=False):
        self.carlaConnection.disconnectFromServer(withError=withError)
        self.renderContext.quit()

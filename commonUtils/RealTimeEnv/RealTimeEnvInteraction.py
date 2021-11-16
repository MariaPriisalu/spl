# This is for real time interaction with Carla/Unreal side.
# Work in progress but contains all hooks for interacting with the engine in a OpenAI gym compatible interface


from dotmap import DotMap
from . RealTimeNullEnvInteraction import NullRealTimeEnv
from . RealTimeCarEnv import *
from . RealTimePedEnv import *

class CarlaRealTimeEnv(NullRealTimeEnv):
	def __init__(self):
		super(NullRealTimeEnv, self).__init__()


		self.pedestrianAgents = [RealTimePedEnv()]
		self.carsAgents = [RealTimeCarEnv()]


	def getObservation(self):
		ObsDict = DotMap()
		ObsDict.pedestrians = {}
		ObsDict.cars = {}
		ObsDict.engineFrame = -1


	def getReward(self):
		return 0


	def isEnvFinished(self):
		return False


	def getDetails(self):
		return None

	def reset(self, initDict):
		obs = self.getObservation()
		return obs

	def action(self, inputActionDict):

		# Return output
		obs = self.getObservation()
		info = self.getDetails()
		done = self.isEnvFinished()
		reward = self.getReward()

		return obs, reward, done, info
		pass


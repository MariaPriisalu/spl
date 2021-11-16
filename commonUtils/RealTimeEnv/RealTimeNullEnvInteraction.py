
from dotmap import DotMap

class NullRealTimeEnv:
	def __init__(self):
		pass

	def reset(self, initDict):
		obs = None
		return obs

	def action(self, actionDict, training):
		obs = None
		reward = None
		done = None
		info = None
		return obs, reward, done, info
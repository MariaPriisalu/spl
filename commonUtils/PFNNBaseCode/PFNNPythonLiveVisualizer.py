import threading
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10
skeletonColor = "#3498db"

class PFNNLiveVisualizer:
	# Warning using TKK does very slow rendering but solves some compatibility issues
	def __init__(self, jointParents, poseData, getBonePosFunc, useTKAgg = False, skeletonColor = "#3498db"):
		if useTKAgg == True:
			mpl.use("TkAgg")

		self.numBones = len(jointParents)
		self.jointsParents = jointParents
		self.getBonePosFunc = getBonePosFunc

        # We are caching the 3D lines and update them on each tick.
		# This is done to optimizer rendering speed
		self.werePlotsCached = False
		self.cached_3DLines = []
		self.skeletonColor = skeletonColor

		self.cached_3DLines = [None] * self.numBones

		# Create the 3D projection and axes
		self.fig = plt.figure()
		self.ax = self.fig.gca(projection='3d')
		self.ax.legend()
		axesViewSize = 100
		xNegLimit = -50
		yNegLimit = -50
		zNegLimit = 0
		self.ax.set_xlim3d([xNegLimit, xNegLimit + axesViewSize])
		self.ax.set_zlim3d([zNegLimit, zNegLimit + axesViewSize])
		self.ax.set_ylim3d([yNegLimit, yNegLimit + axesViewSize])

		# Sometimes you might need this..
		#plt.ion()

	# Draws a line between p1 and p2 points. If this lines was already instanced, just reuse it from the cache
	# If caching the line for the first time, return it so we can cache the result and reuse the instance
	def drawLine(self, p1, p2, cachedLine, color):
		x_values = [p1[0], p2[0]]
		y_values = [p1[1], p2[1]]
		z_values = [p1[2], p2[2]]

		res = None
		if cachedLine is None:
			res = plt.plot(x_values, y_values, z_values, c=color)[0]
		else:
			cachedLine.set_xdata(x_values)
			cachedLine.set_ydata(y_values)
			cachedLine.set_3d_properties(z_values)

		return res

	# Note that in matlab Z is up, while in our simulation Y is up, thath's why we send as parameter to this function the Z offset but we interpret it as Y !
	def updateSkeletonGraph(self, offset_center_X, offset_center_Y, poseData):
		global werePlotsCached
		global cached_plots

		for boneID in range(self.numBones):
			parentBoneID = self.jointsParents[boneID]
			if parentBoneID == -1:
				continue

			# Get bone in world coordinates
			x0, z0, y0 = self.getBonePosFunc(poseData, boneID)
			x1, z1, y1 = self.getBonePosFunc(poseData, parentBoneID)

			# Translate them to local simulation using the given center
			x0 -= offset_center_X
			x1 -= offset_center_X
			y0 -= offset_center_Y
			y1 -= offset_center_Y
			P1 = [x0, y0, z0]
			P2 = [x1, y1, z1]

			res = None
			if not self.werePlotsCached:
				res = self.drawLine(P1, P2, None, self.skeletonColor)
				self.cached_3DLines[boneID] = res
			else:
				self.drawLine(P1, P2, self.cached_3DLines[boneID], self.skeletonColor)

		plt.show(block=False)
		self.fig.canvas.draw()
		plt.pause(0.000001)  # Note: This was added to allow GUI thread process something too such that the matplotlib is ticked. We can do them in different threads if pause is an issue..

		self.werePlotsCached = True

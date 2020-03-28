
import matplotlib.pyplot as plt
import numpy as np
from gw import oneD2twoD, twoD2oneD

def plotWorld(world, plotNow=True):
	
	r,c = world.shape
	plt.figure(figsize=(c,r))

	gridMat = np.zeros(world.shape)
	for i,j in world.obstacle2D:
		gridMat[i,j] = 1.0
	for i,j in world.terminal2D:
		gridMat[i,j] = 0.2
	plt.pcolor(gridMat, edgecolors='k', linewidths=2, cmap='binary', vmin=0, vmax=1)

	U = np.zeros((r,c))
	V = np.zeros((r,c))
	U[:] = np.nan
	V[:] = np.nan
	for (a,b) in world.jump.keys():
		(a2,b2) = world.jump[(a,b)]
		U[a,b] = (b2-b)
		V[a,b] = (a-a2)

	C,R = np.meshgrid(np.arange(0,c)+0.5, np.arange(0,r)+0.5)
	plt.quiver(C, R, U, V, scale=1, units='xy')

	plt.xticks(np.arange(c)+0.5,np.arange(c))
	plt.yticks(np.arange(r)+0.5,np.arange(r))
	plt.gca().invert_yaxis()
	plt.title('Grid World')
	if plotNow:
		plt.show()

def plotGrid(fig, r, c):

	ax = fig.add_subplot(111)
	for i in range(c-1):
		ax.axvline(i+0.5, color='k')
	for i in range(r-1):
		ax.axhline(i+0.5, color='k')
	ax.set_xlim(-0.5, c - 0.5)
	ax.set_ylim(-0.5, r - 0.5)
	ax.set_aspect('equal')

def plotStateValue(data, world, plotNow=True):
	"""
	Args:
		data (numpy array): matrix that holds the state-values for all states in the gridworld
			should be the same size as world.shape
		world (object): GridWorld instance
		plotNow (bool): if true, will call plot.show()
	"""

	if not (data.shape == world.shape):
		data = np.reshape(data,world.shape)

	r,c = data.shape
	plt.figure(figsize=(c+1,r))

	obstMat = np.empty((r,c))
	obstMat[:] = np.nan
	for i,j in world.obstacle2D:
		obstMat[i,j] = 1.0
	for i,j in world.terminal2D:
		obstMat[i,j] = 0.2
	obstMat = np.ma.masked_invalid(obstMat)
	f = plt.pcolor(data, edgecolors='k', linewidths=2, cmap='RdYlGn', vmin=np.min(data), vmax=np.max(data))
	plt.colorbar()
	plt.pcolor(obstMat, edgecolors='k', linewidths=2, cmap='binary', vmin=0.0, vmax=1.0)

	def show_values(pc, fmt="%.2f", **kw):
	    pc.update_scalarmappable()
	    ax = pc.axes
	    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
	        x, y = p.vertices[:-2, :].mean(0)
	        if np.all(color[:3] > 0.5):
	            color = (0.0, 0.0, 0.0)
	        else:
	            color = (1.0, 1.0, 1.0)
	        if (y-0.5,x-0.5) in world.terminal2D:
	        	ax.text(x, y, "T", ha="center", va="center", color=(0.0,0.0,0.0), **kw)
	        elif (y-0.5,x-0.5) not in world.obstacle2D:
	        	ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)
	show_values(f)

	plt.xticks(np.arange(c)+0.5,np.arange(c))
	plt.yticks(np.arange(r)+0.5,np.arange(r))
	plt.gca().invert_yaxis()
	plt.title('State Value Function (V)')
	
	if plotNow:
		plt.show()

def plotStateActionValue(data, world, plotNow=True):
	"""
	Args:
		data (numpy array): matrix that holds the state-action-values for all states in the gridworld
			should be the same size as world.nstates x 5
		world (object): GridWorld instance
		plotNow (bool): if true, will call plot.show()
	"""

	# plot grid
	r,c = world.shape
	fig = plt.figure(figsize=(c+1,r))
	ax = fig.add_subplot(111)
	for i in range(0,3*c,3):
		ax.axvline(i, color='k')
	for i in range(0,3*r,3):
		ax.axhline(i, color='k')
	ax.set_xlim(0, 3*c)
	ax.set_ylim(0, 3*r)

	# plot q values
	qshape = (3*world.shape[0],3*world.shape[1])
	qmask = np.empty(qshape)
	qmask[:] = np.nan
	obstmask = np.empty(qshape)
	obstmask[:] = np.nan

	for i in range(data.shape[0]):
		(r,c) = oneD2twoD(i,world.shape)
		(r,c) = (int(r),int(c))
		if (r,c) in world.obstacle2D:
			obstmask[3*r:3*r+3,3*c:3*c+3] = 1.0
		elif (r,c) in world.terminal2D:
			obstmask[3*r:3*r+3,3*c:3*c+3] = 0.2
		else:
			if (r,c) in world.jump.keys():
				qmask[3*r+1,3*c+1] = data[i,4]  # jump
			else:
				if c > 0 and (r,c-1) not in world.obstacle2D:
					qmask[3*r+1,3*c] = data[i,3]  	# left
				if c < world.shape[1]-1 and (r,c+1) not in world.obstacle2D:
					qmask[3*r+1,3*c+2] = data[i,2]  # right
				if r > 0 and (r-1,c) not in world.obstacle2D:
					qmask[3*r,3*c+1] = data[i,1]  	# up
				if r < world.shape[0]-1 and (r+1,c) not in world.obstacle2D:
					qmask[3*r+2,3*c+1] = data[i,0]  # down
	qmask = np.ma.masked_invalid(qmask)
	obstmask = np.ma.masked_invalid(obstmask)

	plt.pcolor(qmask, cmap='RdYlGn', vmin=np.min(data), vmax=np.max(data))
	plt.colorbar()
	plt.pcolor(obstmask, cmap='binary', vmin=0, vmax=1)

	plt.xticks(np.arange(1.5,3*c,3),np.arange(c))
	plt.yticks(np.arange(1.5,3*r,3),np.arange(r))
	plt.gca().invert_yaxis()
	plt.title('State Action Value Function (Q)')
	if plotNow:
		plt.show()

def plotPolicyPi(P, world, plotNow=True):
	"""
	Args:
		P (numpy array): matrix that holds the policy of the agent (P^pi)
		world (object): GridWorld instance
		plotNow (bool): if true, will call plot.show()
	"""

	r,c = world.shape
	C,R = np.meshgrid(np.arange(0,c), np.arange(0,r))
	fig = plt.figure(figsize=(c+1,r))
	plotGrid(fig, r, c)

	# loop through all states to find direction for each one
	U = np.zeros((r,c))
	V = np.zeros((r,c))
	for i in range(P.shape[0]):
		(a,b) = oneD2twoD(i, world.shape)
		(a,b) = (int(a),int(b))
		inds = np.nonzero(P[i,:])[0]
		if len(inds)>0 and i not in world.terminal and i not in world.obstacle:
			for j in inds:
				if (j == i+1):  	# right
					U[a,b] = 1
				elif (j == i-1):  	# left 
					U[a,b] = -1
				elif (j == i+c):  	# down
					V[a,b] = -1
				elif (j == i-c):  	# up
					V[a,b] = 1
				else:				# jump
					(x,y) = oneD2twoD(j, world.shape)
					U[a,b] = 2*(y-b)
					V[a,b] = 2*(a-x)
		else:
			U[a,b] = np.nan
			V[a,b] = np.nan
		
	plt.quiver(C, R, U, V, scale=2, units='xy')

	# plot black squares for obstacles
	C2,R2 = np.meshgrid(np.arange(0,c+1), np.arange(0,r+1))
	obstMat = np.empty((r+1,c+1))
	obstMat[:] = np.nan
	for i,j in world.obstacle2D:
		obstMat[i,j] = 1.0
	for i,j in world.terminal2D:
		obstMat[i,j] = 0.2
	obstMatM = np.ma.masked_invalid(obstMat)
	plt.pcolormesh(C2-0.5,R2-0.5,obstMatM,cmap='binary',vmin=0,vmax=1)

	plt.xticks(np.arange(c),np.arange(c))
	plt.yticks(np.arange(r),np.arange(r))
	plt.gca().invert_yaxis()
	plt.title('Policy')
	if plotNow:
		plt.show()

def plotGreedyPolicyQ(Q, world, plotNow=True):
	"""
	Args:
		Q (numpy array): matrix that holds the state-action-values for all states in the gridworld
			should be the same size as world.nstates x 5
		world (object): GridWorld instance
		plotNow (bool): if true, will call plot.show()
	"""

	r,c = world.shape
	C,R = np.meshgrid(np.arange(0,c), np.arange(0,r))
	fig = plt.figure()
	plotGrid(fig, r, c)

	# loop through all states to find direction for each one
	actionlist = np.array(['D','U','R','L','J'])
	U = np.zeros((r,c))
	V = np.zeros((r,c))
	for i in range(Q.shape[0]):
		(a,b) = oneD2twoD(i, world.shape)
		(a,b) = (int(a),int(b))
		if i not in world.terminal and i not in world.obstacle:
			world.set_state(i)
			available_actions = actionlist[world.get_actions()]
			action = available_actions[np.argmax(Q[i,world.get_actions()])]
			if (action == 'D'):
				V[a,b] = -1
			elif (action == 'U'):
				V[a,b] = 1
			elif (action == 'R'):
				U[a,b] = 1
			elif (action == 'L'):
				U[a,b] = -1
			else:
				(x,y) = world.jump[(a,b)]
				U[a,b] = 1.5*(y-b)
				V[a,b] = 1.5*(a-x)
		else:
			U[a,b] = np.nan
			V[a,b] = np.nan
	plt.quiver(C, R, U, V, scale=2, units='xy')

	# plot black squares for obstacles
	C2,R2 = np.meshgrid(np.arange(0,c+1), np.arange(0,r+1))
	obstMat = np.empty((r+1,c+1))
	obstMat[:] = np.nan
	for i,j in world.obstacle2D:
		obstMat[i,j] = 1.0
	for i,j in world.terminal2D:
		obstMat[i,j] = 0.2
	obstMatM = np.ma.masked_invalid(obstMat)
	plt.pcolormesh(C2-0.5,R2-0.5,obstMatM,cmap='binary',vmin=0,vmax=1)

	plt.gca().invert_yaxis()
	if plotNow:
		plt.show()

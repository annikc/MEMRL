'''
Set up functions for plotting gridworld environment
TODO: Interactive policy plotting

Author: Annik Carson
-- Oct 2019
'''

# =====================================
#           IMPORT MODULES            #
# =====================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.cm as cmx
import matplotlib.patches as patches

from os import listdir
from os.path import isfile, join
from scipy.stats import entropy
import imageio

# =====================================
#              FUNCTIONS
# =====================================

def running_mean(x, N):
	cumsum = np.cumsum(np.insert(x, 0, 0))
	return (cumsum[N:] - cumsum[:-N]) / float(N)

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def softmax(x,T=1):
	 """Compute softmax values for each sets of scores in x."""
	 e_x = np.exp((x - np.max(x))/T)
	 return e_x / e_x.sum(axis=0) # only difference

def make_arrows(action, probability):
	'''
	:param action:
	:param probability:
	:return:
	'''
	if probability == 0:
		dx, dy = 0, 0
		head_w, head_l = 0, 0
	else:
		dxdy = [(0.0, 0.25),  #D
				(0.0, -0.25), #U
				(0.25, 0.0),  #R
				(-0.25, 0.0), #L
				(0.1,-0.1), # points right and up #J
				(-0.1,0.1), # points left and down # P
				]
		dx,dy = dxdy[action]

		head_w, head_l = 0.1, 0.1

	return dx, dy, head_w, head_l

# =====================================
#          PLOTTING FUNCTIONS
# =====================================
def plot_softmax(x, T=1):
	f, axarr = plt.subplots(2, sharex=True)
	axarr[0].bar(np.arange(len(x)), x)
	y = softmax(x, T)
	axarr[1].bar(np.arange(len(x)), y)
	plt.show()

def plot_env(maze, save=False):
	'''
	:param maze: the environment object
	:param save: bool. save figure in current directory
	:return: None
	'''
	fig  = plt.figure()
	axis = fig.add_axes([0.05, 0.05, .9, .9]) # [left, bottom, width, height]
	ax   = fig.gca()

	# plot basic grid
	axis.pcolor(maze.grid, cmap = 'bone', vmax =1, vmin = 0)

	# add patch for agent location (blue)
	agent_y, agent_x = maze.cur_state
	ax.add_patch(plt.Circle((agent_y + .5, agent_x + .5), 0.35, fc='b'))

	# add patch for reward location/s (red)
	for rwd_loc in maze.rwd_loc:
		rwd_y, rwd_x = rwd_loc
		#ax.add_patch(plt.Circle((rwd_y+.5, rwd_x+.5), 0.35, fc='r'))
		ax.add_patch(plt.Rectangle((rwd_y, rwd_x), width=1, height=1, linewidth=1, ec='white', fill=False))
	ax.set_aspect('equal')

	if save:
		plt.savefig('../data/figures/{}environment.svg'.format(maze.maze_type), format='svg', pad_inches=2)


def plot_valmap(maze, value_array, save=False, **kwargs):
	'''
	:param maze: the environment object
	:param value_array: array of state values
	:param save: bool. save figure in current directory
	:return: None
	'''
	show = kwargs.get('show', True)
	title = kwargs.get('title', 'State Value Estimates')
	directory = kwargs.get('directory', '../data/figures/')
	filetype = kwargs.get('filetype', 'png')
	rewards = kwargs.get('rwds', maze.rewards)
	vals = value_array.copy()
	fig = plt.figure(figsize=(7,5))
	ax1 = fig.add_axes([0, 0, 0.85, 0.85])
	axc = fig.add_axes([0.75, 0, 0.05, 0.85])
	vmin, vmax = kwargs.get('v_range', [0,1])
	cmap = plt.cm.Spectral_r
	cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
	for i in maze.obstacles_list:
		vals[i] = np.nan
	cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
	ax1.pcolor(vals, cmap=cmap, vmin = vmin, vmax = vmax)

	# add patch for reward location/s (red)
	for rwd_loc in rewards:
		rwd_r, rwd_c = rwd_loc
		ax1.add_patch(plt.Rectangle((rwd_c, rwd_r), width=0.99, height=1, linewidth=1, ec='white', fill=False))


	ax1.set_aspect('equal')
	ax1.invert_yaxis()
	ax1.set_title(title)

	if save:
		plt.savefig(f'{directory}{title}.{filetype}', format=f'{filetype}', bbox_inches='tight')
	if show:
		plt.show()

	plt.close()

def plot_polmap(maze, policy_array, save=False, **kwargs):
	'''
	:param maze: the environment object
	:param save: bool. save figure in current directory
	:return: None
	'''
	show = kwargs.get('show', True)
	title = kwargs.get('title', 'Most Likely Action from Policy')
	directory = kwargs.get('directory', '../data/figures/')
	filetype = kwargs.get('filetype', 'png')
	rewards = kwargs.get('rwds', maze.rewards)
	fig = plt.figure(figsize=(7,5))
	ax1 = fig.add_axes([0, 0, 0.85, 0.85])
	axc = fig.add_axes([0.75, 0, 0.05, 0.85])

	cmap = plt.cm.Spectral_r
	cNorm = colors.Normalize(vmin=0, vmax=1)
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

	# make base grid
	ax1.pcolor(maze.grid, vmin=0, vmax=1, cmap='bone')
	# add patch for reward location/s (red)
	for rwd_loc in rewards:
		rwd_r, rwd_c = rwd_loc
		ax1.add_patch(plt.Rectangle((rwd_c, rwd_r), width=0.99, height=1, linewidth=1, ec='white', fill=False))

	chance_threshold = kwargs.get('threshold',0.18)  #np.round(1 / len(maze.actionlist), 6)


	cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
	for i in range(maze.r):
		for j in range(maze.c):
			action = np.argmax(tuple(policy_array[i][j]))
			prob = max(policy_array[i][j])

			dx1, dy1, head_w, head_l = make_arrows(action, prob)
			if prob > chance_threshold:
				if (dx1, dy1) == (0, 0):
					pass
				else:
					colorVal1 = scalarMap.to_rgba(prob)
					ax1.arrow(j+0.5, i+0.5, dx1, dy1, head_width=0.3, head_length=0.2, color=colorVal1)
			else:
				pass
	ax1.set_aspect('equal')
	ax1.set_title(title)
	ax1.invert_yaxis()

	if save:
		plt.savefig(f'{directory}{title}.{filetype}', format=f'{filetype}', bbox_inches='tight')
	if show:
		plt.show()
	plt.close()

def plot_pref_pol(maze, policy_array, save=False, **kwargs):
	'''
		:param maze: the environment object
		:param save: bool. save figure in current directory
		:return: None
		'''
	show = kwargs.get('show', True)
	title = kwargs.get('title', 'Most Likely Action from Policy')
	directory = kwargs.get('directory', '../data/figures/')
	filetype = kwargs.get('filetype', 'png')
	vmax = kwargs.get('upperbound', 2)
	rewards = kwargs.get('rwds', maze.rewards)
	fig = plt.figure(figsize=(7, 5))
	ax1 = fig.add_axes([0, 0, 0.85, 0.85])
	axc = fig.add_axes([0.75, 0, 0.05, 0.85])

	cmap = plt.cm.Spectral_r
	cNorm = colors.Normalize(vmin=0, vmax=vmax)
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
	cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
	# make base grid
	ax1.pcolor(maze.grid, vmin=0, vmax=vmax, cmap='bone')
	# add patch for reward location/s (red)
	for rwd_loc in rewards:
		rwd_r, rwd_c = rwd_loc
		ax1.add_patch(plt.Rectangle((rwd_c, rwd_r), width=0.99, height=1, linewidth=1, ec='white', fill=False))

	chance_threshold = kwargs.get('threshold', 0.18)  # np.round(1 / len(maze.actionlist), 6)

	for i in range(maze.r):
		for j in range(maze.c):
			policy = tuple(policy_array[i, j])

			dx, dy = 0.0, 0.0
			for ind, k in enumerate(policy):
				action = ind
				prob = k
				if i==1 and j==1:
					print(action, prob)
				if prob < 0.01:
					pass
				else:
					dx1, dy1, head_w, head_l = make_arrows(action, prob)
					dx += dx1*prob
					dy += dy1*prob
					if i == 1 and j == 1:
						print(ind, k, dx1, dy1)
			if dx ==0.0 and dy == 0.0:
				pass
			else:
				colorVal1 = scalarMap.to_rgba(entropy(policy))
				ax1.arrow(j + 0.5, i + 0.5, dx, dy, head_width=0.3, head_length=0.5, color=colorVal1)


	ax1.set_aspect('equal')
	ax1.set_title(title)
	ax1.invert_yaxis()

	if save:
		plt.savefig(f'{directory}{title}.{filetype}', format=f'{filetype}', bbox_inches='tight')
	if show:
		plt.show()
	plt.close()
def plot_optimal(maze, policy_array, save=False, **kwargs):
	'''
	:param maze: the environment object
	:param save: bool. save figure in current directory
	:return: None
	'''
	show = kwargs.get('show', True)
	title = kwargs.get('title', 'Most Likely Action from Policy')
	directory = kwargs.get('directory', '../data/figures/')
	filetype = kwargs.get('filetype', 'png')
	rewards = kwargs.get('rwds', maze.rewards)
	fig = plt.figure(figsize=(7,5))
	ax1 = fig.add_axes([0, 0, 0.85, 0.85])
	axc = fig.add_axes([0.75, 0, 0.05, 0.85])

	cmap = plt.cm.Spectral_r
	cNorm = colors.Normalize(vmin=0, vmax=1)
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
	cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
	# make base grid
	ax1.pcolor(maze.grid, vmin=0, vmax=1, cmap='bone')
	# add patch for reward location/s (red)
	for rwd_loc in rewards:
		rwd_r, rwd_c = rwd_loc
		ax1.add_patch(plt.Rectangle((rwd_c, rwd_r), width=0.99, height=1, linewidth=1, ec='white', fill=False))

	chance_threshold = kwargs.get('threshold',0.18)  #np.round(1 / len(maze.actionlist), 6)


	for i in range(maze.r):
		for j in range(maze.c):
			policy = tuple(policy_array[i,j])

			dx, dy = 0,0
			for ind, k in enumerate(policy):
				if i == 0 and j ==0:
					print(ind,k)
				action = ind
				prob = k
				if prob < 0.01:
					pass
				else:
					dx1, dy1, head_w, head_l = make_arrows(action, prob)
					dx += dx1
					dy += dy1
			colorVal1 = scalarMap.to_rgba(entropy(policy))
			ax1.arrow(j+0.5, i+0.5, dx/2, dy/2, head_width=0.25, head_length=0.5, color=colorVal1)

	ax1.set_aspect('equal')
	ax1.set_title(title)
	ax1.invert_yaxis()

	if save:
		plt.savefig(f'{directory}{title}.{filetype}', format=f'{filetype}', bbox_inches='tight')
	if show:
		plt.show()
	plt.close()

def plot_memory_estimates(maze, value_array, save=False, **kwargs):
	'''
	:param maze: the environment object
	:param value_array: array of state values
	:param save: bool. save figure in current directory
	:return: None
	'''
	show = kwargs.get('show', True)
	title = kwargs.get('title', 'State Value Estimates')
	directory = kwargs.get('directory', '../data/figures/')
	filetype = kwargs.get('filetype', 'png')
	rewards = kwargs.get('rwds', maze.rewards)


	vals = value_array.copy()



	fig = plt.figure(figsize=(7,5))
	ax1 = fig.add_axes([0, 0, 0.85, 0.85])
	axc = fig.add_axes([0.75, 0, 0.05, 0.85])
	vmin, vmax = kwargs.get('v_range', [0,1])
	cmap = plt.cm.Spectral_r
	cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
	for i in maze.obstacles_list:
		vals[i] = np.nan
	cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
	ax1.pcolor(vals, cmap=cmap, vmin = vmin, vmax = vmax)

	# add patch for reward location/s (red)
	for rwd_loc in rewards:
		rwd_r, rwd_c = rwd_loc
		ax1.add_patch(plt.Rectangle((rwd_c, rwd_r), width=0.99, height=1, linewidth=1, ec='black', fill=False))


	ax1.set_aspect('equal')
	ax1.invert_yaxis()
	ax1.set_title(title)

	if save:
		plt.savefig(f'{directory}{title}.{filetype}', format=f'{filetype}', bbox_inches='tight')
	if show:
		plt.show()

	plt.close()





############## junks
def make_arrows_old(action, probability):
	'''
	:param action:
	:param probability:
	:return:
	'''
	if probability == 0:
		dx, dy = 0, 0
		head_w, head_l = 0, 0
	else:
		if action == 0:  # N
			dx = 0
			dy = -.25
		elif action == 1:  # E
			dx = .25
			dy = 0
		elif action == 2:  # W
			dx = -.25
			dy = 0
		elif action == 3:  # S
			dx = 0
			dy = .25
		elif action == 4:  # stay
			dx = -.1
			dy = -.1
		elif action == 5:  # poke
			dx = .1
			dy = .1

		head_w, head_l = 0.1, 0.1

	return dx, dy, head_w, head_l

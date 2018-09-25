'''
Plot functions used for AC Agent in RL gridworld task
Author: Annik Carson 
-- June 2018
'''

# =====================================
#           IMPORT MODULES            #
# =====================================
from __future__ import division, print_function

import numpy as np
np.random.seed(12345)

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.cm as cmx
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

from sklearn.neighbors import NearestNeighbors

import scipy.stats as st

import os
from os import listdir
from os.path import isfile, join

import imageio



###################################
#================================
# Default Environment Parameters
#================================ 
height = 7
width = 7

mazetype = 'none'
portshift = 'none'

obs_rho = 0.0 #obstacle density

# specify number of place cells and 
# full width half max of gaussian distribution of pc sensitivity
place_cells = 850
fwhm = 0.25 # NB: place cell full width half max must be in (0,1)


##################################

class PlaceCells(object):
	def __init__(self, num_cells, grid, **kwargs):
		self.num_cells = num_cells
		self.gx, self.gy = grid.x, grid.y
		self.field_width = kwargs.get('fwhm', 0.25)
		
		self.x = np.random.uniform(0,1,(self.num_cells,))
		self.y = np.random.uniform(0,1,(self.num_cells,))
		
	def activity(self, position):
		x0 = (position[0]/self.gx)+float(1/(2*self.gx))
		y0 = (position[1]/self.gy)+float(1/(2*self.gy))
		
		pc_activities = (np.exp(-((self.x-x0)**2 + (self.y-y0)**2) / self.field_width**2))
		state_vec = pc_activities.round(decimals = 10)
		return np.reshape(state_vec, (1,len(state_vec)))

class gridworld(object): 
	def __init__(self, grid_size, **kwargs):
		self.y, self.x = grid_size
		self.rho = kwargs.get('rho', 0.0)
		self.bound = kwargs.get('walls', False)
		self.maze_type = kwargs.get('maze_type', 'none')
		if self.bound:
			g1, u1, o1 = self.grid_maker()
			self.grid, self.useable, self.obstacles = self.grid_walls(g1, u1, o1)
		else:
			self.grid, self.useable, self.obstacles = self.grid_maker()

		self.actionlist = kwargs.get('actionlist', ['N', 'E', 'W', 'S', 'stay', 'poke'])
		self.rwd_action = kwargs.get('rewarded_action', 'poke')

		if self.maze_type == 'tmaze':
			self.rwd_loc = [self.useable[0]]
			self.port_shift = kwargs.get('port_shift', 'none')
			
		if self.maze_type == 'triple_reward':
			self.rwd_loc = [(self.x-1, 0), (self.x-1, self.y-1), (0, self.y-1)]
			self.orig_rwd_loc = [(self.x-1, 0), (self.x-1, self.y-1), (0, self.y-1)]
			self.starter = kwargs.get('t_r_start', (0,0))

		else:
			if self.maze_type == 'none' and self.rho == 0.0:
				self.rwd_loc = [(np.random.randint(self.x-2), np.random.randint(self.y-2))]
			else:
				rwd_choice = np.random.choice(len(self.useable))
				self.rwd_loc = [self.useable[rwd_choice]]
			self.orig_rwd_loc = []
		#print(type(self.rwd_loc),self.rwd_loc)

		
		start_choice = np.random.choice(len(self.useable))
		self.start_loc = self.useable[start_choice]
		if self.maze_type=='triple_reward':
			self.start_loc = self.starter
		
		
		#self.num_placecells = kwargs.get('num_pc', 500)
		#self.fwhm = kwargs.get('pc_fwhm', 6)
		#self.pcs = PlaceCells(num_cells=self.num_placecells, grid=self,fwhm = self.fwhm)
		self.reset_env()

		self.empty_map = self.make_map(self.grid, False)
		#self.init_value_map = self.empty_map
		#self.init_policy_map = self.empty_map
	
	def grid_maker(self):
		''' 
		Default grid is empty -- all squares == 0
		If env_type given, set some squares to == 1 
				(agent cannot occupy these squares)
		In the case of the T maze, set all squares =1 
				and just rewrite which squares the agent may 
				occupy (i.e. grid = 0 at these points)
		'''

		env_types = ['none', 'bar','room','tmaze', 'triple_reward']
		if self.maze_type not in env_types:
			print("Environment Type '{0}' Not Recognized. \nOptions are: {1} \nDefault is Open Field (maze_type = 'none')".format(self.maze_type, env_types))

		grid = np.zeros((self.y,self.x), dtype=int)
				
		if self.maze_type == 'bar':
			self.rho = 0
			barheight = 3
			for i in range(self.x-2): 
				grid[barheight][i+1] = 1
		
		elif self.maze_type == 'room':
			self.rho = 0
			vwall = int(self.x/2)
			hwall = int(self.y/2)
			
			#make walls
			for i in range(self.x):
				grid[vwall][i] = 1
			for i in range(self.y): 
				grid[i][hwall] = 1
				
			# make doors
			grid[vwall][np.random.choice(np.arange(0,vwall))] = 0 
			grid[vwall][np.random.choice(np.arange(vwall+1, self.x))] = 0 
			
			grid[np.random.choice(np.arange(0,hwall))][hwall] = 0 
			grid[np.random.choice(np.arange(hwall+1, self.y))][hwall] = 0 
		
		elif self.maze_type == 'tmaze': 
			self.rho = 0
			self.possible_ports = []
			grid = np.ones((self.y, self.x), dtype=int)
			h1, v1 = int(self.x/2), 0
			if h1%2==0:
				for i in range(self.x):
					grid[v1][i] = 0
					if i == 0:
						self.possible_ports.append((i,v1))
					elif i == self.x-1:
						self.possible_ports.append((i,v1))
			else: 
				for i in range(self.x):
					grid[v1][i] = 0
					if i == 0:
						self.possible_ports.append((i,v1))
					elif i == self.x-1:
						self.possible_ports.append((i,v1))

			if self.y > int(self.x/2):
				for i in range(self.y):
					grid[i][h1] = 0
					if i == self.y-1:
						self.possible_ports.append((h1,i))
			else:
				for i in range(self.y):
					grid[i][h1] = 0
					if i == self.y-1:
						self.possible_ports.append((h1,i))

		# for env_type = none, can set randomized obstacles on the grid, specified by density rho
		if self.rho != 0:
			maze = np.vstack([[np.random.choice([0,1], p = [1-self.rho, self.rho]) for _ in range(self.x)] for _ in range(self.y)])
			grid = grid + maze

		# lists of tuples storing locations of open grid space and obstacles (unusable grid space)
		useable_grid = list(zip(np.where(grid==0)[1], np.where(grid==0)[0]))
		obstacles = list(zip(np.where(grid==1)[1], np.where(grid==1)[0]))
		
		return grid, useable_grid, obstacles

	def grid_walls(self, grid, useable_grid, obstacles):
		new_grid = np.ones((self.y+2, self.x+2))
		for i in xrange(self.y):
			for j in xrange(self.x):
				new_grid[i+1][j+1] = grid[i][j]
		new_useable = [tuple(np.add(x, (1,1))) for x in useable_grid]
		new_obstacle =[tuple(np.add(x, (1,1))) for x in obstacles]
		return new_grid, new_useable, new_obstacle
	
	def make_map(self, grid, pol=False):
		'''
		Set up a map for the agent to record its policy and value
			estimates as it navigates the grid
		'''
		if pol:
			pv_map = np.zeros(grid.shape, dtype=[('N', 'f8'), ('E', 'f8'),('W', 'f8'), ('S', 'f8'),('stay', 'f8'), ('poke', 'f8')])
			pv_map[grid == 1] = (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)

		else:
			pv_map = np.zeros(grid.shape)
			pv_map[grid == 1] = np.nan
		return pv_map
	
	def reset_env(self): 
		self.cur_state = self.start_loc
		self.last_action = 'NA'
		self.rwd = 0
		
		self.done = False
		self.reward_tally = {}
		#if self.maze_type == 'triple_reward':
		for i in self.orig_rwd_loc:
			self.reward_tally[i] = 0

	def start_trial(self):
		start_choice = np.random.choice(len(self.useable))
		self.start_loc = self.useable[start_choice]
		
		self.cur_state = self.start_loc
		self.last_action = 'NA'
		self.rwd = 0 
		
		self.done = False
		if self.maze_type == 'triple_reward':
			self.start_loc = self.starter
		self.reward_tally = {}
		for i in self.orig_rwd_loc:
			self.reward_tally[i] = 0
	
	def move(self, action):
		if action == 'N': 
			next_state = (self.cur_state[0], self.cur_state[1]-1)
			if next_state in self.useable: 
				self.cur_state = next_state
		elif action == 'E': 
			next_state = (self.cur_state[0]+1, self.cur_state[1])
			if next_state in self.useable: 
				self.cur_state = next_state
		elif action == 'W': 
			next_state = (self.cur_state[0]-1, self.cur_state[1])
			if next_state in self.useable: 
				self.cur_state = next_state
		elif action == 'S': 
			next_state = (self.cur_state[0], self.cur_state[1]+1)
			if next_state in self.useable: 
				self.cur_state = next_state
		
		self.get_reward(action)
		self.last_action = action
		return self.cur_state
	
	def get_reward(self, action):
		if (action == 'poke') & (self.cur_state in self.rwd_loc):
			self.rwd = 1
			self.done = True
			if self.maze_type == 'tmaze':
				if self.port_shift in ['equal', 'left', 'right']:
					self.shift_rwd(self.port_shift)
			self.reward_tally[self.cur_state] += 1

		else:
			self.rwd = 0
			self.done = False

	def shift_rwd(self,shift):
		port_rwd_probabilities = [0.333, 0.333, 0.333]
		current_rewarded_port = self.rwd_loc[0]

		if shift == 'equal':
			dir_prob = [0.5, 0.5]

		elif shift == 'left':
			dir_prob = [0.95, 0.05]

		elif shift == 'right':
			dir_prob = [0.05, 0.95]

		if self.rwd_loc[0] in self.possible_ports:
			poked_port = self.possible_ports.index(self.rwd_loc[0])
			right_port = (self.possible_ports.index(self.rwd_loc[0])+1)%3
			left_port = (self.possible_ports.index(self.rwd_loc[0])+2)%3

			port_rwd_probabilities[poked_port] = 0
			port_rwd_probabilities[left_port] = dir_prob[0]*port_rwd_probabilities[left_port]
			port_rwd_probabilities[right_port] = dir_prob[1]*port_rwd_probabilities[right_port]

			port_rwd_probabilities = [rp/sum(port_rwd_probabilities) for rp in port_rwd_probabilities]
			new_rwd_loc = np.random.choice(3,1,p=port_rwd_probabilities)
			self.rwd_loc = [self.possible_ports[new_rwd_loc[0]]]
		else:
			print('is not works good')


		


class action_wrapper(object): 
	def __init__(self, actionlist):
		self.n = len(actionlist)
		self.actionlist = actionlist
		
class gymworld(object):
	def __init__(self, gridworld):
		self.env = gridworld
		self.action_space = action_wrapper(self.env.actionlist)
		self.state = self.env.cur_state
		self.observation_space = self.env.cur_state[0]
		self.reward = self.env.rwd

	def reset(self):
		self.env.start_trial()
		return self.env.cur_state

	def step(self,action):
		action_string = self.env.actionlist[action]
		observation = self.env.move(action_string)
		self.state = observation
		self.reward = self.env.rwd
		done = False
		info = None
		return observation, self.reward, done, info


def get_frame(maze):
	#grid
	grid = maze.grid
	#location of reward
	rwd_position = np.zeros_like(maze.grid)
	rwd_position[maze.rwd_loc[0][1], maze.rwd_loc[0][0]] = 1
	#location of agent
	agt_position = np.zeros_like(maze.grid)
	agt_position[maze.cur_state[1], maze.cur_state[0]] = 1
	
	return np.array((grid, rwd_position, agt_position)) #np.transpose(np.array((grid, rwd_position, agt_position)),axes=[1,2,0])







def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def softmax(x,T=1):
	 """Compute softmax values for each sets of scores in x."""
	 e_x = np.exp((x - np.max(x))/T)
	 return e_x / e_x.sum(axis=0) # only difference

def plot_softmax(x, T=1):
	f, axarr = plt.subplots(2, sharex=True)
	axarr[0].bar(np.arange(len(x)), x)
	y = softmax(x, T)    
	axarr[1].bar(np.arange(len(x)), y) 
	plt.show()


# =====================================
#        DEFINE PLOT FUNCTIONS        #
# =====================================

def running_mean(l, N):
	sum = 0
	result = list( 0 for x in l)

	for i in range( 0, N ):
		sum = sum + l[i]
		result[i] = sum / (i+1)

	for i in range( N, len(l) ):
		sum = sum - l[i-N] + l[i]
		result[i] = sum / N

	return result


def make_env_plots(maze, env=True, pc_map=False, pcs=None, pc_vec=False, save=False):
	grid = maze.grid
	useable_grid = maze.useable
	agent_loc = maze.cur_state
	
	if env: 
		# plot maze -- agent (blue) and reward (red)
		fig = plt.figure(0)

		axis  = fig.add_axes([0.05, 0.05, .9, .9]) # [left, bottom, width, height]

		ax = fig.gca()
		axis.pcolor(grid, cmap = 'bone', vmax =1, vmin = 0)

		for rwd_loc in maze.rwd_loc:
			rwd_v, rwd_h = rwd_loc
			ax.add_patch(plt.Circle((rwd_v+.5, rwd_h+.5), 0.35, fc='r'))
		
		agent_v, agent_h = agent_loc
		ax.add_patch(plt.Circle((agent_v+.5, agent_h+.5), 0.35, fc='b'))
		if maze.maze_type == 'tmaze':
			pass
			#for port in maze.possible_ports:
			#	ax.add_patch(plt.Circle(np.add(port, (0.5,0.5)), 0.25, fc='g'))

		ax.invert_yaxis()
		#plt.colorbar()
		#if maze.y == maze.x: 
			#plt.axis('tight')
		ax.set_aspect('equal')
		if save:
			plt.savefig('{}environment.svg'.format(maze.maze_type), format='svg', pad_inches =2)
		
	if pc_map:
		cmap = plt.cm.jet
		vmin = 0
		vmax = max(pcs.activity(maze.cur_state)[0])
		cNorm  = colors.Normalize(vmin=0, vmax=vmax)
		scalarMap = cmx.ScalarMappable(norm = cNorm, cmap=cmap)
		# plot place cells 
		# circle radius given by fwhm of place cells (???)
		fig = plt.figure(1)
		ax  = fig.add_axes([0, 0.1, 0.6, 0.85]) # [left, bottom, width, height]
		axc = fig.add_axes([0.63, 0.1, 0.03, 0.85])

	
		for i in range(len(pcs.x)):

			colorVal = scalarMap.to_rgba(np.squeeze(pcs.activity(maze.cur_state))[i])
			ax.add_patch(patches.Circle((pcs.x[i], pcs.y[i]), 0.025, fc=colorVal, ec='none', alpha=0.5))
			ax.set_ylim([1,0])
		cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
		ax.axis('equal')
		if maze.y == maze.x:
			ax.axis('tight')
		#ax.grid('on')
		plt.show()

	if pc_vec:
		cmap = plt.cm.jet
		vmin = 0
		vmax = max(pcs.activity(maze.cur_state)[0])
		cNorm  = colors.Normalize(vmin=0, vmax=vmax)
		scalarMap = cmx.ScalarMappable(norm = cNorm, cmap=cmap)

		# plot input vector
		fig = plt.figure(2)
		ax  = fig.add_axes([0, 0.25, 0.7, 0.15]) # [left, bottom, width, height]
		axc = fig.add_axes([0, 0.1, 0.7, 0.07])

		ax.pcolor(pcs.activity(maze.cur_state), vmin=0, vmax=1, cmap=cmap)
		ax.set_yticklabels([''])
		cb2 = colorbar.ColorbarBase(axc, cmap = cmap, norm = cNorm, orientation='horizontal')
		plt.show()

def make_gif(mypath, mazetype):
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	intfiles = [int(f) for f in onlyfiles]
	intfiles.sort()
	if mazetype =='none':
		gifname = './valplots/gifs/grid{}.gif'.format(obs_rho)
	else: 
		gifname = './valplots/gifs/{}.gif'.format(mazetype)

	with imageio.get_writer(gifname, mode='I', duration=0.5) as writer:
				for filename in intfiles:
					image = imageio.imread(mypath+str(filename))
					writer.append_data(image)
	print( "Gif file saved at ", gifname)


def make_arrows(action, probability):
	if probability == 0: 
		dx, dy = 0, 0
		head_w, head_l = 0,0
	
	else: 
		if action == 0: #N
			dx = 0
			dy = .25
		elif action == 1: #E
			dx = .25
			dy = 0
		elif action == 2: #W
			dx = -.25
			dy = 0
		elif action == 3: #S
			dx = 0
			dy = -.25
		elif action == 4: #stay
			dx = -.1
			dy = -.1
		elif action ==5: #poke 
			dx = .1
			dy = .1
		
		head_w, head_l = 0.1, 0.1
		
	return dx,dy, head_w, head_l



class artist_instance:
	def __init__(self, xy=None, rad=None):
		self.xy=xy if xy is not None else (1,1)
		self.radius = rad if rad is not None else 0.1
		self.fc = 'w'
		self.ec = 'r'
	def art(self):
		return getattr(patches,'Circle')(xy=self.xy,radius=self.radius,fc=self.fc,ec=self.ec)

def print_value_maps(maze,val_maps,**kwargs):
	maptype = kwargs.get('type', 'value')
	value_min, value_max = kwargs.get('val_range', (np.nanmin(val_maps),np.nanmax(val_maps)))
	mazetype = maze.maze_type
	obs_rho = maze.rho
	#rwd_loc = maze.rwd_loc[0]
	maps = kwargs.get('maps', 'all')
	title = kwargs.get('title',None)
	save_dir = kwargs.get('save_dir', None)
	if title == None:
		if mazetype == 'none':
			if obs_rho != 0.0:
				save_string = './figures/grid_obs{}_{}map.svg'.format(obs_rho,maptype)
			else:
				save_string = './figures/{}.svg'.format(title.replace(" ",""))
		else: 
			save_string = './figures/{}_{}map.svg'.format(mazetype,maptype)
	else:
		if mazetype == 'none':
			if obs_rho !=0.0:
				save_string = './figures/grid_obs{}_{}.svg'.format(obs_rho,title.replace(" ",""))
			else:
				save_string = './figures/{}.svg'.format(title.replace(" ",""))
		else: 
			save_string = './figures/{}_{}map.svg'.format(mazetype,title.replace(" ", ""))
	
	if save_dir != None:
		save_string = save_string.replace("./figures/", save_dir)
	
	if maps == 'all':
		plotrows = 4
		plotcols = 5
		fig, axes = plt.subplots(nrows=plotrows, ncols=plotcols, sharex=True, sharey =True)
		items = np.linspace(0, len(val_maps)-1, plotrows*plotcols)
		rp_s = []
		for rwd_loc in maze.rwd_loc:
			rp_s.append(artist_instance(xy=np.add(rwd_loc,(0.5,0.5)), rad=.2))

		for i, ax in enumerate(axes.flat):
			data = val_maps[int(items[i])]
			im = ax.pcolor(data, cmap= 'Spectral_r', vmin=value_min, vmax=value_max)
			im.cmap.set_under('w', 1.0)
			im.cmap.set_over('r', 1.0)
			im.cmap.set_bad('w', 1.0)
			ax.axis('off')
			ax.set_aspect('equal')
			for rwd_patch in rp_s:
				ax.add_patch(rwd_patch.art())
			ax.set_title('{}'.format(int(items[i])))

		if title != None:
			plt.suptitle(title)

		axes[0,0].invert_yaxis()

		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		fig.colorbar(im, cax=cbar_ax)
		plt.savefig(save_string, format='svg')
		plt.show()

	elif type(maps) == list:
		plotrows = 4
		plotcols = 5
		fig, axes = plt.subplots(nrows=plotrows, ncols=plotcols, sharex=True, sharey =True)
		items = np.linspace(0, len(val_maps)-1, plotrows*plotcols)
		rp_s = []
		for rwd_loc in maze.rwd_loc:
			rp_s.append(artist_instance(xy=np.add(rwd_loc,(0.5,0.5)), rad=.2))

		for i, ax in enumerate(axes.flat):
			data = val_maps[int(items[i])]
			im = ax.pcolor(data, cmap= 'Spectral_r', vmin=value_min, vmax=value_max)
			im.cmap.set_under('w', 1.0)
			im.cmap.set_over('r', 1.0)
			im.cmap.set_bad('w', 1.0)
			ax.axis('off')
			ax.set_aspect('equal')
			for rwd_patch in rp_s:
				ax.add_patch(rwd_patch.art())
			ax.set_title('{}'.format(int(items[i])))

		if title != None:
			plt.suptitle(title)

		axes[0,0].invert_yaxis()

		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		fig.colorbar(im, cax=cbar_ax)
		plt.savefig(save_string, format='svg')
		plt.show()


	else:
		if type(maps) == int:
			data = val_maps[maps]
			trial = maps%len(val_maps)
			fig = plt.figure(1)
			im = plt.imshow(data, vmin=np.nanmin(data+10), vmax=np.nanmax(data+5), cmap='Spectral_r', interpolation='none')
			rp_s = []
			for rwd_loc in maze.rwd_loc:
				rp_s.append(artist_instance(rwd_loc, rad=.2))
			im.cmap.set_bad('w', 1.0)
			for rwd_patch1 in rp_s:
				fig.axes[0].add_patch(rwd_patch1.art())
			#plt.imshow(data, vmin=1.32, vmax=1.5, cmap='jet', interpolation='none')
			plt.title('Trial {}'.format(trial))
			plt.colorbar()
			plt.show()
		else:
			print("Must specify which map to print (integer value) else specify 'all' ")

### from junk.ipynb -- need to make compatible 
#	data = val_maps[-1].copy()
#	data[np.where(data>0)] = 0
#
#	## Plot actual choice
#	fig = plt.figure()
#
#	cmap = plt.cm.Spectral_r
#	cNorm  = colors.Normalize(vmin=0, vmax=1)
#	scalarMap = cmx.ScalarMappable(norm = cNorm, cmap=cmap)
#
#
#	ax1  = fig.add_axes([0.04, 0, 0.4, 0.85]) # [left, bottom, width, height]
#	ax2   = fig.add_axes([0.47, 0, 0.4, 0.85]) # [left, bottom, width, height]
#	axc = fig.add_axes([0.89, 0.125, 0.05, 0.6])
#
#	cb1 = mpl.colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
#
#	ax1.imshow(data, vmin=0, vmax=1, cmap='bone', interpolation='none')
#	ax1.add_patch(patches.Circle(maze.rwd_loc, 0.35, fc='w'))
#
#	ax2.imshow(data, vmin=0, vmax=1, cmap='bone', interpolation='none')
#	ax2.add_patch(patches.Circle(maze.rwd_loc, 0.35, fc='w'))
#
#
#	# p_field indicies
#	# 0 - choice, 
#	# 1 - list(tfprob)[choice], 
#	# 2 - list(tfprob).index(max(list(tfprob))),
#	# 3 - max(list(tfprob)), 
#	# 4 - i)
#
#	for i in range(0, p_field.shape[0]):
#	    for j in range(0, p_field.shape[1]):
#	        dx1, dy1, head_w1, head_l1 = make_arrows(p_field[i][j][0], p_field[i][j][1]) 
#	        if (dx1, dy1) == (0,0):
#	            pass
#	        else:
#	            colorVal1 = scalarMap.to_rgba(p_field[i][j][1])
#	            ax1.arrow(j, i, dx1, dy1, head_width =0.3, head_length =0.2, color=colorVal1, alpha = 1 - ((len(val_maps)-p_field[i][j][4])/len(val_maps)))
#	            
#	        dx2, dy2, head_w2, head_l2 = make_arrows(p_field[i][j][2], p_field[i][j][3])
#	        if (dx2, dy2) == (0,0):
#	            pass
#	        else:
#	            colorVal2 = scalarMap.to_rgba(p_field[i][j][3])
#	            ax2.arrow(j, i, dx2, dy2, head_width =0.3, head_length =0.2, color=colorVal2, alpha = 1 - ((len(val_maps)-p_field[i][j][4])/len(val_maps)))
#	            
#	ax1.set_title("Chosen Action")
#	ax2.set_title("Most likely choice")
#
#	plt.savefig('./figures/{}choice_field.svg'.format(mazetype),format ='svg')
#	plt.show()

class KLD_holder(object):
	def __init__(self,gridworld,**kwargs):
		self.flag = kwargs.get('track', 'KLD')
		self.y 		 = gridworld.y
		self.x 		 = gridworld.x
		self.num_act = len(gridworld.actionlist)
		if self.flag == 'KLD':
			self.map = np.zeros((self.y, self.x))
			self.map.fill(np.nan)
			self.op 	= opt_pol_map(gridworld)
		elif self.flag == 'policy':
			self.map 	 = np.zeros((self.y, self.x, self.num_act))
		else:
			print("Flag error. Track 'KLD' (default) or 'policy' as keyword")
		

	def update(self,state, policy):
		#policy must be list or array
		if self.flag == 'policy':
			self.map[state[1], state[0]] = policy

		elif self.flag =='KLD':
			optimal = self.op[state[1], state[0]]
			self.map[state[1], state[0]] = st.entropy(optimal,policy)
			
	def reset(self):
		if self.flag == 'policy':
			self.map = np.zeros((self.y, self.x, self.num_act))
		elif self.flag == 'KLD':
			self.map = np.zeros((self.y, self.x))
		

def opt_pol_map(gridworld):
	optimal_policy = np.zeros((gridworld.y, gridworld.x, len(gridworld.actionlist)))

	for location in gridworld.useable:
		xdim,ydim=location
		xrwd,yrwd=gridworld.rwd_loc[0]
		
		if xdim<xrwd:
			optimal_policy[ydim,xdim][1] = 1
			if ydim<yrwd:
				optimal_policy[ydim,xdim][3] = 1
			elif ydim>yrwd:
				optimal_policy[ydim,xdim][0] = 1
				
		elif xdim>xrwd:
			optimal_policy[ydim,xdim][2] = 1
			if ydim<yrwd:
				optimal_policy[ydim,xdim][3] = 1
			elif ydim>yrwd:
				optimal_policy[ydim,xdim][0] = 1
		else:
			if ydim<yrwd:
				optimal_policy[ydim,xdim][3] = 1
			elif ydim>yrwd:
				optimal_policy[ydim,xdim][0] = 1
			else:
				optimal_policy[ydim,xdim][5] = 1
		
		optimal_policy[ydim,xdim] = softmax(optimal_policy[ydim,xdim],T=0.01)

	return optimal_policy


def save_value_map(vm, maze, trial, savedir):
    data = vm
    fig = plt.figure()
    im = plt.imshow(data, vmin = 0, vmax = 40, cmap='Spectral_r', interpolation ='none')
    rp_s = []
    for reward_loc in maze.rwd_loc:
        rp_s.append(eu.artist_instance(xy=np.add(reward_loc,(0,0)), rad = 0.2))
    im.cmap.set_bad('w', 1.0)
    for rp1 in rp_s:
        fig.axes[0].add_patch(rp1.art())
    plt.colorbar()
    plt.title('Trial {}'.format(trial))
    plt.savefig(savedir+'trial_{}'.format(trial))
    plt.close()
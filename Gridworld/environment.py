'''
Plot functions used for AC Agent in RL gridworld task
Author: Annik Carson 
-- Feb 2018
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

import os
from os import listdir
from os.path import isfile, join

import imageio



###################################
#========================
# Environment Parameters
#======================== 
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
		self.y = np.random.uniform(0,1, (self.num_cells,))
		
	def activity(self, position):
		x0 = (position[0]/self.gx)+float(1/(2*self.gx))
		y0 = (position[1]/self.gy)+float(1/(2*self.gy))
		
		pc_activities = (np.exp(-((self.x-x0)**2 + (self.y-y0)**2) / self.field_width**2))
		return pc_activities

class gridworld(object): 
	def __init__(self, grid_size, **kwargs):
		self.y, self.x = grid_size
		self.rho = kwargs.get('rho', 0)
		self.maze_type = kwargs.get('maze_type', 'none')
		self.grid, self.useable, self.obstacles = self.grid_maker()

		self.actionlist = kwargs.get('actionlist', ['N', 'E', 'W', 'S', 'stay', 'poke'])
		self.rwd_action = kwargs.get('rewarded_action', 'poke')
		
		if self.maze_type == 'tmaze':
			self.rwd_loc = self.useable[0]
			self.port_shift = kwargs.get('port_shift', 'none')
			
		else:
			rwd_choice = np.random.choice(len(self.useable))
			self.rwd_loc = self.useable[rwd_choice]
		
		start_choice = np.random.choice(len(self.useable))
		self.start_loc = self.useable[start_choice]
		
		self.fwhm = kwargs.get('pc_fwhm', 6)
		self.num_placecells = kwargs.get('num_pc', 500)
		self.pcs = PlaceCells(num_cells=self.num_placecells, grid=self,fwhm = self.fwhm)
		self.reset_env()

		self.empty_map = self.make_map(self.grid)
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

		env_types = ['none', 'bar','room','tmaze']
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
	
	def make_map(self, grid):
		'''
		Set up a map for the agent to record its policy and value
			estimates as it navigates the grid
		'''
		pv_map = np.zeros(grid.shape)
		pv_map[grid == 1] = np.nan
		return pv_map
	
	def reset_env(self):        
		self.cur_state = self.start_loc
		self.last_action = 'NA'
		self.net_state = self.mk_state()
		self.rwd = 0
		
		self.done = False
		
	def start_trial(self):
		#rwd_choice = np.random.choice(len(self.useable))
		#self.rwd_loc = self.useable[rwd_choice]
		
		start_choice = np.random.choice(len(self.useable))
		self.start_loc = self.useable[start_choice]
		
		self.cur_state = self.start_loc
		self.last_action = 'NA'
		self.net_state = self.mk_state()
		self.rwd = 0 
		
		self.done = False
	
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
		self.net_state = self.mk_state()
		self.last_action = action
		return self.net_state
	
	def get_reward(self, action):
		if (action == 'poke') & (self.cur_state == self.rwd_loc):
			self.rwd = 1
			self.done = True
			if self.maze_type == 'tmaze':
				if self.port_shift in ['equal', 'left', 'right']:
					self.shift_rwd(self.port_shift)
		elif (self.cur_state == (2,2)):
			self.rwd = -1 
		else:
			self.rwd = 0
			self.done = False

	def shift_rwd(self,shift):
		port_rwd_probabilities = [0.333, 0.333, 0.333]
		current_rewarded_port = self.rwd_loc

		if shift == 'equal':
			dir_prob = [0.5, 0.5]

		elif shift == 'left':
			dir_prob = [0.95, 0.05]

		elif shift == 'right':
			dir_prob = [0.05, 0.95]

		if self.rwd_loc in self.possible_ports:
			poked_port = self.possible_ports.index(self.rwd_loc)
			right_port = (self.possible_ports.index(self.rwd_loc)+1)%3
			left_port = (self.possible_ports.index(self.rwd_loc)+2)%3

			port_rwd_probabilities[poked_port] = 0
			port_rwd_probabilities[left_port] = dir_prob[0]*port_rwd_probabilities[left_port]
			port_rwd_probabilities[right_port] = dir_prob[1]*port_rwd_probabilities[right_port]

			port_rwd_probabilities = [rp/sum(port_rwd_probabilities) for rp in port_rwd_probabilities]
			new_rwd_loc = np.random.choice(3,1,p=port_rwd_probabilities)
			self.rwd_loc = self.possible_ports[new_rwd_loc[0]]
		else:
			print('is not works good')


		
	def mk_state(self, **kwargs):
		state = kwargs.get('state', self.cur_state)        
		#s = np.zeros((self.y, self.x))
		#s[self.cur_state[1]][self.cur_state[0]] = 1 
		#s = np.reshape(s, (self.y*self.x))
		s = self.pcs.activity(state)
		
		a = np.zeros(len(self.actionlist))
		if self.last_action in self.actionlist:
			a[self.actionlist.index(self.last_action)] = 1 
		s = np.append(s, a)
		s = np.reshape(s, (1, len(s)))
		
		return s

class action_wrapper(object): 
	def __init__(self, actionlist):
		self.n = len(actionlist)
		self.actionlist = actionlist
		
class gymworld(object):
	def __init__(self, gridworld):
		self.env = gridworld
		self.action_space = action_wrapper(self.env.actionlist)
		self.state = self.env.net_state
		self.observation_space = np.reshape(self.state, (self.state.shape[1], self.state.shape[0]))
		self.reward = self.env.rwd

	def reset(self):
		self.env.start_trial()
		self.state = self.env.net_state
		return self.state

	def step(self,action):
		action_string = self.env.actionlist[action]
		observation = self.env.move(action_string)
		self.state = observation
		self.reward = self.env.rwd
		done = False
		info = None
		return observation, self.reward, done, info


def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def softmax(x):
	 """Compute softmax values for each sets of scores in x."""
	 e_x = np.exp(x - np.max(x))
	 return e_x / e_x.sum(axis=0) # only difference

def plot_softmax(x):
	f, axarr = plt.subplots(2, sharex=True)
	axarr[0].bar(np.arange(len(x)), x)
	y = softmax(x)    
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


def make_env_plots(maze, env=True, pc_map=False, pc_vec=False):
	grid = maze.grid
	useable_grid = maze.useable
	rwd_loc = maze.rwd_loc
	agent_loc = maze.cur_state
	
	cmap = plt.cm.jet
	cNorm  = colors.Normalize(vmin=0, vmax=max(maze.pcs.activity(maze.cur_state)))
	scalarMap = cmx.ScalarMappable(norm = cNorm, cmap=cmap)


	if env: 
		# plot maze -- agent (blue) and reward (red)
		fig = plt.figure(0)

		axis  = fig.add_axes([0.05, 0.05, .9, .9]) # [left, bottom, width, height]

		ax = fig.gca()
		axis.pcolor(grid, cmap = 'bone', vmax =1, vmin = 0)

		rwd_v, rwd_h = rwd_loc

		agent_v, agent_h = agent_loc

		ax.add_patch(plt.Circle((rwd_v+.5, rwd_h+.5), 0.35, fc='r'))
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

		plt.savefig('{}environment.svg'.format(maze.maze_type), format='svg', pad_inches =2)
		
	if pc_map:
		# plot place cells 
		# circle radius given by fwhm of place cells (???)
		fig = plt.figure(1)
		ax  = fig.add_axes([0, 0.1, 0.6, 0.85]) # [left, bottom, width, height]
		axc = fig.add_axes([0.63, 0.1, 0.03, 0.85])

	
		for i in range(len(maze.pcs.x)):
			colorVal = scalarMap.to_rgba(maze.pcs.activity(maze.cur_state)[i])
			ax.add_patch(patches.Circle((maze.pcs.x[i], maze.pcs.y[i]), 0.025, fc=colorVal, ec='none', alpha=0.5))
			ax.set_ylim([1,0])
		cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
		ax.axis('equal')
		if maze.y == maze.x:
			ax.axis('tight')
		#ax.grid('on')
		plt.show()

	if pc_vec:
		# plot input vector
		fig = plt.figure(2)
		ax  = fig.add_axes([0, 0.25, 0.7, 0.15]) # [left, bottom, width, height]
		axc = fig.add_axes([0, 0.1, 0.7, 0.07])

		ax.pcolor(maze.pcs.activity(maze.cur_state).reshape(1,maze.pcs.activity(maze.cur_state).shape[0]), vmin=0, vmax=1, cmap=cmap)
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
			dy = -.25
		elif action == 1: #E
			dx = .25
			dy = 0
		elif action == 2: #W
			dx = -.25
			dy = 0
		elif action == 3: #S
			dx = 0
			dy = .25
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
		self.ec = 'b'
	def art(self):
		return getattr(patches,'Circle')(xy=self.xy,radius=self.radius,fc=self.fc,ec=self.ec)

def print_value_maps(maze,val_maps,print_last=False, **kwargs):
	mazetype = maze.maze_type
	obs_rho = maze.rho
	rwd_loc = maze.rwd_loc
	maps = kwargs.get('maps', 'all')
	
	if maps == 'all':
		plotrows = 4
		plotcols = 5
		fig, axes = plt.subplots(nrows=plotrows, ncols=plotcols, sharex=True, sharey =True)
		items = np.linspace(0, len(val_maps)-1, plotrows*plotcols)
		rwd_patch=artist_instance(xy=np.add(rwd_loc,(0.5,0.5)), rad=.2)

		for i, ax in enumerate(axes.flat):
			data = val_maps[int(items[i])]
			im = ax.pcolor(data, cmap= 'Spectral_r', vmin=np.nanmin(val_maps), vmax=np.nanmax(val_maps))
			im.cmap.set_under('w', 1.0)
			im.cmap.set_over('r', 1.0)
			im.cmap.set_bad('w', 1.0)
			ax.axis('off')
			ax.set_aspect('equal')
			ax.add_patch(rwd_patch.art())
			ax.set_title('{}'.format(int(items[i])))

		axes[0,0].invert_yaxis()

		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		fig.colorbar(im, cax=cbar_ax)
		if mazetype == 'none':
			plt.savefig('./figures/grid_obs{}_valuemap.svg'.format(obs_rho), format='svg')
		else: 
			plt.savefig('./figures/{}_valuemap.svg'.format(mazetype), format='svg')
		plt.show()

	else:
		if type(maps) == int:
			data = val_maps[maps]
			trial = maps%len(val_maps)
			fig = plt.figure(1)
			im = plt.imshow(data, vmin=np.nanmin(data), vmax=np.nanmax(data), cmap='Spectral_r', interpolation='none')
			rwd_patch1 = artist_instance(rwd_loc, rad=.2)
			im.cmap.set_bad('w', 1.0)
			fig.axes[0].add_patch(rwd_patch1.art())
			#plt.imshow(data, vmin=1.32, vmax=1.5, cmap='jet', interpolation='none')
			plt.title('Trial {}'.format(trial))
			plt.colorbar()
			plt.show()
		else:
			print("Must specify which map to print (integer value) else specify 'all' ")
'''
Plot functions used for AC Agent in RL gridworld task
Author: Annik Carson 
-- January 2019
'''

# =====================================
#           IMPORT MODULES            #
# =====================================
from __future__ import division, print_function

import numpy as np
np.random.seed(12345)


class gridworld(object): 
	def __init__(self, grid_params, **kwargs):
		self.y 				= kwargs.get('y_height',grid_params['y_height'])
		self.x 				= kwargs.get('x_width',grid_params['x_width'])
		self.rho 			= kwargs.get('rho',grid_params['rho'])
		self.bound 			= kwargs.get('walls',grid_params['walls'] )
		self.maze_type 		= kwargs.get('maze_type',grid_params['maze_type'])
		self.port_shift		= kwargs.get('port_shift',grid_params['port_shift'])

		self.actionlist 	= kwargs.get('actionlist', ['N', 'E', 'W', 'S', 'stay', 'poke'])
		self.rwd_action 	= kwargs.get('rewarded_action', 'poke')
		self.barheight 		= kwargs.get('barheight', 3)

		self.grid, self.useable, self.obstacles = self.grid_maker()


		if self.maze_type == 'tmaze':
			self.rwd_loc 		= [self.useable[0]]
			
		if self.maze_type == 'triple_reward':
			self.rwd_loc 		= [(self.x-1, 0), (self.x-1, self.y-1), (0, self.y-1)]
			self.orig_rwd_loc 	= [(self.x-1, 0), (self.x-1, self.y-1), (0, self.y-1)]
			self.starter 		= kwargs.get('t_r_start', (0,0))

		else:
			rwd_choice 			= np.random.choice(len(self.useable))
			self.rwd_loc 		= [self.useable[rwd_choice]]
			self.orig_rwd_loc 	= []
		
		start_choice 	= np.random.choice(len(self.useable))
		self.start_loc 	= self.useable[start_choice]
		
		if self.maze_type=='triple_reward':
			self.start_loc = self.starter
		
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
			barheight = self.barheight
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

		if self.bound: 
			grid_bound = np.ones((self.y+2, self.x+2), dtype=int)
			grid_bound[1:-1][:,1:-1] = grid
			grid = grid_bound


		# lists of tuples storing locations of open grid space and obstacles (unusable grid space)
		useable_grid = list(zip(np.where(grid==0)[1], np.where(grid==0)[0]))
		obstacles = list(zip(np.where(grid==1)[1], np.where(grid==1)[0]))
		
		return grid, useable_grid, obstacles
	
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

	def set_rwd(self, rwd_loc):
		if not isinstance(rwd_loc, list):
			print("must be list of tuples")
		self.rwd_loc = rwd_loc
		if self.maze_type is not 'triple_reward':
			for i in self.rwd_loc: 
				self.orig_rwd_loc.append(i)

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


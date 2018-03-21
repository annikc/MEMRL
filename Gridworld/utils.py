from __future__ import division

import torch 
from torch.autograd import Variable
from torch import autograd, optim, nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

import os
from os import listdir
from os.path import isfile, join

import imageio



# =====================================
#  		    OBJECT CLASSES			  #
# =====================================

class gridworld(): 
	def __init__(self, gridSize, showgrid=False, **kwargs):
		self.y, self.x = gridSize
		self.rho = kwargs.get('rho', 0.3)
		self.maze_type = kwargs.get('maze_type', None)
		self.grid, self.useable, self.obstacles = self.grid_maker()
		
		self.value_map = self.make_value_map(self.grid)
		self.init_value_map = self.value_map

		#### TO DO -- COMPLETE THIS CODE
		#self.policy_map = self.make_policy_map(#####)
		#self.init_policy_map = self.policy_map
		
		self.actionlist = kwargs.get('actionlist', ['N', 'E', 'W', 'S', 'stay', 'poke'])
		self.rwd_action = kwargs.get('rewarded_action', 'poke')
		
		if self.maze_type == 'tmaze':
			self.rwd_loc = self.useable[0]
		else:
			rwd_choice = np.random.choice(len(self.useable))
			self.rwd_loc = self.useable[rwd_choice]
		
		start_choice = np.random.choice(len(self.useable))
		self.start_loc = self.useable[start_choice]
		
		self.fwhm = kwargs.get('pc_fwhm', 6)
		self.num_placecells = kwargs.get('num_pc', 500)
		self.pcs = PlaceCells(num_cells=self.num_placecells, grid=self,fwhm = self.fwhm)
		self.reset_env()
		
		if showgrid: 
			plot_grid(self)
	
	def grid_maker(self):
		env_types = ['none', 'bar','room','tmaze']
		if self.maze_type not in env_types:
			print("Environment Type '{0}' Not Recognized. \nOptions are: {1} \nDefault is Open Field (maze_type = 'none')".format(self.maze_type, env_types))

		grid = np.zeros((self.y,self.x), dtype=int)
		
		if self.maze_type == 'tmaze': 
			self.rho = 0
			grid = np.ones((self.y, self.x), dtype=int)
			
			h1 = int(self.x/2)
			v1 = 0

			
			if h1%2==0:
				for i in range(self.x):
					grid[v1][i] = 0
			else: 
				for i in range(self.x-1):
					grid[v1][i+1] = 0

			if self.y > int(self.x/2):
				for i in range(int(self.x/2)):
					grid[i][h1] = 0
			else:
				for i in range(self.y):
					grid[i][h1] = 0

					
		elif self.maze_type == 'bar':
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
			

		if self.rho != 0: 
			maze = np.vstack([[np.random.choice([0,1], p = [1-self.rho, self.rho]) for _ in range(self.x)] for _ in range(self.y)])
			grid = grid + maze

		# lists of tuples storing locations of open grid space and obstacles (unusable grid space)
		useable_grid = zip(np.where(grid==0)[1], np.where(grid==0)[0])
		obstacles = zip(np.where(grid==1)[1], np.where(grid==1)[0])
		
		return grid, useable_grid, obstacles
	
	def make_value_map(self, grid):
		value_map = np.zeros(grid.shape)
		value_map[grid == 1] = np.nan
		return value_map
	
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
		else:
			self.rwd = 0
			self.done = False
		
	def mk_state(self):        
		#s = np.zeros((self.y, self.x))
		#s[self.cur_state[1]][self.cur_state[0]] = 1 
		#s = np.reshape(s, (self.y*self.x))
		s = self.pcs.activity(self.cur_state)
		
		a = np.zeros(len(self.actionlist))
		if self.last_action in self.actionlist:
			a[self.actionlist.index(self.last_action)] = 1 
		s = np.append(s, a)
		s = np.reshape(s, (1, len(s)))
		
		return s

class PlaceCells(object):
	def __init__(self, num_cells, grid, **kwargs):
		self.num_cells = num_cells
		self.gx, self.gy = grid.x, grid.y
		self.field_width = kwargs.get('fwhm', 6)
		
		self.x = np.random.uniform(0,1,(self.num_cells,))
		self.y = np.random.uniform(0,1, (self.num_cells,))
		
	def activity(self, position):
		x0 = (position[0]/self.gx)+float(1/(2*self.gx))
		y0 = (position[1]/self.gy)+float(1/(2*self.gy))
		
		pc_activities = (np.exp(-((self.x-x0)**2 + (self.y-y0)**2) / self.field_width**2))
		return pc_activities


# set up network
class AC_Net(nn.Module):
    def __init__(self, dims):        
        super(AC_Net, self).__init__()
        # dims is a list of dimensions of each layer [dimension_of_input, dim_of_hiddenlayer1, ... dim_of_hiddenlayerN, dim_of_output]
        self.layers = dims
        if len(dims)>2: 
            self.hidden = []
            for i in range(len(dims)-2):
                self.hidden.append(nn.Linear(dims[i], dims[i+1]))
            self.h_layers = nn.ModuleList(self.hidden)
            
        self.p_in = nn.Linear(dims[-2],dims[-1])
        self.v_in = nn.Linear(dims[-2],1)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
        
        self.saved_actions = []
        self.rewards = []
        
    def forward(self, x):
        if len(self.layers)>2:
            for i in range(len(self.hidden)):
                x = F.relu(self.hidden[i](x))
        pol = F.softmax(self.p_in(x),dim=1)
        val = self.v_in(x)

        return pol, val

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
def select_action(model,state):
    policy_, value_ = model(state)
    a = Categorical(policy_)
    action = a.sample()
    model.saved_actions.append(SavedAction(a.log_prob(action), value_))
    return action.data[0], policy_.data[0], value_.data[0]


def finish_trial(model, discount_factor, optimizer):
    R = 0
    returns_ = discount_rwds(np.asarray(model.rewards), gamma=discount_factor)
    saved_actions = model.saved_actions
    
    policy_losses = []
    value_losses = []
    
    returns_ = torch.Tensor(returns_)
    #returns_ = (returns_ - returns_.mean()) / (returns_.std() + np.finfo(np.float32).eps)
    for (log_prob, value), r in zip(saved_actions, returns_):
        rpe = r - value.data[0, 0]
        policy_losses.append(-log_prob * rpe)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
    optimizer.zero_grad()
    loss = torch.cat(policy_losses).sum() + torch.cat(value_losses).sum()
    loss.backward(retain_graph=False)
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]

# =====================================
# 			   FUNCTIONS			  #
# =====================================

def plot_grid(env):
	grid = env.grid
	useable_grid = env.useable
	rwd_loc = env.rwd_loc
	agent_loc = env.cur_state

	fig = plt.figure()
	ax = fig.gca()
	plt.pcolor(grid, cmap = 'bone', vmax =1, vmin = 0)

	rwd_v, rwd_h = rwd_loc

	agent_v, agent_h = agent_loc

	ax.add_patch(plt.Circle((rwd_v+.5, rwd_h+.5), 0.35, fc='r'))
	ax.add_patch(plt.Circle((agent_v+.5, agent_h+.5), 0.35, fc='b'))
	
	ax.invert_yaxis()
	#plt.colorbar()
	plt.axis('equal')
	
	#plt.show()


#========================
# Environment Plots
#======================== 
def make_env_plots(maze, env=True, pc_map=True, pc_vec=True):
	grid = maze.grid
	useable_grid = maze.useable
	rwd_loc = maze.rwd_loc
	agent_loc = maze.cur_state
	
	cmap = plt.cm.jet
	cNorm  = colors.Normalize(vmin=0, vmax=max(maze.pcs.activity(maze.cur_state)))
	scalarMap = cmx.ScalarMappable(norm = cNorm, cmap=cmap)


	if env: 
		# plot maze -- agent (blue) and reward (red)
		fig = plt.figure()
		axis  = fig.add_axes([0, 0, .7, 1]) # [left, bottom, width, height]

		ax = fig.gca()
		axis.pcolor(grid, cmap = 'bone', vmax =1, vmin = 0)

		rwd_v, rwd_h = rwd_loc

		agent_v, agent_h = agent_loc

		ax.add_patch(plt.Circle((rwd_v+.5, rwd_h+.5), 0.35, fc='r'))
		ax.add_patch(plt.Circle((agent_v+.5, agent_h+.5), 0.35, fc='b'))

		ax.invert_yaxis()
		#plt.colorbar()
		plt.axis('tight')
		
	if pc_map:
		# plot place cells 
		# circle radius given by fwhm of place cells (???)
		fig = plt.figure()
		ax  = fig.add_axes([0, 0.1, 0.6, 0.85]) # [left, bottom, width, height]
		axc = fig.add_axes([0.63, 0.1, 0.03, 0.85])

	
		for i in range(len(maze.pcs.x)):
			colorVal = scalarMap.to_rgba(maze.pcs.activity(maze.cur_state)[i])
			ax.add_patch(patches.Circle((maze.pcs.x[i], maze.pcs.y[i]), 0.025, fc=colorVal, ec='none', alpha=0.5))
			ax.set_ylim([1,0])
		cb1 = mpl.colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
		ax.axis('equal')
		ax.axis('tight')
		ax.grid('on')
		plt.show()

	if pc_vec:
		# plot input vector
		fig = plt.figure()
		ax  = fig.add_axes([0, 0.25, 0.7, 0.15]) # [left, bottom, width, height]
		axc = fig.add_axes([0, 0.1, 0.7, 0.07])

		ax.pcolor(maze.pcs.activity(maze.cur_state).reshape(1,maze.pcs.activity(maze.cur_state).shape[0]), vmin=0, vmax=1, cmap=cmap)
		ax.set_yticklabels([''])
		cb2 = mpl.colorbar.ColorbarBase(axc, cmap = cmap, norm = cNorm, orientation='horizontal')
		plt.show()


# Functions for computing relevant terms for weight updates after trial runs
def discount_rwds(r, gamma = 0.99): 
	disc_rwds = np.zeros_like(r)
	running_add = 0
	for t in reversed(xrange(0, r.size)): 
		running_add = running_add*gamma + r[t]
		disc_rwds[t] = running_add
	return disc_rwds

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

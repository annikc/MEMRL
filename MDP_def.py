################## Outline for Behavioural Experiment MDP ###################
'''
This file acts to set up all of the necessary architecture to run the MDP task. 
This includes definitions for:
	- the simulated environment
	- the trials run within this environment
	- the states, actions, and rewards available in trials
	- the neural network used for learning

################################### NOTES ###################################

State = location x texture (locations = {1234567}, textures = {ABC}) 
Some state transitions will be 0 (b/c cannot transition from 1 to 7, eg.) 
	i.e. would have prob == 1 of moving from 1a to 2a if agent took action == south. 
	have prob == 1 of staying in 1a if agent took action == stay or action == nosepoke.
Reward ==1 if nosepoke in a rewarded chamber.


To represent in NN want to convert to series of distr representations. 
Have set of units that encodes location; set of units that encodes current texture all in one layer 
(i.e. input layer for NN). 
At any given time, have some activation vector in the input layer based on current state info.

Hidden layers = 2 for now. Can randomly assign some place cells to this environment 
i.e. each unit will have some activity fxn determined by loc of agent. 
Using sigmoid units for right now. 
Rate code a neuron would have. I.e. want each neuron to have activity pattern based on location (hard coded) 
Likewise for textures, would have distributed representation (i.e. 10 units activated to different degrees by each pattern). 
20 loc units 10 tex units

30 dimensional input vector to NN. 
Forward pass feed up to some H.0 layer via some set of syn (W.0). H.0 = sig*W.0(in)
Then passing through to next H.1 layer via some set of syn W.1. H.1 = sig*W.1(H.0)
Output to V and PI through weights W.v and W.pi. 
V = (linear fxn)W.v(H.1) and 
PI = (softmax)W.pi(H.1) [Check https://en.wikipedia.org/wiki/Softmax_function]

Takes current state as input and outputs action advantages and value estimate 
'''

import numpy as np 
import matplotlib.pyplot as plt 
import pdb

################################ TO DO LIST ################################
''' 
Forward pass
	- have NN take input ("state" vector) 
		- state vector contains 17 units:
			- 7 location units (#0-6)
				- 0 = home
				- 1 = W1
				- 2 = W2
				- 3 = E1
				- 4 = E2
				- 5 = S1 
				- 6 = S2
			- 4 texture units (#7-10)
				- 7 = null
				- 8 = A
				- 9 = B
				- 10 = C
			- 6 choice/action units (#11-17)
				- 11 = stay
				- 12 = poke
				- 13 = north
				- 14 = east
				- 15 = west
				- 16 = south
	- pass state vector through random weights (chosen from normal distribution) W_{0} to hidden layer h_{0}
		- h_{0} = sigmoid*(W_{0}*s + b_{0})
	- pass activity of hidden layer h_{0} through random weights W_{1} to hidden layer h_{1}
		- h_{1} = sigmoid*(W_{1}*h_{0} + b_{1})


'''
#------------------------------------------------------
""" 					Set up						"""
#------------------------------------------------------
# Define what each state is and what action will be rewarded in that state
class State:
	def __init__(self, location, texture, **kwargs):
		self.loc = location
		self.tex = texture
		self.reward_action = kwargs.get('rewarded_action', 'poke')


class Environment:
	def __init__(self, **kwargs):
		'''
		Define the physical limitations of the environment - what and where everything is, how transitions between 
		states work in the environment
		'''
		self.all_loc = kwargs.get('locations', ['HOME', 'W2', 'W1', 'E1', 'E2', 'S1', 'S2'])
		self.all_tex = kwargs.get('textures', ['A', 'B', 'C', 'null'])
		self.tex_loc = kwargs.get('setup', {'W2':'A', 'W1':'A', 'HOME':'null', 'E1':'B', 'E2':'B', 'S1':'C', 'S2':'C'}) 

		self.all_actions = kwargs.get('actions',['N','E','S','W','stay','poke'])
		self.geom = kwargs.get('geometry', {'HOME':{'N':'HOME','E':'E1','S':'S1','W':'W1','stay':'HOME','poke':'HOME'},
											'W2':{'N':'W2', 'E':'W1', 'S':'W2', 'W':'W2', 'stay':'W2', 'poke':'W2'}, 
											'W1':{'N':'W1','E':'HOME','S':'W1','W':'W2','stay':'W1','poke':'W1'}, 
											'S1':{'N':'HOME','E':'S1','S':'S2','W':'S1','stay':'S1','poke':'S1'},
											'S2':{'N':'S1','E':'S2','S':'S2','W':'S2','stay':'S2','poke':'S2'},
											'E1':{'N':'E1','E':'E2','S':'E1','W':'HOME','stay':'E1','poke':'E1'},
											'E2':{'N':'E2','E':'E2','S':'E2','W':'E1','stay':'E2','poke':'E2'}
			})

		self.tex_reward = kwargs.get('rewarded_textures', {'A':1, 'B':1, 'C':0, 'null':0})
		self.loc_reward = kwargs.get('rewarded_locations', {'HOME':0, 'W2':1, 'W1':0, 'E1':0, 'E2':1, 'S1':0, 'S2':1})
		self.correct_reward = kwargs.get('actual_reward', 'W2')

		# Now, build the states and transition matrices with the above info
		self.cur_state = State(self.all_loc[0], self.tex_loc[self.all_loc[0]])

		# Track reward and return 
		self.rwd = 0 
		self.rwd_mag = kwargs.get('reward_magnitude', 1)
		self.rtrn = 0
		self.gamma = kwargs.get('gamma', 0.9)


	def state_trans(self, action):
		if (self.correct_reward == self.cur_state.loc) & (self.cur_state.reward_action == action):
			self.rwd = self.rwd_mag
			self.rtrn = self.rwd + (self.gamma)*self.rtrn
		else: 
			self.rwd = 0 
		next_loc = self.geom[self.cur_state.loc][action]
		next_tex = self.tex_loc[next_loc]
		self.cur_state = State(next_loc, next_tex)
		print "Result: State in next step will be", self.cur_state.loc

class Trial:
	def __init__(self, t_set, **kwargs):
		self.trial_set = t_set
		self.num_events = kwargs.get('num_events', 50)
		self.time_is = 0 
		self.events = []
		self.time_is = 0
		self.state_is = self.trial_set.working_env.cur_state
		self.action_is = 'N/A'
		self.reward_is = 'N/A'
		self.return_is = 'N/A'
		self.events.append([self.time_is, self.state_is, self.action_is, self.reward_is, self.return_is])
		
	def add_event(self, action, **kwargs):
		step = self.trial_set.working_env.state_trans(action)
		self.time_is = len(self.events)
		self.state_is = self.trial_set.working_env.cur_state
		self.action_is = action
		self.reward_is = self.trial_set.working_env.rwd
		self.return_is = self.trial_set.working_env.rtrn
		self.events.append([self.time_is, self.state_is, self.action_is, self.reward_is, self.return_is])
		#self.step = self.trial_set.working_env.state_trans(action)		

class Trial_Set: 
	def __init__(self, total_trials, **kwargs):
		self.working_env = Environment()
		self.num_trials = 0 
		self.tot_trials = total_trials
		self.trials = []

	def add_trial(self, **kwargs):
		self.current_trial = Trial(self)
		self.trials.append(self.current_trial)
		if self.num_trials < self.tot_trials:
			self.num_trials += 1
			print " ===== \n Number of trials in this set is", self.num_trials, "\n ====="

	#for i in range(total_trials):
	#	add_trial()

#class Network: 
#	def __init__(self, **kwargs):
#		self.h_units = kwargs.get('hidden_units' = 100) 
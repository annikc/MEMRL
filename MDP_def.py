
import numpy as np 
import matplotlib.pyplot as plt 
import pdb

################## Outline for Behavioural Experiment MDP ###################
'''
This file acts to set up all of the necessary architecture to run the MDP task. 
This includes definitions for:
	- the simulated environment
	- the trials run within this environment
	- the states, actions, and rewards available in trials
	- the neural network used for learning

################################### NOTES ###################################

State = location x texture (locations = 1234567, textures = ABC) 
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

#------------------------------------------------------
""" 					Set up						"""
#------------------------------------------------------
class State(object):
	'''
	State objects contain about the location and texture the agent encounters at a given timestep, as well as which
	actions are to be rewarded. In our current formulation, the agent must poke to get a reward. 
	'''
	def __init__(self, location, texture, **kwargs):
		self.loc = location
		self.tex = texture
		self.reward_action = kwargs.get('rewarded_action', 'poke')

class Environment(object):
	'''
	Lay out rules of environment, define reward conditions, set up operation for transitioning between states
		Environment objects define the rules of the agent's world: 
			what and where everything is
			the consequences of the agent's actions/transitions between states
			where rewards exist in the world
		Environment keeps track of the current state of the agent
		Environment defines reward magnitude and the agent's discount rate 
	'''
	def __init__(self, **kwargs):

		# Define physical parameters of environment
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
		# Define which textures are rewarded
		self.tex_reward = kwargs.get('rewarded_textures', {'A':1, 'B':0, 'C':0, 'null':0})
		loc_reward = kwargs.get('rewarded_locations', {'HOME':0, 'W2':1, 'W1':0, 'E1':0, 'E2':1, 'S1':0, 'S2':1})
		self.correct_reward = kwargs.get('actual_reward', 'W2')

		# Create state object for the agent to access location/texture information 
		self.cur_state = State(self.all_loc[0], self.tex_loc[self.all_loc[0]])

		# Track reward and return 
		self.rwd = 0 
		self.rwd_mag = kwargs.get('reward_magnitude', 1)
		self.rtrn = 0
		self.gamma = kwargs.get('gamma', 0)


	def state_trans(self, action):
		'''
		Move agent between states based on selected action
		If selected action is correct, update reward and return variables
		'''
		if (self.correct_reward == self.cur_state.loc) & (self.cur_state.reward_action == action):
			self.rwd = self.rwd_mag
		else: 
			self.rwd = 0
		self.rtrn = self.rwd + (self.gamma)*self.rtrn
		# update agent's state based on selected action
		next_loc = self.geom[self.cur_state.loc][action]
		next_tex = self.tex_loc[next_loc]
		self.cur_state = State(next_loc, next_tex)

class Trial(object):
	'''
	Define a set of events the agent will run through
	'''
	def __init__(self, t_set, **kwargs):
		self.trial_set = t_set
		self.num_events = kwargs.get('num_events', 500)
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
	

class Trial_Set(object): 
	def __init__(self, env_name, total_trials, **kwargs):
		self.num_trial_events = kwargs.get('num_events', 500)
		self.working_env = env_name
		self.num_trials = 0
		self.tot_trials = total_trials
		self.trials = []
		print " =========== \n Number of Trials in Set is:", self.tot_trials, '(Events per Trial', self.num_trial_events, ")\n ==========="

	def add_trial(self, **kwargs):
		self.current_trial = Trial(self, num_events=self.num_trial_events)
		self.trials.append(self.current_trial)
		self.working_env.rtrn = 0
		if self.num_trials < self.tot_trials:
			self.num_trials += 1

class OutputLayer(object): 
	def __init__(self, num_inputs = 50, num_actions = 6):

		# number of inputs to the output layer from last hidden layer
		self.l = num_inputs
		# number of outputs from the layer 
		self.k = num_actions

		# values of the policy units in the output layer before softmax transformation
		self.policy_units_input = np.zeros((self.k,1))
		# values of the softmax transformation
		self.soft_policy = np.zeros((self.k,1))
		# which action gets selected -- want a number instead of a string  ## concatenate the output vector here with the loc/tex vectors to act as new state 
		self.selected_action = np.zeros((self.k,1))
		# value of value unit in the output layer
		self.value_unit = np.zeros((1,1))

		# input vector
		self.inputs = np.zeros((self.l,1))

		# policy weights  
		self.W_A = np.zeros((self.k,self.l))
		# policy biases
		self.b_A = np.zeros((self.k,1))
		# value weights
		self.W_V = np.zeros((1,self.l))
		# value bias
		self.b_V = np.zeros((1,1))
	
		# value gradients
		self.dLv_dWv = np.zeros((1,self.l))
		self.dLv_dbv = np.zeros((1,1))

		# policy gradients
		self.dLpi_dWa = np.zeros((self.k,self.l))
		self.dLpi_dba = np.zeros((self.k,1))

		# gradient sums
		self.dWv = np.zeros((1,self.l))
		self.dWa = np.zeros((self.k,self.l))
		self.dbv = np.zeros((1,1))
		self.dba = np.zeros((self.k,1))

	def f_step(self, inputs, temperature = 1.0):
		
		# calculate the unit activities
		self.inputs 		= inputs # store the inputs
		self.linsum			= np.dot(self.W_A, self.inputs) + self.b_A # calculate the linear sum into softmax
		self.soft_policy 	= softmax(self.linsum, temperature) # calculate the softmax output (i.e. policies)
		self.value_unit 	= np.dot(self.W_V, self.inputs) + self.b_V # calculate the value unit

		# select an action
		choice                       = np.random.choice(np.arange(self.k), 1, list(self.soft_policy[:,0]))[0] #p=[0,0,0,0.5,0,0.5])[0] #
		self.selected_action[:]      = 0.0
		self.selected_action[choice] = 1.0

		return (self.selected_action, self.value_unit, self.soft_policy)

	def accumulate_gradients(self, delta):
		# calculating gradients
		#d_linsum_d_Wa = np.multiply.outer(self.inputs[:,0], np.eye(self.k))
		ind = np.where(self.selected_action == 1.0)
		pi_vec = -self.soft_policy
		pi_vec[ind[0][0]] = 1-self.soft_policy[ind[0][0]]

		# store gradient to be passed to next layer
		self.bprop_v_partial = -delta
		try:
			self.bprop_pi_partial = pi_vec.T
		except:
			pdb.set_trace()
		# value weight gradients
		self.dLv_dWv = (self.bprop_v_partial*self.inputs).T
		self.dLv_dbv = self.bprop_v_partial

		# action weight gradients
		self.dLpi_dWa = delta*(np.outer(pi_vec, self.inputs))
		self.dLpi_dba = delta*pi_vec
		
		#if delta != 0.0:
		#	pdb.set_trace()
		# accumulate the gradients
		self.dWv = self.dWv + self.dLv_dWv
		self.dWa = self.dWa + self.dLpi_dWa
		self.dbv = self.dbv + self.dLv_dbv
		self.dba = self.dba + self.dLpi_dba

	def update_weights(self,learning_rate):

		self.W_A = self.W_A - learning_rate*self.dWa
		self.W_V = self.W_V - learning_rate*self.dWv
		self.b_A = self.b_A - learning_rate*self.dba
		self.b_V = self.b_V - learning_rate*self.dbv

		self.dWv = 0
		self.dWa = 0
		self.dbv = 0
		self.dba = 0

	def initialize_weights(self,sigma):

		self.W_A = 0.00*np.random.normal(0.0,sigma,(self.k,self.l))
		self.W_V = 0.00*np.random.normal(0.0,sigma,(1,self.l))
		self.b_A = 0.00*np.random.normal(0.0,sigma,(self.k,1))
		self.b_V = 0.00*np.random.normal(0.0,sigma,(1,1))

	def reset_gradients(self):

		self.dWv[:] = 0
		self.dWa[:] = 0
		self.dbv[:] = 0
		self.dba[:] = 0

class HiddenLayer(object): 

	def __init__(self, num_inputs = 50, num_units = 50):

		# number of input units
		self.m = num_inputs
		# number of units in this layer 
		self.n = num_units
		
		# linear input to the units (given to sigmoid)
		self.linsum   = np.zeros((self.n,1))
		# unit activities (after sigmoid)
		self.activity = np.zeros((self.n,1))

		# input vector
		self.inputs = np.zeros((self.m,1))

		# weights  
		self.W = np.zeros((self.n,self.m))
		# biases
		self.b = np.zeros((self.n,1))
	
		# value gradients
		self.dLv_dW = np.zeros((self.n,self.m))
		self.dLv_db = np.zeros((self.n,1))

		# policy gradients
		self.dLpi_dW = np.zeros((self.n,self.m))
		self.dLpi_db = np.zeros((self.n,1))

		# gradient sums
		self.dW = np.zeros((self.n,self.m))
		self.db = np.zeros((self.n,1))


	def f_step(self, inputs):
		
		# calculate the unit activities
		self.inputs   = inputs # store the inputs
		self.linsum   = np.dot(self.W, self.inputs) + self.b # calculate the linear sum into sigmoid
		self.activity = sigmoid(self.linsum) # calculate the sigmoid output

		return self.activity

	def accumulate_gradients(self, delta, layer_above):
		'''
		Need to make sure is returning correct shapes for dLpi_dW, etc
		'''
		# calculating gradients 
		d_activ = self.activity*(1-self.activity) 
		d_activity_d_linsum = np.diag(d_activ[:,0]) # matrices A and C
		d_linsum_d_W = np.multiply.outer(self.inputs[:,0], np.eye(self.n)) # tensors B and D

		if type(layer_above).__name__ == 'OutputLayer': 
			cur_layer_dv_calc = np.dot(layer_above.bprop_v_partial, np.dot(layer_above.W_V, d_activity_d_linsum))
			cur_layer_dpi_calc = np.outer(np.dot(layer_above.bprop_pi_partial, layer_above.W_A).T, self.activity)
			print "cur_layer_dpi_calc shape is ", cur_layer_dpi_calc.shape
		else: 
			cur_layer_dv_calc = np.dot(layer_above.bprop_v_partial, np.dot(layer_above.W, d_activity_d_linsum))
			cur_layer_dpi_calc = np.outer(np.dot(layer_above.bprop_pi_partial, layer_above.W).T, self.activity)
			print "cur_layer_dpi_calc shape is ", cur_layer_dpi_calc.shape

		# store gradient to be passed to next layer
		self.bprop_v_partial = cur_layer_dv_calc
		self.bprop_pi_partial = cur_layer_dpi_calc

		# store gradients for value calculations
		self.dLv_dWv = np.dot(cur_layer_dv_calc, d_linsum_d_W)
		self.dLv_dbv = cur_layer_dv_calc

		self.dLpi_dW = delta*np.dot(cur_layer_dpi_calc, d_linsum_d_W)
		self.dLpi_db = delta*cur_layer_dpi_calc 

		# accumulate the gradients
		self.dW = self.dW + self.dLv_dW + self.dLpi_dW
		self.db = self.db + self.dLv_db + self.dLpi_db

	def update_weights(self,learning_rate):

		self.W = self.W - learning_rate*self.dW
		self.b = self.b - learning_rate*self.db

	def initialize_weights(self,sigma):
		
		self.W = np.random.normal(0.0,sigma,(self.n,self.m))
		self.b = np.random.normal(0.0,sigma,(self.n,1))

	def reset_gradients(self):
		
		self.dW[:] = 0
		self.db[:] = 0

class Network(object):
	'''
	Gets current state as argument (not for init)

	'''
	def __init__(self, num_units = [17,50,50,6], learning_rate = [0.001,0.001,0.001,0.001], temperature  = 1.0):
		
		# check the number of layers
		self.num_layers = len(num_units)
		self.has_hidden = self.num_layers > 2
		if self.num_layers < 2:
			print("Network must have at least two layers")
			raise
		self.num_hidden = self.num_layers - 2

		# the number of units in each layer
		self.num_units = num_units

		# store the learning rate
		self.learning_rate = learning_rate

		# store the temperature
		self.temperature = temperature

		# create the input layer
		self.input_layer = np.zeros((self.num_units[0],1))

		# create the hidden layers
		if self.num_hidden > 0:
			self.hidden_layers = []
			for i in range(self.num_hidden):
				self.hidden_layers.append(HiddenLayer(self.num_units[i],self.num_units[i+1]))
				self.hidden_layers[i].initialize_weights(1)

		# create the output layer
		self.output_layer = OutputLayer(self.num_units[-2],self.num_units[-1])
		self.output_layer.initialize_weights(1)

		#initialize prediction error tracking 
		self.delta = 0

		# initialize current action and value
		self.action = np.zeros((self.num_units[-1],1))
		self.value  = np.zeros((1,1))

	def forward_pass(self,state_vector):
		
		# set the input to the networ
		self.input_layer = state_vector
		activity         = self.input_layer

		# forward pass through hidden layers
		if self.has_hidden:
			# do a forward passs through the subsequeent hidden layers
			for i in range(self.num_hidden):
				activity = self.hidden_layers[i].f_step(activity)

		# do the forward step in the output layer
		(self.action, self.value, self.policy_calc) = self.output_layer.f_step(activity,self.temperature)

	def accumulate_gradients(self,delta):
		if self.has_hidden: 
			for i in range(self.num_hidden):
				self.hidden_layers[i].accumulate_gradients(delta)
		
		self.output_layer.accumulate_gradients(delta)


	def update_weights(self):
		if self.has_hidden: 
			for i in range(self.num_hidden):
				self.hidden_layers[i].update_weights(self.learning_rate[i+1])
		self.output_layer.update_weights(self.learning_rate[-1])
	
	def reset_gradients(self):
		if self.has_hidden:
			for i in range(self.num_hidden):
				# reset value gradients for hidden layers
				self.hidden_layers[i].reset_gradients() 
		self.output_layer.reset_gradients()
		


def test_options(data_storage, environment, network):
	# run through possible state options and evaluate learned choices
	for i in range(len(environment.all_loc)):
		test_state = State(environment.all_loc[i], environment.tex_loc[environment.all_loc[i]])
		environment.cur_state = test_state
		NN_input = make_state_vector(test_state.loc, test_state.tex, 'N/A')
		
		# run through network to generate policy calculation
		network.forward_pass(NN_input)
		policy_calc = network.policy_calc[:,0]

		for j in range(network.output_layer.k):
			data_storage[test_state.loc][j].append(policy_calc[j])

def plot_test_options(data_storage, trial_set):
	# for plotting in this specific environment 
	home_store = data_storage['HOME']
	W1_store = data_storage['W1']
	W2_store = data_storage['W2']
	E1_store = data_storage['E1']
	E2_store = data_storage['E2']
	S1_store = data_storage['S1']
	S2_store = data_storage['S2']
	store_trialnum = np.arange(trial_set.tot_trials)
	# row and column sharing
	f, axarr = plt.subplots(3, 5, figsize=(12,8))
	axarr[0, 0].plot(store_trialnum, W2_store[0], 'r', label='North')
	axarr[0, 0].plot(store_trialnum, W2_store[1], 'k', label='East')
	axarr[0, 0].plot(store_trialnum, W2_store[2], 'y', label='South')
	axarr[0, 0].plot(store_trialnum, W2_store[3], 'g-o', label='West')
	axarr[0, 0].plot(store_trialnum, W2_store[4], 'c', label='Stay')
	axarr[0, 0].plot(store_trialnum, W2_store[5], 'b-o', label='Poke')
	axarr[0, 0].set_title('W2')

	axarr[0, 1].plot(store_trialnum, W1_store[0], 'r', label='North')
	axarr[0, 1].plot(store_trialnum, W1_store[1], 'k', label='East')
	axarr[0, 1].plot(store_trialnum, W1_store[2], 'y', label='South')
	axarr[0, 1].plot(store_trialnum, W1_store[3], 'g-o', label='West')
	axarr[0, 1].plot(store_trialnum, W1_store[4], 'c', label='Stay')
	axarr[0, 1].plot(store_trialnum, W1_store[5], 'b-o', label='Poke')
	axarr[0, 1].set_title('W1')

	axarr[0, 2].plot(store_trialnum, home_store[0], 'r', label='North')
	axarr[0, 2].plot(store_trialnum, home_store[1], 'k', label='East')
	axarr[0, 2].plot(store_trialnum, home_store[2], 'y', label='South')
	axarr[0, 2].plot(store_trialnum, home_store[3], 'g-o', label='West')
	axarr[0, 2].plot(store_trialnum, home_store[4], 'c', label='Stay')
	axarr[0, 2].plot(store_trialnum, home_store[5], 'b-o', label='Poke')
	axarr[0, 2].set_title('HOME')

	axarr[0, 3].plot(store_trialnum, E1_store[0], 'r', label='North')
	axarr[0, 3].plot(store_trialnum, E1_store[1], 'k', label='East')
	axarr[0, 3].plot(store_trialnum, E1_store[2], 'y', label='South')
	axarr[0, 3].plot(store_trialnum, E1_store[3], 'g-o', label='West')
	axarr[0, 3].plot(store_trialnum, E1_store[4], 'c', label='Stay')
	axarr[0, 3].plot(store_trialnum, E1_store[5], 'b-o', label='Poke')
	axarr[0, 3].set_title('E1')

	axarr[0, 4].plot(store_trialnum, E2_store[0], 'r', label='North')
	axarr[0, 4].plot(store_trialnum, E2_store[1], 'k', label='East')
	axarr[0, 4].plot(store_trialnum, E2_store[2], 'y', label='South')
	axarr[0, 4].plot(store_trialnum, E2_store[3], 'g-o', label='West')
	axarr[0, 4].plot(store_trialnum, E2_store[4], 'c', label='Stay')
	axarr[0, 4].plot(store_trialnum, E2_store[5], 'b-o', label='Poke')
	axarr[0, 4].set_title('E2')

	axarr[1, 2].plot(store_trialnum, S1_store[0], 'r', label='North')
	axarr[1, 2].plot(store_trialnum, S1_store[1], 'k', label='East')
	axarr[1, 2].plot(store_trialnum, S1_store[2], 'y', label='South')
	axarr[1, 2].plot(store_trialnum, S1_store[3], 'g-o', label='West')
	axarr[1, 2].plot(store_trialnum, S1_store[4], 'c', label='Stay')
	axarr[1, 2].plot(store_trialnum, S1_store[5], 'b-o', label='Poke')
	axarr[1, 2].set_title('S1')

	axarr[2, 2].plot(store_trialnum, S2_store[0], 'r', label='North')
	axarr[2, 2].plot(store_trialnum, S2_store[1], 'k', label='East')
	axarr[2, 2].plot(store_trialnum, S2_store[2], 'y', label='South')
	axarr[2, 2].plot(store_trialnum, S2_store[3], 'g-o', label='West')
	axarr[2, 2].plot(store_trialnum, S2_store[4], 'c', label='Stay')
	axarr[2, 2].plot(store_trialnum, S2_store[5], 'b-o', label='Poke')
	axarr[2, 2].set_title('S2')

	axarr[1,0].axis('off')
	axarr[1,1].axis('off')
	axarr[2,0].axis('off')
	axarr[2,1].axis('off')
	axarr[-2,-2].axis('off')
	axarr[-1,-2].axis('off')
	axarr[-2,-1].axis('off')
	axarr[-1,-1].axis('off')

	axarr[0,0].legend(loc='center left', bbox_to_anchor=(4.55, -1.5))
	axarr[0,0].set_ylim([-0.05, 1.05])

	#plt.savefig('returnplot.png')
	plt.show()



def make_state_vector(loc_var, tex_var, last_act):
	input_vec = np.zeros((17,1))
	if loc_var == 'HOME':
		input_vec[0] = 1
	elif loc_var == 'W1':
		input_vec[1] = 1 
	elif loc_var == 'W2':
		input_vec[2] = 1 
	elif loc_var == 'E1':
		input_vec[3] = 1 
	elif loc_var == 'E2':
		input_vec[4] = 1
	elif loc_var == 'S1':
		input_vec[5] = 1
	elif loc_var == 'S2':
		input_vec[6] = 1

	if tex_var == 'null':
		input_vec[7] = 1
	elif tex_var == 'A':
		input_vec[8] = 1
	elif tex_var == 'B':
		input_vec[9] = 1
	elif tex_var == 'C':
		input_vec[10] = 1

	if last_act == 'N':
		input_vec[11] = 1
	elif last_act == 'E':
		input_vec[12] = 1
	elif last_act == 'W':
		input_vec[13] = 1
	elif last_act == 'S':
		input_vec[14] = 1
	elif last_act == 'stay':
		input_vec[15] = 1
	elif last_act == 'poke':
		input_vec[16] = 1

	return input_vec


# Define mathematical functions
def softmax(x, T=1):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x/T)/np.sum(np.exp(x/T), axis=0)

def sigmoid(alpha):
     y = 1 / (1 + np.exp(-alpha))
     return y

#------------------------------------------------------
""" 		Generate Trial Run Visualization		"""
#------------------------------------------------------
def plot_action_choices(choice): 
	plt.scatter(step_no, return_is)


# Define plotting functions
def plot_walk(agent_loc, step_no): 
	# Create grid for plotting 
	height = 3
	width = 5
	env_map = [[]]*height
	env_labels =  [[]]*height
	for i in range(len(env_map)):
		env_map[i] = [[]]*width
		env_labels[i] = [[]]*width

	env_labels[0][0] = 'W2'

	if agent_loc == 'W2':
		for i in range(len(env_map)):
			for j in range(len(env_map[i])):
				env_map[i][j] = (1,1,1)
		env_map[0][0] = (1,0,0)
	elif agent_loc == 'W1':
		for i in range(len(env_map)):
			for j in range(len(env_map[i])):
				env_map[i][j] = (1,1,1)
		env_map[0][1] = (1,0,0)
	elif agent_loc == 'HOME':
		for i in range(len(env_map)):
			for j in range(len(env_map[i])):
				env_map[i][j] = (1,1,1)
		env_map[0][2] = (1,0,0)
	elif agent_loc == 'E1': 
		for i in range(len(env_map)):
			for j in range(len(env_map[i])):
				env_map[i][j] = (1,1,1)
		env_map[0][3] = (1,0,0) 
	elif agent_loc == 'E2': 
		for i in range(len(env_map)):
			for j in range(len(env_map[i])):
				env_map[i][j] = (1,1,1)
		env_map[0][4] = (1,0,0)
	elif agent_loc == 'S1':
		for i in range(len(env_map)):
			for j in range(len(env_map[i])):
				env_map[i][j] = (1,1,1)
		env_map[1][2] = (1,0,0)
	elif agent_loc == 'S2':
		for i in range(len(env_map)):
			for j in range(len(env_map[i])):
				env_map[i][j] = (1,1,1)
		env_map[2][2] = (1,0,0)	

	#props = dict(facecolor='none', alpha=0.7)
	plt.text(-0.05, 0, '$W2$')
	plt.text(0.95, 0, '$W1$')
	plt.text(1.75, 0, '$HOME$')
	plt.text(2.95, 0, '$E1$')
	plt.text(3.95, 0, '$E2$')
	plt.text(1.95, 1, '$S1$')
	plt.text(1.95, 2, '$S2$')
	plt.text(3.95, -0.75, 't = {0}'.format(step_no))

	plt.imshow(env_map, interpolation="none", aspect=1)
	#plt.show()

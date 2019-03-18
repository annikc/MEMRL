#### Model Free Agent 

'''
Object Classes and Relevant Functions for Actor Critic Agent
Author: Annik Carson 
--  June 2018
'''

# =====================================
#           IMPORT MODULES            #
# =====================================
from __future__ import division, print_function

import numpy as np

import torch 
from torch.autograd import Variable
from torch import autograd, optim, nn
import torch.nn.functional as F
from torch.distributions import Categorical

import stategen as sg

from collections import namedtuple

import pdb

# =====================================
# CLASSES
# =====================================
# Network Class
class AC_Net(nn.Module):
	'''
	An actor-critic neural network class. Takes sensory inputs and generates a policy and a value estimate.
	'''

	# ================================
	def __init__(self, agent_params, **kwargs):
		input_dimensions  = kwargs.get('input_dimensions', agent_params['input_dims'])
		action_dimensions = kwargs.get('action_dimensions', agent_params['action_dims'])
		batch_size        = kwargs.get('batch_size', 4)
		hidden_types      = kwargs.get('hidden_types', agent_params['hid_types'])
		hidden_dimensions = kwargs.get('hidden_dimensions', agent_params['hid_dims'])
		rfsize            = kwargs.get('rfsize', 4)
		padding           = kwargs.get('padding', 1)
		stride            = kwargs.get('stride', 1)
		
		'''
		def __init__(self, input_dimensions, action_dimensions, 
		batch_size=4, hidden_types=[], hidden_dimensions=[],
		rfsize=4, padding=1, stride=1):
		
		AC_Net(input_dimensions, action_dimensions, hidden_types=[], hidden_dimensions=[])

		Create an actor-critic network class.

		Required arguments:
			- input_dimensions (int): the dimensions of the input space
			- action_dimensions (int): the number of possible actions

		Optional arguments:
			- batch_size (int): the size of the batches (default = 4).
			- hidden_types (list of strings): the type of hidden layers to use, options are 'linear',
											  'lstm', 'gru'. If list is empty no hidden layers are
											  used (default = []).
			- hidden_dimensions (list of ints): the dimensions of the hidden layers. Must be a list of
												equal length to hidden_types (default = []).
			- TODO insert new args
		'''
	   
		# call the super-class init 
		super(AC_Net, self).__init__()


		# store the input dimensions
		self.input_d = input_dimensions
		# determine input type
		if type(input_dimensions) == int:
			assert (hidden_types[0] == 'linear' or hidden_types[0] == 'lstm' or hidden_types[0] == 'gru')
			self.input_type = 'vector'
		elif type(input_dimensions) == tuple:
			assert (hidden_types[0] == 'conv' or hidden_types[0] == 'pool')
			self.input_type = 'frame'

		# store the batch size
		self.batch_size = batch_size

		# check that the correct number of hidden dimensions are specified
		assert len(hidden_types) is len(hidden_dimensions)
		
		# check whether we're using hidden layers
		if not hidden_types:

			self.layers = [input_dimensions,action_dimensions]

			# no hidden layers, only input to output, create the actor and critic layers
			self.actor = nn.Linear(input_dimensions, action_dimensions)
			self.critic = nn.Linear(input_dimensions, 1)
			self.output = nn.ModuleList([self.actor, self.critic])

		else:
			# to store a record of the last hidden states
			self.hx = []
			self.cx = []
		
			# create the hidden layers
			self.hidden = nn.ModuleList()
			for i,htype in enumerate(hidden_types):
				
				# check that the type is an accepted one
				assert htype in ['linear','lstm','gru', 'conv', 'pool']

				# get the input dimensions
				if i is 0:
					input_d  = input_dimensions
				else:
					if hidden_types[i-1] in ['conv','pool'] and not htype in ['conv','pool']:
						input_d = int(np.prod(hidden_dimensions[i-1]))
					else:
						input_d = hidden_dimensions[i-1]

				# get the output dimensions
				if not htype in ['conv','pool']:
					output_d = hidden_dimensions[i]
				elif htype in ['conv','pool']:
					output_d = list((0,0,0))
					if htype is 'conv':
						output_d[0] = int(np.floor((input_d[0] + 2*padding - rfsize)/stride) + 1)
						output_d[1] = int(np.floor((input_d[1] + 2*padding - rfsize)/stride) + 1)
						#pdb.set_trace()
						assert output_d[0] == hidden_dimensions[i][0], (hidden_dimensions[i][0], output_d[0])
						assert output_d[1] == hidden_dimensions[i][1]
						output_d[2] = hidden_dimensions[i][2]
					elif htype is 'pool':
						output_d[0] = int(np.floor((input_d[0] +2*padding - (rfsize-1) -1)/stride  +1 ))
						output_d[1] = int(np.floor((input_d[0] +2*padding - (rfsize-1) -1)/stride  +1 ))
						assert output_d[0] == hidden_dimensions[i][0]
						assert output_d[1] == hidden_dimensions[i][1]
						output_d[2] = hidden_dimensions[i][2]
					output_d = tuple(output_d)
				
				# construct the layer
				if htype is 'linear':
					self.hidden.append(nn.Linear(input_d, output_d))
					self.hx.append(None)
					self.cx.append(None)
				elif htype is 'lstm':
					self.hidden.append(nn.LSTMCell(input_d, output_d))
					self.hx.append(Variable(torch.zeros(self.batch_size,output_d)))
					self.cx.append(Variable(torch.zeros(self.batch_size,output_d)))
				elif htype is 'gru':
					self.hidden.append(nn.GRUCell(input_d, output_d))
					self.hx.append(Variable(torch.zeros(self.batch_size,output_d)))
					self.cx.append(None)
				elif htype is 'conv':
					#pdb.set_trace()
					self.hidden.append(nn.Conv2d(input_d[2],output_d[2],rfsize,padding=padding,stride=stride))
					self.hx.append(None)
					self.cx.append(None)
				elif htype is 'pool':
					self.hidden.append(nn.MaxPool2d(rfsize,padding=padding,stride=stride))
					self.hx.append(None)
					self.cx.append(None)
			# create the actor and critic layers
			self.layers = [input_dimensions]+hidden_dimensions+[action_dimensions]

			self.actor = nn.Linear(output_d, action_dimensions)
			self.critic = nn.Linear(output_d, 1)
			self.output = nn.ModuleList([self.actor, self.critic])
		# store the output dimensions
		self.output_d = output_d

		# to store a record of actions and rewards	
		self.saved_actions = []
		self.rewards = []


	# ================================
	def forward(self, x, temperature=1):
		'''
		forward(x):

		Runs a forward pass through the network to get a policy and value.

		Required arguments:
			- x (torch.Tensor): sensory input to the network, should be of size batch x input_d

		'''

		# check the inputs
		if type(self.input_d) == int:
			assert x.shape[-1] == self.input_d
		elif type(self.input_d) == tuple:
			assert (x.shape[2], x.shape[3], x.shape[1]) == self.input_d
			if not  (isinstance(self.hidden[0],nn.Conv2d) or isinstance(self.hidden[0],nn.MaxPool2d)):
				raise Exception('image to non {} layer'.format(self.hidden[0]))
		else:	
			pdb.set_trace()

		# pass the data through each hidden layer
		for i, layer in enumerate(self.hidden):
			# squeeze if last layer was conv/pool and this isn't
			if i > 0:
				if (isinstance(self.hidden[i-1],nn.Conv2d) or isinstance(self.hidden[i-1],nn.MaxPool2d)) and \
				not (isinstance(layer,nn.Conv2d) or isinstance(layer,nn.MaxPool2d)):
					x = x.view(1, -1)
			# run input through the layer depending on type
			if isinstance(layer, nn.Linear):
				x = F.relu(layer(x))
				lin_activity = x
			elif isinstance(layer, nn.LSTMCell):
				x, cx = layer(x, (self.hx[i], self.cx[i]))
				self.hx[i] = x.clone()
				self.cx[i] = cx.clone()
			elif isinstance(layer, nn.GRUCell):
				x = layer(x, self.hx[i])
				self.hx[i] = x.clone()
			elif isinstance(layer, nn.Conv2d):
				x = F.relu(layer(x))
			elif isinstance(layer, nn.MaxPool2d):
				x = layer(x)
		# pass to the output layers
		policy = F.softmax(self.actor(x)/temperature, dim=1)
		value  = self.critic(x)
		
		if isinstance(self.hidden[-1], nn.Linear):
			return policy, value, lin_activity
		else:
			return policy, value

	# ===============================
	def reinit_hid(self):
		# to store a record of the last hidden states
		self.hx = []
		self.cx = []
	
		for i, layer in enumerate(self.hidden):
			if isinstance(layer, nn.Linear):
				pass
			elif isinstance(layer, nn.LSTMCell):
				self.hx.append(Variable(torch.zeros(self.batch_size,layer.hidden_size)))
				self.cx.append(Variable(torch.zeros(self.batch_size,layer.hidden_size)))
			elif isinstance(layer, nn.GRUCell):
				self.hx.append(Variable(torch.zeros(self.batch_size,layer.hidden_size)))
				self.cx.append(None)
			elif isinstance(layer, nn.Conv2d):
				pass
			elif isinstance(layer, nn.MaxPool2d):
				pass

	#def calc_conv_dims(self, )

def conv_output(input_tuple, **kwargs): 
	h_in, w_in, channels = input_tuple
	padding = kwargs.get('padding', 1) ## because this is 1 in MF, default 0
	dilation = kwargs.get('dilation', 1) # default 1
	kernel_size = kwargs.get('rfsize', 4 ) # set in MF
	stride = kwargs.get('stride', 1) # set in MF, default 1 
	
	h_out = int(np.floor(((h_in +2*padding - dilation*(kernel_size-1) - 1)/stride)+1))
	w_out = int(np.floor(((w_in +2*padding - dilation*(kernel_size-1) - 1)/stride)+1))
	
	return (h_out, w_out, channels)


# =====================================
# FUNCTIONS FOR STATE INPUT GENERATION
# =====================================
# Place cell activity vector 

# Gridworld frame tensor 



# =====================================
# FUNCTIONS FOR END OF TRIAL
# =====================================
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def select_action(model,policy_, value_):
	a = Categorical(policy_)
	action = a.sample()
	model.saved_actions.append(SavedAction(a.log_prob(action), value_))
	
	return action.data[0], policy_.data[0], value_.data[0]

 
def sample_select_action(model,state, **kwargs):
	get_lin = kwargs.get('getlin', False)
	if get_lin:
		policy_, value_, linear_activity_ = model(state)
	else:
		policy_, value_ = model(state)[0:2]
	a = Categorical(policy_)
	action = a.sample()
	#model.saved_actions.append(SavedAction(a.log_prob(action), value_))
	if get_lin: 
		return action.data[0], policy_.data[0], value_.data[0], linear_activity_.data[0]
	else: 
		return action.data[0], policy_.data[0], value_.data[0]
 

# Functions for computing relevant terms for weight updates after trial runs
def finish_trial(model, discount_factor, optimizer):
	'''
	finish_trial(model
	Finishes a given training trial and backpropagates.
	'''

	# set the return to zero
	R = 0
	returns_ = discount_rwds(np.asarray(model.rewards), gamma=discount_factor)
	saved_actions = model.saved_actions
	
	policy_losses = []
	value_losses = []
	
	returns_ = torch.Tensor(returns_)
	#pdb.set_trace()
	#returns_ = (returns_ - returns_.mean()) / (returns_.std() + np.finfo(np.float32).eps)
	for (log_prob, value), r in zip(saved_actions, returns_):
		rpe = r - value.data[0, 0]
		policy_losses.append(-log_prob * rpe)
		value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([[r]]))).unsqueeze(-1))
	optimizer.zero_grad()
	p_loss = torch.cat(policy_losses).sum()
	v_loss = torch.cat(value_losses).sum()
	total_loss = p_loss + v_loss
	total_loss.backward(retain_graph=False)
	optimizer.step()
	del model.rewards[:]
	del model.saved_actions[:]

	return p_loss, v_loss

def discount_rwds(r, gamma = 0.99): 
	disc_rwds = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)): 
		running_add = running_add*gamma + r[t]
		disc_rwds[t] = running_add
	return disc_rwds

def generate_values(maze, model):
	value_map = maze.empty_map

	for loc in maze.useable:
		state = Variable(torch.FloatTensor(sg.get_frame(maze,agtlocation=loc)))
		policy, value = sample_select_action(model,state)[1:3]
		value_map[loc[1]][loc[0]] = value
		
	return value_map
	
def generate_values_old(maze, model,**kwargs):
	value_map = maze.empty_map
	EC = kwargs.get('EC', None)
	pcs = kwargs.get('pcs', None)
	if EC!=None:
		EC_pol_map = maze.make_map(maze.grid, pol=True)
		MF_pol_map = maze.make_map(maze.grid, pol=True)
	for loc in maze.useable:
		if model.input_type == 'vector':
			state = Variable(torch.FloatTensor(pcs.activity(loc)))
			policy, value = sample_select_action(model,state)[1:3]
		
		elif model.input_type == 'frame':
			state = Variable(torch.FloatTensor(sg.get_frame(maze,agtlocation=loc)))
			if isinstance (model.hidden[-1], nn.Linear):
				policy, value, lin_act = sample_select_action(model,state, getlin=True)[1:4]
			else: 
				policy, value = sample_select_action(model,state)[1:3]
		
		value_map[loc[1]][loc[0]] = value
		if EC != None:
			if model.input_type == 'vector':
				EC_pol = EC.recall_mem(tuple(state.data[0]))
			elif model.input_type == 'frame':
				EC_pol = EC.recall_mem(tuple(lin_act.view(-1)))
			EC_pol_map[loc[1]][loc[0]] = tuple(EC_pol.data[0])
			MF_pol_map[loc[1]][loc[0]] = tuple(policy)

	if EC == None:
		return value_map
	else:
		return EC_pol_map, MF_pol_map
	
def make_agent(agent_params):
	if agent_params['load_model']: 
		MF = torch.load(agent_params['load_dir']) # load previously saved model
	else:
		MF = AC_Net(agent_params)
	
	opt = optim.Adam(MF.parameters(), lr = agent_params['eta'])
	return MF, opt
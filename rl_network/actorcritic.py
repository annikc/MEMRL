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

from collections import namedtuple

import pdb


# =====================================
# CLASSES
# =====================================

class AC_Net(nn.Module):
	'''
	An actor-critic neural network class. Takes sensory inputs and generates a policy and a value estimate.
	'''

	# ================================
	def __init__(self, input_dimensions, action_dimensions, hidden_types=[], hidden_dimensions=[]):
		'''
		AC_Net(input_dimensions, action_dimensions, hidden_types=[], hidden_dimensions=[])

		Create an actor-critic network class.

		Required arguments:
			- input_dimensions (int): the dimensions of the input space
			- action_dimensions (int): the number of possible actions

		Optional arguments:
			- hidden_types (list of strings): the type of hidden layers to use, options are 'linear',
			                                  'lstm', 'gru'. If list is empty no hidden layers are
			                                  used (default = []).
			- hidden_dimensions (list of ints): the dimensions of the hidden layers. Must be a list of
			                                    equal length to hidden_types (default = []).
		'''
       
		# call the super-class init 
		super(AC_Net, self).__init__()

		# store the input dimensions
		self.input_d = input_dimensions

		# check that the correct number of hidden dimensions are specified
		assert len(hidden_types) is len(hidden_dimensions)
		
		# check whether we're using hidden layers
		if not hidden_types:

			# no hidden layers, only input to output, create the actor and critic layers
			self.actor = nn.Linear(input_dimensions, action_dimensions)
			self.critic = nn.Linear(input_dimensions, 1)
			self.output = nn.ModuleList([self.actor, self.critic])

		else:
 
			# create the hidden layers
			self.hidden = nn.ModuleList()
			for i,htype in enumerate(hidden_types):
				
				# check that the type is an accepted one
				assert htype in ['linear','lstm','gru']

				# get the dimensions
				if i is 0:
					input_d  = input_dimensions
				else:
					input_d  = hidden_dimensions[i-1]
				output_d = hidden_dimensions[i]

				# construct the layer
				if htype is 'linear':
					self.hidden.append(nn.Linear(input_d, output_d))
				elif htype is 'lstm':
					self.hidden.append(nn.LSTMCell(input_d, output_d))
				elif htype is 'gru':
					self.hidden.append(nn.GRUCell(input_d, output_d))

			# create the actor and critic layers
			self.actor = nn.Linear(output_d, action_dimensions)
			self.critic = nn.Linear(output_d, 1)
			self.output = nn.ModuleList([self.actor, self.critic])

		# to store a record of actions and rewards	
		self.saved_actions = []
		self.rewards = []
		
		# initialize the weights ? do we need this?
#		for m in self.modules():
#			if isinstance(m, nn.Linear):
#				m.weight.data.normal_(0, 0.001)
#				m.bias.data.zero_()

	# ================================
	def forward(self, x, h=None, c=None):
		'''
		forward(x, h=None, c=None):

		Runs a forward pass through the network to get a policy and value.

		Required arguments:
			- x (torch.Tensor): sensory input to the network, should be of size batch x input_d

		Optional arguments:
			- h (list of torch.Tensor objests): previous hidden states for recurrent units in a
			                                    list of length = num layers (entry of None for
			                                    Linear layers). Input None if no recurrent layers
			                                    (default = None). 
			- c (list of torch.Tensor objests): previous cell states for LSTM units in a
			                                    list of length = num layers (entry of None for
			                                    Linear or GRU layers). Input None if no LSTM layers
			                                    (default = None). 
		'''

		# check the inputs
		assert x.shape[-1] is self.input_d
		if not h is None:
			assert len(h) is len(self.hidden)
		if not c is None:
			assert len(c) is len(self.hidden)

		# initialize lists to store recurrent hidden states
		hx = []
		cx = []

		# pass the data through each hidden layer
		for i, layer in enumerate(self.hidden):
			if isinstance(layer, nn.Linear):
				x = F.relu(layer(x))
			elif isinstance(layer, nn.LSTMCell):
				x, ci = layer(x, (h[i], c[i]))
				hx.append(x.clone())
				cx.append(ci.clone())
			elif isinstance(layer, nn.GRUCell):
				x = layer(x, h[i])
				hx.append(x.clone())

		# pass to the output layers
		policy = F.softmax(self.actor(x), dim=1)
		value  = self.critic(x)

		return policy, value, (hx, cx)

# =====================================
# FUNCTIONS
# =====================================

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
	#returns_ = (returns_ - returns_.mean()) / (returns_.std() + np.finfo(np.float32).eps)
	for (log_prob, value), r in zip(saved_actions, returns_):
		rpe = r - value.data[0, 0]
		policy_losses.append(-log_prob * rpe)
		value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
	optimizer.zero_grad()
	p_loss = torch.cat(policy_losses).sum()
	v_loss = torch.cat(value_losses).sum()
	total_loss = p_loss + v_loss
	total_loss.backward(retain_graph=False)
	optimizer.step()
	del model.rewards[:]
	del model.saved_actions[:]

	return p_loss, v_loss


# Functions for computing relevant terms for weight updates after trial runs
def discount_rwds(r, gamma = 0.99): 
	disc_rwds = np.zeros_like(r)
	running_add = 0
	for t in reversed(xrange(0, r.size)): 
		running_add = running_add*gamma + r[t]
		disc_rwds[t] = running_add
	return disc_rwds

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
#def select_action(model,policy_,value_):
#    a = Categorical(policy_)
#    action = a.sample()
#    model.saved_actions.append(SavedAction(a.log_prob(action), value_))
#    return action, policy_, value_.data[0]

def select_action(model,state):
	policy_, value_ = model(state)
	a = Categorical(policy_)
	action = a.sample()
	#model.saved_actions.append(SavedAction(a.log_prob(action), value_))
	return action.data[0], policy_.data[0], value_.data[0]

def select_action_end(model,policy_, value_):
	a = Categorical(policy_)
	action = a.sample()
	model.saved_actions.append(SavedAction(a.log_prob(action), value_))
	return action.data[0], policy_.data[0], value_.data[0]


def generate_values(maze, model,EC):
	value_map = maze.empty_map
	if EC!=None:
		EC_pol_map = maze.make_map(maze.grid, pol=True)
		MF_pol_map = maze.make_map(maze.grid, pol=True)
	for loc in maze.useable:
		state = Variable(torch.FloatTensor(maze.mk_state(state=loc)))
		policy, value = select_action(model,state)[1:3]
		value_map[loc[1]][loc[0]] = value
		if EC != None:
			EC_pol = EC.recall_mem(tuple(state.data[0]))
			EC_pol_map[loc[1]][loc[0]] = tuple(EC_pol.data[0])
			MF_pol_map[loc[1]][loc[0]] = tuple(policy)
	if EC == None:
		return value_map
	else:
		return EC_pol_map, MF_pol_map
	

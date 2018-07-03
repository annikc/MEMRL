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



# =====================================
#        DEFINE OBJECT CLASSES        #
# =====================================

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
			
		
		# insert LSTM (n,1,dims[-2]) = (#events, #trials, #inputs)
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
	p_loss = torch.cat(policy_losses).sum()
	v_loss = torch.cat(value_losses).sum()
	total_loss = p_loss + v_loss
	total_loss.backward(retain_graph=False)
	optimizer.step()
	del model.rewards[:]
	del model.saved_actions[:]

	return p_loss, v_loss
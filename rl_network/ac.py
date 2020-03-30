# =============================================================================
# Object Classes and Relevant Functions for Actor Critic Agent
# Author Annik Carson
# Updated Feb 2020
# =============================================================================

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
import stategen as sg

# =====================================
#              FUNCTIONS
# =====================================
def softmax(x, T=1):
	e_x = np.exp((x - np.max(x))/T)
	return np.round(e_x / e_x.sum(axis=0),8) # only difference

def discount_rwds(r, gamma = 0.99):
	disc_rwds = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		running_add = running_add*gamma + r[t]
		disc_rwds[t] = running_add
	return disc_rwds

# =====================================
#       ACTOR CRITIC NETWORK CLASS
# =====================================
class AC_Net(nn.Module):
	SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
	def __init__(self, agent_params, **kwargs):
		'''
		Create an actor-critic network class

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
		input_dimensions  = kwargs.get('input_dimensions',  agent_params['input_dims'])
		action_dimensions = kwargs.get('action_dimensions', agent_params['action_dims'])
		hidden_types      = kwargs.get('hidden_types',      agent_params['hid_types'])
		lin_dims 		  = kwargs.get('linear_dimensions', agent_params['lin_dims'])

		if 'num_channels' not in agent_params.keys():
			self.num_channels  = kwargs.get('num_channels', 3)
		else:
			self.num_channels  = agent_params['num_channels']

		if 'rfsize' not in agent_params.keys():  # kernel size
			self.rfsize        = kwargs.get('rfsize', 4)
		else:
			self.rfsize        = agent_params['rfsize']
		if 'padding' not in agent_params.keys():
			self.padding       = kwargs.get('padding', 1)
		else:
			self.padding 	   = agent_params['padding']
		if 'dilation' not in agent_params.keys():
			self.dilation  	   = 1
		else:
			self.dilation      = kwargs.get('dilation', 1)
		if 'stride' not in agent_params.keys():
			self.stride        = kwargs.get('stride', 1)
		else:
			self.stride        = agent_params['stride']
		if 'batch_size' not in agent_params.keys():
			self.batch_size   = kwargs.get('batch_size', 1)
		else:
			self.batch_size    = agent_params['batch_size']

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

		# check that the correct number of hidden dimensions are specified
		assert len(lin_dims) is sum(1 for p in hidden_types if p in ['linear', 'lstm', 'gru'])

		# check whether we're using hidden layers
		if not hidden_types:
			self.layers = [input_dimensions,action_dimensions]

			# no hidden layers, only input to output, create the actor and critic layers
			self.output = nn.ModuleList([
				nn.Linear(input_dimensions, action_dimensions), # ACTOR
				nn.Linear(input_dimensions, 1)])				# CRITIC
		else:
			# to store a record of the last hidden states
			self.hx = []
			self.cx = []

			# create the hidden layers
			self.hidden = nn.ModuleList()
			self.hidden_dimensions = []
			j = 0
			for i,htype in enumerate(hidden_types):
				# check that the type is an accepted one
				assert htype in ['linear','lstm','gru', 'conv', 'pool']

				# get the input dimensions
				if i is 0:
					input_d  = input_dimensions
				else:
					if hidden_types[i-1] in ['conv','pool'] and not htype in ['conv','pool']:
						input_d = int(np.prod(self.hidden_dimensions[i-1]))

					else:
						input_d = self.hidden_dimensions[i-1]

				# get the output dimensions
				if not htype in ['conv','pool']:
					output_d = lin_dims[j]
					j += 1
				elif htype in ['conv','pool']:
					output_d = list((0,0,0))
					if htype is 'conv':
						output_d[0] = int(np.floor((input_d[0] + 2*self.padding - self.dilation*(self.rfsize-1) - 1)/self.stride) + 1)
						output_d[1] = int(np.floor((input_d[1] + 2*self.padding - self.dilation*(self.rfsize-1) - 1)/self.stride) + 1)
						output_d[2] = self.num_channels
					elif htype is 'pool':
						output_d[0] = int(np.floor((input_d[0] +2*self.padding - self.dilation*(self.rfsize-1) -1)/self.stride  +1 ))
						output_d[1] = int(np.floor((input_d[1] +2*self.padding - self.dilation*(self.rfsize-1) -1)/self.stride  +1 ))
						output_d[2] = self.num_channels
					output_d = tuple(output_d)

				self.hidden_dimensions.append(output_d)

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
					self.hidden.append(nn.Conv2d(input_d[2],output_d[2],kernel_size=self.rfsize,padding=self.padding,stride=self.stride,dilation=self.dilation))
					self.hx.append(None)
					self.cx.append(None)
				elif htype is 'pool':
					self.hidden.append(nn.MaxPool2d(kernel_size=self.rfsize,padding=self.padding,stride=self.stride,dilation=self.dilation))
					self.hx.append(None)
					self.cx.append(None)

			# create the actor and critic layers
			self.layers = [input_dimensions]+self.hidden_dimensions+[action_dimensions]
			self.output = nn.ModuleList([
				nn.Linear(output_d, action_dimensions), #actor
				nn.Linear(output_d, 1)                  #critic
			])
		# store the output dimensions
		self.output_d = output_d

		# to store a record of actions and rewards
		self.saved_actions = []
		self.rewards = []

		self.optimizer = None

	def forward(self, x, temperature=1):
		# check the inputs
		if type(self.input_d) == int:
			assert x.shape[-1] == self.input_d
		elif type(self.input_d) == tuple:
			#assert (x.shape[1], x.shape[2], x.shape[0]) == self.input_d
			if not  (isinstance(self.hidden[0],nn.Conv2d) or isinstance(self.hidden[0],nn.MaxPool2d)):
				raise Exception('image to non {} layer'.format(self.hidden[0]))

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
				self.conv = x
			elif isinstance(layer, nn.MaxPool2d):
				x = layer(x)
		# pass to the output layers
		policy = F.softmax(self.output[0](x), dim=1)
		value  = self.output[1](x)

		if isinstance(self.hidden[-1], nn.Linear):
			return policy, value, lin_activity
		else:
			return policy, value

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

	def select_action(self, policy, value):
		a = Categorical(policy)
		action = a.sample()
		self.saved_actions.append(SavedAction(a.log_prob(action), value))
		return action.item(), policy.data[0], value.item()

	def select_ec_action(self, model, mf_policy_, mf_value_, ec_policy_):
		a = Categorical(ec_policy_)
		b = Categorical(mf_policy_)
		action = a.sample()
		self.saved_actions.append(SavedAction(b.log_prob(action), mf_value_))
		return action.item(), mf_policy_.data[0], mf_value_.item()

def make_agent(agent_params, **kwargs):
	opt = kwargs.get('optimizer_type', optim.Adam)
	if 'freeze_weights' in agent_params.keys():
		freeze_weights = agent_params['freeze_weights']
	else:
		freeze_weights = False

	if agent_params['load_model']:
		MF = torch.load(agent_params['load_dir']) # load previously saved model
	else:
		MF = AC_Net(agent_params)

	if freeze_weights:
		freeze = []
		unfreeze = []
		for i, nums in MF.named_parameters():
			if i[0:6] == 'output':
				unfreeze.append(nums)
			else:
				freeze.append(nums)
		MF.optimizer = opt([{'params': freeze, 'lr': 0.0}, {'params': unfreeze, 'lr': agent_params['eta']}], lr=0.0)
	else:
		critic = []
		others = []
		for i, nums in MF.named_parameters():
			if i[0:8] == 'output.1': #critic
				critic.append(nums)
			else:
				others.append(nums)
		MF.optimizer = opt(MF.parameters(), lr= agent_params['eta'])
	return MF


# Functions for computing relevant terms for weight updates after trial runs
def finish_trial(model, discount_factor, optimizer, **kwargs):
	policy_losses = []
	value_losses  = []
	saved_actions = model.saved_actions
	returns_ 	  = torch.Tensor(discount_rwds(np.asarray(model.rewards), gamma=discount_factor))

	EC     = kwargs.get('cache', None)
	buffer = kwargs.get('buffer', None)

	if EC is not None:
		if buffer is not None:
			mem_dict = {}
			timesteps, states, actions, readable, trial = buffer
			#timesteps = buffer[0]
			#states    = buffer[1]
			#actions   = buffer[2]
			#readable  = buffer[3]
			#trial 	  = buffer[4]
		else:
			raise Exception('No memory buffer provided for kwarg "buffer=" ')

		for (log_prob, value), r, t_, s_, a_, rdbl in zip(saved_actions, returns_, timesteps, states, actions, readable):
			rpe = r - value.item()
			policy_losses.append(-log_prob * rpe)
			value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([[r]]))).unsqueeze(-1))

			mem_dict['activity'] = s_
			mem_dict['action']   = a_
			mem_dict['delta']    = r ## trial change
			mem_dict['timestamp']= t_
			mem_dict['readable'] = rdbl
			mem_dict['trial']    = trial
			EC.add_mem(mem_dict)
	else:
		for (log_prob, value), r in zip(saved_actions, returns_):
			rpe = r - value.item()
			policy_losses.append(-log_prob * rpe)
			value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([[r]]))).unsqueeze(-1))

	model.optimizer.zero_grad()
	p_loss, v_loss = torch.cat(policy_losses).sum(), torch.cat(value_losses).sum()
	total_loss = p_loss + v_loss
	total_loss.backward(retain_graph=False)
	model.optimizer.step()

	del model.rewards[:]
	del model.saved_actions[:]

	return p_loss, v_loss



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
	if EC is not None:
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
		if EC is not None:
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


def snapshot(maze, agent):
	val_array = np.empty(maze.grid.shape)
	pol_array = np.zeros(maze.grid.shape, dtype=[('N', 'f8'), ('E', 'f8'), ('W', 'f8'), ('S', 'f8'), ('stay', 'f8'), ('poke', 'f8')])
	# cycle through all available states
	for i in maze.useable:
		#maze.cur_state = i
		state = torch.Tensor(maze.get_frame(agtlocation=i))
		policy_, value_ = agent(state)[0:2]
		val_array[i[1], i[0]] = value_.item()
		pol_array[i[1], i[0]] = tuple(policy_.detach().numpy()[0])

	return val_array, pol_array



def mem_snapshot(maze, EC, trial_timestamp,**kwargs):
	envelope = kwargs.get('decay', 50)
	mem_temp = kwargs.get('mem_temp', 1)
	mpol_array = np.zeros(maze.grid.shape, dtype=[('N', 'f8'), ('E', 'f8'), ('W', 'f8'), ('S', 'f8'), ('stay', 'f8'), ('poke', 'f8')])
	get_vals = kwargs.get('get_vals', False)
	if get_vals:
		mval_array = np.zeros(maze.grid.shape, dtype=[('N', 'f8'), ('E', 'f8'), ('W', 'f8'), ('S', 'f8'), ('stay', 'f8'), ('poke', 'f8')])

	# cycle through readable states
	for i in EC.cache_list.values():
		xval = i[2][0]
		yval = i[2][1]

		memory       = np.nan_to_num(i[0])
		deltas       = memory[:,0]
		times        = abs(trial_timestamp - memory[:,1])
		pvals 		 = EC.make_pvals(times, envelope=envelope)

		policy = softmax(  np.multiply(deltas, pvals), T=mem_temp) #np.multiply(sim,deltas))
		mpol_array[yval][xval] = tuple(policy)
		if get_vals:
			mval_array[yval][xval] = tuple(deltas)
	if get_vals:
		return mpol_array, mval_array
	else:
		return mpol_array



#### JUNKYARD
def reset_agt(maze, agent_params, **kwargs):
	if agent_params['load_model'] == True:
		if agent_params['rwd_placement'] == 'training_loc':
			rwd_placement = [(int(maze.x / 2), int(maze.y / 2))]
		if agent_params['rwd_placement'] == 'moved_loc':
			rwd_placement = [(int(3 * maze.x / 4), int(maze.y / 4))]
	else:
		rwd_placement = [(int(maze.x / 2), int(maze.y / 2))]

	rwd_location = kwargs.get('rwd_placement', rwd_placement)
	freeze = kwargs.get('freeze_weights', False)
	maze.set_rwd(rwd_location)

	# make agent

	agent_params = gen_input(maze, agent_params)
	MF, opt = make_agent(agent_params, freeze)

	run_dict = {
		'NUM_EVENTS': 300,
		'NUM_TRIALS': 2000,
		'environment': maze,
		'agent': MF,
		'optimizer': opt,
		'agt_param': agent_params
	}
	return run_dict

def conv_output(input_tuple, **kwargs):
	h_in, w_in, channels = input_tuple
	padding = kwargs.get('padding', 1) ## because this is 1 in MF, default 0
	dilation = kwargs.get('dilation', 1) # default 1
	kernel_size = kwargs.get('rfsize', 4 ) # set in MF
	stride = kwargs.get('stride', 1) # set in MF, default 1

	h_out = int(np.floor(((h_in +2*padding - dilation*(kernel_size-1) - 1)/stride)+1))
	w_out = int(np.floor(((w_in +2*padding - dilation*(kernel_size-1) - 1)/stride)+1))

	return (h_out, w_out, channels)

def gen_input(maze, agt_dictionary, **kwargs):
	num_channels = 3
	agt_dictionary['num_channels'] = num_channels
	hidden_layer_types = kwargs.get('hid_types', ['conv', 'pool', 'linear'])

	if maze.bound:
		agt_dictionary['input_dims'] = (maze.y+2, maze.x+2, agt_dictionary['num_channels'])
	else:
		agt_dictionary['input_dims'] = (maze.y, maze.x, agt_dictionary['num_channels'])

	agt_dictionary['hid_types'] = hidden_layer_types
	for ind, i in enumerate(hidden_layer_types):
		if ind == 0:
			agt_dictionary['hid_dims'] = [conv_output(agt_dictionary['input_dims'], rfsize=agt_dictionary['rfsize'])]
		else:
			if i == 'conv' or i == 'pool':
				agt_dictionary['hid_dims'].append(conv_output(agt_dictionary['hid_dims'][ind-1], rfsize=agt_dictionary['rfsize']))
			elif i == 'linear':
				agt_dictionary['hid_dims'].append(agt_dictionary['lin_dims'])

	agt_dictionary['maze'] = maze

	return agt_dictionary

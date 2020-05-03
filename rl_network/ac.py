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
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def softmax(x, T=1):
	e_x = np.exp((x - np.max(x))/T)
	return np.round(e_x / e_x.sum(axis=0),8)

def discount_rwds(r, gamma = 0.99):
	disc_rwds = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		running_add = running_add*gamma + r[t]
		disc_rwds[t] = running_add
	return disc_rwds

def make_agent(agent_params, **kwargs):
	opt = kwargs.get('optimizer_type', optim.Adam)
	if 'freeze_weights' in agent_params.keys():
		freeze_weights = agent_params['freeze_weights']
	else:
		freeze_weights = False

	if agent_params['load_model']:
		MF = torch.load(agent_params['load_dir']) # load previously saved model
	else:
		MF = ActorCritic(agent_params)

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




# =====================================
#       ACTOR CRITIC NETWORK CLASS
# =====================================
class ActorCritic(nn.Module):
	def __init__(self, agent_params, **kwargs):
		# call the super-class init
		super(ActorCritic, self).__init__()
		self.gamma = agent_params['gamma'] # discount factor

		self.input_dims  = agent_params['input_dims']
		self.action_dims = agent_params['action_dims']

		if 'rfsize' not in agent_params.keys():
			self.rfsize  = kwargs.get('rfsize', 4)
		else:
			self.rfsize  = agent_params['rfsize']
		if 'padding' not in agent_params.keys():
			self.padding = kwargs.get('padding', 1)
		else:
			self.padding = agent_params['padding']
		if 'dilation' not in agent_params.keys():
			self.dilation= 1
		else:
			self.dilation= kwargs.get('dilation', 1)
		if 'stride' not in agent_params.keys():
			self.stride  = kwargs.get('stride', 1)
		else:
			self.stride  = agent_params['stride']
		if 'batch_size' not in agent_params.keys():
			self.batch_size= kwargs.get('batch_size', 1)
		else:
			self.batch_size= agent_params['batch_size']


		if 'hidden_types' in agent_params.keys():

			if len(agent_params['hidden_dims']) != len(agent_params['hidden_types']):
				raise Exception('Incorrect specification of hidden layer dimensions')

			hidden_types = agent_params['hidden_types']
			# create lists for tracking hidden layers
			self.hidden = nn.ModuleList()
			self.hidden_dims   = agent_params['hidden_dims']

			self.hx = []
			self.cx = []
			# calculate dimensions for each layer
			for ind, htype in enumerate(hidden_types):
				if htype not in ['linear', 'lstm', 'gru', 'conv', 'pool']:
					raise Exception(f'Unrecognized type for hidden layer {ind}')
				if ind==0:
					input_d = self.input_dims
				else:
					if hidden_types[ind-1] in ['conv', 'pool'] and not htype in ['conv', 'pool']:
						input_d = int(np.prod(self.hidden_dims[ind-1]))

					else:
						input_d = self.hidden_dims[ind-1]

				if htype in ['conv','pool']:
					output_d = tuple(self.conv_output(input_d))
					self.hidden_dims[ind] = output_d

				else:
					output_d = self.hidden_dims[ind]

				# construct the layer
				if htype is 'linear':
					self.hidden.append(nn.Linear(input_d, output_d))
					self.hx.append(None)
					self.cx.append(None)
				elif htype is 'lstm':
					self.hidden.append(nn.LSTMCell(input_d, output_d))
					self.hx.append(Variable(torch.zeros(self.batch_size, output_d)))
					self.cx.append(Variable(torch.zeros(self.batch_size, output_d)))
				elif htype is 'gru':
					self.hidden.append(nn.GRUCell(input_d, output_d))
					self.hx.append(Variable(torch.zeros(self.batch_size, output_d)))
					self.cx.append(None)
				elif htype is 'conv':
					in_channels = input_d[0]
					out_channels = output_d[0]
					self.hidden.append(nn.Conv2d(in_channels, out_channels, kernel_size=self.rfsize, padding=self.padding, stride=self.stride, dilation=self.dilation))
					self.hx.append(None)
					self.cx.append(None)
				elif htype is 'pool':
					self.hidden.append(nn.MaxPool2d(kernel_size=self.rfsize, padding=self.padding, stride=self.stride, dilation=self.dilation))
					self.hx.append(None)
					self.cx.append(None)

			# create the actor and critic layers
			self.layers = [self.input_dims]+self.hidden_dims+[self.action_dims]
			self.output = nn.ModuleList([
				nn.Linear(output_d, self.action_dims), #actor
				nn.Linear(output_d, 1)                 #critic
			])

		else:
			self.layers = [self.input_dims, self.action_dims]
			self.output = nn.ModuleList([nn.Linear(input_dimensions, action_dimensions),  # ACTOR
										 nn.Linear(input_dimensions, 1)])  # CRITIC
		self.output_d = self.hidden_dims[-1]

		self.saved_actions = []
		self.saved_rewards = []

		self.optimizer = optim.Adam(self.parameters(), lr=agent_params['eta'])

	def conv_output(self, input_tuple, **kwargs):
		channels, h_in, w_in = input_tuple
		padding = kwargs.get('padding', self.padding)
		dilation = kwargs.get('dilation', self.dilation)
		kernel_size = kwargs.get('rfsize', self.rfsize)
		stride = kwargs.get('stride', self.stride)

		h_out = int(np.floor(((h_in +2*padding - dilation*(kernel_size-1) - 1)/stride)+1))
		w_out = int(np.floor(((w_in +2*padding - dilation*(kernel_size-1) - 1)/stride)+1))

		return (channels, h_out, w_out)

	def forward(self, x, temperature=1, **kwargs):
		get_lin_act = kwargs.get('lin_act', None)
		# check the inputs
		if type(self.input_dims) == int:
			assert x.shape[-1] == self.input_dims
		elif type(self.input_dims) == tuple:
			if x.shape[0] == 1:
				assert self.input_dims == tuple(x.shape[1:]) # x.shape[0] is the number of items in the batch
			if not  (isinstance(self.hidden[0],nn.Conv2d) or isinstance(self.hidden[0],nn.MaxPool2d)):
				raise Exception(f'image to non {self.hidden[0]} layer')

		# pass the data through each hidden layer
		for i, layer in enumerate(self.hidden):
			# squeeze if last layer was conv/pool and this isn't
			if i > 0:
				if (isinstance(self.hidden[i-1],nn.Conv2d) or isinstance(self.hidden[i-1],nn.MaxPool2d)) and \
				not (isinstance(layer,nn.Conv2d) or isinstance(layer,nn.MaxPool2d)):
					x = x.view(x.shape[0],-1)

			# run input through the layer depending on type
			if isinstance(layer, nn.Linear):
				x = F.relu(layer(x))
				if i == get_lin_act:
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
			if i == get_lin_act:
				lin_activity = x
		# pass to the output layers
		policy = F.softmax(self.output[0](x), dim=1)
		value  = self.output[1](x)

		if get_lin_act is not None:
			return policy, value, lin_activity
			#if isinstance(self.hidden[get_lin_act], nn.Linear):
			#	return policy, value, lin_activity
			#else:
			#	raise Exception('Layer Specified by parameter lin_act is not a linear layer')
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
		return action.item() #, policy.data[0], value.item()

	def select_ec_action(self, mf_policy_, mf_value_, ec_policy_):
		a = Categorical(ec_policy_)
		b = Categorical(mf_policy_)
		action = a.sample()
		self.saved_actions.append(SavedAction(b.log_prob(action), mf_value_))
		return action.item() #, mf_policy_.data[0], mf_value_.item()

	# Functions for computing relevant terms for weight updates after trial runs
	# TODO
	def finish_trial(self, **kwargs):
		policy_losses = []
		value_losses  = []
		saved_actions = self.saved_actions
		returns_ 	  = torch.Tensor(discount_rwds(np.asarray(self.saved_rewards), gamma=self.gamma))

		for (log_prob, value), r in zip(saved_actions, returns_):
			rpe = r - value.item()
			policy_losses.append(-log_prob * rpe)
			value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([[r]]))).unsqueeze(-1))

		self.optimizer.zero_grad()
		p_loss, v_loss = torch.cat(policy_losses).sum(), torch.cat(value_losses).sum()
		total_loss = p_loss + v_loss
		total_loss.backward(retain_graph=False)

		self.optimizer.step()

		del self.saved_rewards[:]
		del self.saved_actions[:]
		return p_loss, v_loss

	def finish_trial_EC(self, **kwargs):
		policy_losses = []
		value_losses = []
		saved_actions = self.saved_actions
		returns_ = torch.Tensor(discount_rwds(np.asarray(self.saved_rewards), gamma=self.gamma))

		EC = kwargs.get('cache', None)
		buffer = kwargs.get('buffer', None)

		if EC is not None:
			if buffer is not None:
				mem_dict = {}
				timesteps, states, actions, readable, trial = buffer
			# timesteps = buffer[0]
			# states    = buffer[1]
			# actions   = buffer[2]
			# readable  = buffer[3]
			# trial 	  = buffer[4]
			else:
				raise Exception('No memory buffer provided for kwarg "buffer=" ')

			for (log_prob, value), r, t_, s_, a_, rdbl in zip(saved_actions, returns_, timesteps, states, actions,
															  readable):
				rpe = r - value.item()
				policy_losses.append(-log_prob * rpe)
				value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([[r]]))).unsqueeze(-1))

				mem_dict['activity'] = s_
				mem_dict['action'] = a_
				mem_dict['delta'] = r  ## trial change
				mem_dict['timestamp'] = t_
				mem_dict['readable'] = rdbl
				mem_dict['trial'] = trial
				EC.add_mem(mem_dict)
		else:
			for (log_prob, value), r in zip(saved_actions, returns_):
				rpe = r - value.item()
				policy_losses.append(-log_prob * rpe)
				value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([[r]]))).unsqueeze(-1))

		self.optimizer.zero_grad()
		p_loss, v_loss = torch.cat(policy_losses).sum(), torch.cat(value_losses).sum()
		total_loss = p_loss + v_loss
		total_loss.backward(retain_graph=False)
		self.optimizer.step()

		del self.saved_rewards[:]
		del self.saved_actions[:]

		return p_loss, v_loss


#++++++++++++++++++++++++++++++++++++++++++++++++++
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
	mpol_array = np.zeros(maze.grid.shape, dtype=[('D', 'f8'), ('U', 'f8'), ('R', 'f8'), ('L', 'f8'), ('J', 'f8'), ('P', 'f8')])
	get_vals = kwargs.get('get_vals', False)
	if get_vals:
		mval_array = np.zeros(maze.grid.shape, dtype=[('D', 'f8'), ('U', 'f8'), ('R', 'f8'), ('L', 'f8'), ('J', 'f8'), ('P', 'f8')])
		mval_array[:] = np.nan
	# cycle through readable states
	for i in EC.cache_list.values():
		row, col = i[2]

		memory       = np.nan_to_num(i[0])
		deltas       = i[0][:,0]
		#times        = abs(trial_timestamp - memory[:,1])
		#pvals 		 = EC.make_pvals(times, envelope=envelope)

		policy = softmax( np.nan_to_num(deltas), T=mem_temp) #np.multiply(sim,deltas))
		mpol_array[row, col] = tuple(policy)
		if get_vals:
			mval_array[row,col] = tuple(deltas)
	if get_vals:
		return mpol_array, mval_array
	else:
		return mpol_array


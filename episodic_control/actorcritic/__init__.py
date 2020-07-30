# Actor-Critic Model-Free Control Module Object Class and Related Functions
# Written and maintained by Annik Carson
# Last updated: July 2020
#
# =====================================
#           IMPORT MODULES            #
# =====================================
from __future__ import division, print_function
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from collections import namedtuple

from fxns import discount_rwds

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
# =====================================
#       ACTOR CRITIC NETWORK CLASS
# =====================================
class ActorCritic(torch.nn.Module):
	def __init__(self, agent_params, **kwargs):
		# call the super-class init
		super(ActorCritic, self).__init__()
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

		if 'gamma' not in agent_params.keys():
			self.gamma = kwargs.get('gamma', 0.98)
		else:
			self.gamma = agent_params['gamma']

		if 'eta' not in agent_params.keys():
			self.eta = kwargs.get('eta', 5e-4)
		else:
			self.eta = agent_params['eta']

		self.use_SR = kwargs.get('use_SR', True)


		if 'hidden_types' in agent_params.keys():

			if len(agent_params['hidden_dims']) != len(agent_params['hidden_types']):
				raise Exception('Incorrect specification of hidden layer dimensions')

			self.hidden_types = agent_params['hidden_types']
			# create lists for tracking hidden layers
			self.hidden = torch.nn.ModuleList()
			self.hidden_dims   = agent_params['hidden_dims']

			self.hx = []
			self.cx = []
			# calculate dimensions for each layer
			for ind, htype in enumerate(self.hidden_types):
				if htype not in ['linear', 'lstm', 'gru', 'conv', 'pool']:
					raise Exception(f'Unrecognized type for hidden layer {ind}')
				if ind==0:
					input_d = self.input_dims
				else:
					if self.hidden_types[ind-1] in ['conv', 'pool'] and not htype in ['conv', 'pool']:
						input_d = int(np.prod(self.hidden_dims[ind-1]))

					else:
						input_d = self.hidden_dims[ind-1]

				if htype in ['conv','pool']:
					output_d = tuple(self.conv_output(input_d))
					self.hidden_dims[ind] = output_d

				else:
					output_d = self.hidden_dims[ind]

				# construct the layer
				if htype == 'linear':
					self.hidden.append(torch.nn.Linear(input_d, output_d))
					self.hx.append(None)
					self.cx.append(None)
				elif htype == 'lstm':
					self.hidden.append(torch.nn.LSTMCell(input_d, output_d))
					self.hx.append(Variable(torch.zeros(self.batch_size, output_d)))
					self.cx.append(Variable(torch.zeros(self.batch_size, output_d)))
				elif htype == 'gru':
					self.hidden.append(torch.nn.GRUCell(input_d, output_d))
					self.hx.append(Variable(torch.zeros(self.batch_size, output_d)))
					self.cx.append(None)
				elif htype == 'conv':
					in_channels = input_d[0]
					out_channels = output_d[0]
					self.hidden.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=self.rfsize, padding=self.padding, stride=self.stride, dilation=self.dilation))
					self.hx.append(None)
					self.cx.append(None)
				elif htype == 'pool':
					self.hidden.append(torch.nn.MaxPool2d(kernel_size=self.rfsize, padding=self.padding, stride=self.stride, dilation=self.dilation))
					self.hx.append(None)
					self.cx.append(None)

			# create the actor and critic layers
			self.layers = [self.input_dims]+self.hidden_dims+[self.action_dims]
			self.output = torch.nn.ModuleList([
				torch.nn.Linear(output_d, self.action_dims), #actor
				torch.nn.Linear(output_d, 1)                 #critic
			])
			if self.use_SR:
				self.SR = torch.nn.Linear(output_d, output_d) # psi

		else:
			self.layers = [self.input_dims, self.action_dims]
			self.output = torch.nn.ModuleList([torch.nn.Linear(input_dimensions, action_dimensions),  # ACTOR
										 torch.nn.Linear(input_dimensions, 1)])  # CRITIC
		output_d = self.hidden_dims[-1]

		self.saved_actions = []
		self.saved_rewards = []
		self.saved_phi     = []
		self.saved_psi     = []

		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.eta)

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
		# check the inputs
		if type(self.input_dims) == int:
			assert x.shape[-1] == self.input_dims
		elif type(self.input_dims) == tuple:
			if x.shape[0] == 1:
				assert self.input_dims == tuple(x.shape[1:]) # x.shape[0] is the number of items in the batch
			if not  (isinstance(self.hidden[0],torch.nn.Conv2d) or isinstance(self.hidden[0],torch.nn.MaxPool2d)):
				raise Exception(f'image to non {self.hidden[0]} layer')

		# pass the data through each hidden layer
		for i, layer in enumerate(self.hidden):
			# squeeze if last layer was conv/pool and this isn't
			if i > 0:
				if (isinstance(self.hidden[i-1],torch.nn.Conv2d) or isinstance(self.hidden[i-1],torch.nn.MaxPool2d)) and \
				not (isinstance(layer,torch.nn.Conv2d) or isinstance(layer,torch.nn.MaxPool2d)):
					x = x.view(x.shape[0],-1)

			# run input through the layer depending on type
			if isinstance(layer, torch.nn.Linear):
				x = F.relu(layer(x))
			elif isinstance(layer, torch.nn.LSTMCell):
				x, cx = layer(x, (self.hx[i], self.cx[i]))
				self.hx[i] = x.clone()
				self.cx[i] = cx.clone()
			elif isinstance(layer, torch.nn.GRUCell):
				x = layer(x, self.hx[i])
				self.hx[i] = x.clone()
			elif isinstance(layer, torch.nn.Conv2d):
				x = F.relu(layer(x))
				self.conv = x
			elif isinstance(layer, torch.nn.MaxPool2d):
				x = layer(x)
		# pass to the output layers
		policy = F.softmax(self.output[0](x), dim=1)
		value  = self.output[1](x)
		if self.use_SR:
			phi = x
			psi = self.SR(x)
			return policy, value, phi, psi
		else:
			return policy, value , x

	def reinit_hid(self):
		# to store a record of the last hidden states
		self.hx = []
		self.cx = []

		for i, layer in enumerate(self.hidden):
			if isinstance(layer, torch.nn.Linear):
				pass
			elif isinstance(layer, torch.nn.LSTMCell):
				self.hx.append(Variable(torch.zeros(self.batch_size,layer.hidden_size)))
				self.cx.append(Variable(torch.zeros(self.batch_size,layer.hidden_size)))
			elif isinstance(layer, torch.nn.GRUCell):
				self.hx.append(Variable(torch.zeros(self.batch_size,layer.hidden_size)))
				self.cx.append(None)
			elif isinstance(layer, torch.nn.Conv2d):
				pass
			elif isinstance(layer, torch.nn.MaxPool2d):
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

	def finish_trial(self, **kwargs):
		policy_losses = []
		value_losses  = []
		saved_actions = self.saved_actions
		returns_ 	  = torch.Tensor(discount_rwds(np.asarray(self.saved_rewards), gamma=self.gamma))

		if self.use_SR:
			phis = self.saved_phi
			psis = self.saved_psi
			psi_losses = []

		for (log_prob, value), r in zip(saved_actions, returns_):
			rpe = r - value.item()
			policy_losses.append(-log_prob * rpe)
			value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([[r]]))).unsqueeze(-1))
		if self.use_SR:
			for t in range(len(phis[:-1])):
				next_psi = psis[t + 1]
				psi_hat = phis[t] + self.gamma * next_psi
				loss = torch.nn.MSELoss(reduction='mean')
				l_psi = loss(psi_hat, psis[t]).view(-1)

				psi_losses.append(l_psi)

		self.optimizer.zero_grad()
		if self.use_SR:
			p_loss, v_loss, psi_loss =  torch.cat(policy_losses).sum(), torch.cat(value_losses).sum(), torch.cat(psi_losses).sum()
			total_loss = p_loss + v_loss + psi_loss
			total_loss.backward(retain_graph=False)
			self.optimizer.step()

			del self.saved_rewards[:]
			del self.saved_actions[:]
			del self.saved_phi[:]
			del self.saved_psi[:]

			return p_loss, v_loss, psi_loss

		else:
			p_loss, v_loss = torch.cat(policy_losses).sum(), torch.cat(value_losses).sum()
			total_loss = p_loss + v_loss
			total_loss.backward(retain_graph=False)
			self.optimizer.step()

			del self.saved_rewards[:]
			del self.saved_actions[:]
			del self.saved_phi[:]
			del self.saved_psi[:]

			return p_loss, v_loss

	def finish_trial_EC(self, **kwargs):
		policy_losses = []
		value_losses = []
		saved_actions = self.saved_actions
		returns_ = torch.Tensor(discount_rwds(np.asarray(self.saved_rewards), gamma=self.gamma))
		if self.use_SR:
			phis = self.saved_phi
			psis = self.saved_psi
			psi_losses = []

		EC = kwargs.get('cache', None)
		buffer = kwargs.get('buffer', None)

		if EC != None:
			if buffer != None:
				mem_dict = {}
				timesteps, states, actions, readable, trial = buffer
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

		if self.use_SR:
			for t in range(len(phis[:-1])):
				next_psi = psis[t + 1]
				psi_hat = phis[t] + self.gamma * next_psi
				loss = torch.nn.MSELoss(reduction='mean')
				l_psi = loss(psi_hat, psis[t]).view(-1)

				psi_losses.append(l_psi)

		self.optimizer.zero_grad()
		if self.use_SR:
			p_loss, v_loss, psi_loss = torch.cat(policy_losses).sum(), torch.cat(value_losses).sum(), torch.cat(
				psi_losses).sum()
			total_loss = p_loss + v_loss + psi_loss
			total_loss.backward(retain_graph=False)
			self.optimizer.step()

			del self.saved_rewards[:]
			del self.saved_actions[:]
			del self.saved_phi[:]
			del self.saved_psi[:]

			return p_loss, v_loss, psi_loss

		else:
			p_loss, v_loss = torch.cat(policy_losses).sum(), torch.cat(value_losses).sum()
			total_loss = p_loss + v_loss
			total_loss.backward(retain_graph=False)
			self.optimizer.step()

			del self.saved_rewards[:]
			del self.saved_actions[:]
			del self.saved_phi[:]
			del self.saved_psi[:]

			return p_loss, v_loss

# =====================================
#              FUNCTIONS
# =====================================
def make_agent(agent_params, **kwargs):
	opt = kwargs.get('optimizer_type', torch.optim.Adam)
	if 'freeze_w' in agent_params.keys():
		freeze_weights = agent_params['freeze_w']
	else:
		freeze_weights = False

	if agent_params['architecture'] == 'A':
		use_SR = False
	elif agent_params['architecture'] == 'B':
		use_SR = True

	if agent_params['load_model']:
		MF = torch.load(agent_params['load_dir']) # load previously saved model
	else:
		MF = ActorCritic(agent_params, use_SR=use_SR)

	if freeze_weights:
		freeze = []
		unfreeze = []
		for i, nums in MF.named_parameters():
			if i[0:6] == 'output' or i[0:2]=='SR':
				print(i)
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

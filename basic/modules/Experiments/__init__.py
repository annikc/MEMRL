# write experiment class
# expt class should take agent and environment
# functions for stepping through events/trials, updating,
# collecting data, writing data
# Annik Carson Nov 20, 2020
# TODO
# =====================================
#           IMPORT MODULES            #
# =====================================
import numpy as np
import time
import pickle, csv
import uuid
import torch

class expt(object):
	def __init__(self, agent, environment, **kwargs):
		self.env = environment
		self.agent = agent
		# self.rep_learner = rep_learner  #TODO add in later
		self.data = self.reset_data_logs()
		self.agent.counter = 0

	def record_log(self, expt_type, env_name, n_trials, **kwargs): ## TODO -- set up logging
		parent_folder = kwargs.get('dir', './Data/')
		log_name     = kwargs.get('file', 'test_bootstrap.csv')
		load_from = kwargs.get('load_from', ' ')

		save_id = uuid.uuid4()
		timestamp = time.asctime(time.localtime())

		expt_log = [
		'save_id',  # uuid
		'experiment_type',  # int
		'load_from',  # str
		'num_trials',  # int
		'num_events',  # int
		'ENVIRONMENT',  # str
		'shape',  # tuple
		'rho',  # float
		'rewards',  # dict
		'action_list',  # list
		'rwd_action',  # str
		'step_penalization',  # float
		'useable',  # list
		'obstacle2D',  # list
		'terminal2D',  # list
		'jump',  # list or NoneType
		'random_start',  # bool
		'AGENT',  # arch
		'use_SR',  # bool
		'freeze_weights',  # bool
		'layers',  # list
		'hidden_types',  # list
		'gamma',  # float
		'eta',  # float
		'optimizer',  # torch optim. class
		'MEMORY',  # # string*
		'cache_limit',  # int
		'use_pvals',  # bool
		'memory_envelope',  # int
		'mem_temp',  # float
		'alpha',  # float   # memory mixing parameters
		'beta'  # int
		]

		log_jam = [save_id, env_name, expt_type, n_trials]

		# write to logger
		with open(parent_folder + log_name, 'a+', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(log_jam)

		# save data
		with open(f'{parent_folder}results/{save_id}_data.p', 'wb') as savedata:
			pickle.dump(self.data, savedata)
		# save agent weights
		torch.save(self.agent.MFC, f=f'{parent_folder}agents/{save_id}.pt')
		# save episodic dictionary
		if self.agent.EC != None:
			with open(f'{parent_folder}ec_dicts/{save_id}_EC.p', 'wb') as saveec:
				pickle.dump(self.agent.EC.cache_list, saveec)

	def reset_data_logs(self):
		data_log = {'total_reward': [],
					'loss': [[], []],
					'trial_length': [],
					'EC_snap': [],
					'P_snap': [],
					'V_snap': []
					}
		return data_log

	def representation_learning(self):
		# TODO
		# to be run before experiment to learn representations of states
		pass

	def end_of_trial(self, trial):
		p, v = self.agent.finish_()

		self.data['total_reward'].append(self.reward_sum)
		self.data['loss'][0].append(p)
		self.data['loss'][1].append(v)

		if trial == 0:
			self.running_rwdavg = self.reward_sum
		else:
			self.running_rwdavg = ((trial) * self.running_rwdavg + self.reward_sum) / (trial + 2)

		if trial % self.print_freq == 0:
			print(f"Episode: {trial}, Score: {self.reward_sum} (Running Avg:{self.running_rwdavg}) [{time.time() - self.t}s]")
			self.t = time.time()

	def run(self, NUM_TRIALS, NUM_EVENTS, **kwargs):
		self.print_freq = kwargs.get('printfreq', 100)
		self.reset_data_logs()
		self.t = time.time()

		for trial in range(NUM_TRIALS):
			state = self.env.reset()
			self.reward_sum = 0

			for event in range(NUM_EVENTS):
				# get state info from environment
				state_representation = self.get_representation(state)
				readable = 0

				# get action from agent
				action, log_prob, expected_value = self.agent.get_action(state_representation)
				# take step in environment
				next_state, reward, done, info = self.env.step(action)

				# end of event
				target_value = 0
				self.reward_sum += reward

				self.agent.log_event(episode=trial, event=self.agent.counter,
									 state=state_representation, action=action, reward=reward, next_state=next_state,
									 log_prob=log_prob, expected_value=expected_value, target_value=target_value,
									 done=done, readable_state=readable)
				self.agent.counter += 1
				state = next_state
				if done:
					break

			self.end_of_trial(trial)

class discrete_state_Experiment(expt):
	def __init__(self, agent, environment, **kwargs):
		super(discrete_state_Experiment, self).__init__(agent, environment)

	def get_representation(self, state):
		# TODO
		# use self.representation_network
		# pass observation from environment
		# output representation to be used for self.agent input
		##### trivial representation: one-hot rep of state
		onehot_state = np.zeros(self.env.observation_space.n)
		onehot_state[state] = 1

		return onehot_state



class cont_state_Experiment(expt):
	def __init__(self, agent, environment, **kwargs):
		super(cont_state_Experiment, self).__init__(agent, environment)

	def get_representation(self, state):
		return state




class gridworldExperiment(expt):
	def __init__(self, agent, environment, **kwargs):
		super(gridworldExperiment, self).__init__(agent, environment)

		# temp
		# only for gridworld environment
		self.sample_obs, self.sample_states = self.env.get_sample_obs()
		self.sample_reps = self.get_reps()
		# / temp

	def get_reps(self):
		reps = []
		for i in self.sample_states:
			j = self.env.twoD2oneD(i)
			r = np.zeros(self.env.nstates)
			r[j] = 1
			reps.append(r)
		return reps

	def get_representation(self, state):
		# TODO
		# use self.representation_network
		# pass observation from environment
		# output representation to be used for self.agent input
		##### trivial representation: one-hot rep of state
		onehot_state = np.zeros(self.env.nstates)
		onehot_state[state] = 1

		return onehot_state

	def snapshot(self):
		# initialize empty data frames
		pol_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])
		val_grid = np.empty(self.env.shape)

		mem_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])

		# forward pass through network
		pols, vals = self.agent.MFC(self.sample_obs)

		# populate with data from network
		for s, p, v in zip(self.sample_states, pols, vals):
			pol_grid[s] = tuple(p.data.numpy())
			val_grid[s] = v.item()

		for ind, rep in enumerate(self.sample_reps):
			mem_pol = self.agent.EC.recall_mem(tuple(rep))
			state = self.sample_states[ind]
			mem_grid[state] = tuple(mem_pol)

		return pol_grid, val_grid, mem_grid
	'''
	def run(self, NUM_TRIALS, NUM_EVENTS, **kwargs):
		print_freq = kwargs.get('printfreq', 100)
		self.render = kwargs.get('render', False)

		self.reset_data_logs()
		t = time.time()
		for trial in range(NUM_TRIALS):
			if self.render:
				self.env.figure[0].canvas.set_window_title(f'Trial {trial}')
			self.env.reset()
			self.reward_sum = 0

			for event in range(NUM_EVENTS):
				## get observation from environment
				state = self.env.state
				state_representation = self.get_representation(state)
				readable = 0

				# get action from agent
				action, log_prob, expected_value = self.agent.get_action(state_representation)
				# take step in environment
				next_state, reward, done, info = self.env.step(action)

				# end of event
				target_value = 0
				self.reward_sum += reward

				self.agent.log_event(episode=trial, event=self.agent.counter,
									 state=state_representation, action=action, reward=reward, next_state=next_state,
									 log_prob=log_prob, expected_value=expected_value, target_value=target_value,
									 done=done, readable_state=readable)
				self.agent.counter += 1
				if self.render:
					self.env.render()
				if done:
					break

			self.end_of_trial(trial)
	'''

#TODO write bootstrap as more general class
class gridworldBootstrap(gridworldExperiment):
	def __init__(self, agent, environment):
		super(gridworldBootstrap,self).__init__(agent, environment)
		# temp
		self.policy_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])
		self.value_grid  = np.empty(self.env.shape)
		#\temp

	def take_snapshot(self):
		states2d = self.sample_states
		reps = self.sample_reps

		#get EC policies
		EC_pols = self.policy_grid.copy()

		#get MF policies, values
		MF_pols = self.policy_grid.copy()
		MF_vals = self.value_grid.copy()

		for rep, s in zip(reps, states2d):
			p, v = self.agent.MFC(rep)
			MF_vals[s[0], s[1]] = v
			MF_pols[s[0], s[1]] = tuple(p)

			ec_p = self.agent.EC.recall_mem(tuple(rep), timestep=self.agent.counter)
			EC_pols[s[0],s[1]] = tuple(ec_p)

		self.data['V_snap'].append(MF_vals)
		self.data['P_snap'].append(MF_pols)
		self.data['EC_snap'].append(EC_pols)

	def run(self, NUM_TRIALS, NUM_EVENTS, **kwargs):
		self.print_freq = kwargs.get('printfreq', 100)
		self.reset_data_logs()
		self.data['bootstrap_reward'] = []
		self.data['trajectories'] = []
		self.t = time.time()

		for trial in range(NUM_TRIALS):
			for set in range(2): ## set 0: episodic control, use this for weight updates; set 1: MF control, no updates
				state = self.env.reset()
				state = np.random.choice([0,19,399,380])
				self.env.set_state(state)
				self.reward_sum = 0
				if set == 0:
					self.agent.get_action = self.agent.EC_action
				else:
					self.agent.get_action = self.agent.MF_action
				#print(f'trial{trial}, {self.agent.get_action.__name__}')
				for event in range(NUM_EVENTS):
					state_representation = self.get_representation(state)
					readable = state

					# get action from agent
					action, log_prob, expected_value = self.agent.get_action(state_representation)
					# take step in environment
					next_state, reward, done, info = self.env.step(action)
					#print(self.env.oneD2twoD(state), self.env.action_list[action], self.env.oneD2twoD(next_state))
					# end of event
					target_value = 0
					self.reward_sum += reward

					self.agent.log_event(episode=trial, event=self.agent.counter,
										 state=state_representation, action=action, reward=reward, next_state=next_state,
										 log_prob=log_prob, expected_value=expected_value, target_value=target_value,
										 done=done, readable_state=readable)
					self.agent.counter += 1
					state = next_state
					if done:
						break

				if set == 0:
					self.end_of_trial(trial)
				elif set ==1:
					# temp
					trajs = np.vstack(self.agent.transition_cache.transition_cache)
					sts, acts, rwds = trajs[:,10], trajs[:,3], trajs[:,4]
					data_package = [(x,y,z) for x, y,z in zip(sts,acts,rwds)]
					#print(data_package)
					self.data['trajectories'].append(data_package)
					# \temp
					self.data['bootstrap_reward'].append(self.reward_sum)
					self.agent.transition_cache.clear_cache()

			self.take_snapshot()

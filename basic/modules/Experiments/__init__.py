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
import os

class expt(object):
	def __init__(self, agent, environment, **kwargs):
		self.env = environment
		self.agent = agent
		# self.rep_learner = rep_learner  #TODO add in later
		self.data = self.reset_data_logs()
		self.agent.counter = 0
		self.pol_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])
		self.val_grid = np.empty(self.env.shape)

	def update_ledger(self, parent_folder, file_name, info_list):
		with open(parent_folder + file_name, 'a+', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(info_list)

	def save_objects(self, parent_folder, save_id):
		# save data
		with open(f'{parent_folder}results/{save_id}_data.p', 'wb') as savedata:
			pickle.dump(self.data, savedata)
		# save agent weights
		torch.save(self.agent.MFC.state_dict(), f=f'{parent_folder}agents/{save_id}.pt')
		# save episodic dictionary
		if self.agent.EC != None:
			with open(f'{parent_folder}ec_dicts/{save_id}_EC.p', 'wb') as saveec:
				pickle.dump(self.agent.EC.cache_list, saveec)

	def record_log(self, env_name, representation_type, n_trials, n_steps, **kwargs): ## TODO -- set up logging
		parent_folder = kwargs.get('dir', './Data/')
		log_name      = kwargs.get('file', 'test_bootstrap.csv')
		load_from     = kwargs.get('load_from', ' ')
		mock_log      = kwargs.get('mock_log', False)

		save_id = uuid.uuid4()
		timestamp = time.asctime(time.localtime())

		field_names = [
		'timestamp', #datetime experiment recorded
		'save_id',  # uuid
		'load_from',  # str
		'num_trials',  # int
		'num_events',  # int
		'env_name',  # str
		'representation', # str
		'MF_input_dims',  # arch
		'MF_fc1_dims',  # bool
		'MF_fc2_dims',  # bool
		'MF_lr',  # list
		'MF_temp',  # list
		'MF_gamma',  # float
		'EC_cache_limit',  # float
		'EC_temp',  # torch optim. class
		'EC_mem_decay',  # # string
		'EC_use_pvals',  # bool
		'EC_similarity_meas', # string
		'extra_info'
		]
		run_data = [timestamp, save_id, load_from, n_trials, n_steps, env_name, representation_type]
		network_keys = ['input_dims', 'fc1_dims', 'fc2_dims', 'lr', 'temperature']
		ec_keys = ['cache_limit', 'mem_temp', 'memory_envelope', 'use_pvals']
		agent_data = [self.agent.MFC.__dict__[k] for k in network_keys] + [self.agent.gamma]
		if self.agent.EC != None:
			ec_data = [self.agent.EC.__dict__[k] for k in ec_keys]
			ec_data.append(self.agent.EC.__dict__['distance_metric'])
		else:
			ec_data = ["None" for k in ec_keys] + ["None"]

		extra_info = kwargs.get('extra', [])

		log_jam = run_data + agent_data + ec_data + extra_info

		# write to logger
		if not os.path.exists(parent_folder+log_name):
			self.update_ledger(parent_folder,log_name,info_list=field_names)

		if mock_log:
			log_jam = log_jam+['mock log']

		self.update_ledger(parent_folder,log_name,info_list=log_jam)

		if not mock_log: ## can turn on flag to write to csv without saving files
			self.save_objects(parent_folder,save_id)
		print(f'Logged with ID {save_id}')

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

	def snapshot(self, states, observations):
		# initialize empty data frames
		pol_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])
		val_grid = np.empty(self.env.shape)

		mem_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])

		# forward pass through network
		pols, vals = self.agent.MFC(observations)

		# populate with data from network
		for s, p, v in zip(states, pols, vals):
			pol_grid[s] = tuple(p.data.numpy())
			val_grid[s] = v.item()

		return pol_grid, val_grid

	def end_of_trial(self, trial, logsnap=False):
		p, v = self.agent.finish_()

		# temp
		if logsnap:
			if trial % self.print_freq==0:
				states = [self.env.oneD2twoD(x) for x in list(self.agent.state_reps.keys())]
				observations = list(self.agent.state_reps.values())
				MF_pols, MF_vals = self.snapshot(states,observations)
				self.data['V_snap'].append(MF_vals)
				self.data['P_snap'].append(MF_pols)
		# /temp

		self.data['total_reward'].append(self.reward_sum) # removed for bootstrap expts
		self.data['loss'][0].append(p)
		self.data['loss'][1].append(v)

		if trial <= 10:
			self.running_rwdavg = np.mean(self.data['total_reward'])
		else:
			self.running_rwdavg = np.mean(self.data['total_reward'][-self.print_freq:])

		if trial % self.print_freq == 0:
			print(f"Episode: {trial}, Score: {self.reward_sum} (Running Avg:{self.running_rwdavg}) [{time.time() - self.t}s]")
			self.t = time.time()

	def single_step(self,trial):
		# get representation for given state of env. TODO: should be in agent to get representation?
		state_representation = self.agent.get_state_representation(self.state)
		readable = self.state

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
		self.state = next_state
		return done

	def run(self, NUM_TRIALS, NUM_EVENTS, **kwargs):
		self.print_freq = kwargs.get('printfreq', 100)
		self.reset_data_logs()
		self.t = time.time()
		logsnap = kwargs.get('snapshot_logging', False)

		for trial in range(NUM_TRIALS):
			self.state = self.env.reset()
			self.reward_sum = 0

			for event in range(NUM_EVENTS):
				done = self.single_step(trial)

				if done:
					break

			self.end_of_trial(trial,logsnap=logsnap)

class conv_expt(expt):
	def __init__(self, agent, environment):
		super().__init__(agent,environment)

	def record_log(self, env_name, representation_type, n_trials, n_steps, **kwargs): ## TODO -- set up logging
		parent_folder = kwargs.get('dir', './Data/')
		log_name      = kwargs.get('file', 'test_bootstrap.csv')
		load_from     = kwargs.get('load_from', ' ')
		mock_log      = kwargs.get('mock_log', False)

		save_id = uuid.uuid4()
		timestamp = time.asctime(time.localtime())

		field_names = [
		'timestamp', #datetime experiment recorded
		'save_id',  # uuid
		'load_from',  # str
		'num_trials',  # int
		'num_events',  # int
		'env_name',  # str
		'representation', # str
		'MF_input_dims',  # arch
		'MF_hidden_types',
		'MF_hidden_dims',
		'MF_lr',  # list
		'MF_temp',  # list
		'MF_gamma',  # float
		'EC_cache_limit',  # float
		'EC_temp',  # torch optim. class
		'EC_mem_decay',  # # string
		'EC_use_pvals',  # bool
		'EC_similarity_meas', # string
		'extra_info'
		]
		run_data = [timestamp, save_id, load_from, n_trials, n_steps, env_name, representation_type]
		network_keys = ['input_dims', 'hidden_types', 'hidden_dims', 'lr', 'temperature']
		ec_keys = ['cache_limit', 'mem_temp', 'memory_envelope', 'use_pvals']
		agent_data = [self.agent.MFC.__dict__[k] for k in network_keys] + [self.agent.gamma]
		if self.agent.EC != None:
			ec_data = [self.agent.EC.__dict__[k] for k in ec_keys]
			ec_data.append(self.agent.EC.__dict__['distance_metric'])
		else:
			ec_data = ["None" for k in ec_keys] + ["None"]

		extra_info = kwargs.get('extra', [])

		log_jam = run_data + agent_data + ec_data + extra_info

		# write to logger
		if not os.path.exists(parent_folder+log_name):
			with open(parent_folder + log_name, 'a+', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(field_names)

		with open(parent_folder + log_name, 'a+', newline='') as file:
			writer = csv.writer(file)
			if mock_log:
				writer.writerow(log_jam+["mock log"])
			else:
				writer.writerow(log_jam)

		if not mock_log: ## can turn on flag to write to csv without saving files
			# save data
			with open(f'{parent_folder}results/{save_id}_data.p', 'wb') as savedata:
				pickle.dump(self.data, savedata)
			# save agent weights
			torch.save(self.agent.MFC.state_dict(), f=f'{parent_folder}agents/{save_id}.pt')
			# save episodic dictionary
			if self.agent.EC != None:
				with open(f'{parent_folder}ec_dicts/{save_id}_EC.p', 'wb') as saveec:
					pickle.dump(self.agent.EC.cache_list, saveec)
		print(f'Logged with ID {save_id}')

class flat_expt(expt):
	def __init__(self, agent, environment):
		super().__init__(agent,environment)

	def record_log(self, env_name, representation_type, n_trials, n_steps, **kwargs): ## TODO -- set up logging
		parent_folder = kwargs.get('dir', './Data/')
		log_name      = kwargs.get('file', 'test_bootstrap.csv')
		load_from     = kwargs.get('load_from', ' ')
		mock_log      = kwargs.get('mock_log', False)

		save_id = uuid.uuid4()
		timestamp = time.asctime(time.localtime())

		field_names = [
		'timestamp', #datetime experiment recorded
		'save_id',  # uuid
		'load_from',  # str
		'num_trials',  # int
		'num_events',  # int
		'env_name',  # str
		'representation', # str
		'MF_input_dims',  # arch
		'MF_lr',  # list
		'MF_temp',  # list
		'MF_gamma',  # float
		'EC_cache_limit',  # float
		'EC_temp',  # torch optim. class
		'EC_mem_decay',  # # string
		'EC_use_pvals',  # bool
		'EC_similarity_meas', # string
		'extra_info'
		]
		run_data = [timestamp, save_id, load_from, n_trials, n_steps, env_name, representation_type]
		network_keys = ['input_dims', 'lr', 'temperature']
		ec_keys = ['cache_limit', 'mem_temp', 'memory_envelope', 'use_pvals']
		agent_data = [self.agent.MFC.__dict__[k] for k in network_keys] + [self.agent.gamma]
		if self.agent.EC != None:
			ec_data = [self.agent.EC.__dict__[k] for k in ec_keys]
			ec_data.append(self.agent.EC.__dict__['distance_metric'])
		else:
			ec_data = ["None" for k in ec_keys] + ["None"]

		extra_info = kwargs.get('extra', [])

		log_jam = run_data + agent_data + ec_data + extra_info

		# write to logger
		if not os.path.exists(parent_folder+log_name):
			with open(parent_folder + log_name, 'a+', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(field_names)

		with open(parent_folder + log_name, 'a+', newline='') as file:
			writer = csv.writer(file)
			if mock_log:
				writer.writerow(log_jam+["mock log"])
			else:
				writer.writerow(log_jam)

		if not mock_log: ## can turn on flag to write to csv without saving files
			# save data
			with open(f'{parent_folder}results/{save_id}_data.p', 'wb') as savedata:
				pickle.dump(self.data, savedata)
			# save agent weights
			torch.save(self.agent.MFC.state_dict(), f=f'{parent_folder}agents/{save_id}.pt')
			# save episodic dictionary
			if self.agent.EC != None:
				with open(f'{parent_folder}ec_dicts/{save_id}_EC.p', 'wb') as saveec:
					pickle.dump(self.agent.EC.cache_list, saveec)
		print(f'Logged with ID {save_id}')

class shallow_expt(expt):
	def __init__(self, agent, environment):
		super().__init__(agent,environment)

	def record_log(self, env_name, representation_type, n_trials, n_steps, **kwargs): ## TODO -- set up logging
		parent_folder = kwargs.get('dir', './Data/')
		log_name      = kwargs.get('file', 'test_bootstrap.csv')
		load_from     = kwargs.get('load_from', ' ')
		mock_log      = kwargs.get('mock_log', False)

		save_id = uuid.uuid4()
		timestamp = time.asctime(time.localtime())

		field_names = [
		'timestamp', #datetime experiment recorded
		'save_id',  # uuid
		'load_from',  # str
		'num_trials',  # int
		'num_events',  # int
		'env_name',  # str
		'representation', # str
		'MF_input_dims',  # arch
		'MF_hidden_dims', # arch
		'MF_lr',  # list
		'MF_temp',  # list
		'MF_gamma',  # float
		'EC_cache_limit',  # float
		'EC_temp',  # torch optim. class
		'EC_mem_decay',  # # string
		'EC_use_pvals',  # bool
		'EC_similarity_meas', # string
		'extra_info'
		]
		run_data = [timestamp, save_id, load_from, n_trials, n_steps, env_name, representation_type]
		network_keys = ['input_dims', 'hidden_dims', 'lr', 'temperature']
		ec_keys = ['cache_limit', 'mem_temp', 'memory_envelope', 'use_pvals']
		agent_data = [self.agent.MFC.__dict__[k] for k in network_keys] + [self.agent.gamma]
		if self.agent.EC != None:
			ec_data = [self.agent.EC.__dict__[k] for k in ec_keys]
			ec_data.append(self.agent.EC.__dict__['distance_metric'])
		else:
			ec_data = ["None" for k in ec_keys] + ["None"]

		extra_info = kwargs.get('extra', [])

		log_jam = run_data + agent_data + ec_data + extra_info

		# write to logger
		if not os.path.exists(parent_folder+log_name):
			with open(parent_folder + log_name, 'a+', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(field_names)

		with open(parent_folder + log_name, 'a+', newline='') as file:
			writer = csv.writer(file)
			if mock_log:
				writer.writerow(log_jam+["mock log"])
			else:
				writer.writerow(log_jam)

		if not mock_log: ## can turn on flag to write to csv without saving files
			# save data
			with open(f'{parent_folder}results/{save_id}_data.p', 'wb') as savedata:
				pickle.dump(self.data, savedata)
			# save agent weights
			torch.save(self.agent.MFC.state_dict(), f=f'{parent_folder}agents/{save_id}.pt')
			# save episodic dictionary
			if self.agent.EC != None:
				with open(f'{parent_folder}ec_dicts/{save_id}_EC.p', 'wb') as saveec:
					pickle.dump(self.agent.EC.cache_list, saveec)
		print(f'Logged with ID {save_id}')



class flat_random_walk(flat_expt):
	def __init__(self, agent, environment):
		super().__init__(agent,environment)

	def single_step(self,trial):
		# get representation for given state of env. TODO: should be in agent to get representation?
		state_representation = self.agent.get_state_representation(self.state)
		readable = self.state

		# get action from agent
		action, log_prob, expected_value = self.agent.get_action(state_representation)
		action = np.random.choice(4)
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
		self.state = next_state
		return done

class flat_dist_return(flat_expt):
	def __init__(self,agent,envrionment):
		super().__init__(agent,envrionment)
		self.data['dist_rtn'] = []
		self.print_flag = True

	def single_step(self,trial):
		# get representation for given state of env.
		state_representation = self.agent.get_state_representation(self.state)
		readable = self.state

		# get action from agent
		action, log_prob, expected_value, distance, ec_readable = self.agent.get_action(state_representation)
		#get distance from EC

		# take step in environment
		next_state, reward, done, info = self.env.step(action)

		# end of event
		target_value = 0
		self.reward_sum += reward

		self.agent.log_event(episode=trial, event=self.agent.counter,
							 state=state_representation, action=action, reward=reward, next_state=next_state,
							 log_prob=log_prob, expected_value=expected_value, target_value=target_value,
							 done=done, readable_state=readable, distance=distance, ec_readable=ec_readable)
		self.agent.counter += 1
		self.state = next_state
		return done

	def end_of_trial(self, trial, logsnap=False):
		p, v = self.agent.finish_()

		if self.print_flag:
			print(len(self.agent.EC.cache_list.keys()))

		if len(self.agent.EC.cache_list.keys()) == self.agent.EC.cache_limit and self.print_flag:
			print(f"cache limit hit on trial {trial}")
			self.print_flag = False

		# temp
		if logsnap:
			states = [self.env.oneD2twoD(x) for x in list(self.agent.state_reps.keys())]
			observations = list(self.agent.state_reps.values())
			MF_pols, MF_vals = self.snapshot(states,observations)
			self.data['V_snap'].append(MF_vals)
			self.data['P_snap'].append(MF_pols)
		# /temp

		self.data['total_reward'].append(self.reward_sum) # removed for bootstrap expts
		self.data['loss'][0].append(p)
		self.data['loss'][1].append(v)
		self.data['dist_rtn'].append(self.agent.avg_dist_rtn[-1])

		if trial <= 10:
			self.running_rwdavg = np.mean(self.data['total_reward'])
		else:
			self.running_rwdavg = np.mean(self.data['total_reward'][-self.print_freq:])

		if trial % self.print_freq == 0:
			print(f"Episode: {trial}, Score: {self.reward_sum} (Running Avg:{self.running_rwdavg}) [{time.time() - self.t}s]")
			self.t = time.time()

class flat_ec_pol_track(flat_expt):
	def __init__(self,agent,envrionment):
		super().__init__(agent,envrionment)
		self.data['ec_dicts'] = []
		self.print_flag = True

	def end_of_trial(self, trial,logsnap):
		p, v = self.agent.finish_()

		self.data['total_reward'].append(self.reward_sum) # removed for bootstrap expts
		self.data['loss'][0].append(p)
		self.data['loss'][1].append(v)

		self.data['ec_dicts'].append(self.agent.EC.cache_list.copy())


		if trial <= 10:
			self.running_rwdavg = np.mean(self.data['total_reward'])
		else:
			self.running_rwdavg = np.mean(self.data['total_reward'][-self.print_freq:])

		if trial % self.print_freq == 0:
			print(f"Episode: {trial}, Score: {self.reward_sum} (Running Avg:{self.running_rwdavg}) [{time.time() - self.t}s]")
			self.t = time.time()


class gridworldExperiment(expt):
	def __init__(self, agent, environment, **kwargs):
		super(gridworldExperiment, self).__init__(agent, environment)

		# temp
		# only for gridworld environment
		self.sample_obs, self.sample_states = self.env.get_sample_obs()
		self.sample_reps = self.get_reps()

		self.policy_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])
		self.value_grid  = np.empty(self.env.shape)
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

		if self.agent.EC is not None:
			for ind, rep in enumerate(self.sample_reps):
				mem_pol = self.agent.EC.recall_mem(tuple(rep))
				state = self.sample_states[ind]
				mem_grid[state] = tuple(mem_pol)

			return pol_grid, val_grid, mem_grid

		else: return pol_grid, val_grid, []

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

			if self.agent.EC != None:
				ec_p = self.agent.EC.recall_mem(tuple(rep), timestep=self.agent.counter)
				EC_pols[s[0],s[1]] = tuple(ec_p)

		self.data['V_snap'].append(MF_vals)
		self.data['P_snap'].append(MF_pols)
		if self.agent.EC != None:
			self.data['EC_snap'].append(EC_pols)

	def run(self, NUM_TRIALS, NUM_EVENTS, **kwargs):
		self.print_freq = kwargs.get('printfreq', 100)
		log_snapshot = kwargs.get('snap', True)
		self.reset_data_logs()
		self.t = time.time()

		for trial in range(NUM_TRIALS):
			self.state = self.env.reset()
			self.reward_sum = 0

			for event in range(NUM_EVENTS):
				done = self.single_step(trial)

				if done:
					break

			self.take_snapshot() # only differentce from expt.run()
			if log_snapshot:
				self.end_of_trial(trial)


class Bootstrap_flat(gridworldExperiment):
	def __init__(self, agent, environment):
		super().__init__(agent, environment)
		self.policy_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])
		self.value_grid  = np.empty(self.env.shape)

	def track_trajectories(self,set):
		trajs = np.vstack(self.agent.transition_cache.transition_cache)
		sts, acts, rwds = trajs[:,10], trajs[:,3], trajs[:,4]
		data_package = [(x,y,z) for x, y,z in zip(sts,acts,rwds)]
		self.data['trajectories'][set].append(data_package)

	def end_of_trial(self, trial, set):
		if set == 0: # EC
			data_key = 'total_reward'
			p, v = self.agent.finish_()

			self.data['loss'][0].append(p)
			self.data['loss'][1].append(v)

		elif set == 1: # MF
			data_key = 'bootstrap_reward'
			# compute loss for MF guided trajectories, but don't use it to update weights
			p, v = self.agent.calc_loss()
			# use trajectories from MF guided trials in EC cache - try option without storing to EC
			self.agent.EC_storage()                   # usually handled in self.agent.finish_()
			self.agent.transition_cache.clear_cache() # usually handled in self.agent.finish_()

			self.data['mf_loss'][0].append(p)
			self.data['mf_loss'][1].append(v)

		else:
			raise Exception('Invalid Set Argument')

		# record cumulative reward to correct data structure
		self.data[data_key].append(self.reward_sum)

		if trial <= 10:
			self.running_rwdavg = np.mean(self.data[data_key])
		else:
			self.running_rwdavg = np.mean(self.data[data_key][-10:-1])

		if trial % self.print_freq == 0:
			print(f"Episode: {trial}, {['EC','MF'][set]} Score: {self.reward_sum} (Running Avg:{self.running_rwdavg}) [{time.time() - self.t}s]")
			self.t = time.time()

	def run(self, NUM_TRIALS, NUM_EVENTS, **kwargs):
		self.print_freq = kwargs.get('printfreq', 100)
		self.reset_data_logs()
		self.data['bootstrap_reward'] = []
		self.data['mf_loss'] = [[],[]]
		self.data['trajectories'] = [[],[]]
		self.data['cache_size'] = []
		self.t = time.time()

		for trial in range(NUM_TRIALS):
			for set in [0,1]:  ## set 0: episodic control, use this for weight updates; set 1: MF control, no updates
				self.state = self.env.reset()
				self.reward_sum = 0

				# set action picker
				if set == 0:
					self.agent.get_action = self.agent.EC_action
				else:
					self.agent.get_action = self.agent.MF_action

				# get a trajectory in the environment up to NUM_EVENTS steps
				for event in range(NUM_EVENTS):
					done = self.single_step(trial)
					if done:
						break

				self.end_of_trial(trial, set)

				# keep track only for printing at the end of a trial cycle (EC/MF set)
				if set == 0:
					ecrwd = self.reward_sum
				elif set ==1:
					mfrwd = self.reward_sum

			#print(f'{trial}: EC {ecrwd:.3f}  /  MF {mfrwd:.3f}')
			#self.take_snapshot()

	def record_log(self, env_name, representation_type, n_trials, n_steps, **kwargs): ## TODO -- set up logging
		parent_folder = kwargs.get('dir', './Data/')
		log_name      = kwargs.get('file', 'test_bootstrap.csv')
		load_from     = kwargs.get('load_from', ' ')
		mock_log      = kwargs.get('mock_log', False)

		save_id = uuid.uuid4()
		timestamp = time.asctime(time.localtime())

		field_names = [
		'timestamp', #datetime experiment recorded
		'save_id',  # uuid
		'load_from',  # str
		'num_trials',  # int
		'num_events',  # int
		'env_name',  # str
		'representation', # str
		'MF_input_dims',  # arch
		'MF_lr',  # list
		'MF_temp',  # list
		'MF_gamma',  # float
		'EC_cache_limit',  # float
		'EC_temp',  # torch optim. class
		'EC_mem_decay',  # # string
		'EC_use_pvals',  # bool
		'EC_similarity_meas', # string
		'extra_info'
		]
		run_data = [timestamp, save_id, load_from, n_trials, n_steps, env_name, representation_type]
		network_keys = ['input_dims', 'lr', 'temperature']
		ec_keys = ['cache_limit', 'mem_temp', 'memory_envelope', 'use_pvals']
		agent_data = [self.agent.MFC.__dict__[k] for k in network_keys] + [self.agent.gamma]
		if self.agent.EC != None:
			ec_data = [self.agent.EC.__dict__[k] for k in ec_keys]
			ec_data.append(self.agent.EC.__dict__['distance_metric'])
		else:
			ec_data = ["None" for k in ec_keys] + ["None"]

		extra_info = kwargs.get('extra', [])

		log_jam = run_data + agent_data + ec_data + extra_info

		# write to logger
		if not os.path.exists(parent_folder+log_name):
			with open(parent_folder + log_name, 'a+', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(field_names)

		with open(parent_folder + log_name, 'a+', newline='') as file:
			writer = csv.writer(file)
			if mock_log:
				writer.writerow(log_jam+["mock log"])
			else:
				writer.writerow(log_jam)

		if not mock_log: ## can turn on flag to write to csv without saving files
			# save data
			with open(f'{parent_folder}results/{save_id}_data.p', 'wb') as savedata:
				pickle.dump(self.data, savedata)
			# save agent weights
			torch.save(self.agent.MFC.state_dict(), f=f'{parent_folder}agents/{save_id}.pt')
			# save episodic dictionary
			if self.agent.EC != None:
				with open(f'{parent_folder}ec_dicts/{save_id}_EC.p', 'wb') as saveec:
					pickle.dump(self.agent.EC.cache_list, saveec)
		print(f'Logged with ID {save_id}')

class Bootstrap_shallow(Bootstrap_flat):
	def __init__(self, agent, environment):
		super().__init__(agent, environment)
		self.policy_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])
		self.value_grid  = np.empty(self.env.shape)

	def record_log(self, env_name, representation_type, n_trials, n_steps, **kwargs): ## TODO -- set up logging
		parent_folder = kwargs.get('dir', './Data/')
		log_name      = kwargs.get('file', 'test_bootstrap.csv')
		load_from     = kwargs.get('load_from', ' ')
		mock_log      = kwargs.get('mock_log', False)

		save_id = uuid.uuid4()
		timestamp = time.asctime(time.localtime())

		field_names = [
		'timestamp', #datetime experiment recorded
		'save_id',  # uuid
		'load_from',  # str
		'num_trials',  # int
		'num_events',  # int
		'env_name',  # str
		'representation', # str
		'MF_input_dims',  # arch
		'MF_hidden_dims', # arch
		'MF_lr',  # list
		'MF_temp',  # list
		'MF_gamma',  # float
		'EC_cache_limit',  # float
		'EC_temp',  # torch optim. class
		'EC_mem_decay',  # # string
		'EC_use_pvals',  # bool
		'EC_similarity_meas', # string
		'extra_info'
		]
		run_data = [timestamp, save_id, load_from, n_trials, n_steps, env_name, representation_type]
		network_keys = ['input_dims', 'hidden_dims', 'lr', 'temperature']
		ec_keys = ['cache_limit', 'mem_temp', 'memory_envelope', 'use_pvals']
		agent_data = [self.agent.MFC.__dict__[k] for k in network_keys] + [self.agent.gamma]
		if self.agent.EC != None:
			ec_data = [self.agent.EC.__dict__[k] for k in ec_keys]
			ec_data.append(self.agent.EC.__dict__['distance_metric'])
		else:
			ec_data = ["None" for k in ec_keys] + ["None"]

		extra_info = kwargs.get('extra', [])

		log_jam = run_data + agent_data + ec_data + extra_info

		# write to logger
		if not os.path.exists(parent_folder+log_name):
			with open(parent_folder + log_name, 'a+', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(field_names)

		with open(parent_folder + log_name, 'a+', newline='') as file:
			writer = csv.writer(file)
			if mock_log:
				writer.writerow(log_jam+["mock log"])
			else:
				writer.writerow(log_jam)

		if not mock_log: ## can turn on flag to write to csv without saving files
			# save data
			with open(f'{parent_folder}results/{save_id}_data.p', 'wb') as savedata:
				pickle.dump(self.data, savedata)
			# save agent weights
			torch.save(self.agent.MFC.state_dict(), f=f'{parent_folder}agents/{save_id}.pt')
			# save episodic dictionary
			if self.agent.EC != None:
				with open(f'{parent_folder}ec_dicts/{save_id}_EC.p', 'wb') as saveec:
					pickle.dump(self.agent.EC.cache_list, saveec)
		print(f'Logged with ID {save_id}')


#TODO write bootstrap as more general class
class Bootstrap_viewMF(gridworldExperiment):
	def __init__(self, agent, environment):
		super(Bootstrap_viewMF,self).__init__(agent, environment)
		self.policy_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])
		self.value_grid  = np.empty(self.env.shape)
		self.data['weights'] = {'h0':[],'h1':[],'p':[],'v':[]} # h0, h1, p, v

	def track_weights(self):
		for y in range(len(self.agent.MFC.hidden)):
			layer = self.agent.MFC.hidden[y]
			grad_norm    = torch.norm(layer.weight.grad)
			self.data['weights'][f'h{y}'].append(grad_norm)

		for x in range(len(self.agent.MFC.output)):
			layer = self.agent.MFC.output[x]
			grad_norm = torch.norm(layer.weight.grad)
			self.data['weights'][['p','v'][x]].append(grad_norm)

	def track_trajectories(self,set):
		trajs = np.vstack(self.agent.transition_cache.transition_cache)
		sts, acts, rwds = trajs[:,10], trajs[:,3], trajs[:,4]
		data_package = [(x,y,z) for x, y,z in zip(sts,acts,rwds)]
		self.data['trajectories'][set].append(data_package)

	def end_of_trial(self, trial, set):
		if set == 0: # EC
			data_key = 'total_reward'
			p, v = self.agent.finish_()

			self.data['loss'][0].append(p)
			self.data['loss'][1].append(v)

		elif set == 1: # MF
			data_key = 'bootstrap_reward'
			# compute loss for MF guided trajectories, but don't use it to update weights
			p, v = self.agent.calc_loss()
			# use trajectories from MF guided trials in EC cache - try option without storing to EC
			self.agent.EC_storage()                   # usually handled in self.agent.finish_()
			self.agent.transition_cache.clear_cache() # usually handled in self.agent.finish_()

			self.data['mf_loss'][0].append(p)
			self.data['mf_loss'][1].append(v)

		else:
			raise Exception('Invalid Set Argument')

		# record cumulative reward to correct data structure
		self.data[data_key].append(self.reward_sum)

		if trial <= 10:
			self.running_rwdavg = np.mean(self.data[data_key])
		else:
			self.running_rwdavg = np.mean(self.data[data_key][-10:-1])

		if trial % self.print_freq == 0:
			print(f"Episode: {trial}, {['EC','MF'][set]} Score: {self.reward_sum} (Running Avg:{self.running_rwdavg}) [{time.time() - self.t}s]")
			self.t = time.time()

	def run(self, NUM_TRIALS, NUM_EVENTS, **kwargs):
		self.print_freq = kwargs.get('printfreq', 100)
		self.reset_data_logs()
		self.data['bootstrap_reward'] = []
		self.data['mf_loss'] = [[],[]]
		self.data['trajectories'] = [[],[]]
		self.data['cache_size'] = []
		self.t = time.time()

		for trial in range(NUM_TRIALS):
			for set in [0,1]:  ## set 0: episodic control, use this for weight updates; set 1: MF control, no updates
				self.state = self.env.reset()
				self.reward_sum = 0

				# set action picker
				if set == 0:
					self.agent.get_action = self.agent.EC_action
				else:
					self.agent.get_action = self.agent.MF_action

				# get a trajectory in the environment up to NUM_EVENTS steps
				for event in range(NUM_EVENTS):
					done = self.single_step(trial)
					if done:
						break

				self.end_of_trial(trial, set)

				# keep track only for printing at the end of a trial cycle (EC/MF set)
				if set == 0:
					ecrwd = self.reward_sum
				elif set ==1:
					mfrwd = self.reward_sum

			print(f'{trial}: EC {ecrwd:.3f}  /  MF {mfrwd:.3f}')
			self.take_snapshot()
			self.track_weights()

class Bootstrap_forgetfulMF(Bootstrap_viewMF):
	## same as Bootstrap_viewMF but do not record MF trials to EC
	def __init__(self, agent, environment):
		super(Bootstrap_forgetfulMF,self).__init__(agent, environment)

	def end_of_trial(self, trial, set):
		if set == 0: # EC
			data_key = 'total_reward'
			p, v = self.agent.finish_()

			self.data['loss'][0].append(p)
			self.data['loss'][1].append(v)

		elif set == 1: # MF
			data_key = 'bootstrap_reward'
			# compute loss for MF guided trajectories, but don't use it to update weights
			p, v = self.agent.calc_loss()
			self.agent.transition_cache.clear_cache() # usually handled in self.agent.finish_()

			self.data['mf_loss'][0].append(p)
			self.data['mf_loss'][1].append(v)

		else:
			raise Exception('Invalid Set Argument')

		# record cumulative reward to correct data structure
		self.data[data_key].append(self.reward_sum)

		if trial <= 10:
			self.running_rwdavg = np.mean(self.data[data_key])
		else:
			self.running_rwdavg = np.mean(self.data[data_key][-10:-1])

		if trial % self.print_freq == 0:
			print(f"Episode: {trial}, {['EC','MF'][set]} Score: {self.reward_sum} (Running Avg:{self.running_rwdavg}) [{time.time() - self.t}s]")
			self.t = time.time()

class Bootstrap_interleaved(Bootstrap_viewMF):
	def __init__(self, agent, environment):
		super(Bootstrap_interleaved,self).__init__(agent, environment)

	def end_of_trial(self, trial, set):
		## regardless of action controller, use trajectory to update weights
		## this also means MF trajectories get written to EC cache
		p, v = self.agent.finish_()
		if set == 0: # EC
			data_key = 'total_reward'
			self.data['loss'][0].append(p)
			self.data['loss'][1].append(v)

		elif set == 1: # MF
			data_key = 'bootstrap_reward'
			self.data['mf_loss'][0].append(p)
			self.data['mf_loss'][1].append(v)
			self.track_weights()

		else:
			raise Exception('Invalid Set Argument')

		# record cumulative reward to correct data structure
		self.data[data_key].append(self.reward_sum)

		if trial <= 10:
			self.running_rwdavg = np.mean(self.data[data_key])
		else:
			self.running_rwdavg = np.mean(self.data[data_key][-10:-1])

		if trial % self.print_freq == 0:
			print(f"Episode: {trial}, {['EC','MF'][set]} Score: {self.reward_sum} (Running Avg:{self.running_rwdavg}) [{time.time() - self.t}s]")
			self.t = time.time()

class Bootstrap_verbose(Bootstrap_viewMF):
	## same as Bootstrap_viewMF but do not record MF trials to EC
	def __init__(self, agent, environment):
		super(Bootstrap_verbose,self).__init__(agent, environment)

	def EC_storage(self):
		mem_dict = {}
		buffer = np.vstack(self.agent.transition_cache.transition_cache)

		states    = buffer[:,2]
		actions   = buffer[:,3]
		returns   = buffer[:,8]
		timesteps = buffer[:,1]
		readable  = buffer[:,10]
		trial     = buffer[-1,0]

		for s, a, r, event, rdbl in zip(states,actions,returns,timesteps,readable):
			mem_dict['activity']  = tuple(s)
			mem_dict['action']    = a
			mem_dict['delta']     = r
			mem_dict['timestamp'] = event
			mem_dict['readable']  = rdbl
			mem_dict['trial']     = trial

			print(np.where(s==1.),mem_dict)

			self.agent.EC.add_mem(mem_dict)
			print(self.agent.EC.cache_list[mem_dict['activity']])
			print(self.agent.EC.recall_mem(mem_dict['activity']))

	'''
	def single_step(self,trial):
		# get representation for given state of env. TODO: should be in agent to get representation?
		state_representation = self.get_representation(self.state)
		readable = 0

		# get action from agent
		action, log_prob, expected_value = self.agent.get_action(state_representation)
		# take step in environment
		next_state, reward, done, info = self.env.step(action)

		print(f"EVENT:{self.state}/{action}/{reward}/{next_state}")

		# end of event
		target_value = 0
		self.reward_sum += reward

		self.agent.log_event(episode=trial, event=self.agent.counter,
							 state=state_representation, action=action, reward=reward, next_state=next_state,
							 log_prob=log_prob, expected_value=expected_value, target_value=target_value,
							 done=done, readable_state=readable)
		self.agent.counter += 1
		self.state = next_state
		return done
	'''

	def end_of_trial(self, trial, set):
		if set == 0: # EC
			data_key = 'total_reward'
			p, v = self.agent.update()
			print(self.agent.transition_cache.transition_cache)
			if self.agent.EC != None:
				self.EC_storage()

			self.agent.transition_cache.clear_cache()

			self.data['loss'][0].append(p)
			self.data['loss'][1].append(v)

		elif set == 1: # MF
			data_key = 'bootstrap_reward'
			# compute loss for MF guided trajectories, but don't use it to update weights
			p, v = self.agent.calc_loss()
			# use trajectories from MF guided trials in EC cache - try option without storing to EC
			self.agent.EC_storage()                   # usually handled in self.agent.finish_()
			self.agent.transition_cache.clear_cache() # usually handled in self.agent.finish_()

			self.data['mf_loss'][0].append(p)
			self.data['mf_loss'][1].append(v)

		else:
			raise Exception('Invalid Set Argument')

		# record cumulative reward to correct data structure
		self.data[data_key].append(self.reward_sum)

		if trial <= 10:
			self.running_rwdavg = np.mean(self.data[data_key])
		else:
			self.running_rwdavg = np.mean(self.data[data_key][-10:-1])

		if trial % self.print_freq == 0:
			print(f"Episode: {trial}, {['EC','MF'][set]} Score: {self.reward_sum} (Running Avg:{self.running_rwdavg}) [{time.time() - self.t}s]")
			self.t = time.time()

	def run(self, NUM_TRIALS, NUM_EVENTS, **kwargs):
		self.print_freq = kwargs.get('printfreq', 100)
		self.reset_data_logs()
		self.data['bootstrap_reward'] = []
		self.data['mf_loss'] = [[],[]]
		self.data['trajectories'] = [[],[]]
		self.data['cache_size'] = []
		self.t = time.time()

		for trial in range(NUM_TRIALS):
			for set in [0,1]:  ## set 0: episodic control, use this for weight updates; set 1: MF control, no updates
				self.state = self.env.reset()
				self.reward_sum = 0

				# set action picker
				if set == 0:
					self.agent.get_action = self.agent.EC_action
				else:
					self.agent.get_action = self.agent.MF_action

				# get a trajectory in the environment up to NUM_EVENTS steps
				for event in range(NUM_EVENTS):
					done = self.single_step(trial)
					if done:
						break

				self.end_of_trial(trial, set)

				# keep track only for printing at the end of a trial cycle (EC/MF set)
				if set == 0:
					ecrwd = self.reward_sum
				elif set ==1:
					mfrwd = self.reward_sum

			print(f'{trial}: EC {ecrwd:.3f}  /  MF {mfrwd:.3f}')
			self.take_snapshot()
			self.track_weights()


#### JUNKYARD
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


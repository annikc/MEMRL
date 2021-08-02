# Episodic Memory Module Object Class and Related Functions
# Written and maintained by Annik Carson
# Last updated: July 2020
#
# =====================================
#           IMPORT MODULES            #
# =====================================
from __future__ import division, print_function
import numpy as np
from scipy.spatial.distance import cdist
import sys
sys.path.append('../../../modules/')
from modules.Utils import softmax

class EpisodicMemory(object):
	def __init__(self, entry_size, cache_limit,**kwargs):
		self.cache_list 	 = {}								# memory bank object
		self.cache_limit 	 = cache_limit                      # size of memory bank
		self.n_actions		 = entry_size						# number of rows in each memory unit

		self.mem_temp        = kwargs.get('mem_temp', 1)        # softmax temp for memory recall
		self.memory_envelope = kwargs.get('mem_envelope', 50)   # speed of memory decay
		self.use_pvals       = kwargs.get('pvals', False)

		self.distance_metric = kwargs.get('distance', 'chebyshev')

	def reset_cache(self):
		self.cache_list.clear()

	def calc_envelope(self,halfmax):
		'''
		:param halfmax: x value for which envelope will give sech(x/env) = 0.5
		:return: envelope value
		e^(x/env) = (2+np.sqrt(3)) for sech(x/env) = 0.5
		Hence x/env = np.log(2+np.sqrt(3)) and env = x/ np.log(2+np.sqrt(3))
		'''
		return halfmax/np.log(2+np.sqrt(3))

	def add_mem(self, item):
		activity 	= item['activity']
		action		= item['action']
		delta 		= item['delta'] #return
		timestamp	= item['timestamp']
		trial       = item['trial'] # episode (i.e. collection of transitions)
		#
		readable    = item['readable'] # 2d coordinate just to know what state we're looking at

		# Case 1: memory is not full
		if len(self.cache_list) < self.cache_limit:
			# Case 1a: key does not yet exist
			if activity not in self.cache_list.keys(): # if no key for this state exists already, add new one
				mem_entry = np.empty((self.n_actions, 2))
				mem_entry[:,0] = np.nan # initialize deltas to nan
				mem_entry[:,1] = np.inf # initialize timestamps to inf
				self.cache_list[activity] = [mem_entry, np.inf, None]
			# Case 1b: key exists, add or replace relevant info in mem container
			self.cache_list[activity][0][action] = [delta, trial]
			self.cache_list[activity][1] = timestamp
			self.cache_list[activity][2] = readable
			#self.visited_states.append(readable)
		# Case 2: memory is full
		else:
			# Case 2a: key does not yet exist
			if activity not in self.cache_list.keys():
				# choose key to be removed
				cache_keys = list(self.cache_list.keys())
				persistence_ = [x[1] for x in self.cache_list.values()] # get list of all timestamp flags
				lp = persistence_.index(min(persistence_))              # find entry that was updated the LEAST recently
				old_activity = cache_keys[lp]                           # get key in dictionary corresponding to oldest timestep flag
				del self.cache_list[old_activity]                       # delete item from dictionary with oldest timestamp flag

				# add new mem container
				mem_entry = np.empty((self.n_actions, 2))
				mem_entry[:,0] = np.nan
				mem_entry[:,1] = np.inf # initialize entries to nan
				self.cache_list[activity] = [mem_entry, np.inf, None]
			# Case2b: key exists, add or replace relevant info in mem container
			self.cache_list[activity][0][action] = [delta, trial]
			self.cache_list[activity][1] = timestamp
			self.cache_list[activity][2] = readable

	def recall_mem(self, key, timestep=0, **kwargs):
		'''
		pass in key: get most similar entry and return cosine sim score

		confidence score = scaled by cosine sim

		'''
		mem_temp = kwargs.get('mem_temp', self.mem_temp)
		#specify decay envelope for memory relevance calculation
		envelope = kwargs.get('pval_decay_env', self.memory_envelope)
		
		if len(self.cache_list) == 0:
			random_policy = softmax(np.zeros(self.n_actions))
			return random_policy
		else:
			lin_act, distance = self.similarity_measure(key) # returns the most similar key, as well as the cosine similarity measure
			memory       = np.nan_to_num(self.cache_list[lin_act][0])
			deltas       = memory[:,0]
			similarity = 1 ## using key sim
			if self.use_pvals:
				times = abs(timestep - memory[:, 1])
				pvals = self.make_pvals(times, envelope=envelope)
				policy = softmax( similarity*np.multiply(deltas,pvals), T=mem_temp)
			else:
				policy = softmax( similarity*deltas, T=mem_temp)
			return policy

	def make_pvals(self, p, **kwargs):
		if isinstance(p,int):
			ratio = p/self.memory_envelope
			return np.round(1 / np.cosh(ratio), 8)
		else:
			ratio = np.around(p/self.memory_envelope, 8)
			return np.round(1 / np.cosh(ratio), 8)

	def cosine_sim(self, key):
		# make list of memory keys
		mem_cache = np.asarray(list(self.cache_list.keys()))

		entry = np.asarray(key)
		# compute cosine similarity measure
		mqt = np.dot(mem_cache, entry)
		norm = np.linalg.norm(mem_cache, axis=1) * np.linalg.norm(entry)
		cosine_similarity = mqt / norm

		lin_act = mem_cache[np.argmax(cosine_similarity)]
		return  tuple(lin_act), max(cosine_similarity)

	def similarity_measure(self, key):
		mem_cache = np.asarray(list(self.cache_list.keys()))
		entry 	  = np.asarray(key)
		distance  = cdist([entry], mem_cache, metric=self.distance_metric)[0]

		closest_entry = mem_cache[np.argmin(distance)]
		return tuple(closest_entry), min(distance)

class distance_report_EpisodicMemory(object):
	def __init__(self, entry_size, cache_limit,**kwargs):
		self.cache_list 	 = {}								# memory bank object
		self.cache_limit 	 = cache_limit                      # size of memory bank
		self.n_actions		 = entry_size						# number of rows in each memory unit

		self.mem_temp        = kwargs.get('mem_temp', 1)        # softmax temp for memory recall
		self.memory_envelope = kwargs.get('mem_envelope', 50)   # speed of memory decay
		self.use_pvals       = kwargs.get('pvals', False)

		self.distance_metric = kwargs.get('distance', 'euclidean')

	def reset_cache(self):
		self.cache_list.clear()

	def calc_envelope(self,halfmax):
		'''
		:param halfmax: x value for which envelope will give sech(x/env) = 0.5
		:return: envelope value
		e^(x/env) = (2+np.sqrt(3)) for sech(x/env) = 0.5
		Hence x/env = np.log(2+np.sqrt(3)) and env = x/ np.log(2+np.sqrt(3))
		'''
		return halfmax/np.log(2+np.sqrt(3))

	def add_mem(self, item):
		activity 	= item['activity']
		action		= item['action']
		delta 		= item['delta'] #return
		timestamp	= item['timestamp']
		trial       = item['trial'] # episode (i.e. collection of transitions)
		#
		readable    = item['readable'] # 2d coordinate just to know what state we're looking at

		# Case 1: memory is not full
		if len(self.cache_list) < self.cache_limit:
			# Case 1a: key does not yet exist
			if activity not in self.cache_list.keys(): # if no key for this state exists already, add new one
				mem_entry = np.empty((self.n_actions, 2))
				mem_entry[:,0] = np.nan # initialize deltas to nan
				mem_entry[:,1] = np.inf # initialize timestamps to inf
				self.cache_list[activity] = [mem_entry, np.inf, None]
			# Case 1b: key exists, add or replace relevant info in mem container
			self.cache_list[activity][0][action] = [delta, trial]
			self.cache_list[activity][1] = timestamp
			self.cache_list[activity][2] = readable
			#self.visited_states.append(readable)
		# Case 2: memory is full
		else:
			# Case 2a: key does not yet exist
			if activity not in self.cache_list.keys():
				# choose key to be removed
				cache_keys = list(self.cache_list.keys())
				persistence_ = [x[1] for x in self.cache_list.values()] # get list of all timestamp flags
				lp = persistence_.index(min(persistence_))              # find entry that was updated the LEAST recently
				old_activity = cache_keys[lp]                           # get key in dictionary corresponding to oldest timestep flag
				del self.cache_list[old_activity]                       # delete item from dictionary with oldest timestamp flag

				# add new mem container
				mem_entry = np.empty((self.n_actions, 2))
				mem_entry[:,0] = np.nan
				mem_entry[:,1] = np.inf # initialize entries to nan
				self.cache_list[activity] = [mem_entry, np.inf, None]
			# Case2b: key exists, add or replace relevant info in mem container
			self.cache_list[activity][0][action] = [delta, trial]
			self.cache_list[activity][1] = timestamp
			self.cache_list[activity][2] = readable

	def recall_mem(self, key, timestep=0, **kwargs):
		'''
		pass in key: get most similar entry and return cosine sim score

		confidence score = scaled by cosine sim

		'''
		mem_temp = kwargs.get('mem_temp', self.mem_temp)
		#specify decay envelope for memory relevance calculation
		envelope = kwargs.get('pval_decay_env', self.memory_envelope)

		report_dist = kwargs.get('report_dist', False)


		if len(self.cache_list) == 0:
			policy = softmax(np.zeros(self.n_actions))
			distance = np.nan
			readable_state = np.nan
		else:
			lin_act, distance = self.similarity_measure(key) # returns the most similar key, as well as the cosine similarity measure
			readable_state = self.cache_list[lin_act][2]
			memory       = np.nan_to_num(self.cache_list[lin_act][0])
			deltas       = memory[:,0]
			similarity = 1 ## using key sim
			if self.use_pvals:
				times = abs(timestep - memory[:, 1])
				pvals = self.make_pvals(times, envelope=envelope)
				policy = softmax( similarity*np.multiply(deltas,pvals), T=mem_temp)
			else:
				policy = softmax( similarity*deltas, T=mem_temp)
		if report_dist:
			return policy, distance, readable_state
		else:
			return policy

	def make_pvals(self, p, **kwargs):
		if isinstance(p,int):
			ratio = p/self.memory_envelope
			return np.round(1 / np.cosh(ratio), 8)
		else:
			ratio = np.around(p/self.memory_envelope, 8)
			return np.round(1 / np.cosh(ratio), 8)

	def cosine_sim(self, key):
		# make list of memory keys
		mem_cache = np.asarray(list(self.cache_list.keys()))

		entry = np.asarray(key)
		# compute cosine similarity measure
		mqt = np.dot(mem_cache, entry)
		norm = np.linalg.norm(mem_cache, axis=1) * np.linalg.norm(entry)
		cosine_similarity = mqt / norm

		lin_act = mem_cache[np.argmax(cosine_similarity)]
		return  tuple(lin_act), max(cosine_similarity)

	def similarity_measure(self, key):
		mem_cache = np.asarray(list(self.cache_list.keys()))
		entry 	  = np.asarray(key)
		distance  = cdist([entry], mem_cache, metric=self.distance_metric)[0]

		closest_entry = mem_cache[np.argmin(distance)]
		return tuple(closest_entry), min(distance)

class random_forget_EC(EpisodicMemory):
	def __init__(self, entry_size, cache_limit, **kwargs):
		super(random_forget_EC, self).__init__(entry_size,cache_limit,**kwargs)
		self.forgotten_states={}

	def add_mem(self, item):
		activity 	= item['activity']
		action		= item['action']
		delta 		= item['delta'] #return
		timestamp	= item['timestamp']
		trial       = item['trial'] # episode (i.e. collection of transitions)
		#
		readable    = item['readable'] # 2d coordinate just to know what state we're looking at

		# Case 1: memory is not full
		if len(self.cache_list) < self.cache_limit:
			# Case 1a: key does not yet exist
			if activity not in self.cache_list.keys(): # if no key for this state exists already, add new one
				mem_entry = np.empty((self.n_actions, 2))
				mem_entry[:,0] = np.nan # initialize deltas to nan
				mem_entry[:,1] = np.inf # initialize timestamps to inf
				self.cache_list[activity] = [mem_entry, np.inf, None]
			# Case 1b: key exists, add or replace relevant info in mem container
			self.cache_list[activity][0][action] = [delta, trial]
			self.cache_list[activity][1] = timestamp
			self.cache_list[activity][2] = readable
			#self.visited_states.append(readable)
		# Case 2: memory is full
		else:
			# Case 2a: key does not yet exist
			if activity not in self.cache_list.keys():
				# choose key to be removed
				cache_keys = list(self.cache_list.keys())

				rand_index = np.random.choice(len(cache_keys))
				old_activity = cache_keys[rand_index]                   # get key in dictionary corresponding to random index

				## new for this class -- record the state index of the key to be discarded
				forgotten_state = self.cache_list[old_activity][2] #readable_state
				if forgotten_state in self.forgotten_states.keys():
					self.forgotten_states[forgotten_state] +=1
				else:
					self.forgotten_states[forgotten_state] = 1

				del self.cache_list[old_activity]                       # delete item from dictionary

				# add new mem container
				mem_entry = np.empty((self.n_actions, 2))
				mem_entry[:,0] = np.nan
				mem_entry[:,1] = np.inf # initialize entries to nan
				self.cache_list[activity] = [mem_entry, np.inf, None]
			# Case2b: key exists, add or replace relevant info in mem container
			self.cache_list[activity][0][action] = [delta, trial]
			self.cache_list[activity][1] = timestamp
			self.cache_list[activity][2] = readable

class EC_track_forgotten_states(EpisodicMemory):
	def __init__(self,entry_size,cache_limit,**kwargs):
		super(EC_track_forgotten_states, self).__init__(entry_size,cache_limit,**kwargs)
		self.forgotten_states={}

	def add_mem(self, item):
		activity 	= item['activity']
		action		= item['action']
		delta 		= item['delta'] #return
		timestamp	= item['timestamp']
		trial       = item['trial'] # episode (i.e. collection of transitions)
		#
		readable    = item['readable'] # 2d coordinate just to know what state we're looking at

		# Case 1: memory is not full
		if len(self.cache_list) < self.cache_limit:
			# Case 1a: key does not yet exist
			if activity not in self.cache_list.keys(): # if no key for this state exists already, add new one
				mem_entry = np.empty((self.n_actions, 2))
				mem_entry[:,0] = np.nan # initialize deltas to nan
				mem_entry[:,1] = np.inf # initialize timestamps to inf
				self.cache_list[activity] = [mem_entry, np.inf, None]
			# Case 1b: key exists, add or replace relevant info in mem container
			self.cache_list[activity][0][action] = [delta, trial]
			self.cache_list[activity][1] = timestamp
			self.cache_list[activity][2] = readable
			#self.visited_states.append(readable)
		# Case 2: memory is full
		else:
			# Case 2a: key does not yet exist
			if activity not in self.cache_list.keys():
				# choose key to be removed
				cache_keys = list(self.cache_list.keys())
				persistence_ = [x[1] for x in self.cache_list.values()] # get list of all timestamp flags
				lp = persistence_.index(min(persistence_))              # find entry that was updated the LEAST recently
				old_activity = cache_keys[lp]                           # get key in dictionary corresponding to oldest timestep flag

				## new for this class -- record the state index of the key to be discarded
				forgotten_state = self.cache_list[old_activity][2] #readable_state
				if forgotten_state in self.forgotten_states.keys():
					self.forgotten_states[forgotten_state] +=1
				else:
					self.forgotten_states[forgotten_state] = 1

				del self.cache_list[old_activity]                       # delete item from dictionary with oldest timestamp flag

				# add new mem container
				mem_entry = np.empty((self.n_actions, 2))
				mem_entry[:,0] = np.nan
				mem_entry[:,1] = np.inf # initialize entries to nan
				self.cache_list[activity] = [mem_entry, np.inf, None]
			# Case2b: key exists, add or replace relevant info in mem container
			self.cache_list[activity][0][action] = [delta, trial]
			self.cache_list[activity][1] = timestamp
			self.cache_list[activity][2] = readable


class forget_least_recently_accessed_memory(EC_track_forgotten_states):
	def __init__(self,entry_size,cache_limit,**kwargs):
		super(EC_track_forgotten_states, self).__init__(entry_size,cache_limit,**kwargs)
		self.forgotten_states={}

	def recall_mem(self, key, timestep, **kwargs):
		'''
		pass in key: get most similar entry and return cosine sim score

		confidence score = scaled by cosine sim

		'''
		mem_temp = kwargs.get('mem_temp', self.mem_temp)
		#specify decay envelope for memory relevance calculation
		envelope = kwargs.get('pval_decay_env', self.memory_envelope)


		if len(self.cache_list) == 0:
			random_policy = softmax(np.zeros(self.n_actions))
			return random_policy
		else:
			lin_act, distance = self.similarity_measure(key) # returns the most similar key, as well as the cosine similarity measure
			memory       = np.nan_to_num(self.cache_list[lin_act][0])
			## update timestep in memory
			self.cache_list[lin_act][1] = timestep
			deltas       = memory[:,0]
			similarity = 1 ## using key sim
			if self.use_pvals:
				times = abs(timestep - memory[:, 1])
				pvals = self.make_pvals(times, envelope=envelope)
				policy = softmax( similarity*np.multiply(deltas,pvals), T=mem_temp)
			else:
				policy = softmax( similarity*deltas, T=mem_temp)
			return policy


class RandomPolicy_EC(EpisodicMemory):
    def __init__(self, entry_size, cache_limit):
        super(RandomPolicy_EC, self).__init__(entry_size, cache_limit)

    def recall_mem(self, key):
        # regardless of state return a random policy
        return softmax(np.ones(self.entry_size))

    def add_mem(self, item):
        # don't need to keep records -- will just slow computations down
        pass

def calc_envelope(halfmax):
	'''
	:param halfmax: x value for which envelope will give sech(x/env) = 0.5
	:return: envelope value
	e^(x/env) = (2+np.sqrt(3)) for sech(x/env) = 0.5
	Hence x/env = np.log(2+np.sqrt(3)) and env = x/ np.log(2+np.sqrt(3))
	'''
	return halfmax/np.log(2+np.sqrt(3))


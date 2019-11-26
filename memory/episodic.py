#### Episodic Memory Cache

'''
Object Classes and Relevant Functions for Episodic Memory Module
Author: Annik Carson 
--  June 2018
'''

# =====================================
#           IMPORT MODULES            #
# =====================================
from __future__ import division, print_function
import numpy as np

class ep_mem(object):
	def __init__(self, model, cache_limit,**kwargs):
		self.cache_list 		= {}								# memory bank object
		self.cache_limit 		= cache_limit                       # size of memory bank
		self.n_actions			= model.layers[-1]					# number of rows in each memory unit

		self.memory_envelope 	= kwargs.get('mem_envelope', 50)    # speed of memory decay

		##
		self.mem_factor            = 0.5
		self.reward_unseen         = True
		self.time_since_last_reward= 0
		self.confidence_score      = 0
		self.cs_max                = 0

	def reset_cache(self):
		self.cache_list.clear()

	def add_mem(self, item):
		activity 	= item['activity']
		action		= item['action']
		delta 		= item['delta']
		timestamp	= item['timestamp']
		trial       = item['trial']
		#
		readable    = item['readable']

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

	def recall_mem(self, key, timestep, **kwargs):
		'''
		pass in key: get most similar entry and return cosine sim score

		confidence score = scaled by cosine sim

		'''
		if len(self.cache_list) == 0:
			random_policy = softmax(np.zeros(6))
			return random_policy
		else:
			#specify decay envelope for memory relevance calculation
			envelope = kwargs.get('decay', self.memory_envelope)

			# returns the most similar key, as well as the cosine similarity measure
			lin_act, similarity = self.cosine_sim(key)


			memory       = np.nan_to_num(self.cache_list[lin_act][0])

			deltas       = memory[:,0]
			times        = abs(timestep - memory[:,1])
			pvals 		 = self.make_pvals(times, envelope=envelope)
			policy = softmax( similarity*np.multiply(deltas, pvals), T=1) #np.multiply(sim,deltas))
			return policy

	def make_pvals(self, p, **kwargs):
		envelope = kwargs.get('envelope', self.memory_envelope)

		return np.round(1 / np.cosh(p / envelope), 8)

	# retrieve relevant items from memory
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

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def softmax(x, T=1):
	e_x = np.exp((x - np.max(x))/T)
	return np.round(e_x / e_x.sum(axis=0),8) # only difference

def plot_softmax(x):
	f, axarr = plt.subplots(2, sharex=True)
	axarr[0].bar(np.arange(len(x)), x)
	y = softmax(x)    
	axarr[1].bar(np.arange(len(x)), y) 
	plt.show()

def calc_env(halfmax):
	'''
	:param halfmax: x value for which envelope will give sech(x/env) = 0.5
	:return: envelope value
	e^(x/env) = (2+np.sqrt(3)) for sech(x/env) = 0.5
	Hence x/env = np.log(2+np.sqrt(3)) and env = x/ np.log(2+np.sqrt(3))
	'''
	return int(halfmax/np.log(2+np.sqrt(3)))

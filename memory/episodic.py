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

import time 
import numpy as np

import torch 
from torch.autograd import Variable
from torch import autograd, optim, nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import namedtuple

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

class ep_mem(object):
	def __init__(self, model, cache_limit,**kwargs):
		self.cache_list 		= {}								# memory bank object
		self.cache_limit 		= cache_limit                       # size of memory bank
		self.n_actions			= model.layers[-1]					# number of rows in each memory unit



		self.memory_envelope 	= kwargs.get('mem_envelope', 50)    # speed of memory decay


		#self.key_length	 	= model.layers[-2]
		#self.key_dtype			= [(f'{i}', 'f8') for i in range(self.key_length)]


		##
		num_inputs                 = model.layers[0]
		self.mem_factor            = 0.5
		self.reward_unseen         = True
		self.time_since_last_reward= 0
		self.confidence_score      = 0
		self.cs_max                = 0

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
			print("hello world")
			# Case 2a: key does not yet exist
			if activity not in self.cache_list.keys():
				# choose key to be removed
				cache_keys = list(self.cache_list.keys())
				persistence_ = [t for e, t in self.cache_list.values()] # get list of all timestamp flags
				print(persistence_)
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



	def reset_cache(self):
		self.cache_list.clear()

	def make_pvals(self, p, **kwargs):
		policy_ = kwargs.get('pol_id', None)
		mfc = kwargs.get('mfc', 1)
		envelope = kwargs.get('envelope', self.memory_envelope)
		if policy_ is not None:
			if policy_ == 'MF':
				return np.round(1 / np.cosh(p / self.memory_envelope), 8)
			elif policy_ == 'EC':
				return mfc*np.round(1 / np.cosh(p / self.memory_envelope), 8)
		else:
			return np.round(1 / np.cosh(p / self.memory_envelope), 8)

	# retrieve relevant items from memory
	def cosine_sim(self, key, **kwargs):
		similarity_threshold = kwargs.get('threshold', 0.9)

		mem_cache = np.asarray(list(self.cache_list.keys()))
		entry = np.asarray(key)

		mqt = np.dot(mem_cache, entry)
		norm = np.linalg.norm(mem_cache, axis=1) * np.linalg.norm(entry)

		cosine_similarity = mqt / norm

		index = np.argmax(cosine_similarity)
		similar_activity = mem_cache[index]
		if max(cosine_similarity) >= similarity_threshold:
			return similar_activity, index, max(cosine_similarity)

		else:
			# print('max memory similarity:', max(cosine_similarity))
			return [], [], max(cosine_similarity)

	def recall_mem(self, key, timestep, **kwargs):
		'''
		pass in key: get most similar entry and return cosine sim score

		confidence score = scaled by cosine sim

		'''
		envelope = kwargs.get('env', self.memory_envelope)
		#print(len(key), "====")
		mem_, i, sim = self.cosine_sim(key,threshold=0)
		#eprint(len(mem_), "####")
		memory       = np.nan_to_num(self.cache_list[tuple(mem_)][0])
		deltas       = memory[:,0]
		times        = abs(timestep - memory[:,1])
		pvals 		 = self.make_pvals(times, envelope=envelope)

		policy = softmax(np.multiply(sim,deltas))  #np.multiply(deltas, pvals), T= 0.1)
		return policy



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

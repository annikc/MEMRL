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

printflag = False

class ep_mem(object):
	def __init__(self, model, cache_limit,**kwargs):
		self.cache_limit 		= cache_limit
		self.memory_envelope 	= kwargs.get('persistence', 10)
		self.cache_list 		= {}
		self.n_actions			= model.layers[-1]



		##
		num_inputs                 = model.layers[0]


		self.mem_factor            = 0.5
		self.reward_unseen         = True
		self.time_since_last_reward= 0
		self.confidence_score      = 0
		self.cs_max                = 0

		self.stupid_df = [[],[],[],[]]

	def reset_cache(self):
		self.cache_list.clear()
		self.stupid_df = [[], [], [], []]
		
	def make_pvals(self, p):
		return 1 / np.cosh(p / self.memory_envelope)

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

	def add_mem(self, item):
		activity 	= item['activity']
		action		= item['action']
		delta 		= item['delta']
		timestamp	= item['timestamp']
		'''
		1. Memory is not full
			a. key does not yet exist
				- make mem container w this key 
				- add relevant info in container
			b. key already exists
				- add or replace relevant info in mem container
				
		2. Memory is full
			a. key does not exist in mem
				- choose key to replace -- should just choose oldest? 

			b. key already exists
				- add or replace relevant info in mem container
		'''
		# if memory is not full

		if len(list(self.cache_list.keys())) < self.cache_limit:
			tx = time.time()#0
			if activity not in self.cache_list.keys(): # if no key for this state exists already, add new one
				mem_entry = np.empty((self.n_actions, 2))
				mem_entry[:,0] = np.nan # initialize entries to nan
				mem_entry[:,1] = np.inf
				self.cache_list[activity] = [mem_entry, np.inf]
			self.stupid_df[0].append(time.time()-tx)
			tx = time.time() #1
			# add or replace relevant info in mem container
			self.cache_list[activity][0][action] = [delta, timestamp]
			self.cache_list[activity][1] = timestamp
			self.stupid_df[1].append(time.time()-tx)
		# if memory is full
		else:

			# if there is no item in memory that matches
			if activity not in self.cache_list.keys():
				tx = time.time()  # 2
				# choose key to be removed

				cache_items = np.asarray(list(self.cache_list.items())) # . values

				persistence_ = [t for e, t in cache_items[:,1]]

				lp = persistence_.index(min(persistence_))
				old_activity = cache_items[lp,0]
				del self.cache_list[old_activity]
				self.stupid_df[2].append(time.time()-tx)

				# add new mem container
				mem_entry = np.empty((self.n_actions, 2))
				mem_entry[:,0] = np.nan
				mem_entry[:,1] = np.inf # initialize entries to nan
				self.cache_list[activity] = [mem_entry, np.inf]

			tx = time.time()  # 3
			# add or replace relevant info in mem container
			mem_entry = self.cache_list[activity][0]
			persistence_value = timestamp * max(abs(mem_entry[:,0]))

			self.cache_list[activity][0][action] = [delta, timestamp]
			self.cache_list[activity][1] = persistence_value
			self.stupid_df[3].append(time.time()-tx)

	def recall_mem(self, key, timestep, **kwargs):
		'''
		pass in key: get most similar entry and return cosine sim score

		confidence score = scaled by cosine sim

		'''
		envelope = kwargs.get('env', 50)

		mem_, i, sim = cosine_sim(key,threshold=0.5)
		memory       = self.cache_list[tuple(mem_)]
		deltas       = memory[0]
		times        = timestep - np.nan_to_num(memory[1])

		policy = softmax(sim*(np.multiply(deltas, self.make_pvals(times,envelope=envelope))))

		return policy


def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def softmax(x, T=1):
	 """Compute softmax values for each sets of scores in x."""
	 e_x = np.exp((x - np.max(x))/T)
	 return e_x / e_x.sum(axis=0) # only difference

def plot_softmax(x):
	f, axarr = plt.subplots(2, sharex=True)
	axarr[0].bar(np.arange(len(x)), x)
	y = softmax(x)    
	axarr[1].bar(np.arange(len(x)), y) 
	plt.show()

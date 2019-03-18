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
		num_inputs                 = model.layers[0]
		self.num_actions           = model.layers[-1]
		self.cache_limit           = cache_limit
		self.memory_envelope       = kwargs.get('pers', 10)
		self.cache_list            = {}

		self.mem_factor            = 0.5
		self.reward_unseen         = True
		self.time_since_last_reward= 0
		self.confidence_score      = 0
		self.cs_max                = 0
		
	def reset_cache(self):
		self.cache_list.clear()
		
	def make_onehot(self,action,delta):
		onehot        = np.zeros(self.num_actions)
		onehot[action]= 1

		return onehot
	
	def make_pvals(self,cur_time,mem_dict,**kwargs):
		self.memory_envelope = kwargs.get('envelope', 10)
		
		cache_times	= np.array([v[1] for k,v in mem_dict.items()])
		cache_delts	= np.array([np.linalg.norm(v[0], axis=0) for k, v in mem_dict.items()])
			   
		p          = ((cur_time - cache_times)/cur_time) #*cache_delts
		pvals      = 1-1/np.cosh(p/self.memory_envelope)
		return pvals
		
	def reward_update(self, trialstart_stamp, timestamp, reward, **kwargs):
		if printflag:
			starttime = time.time()
		scale_val = 0.1*kwargs.get('scale', 5)
		
		# get subset of dict within trial
		mem_buffer = {k: v for k,v in self.cache_list.items() if v[1]>trialstart_stamp}
		
		# make pvals on subset		
		mem_pvals  = self.make_pvals(timestamp, mem_buffer,  envelope=10)
		
		# modify dict at relevant locations w scaled onehots
		for ind, key in enumerate(mem_buffer.keys()):
			rescaled_vec = scale_val*(reward + mem_pvals[ind])*self.cache_list[key][0]
			og_timestamp = self.cache_list[key][1]
			og_loc       = self.cache_list[key][2]

			self.cache_list[key] = (rescaled_vec, og_timestamp, og_loc)
		mem_buffer.clear()

		if printflag:
			print("reward_update takes {0:.5f}s ".format(time.time() - starttime))


	def mem_knn(self, entry, **kwargs):
		if printflag:
			starttime = time.time()

		if type(entry) != tuple: 
			entry = tuple(entry)

		## find K nearest neighbours to memory 
		# used for storage and recall 
		k                 = kwargs.get('k', 3)
		distance_threshold= kwargs.get('dist_thresh', 0.05)
		
		activities = self.cache_list.keys()
		if len(self.cache_list.keys())>k:
			nbrs               = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(activities)
			distances, indices = nbrs.kneighbors([entry])
			indices            = indices[0][distances[0] < distance_threshold]
			distances          = distances[0][distances[0] < distance_threshold]
			knn_acts           = [activities[i] for i in indices]
		else:
			knn_acts           = []
			distances          = []

		if printflag:
			print("mem_knn takes {0:.5f}s ".format(time.time() - starttime))

		return knn_acts, distances

	def cosine_sim(self, entry, **kwargs):
		if printflag:
			starttime = time.time()
		
		similarity_threshold = kwargs.get('threshold', 0.9)
		
		mem_cache 	= np.asarray(self.cache_list.keys())
		entry 		= np.asarray(entry)

		mqt 		= np.matmul(mem_cache, entry.T)
		norm 		= np.linalg.norm(mem_cache, axis = 1).reshape(-1,1)*np.linalg.norm(entry).reshape(-1,1) 

		cosine_similarity = mqt/norm

		if max(cosine_similarity) >= similarity_threshold:
			index = [np.argmax(cosine_similarity)]
			similar_activity = [mem_cache[index]]
			return similar_activity, index

		else:
			print('max memory similarity:', max(cosine_similarity))
			return [],[]
		
		if printflag:
			print("cosine_sim takes {0:.5f}s ".format(time.time() - starttime))

	def add_mem(self,add_mem_args, **kwargs):

		#### WSNT TO UPDATE THIS TO USE SIMILARITY MEAS
		if printflag:
			starttime = time.time()
		activity 	= add_mem_args['activity']
		action 	 	= add_mem_args['action']
		delta 	 	= add_mem_args['delta']
		timestamp 	= add_mem_args['timestamp']
		if 'state' in add_mem_args.keys():
			grid_state = add_mem_args['state']
		else:
			grid_state = None

		'''
		Add memory to cache list (dictionary)
		Inputs: 
			activity    - a tuple of values from the state vector 
							(pc activities)
			action      - an integer value representing the chosen action
			delta       - a float value (scalar) representing 'reward prediction 
							error'/eligibility traces
			timestamp   - when the memory was taken -- used to caculate memory persistence

		Items in the dictionary are called using activity tuple as the key. The corresponding entry
		is itself a tuple. entry[0] is an n-dimensional vector where n is the number of possible actions.
		This vector is zero-valued except at the position corresponding to the chosen action, which is set
		to the reward prediction error/eligibility trace value (can think of it as a one-hot vector scaled
		by RPE). 

		If dictionary is not full, will just add memory with pc activity tuple used as key. 
		If dictionary has reached the cache_limit, will calculate the k-nearest neighbours in the dictionary
		and replace the entry with the lowest persistence value (a function of the timestep at which the memory
		was recorded). 

		'''
		## activity is pc_activites
		##     --> to be a dict key must be a tuple (immutable type)
		## item is (delta*action_onehot, timestamp) where delta = r_t - v(s_t) - v(s_t-1)
		mixing      = kwargs.get('mixing', False)
		keep_hist   = kwargs.get('keep_hist', False)
		hist_scaling= 0.9
		item        = self.make_onehot(action,delta)
		
		if len(self.cache_list) < self.cache_limit:
			#if activity not in self.cache_list.keys():
			self.cache_list[activity] = item, timestamp, grid_state
		else:
			## determine entry to get rid of
			# KNN to see most similar entries
			# replace neighbour with lowest persistence
			# no item closer than threshold, replace item in all mem with lowest persistence

			act_keys, distances = self.mem_knn(activity)
			
			if len(act_keys) > 0:
				#use it
				# make new dictionary of just the k nn entries and their data
				cache_knn         = {k: self.cache_list[k] for k in act_keys}
				# of those entries pick the one with lowest persistence
				persistence_values= self.make_pvals(timestamp, cache_knn)
				lp_entry_key      = cache_knn.keys()[np.argmin(persistence_values)] 
				lp_entry_data     = self.cache_list[lp_entry_key]
				
				
				# make new entry out of convex combination of old entry and current information
				### MIXING
				if mixing:
					p_val          = min(persistence_values)
					new_entry_key  = p_val*np.asarray(lp_entry_key) + (1-p_val)*np.asarray(activity)
					activity       = tuple(new_entry_key)
					new_entry_data_= p_val*lp_entry_data[0] + (1-p_val)*item
					item           = new_entry_data_#Variable(torch.from_numpy(new_entry_data_)).type(torch.FloatTensor)
					
				if keep_hist:
					old_item                 = self.cache_list[lp_entry_key][0]
					del self.cache_list[lp_entry_key]
					self.cache_list[activity]= item+(hist_scaling*old_item), timestamp, grid_state
				else:
					del self.cache_list[lp_entry_key]
					self.cache_list[activity]= item, timestamp, grid_state
				del cache_knn
				
			else:
				persistence_values= self.make_pvals(timestamp, self.cache_list)
				lp_entry_key      = self.cache_list.keys()[np.argmin(persistence_values)]

				if keep_hist:
					old_item                 = self.cache_list[lp_entry_key][0]
					del self.cache_list[lp_entry_key]
					self.cache_list[activity]= item+(hist_scaling*old_item), timestamp, grid_state
				else:
					del self.cache_list[lp_entry_key]
					self.cache_list[activity]= item, timestamp, grid_state
		
		if printflag:
			print("add_mem takes {0:.5f}s ".format(time.time() - starttime))

	
	def add_mem_new(self,add_mem_args, **kwargs):
		'''
		Add memory to cache list (dictionary)
		Inputs: 
			activity    - a tuple of values from the state vector 
							(pc activities)
			action      - an integer value representing the chosen action
			delta       - a float value (scalar) representing 'reward prediction 
							error'/eligibility traces
			timestamp   - when the memory was taken -- used to caculate memory persistence

		Items in the dictionary are called using activity tuple as the key. The corresponding entry
		is itself a tuple. entry[0] is an n-dimensional vector where n is the number of possible actions.
		This vector is zero-valued except at the position corresponding to the chosen action, which is set
		to the reward prediction error/eligibility trace value (can think of it as a one-hot vector scaled
		by RPE). 

		If dictionary is not full, will just add memory with pc activity tuple used as key. 
		If dictionary has reached the cache_limit, will calculate the k-nearest neighbours in the dictionary
		and replace the entry with the lowest persistence value (a function of the timestep at which the memory
		was recorded). 

		'''
		## activity is pc_activites
		##     --> to be a dict key must be a tuple (immutable type)
		## item is (delta*action_onehot, timestamp) where delta = r_t - v(s_t) - v(s_t-1)
		
		if printflag:
			starttime = time.time()
		activity 	= add_mem_args['activity']
		action 	 	= add_mem_args['action']
		delta 	 	= add_mem_args['delta']
		timestamp 	= add_mem_args['timestamp']
		if 'state' in add_mem_args.keys():
			grid_state = add_mem_args['state']
		else:
			grid_state = None

		metric 		= kwargs.get('metric', 'use_knn')
		mixing      = kwargs.get('mixing', False)
		keep_hist   = kwargs.get('keep_hist', False) ## if we use keep hist, need to fix 
		hist_scaling= 0.9
		item        = self.make_onehot(action,delta)
		if len(self.cache_list) < self.cache_limit:
			#if activity not in self.cache_list.keys():
			self.cache_list[activity] = item, timestamp, grid_state
		else:
			## determine entry to get rid of
			# KNN to see most similar entries
			# replace neighbour with lowest persistence
			# no item closer than threshold, replace item in all mem with lowest persistence
			if metric == 'use_knn':
				act_keys, distances = self.mem_knn(activity)
			elif metric == 'use_cos':
				act_keys, similarity = self.cosine_sim(activity)
			
				if len(act_keys) > 0:
					#use it
					# make new dictionary of just the k nn entries and their data
					cache_knn         = {k: self.cache_list[k] for k in act_keys}
					# of those entries pick the one with lowest persistence
					persistence_values= self.make_pvals(timestamp, cache_knn)
					lp_entry_key      = cache_knn.keys()[np.argmin(persistence_values)] 
					lp_entry_data     = self.cache_list[lp_entry_key]

					del cache_knn
					
				else:
					persistence_values= self.make_pvals(timestamp, self.cache_list)
					lp_entry_key      = self.cache_list.keys()[np.argmin(persistence_values)]

				if keep_hist:
					old_item          = self.cache_list[lp_entry_key][0]
					del self.cache_list[lp_entry_key]
					self.cache_list[activity]= item+(hist_scaling*old_item), timestamp, grid_state
				else:
					del self.cache_list[lp_entry_key]
					self.cache_list[activity]= item, timestamp, grid_state

			if printflag:
				print("add_mem takes {0:.5f}s ".format(time.time() - starttime))


	def compute_confidence(self, rwd):
		if printflag:
			starttime = time.time()

		if self.reward_unseen: 
			cs = 0.0
		else:
			cs = (self.cs_max + self.mem_factor)*np.exp(-self.time_since_last_reward/10)
		
		if cs > 1.0:
			cs = 1.0
			
		if rwd == 1:
			self.reward_unseen         = False
			self.cs_max                = cs
			self.time_since_last_reward= 0
		
		self.time_since_last_reward += 1
		self.confidence_score        = cs

		if printflag:
			print("compute_confidence takes {}s ".format(time.time() - starttime))


	def recall_mem(self, activity, **kwargs):
		if printflag: 
			starttime = time.time()
		metric = kwargs.get("metric", 'use_knn')
		if type(activity) != tuple:
			activity = tuple(activity)

		# KNN to see most similar entries
		if metric == 'use_knn':
			act_keys, distances = self.mem_knn(activity)
		elif metric == 'use_cos':
			act_keys, distances = self.cosine_sim(activity)

		if len(act_keys) > 0:
			# make new dictionary of just the k nn entries and their data
			cache_knn = {k: self.cache_list[k] for k in act_keys}
			cache_actions = [v[0] for k,v in cache_knn.items()]
			rec_vec = sum(cache_actions)
			policy_EC = softmax(rec_vec)

		else:
			policy_EC = softmax(np.zeros(self.num_actions))

		if printflag:
			print("recall_mem takes {}s ".format(time.time() - starttime))


		return Variable(torch.from_numpy(policy_EC).type(torch.FloatTensor).unsqueeze(0))

	def composite_policy(self, policy_MF, policy_EC, rwd):
		if printflag:
			starttime = time.time()
		self.compute_confidence(rwd)
		#policy MF and EC should be torch variables
		policy = (1-self.confidence_score)*policy_EC.data + self.confidence_score*policy_MF.data
		
		if printflag:
			print("composite_policy takes {}s ".format(time.time() - starttime))

		return Variable(policy)

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

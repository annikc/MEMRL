#### Episodic Memory Cache

'''
Object Classes and Relevant Functions for Episodic Memory Module
Author: Annik Carson 
--  March 2018
'''

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


class ep_mem(object):
    def __init__(self, model, cache_limit,**kwargs):
        num_inputs = model.layers[0]
        self.num_actions = model.layers[-1]
        self.cache_limit = cache_limit
        self.memory_envelope = kwargs.get('pers', 10)
        self.cache_list = {}

        #self.reward_unseen = True
        #self.cs_max = 0
    
    def reset_cache(self):
        self.cache_list = {}
        
    def make_onehot(self,action,delta):
        onehot = np.zeros(self.num_actions)
        onehot[action] = delta
        return onehot
    
    def make_pvals(self,cur_time,mem_dict):
        cache_times = np.array([v[1] for k,v in mem_dict.items()])
        cache_delts = np.array([np.linalg.norm(v[0], axis=0) for k, v in mem_dict.items()])
               
        p = ((cur_time - cache_times)/cur_time)*cache_delts
        pvals = 1-1/np.cosh(p/self.memory_envelope)
        return pvals
        

    def mem_knn(self, entry, **kwargs):
        ## find K nearest neighbours to memory 
        # used for storage and recall 
        k = kwargs.get('k', 3)
        distance_threshold = kwargs.get('dist_thresh', 0.1)
        
        activities = self.cache_list.keys()
        if len(self.cache_list.keys())>k:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(activities)
            distances, indices = nbrs.kneighbors([entry])
            indices = indices[0][distances[0] < distance_threshold]
            distances = distances[0][distances[0] < distance_threshold]
            knn_acts = [activities[i] for i in indices]
        else:
            knn_acts = []
            distances = []
        return knn_acts, distances
    
    def add_mem(self,activity,action,delta,timestamp,**kwargs):
        ## activity is pc_activites
        ##     --> to be a dict key must be a tuple (immutable type)
        ## item is (delta*action_onehot, timestamp) where delta = r_t - v(s_t) - v(s_t-1)
        
        item = self.make_onehot(action,delta)
        
        if len(self.cache_list) < self.cache_limit: 
            self.cache_list[activity] = item, timestamp
            
        else:
            ## determine entry to get rid of
            # KNN to see most similar entries
            # replace neighbour with lowest persistence
            # no item closer than threshold, replace item in all mem with lowest persistence
            act_keys, distances = self.mem_knn(activity)
            
            if len(act_keys) > 0:
                #use it
                # make new dictionary of just the k nn entries and their data
                cache_knn = {k: self.cache_list[k] for k in act_keys}
                # of those entries pick the one with lowest persistence
                persistence_values = self.make_pvals(timestamp, cache_knn)
                lp_entry_key = cache_knn.keys()[np.argmin(persistence_values)] 
                lp_entry_data = self.cache_list[lp_entry_key]
                p_val = min(persistence_values)
                
                # make new entry out of convex combination of old entry and current information
                new_entry_key = tuple(p_val*np.asarray(lp_entry_key) + (1-p_val)*np.asarray(activity))
                new_entry_data = p_val*lp_entry_data[0] + (1-p_val)*item
                
                del self.cache_list[lp_entry_key]
                self.cache_list[new_entry_key] = new_entry_data, timestamp
                del cache_knn

            else:
                persistence_values = self.make_pvals(timestamp, self.cache_list)
                lp_entry_key = self.cache_list.keys()[np.argmin(persistence_values)]
                del self.cache_list[lp_entry_key]
                self.cache_list[activity] = item, timestamp
                
    def recall_mem(self, activity,policy,confidence,**kwargs):
        ## activity is pc_activites - to be a dict key must be a tuple (immutable type)
        ## item is (delta*action_onehot, persistence) where delta = r_t - v(s_t) - v(s_t-1)

        # KNN to see most similar entries
        act_keys, distances = self.mem_knn(activity)

        if len(act_keys) > 0:
            # make new dictionary of just the k nn entries and their data
            cache_knn = {k: self.cache_list[k] for k in act_keys}
            cache_actions = [v[0] for k,v in cache_knn.items()]
            composite_mem_policy = softmax(sum(cache_actions))
        else:
            composite_mem_policy = policy

        composite_policy = (1-confidence)*composite_mem_policy + confidence*policy
        
        return composite_policy

    def mem_confidence(self, rwd, cs_max, time_since_last_reward, no_r):
        self.factor = 0.5
        
        if no_r: 
            cs = 0
        else:
            cs = (cs_max + self.factor)*np.exp(-time_since_last_reward/10)
        
        if cs > 1.0:
            cs = 1
            
        if rwd == 1:
            no_r = False

            cs_max = cs
            time_since_last_reward = 0
        
        time_since_last_reward += 1
        return cs, cs_max, time_since_last_reward

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def softmax(x):
     """Compute softmax values for each sets of scores in x."""
     e_x = np.exp(x - np.max(x))
     return e_x / e_x.sum(axis=0) # only difference

def plot_softmax(x):
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].bar(np.arange(len(x)), x)
    y = softmax(x)    
    axarr[1].bar(np.arange(len(x)), y) 
    plt.show()

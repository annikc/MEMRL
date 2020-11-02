# =====================================
#  Stores Transition Data to np Arrays
# =====================================
#       Function Descriptions
# =====================================
# store_transition = stores transition data 
# clear_transitions = used to setup numpy arrays to store transition data
# get_transitions = returns numpy arrays of all transitions currently saved
# sample_transitions = returns a sample of transitions currently saved in transition Cache 
# save_transitions = TO DO 

from collections import namedtuple
import numpy as np

class Transition_Cache():
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.transition_cache = []
        self.cache_cntr = 0

    def store_transition(self, transition):
        
        if len(self.transition_cache) < self.cache_size:
            self.transition_cache.append(transition)
        else: 
            self.transition_cache[self.cache_cntr] = transition
            self.cache_cntr += 1 if self.cache_cntr < self.cache_size else 0
    
    # clear buffer - can be used for MC methods at end of episode
    def clear_cache(self):
        self.transition_cache = []
        self.cache_cntr = 0

    # samples transitions - can be used for methods that require buffer
    def sample_transition_cache(self, batch_size):
        # find out how many memories are in our buffer 
        amount_stored = len(self.transition_cache)
        
        # get a list of random index numbers and sample the cache
        batch = np.random.choice(amount_stored, batch_size, replace =False)
        transition_batch = self.transition_cache[batch]

        return transition_batch








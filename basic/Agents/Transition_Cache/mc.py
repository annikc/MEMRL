# =====================================
#       Monte Carlo Update System
# =====================================
#       Function Descriptions
# =====================================
# store_transition = stores transition log_prob, critic value, reward, and done flags for each transition during episode 
# clear_buffer = clears stored data from the buffer created by store_transitions
# get_buffer = returns a zip((log_prob, value), disc_rewards) - Note (log_prob, value) is a namedtuple
# discount_rwds = takes the list of rewards collected in episode and returns their discount value () (uses rewards-to-go - i.e. backward iteration)

from collections import namedtuple
import numpy as np
import torch as T

class MCBuffer():
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.action_memory = [] 
        self.reward_memory = []
        self.terminal_memory = []
        self.SavedAction = namedtuple('SavedAction', ['log_prob', 'critic_value'])

    def store_transition(self, log_prob, critic_value, reward, done):
        #  set the numpy arrays at the index = parameters we passed in
        self.action_memory.append(self.SavedAction(log_prob, critic_value))
        self.reward_memory.append(reward)
        self.terminal_memory.append(done)
    
    # function that returns all the buffer information from the episode
    def get_buffer(self):
        target_values = T.Tensor(self.discount_rwds(np.asarray(self.reward_memory)))
        return zip(self.action_memory, target_values)

    def discount_rwds(self, r):
        disc_rwds = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add*self.gamma + r[t]
            disc_rwds[t] = running_add
        return disc_rwds

    # clear buffer - can add save-to-memory functionality here for EC
    def clear_buffer(self):
        self.action_memory = [] 
        self.reward_memory = []
        self.terminal_memory = []





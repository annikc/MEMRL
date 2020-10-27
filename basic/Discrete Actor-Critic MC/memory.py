from __future__ import divition, print_function
import numpy as np
from futils import softmax

class EpisodicMemory(object):
    def __init__(self, max_size, input_dims, n_actions):
        self.mem_size = max_size
        self.
        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.new_state_memory = np.zeros(self.mem_size, *input_dims)
        self.action_memory = np.zeros(self.mem_size, n_actions)
        self.reward_memory = np.zeros(self.mem_size)

        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, episode, state, action, reward, state_, done):
        index = mem_7


        self.state_memory[index] = 

    def reset_cache(self):
        self.cache_list.clear()

    # Not sure what this is used for atm
    def calc_envelope(self, halfmax):
        return halfmax/np.log(2+np.sqrt(3))

    def recall(self, representation):
        

        
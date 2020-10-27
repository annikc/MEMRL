

import numpy as np

class MCMemory():
    def __init__(self, max_size, input_shape, n_actions):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.mem_size = max_size
        self.mem_cntr = 0  # holds position of first available memory
        self.state_memory = []
        self.new_state_memory = []  
        self.action_memory = [] 
        self.log_prob_memory = []
        self.reward_memory = [] 
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_memory(self, state, action, log_prob, reward, state_, done):
        # finds index of first available memory = when end is reached oldest memory is replaced
        index = self.mem_cntr % self.mem_size 
        #  set the numpy arrays at the index = parameters we passed in
        self.log_prob_memory[index] = log_prob
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def clear_memory(self):
        self.mem_cntr = 0  
        self.state_memory = np.zeros((self.mem_size, *self.input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *self.input_shape))  
        self.action_memory = np.zeros((self.mem_size, self.n_actions))  
        self.reward_memory = np.zeros(self.mem_size)  
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    # function that handles our buffer 

    def get_memories(self):
        states = self.state_memory
        actions = self.action_memory
        log_probs = self.log_prob_memory
        rewards = self.reward_memory
        next_states = self.new_state_memory
        dones = self.terminal_memory

        return states, actions, log_probs, rewards, next_states, dones





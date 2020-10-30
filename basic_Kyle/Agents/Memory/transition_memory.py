# =====================================
#  Stores Transition Data to np Arrays
# =====================================
#       Function Descriptions
# =====================================
# store_transition = stores transition data 
# clear_transitions = used to setup numpy arrays to store transition data
# get_transitions = returns numpy arrays of all transitions currently saved
# sample_transitions = returns a sample of transitions currently saved in transition memory arrays
# save_transitions = TO DO 

import numpy as np

class Transition_Memory():
    def __init__(self, mem_size, input_dims):
        self.mem_size = mem_size
        self.input_dims = input_dims
        self.clear_memory()
 
    def store_transition(self, state, action, reward, new_state, log_prob, model_value, done):
        index = self.mem_cntr
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.log_prob_memory[index] = log_prob
        self.model_value_memory[index] = model_value
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
    # function that returns all the transitions currently saved in memory
    def get_transitions(self):
        states = self.state_memory[:self.mem_cntr]
        actions = self.action_memory[:self.mem_cntr]
        rewards = self.reward_memory[:self.mem_cntr]
        new_states = self.new_state_memory[:self.mem_cntr]
        log_probs = self.log_prob_memory[:self.mem_cntr]
        model_values = self.model_value_memory[:self.mem_cntr]
        dones = self.terminal_memory[:self.mem_cntr]

        return states, actions, rewards, new_states, log_probs, model_values, dones

    # clear buffer - can be used for MC methods at end of episode
    def clear_memory(self):
        self.state_memory = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32) # pytorch is particular about data types and enforces type checking 
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)  # we use an int because we have a discrete action space. 
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *self.input_dims), dtype = np.float32)
        self.log_prob_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.model_value_memory = np.zeros(self.mem_size, dtype=np.float32) 
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.mem_cntr = 0

    # samples transitions - can be used for methods that require buffer
    def sample_transitions(self, batch_size):
        # find out how many memories are in our buffer 
        max_mem = min(self.mem_cntr, self.mem_size)
        
        # get a list of random index numbers to sample memory arrays
        batch = np.random.choice(max_mem, batch_size, replace =False)

        # sample memories
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        log_probs = self.log_prob_memory[batch]
        model_values = self.model_value_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, log_probs, model_values, dones

    #def save_transitions(self): # TO DO - this can be used for implemeting episodic memory








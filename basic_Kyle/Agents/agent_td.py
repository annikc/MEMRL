# =====================================
#     Creates MC Actor-Critc Agent
# =====================================
#       Function Descriptions
# =====================================
# get_action_log_value = takes observation (state) and returns the action from actor network (i.e. policy), log_value from actor, and value from critic networ 
# mc_learn = Uses stored transition from the episode to calculate losses and updates the actor and critic networks
import torch as T
import torch.nn.functional as F
import numpy as np
from Agents.Transition_Cache.transition_cache import Transition_Cache
from torch.autograd import Variable

class Agent_TD():
    def __init__(self, policy_network, value_network, trans_cache_size=100000, 
                    gamma=0.99, TD=True, epsilon=0.99, eps_min=0.01, eps_dec=5e-4, batch_size=32, replace=1000): #alpha/beta = agent/critic lrs
        self.log_probs = None
        self.gamma = gamma
        self.q_eval = policy_network 
        self.q_next = value_network  
        self.transition_cache = Transition_Cache(trans_cache_size) 
        self.TD = TD
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.current_transition = None
        self.action_space = [i for i in range(policy_network.output_dims)]
        self.log_prob = None
        self.replace_target_cnt = replace
        self.learn_step_counter = 0

    def get_action(self, observation): 
        state = T.tensor([observation], dtype=float).to(self.q_eval.device)
        actions = self.q_eval.forward(state)
        if np.random.random() > self.epsilon:
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        # get the current q value (expected value of taking the action)
        expected_value = actions[action]
        
        return action, self.log_prob, expected_value

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def learn(self):
        # Learning updates the value network. 
        # 1. let the agent play a bunch of games just randomly selecting actions until buffer full
        # 2. start learning once you've filled up the batch_size not the whole memory (below)
        if len(self.transition_cache.transition_cache) < self.batch_size:
            return 
        
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        rewards, expected_values, next_states, terminals = self.sample_transition_cache()

        # Convert numpy arrays from memory to tensors
        rewards = T.tensor(rewards).to(self.q_eval.device)
        expected_values = T.tensor(expected_values).to(self.q_eval.device)
        next_states = T.tensor(next_states).to(self.q_eval.device)
        terminals = T.tensor(terminals).to(self.q_eval.device)

        next_state_values = self.q_next.forward(next_states).max(dim=1)[0] 
        next_state_values[terminals] = 0.0

        # the target value is based on taking the greedy action in the next state (agent may not do this!)
        target_values = rewards + self.gamma*next_state_values

        #print(target_values, expected_values)

        # Calculate the loss
        loss = self.q_eval.loss(expected_values, target_values)
        #print(loss)

        loss.backward()
        self.q_eval.optimizer.step()
    
        # Each time we learn we need to decrese the epsilon value by 1 unit of decrement 
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        self.learn_step_counter += 1
 
    # agents transition Cache
    def store_transition(self):
        self.transition_cache.store_transition(self.current_transition)

    def clear_transition_cache(self):
        self.transition_cache.clear_cache()
    
    def sample_transition_cache(self):
        return self.transition_cache.sample_transition_cache(self.batch_size)
    

                
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
from Agents.Transition_Cache import Transition_Cache
from torch.autograd import Variable

class Agent_MC_2N():
    def __init__(self, policy_network, value_network, trans_cache_size=100000, 
                    gamma=0.99, TD=False):
        self.log_probs = None
        self.gamma = gamma
        self.policy_network = policy_network # actor network output is action dimensions
        self.value_network = value_network  # critic netowrk outputs is value estimate 
        self.transition_cache = Transition_Cache(trans_cache_size) 
        self.TD = TD

    def get_action(self, observation): 
        state = self.policy_network(observation)
        # the following could be turned into a function within agent_utilities
        probabilities = F.softmax(state)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        expected_value = self.value_network(observation)
        
        # Consistent with learning agent
        #print(action.item(), log_prob, expected_value.view(-1))

        return action.item(), log_prob, expected_value.view(-1)

    def learn(self):
        policy_losses = 0
        value_losses = 0

        # Calculates the discounted rewards - saves to expected values
        self.transition_cache.transition_cache = self.discount_rwds(self.transition_cache.transition_cache, self.gamma)

        # Gets policy and value losses
        policy_losses, value_losses = self.episode_losses(self.transition_cache.transition_cache)

        self.policy_network.optimizer.zero_grad()
        self.value_network.optimizer.zero_grad()

        (policy_losses + value_losses).backward()
        self.policy_network.optimizer.step()
        self.value_network.optimizer.step()
 
    # agents transition Cache
    def store_transition(self, transition):
        self.transition_cache.store_transition(transition)

    def clear_transition_cache(self):
        self.transition_cache.clear_cache()
    
    # discount_rwds + episode losses could be moved to another module
    def discount_rwds(self, transitions, gamma = 0.99):
        running_add = 0
        for t in reversed(range(len(transitions))):
            running_add = running_add*gamma + transitions[t].reward
            transitions[t] = transitions[t]._replace(target_value = running_add)
            #print (transitions[t].target_value)

        return transitions

    def episode_losses(self, transitions):
        policy_losses = 0
        value_losses = 0
        for transition in transitions: 
        # delta = reward + gamma*critic_value*(1-done)
            target_value = transition.target_value
            expected_value = transition.expected_value
            delta = target_value - expected_value.item()
            #print(f"T: {target_value}, E: {expected_value}, D: {delta}")
            log_prob = transition.log_prob  # This is the delta variable (i.e target_value*self.gamma + observed_value)
            policy_loss = (-log_prob * delta)
            return_bs = Variable(T.Tensor([[target_value]])).unsqueeze(-1) # used to make the shape work out
            #print(observed_value.shape, return_bs.shape) # shape error - you can pass batch 
            value_loss = (F.smooth_l1_loss(expected_value, return_bs))
            policy_losses += policy_loss
            value_losses += value_loss
            #print(f"P LOSS: {policy_losses}, V LOSS: {value_losses}")

        return policy_losses, value_losses
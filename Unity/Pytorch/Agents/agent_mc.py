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
from .Caches.transition_cache import Transition_Cache
from torch.autograd import Variable
from collections import namedtuple

Transition = namedtuple('Transition', 'episode, transition, state, action, reward, \
                                next_state, log_prob, expected_value, target_value, done')

class Agent_MC():
    def __init__(self, network, memory=None, **kwargs): #alpha/beta = agent/critic lrs
        # Optional arguments (kwargs)
        self.MFC_valnet = kwargs.get('value_network', None)  # allows a seperate network to be used to calculate value
        self.cache_size = kwargs.get('cache_size', 10000)
        self.gamma = kwargs.get('gamma', 0.98)
        self.TD = kwargs.get('td_learn', False)
        # set networks
        self.MFC = network # actor network output is action dimensions

        # create new transition cache
        self.transition_cache = Transition_Cache(self.cache_size)

        self.policy = None
        self.expected_value = None 
        self.action = None
        self.log_prob = None
        self.target_value = 0

    def MF_action(self, observation): 
        # get policy and value from 1 network or seperate networks
        self.state = observation
        self.policy, value = self.MFC(self.state)
        if self.MFC_valnet is not None:
           _, value = self.MFC_valnet(self.state)

        self.expected_value = value.view(-1)
        
        # calculate action based on policy returned
        a = T.distributions.Categorical(self.policy)
        selected_action = a.sample()
        self.action = selected_action.item()
        self.log_prob = a.log_prob(selected_action)

        return self.action

    def EC_action(self, observation):
        return
        # MF_Policy, value = self.MFC(observation)
        # if self.MFC_valnet is not None:
        #     _, value = self.MFC_valnet(observation)


    def learn(self, clear_cache = True):
        policy_losses = 0
        value_losses = 0

        # Calculates the discounted rewards - saves to expected values
        self.transition_cache.transition_cache = self.discount_rwds(self.transition_cache.transition_cache)

        # Gets policy and value losses
        policy_losses, value_losses = self.episode_losses(self.transition_cache.transition_cache)

        self.MFC.optimizer.zero_grad()
        if self.MFC_valnet is not None:
            self.MFC_valnet.optimizer.zero_grad()

        (policy_losses + value_losses).backward()
        self.MFC.optimizer.step()

        if self.MFC_valnet is not None:
            self.MFC_valnet.optimizer.step()
        
        if clear_cache: self.transition_cache.clear_cache() 
 
    # agents transition Cache
    def store_transition(self, episode, transition, reward, next_state, done):
        # creates named tuple and adds it to the cache
        transition = Transition(episode=episode, transition=transition, state=self.state, action=self.action, 
                                                    reward=reward, next_state=next_state, log_prob=self.log_prob, 
                                                    expected_value=self.expected_value, target_value=self.target_value, done=done)
        self.transition_cache.store_transition(transition)

    def clear_transition_cache(self):
        self.transition_cache.clear_cache()
    
    # discount_rwds + episode losses could be moved to another module
    def discount_rwds(self, transitions):
        running_add = 0
        for t in reversed(range(len(transitions))):
            running_add = running_add*self.gamma + transitions[t].reward
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
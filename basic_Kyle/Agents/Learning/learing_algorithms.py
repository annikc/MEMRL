import numpy as np
import torch as T
import torch.nn.functional as F


    
class Monte_Carlo_Learning():
    def __init__(self, gamma):
        self.gamma = gamma
    
    def learn(self, policy_network, value_network, transition_memory):
        policy_losses = 0
        value_losses = 0
        # Get transition information
        states, actions, rewards, new_states, log_probs, model_values, dones = transition_memory.get_transitions()
        #print(log_probs[:10])
        #print(rewards)
        #print(model_values[:10])

        # Calculate discounted rewards
        target_values = self.discount_rwds(rewards)

        # target_values are calucated using rewards-to-go backward iteration method (discounted_rwds function in mc_buffer)
        for index in range(len(log_probs)):
            # delta = reward + gamma*critic_value*(1-done)
            delta = target_value - model_values[index]  # This is the delta variable (i.e target_value*self.gamma + observed_value)
            policy_loss = (-log_prob * delta)
            value_loss = (delta**2)
            policy_losses += policy_loss
            value_losses += value_loss
            print(policy_losses, value_losses)
        
        policy_network.optimizer.zero_grad()
        value_network.optimizer.zero_grad()


        (policy_losses + value_losses).backward()
        policy_network.optimizer.step()
        value_network.optimizer.step()

        return policy_network, value_network


    def discount_rwds(self, r):
        disc_rwds = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(len(r))):
            running_add = running_add*self.gamma + r[t]
            disc_rwds[t] = running_add
        return disc_rwds

class TD_Buffer_Learning():

    
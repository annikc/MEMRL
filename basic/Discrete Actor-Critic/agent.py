from network import Network
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Agent():
    def __init__(self, alpha, beta, input_dims, gamma=0.99, l1_size = 256, l2_size = 256, n_actions = 3): #alpha/beta = agent/critic lrs
        self.gamma = gamma
        # AC works by updating the actor network with the graident along the policy
        # the policy is just a probability distribution and we back propogate the log of that 
        # distribution through the actor network for loss minimization the actor network
        self.log_probs = None
        # create the actor critic network using our generic network 
        # the actor network 
        self.actor = Network(alpha, input_dims, l1_size, l2_size, n_actions)
        self.critic = Network(beta, input_dims, l1_size, l2_size, n_actions=1)

    
    def choose_action(self, observation):
        # Remember we need to activate fc3 + probabilities need to sum to 1
        # to do this we use softmax activation funtion 
        probabilities = F.softmax(self.actor.forward(observation))
        # we then create a distribution that is modelled on the probabilities 
        action_probs = T.distributions.Categorical(probabilities)
        # then we get the action by sampling the action probability space 
        action = action_probs.sample()
        # Now we need the log probability of our sample to perform back-prop
        self.log_probs = action_probs.log_prob(action)

        # then we want to return our action as a integer using .item() since thats what 
        # OpenAI uses (action is currently a cuda tesnor)
        return action.item()



    # AC method is a TD method which requires us to calculate the delta (or error)
    # between what the model predicted and what the actual outcome was
    # we need the done flag so that when we transition into the terminal state
    # there are no feature rewards so the value of the next state is identically 0
    def learn(self, state, reward, new_state, done):
        # for all pytorch programs you want to zero out the gradients for the optimizer
        # at the beginning of your learning function. we do this for both actor and critic
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        # next we get the value of the state and the value of the next state
        # from our critic network 
        critic_value = self.critic.forward(state)
        critic_value_ = self.critic.forward(new_state)

        # next we calculate the TD error (i.e. delta)
        # we augment calculation with 1-int(done) so that we don't update 
        # when episode is done
        delta = ((reward + self.gamma*critic_value*(1-int(done))) - critic_value_)
        # we use delta to calculate both the actor and critic losses
        # the modifies the action probabilities in the direction that maximizes
        # future reward
        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        # we back propogate the sum of the losses through the network 
        (actor_loss + critic_loss).backward()

        # then we optimize
        self.actor.optimizer.step()
        self.critic.optimizer.step()









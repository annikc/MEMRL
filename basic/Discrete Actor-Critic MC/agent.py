from network import Network
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from mc_update import MCMemory

class Agent():
    def __init__(self, alpha, beta, input_dims, gamma=0.99, l1_size = 256, l2_size = 256, n_actions = 3): #alpha/beta = agent/critic lrs
        self.gamma = gamma
        # AC works by updating the actor network with the graident along the policy
        # the policy is just a probability distribution and we back propogate the log of that 
        # distribution through the actor network for loss minimization the actor network
        self.log_probs = None
        #create the actor critic network using our generic network 
        self.actor = Network(alpha, input_dims, l1_size, l2_size, n_actions)
        self.critic = Network(beta, input_dims, l1_size, l2_size, n_actions)
        self.memory = MCMemory(input_dims, n_actions)

    # def state_represention(self, state):
    #      # pass state info through our environemnt represention Network to get the 
    #      # state representation 
    #      state_representation = F.softmax(self.env_representation.forward(state))
    #      # pass representation to memory to get similar state representations 
    #      action_probs = T.distributions.Categorical(probabilities)
    #      # compare the action 

    
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
        return action.item(), self.log_prob

    def encode_memory(self, state, action, log_prob, reward, new_state, done):
        self.memory.store_memory(state, action, log_prob, reward, new_state, done)

    # AC method is a TD method which requires us to calculate the delta (or error)
    # between what the model predicted and what the actual outcome was
    # we need the done flag so that when we transition into the terminal state
    # there are no feature rewards so the value of the next state is identically 0
    def learn(self):
        
        # Get the memories for the episode
        states, actions, log_probs, rewards, new_states, dones = \
            self.memory.get_memories()

        # transform memories into tensors that can be passed through networks
        state = T.tensor(states, dtype=T.float).to(self.actor.device)
        action = T.tensor(actions, dtype=T.float).to(self.actor.device)
        log_probs = T.tensor(log_probs, dtype=T.float).to(self.actor.device)
        reward = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        new_state = T.tensor(new_states, dtype=T.float).to(self.actor.device)
        done = T.tensor(dones).to(self.actor.device)

        # Calculate Losses 
        actions = self.actor.forward(state)
        critic = self.critic.forward(state)
        critic_value = critic.view(-1)

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

    def reward_to_go(self, rewards):
        n = len(rewards)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs









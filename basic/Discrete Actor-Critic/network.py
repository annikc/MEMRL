import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


### here we create two different classes one for the actor and one for the critic network
# the actor network tells the agent what actions to take given the policy approximation
# the other network is the critic which says whether the action is good or bad based on
# its approximation of the value of the state-action pair (learns values of actions)


class Network(nn.Module):
    def __init__(self, lr, input_dim, fc1_dims, fc2_dims, n_actions):
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # network connections 
        self.fc1 = nn.Linear(self.input_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        # optimizer - parameters come fomr nn.Module 
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # setup device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cup:0')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device) # transforms np float array to cuda tensor
        # feed observation through layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # this is activated later when we select an action 

        return x


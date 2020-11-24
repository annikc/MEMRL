import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Network(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Network, self).__init__()
        self.input_dims = input_dims
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        # print("input", input_dims)
        # print("lr", lr)
        # print("fc1", fc1_dims)
        # print("fc2", fc2_dims)
        # print("n_actions", n_actions)

        # network connections 
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.policy = nn.Linear(self.fc2_dims, self.n_actions)
        self.value = nn.Linear(self.fc2_dims, 1)

        # optimizer - parameters come fomr nn.Module 
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # loss function for td learning
        self.loss = nn.MSELoss()

        # setup device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device) 
        phi = F.relu(self.fc1(state))
        psi = F.relu(self.fc2(phi))
        policy = (self.policy(psi))
        policy = F.softmax(policy)
        
        value = self.value(psi) 

        return policy, value
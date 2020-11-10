import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FullyConnected_AC(nn.Module):
    def __init__(self, lr, input_dim, fc1_dims, fc2_dims, n_actions):
        super(FullyConnected_AC, self).__init__()
        self.input_dim = input_dim
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # network connections 
        self.fc1 = nn.Linear(*self.input_dim, self.fc1_dims)
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
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy(x),dim=0)
        value = self.value(x) # this is activated differently by the actor and critic networks

        return policy, value
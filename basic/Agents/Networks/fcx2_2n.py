import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FullyConnected_2N(nn.Module):
    def __init__(self, lr, input_dim, fc1_dims, fc2_dims, output_dims):
        super(FullyConnected_2N, self).__init__()
        self.input_dim = input_dim
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.output_dims = output_dims

        # network connections 
        self.fc1 = nn.Linear(*self.input_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.output_dims)

        # optimizer - parameters come fomr nn.Module 
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # setup device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device) 
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # this is activated differently by the actor and critic networks

        return x
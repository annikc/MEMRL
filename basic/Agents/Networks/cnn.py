# =====================================
#            Creates CNN
# =====================================
#       Function Descriptions
# =====================================
# Forward = forwards data through the network - output is generic and can be activated differently for actor and critic 

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CNN_AC(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, n_actions):
        super(CNN_AC, self).__init__()
        self.lr = lr
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 4, stride=1)

        fc_input_dims = self.conv_output_dims(input_dims)
        
        self.fc1_dims = fc1_dims
        self.n_actions = n_actions

        # network connections 
        self.fc1 = nn.Linear(fc_input_dims, self.fc1_dims)
        self.policy = nn.Linear(self.fc1_dims, n_actions)
        self.value = nn.Linear(self.fc1_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # loss function for td learning
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, observation):
        state_tensor = T.Tensor(observation[None,...]).to(self.device) 
        conv1 = F.relu(self.conv1(state_tensor))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # Note: conv3 shape is BS x n_filters x H x W
        # conv_state shape is BS x (n_filters * H * W)
        conv_state = conv3.view(conv3.size()[0], -1)

        flat1 = F.relu(self.fc1(conv_state))
        policy = F.softmax(self.policy(flat1))
        value = self.value(flat1) # do softmax here
          # we can activate this differently depending on whether it's the actor or critic network

        return policy, value

## write networks

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class SRNet(nn.Module):
    def __init__(self):
        super(SRNet, self).__init__()

        self.rep_layer_dims = 400
        hidden_units_sr = (self.rep_layer_dims * 4,) ## currently only using one layer from phi to psi (sr)
        in_channels = 3

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1)
        self.rep = nn.Linear(5 * 5 * 32, self.rep_layer_dims)

        self.lin1 = nn.Linear(self.rep_layer_dims, 5 * 5 * 32)
        self.dconv1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=1)
        self.dconv2 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1)
        self.dconv3 = nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2)

        self.sr = nn.Linear(self.rep_layer_dims, self.rep_layer_dims)

    def forward(self, x):
        x = torch.Tensor(x)
        batch_size = x.shape[0]

        # encode
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # cast x into correct shape
        x = x.view(batch_size, 32 * 5 * 5)

        # get latent representation
        phi = F.relu(self.rep(x))

        # get SR
        psi = F.relu(self.sr(phi))


        x = F.relu(self.lin1(phi))
        # cast x into correct shape
        x = x.view(batch_size, 32, 5, 5)

        # decode
        x = F.relu(self.dconv1(x))
        x = F.relu(self.dconv2(x))
        reconstruct = F.relu(self.dconv3(x))
        return phi, psi, reconstruct

## Write Classes for learning state representations
# OSFM - one step forward model -- trains to predict next state from
#        current state and action
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_action(env):
    return np.random.choice(len(env.action_list))

def get_conv_samples(env,maxsteps):
    data_col = [[], [], []]
    onehot_a = np.zeros(env.action_space.n)
    env.reset()
    for step in range(maxsteps):
        s = env.get_state()
        state = env.get_observation()

        action = get_action(env)

        onehot_a[:] = 0
        onehot_a[action] = 1

        s_prime, r, done, __ = env.step(action)
        next_state = env.get_observation()

        data_col[0].append(state)
        data_col[1].append(onehot_a.copy())
        data_col[2].append(next_state)

        # env.render(0.05)

        if step == maxsteps - 1 or done:
            # plt.show(block=True)
            pass

        if done:
            env.reset()
            # break
    return data_col

def get_onehot_samples(env, maxsteps):
    data_col = [[], [], []]
    onehot_a = np.zeros(env.action_space.n)
    env.reset()

    for step in range(maxsteps):
        s = env.get_state
        state = env.get_observation(onehot=True)[0]

        action = get_action(env)
        onehot_a[:] = 0
        onehot_a[action] = 1

        s_prime, r, done, _ = env.step(action)
        next_state = env.get_observation(onehot=True)[0]

        data_col[0].append(state)
        data_col[1].append(np.expand_dims(onehot_a.copy(), 0))
        data_col[2].append(next_state)

        if done:
            env.reset()

    return data_col

class Conv_OSFM(nn.Module):
    def __init__(self):
        super(Conv_OSFM, self).__init__()

        self.rep_layer_dims = 400
        hidden_units_sr = (self.rep_layer_dims * 4,)
        in_channels = 3
        num_actions = 4

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1)
        self.rep = nn.Linear(5 * 5 * 32, self.rep_layer_dims)

        self.lin1 = nn.Linear(self.rep_layer_dims + num_actions, 5 * 5 * 32)
        # self.lin2  = nn.Linear(5*5*32, 5*5*32)
        self.dconv1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=1)
        self.dconv2 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1)
        self.dconv3 = nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2)

        self.sr = nn.Linear(self.rep_layer_dims, self.rep_layer_dims)

    def forward(self, state, action):
        action = torch.Tensor(action)  # torch.unsqueeze(torch.Tensor(action),-1)
        x = torch.Tensor(state)
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
        psi = F.relu(self.sr(phi))  ## no relu

        x = torch.cat((phi, action), 1)
        x = F.relu(self.lin1(x))

        # x = F.relu(self.lin2(x))

        # cast x into correct shape
        x = x.view(batch_size, 32, 5, 5)

        # decode
        x = F.relu(self.dconv1(x))

        x = F.relu(self.dconv2(x))

        reconstruct = F.tanh(self.dconv3(x))

        return phi, psi, reconstruct

class FC_OSFM(nn.Module):
    def __init__(self):
        super(FC_OSFM, self).__init__()
        num_actions = 4
        self.lin1 = nn.Linear(400, 300)
        self.lin2 = nn.Linear(300, 200)
        self.lin3 = nn.Linear(200, 100)
        self.rep = nn.Linear(100, 400)

        self.rlin1 = nn.Linear(400 + num_actions, 200)
        self.rlin2 = nn.Linear(200, 300)
        self.rlin3 = nn.Linear(300, 400)

        self.sr = nn.Linear(400, 400)

    def forward(self, state, action):
        a = torch.Tensor(action)
        x = torch.Tensor(state)

        x = F.relu(self.lin1(x))

        x = F.relu(self.lin2(x))

        x = F.relu(self.lin3(x))

        phi = F.relu(self.rep(x))

        psi = F.relu(self.sr(phi))

        x = torch.cat((phi, a), -1)
        x = F.relu(self.rlin1(x))

        x = F.relu(self.rlin2(x))

        reconst = F.tanh(self.rlin3(x))

        return phi, psi, reconst




### JUNKYARD
def plot_phi(phi):
    data = phi[0].detach().numpy()
    print(data.shape)
    plt.imshow(data, aspect='auto')
    plt.show()

def plot_frames(obsr):
    obs = obsr[0]

    fig, axs = plt.subplots(1, 3, sharey=True)
    cmap = 'bone_r'
    titles = ['Grid', 'Reward', 'Agent']
    for i in range(3):
        ax = axs[i]
        pcm = ax.pcolor(obs[i], cmap=cmap, vmin=0, vmax=1)
        fig.colorbar(pcm, ax=ax, shrink=0.4)
        ax.set_title(titles[i])
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    plt.show()
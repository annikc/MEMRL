import torch as T
import torch.nn as nn # to handle layers
import torch.nn.functional as F # for activation function
import torch.optim as optim # for optimizer
import numpy as np

class DQN(nn.Module):
    def __init__(self,input_dims, fc1_dims, fc2_dims, n_actions, lr):
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims   = fc1_dims
        self.fc2_dims   = fc2_dims
        self.output_dims= n_actions

        self.fc1  = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2  = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3  = nn.Linear(self.fc2_dims, self.output_dims)

        self.lr         = lr
        self.optimizer  = optim.Adam(self.parameters(), lr=self.lr)
        self.loss       = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class DQ_agent(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_decay=1e-4,
                 replace_target=0):
        self.gamma   = gamma
        self.set_epsilon(epsilon)
        self.eps_min = eps_end
        self.eps_dec = eps_decay
        self.lr      = lr
        self.action_space = [i for i in range(n_actions)]

        self.mem_size     = max_mem_size
        self.batch_size   = batch_size
        self.mem_cntr     = 0
        self.iter_cntr = 0
        self.replace_target = replace_target

        FC1_DIMS = 256
        FC2_DIMS = 256

        self.Q_eval = DQN(input_dims, n_actions=n_actions, fc1_dims=FC1_DIMS,
                          fc2_dims=FC2_DIMS, lr=self.lr)
        if self.replace_target == 0:
            self.Q_next = self.Q_eval
        else:
            self.Q_next = DQN(input_dims, n_actions=n_actions,
                              fc1_dims=FC1_DIMS, fc2_dims=FC2_DIMS, lr=self.lr)
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        # memory buffers
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory    = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory    = np.zeros(self.mem_size, dtype=np.float32)

        # value of terminal state is always 0, so there are no future rewards
        # to attain
        self.terminal_memory  = np.zeros(self.mem_size, dtype=np.bool)


    def store_transition(self, state, action, reward, next_state, done):
        # wrap around and rewrite earliest memories after memory is full
        index = self.mem_cntr % self.mem_size
        self.state_memory[index]     = state
        self.action_memory[index]    = action
        self.reward_memory[index]    = reward
        self.new_state_memory[index] = next_state
        self.terminal_memory[index]  = done

        self.mem_cntr += 1

    def choose_action(self,observation):
      # epsilon greedy action selection
        if np.random.random() > self.epsilon:
            # take best action (square brackets makes it a batch of size 1)
            state = T.Tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def set_epsilon(self, epsilon):
        if not (0 <= epsilon <= 1.0):
            raise ValueError("Epsilon must be between 0 and 1.")
        if not hasattr(self, "epsilon"):
            self.epsilon = epsilon
        else:
            if epsilon > self.eps_min:
                self.epsilon = epsilon
            else:
                self.epsilon = self.eps_min

    def learn(self):
        if self.mem_cntr < self.batch_size:
            # don't bother learning if you don't have enough examples in memory
            return
        else:
            self.Q_eval.optimizer.zero_grad()
            max_mem = min(self.mem_size, self.mem_cntr)
            # sample from memory without replacement
            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            batch_state     = T.Tensor(self.state_memory[batch]).to(self.Q_eval.device)

            batch_new_state = T.Tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
            batch_reward    = T.Tensor(self.reward_memory[batch]).to(self.Q_eval.device)
            batch_terminal  = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
            # batch_action doesn't need to be torch tensor
            batch_action    = self.action_memory[batch]

            q_eval = self.Q_eval(batch_state)[batch_index, batch_action]
            q_next = self.Q_next(batch_new_state)
            q_next[batch_terminal] = 0.0

            # max along action dimension (retain 0th val since max returns
            # [value, index], and we just want value)
            q_target = batch_reward + self.gamma * T.max(q_next, dim=1)[0]

            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward() # backprop the error
            self.Q_eval.optimizer.step() # weight update

            # now decrease epsilon value
            self.set_epsilon(self.epsilon - self.eps_dec)

            self.iter_cntr += 1
            if (self.replace_target != 0) and (self.iter_cntr % self.replace_target == 0):
                self.Q_next.load_state_dict(self.Q_eval.state_dict())
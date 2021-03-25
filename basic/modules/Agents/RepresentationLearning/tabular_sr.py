import numpy as np
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA

import sys
import gym


cmap = plt.cm.viridis
cmap.set_bad(color='white')

def onehot(state_index, nstates):
    vec = np.zeros(nstates)
    vec[state_index] = 1
    return vec

def twohot(value, max_value):
    vec_1 = np.zeros(max_value)
    vec_2 = np.zeros(max_value)
    vec_1[value[0]] = 1
    vec_2[value[1]] = 1
    return np.concatenate([vec_1, vec_2])

def mask_grid(grid, blocks, mask_value=-100):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if [i,j] in blocks:
                grid[i,j] = mask_value
    grid = np.ma.masked_where(grid == mask_value, grid)
    return grid

class Tabular_SR_Agent(object):
  def __init__(self, env, gamma, learning_rate, epsilon=1):
    self.env = env
    self.state_space = env.nstates
    self.action_space = env.action_space.n
    self.M          = np.zeros((self.action_space, self.state_space, self.state_space))
    self.reward_est = np.zeros(self.state_space)
    self.q_table    = np.zeros((self.state_space, self.action_space)) ## come back to this

    self.lr = learning_rate
    self.gamma = gamma
    self.epsilon = epsilon

    self.transitions = []

  def onehot(self, state_index):
    vec = np.zeros(self.state_space)
    vec[state_index] = 1
    return vec

  def choose_action(self, state):
    if np.random.random()>self.epsilon:
      #action = np.argmax(self.q_table[state])
      action = np.argmax(self.get_q_estimate(state))
    else:
      action=np.random.choice(self.action_space)
    return action

  def update_q_table(self, current_state, current_action, reward, new_state):
    # this function describes how the Q table gets updated so the agent can make
    # better choices based on what it has experienced from the environment
    current_q = self.q_table[current_state, current_action]
    max_future_q = np.max(self.q_table[new_state,:])

    new_q = (1-self.lr)*current_q + self.lr*(reward + self.gamma*max_future_q)
    self.q_table[current_state, current_action] = new_q

  def get_q_estimate(self, state):
    reward_est = self.reward_est
    # given my current state, get successor states for all actions
    # multiply with onehot reward estimate to get column with M values across actions
    q_est = np.matmul(self.M[:,state,:], reward_est)

    return q_est


  def update(self):
    s, a, s_, r, done = self.transitions[-2]
    a_                = self.transitions[-1][1]

    # update successor matrix self.M
    if done:
      sr_error = self.onehot(s) + self.gamma*self.onehot(s_) - self.M[a,s,:]
    else:
      sr_error = self.onehot(s) + self.gamma*self.M[a_,s_,:] - self.M[a,s,:]

    self.M[a,s,:] += self.lr * sr_error

    #update reward estimate self.reward_est
    reward_error = r - self.reward_est[s_]
    self.reward_est[s_] += self.lr*reward_error

    return sr_error, reward_error


  def step_in_env(self,state):
    action = self.choose_action(state)
    next_state, reward, done, _ = self.env.step(action)
    self.transitions.append([state,action,next_state,reward,done])
    return next_state

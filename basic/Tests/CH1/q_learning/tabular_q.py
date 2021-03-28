import numpy as np
import matplotlib.pyplot as plt
import gym
from modules.Agents import Tabular_Q_Agent as Q_Agent
from modules.Experiments.tabular_q_expt import Q_Expt, pref_Q_action

from modules.Utils import running_mean as rm

# create environment
env_name   = 'gym_grid:gridworld-v1'
env = gym.make(env_name)
plt.close()

N_epsiodes = 5000
# create agent
agent = Q_Agent(env,end_eps_decay=N_epsiodes/2)

# create experiment
expt = Q_Expt(agent, env)

expt.run(N_epsiodes)

collected_rewards = expt.data['total_reward']
pref_action = pref_Q_action(env, agent.q_table)

# plot results
fig,ax = plt.subplots(1,2)
ax[0].plot(collected_rewards)
ax[1].imshow(pref_action, interpolation='none')
ax[1].axis('off')
plt.show()

'''





fig,ax = plt.subplots(2,1,sharex=False)
agg_ep_rewards = run.agg_ep_rewards
ax[0].plot(agg_ep_rewards['ep'], agg_ep_rewards['avg'], label='avg')
ax[0].plot(agg_ep_rewards['ep'], agg_ep_rewards['min'], label='min')
ax[0].plot(agg_ep_rewards['ep'], agg_ep_rewards['max'], label='max')
ax[0].legend(loc=0)
a = ax[1].imshow(pref_action, interpolation='none')
cbar = fig.colorbar(a, ax=ax[1],ticks=[0,1,2,3])
cbar.ax.set_yticklabels(['Down','Up','Right','Left'])


plt.show()
'''
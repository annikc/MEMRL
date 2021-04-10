####
# File uses basic fully connected actor critic architecture to train in a gridworld environment
# State inputs are one-hot vectors of discrete grid states -- similar experiments are done in CH2/ for different
# styles of representation of state (i.e. place cell, successor representation)
# No memory module is used in this run
# Data can be logged by calling the .record_log() function of the experiment class
#
#####
import numpy as np
import gym
import matplotlib.pyplot as plt
import sys
sys.path.append('../../modules/')
from basic.modules.Agents.Networks import flex_ActorCritic as Network
from basic.modules.Agents.Networks import params
from basic.modules.Agents import Agent
from basic.modules.Agents.RepresentationLearning.learned_representations import convs, reward_convs

rep_types = {'convs':convs, 'reward_convs':reward_convs}

from basic.modules.Experiments import expt
#from modules.Utils.gridworld_plotting import plot_pref_pol, plot_polmap





## set parameters for run -- environment, representation type
write_to_file = 'conv_mf_training.csv'
env_name = 'gridworld:gridworld-v5'
representation_type = 'reward_convs'
num_trials = 5000
num_events = 250

for _ in range(2):
    env = gym.make(env_name)
    plt.close()

    state_reps, representation_name, input_dims, _ = rep_types[representation_type](env)

    network_parameters = params(env)
    print(network_parameters.__dict__)
    network = Network(network_parameters)

    print(network)
    agent = Agent(network, state_representations=state_reps)

    ex = expt(agent, env)
    ex.run(num_trials,num_events,printfreq=10)

    ex.conv_record_log(env_name=env_name, representation_type=representation_name,
                      n_trials=num_trials, n_steps=num_events,
                      dir='../../Data/', file=write_to_file)

fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(ex.data['total_reward'])
ax[1].plot(ex.data['loss'][0], label='P_loss')
ax[1].plot(ex.data['loss'][1], label='V_loss')
ax[1].legend(bbox_to_anchor=(1.05,0.95))
plt.show()

'''
for _ in range(1):
    ## generate the environment object
    env = gym.make(env_name)
    #plt.close()

    ## get state representations to be used
    state_reps, representation_name, input_dims, _ = rep_types[representation_type](env)

    ## create an actor-critic network and associated agent
    network = Network(input_dims=[input_dims], fc1_dims=200, fc2_dims=200, output_dims=env.action_space.n, lr=0.00005)
    memory = Memory(entry_size=env.action_space.n, cache_limit=400, mem_temp=1)
    agent = Agent(network, state_representations=state_reps, memory=memory)

    # create an experiment class instance
    ex = expt(agent, env)

    ex.run(num_trials, num_events)

    ex.record_log(env_name=env_name, representation_type=representation_name,
                  n_trials=num_trials, n_steps=num_events,
                  dir='./Data/', file=write_to_file)

# print results of training
fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(ex.data['total_reward'])
ax[1].plot(ex.data['loss'][0], label='P_loss')
ax[1].plot(ex.data['loss'][1], label='V_loss')
ax[1].legend(bbox_to_anchor=(1.05,0.95))


ep_pols = np.zeros(env.shape, dtype=[(x,'f8') for x in env.action_list])

for state_i in range(env.nstates):
    state_obs = state_reps[state_i]
    h = ex.agent.EC.recall_mem(state_obs)
    ep_pols[env.oneD2twoD(state_i)] = tuple(h)

plot_pref_pol(env, ep_pols)

plt.show()
'''

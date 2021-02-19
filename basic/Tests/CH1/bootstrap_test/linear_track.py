import torch
import gym
import numpy as np
import modules.Agents.Networks as nets
import modules.Agents.EpisodicMemory as Memory
from modules.Agents import Agent
from modules.Experiments import gridworldExperiment as expt
import matplotlib.pyplot as plt
from modules.Utils import running_mean as rm

env_name   = 'gym_grid:gridworld-v112'
network_id = None # '97b5f281-a60e-4738-895d-191a04edddd6'
ntrials    = 1000

# create environment
env = gym.make(env_name)
plt.show()

# generate network
if network_id == None:
    # generate parameters for network from environment observation shape
    params = nets.fc_params(env)
    params.lr = 0.005
    params.hidden_types = ['linear']
    params.hidden_dims = [40]
    params.temp = 1.1
    network = nets.ActorCritic(params)
else:
    network = torch.load(f=f'./Data/agents/load_agents/{network_id}.pt')

memtemp = 1
memory = Memory.EpisodicMemory(cache_limit=400, entry_size=env.action_space.n, mem_temp=memtemp)

agent = Agent(network, memory=memory)
agent.get_action = agent.MF_action

opt_values = np.zeros(env.nstates)
reward_loc = env.twoD2oneD(list(env.rewards.keys())[0])
opt_values[reward_loc] = list(env.rewards.values())[0]

for ind in reversed(range(len(opt_values)-1)):
    opt_values[ind] = env.step_penalization + agent.gamma*opt_values[ind+1]


ntrials = 1000
nevents = 100
run = expt(agent, env)
run.data['opt_values'] = opt_values

run.run(NUM_TRIALS=ntrials, NUM_EVENTS=nevents)
run.record_log(dir='../../../Data/' ,file='linear_track.csv', expt_type=f'{type(run).__name__}_lr{params.lr}', env_name=env_name, n_trials=ntrials, n_steps=nevents)


def get_V_norm_evol():
    normie = []
    for x in run.data['V_snap']:
        #print(x)
        #print(x-run.data['opt_values'])

        normie.append(np.linalg.norm(x-run.data['opt_values']))

    return normie

norm = get_V_norm_evol()
plt.plot(norm)
plt.show()

def plot_reward_loss():
    smoothing = 10
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[0].plot(rm(run.data['total_reward'],smoothing), 'k', alpha=0.5)
    if "bootstrap_reward" in run.data.keys():
        ax[0].plot(rm(run.data['bootstrap_reward'],smoothing),'r')

    ax[1].plot(rm(run.data['loss'][0], smoothing), label='ec_p')
    ax[2].plot(rm(run.data['loss'][1], smoothing), label='ec_v')

    if "mf_loss" in run.data.keys():
        ax[1].plot(rm(run.data['mf_loss'][0], smoothing), label='mf_p')
        ax[2].plot(rm(run.data['mf_loss'][1], smoothing), label='mf_v')

    ax[1].legend(loc=0)
    ax[2].legend(loc=0)

    plt.show()
    plt.close()
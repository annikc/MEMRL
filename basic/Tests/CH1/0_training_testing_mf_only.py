import torch
import gym
import numpy as np
from modules.Agents.RepresentationLearning.learned_representations import onehot, random, place_cell, sr, latents
import modules.Agents.Networks as nets
from modules.Agents import Agent
from modules.Experiments import expt
import matplotlib.pyplot as plt
from modules.Utils import running_mean as rm
import argparse

# set up arguments to be passed in and their defauls
parser = argparse.ArgumentParser()
parser.add_argument('-v', type=int, default=1)
parser.add_argument('-rep', default='onehot')
parser.add_argument('-lr', default=0.0005)
parser.add_argument('-cache', type=int, default=100)
parser.add_argument('-dist', default='chebyshev')
args = parser.parse_args()

# parameters set with command line arugments
version         = args.v
rep_type        = args.rep
learning_rate   = args.lr
cache_size      = args.cache
distance_metric = args.dist

print(args)
write_to_file = 'naive_mf.csv'
directory = './Data/' # ../../Data if you are in Tests/CH2
env_name = f'gridworld:gridworld-v{version}'

num_trials = 5000
num_events = 250

# make gym environment
env = gym.make(env_name)
plt.close()
cache_size_for_env = int(len(env.useable)*(cache_size/100))
print(env.rewards)
rep_types = {'onehot':onehot, 'random':random, 'place_cell':place_cell, 'sr':sr, 'latent':latents}
state_reps, representation_name, input_dims, _ = rep_types[rep_type](env)


# load weights to head_ac network from previously learned agent
AC_head_agent = nets.fc_ActorCritic([input_dims], fc1_dims=200, fc2_dims=200, output_dims=env.action_space.n, lr=learning_rate)

memory = None #Memory.EpisodicMemory(cache_limit=cache_size_for_env, entry_size=env.action_space.n)

agent = Agent(AC_head_agent, memory=memory, state_representations=state_reps)


run = expt(agent, env)
run.run(NUM_TRIALS=num_trials, NUM_EVENTS=num_events)

test_env = gym.make(env_name+'1')
plt.close()
print(test_env.rewards)
test_run = expt(agent, test_env)
test_run.data = run.data
test_run.run(NUM_TRIALS=num_trials*2,NUM_EVENTS=num_events)

test_run.record_log(env_name, representation_name,num_trials*3,num_events,dir=directory, file=write_to_file)

'''
smoothing=10
plt.figure()
plt.plot(rm(run.data['total_reward'],smoothing), c='k', alpha=0.5)
if 'bootstrap_reward' in run.data.keys():
    plt.plot(rm(run.data['bootstrap_reward'],smoothing), c='r')
plt.show()
'''
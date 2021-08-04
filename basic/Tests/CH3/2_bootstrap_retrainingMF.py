import torch
import gym
import pickle
import pandas as pd
import numpy as np
from modules.Agents.RepresentationLearning.learned_representations import onehot, random, place_cell, sr, latents
import modules.Agents.Networks as nets
import modules.Agents.EpisodicMemory as Memory

from modules.Agents import Agent
from modules.Experiments import Bootstrap as expt
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

reps_for_reads = {'onehot':'onehot', 'sr':'analytic successor', 'place_cell':'place_cell','random':'random'}
env_name = f'gridworld:gridworld-v{version}'
directory = './Data/' # ../../Data if you are in Tests/CH2

df = pd.read_csv(directory+'naive_mf.csv')
gb = df.groupby(['env_name','representation'])["save_id"]
print(env_name,rep_type)
id_list = list(gb.get_group((env_name, reps_for_reads[rep_type])))
load_from = id_list[np.random.choice(len(id_list))]
print(f"Loading: {env_name[-12:]} {rep_type} {load_from}" )

print(args)
write_to_file = 'bootstrap_retrain_mf.csv'


env_name = f'gridworld:gridworld-v{version}1'

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
AC_head_agent = nets.flat_ActorCritic(input_dims, env.action_space.n, lr=learning_rate)


if load_from != None:
    AC_head_agent.load_state_dict(torch.load(directory+f'agents/{load_from}.pt'))
    print(f"weights loaded from {load_from}")

memory = Memory.EpisodicMemory(cache_limit=cache_size_for_env, entry_size=env.action_space.n)

agent = Agent(AC_head_agent, memory=memory, state_representations=state_reps)


run = expt(agent, env)
run.run(NUM_TRIALS=num_trials, NUM_EVENTS=num_events)
run.record_log(env_name, representation_name,num_trials,num_events,dir=directory, file=write_to_file, load_from=load_from)


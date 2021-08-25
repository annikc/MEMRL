# Tests/CH2/_3ec_distance_test.py
import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import torch
import sys
sys.path.append('../../modules')
from modules.Agents.Networks import flat_ActorCritic as head_AC
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents.RepresentationLearning.learned_representations import onehot, random, place_cell, sr, latents
from modules.Agents import Agent
from modules.Experiments import flat_expt
import pandas as pd
sys.path.append('../../../')
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


# parameters set for this file
relative_path_to_data = './Data/' # from within Tests/CH1
write_to_file         = f'train_test_ec.csv'
training_env_name     = f'gridworld:gridworld-v{version}'
test_env_name         = training_env_name+'1'
num_trials = 5000
num_events = 250

cache_limits = {'gridworld:gridworld-v11':{100:400, 75:300, 50:200, 25:100},
                'gridworld:gridworld-v31':{100:365, 75:273, 50:182, 25:91},
                'gridworld:gridworld-v41':{100:384, 75:288, 50:192, 25:96},
                'gridworld:gridworld-v51':{100:286, 75:214, 50:143, 25:71}}

cache_size_for_env = int(cache_limits[test_env_name][100] *(cache_size/100))

# make new env to run test in
env = gym.make(training_env_name)
plt.close()

rep_types = {'onehot':onehot, 'random':random, 'place_cell':place_cell, 'sr':sr}
if rep_type == 'latents':
    conv_ids = {'gridworld:gridworld-v1':'c34544ac-45ed-492c-b2eb-4431b403a3a8',
                'gridworld:gridworld-v3':'32301262-cd74-4116-b776-57354831c484',
                'gridworld:gridworld-v4':'b50926a2-0186-4bb9-81ec-77063cac6861',
                'gridworld:gridworld-v5':'15b5e27b-444f-4fc8-bf25-5b7807df4c7f'}
    run_id = conv_ids[f'gridworld:gridworld-v{version}']
    agent_path = relative_path_to_data+f'agents/saved_agents/{run_id}.pt'
    state_reps, representation_name, input_dims, _ = latents(env, agent_path)
else:
    state_reps, representation_name, input_dims, _ = rep_types[rep_type](env)

AC_head_agent = head_AC(input_dims, env.action_space.n, lr=learning_rate)

df = pd.read_csv(relative_path_to_data+write_to_file)
df_gb = df.groupby(['env_name','representation'])["save_id"]
id = list(df_gb.get_group((test_env_name,representation_name)))[0]
print(id)

memory = Memory(entry_size=env.action_space.n, cache_limit=cache_size_for_env, distance=distance_metric)
with open(relative_path_to_data+f'/ec_dicts/{id}_EC.p', 'rb') as f:
    load_mem = pickle.load(f)
memory.cache_list = load_mem

agent = Agent(AC_head_agent, memory=memory, state_representations=state_reps)

#run = flat_expt(agent, env)
#run.run(NUM_TRIALS=num_trials, NUM_EVENTS=num_events)

test_env = gym.make(test_env_name)
plt.close()
print(test_env.rewards)
test_run = flat_expt(agent, test_env)
#test_run.data = run.data
test_run.run(NUM_TRIALS=num_trials*3,NUM_EVENTS=num_events)

test_run.record_log(test_env_name, representation_name,num_trials*3,num_events,dir=relative_path_to_data, file=write_to_file,load_from=id)
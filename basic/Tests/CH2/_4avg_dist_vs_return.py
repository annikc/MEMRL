# Tests/CH2/_2ec_rep_test.py
import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import torch
import sys
sys.path.append('../../modules')
from modules.Agents.Networks import flat_ActorCritic as head_AC
from modules.Agents.EpisodicMemory import distEM as Memory
from modules.Agents.RepresentationLearning.learned_representations import onehot, random, place_cell, sr, latents
from modules.Agents import distAgent as Agent
from modules.Experiments import flat_dist_return as flat_expt
sys.path.append('../../../')
import argparse

# set up arguments to be passed in and their defauls
parser = argparse.ArgumentParser()
parser.add_argument('-v', type=int, default=1)
parser.add_argument('-rep', default='onehot')
parser.add_argument('-load', type=bool, default=True)
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
write_to_file         = 'ec_avg_dist_rtn.csv'
training_env_name     = f'gridworld:gridworld-v{version}'
test_env_name         = training_env_name+'1'
num_trials = 1000
num_events = 250

cache_limits = {'gridworld:gridworld-v11':{100:400, 75:300, 50:200, 25:100},
                'gridworld:gridworld-v31':{100:365, 75:273, 50:182, 25:91},
                'gridworld:gridworld-v41':{100:384, 75:288, 50:192, 25:96},
                'gridworld:gridworld-v51':{100:286, 75:214, 50:143, 25:71}}

cache_size_for_env = int(cache_limits[test_env_name][100] *(cache_size/100))   #cache_limits[test_env_name][cache_size]

# make new env to run test in
env = gym.make(test_env_name)
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

memory = Memory(entry_size=env.action_space.n, cache_limit=cache_size_for_env, distance=distance_metric)

agent = Agent(AC_head_agent, memory=memory, state_representations=state_reps)

ex = flat_expt(agent, env)
print(f"Experiment running {env.unwrapped.spec.id} \nRepresentation: {representation_name} \nCache Limit:{cache_size_for_env}")
ex.run(num_trials,num_events,snapshot_logging=False)

ex.record_log(env_name=test_env_name, representation_type=representation_name,
              n_trials=num_trials, n_steps=num_events,
              dir=relative_path_to_data, file=write_to_file)


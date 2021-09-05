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
import pandas as pd
import torch
import sys
sys.path.append('../../modules/')
sys.path.append('../../../basic')
from modules.Agents.Networks import flex_ActorCritic as Network
from modules.Agents.Networks import shallow_ActorCritic
from modules.Agents.Networks import conv_PO_params, conv_FO_params
from modules.Agents import Agent
from modules.Agents.RepresentationLearning.learned_representations import convs, reward_convs, onehot, random, place_cell, sr
from modules.Experiments import Bootstrap_shallow as expt
import argparse

from modules.Agents.EpisodicMemory import distEM as Memory



# set up arguments to be passed in and their defauls
parser = argparse.ArgumentParser()
parser.add_argument('-v', type=int, default=4)
parser.add_argument('-rep', default='conv')
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

## set parameters for run
write_to_file         = 'conv_mf_bootstrap.csv'
relative_path_to_data = './Data/' # from within Tests/CH1
env_name              = f'gridworld:gridworld-v{version}'
num_trials            = 5000
num_events            = 250

# valid representation types for this experiment
rep_types = {'conv':convs, 'reward_conv':reward_convs}
param_set = {'conv': conv_PO_params, 'reward_conv': conv_FO_params}

df = pd.read_csv(relative_path_to_data+'conv_mf_training.csv')
groups_to_split = ['env_name','representation']
df_gb = df.groupby(groups_to_split)["save_id"]

id_list = list(df_gb.get_group((env_name, rep_type)))
agent_id = np.random.choice(id_list)
print(env_name, rep_type, agent_id)

# saved weights
saved_network = torch.load(relative_path_to_data+f'agents/saved_agents/{agent_id}.pt')

# load agent weights into new network
network = shallow_ActorCritic(input_dims=600, hidden_dims=400,output_dims=4,lr=5e-4)
new_state_dict = {}
for key in saved_network.keys():
    if key[0:6] == 'output':
        if key[7] == '0':
            new_key = 'pol'+key[8:]
            new_state_dict[new_key] = saved_network[key]
        elif key[7] == '1':
            new_key = 'val'+key[8:]
            new_state_dict[new_key] = saved_network[key]
    elif key[0:8] =='hidden.5':
        new_key = 'hidden'+key[8:]
        new_state_dict[new_key] = saved_network[key]

network.load_state_dict(new_state_dict)
print(network)


# instantiate the environment for the experiment
env = gym.make(env_name+'1')
plt.close()

# get representation type, associated parameters to specify the network dimensions
state_reps, representation_name, input_dims, _ = rep_types[rep_type](env)
params = param_set[rep_type]
conv_net = Network(params(env))
conv_net.load_state_dict(saved_network)

latent_state_reps = {}
for key, value in state_reps.items():
    out = conv_net(value)
    vec = conv_net.test_activity[0].detach().numpy()
    latent_state_reps[key] = vec

cache_limits = {'gridworld:gridworld-v11':{100:400, 75:300, 50:200, 25:100},
                'gridworld:gridworld-v31':{100:365, 75:273, 50:182, 25:91},
                'gridworld:gridworld-v41':{100:384, 75:288, 50:192, 25:96},
                'gridworld:gridworld-v51':{100:286, 75:214, 50:143, 25:71}}
cache_size_for_env = int(cache_limits[env_name+'1'][100] *(cache_size/100))
memory = Memory(entry_size=env.action_space.n, cache_limit=cache_size_for_env, distance=distance_metric)

# reinitalize agent with new network
agent = Agent(network, memory, state_representations=latent_state_reps)


#verify_env = gym.make(env_name)
#ver_ex = expt(agent,verify_env)

# expt - redefines logging function to keep track of network details
ex = expt(agent, env)
ex.run(num_trials,num_events)
ex.record_log(env_name=env_name, representation_type=representation_name, n_trials=num_trials, n_steps=num_events, dir=relative_path_to_data, file=write_to_file, load_from=agent_id)

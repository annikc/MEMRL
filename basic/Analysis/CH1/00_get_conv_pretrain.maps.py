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
import torch
import matplotlib.pyplot as plt
import pandas as pd
from modules.Agents.Networks import flex_ActorCritic as Network
from modules.Agents.Networks import conv_PO_params, conv_FO_params
from modules.Agents import conv_randomwalk_agent as Agent
from modules.Agents.RepresentationLearning.learned_representations import convs, reward_convs
from modules.Experiments import conv_expt


## set parameters for run
write_to_file         = 'conv_mf_pretraining.csv'
relative_path_to_data = '../../Data/' # from within Tests/CH1
df = pd.read_csv(relative_path_to_data+write_to_file)
df_gb = df.groupby(['env_name','representation'])['save_id']

# valid representation types for this experiment
rep_types = {'conv':convs, 'rwd_conv':reward_convs}
param_set = {'conv': conv_PO_params, 'rwd_conv': conv_FO_params}

representation_type = 'rwd_conv'
env_name = 'gridworld:gridworld-v01'
env = gym.make(env_name)
plt.close()

# get representation type, associated parameters to specify the network dimensions
state_reps, representation_name, input_dims, _ = rep_types[representation_type](env)
params = param_set[representation_type]
network_parameters = params(env)

# make a new network instance
network = Network(network_parameters)

agent_id = list(df_gb.get_group((env_name,representation_type)))[0]

state_dict = torch.load(relative_path_to_data+f'agents/{agent_id}.pt')
network.load_state_dict(state_dict)

val_array = np.zeros((20,20))
for k, v in state_reps.items():
    coord = env.oneD2twoD(k)
    p, v = network(v)
    v = v.item()
    print(p.detach().numpy())
    val_array[coord] = v

a= plt.imshow(val_array, vmin=-1, vmax=0)
plt.colorbar(a)
plt.show()
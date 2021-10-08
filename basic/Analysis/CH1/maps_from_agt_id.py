import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Analysis.analysis_utils import structured_unstructured, LINCLAB_COLS, plot_specs, analysis_specs, fade, convert_pol_array_to_pref_dir
from modules.Utils.gridworld_plotting import plot_pref_pol, plot_valmap
import torch
import gym
import numpy as np
from modules.Agents.RepresentationLearning.learned_representations import onehot, random, place_cell, sr, convs, conv_PO_params
import modules.Agents.Networks as nets
import modules.Agents.EpisodicMemory as Memory

from modules.Agents import Agent


# get data frame
parent_path = '../../Data/'
#df = pd.read_csv(parent_path+'train_test_shallowAC.csv')
df = pd.read_csv(parent_path+'conv_mf_retraining.csv')
#df['representation'] = df['representation'].apply(structured_unstructured)
print(df.representation.unique())
groups_to_split = ['env_name','representation','num_trials']
gb = df.groupby(groups_to_split)["save_id"]

cache_limits = analysis_specs['cache_limits']

## get agent ids matching specs

env_name = 'gridworld:gridworld-v41'
rep      = 'conv'
pct      = 100
index    = 5000
num_trials = 25000
id_list  = list(gb.get_group((env_name[0:23], rep, num_trials)))
print(id_list)
load_from = np.random.choice(id_list)
print(load_from)


reps_for_reads = {'onehot':'onehot', 'sr':'analytic successor', 'place_cell':'place_cell','random':'random'}

# make gym environment
env = gym.make(env_name)
plt.close()
rep_types = {'unstructured':onehot, 'random':random, 'place_cell':place_cell, 'structured':sr, 'conv':convs}
if rep == 'conv':
    # saved weights
    saved_network = torch.load('../../Data/agents/6a956906-c06c-47ef-aad1-3593fb9068d1.pt')

    # load agent weights into new network
    network = nets.shallow_ActorCritic(input_dims=600, hidden_dims=400,output_dims=4,lr=5e-4)
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


    # instantiate the environment for the experiment
    #env = gym.make(env_name+'1')
    #plt.close()

    # get representation type, associated parameters to specify the network dimensions
    state_reps, representation_name, input_dims, _ = rep_types[rep](env)
    params = conv_PO_params
    conv_net = nets.flex_ActorCritic(params(env))
    conv_net.load_state_dict(saved_network)

    latent_state_reps = {}
    for key, value in state_reps.items():
        out = conv_net(value)
        vec = conv_net.test_activity[0].detach().numpy()
        latent_state_reps[key] = vec

    state_reps = latent_state_reps
    representation_name = 'latents'
    input_dims = 400
    # load weights to head_ac network from previously learned agent
    AC_head_agent = nets.shallow_ActorCritic(600, hidden_dims=400, output_dims=env.action_space.n, lr=0.005)

else:
    state_reps, representation_name, input_dims, _ = rep_types[rep](env)
    AC_head_agent = nets.shallow_ActorCritic(input_dims, hidden_dims=200, output_dims=env.action_space.n, lr=0.005)



if load_from != None:
    AC_head_agent.load_state_dict(torch.load(parent_path+f'agents/{load_from}.pt'))
    print(f"weights loaded from {load_from}")

memory = Memory.EpisodicMemory(cache_limit=400, entry_size=env.action_space.n)
agent = Agent(AC_head_agent, memory=memory, state_representations=state_reps)


## generate pol maps by getting policies for each state in state reps
pol_array = np.zeros((20,20), dtype=[(x,'f8') for x in env.action_list])
val_array = np.zeros((20,20))
for s in state_reps:
    p,v = AC_head_agent(state_reps[s])
    coord = env.oneD2twoD(s)
    pol_array[coord] = tuple(p.detach().numpy())
    val_array[coord] = v.item()

plot_pref_pol(env, pol_array,save=True,directory='../figures/CH1/bits_and_bobs/',title=f'{env_name[-2:]}_{rep}_{pct}')
plot_valmap(env,val_array, v_range=[4,10],save=True, directory='../figures/CH1/bits_and_bobs/',title=f'{env_name[-2:]}_{rep}',cmap=plt.cm.viridis)

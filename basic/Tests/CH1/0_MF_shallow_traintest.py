import torch
import gym
import numpy as np
import pandas as pd
from modules.Agents.RepresentationLearning.learned_representations import convs, reward_convs
import modules.Agents.Networks as nets
from modules.Agents.Networks import flex_ActorCritic as Network
from modules.Agents.Networks import conv_PO_params, conv_FO_params
from modules.Agents import Agent
from modules.Experiments import shallow_expt as expt
import matplotlib.pyplot as plt
import argparse


# set up arguments to be passed in and their defaults
parser = argparse.ArgumentParser()
parser.add_argument('-v', type=int, default=1)
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

print(args)
## load in network weights
def load_network(env_version, rep_type, data_dir):
    env_name = f'gridworld:gridworld-v0{env_version}'
    read_from_file = 'conv_mf_pretraining400.csv'
    # build the environment
    env = gym.make(env_name)
    plt.close()
    print(env_name, env.rewards)

    # get parameters for network
    param_set = {'conv': conv_PO_params, 'rwd_conv': conv_FO_params}
    params = param_set[rep_type]
    network_parameters = params(env)

    # make a new network instance
    network = Network(network_parameters)

    # get id of saved weights
    df = pd.read_csv(data_dir+read_from_file)
    df_gb = df.groupby(['env_name','representation'])['save_id']

    df_rep = {'conv': 'conv', 'rwd_conv': 'reward_conv'}
    id_list = list(df_gb.get_group((env_name, df_rep[rep_type])))

    agent_id = id_list[np.random.choice(len(id_list))]
    state_dict = torch.load(data_dir+f'agents/{agent_id}.pt')
    network.load_state_dict(state_dict)

    print(env_name, rep_type, agent_id)
    return network, state_dict

def get_net_activity(env_name, rep_type, net, show_input=False):
    # build the environment
    env = gym.make(env_name)
    print('env rewards:', env.rewards)
    plt.close()

    rep_types = {'conv':convs, 'rwd_conv':reward_convs}

    state_reps, representation_name, input_dims, _= rep_types[rep_type](env)

    h0_reps = {}
    h1_reps = {}
    for key, value in state_reps.items():
        if show_input:
            if key == 0:
                fig,ax = plt.subplots(1,value.shape[1])
                for i in range(value.shape[1]):
                    ax[i].imshow(value[0,i,:,:],cmap='bone_r')
                plt.show()

        p,v = net(value)
        h0_reps[key] = net.test_activity.detach().numpy()
        h1_reps[key] = net.h_act.detach().numpy()

    return h0_reps, h1_reps

write_to_file = 'train_only_CONVshallowAC.csv'
directory = '../../Data/' # ../../Data if you are in Tests/CH2
env_name = f'gridworld:gridworld-v{version}'
# make gym environment
env = gym.make(env_name)
plt.close()
print(env.rewards)

num_trials = 5000
num_events = 250

net, state_dict = load_network(version,rep_type,directory)
# load weights to head_ac network from previously learned agent
AC_head_agent = nets.shallow_ActorCritic(input_dims=400, hidden_dims=400, output_dims=env.action_space.n, lr=learning_rate)
top_layer_dict = {}
top_layer_dict['hidden.weight'] = state_dict['hidden.5.weight']
top_layer_dict['hidden.bias'] = state_dict['hidden.5.bias']
top_layer_dict['pol.weight'] = state_dict['output.0.weight']
top_layer_dict['pol.bias'] = state_dict['output.0.bias']
top_layer_dict['val.weight'] = state_dict['output.1.weight']
top_layer_dict['val.bias'] = state_dict['output.1.bias']
#AC_head_agent.load_state_dict(top_layer_dict)

# get state inputs
h0, h1 = get_net_activity(env_name,rep_type,net)
state_reps, representation_name, = h0, f'h0_{rep_type}_latents'

memory = None #Memory.EpisodicMemory(cache_limit=cache_size_for_env, entry_size=env.action_space.n)

agent = Agent(AC_head_agent, memory=memory, state_representations=state_reps)


run = expt(agent, env)
run.run(NUM_TRIALS=num_trials, NUM_EVENTS=num_events)
run.record_log(env_name, representation_name,num_trials,num_events,dir=directory, file=write_to_file)

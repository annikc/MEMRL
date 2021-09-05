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
import pickle
from modules.Agents.Networks import flex_ActorCritic as Network
from modules.Agents.Networks import conv_PO_params, conv_FO_params
from modules.Agents.RepresentationLearning.learned_representations import convs, reward_convs, sr
from modules.Utils.gridworld_plotting import plot_polmap, plot_valmap, plot_pref_pol


## read in run IDs corresponding to rep types
read_from_file  = 'conv_mf_pretraining400.csv'
data_dir        = '../../Data/' # from within Tests/CH1

def save_obj(obj, name):
    with open('../../modules/Agents/RepresentationLearning/Learned_Rep_pickles/'+ name + '.p', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

## load in network weights
def load_network(env_name, rep_type):
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
    return network

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

def get_policy_value(env_name, rep_type, net):
    # build the environment
    env = gym.make(env_name)
    print('env rewards:', env.rewards)
    plt.close()

    rep_types = {'conv':convs, 'rwd_conv':reward_convs}

    state_reps, representation_name, input_dims, _= rep_types[rep_type](env)

    vals = np.zeros((20,20))
    vals[:] = np.nan
    pols = np.empty((20,20,4))
    for key, value in state_reps.items():
        coord = env.oneD2twoD(key)
        p,v = net(value)
        vals[coord] = v
        pols[coord,:] = tuple(p.detach().numpy()[0])
    return vals, pols

def save_latent_state_reps():
    for version in ['1','3','4','5']:
        for rep_type in ['conv','rwd_conv']:
            env_base  = 'gridworld:gridworld-v'
            pretrain  = env_base+'0'+version
            train_env = env_base+version
            test_env  = env_base+version+'1'

            net = load_network(pretrain,rep_type)
            fig,ax = plt.subplots(2,1,sharex=True)

            h0, h1 = get_net_activity(train_env,rep_type,net)
            save_obj(h0, f'{rep_type}/h0_v{version}')
            save_obj(h1, f'{rep_type}/h1_v{version}')

            h0, h1 = get_net_activity(test_env,rep_type,net)
            save_obj(h0, f'{rep_type}/h0_v{version}1')
            save_obj(h1, f'{rep_type}/h1_v{version}1')


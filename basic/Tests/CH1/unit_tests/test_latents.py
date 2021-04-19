import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import torch
import sys
sys.path.append('../../../modules')
from modules.Agents.Networks import flat_ActorCritic as head_AC
from modules.Agents.Networks import flex_ActorCritic as ac_net
from modules.Agents.RepresentationLearning.learned_representations import convs, reward_convs, onehot, place_cell, latents
from modules.Agents import Agent
from modules.Experiments import expt
from modules.Utils.gridworld_plotting import plot_polmap
from modules.Agents.Networks.load_network_weights import load_saved_head_weights
sys.path.append('../../../../')

save = False

conv_example_run_ids = {'gridworld:gridworld-v1':'c34544ac-45ed-492c-b2eb-4431b403a3a8',
                        'gridworld:gridworld-v2':'9a12edd8-a978-4e6b-a9f9-e09e0e35c534',
                        'gridworld:gridworld-v3':'32301262-cd74-4116-b776-57354831c484',
                        'gridworld:gridworld-v4':'b50926a2-0186-4bb9-81ec-77063cac6861',
                        'gridworld:gridworld-v5':'15b5e27b-444f-4fc8-bf25-5b7807df4c7f'} # example of each gridworld - v1, v2, v3, v4, v5 using conv representations (no rewards in input tensor)

rwd_conv_example_run_ids = {'gridworld:gridworld-v1':'990b45e3-49a6-49e0-8b85-e1dbbd865504',
                        'gridworld:gridworld-v2':'a465d78c-153e-4922-8011-2f13b5e93926',
                        'gridworld:gridworld-v3':'4ebe79ad-c5e6-417c-8823-a5fceb65b4e0',
                        'gridworld:gridworld-v4':'062f76a0-ce05-4cce-879e-2c3e7d00d543',
                        'gridworld:gridworld-v5':'fee85163-212e-4010-b90a-580e6671a454'} # example of each using conv_reward representation (rewards in input tensor)

ids = {'conv': conv_example_run_ids, 'rwd_conv':rwd_conv_example_run_ids}

latent_type = 'rwd_conv'
env_version = 4
# make environment to get input states
training_env_name = f'gridworld:gridworld-v{env_version}'
testing_env_name = training_env_name+'1'


train_env = gym.make(training_env_name)
plt.close()

test_env = gym.make(testing_env_name)
plt.close()

env= train_env

example_ids = ids[latent_type]
run_id = example_ids[training_env_name]
path_to_agent = f'./../../../Data/agents/{run_id}.pt'


class gridworldparam():
    def __init__(self,inp):
        self.input_dims = (inp,20,20)
        self.action_dims = 4
        self.hidden_types = ['conv', 'pool', 'conv', 'pool', 'linear', 'linear']
        self.hidden_dims = [None, None, None, None, 600, 400]
        self.lr = 5e-4

# get inputs states
if latent_type=='conv':
    state_reps, name, dim, ___ = convs(env)
elif latent_type == 'rwd_conv':
    state_reps, name, dim, ___ = reward_convs(env)


input_dim = state_reps[list(state_reps.keys())[0]].shape[1]

state_dict = torch.load(path_to_agent)
params = gridworldparam(input_dim)
network = ac_net(params)
network.load_state_dict(state_dict)

ls = {}
for index, inp in state_reps.items():
    # do a forward pass
    network(inp)
    # get hidden_layer activity
    ls[index] = network.h_act.detach().numpy()[0]





test_index = 100
tensor_slices = state_reps[test_index][0].shape[0]
fig,ax = plt.subplots(1,tensor_slices)
for item in range(tensor_slices):
    ax[item].imshow(state_reps[test_index][0][item], cmap='bone_r')
    ax[item].set_aspect('equal')


latent_reps1, _, __, ___ = latents(train_env,path_to_agent,type=latent_type)
latent_reps2, _, __, ___ = latents(test_env,path_to_agent,type=latent_type)

fig, ax = plt.subplots(1,3)
ax[0].imshow(np.asarray([latent_reps1[0]]).T,aspect='auto')
ax[1].imshow(np.asarray([latent_reps2[0]]).T,aspect='auto')
ax[2].imshow(np.asarray([latent_reps2[0]-latent_reps1[0]]).T, aspect='auto')
plt.show()

def plot_inputs_and_latents(env_version, latent_type, test_index):
    training_env_name = f'gridworld:gridworld-v{env_version}'
    testing_env_name = training_env_name+'1'
    env_name = testing_env_name
    env = gym.make(env_name)
    plt.close()

    # get inputs states
    if latent_type=='conv':
        inputs, _, __, ___ = convs(env)
    elif latent_type == 'rwd_conv':
        inputs, _, __, ___ = reward_convs(env)

    tensor_slices = inputs[test_index][0].shape[0]
    fig,ax = plt.subplots(1,tensor_slices + 1)
    for item in range(tensor_slices):
        ax[item].imshow(inputs[test_index][0][item], cmap='bone_r')
        ax[item].set_aspect('equal')
    plt.show()

    example_ids = ids[latent_type]
    run_id = example_ids[training_env_name]

    # get corresponding latent states
    path_to_agent = f'./../../../Data/agents/{run_id}.pt'

    empty = head_AC(400, 4, lr=0.005)
    full = load_saved_head_weights(empty,path_to_agent)

    state_reps, name, dim , _ = latents(env,path_to_agent,type=latent_type)

    policy_map = np.zeros(env.shape, dtype = [(x,'f8') for x in env.action_list])
    for state2d in env.useable:
        latent_state = state_reps[env.twoD2oneD(state2d)]
        pol, val = full(latent_state)
        policy_map[state2d] = tuple(pol)

    plot_polmap(env, policy_map)



def get_learned_latents(training_env_name, testing_env_name, example_run_id, obs_type):
    # make environments to get state observations
    training_env = gym.make(training_env_name)
    plt.close()
    testing_env = env.make(testing_env_name)
    plt.close()

    # get latents

def generate_saved_latents():
    for latent_type in ['conv','rwd_conv']:
        for version in [1,2,3,4,5]:
            # get agent ids to load/save/generate latents
            example_run_ids = ids[latent_type]

            # get environment
            env_id = f'gridworld:gridworld-v{version}'
            run_id = example_run_ids[env_id]

            # make sure saved agent is in the form of a state_dict of weights instead of the agent object
            try:
                convert_agent_to_weight_dict(f'../../Data/network_objs/{run_id}.pt',destination_path=f'./../../Data/agents/{run_id}.pt')
            except:
                pass

            # make gym environment
            env = gym.make(env_id)
            plt.close()

            # save latents by loading network, passing appropriate tensor, getting top fc layer activity
            reps, name, dim, _ = latents(env, f'./../../Data/agents/{run_id}.pt', type=latent_type )

            latent_array = np.zeros((env.nstates,env.nstates))
            for i in reps.keys():
                latent_array[i] = reps[i]

            if save:
                with open(f'../../modules/Agents/RepresentationLearning/Learned_Rep_pickles/{latent_type}{run_id[0:8]}_{env_id[-12:]}.p', 'wb') as f:
                    pickle.dump(file=f, obj=latent_array.copy())

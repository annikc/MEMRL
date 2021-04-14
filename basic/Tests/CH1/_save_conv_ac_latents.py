import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import torch
import sys
sys.path.append('../../modules')
from modules.Agents.Networks import flat_ActorCritic as head_AC
from modules.Agents.RepresentationLearning.learned_representations import convs, onehot, place_cell, latents
from modules.Agents import Agent
from modules.Experiments import expt
from modules.Agents.Networks.load_network_weights import convert_agent_to_weight_dict
sys.path.append('../../../')

latent_type = 'rwd_conv'
save = True
version = 5
for version in [1,2,3,4,5]:
    env_id = f'gridworld:gridworld-v{version}'
    conv_example_run_ids = ['c34544ac-45ed-492c-b2eb-4431b403a3a8',
                       '9a12edd8-a978-4e6b-a9f9-e09e0e35c534',
                       '32301262-cd74-4116-b776-57354831c484',
                       'b50926a2-0186-4bb9-81ec-77063cac6861',
                       '15b5e27b-444f-4fc8-bf25-5b7807df4c7f'] # example of each gridworld - v1, v2, v3, v4, v5 using conv representations (no rewards in input tensor)

    rwd_conv_example_run_ids = ['990b45e3-49a6-49e0-8b85-e1dbbd865504',
                       'a465d78c-153e-4922-8011-2f13b5e93926',
                       '4ebe79ad-c5e6-417c-8823-a5fceb65b4e0',
                       '062f76a0-ce05-4cce-879e-2c3e7d00d543',
                       'fee85163-212e-4010-b90a-580e6671a454'] # example of each using conv_reward representation (rewards in input tensor)
    ids = {'conv': conv_example_run_ids, 'rwd_conv':rwd_conv_example_run_ids}
    example_run_ids = ids[latent_type]

    run_id = example_run_ids[version-1]
    try:
        convert_agent_to_weight_dict(f'../../Data/network_objs/{run_id}.pt',destination_path=f'./../../Data/agents/{run_id}.pt')
    except:
        pass

    # make gym environment
    env = gym.make(env_id)
    plt.close()

    reps, name, dim, _ = latents(env, f'./../../Data/agents/{run_id}.pt', type=latent_type )
    print(reps[0][0:20])

    latent_array = np.zeros((env.nstates,env.nstates))
    for i in reps.keys():
        latent_array[i] = reps[i]

    if save:
        with open(f'../../modules/Agents/RepresentationLearning/Learned_Rep_pickles/conv_{env_id[-12:]}.p', 'wb') as f:
            pickle.dump(file=f, obj=latent_array.copy())

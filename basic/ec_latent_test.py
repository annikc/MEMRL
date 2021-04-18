import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import torch
import sys
sys.path.append('../../modules')
from modules.Agents.Networks import flat_ActorCritic as head_AC
from modules.Agents.Networks import flex_ActorCritic as Network
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents.Networks.load_network_weights import load_saved_head_weights, convert_agent_to_weight_dict
from modules.Agents.RepresentationLearning.learned_representations import latents
from modules.Agents import Agent
from modules.Experiments import flat_expt
sys.path.append('../../../')


write_to_file = 'ec_latent_test.csv'
version = 1
latent_type = 'conv'
training_env_name = f'gridworld:gridworld-v{version}'
test_env_name = training_env_name+'1'

relative_path_to_data = './Data/' # from within Tests/CH1

num_trials = 25000
num_events = 250

rwd_conv_ids = {'gridworld:gridworld-v1':'990b45e3-49a6-49e0-8b85-e1dbbd865504',
           'gridworld:gridworld-v3':'4ebe79ad-c5e6-417c-8823-a5fceb65b4e0',
           'gridworld:gridworld-v4':'062f76a0-ce05-4cce-879e-2c3e7d00d543',
           'gridworld:gridworld-v5':'fee85163-212e-4010-b90a-580e6671a454'}

conv_ids = {'gridworld:gridworld-v1':'c34544ac-45ed-492c-b2eb-4431b403a3a8',
           'gridworld:gridworld-v3':'32301262-cd74-4116-b776-57354831c484',
           'gridworld:gridworld-v4':'b50926a2-0186-4bb9-81ec-77063cac6861',
           'gridworld:gridworld-v5':'15b5e27b-444f-4fc8-bf25-5b7807df4c7f'}
ids = {'conv':conv_ids, 'rwd_conv':rwd_conv_ids}

id_dict = ids[latent_type]
run_id = id_dict[training_env_name]

# make gym environment to load states for getting latent representations
train_env = gym.make(training_env_name)
plt.close()
# make new env to run test in
test_env = gym.make(test_env_name)
plt.close()

# load latent states to use as state representations to actor-critic heads
agent_path = relative_path_to_data+f'agents/{run_id}.pt'

# save latents by loading network, passing appropriate tensor, getting top fc layer activity
state_reps, representation_name, input_dims, _ = latents(train_env, agent_path, type=latent_type)

# load weights to head_ac network from previously learned agent
empty_net = head_AC(input_dims, test_env.action_space.n, lr=0.0005)
AC_head_agent = load_saved_head_weights(empty_net, agent_path)

memory = Memory(entry_size=test_env.action_space.n, cache_limit=400)

agent = Agent(AC_head_agent, memory=memory, state_representations=state_reps)

ex = flat_expt(agent, test_env)
ex.run(num_trials,num_events,snapshot_logging=False)
ex.record_log(env_name=test_env_name, representation_type=representation_name,
              n_trials=num_trials, n_steps=num_events,
              dir=relative_path_to_data, file=write_to_file)

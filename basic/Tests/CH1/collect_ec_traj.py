import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import torch
import sys
import time
sys.path.append('../../modules')
from matplotlib.collections import LineCollection
from modules.Agents.Networks import flat_ActorCritic as head_AC
from modules.Agents.Networks import flex_ActorCritic as Network
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents.Networks.load_network_weights import load_saved_head_weights, convert_agent_to_weight_dict
from modules.Agents.RepresentationLearning.learned_representations import latents, sr, onehot
from modules.Agents import Agent
from modules.Utils.gridworld_plotting import plot_pref_pol
sys.path.append('../../../')
import argparse

# set up arguments to be passed in and their defauls
parser = argparse.ArgumentParser()
parser.add_argument('-v', type=int, default=4)
parser.add_argument('-rep', default='sr')
parser.add_argument('-load', type=bool, default=False)
parser.add_argument('-lr', default=0.0005)
parser.add_argument('-cache', type=int, default=100)
args = parser.parse_args()

# parameters set with command line arugments
version       = args.v
latent_type   = args.rep
load_weights  = args.load # load learned weights from conv net or use new init head weights
learning_rate = args.lr
cache_size    = args.cache

class expt(object):
	def __init__(self, agent, environment, **kwargs):
		self.env = environment
		self.agent = agent
		# self.rep_learner = rep_learner  #TODO add in later
		self.data = self.reset_data_logs()
		self.agent.counter = 0
		self.pol_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])
		self.val_grid = np.empty(self.env.shape)

		self.first_rwd = False
		self.trajectories = []
		# temp

	def reset_data_logs(self):
		data_log = {'total_reward': [],
					'loss': [[], []],
					'trial_length': [],
					'EC_snap': [],
					'P_snap': [],
					'V_snap': [],
					'occupancy':np.zeros(self.env.nstates)
					}
		return data_log

	def snapshot(self, states, observations):
		# initialize empty data frames
		pol_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])
		val_grid = np.empty(self.env.shape)

		mem_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])

		# forward pass through network
		pols, vals = self.agent.MFC(observations)

		# populate with data from network
		for s, p, v in zip(states, pols, vals):
			pol_grid[s] = tuple(p.data.numpy())
			val_grid[s] = v.item()

		return pol_grid, val_grid

	def end_of_trial(self, trial, logsnap=False):
		p, v = self.agent.finish_()

		# temp
		if logsnap:
			if trial % self.print_freq==0:
				states = [self.env.oneD2twoD(x) for x in list(self.agent.state_reps.keys())]
				observations = list(self.agent.state_reps.values())
				MF_pols, MF_vals = self.snapshot(states,observations)
				self.data['V_snap'].append(MF_vals)
				self.data['P_snap'].append(MF_pols)
		# /temp

		self.data['total_reward'].append(self.reward_sum) # removed for bootstrap expts
		self.data['loss'][0].append(p)
		self.data['loss'][1].append(v)

		if trial <= 10:
			self.running_rwdavg = np.mean(self.data['total_reward'])
		else:
			self.running_rwdavg = np.mean(self.data['total_reward'][-self.print_freq:])

		if trial % self.print_freq == 0:
			print(f"Episode: {trial}, Score: {self.reward_sum} (Running Avg:{self.running_rwdavg}) [{time.time() - self.t}s]")
			self.t = time.time()

	def single_step(self,trial):
		# get representation for given state of env. TODO: should be in agent to get representation?
		state_representation = self.agent.get_state_representation(self.state)
		readable = self.state
		self.data['occupancy'][self.state]+=1

		# get action from agent
		action, log_prob, expected_value = self.agent.get_action(state_representation)
		# take step in environment
		next_state, reward, done, info = self.env.step(action)

		# end of event
		target_value = 0
		self.reward_sum += reward

		self.agent.log_event(episode=trial, event=self.agent.counter,
							 state=state_representation, action=action, reward=reward, next_state=next_state,
							 log_prob=log_prob, expected_value=expected_value, target_value=target_value,
							 done=done, readable_state=readable)
		self.agent.counter += 1
		self.state = next_state
		self.trajectories[trial].append(self.env.oneD2twoD(self.state))
		return done

	def run(self, NUM_TRIALS, NUM_EVENTS, **kwargs):
		self.print_freq = kwargs.get('printfreq', 100)
		self.reset_data_logs()
		self.t = time.time()
		logsnap = kwargs.get('snapshot_logging', False)

		for trial in range(NUM_TRIALS):
			self.trajectories.append([])
			self.state = self.env.reset()
			self.reward_sum = 0
			self.trajectories[trial].append(self.env.oneD2twoD(self.state))
			for event in range(NUM_EVENTS):
				done = self.single_step(trial)

				if done:
					break
			if self.reward_sum >0:
				self.first_rwd = True

			self.end_of_trial(trial,logsnap=logsnap)
			if self.first_rwd:
				break

def get_mem_maps(env, cache_list,full_mem=True):
    blank_mem = Memory(cache_limit=400, entry_size=4)
    blank_mem.cache_list = cache_list

    ec_pol_grid = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])
    if full_mem:
        for key, value in state_reps.items():
            twoD = env.oneD2twoD(key)
            sr_rep = value
            pol = blank_mem.recall_mem(sr_rep)

            ec_pol_grid[twoD] = tuple(pol)
    else:
        for ec_key in blank_mem.cache_list.keys():
            twoD = env.oneD2twoD(blank_mem.cache_list[ec_key][2])
            pol  = blank_mem.recall_mem(ec_key)

            ec_pol_grid[twoD] = tuple(pol)

    return ec_pol_grid
# parameters set for this file
relative_path_to_data = '../../Data/' # from within Tests/CH1
write_to_file         = 'ec_latent_test.csv'
training_env_name     = f'gridworld:gridworld-v{version}'
test_env_name         = training_env_name+'1'
num_trials = 5000
num_events = 250

rwd_conv_ids = {'gridworld:gridworld-v1':'990b45e3-49a6-49e0-8b85-e1dbbd865504',
                'gridworld:gridworld-v3':'4ebe79ad-c5e6-417c-8823-a5fceb65b4e0',
                'gridworld:gridworld-v4':'062f76a0-ce05-4cce-879e-2c3e7d00d543',
                'gridworld:gridworld-v5':'fee85163-212e-4010-b90a-580e6671a454'}
conv_ids     = {'gridworld:gridworld-v1':'c34544ac-45ed-492c-b2eb-4431b403a3a8',
                'gridworld:gridworld-v3':'32301262-cd74-4116-b776-57354831c484',
                'gridworld:gridworld-v4':'b50926a2-0186-4bb9-81ec-77063cac6861',
                'gridworld:gridworld-v5':'15b5e27b-444f-4fc8-bf25-5b7807df4c7f'}


# make gym environment to load states for getting latent representations
train_env = gym.make(training_env_name)
plt.close()
# make new env to run test in
test_env = gym.make(test_env_name)
plt.close()

if latent_type == 'conv' or latent_type == 'rwd_conv':
	ids = {'conv':conv_ids, 'rwd_conv':rwd_conv_ids}

	id_dict = ids[latent_type]
	run_id = id_dict[training_env_name]
	# load latent states to use as state representations to actor-critic heads
	agent_path = relative_path_to_data+f'agents/{run_id}.pt'

	# save latents by loading network, passing appropriate tensor, getting top fc layer activity
	state_reps, representation_name, input_dims, _ = latents(train_env, agent_path, type=latent_type)

elif latent_type in ['sr', 'onehot']:
	rep_Type = {'sr':sr, 'onehot':onehot}
	state_reps, representation_name, input_dims, _ = rep_Type[latent_type](test_env)

if load_weights:
    # load weights to head_ac network from previously learned agent
    empty_net = head_AC(input_dims, test_env.action_space.n, lr=learning_rate)
    AC_head_agent = load_saved_head_weights(empty_net, agent_path)
    loaded_from = run_id
else:
    AC_head_agent = head_AC(input_dims, test_env.action_space.n, lr=learning_rate)
    loaded_from = ' '
cache_limits = {'gridworld:gridworld-v11':{100:400, 75:300, 50:200, 25:100},
                'gridworld:gridworld-v31':{100:365, 75:273, 50:182, 25:91},
                'gridworld:gridworld-v41':{100:384, 75:288, 50:192, 25:96},
                'gridworld:gridworld-v51':{100:286, 75:214, 50:143, 25:71}}
cache_size_for_env = cache_limits[test_env_name][cache_size]
print(cache_size_for_env)
memory = Memory(entry_size=test_env.action_space.n, cache_limit=cache_size_for_env)

agent = Agent(AC_head_agent, memory=memory, state_representations=state_reps)

ex = expt(agent, test_env)
ex.run(100,250,printfreq=1)
for i in ex.trajectories:
	print(i)

fig, ax = plt.subplots(1,1)
ax.pcolor(test_env.grid,cmap='bone_r',edgecolors='k', linewidths=0.1)
ax.set_aspect('equal')
ax.invert_yaxis()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax.add_patch(plt.Rectangle(list(test_env.rewards.keys())[0], 1,1, facecolor='g', alpha=0.3))
st_r, st_c = ex.trajectories[-1][0]
ax.add_patch(plt.Circle((st_c+0.5,st_r+0.5),radius=0.4, color='r'))
for i in ex.trajectories[-1:]:
	segs = []
	for ind in range(len(i)-1):
		pos1 = (i[ind][1]+0.5, i[ind][0]+0.5)
		pos2 = (i[ind+1][1]+0.5, i[ind+1][0]+0.5)
		segs.append((pos1,pos2))
	lines = LineCollection(segs,linestyle=':', color='r')
	ax.add_collection(lines)
plt.savefig('../../Analysis/figures/CH1/example_ec_trajectory.svg')
plt.show()

CL = memory.cache_list
ec_pol_grid = get_mem_maps(test_env,CL)
plot_pref_pol(test_env,ec_pol_grid)
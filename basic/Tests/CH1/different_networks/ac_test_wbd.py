# import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import sys
import torch
sys.path.insert(0, '../../../modules/')
from modules.Utils import running_mean as rm

# import actor critic network
from modules.Agents.Networks import ActorCritic as Network

# import agent class wrapper to handle behaviour
from modules.Agents import Agent

# import representation type
from modules.Agents.RepresentationLearning import PlaceCells

# import experiment class to handle run and logging
from modules.Experiments import expt

# get environment
import gym

def one_hot_state(state):
    vec = np.zeros(env.nstates)
    vec[state] = 1
    return vec

def onehot_state_collection(env):
    collection = []
    for state in range(env.nstates):
        vec = one_hot_state(state)
        collection.append(vec)
    return collection

def twoD_states(env):
    twods = []
    for state in range(env.nstates):
        twod = env.oneD2twoD(state)
        twods.append(twod)
    return twods


# make env
env_name = 'gym_grid:gridworld-v4'
env = gym.make(env_name)
#env.set_reward({(5,5):2})
plt.close()

# 1. Tabular Q
# 2. SARSA
# 3. DQN
# 4. ActorCritic
input_dims = 400

### place cell representations
place_cells = PlaceCells(env.shape, input_dims, field_size=0.25)

plt.figure()
plt.scatter(place_cells.cell_centres[:,0], place_cells.cell_centres[:,1])
plt.show()


pc_state_reps = {}
oh_state_reps = {}
for state in env.useable:
	oh_state_reps[env.twoD2oneD(state)] = one_hot_state(env.twoD2oneD(state))
	pc_state_reps[env.twoD2oneD(state)] = place_cells.get_activities([state])[0]


oh_network = Network(input_dims=[input_dims],fc1_dims=200,fc2_dims=200,output_dims=env.action_space.n, lr=0.0005)

oh_network = torch.load('./Data/agents/{load_id}.pt')
oh_agent = Agent(oh_network, state_representations=oh_state_reps)

pc_network = Network(input_dims=[input_dims],fc1_dims=200,fc2_dims=200,output_dims=env.action_space.n, lr=0.0005)
pc_agent = Agent(pc_network, state_representations=pc_state_reps)

ex = expt(oh_agent, env)

ntrials = 5000
nsteps = 250

ex.run(ntrials, nsteps)
ex.record_log('oh_retraining',env_name,ntrials,nsteps, file='ac_representation.csv')






### JUNKYARD
'''



def run(env, agent, NUM_TRIALS, NUM_EVENTS, data, **kwargs):
	t = time.time()
	for trial in range(NUM_TRIALS):
		state = env.reset()
		reward_sum = 0
		for event in range(NUM_EVENTS):
			# get representation for given state of env. TODO: should be in agent to get representation?
			state_representation = agent.get_state_representation(state)
			readable = 0

			# get action from agent
			action, log_prob, expected_value = agent.get_action(state_representation)
			# take step in environment
			next_state, reward, done, info = env.step(action)

			# end of event
			target_value = 0
			reward_sum += reward

			agent.log_event(episode=trial, event=agent.counter,
								 state=state_representation, action=action, reward=reward, next_state=next_state,
								 log_prob=log_prob, expected_value=expected_value, target_value=target_value,
								 done=done, readable_state=readable)
			agent.counter += 1
			state = next_state
			if done:
				break

		p, v = agent.finish_()

		data['total_reward'].append(reward_sum) # removed for bootstrap expts
		data['loss'][0].append(p)
		data['loss'][1].append(v)

		if trial <= 10:
			running_rwdavg = np.mean(data['total_reward'])
		else:
			running_rwdavg = np.mean(data['total_reward'][-10:-1])

		if trial % 100 == 0:
			print(f"Episode: {trial}, Score: {reward_sum} (Running Avg:{running_rwdavg}) [{time.time() - t}s]")
			t = time.time()



run(env, oh_agent,2500,250, oh_data_log)
run(env, pc_agent,2500,250, pc_data_log)

env.set_reward({(15,15):10})

run(env, oh_agent,7500,250, oh_data_log)

run(env, pc_agent,2500,250, pc_data_log)


oh_reward = oh_data_log['total_reward']
pc_reward = pc_data_log['total_reward']
smoothing = 100
plt.figure()
plt.plot(rm(oh_reward,smoothing))
plt.plot(rm(pc_reward,smoothing))
plt.show()






+++++++++++++++++++++++++++++
# get onehot activity vectors
gridworld_onehots = onehot_state_collection(env)

# get place cell activity vectors
# make a collection of place cells
place_cells       = PlaceCells(env.shape,env.nstates, field_size=1/(env.shape[0]-1))


two_d_states = twoD_states(env) # get all states as coords
plot = place_cells.plot_placefields(two_d_states,8)

# get activities for each state
gridworld_pc_reps = place_cells.get_activities(two_d_states)

def get_representation(state, onehot=True):
    if onehot:
        #vec = np.zeros(env.nstates)
        #vec[state] =1
        vec = gridworld_onehots[state]
        return vec
    else:
        #twodstate = env.oneD2twoD(state)
        #vec = place_cells.get_activities([twodstate])
        vec = gridworld_pc_reps[state]
        return vec

'''
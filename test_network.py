### Outline for Behavioural Experiment MDP 
import numpy as np 
import matplotlib.pyplot as plt 
import pdb
import random
from MDP_def import *

print_statements = False
''' 
----------------------
		NOTES 
----------------------
* When, if at all, do we need to reset gradients? (reset_grads function in Network class)
* Need to comment code


----------------------
		TO DO 
----------------------
* Initial pass -- start with getting some agent who selects actions properly -- write functions to plot current state/environment 
	- animated plots 
	- better way of generating grid of environment?
		- look into using TKinter 

* Have agent select actions based on reward maximization

* Write up learning for top layer (policy selection, value)

* Create new plotting functions with pycairo


! Make sure at input layer each state is represented as separate thing
		! Check again for h_0, h_1 -- determine no bugs by passing dummy input vectors 
! Make fork in code: if you get rid of the hidden layers and project state vector to action/policy units 
! Confirm state representations are correct
! try shallow network
! check hidden representations are sensible (i.e. states that are more similar to each other should be closer to each other in hidden layers)

'''



#------------------------------------------------------
""" 					Set up						"""
#------------------------------------------------------
# Create environment with default structure
T_maze = Environment()

# Create trial set with specified number of trials
trialset = Trial_Set(T_maze, total_trials=300, num_events=500)

# Create network for valicy and policy calculations 
network = Network(num_units = [17, 6], learning_rate=[0.0001,0.0001,0.0001,0.0001])

# 
actions_list = trialset.working_env.all_actions # ['N','E','S','W','stay','poke']

#set up lists for storing variables to track
store_return = []
store_reward = []
store_trialnum = []
store_ev = []
reward_test = 0

home_store = [[],[],[],[],[],[]]
E1_store = [[],[],[],[],[],[]]
E2_store = [[],[],[],[],[],[]]
W1_store = [[],[],[],[],[],[]]
W2_store = [[],[],[],[],[],[]]
S1_store = [[],[],[],[],[],[]]
S2_store = [[],[],[],[],[],[]]

choice_data = {'HOME':home_store, 'E1':E1_store, 'E2':E2_store, 'W1':W1_store, 'W2':W2_store, 'S1':S1_store, 'S2':S2_store}

#------------------------------------------------------
""" 				   Run Trial					"""
#------------------------------------------------------
for i in range(trialset.tot_trials):
	if print_statements:
		print " ---------- \n Trial Number:", trialset.num_trials, "\n ----------"
	else:
		pass
	# initial agent starting state to HOME
	T_maze.cur_state = State(T_maze.all_loc[0], T_maze.tex_loc[T_maze.all_loc[0]]) 
	# Add trials to the set
	trialset.add_trial()
	ev_total = 0
	reward_marker = 0
	value_marker = 0 
	# Run through the number of events specified per trial
	for j in range(trialset.current_trial.num_events):
		if print_statements:
			print "-----  \nEVENT #", str(i)+'-'+str(j+1), "\n The current state is ", trialset.current_trial.events[j][1].loc
		else:
			pass
		
		# feed network 
		NN_input = make_state_vector(trialset.current_trial.events[j][1].loc, trialset.current_trial.events[j][1].tex, trialset.current_trial.events[j][2])

		#____ FORWARD PASS THROUGH NETWORK ______
		network.forward_pass(NN_input)
		
		
		rv = (T_maze.rtrn - network.value[0][0])
		value_tracker = network.value[0][0]
		ev_total += 0.5*(rv**2)
		'''
		EDIT CHOICE SELECTION -- NP.NONZERO
		'''
		action_choice = np.random.choice(T_maze.all_actions, 1, p=network.action[:,0])[0]
		if print_statements:
			print "agent chooses", action_choice
		else:
			pass

		# Add events to the trial corresponding to action selections
		trialset.current_trial.add_event(action_choice)

		reward_test += T_maze.rwd
		reward_marker += T_maze.rwd
		if print_statements:
			print "cumulative reward is ", reward_test
		else:
			pass

		network.accumulate_gradients(rv)

	network.update_weights()


	rtrn = T_maze.rtrn
	if print_statements:
			print("Total return for trial # ", i, ", was: ", rtrn)
	else:
		pass
	
	store_return.append(rtrn)
	store_reward.append(reward_marker)
	store_trialnum.append(trialset.num_trials)
	store_ev.append(ev_total)

	test_options(choice_data, T_maze, network)

	network.reset_gradients()		

#plot_test_options(choice_data, trialset)

### Plot error tracking
plt.plot(np.arange(trialset.tot_trials), store_ev/max(store_ev))
#plt.plot(np.arange(trialset.tot_trials), store_reward, 'ro')
plt.plot(np.arange(trialset.tot_trials), store_return/max(store_return), 'g')

plt.show()	


#### plot reward and return over trials
#fig, a1 = plt.subplots(figsize=(12,8))
#return_plot = a1.plot(store_trialnum, store_return)
#reward_plot = a1.plot(store_trialnum, store_reward, 'r')
#a1.set_xlabel('Trial Number')
#a1.set_ylabel('Return')
#a1.set_title('Forward Pass Generated Policy Units')


#fig, a3 = plt.subplots(figsize=(12,8))
#return_plot = a3.plot(store_trialnum, store_ev)
#a3.set_xlabel('Trial Number')
#a3.set_ylabel('Error')
#a3.set_title('Value error for each trial')






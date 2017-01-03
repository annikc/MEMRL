### Outline for Behavioural Experiment MDP 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker 
import pdb
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
from MDP_def import *

print_to_file = False

if print_to_file: 
	import sys
	orig_stdout = sys.stdout
	fil = file('out.txt', 'w')
	sys.stdout = fil
else:
	pass



print_statements = False

plot_value_grads = True
plot_action_grads = True
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
trialset = Trial_Set(T_maze, total_trials=1, num_events=150)

# Create network for valicy and policy calculations 
network = Network(num_units = [17, 6], learning_rate=[0.0001,0.0001,0.0001,0.0001])

# 
actions_list = ['poke', 'W', 'W', 'W', 'W', 'W']#trialset.working_env.all_actions # ['N','E','S','W','stay','poke']

#set up lists for storing variables to track
store_return = []
store_reward = []
store_ev     = []
store_Wv     = []
store_Wa	 = [[], [], [], [], [], []]
store_loc 	 = []
store_act 	 = []

ev_total = 0
reward_marker = 0
value_tracker = 0 

store_trialnum = []

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
		print(" ---------- \n Trial Number:", trialset.num_trials, "\n ----------")

	# initial agent starting state to HOME
	T_maze.cur_state = State(T_maze.all_loc[0], T_maze.tex_loc[T_maze.all_loc[0]]) 
	# Add trials to the set
	trialset.add_trial()

	# Run through the number of events specified per trial
	for j in range(trialset.current_trial.num_events):
		if print_statements:
			print("-----  \nEVENT #", str(i)+'-'+str(j+1), "\n The current state is ", trialset.current_trial.events[j][1].loc)
		
		# feed network 
		NN_input = make_state_vector(trialset.current_trial.events[j][1].loc, trialset.current_trial.events[j][1].tex, trialset.current_trial.events[j][2])

		#____ FORWARD PASS THROUGH NETWORK ______
		network.forward_pass(NN_input)
		
		# value estimation
		value_tracker = network.value[0][0]

		# prediction error
		rv = (T_maze.rtrn - network.value[0][0])
		ev_total = 0.5*(rv**2)


		action_choice = np.random.choice(T_maze.all_actions, 1, p=network.action[:,0])[0]
		if print_statements:
			print("agent chooses", action_choice)
			
		# Add events to the trial corresponding to action selections
		trialset.current_trial.add_event(action_choice)

		if print_statements:
			print("cumulative reward is ", reward_test)

		network.accumulate_gradients(rv)

		if i == trialset.tot_trials-1:
			# gradients 
			wv_tracker = network.output_layer.dWv
			wa_tracker = network.output_layer.dWa
			#print wa_tracker.shape[0]

			#add to apropriate lists
			store_return.append(T_maze.rtrn)
			store_reward.append(T_maze.rwd)
			store_ev.append(ev_total)
			store_Wv.append(wv_tracker)
			for item in range(wa_tracker.shape[0]):
				store_Wa[item].append(wa_tracker[item,:])
			store_loc.append(make_state_vector(trialset.current_trial.events[j][1].loc, tex_var='null', last_act='stay')[0:7])
			store_act.append(list(network.action.T[0]))
		
	rtrn = T_maze.rtrn
	print("Total return for trial # ", i, ", was: ", rtrn)
	network.update_weights()

rtrn = T_maze.rtrn
if print_statements:
		print("Total return for trial # ", i, ", was: ", rtrn)

loc_array = np.squeeze(np.asarray(store_loc))
act_array = np.squeeze(np.asarray(store_act))

# Plot what? 
reward_bars = False



if plot_value_grads == True:

	## Process data for plotting
	# process value weights arrays
	wv_diff = []
	for i in range(len(store_Wv)):
		if i == 0: 
			wv_diff.append(store_Wv[0])
		else: 
			wv_diff.append(store_Wv[i] - store_Wv[i-1])

	wv_array = np.squeeze(np.asarray(store_Wv))
	wv_array1 = np.squeeze(np.asarray(wv_diff))
	wv_loc = wv_array1[:,0:7]
	wv_tex = wv_array1[:,7:11]
	wv_act = wv_array1[:,11:17]

	# Make Value Updates Plot
	fig = plt.figure(1, figsize=(15,12))

	# Add subplots
	num_subplots = 6
	subplot_counter = np.arange(num_subplots)
	splots = {}
	for i in range(num_subplots):
		temp = "ax"+str(i+1)
		splots[temp] = fig.add_subplot(num_subplots,1,i+1)


	# generate colorbars for imshow plots
	caxes = []
	caxes.append(fig.add_axes([0.91, 0.65, 0.01, 0.11]))
	caxes.append(fig.add_axes([0.91, 0.51, 0.01, 0.11]))
	caxes.append(fig.add_axes([0.91, 0.375, 0.01, 0.11]))
	caxes.append(fig.add_axes([0.91, 0.25, 0.005, 0.09]))
	caxes.append(fig.add_axes([0.91, 0.11, 0.005, 0.09]))

	# Subplot 1: Show return, prediction error, reward events 
	splots['ax1'].plot(np.arange(trialset.current_trial.num_events), store_ev, 'r')
	splots['ax1'].plot(np.arange(trialset.current_trial.num_events), store_return, 'g')
	if reward_bars: 
		for i in range(trialset.current_trial.num_events):
			if store_reward[i] == 1: 
				splots['ax1'].axvline(x=i)
	else: 
		splots['ax1'].plot(np.arange(trialset.current_trial.num_events), store_reward, 'bo')
	splots['ax1'].set_xlim(xmin=0, xmax=max(np.arange(trialset.current_trial.num_events)))
	splots['ax1'].set_title('Value Weight Gradients')

	# show difference in accumulated gradients from last time step
	im2 = splots['ax2'].imshow(wv_loc.T, cmap='Greens_r', interpolation='none', aspect='auto')
	splots['ax2'].locator_params(axis='y',nbins=10)
	splots['ax2'].set_yticklabels(['DUMMY LABEL','HOME', 'W1', 'W2', 'E1', 'E2', 'S1', 'S2'])
	splots['ax2'].yaxis.grid()
	plt.colorbar(im2, cax = caxes[0])

	im3 = splots['ax3'].imshow(wv_tex.T, cmap='Greens_r', interpolation='none', aspect='auto')
	splots['ax3'].locator_params(axis='y',nbins=6)
	splots['ax3'].set_yticklabels(['DUMMY LABEL','null', 'A', 'B', 'C'])
	splots['ax3'].yaxis.grid()
	plt.colorbar(im3, cax = caxes[1])

	im4 = splots['ax4'].imshow(wv_act.T, cmap='Greens_r', interpolation='none', aspect='auto')
	splots['ax4'].locator_params(axis='y',nbins=10)
	splots['ax4'].set_yticklabels(['DUMMY LABEL','N', 'E', 'S', 'W', 'STAY', 'POKE'])
	splots['ax4'].yaxis.grid()
	plt.colorbar(im4, cax = caxes[2])

	# show action selection
	im5 = splots['ax5'].imshow(loc_array.T, cmap='bone_r', interpolation='none', aspect='auto')#aspect=0.9, extent=[0,5000,0,555])
	splots['ax5'].locator_params(axis='y',nbins=10)
	splots['ax5'].set_yticklabels(['DUMMY LABEL','HOME', 'W1', 'W2', 'E1', 'E2', 'S1', 'S2'])
	splots['ax5'].tick_params(axis='y')
	plt.colorbar(im5, cax = caxes[3], ticks=[0, 1])

	# show action selection
	im6 = splots['ax6'].imshow(act_array.T, cmap='bone_r', interpolation='none', aspect='auto')#0.8, extent=[0,5000,0,555])
	splots['ax6'].locator_params(axis='y',nbins=10)
	splots['ax6'].set_yticklabels(['DUMMY LABEL','N', 'E', 'S', 'W', 'STAY', 'POKE'])
	splots['ax6'].tick_params(axis='y')
	plt.colorbar(im6, cax = caxes[4], ticks=[0, 1])

	plt.savefig('./plots/value_fig.png')

if plot_action_grads == True:
	weight_dim_list = ['North', 'East', 'West', 'South', 'Stay', 'Poke']
	for item in range(len(store_Wa)):
		# make correct variables: 
		wa_array = np.asarray(store_Wa[item]).T
		## Process data for plotting
		# process value weights arrays
		wa_diff = []
		for i in range(len(store_Wa[item])):
			if i == 0: 
				wa_diff.append(store_Wa[item][0])
			else: 
				wa_diff.append(store_Wa[item][i] - store_Wa[item][i-1])

		wa_array1 = (np.asarray(wa_diff))

		wa_loc = wa_array1[:,0:7]
		wa_tex = wa_array1[:,7:11]
		wa_act = wa_array1[:,11:17]
		# Make Value Updates Plot
		fig = plt.figure(item+2, figsize=(15,12))

		# Add subplots
		num_subplots = 6
		subplot_counter = np.arange(num_subplots)
		splots = {}
		for i in range(num_subplots):
			temp = "ax"+str(i+1)
			splots[temp] = fig.add_subplot(num_subplots,1,i+1)


		# generate colorbars for imshow plots
		caxes = []

		caxes.append(fig.add_axes([0.91, 0.65, 0.01, 0.11]))
		caxes.append(fig.add_axes([0.91, 0.51, 0.01, 0.11]))
		caxes.append(fig.add_axes([0.91, 0.375, 0.01, 0.11]))
		caxes.append(fig.add_axes([0.91, 0.25, 0.005, 0.09]))
		caxes.append(fig.add_axes([0.91, 0.11, 0.005, 0.09]))

		# Subplot 1: Show return, prediction error, reward events 
		splots['ax1'].plot(np.arange(trialset.current_trial.num_events), store_ev, 'r')
		splots['ax1'].plot(np.arange(trialset.current_trial.num_events), store_return, 'g')
		if reward_bars: 
			for i in range(trialset.current_trial.num_events):
				if store_reward[i] == 1: 
					splots['ax1'].axvline(x=i)
		else: 
			splots['ax1'].plot(np.arange(trialset.current_trial.num_events), store_reward, 'bo')
		splots['ax1'].set_xlim(xmin=0, xmax=max(np.arange(trialset.current_trial.num_events)))
		splots['ax1'].set_title('Action Gradient Derivatives for Unit: {}'.format(weight_dim_list[item]))

		# show difference in accumulated gradients from last time step
		im2 = splots['ax2'].imshow(wa_loc.T, cmap='Blues', interpolation='none', aspect='auto')
		splots['ax2'].locator_params(axis='y',nbins=10)
		splots['ax2'].set_yticklabels(['DUMMY LABEL','HOME', 'W1', 'W2', 'E1', 'E2', 'S1', 'S2'])
		splots['ax2'].yaxis.grid()
		plt.colorbar(im2, cax = caxes[0])

		im3 = splots['ax3'].imshow(wa_tex.T, cmap='Blues', interpolation='none', aspect='auto')
		splots['ax3'].locator_params(axis='y',nbins=6)
		splots['ax3'].set_yticklabels(['DUMMY LABEL','null', 'A', 'B', 'C'])
		splots['ax3'].yaxis.grid()
		plt.colorbar(im3, cax = caxes[1])

		im4 = splots['ax4'].imshow(wa_act.T, cmap='Blues', interpolation='none', aspect='auto')
		splots['ax4'].locator_params(axis='y',nbins=10)
		splots['ax4'].set_yticklabels(['DUMMY LABEL','N', 'E', 'S', 'W', 'STAY', 'POKE'])
		splots['ax4'].yaxis.grid()
		plt.colorbar(im4, cax = caxes[2])

		# show action selection
		im5 = splots['ax5'].imshow(loc_array.T, cmap='bone_r', interpolation='none', aspect='auto')#aspect=0.9, extent=[0,5000,0,555])
		splots['ax5'].locator_params(axis='y',nbins=10)
		splots['ax5'].set_yticklabels(['DUMMY LABEL','HOME', 'W1', 'W2', 'E1', 'E2', 'S1', 'S2'])
		splots['ax5'].tick_params(axis='y')
		plt.colorbar(im5, cax = caxes[3], ticks=[0, 1])

		# show action selection
		im6 = splots['ax6'].imshow(act_array.T, cmap='bone_r', interpolation='none', aspect='auto')#0.8, extent=[0,5000,0,555])
		splots['ax6'].locator_params(axis='y',nbins=10)
		splots['ax6'].set_yticklabels(['DUMMY LABEL','N', 'E', 'S', 'W', 'STAY', 'POKE'])
		splots['ax6'].tick_params(axis='y')
		plt.colorbar(im6, cax = caxes[4], ticks=[0, 1])

		plt.savefig('./plots/action_fig_{}.png'.format(weight_dim_list[item]))






if print_to_file:
	sys.stdout = orig_stdout
	fil.close()
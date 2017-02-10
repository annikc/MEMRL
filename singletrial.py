### Outline for Behavioural Experiment MDP 
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use('Agg')
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

plot_value_grads = False
plot_action_grads = False




#------------------------------------------------------
""" 					Set up						"""
#------------------------------------------------------
# Create standard T maze environment with a single trial and a network to decide actions
T_maze = Environment(gamma=0.8)
trialset = Trial_Set(T_maze, total_trials=200, num_events=500)
network = Network(num_units = [17, 6], learning_rate=[0.001, 0.001])

# create tracking variables to plot data later 
reward_marker = 0
value_tracker = 0 
home_store = [[],[],[],[],[],[]]
E1_store = [[],[],[],[],[],[]]
E2_store = [[],[],[],[],[],[]]
W1_store = [[],[],[],[],[],[]]
W2_store = [[],[],[],[],[],[]]
S1_store = [[],[],[],[],[],[]]
S2_store = [[],[],[],[],[],[]]


# labels for plots 
labels = []
labels.append(['HOME', 'W1', 'W2', 'E1', 'E2', 'S1', 'S2'])
labels.append(['null', 'A', 'B', 'C'])
labels.append(['N', 'E', 'S', 'W', 'STAY', 'POKE'])

# arrays for storing trial results
trial_reward = np.zeros((200,1))
trial_preder = np.zeros((200,1))

#------------------------------------------------------
""" 				   Run Trials					"""
#------------------------------------------------------

for i in range(trialset.tot_trials):
	#count = i 
	if print_statements:
		print(" ---------- \n Trial Number:", trialset.num_trials, "\n ----------")

	# initial agent starting state to HOME
	T_maze.cur_state = State(T_maze.all_loc[0], T_maze.tex_loc[T_maze.all_loc[0]]) 
	# Add trials to the set
	trialset.add_trial()

	#--------------------------------
	#### RUN THE CURRENT TRIAL
	#--------------------------------

	# BLAKE'S NEW INITIALIZATION FOR STORING RETURN AND REWARD
	store_reward = np.zeros((trialset.trials[-1].num_events,1))
	store_return = np.zeros((trialset.trials[-1].num_events,1))
	store_value  = np.zeros((trialset.trials[-1].num_events,1))
	store_rv     = np.zeros((trialset.trials[-1].num_events,1))
	store_ev     = np.zeros((trialset.trials[-1].num_events,1))
	store_action = []
	store_policy = []
	store_inputs = []
	store_acname = []
	
	# Run through the number of events specified per trial
	for j in range(trialset.current_trial.num_events):
		if print_statements:
			print "-----  EVENT #" + str(i) + '-' + str(j), "... The current state is ", trialset.current_trial.events[j][1].loc
		
		#____ FORWARD PASS THROUGH NETWORK ______
		# format state information as appropriate input vector 
		NN_input = make_state_vector(trialset.current_trial.events[j][1].loc, trialset.current_trial.events[j][1].tex, trialset.current_trial.events[j][2])
		network.forward_pass(NN_input)
		
		# network computed value
		# value_tracker = network.value[0][0]

		# network computed action 
		'''Agent choses according to prob distr or chooses greedily?  '''
		#action_choice = np.random.choice(T_maze.all_actions, 1, p=network.action[:,0])[0]
		action_choice = np.random.choice(T_maze.all_actions, 1, p=network.output_layer.selected_action.transpose()[0].tolist())[0]

		# Add events to the trial corresponding to action selections
		trialset.current_trial.add_event(action_choice)

		# compute prediction error
		# rv = (T_maze.rtrn - network.value[0][0]) # BLAKE SAYS: NO LONGER NEEDED, CALCULATED BACKWARDS IN TIME NOW!
		# ev_total = 0.5*(rv**2) # BLAKE SAYS: NO LONGER NEEDED, CALCULATED BACKWARDS IN TIME NOW!

		# network.accumulate_gradients(rv) # BLAKE SAYS: NO LONGER NEEDED, CALCULATED BACKWARDS IN TIME NOW!

#		if count == trialset.tot_trials-1:
#			if j == 0:
#				ev_total = 0
#				# store_return = []
#				# store_reward = []
#				# store_ev     = []
#				store_Wv     = []
#				store_Wa	 = [[], [], [], [], [], []]
#				store_loc 	 = []
#				store_act 	 = []
#			# gradients 
#			wv_tracker = network.output_layer.dWv
#			wa_tracker = network.output_layer.dWa
#			#print wa_tracker.shape[0]
#
#			#add to apropriate lists
#			# store_return.append(T_maze.rtrn)
#			# store_reward.append(T_maze.rwd)
#			# store_ev.append(ev_total)
#			store_reward[j] = T_maze.rwd
#			store_value[j]  = network.value[0][0]
#			store_Wv.append(wv_tracker)
#			for item in range(wa_tracker.shape[0]):
#				store_Wa[item].append(wa_tracker[item,:])
#			store_loc.append(make_state_vector(trialset.current_trial.events[j][1].loc, tex_var='null', last_act='stay')[0:7])
#			store_act.append(list(network.action.T[0]))
		
		# store the necessary values for gradient accumulation	
		store_reward[j] = T_maze.rwd
		store_value[j]  = network.value[0][0]
		store_action.append(network.output_layer.selected_action.copy())
		store_policy.append(network.output_layer.soft_policy)
		store_inputs.append(network.output_layer.inputs)
		store_acname.append(action_choice)	

		if print_statements:
			print "agent chooses: " + store_acname[j] + ", index in store: " + str(np.where(store_action[j])[0][0])

		if print_statements:
			print "cumulative reward is: " + str(np.sum(store_reward))
	

	#--------------------------------
	#### ACCUMULATE GRADIENTS FOR THE CURRENT TRIAL
	#--------------------------------
	store_return[-1] = store_value[-1] # R = V(s_t,theta)
	for j in range(trialset.current_trial.num_events-2,0,-1):

		# calculate the return at this time point
		store_return[j] = store_reward[j] + T_maze.gamma*store_return[j+1]

		# calculate the prediction error
		store_rv[j] = store_return[j] - store_value[j]
		store_ev[j] = 0.5*store_rv[j]**2

		# accumulate the gradients
		network.accumulate_gradients(store_rv[j],store_action[j],store_policy[j],store_inputs[j])

		#if store_return[j] > 0:
		#	pdb.set_trace()
		
	# rtrn = T_maze.rtrn # BLAKE SAYS: NO LONGER NEEDED, RETURN TO BE CALCULATED BACKWARDS THROUGH TIME!
	# print("Total return for trial # ", i, ", was: ", store_return[0])
	print "Total reward for trial # " + str(i) + ", was: " + str(np.sum(store_reward)) + ", with total prediction error: " + str(np.sum(store_ev))
	network.update_weights()
	trial_reward[i] = np.sum(store_reward)
	trial_preder[i] = np.sum(store_ev)
		

#rtrn = T_maze.rtrn
#if print_statements:
#		print("Total return for trial # ", i, ", was: ", rtrn)
#
#loc_array = np.squeeze(np.asarray(store_loc))
#act_array = np.squeeze(np.asarray(store_act))
#
## Plot what? 
#reward_bars = False
#
#
#
#if plot_value_grads == True: 
#	## Process data for plotting
#	# process value weights arrays
#	wv_diff = []
#	for i in range(len(store_Wv)):
#		if i == 0: 
#			wv_diff.append(store_Wv[0])
#		else: 
#			wv_diff.append(store_Wv[i] - store_Wv[i-1])
#
#	wv_array = np.squeeze(np.asarray(store_Wv))
#	wv_array1 = np.squeeze(np.asarray(wv_diff))
#	wv_loc = wv_array1[:,0:7]
#	wv_tex = wv_array1[:,7:11]
#	wv_act = wv_array1[:,11:17]
#
#	# Make Value Updates Plot
#	fig = plt.figure(1, figsize=(15,12))
#
#	# generate colorbars for imshow plots
#	caxes = []
#	caxes.append(fig.add_axes([0.91, 0.65, 0.01, 0.11]))
#	caxes.append(fig.add_axes([0.91, 0.51, 0.01, 0.11]))
#	caxes.append(fig.add_axes([0.91, 0.375, 0.01, 0.11]))
#	caxes.append(fig.add_axes([0.91, 0.25, 0.005, 0.09]))
#	caxes.append(fig.add_axes([0.91, 0.11, 0.005, 0.09]))
#
#	# Add subplots
#	num_subplots = 6
#	subplot_counter = np.arange(num_subplots)
#	splots = {}
#	for i in range(num_subplots):
#		temp = "ax"+str(i+1)
#		splots[temp] = fig.add_subplot(num_subplots,1,i+1)
#
#
#	# Subplot 1: Show return, prediction error, reward events 
#	splots['ax1'].plot(store_ev, 'r')
#	splots['ax1'].plot(store_return, 'g')
#	if reward_bars: 
#		for i in range(trialset.current_trial.num_events):
#			if store_reward[i] == 1: 
#				splots['ax1'].axvline(x=i)
#	else: 
#		splots['ax1'].plot(store_reward, 'bo')
#	#splots['ax1'].set_xlim(xmin=0, xmax=max(np.arange(trialset.current_trial.num_events)))
#	splots['ax1'].set_title('Value Weight Gradients')
#
#	# show difference in accumulated gradients from last time step
#	im2 = splots['ax2'].pcolor(wv_loc.T, cmap='Greens_r')
#	splots['ax2'].locator_params(axis='y',nbins=10)
#	splots['ax2'].set_yticklabels(labels[0])
#	splots['ax2'].yaxis.grid()
#	plt.colorbar(im2, cax = caxes[0])
#	center_labels(splots['ax2'].yaxis, labels[0])
#
#	im3 = splots['ax3'].pcolor(wv_tex.T, cmap='Greens_r')
#	splots['ax3'].locator_params(axis='y',nbins=6)
#	splots['ax3'].set_yticklabels(labels[1])
#	splots['ax3'].yaxis.grid()
#	plt.colorbar(im3, cax = caxes[1])
#	center_labels(splots['ax3'].yaxis, labels[1])
#
#	im4 = splots['ax4'].pcolor(wv_act.T, cmap='Greens_r')
#	splots['ax4'].locator_params(axis='y',nbins=10)
#	splots['ax4'].set_yticklabels(labels[2])
#	splots['ax4'].yaxis.grid()
#	plt.colorbar(im4, cax = caxes[2])
#	center_labels(splots['ax4'].yaxis, labels[2])
#
#	# show action selection
#	im5 = splots['ax5'].pcolor(loc_array.T, cmap='bone_r')#aspect=0.9, extent=[0,5000,0,555])
#	splots['ax5'].locator_params(axis='y',nbins=10)
#	splots['ax5'].set_yticklabels(labels[0])
#	splots['ax5'].tick_params(axis='y')
#	plt.colorbar(im5, cax = caxes[3], ticks=[0, 1])
#	center_labels(splots['ax5'].yaxis, labels[0])
#
#	# show action selection
#	im6 = splots['ax6'].pcolor(act_array.T, cmap='bone_r')#0.8, extent=[0,5000,0,555])
#	splots['ax6'].locator_params(axis='y',nbins=10)
#	splots['ax6'].set_yticklabels(labels[2])
#	splots['ax6'].tick_params(axis='y')
#	plt.colorbar(im6, cax = caxes[4], ticks=[0, 1])
#	center_labels(splots['ax6'].yaxis, labels[2])
#
#	plt.savefig('./plots/value_fig.svg')
#	print "Saving figure value_fig.svg"
#
#if plot_action_grads == True:
#	weight_dim_list = ['North', 'East', 'West', 'South', 'Stay', 'Poke']
#	for item in range(len(store_Wa)):
#		# make correct variables: 
#		wa_array = np.asarray(store_Wa[item]).T
#		## Process data for plotting
#		# process value weights arrays
#		wa_diff = []
#		for i in range(len(store_Wa[item])):
#			if i == 0: 
#				wa_diff.append(store_Wa[item][0])
#			else: 
#				wa_diff.append(store_Wa[item][i] - store_Wa[item][i-1])
#
#		wa_array1 = (np.asarray(wa_diff))
#
#		wa_loc = wa_array1[:,0:7]
#		wa_tex = wa_array1[:,7:11]
#		wa_act = wa_array1[:,11:17]
#		# Make Value Updates Plot
#		fig = plt.figure(item+2, figsize=(15,12))
#
#		# Add subplots
#		num_subplots = 6
#		subplot_counter = np.arange(num_subplots)
#		splots = {}
#		for i in range(num_subplots):
#			temp = "ax"+str(i+1)
#			splots[temp] = fig.add_subplot(num_subplots,1,i+1)
#
#
#		# generate colorbars for imshow plots
#		caxes = []
#		caxes.append(fig.add_axes([0.91, 0.65, 0.01, 0.11]))
#		caxes.append(fig.add_axes([0.91, 0.51, 0.01, 0.11]))
#		caxes.append(fig.add_axes([0.91, 0.375, 0.01, 0.11]))
#		caxes.append(fig.add_axes([0.91, 0.25, 0.005, 0.09]))
#		caxes.append(fig.add_axes([0.91, 0.11, 0.005, 0.09]))
#
#		# Subplot 1: Show return, prediction error, reward events 
#		splots['ax1'].plot(store_ev, 'r')
#		splots['ax1'].plot(store_return, 'g')
#		if reward_bars: 
#			for i in range(trialset.current_trial.num_events):
#				if store_reward[i] == 1: 
#					splots['ax1'].axvline(x=i)
#		else: 
#			splots['ax1'].plot(store_reward, 'bo')
#		splots['ax1'].set_xlim(xmin=0, xmax=max(np.arange(trialset.current_trial.num_events)))
#		splots['ax1'].set_title('Action Gradient Derivatives for Unit: {}'.format(weight_dim_list[item]))
#
#		# show difference in accumulated gradients from last time step
#		im2 = splots['ax2'].pcolor(wa_loc.T, cmap='Blues_r')
#		splots['ax2'].locator_params(axis='y',nbins=10)
#		splots['ax2'].set_yticklabels(labels[0])
#		splots['ax2'].yaxis.grid()
#		plt.colorbar(im2, cax = caxes[0])
#
#		im3 = splots['ax3'].pcolor(wa_tex.T, cmap='Blues_r')
#		splots['ax3'].locator_params(axis='y',nbins=6)
#		splots['ax3'].set_yticklabels(labels[1])
#		splots['ax3'].yaxis.grid()
#		plt.colorbar(im3, cax = caxes[1])
#
#		im4 = splots['ax4'].pcolor(wa_act.T, cmap='Blues_r')
#		splots['ax4'].locator_params(axis='y',nbins=10)
#		splots['ax4'].set_yticklabels(labels[2])
#		splots['ax4'].yaxis.grid()
#		plt.colorbar(im4, cax = caxes[2])
#
#		# show action selection
#		im5 = splots['ax5'].pcolor(loc_array.T, cmap='bone_r')#aspect=0.9, extent=[0,5000,0,555])
#		splots['ax5'].locator_params(axis='y',nbins=10)
#		splots['ax5'].set_yticklabels(labels[0])
#		splots['ax5'].tick_params(axis='y')
#		plt.colorbar(im5, cax = caxes[3], ticks=[0, 1])
#
#		# show action selection
#		im6 = splots['ax6'].pcolor(act_array.T, cmap='bone_r')#0.8, extent=[0,5000,0,555])
#		splots['ax6'].locator_params(axis='y',nbins=10)
#		splots['ax6'].set_yticklabels(labels[2])
#		splots['ax6'].tick_params(axis='y')
#		plt.colorbar(im6, cax = caxes[4], ticks=[0, 1])
#
#		plt.savefig('./plots/action_fig_{}.svg'.format(weight_dim_list[item]))
#		print "Saving figure action_fig_{}.svg".format(weight_dim_list[item])





if print_to_file:
	sys.stdout = orig_stdout
	fil.close()

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib import animation
import matplotlib.ticker 
import pdb
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mdp_fns import *

'''
- Parameter tuning 
- Number of hidden hidden_layers
- Learning rates
- Weight initializations
- What is happening when agent gets 0 reward for trial??? 
'''


print_statements = False
plot_value_grads = False
plot_action_grads = False

#------------------------------------------------------
""" 					Set up						"""
#------------------------------------------------------
# Create standard T maze environment  with trial set and network
T_maze = Environment(gamma=0.9, shift='even')
trialset = Trial_Set(T_maze, total_trials=300, num_events=300)
network = Network(num_units = [17, 6], learning_rate=[2e-5, 7.5e-4, 3e-4])

F = open('./what_do.txt','w') 
# create tracking variables to plot data later 
reward_marker = 0
value_tracker = 0 
 
home_store = [[], []]
E1_store = [[],[]]
E2_store = [[],[]]
W1_store = [[],[]]
W2_store = [[],[]]
S1_store = [[],[]]
S2_store = [[],[]]

val_data = {'HOME':home_store, 'E1':E1_store, 'E2':E2_store, 'W1':W1_store, 'W2':W2_store, 'S1':S1_store, 'S2':S2_store}
# labels for plots 
labels = []
labels.append(['HOME', 'W1', 'W2', 'E1', 'E2', 'S1', 'S2'])
labels.append(['null', 'A', 'B', 'C'])
labels.append(['N', 'E', 'S', 'W', 'STAY', 'POKE'])

# arrays for storing trial reward and prediction error
trial_reward = np.zeros((trialset.tot_trials,1)) 
trial_preder = np.zeros((trialset.tot_trials,1))


start_time = time.time()
#------------------------------------------------------
""" 				   Run Trials					"""
#------------------------------------------------------
for i in range(trialset.tot_trials):
	# initial agent starting state to HOME
	#start_loc = T_maze.all_loc[0]
	# initialize agent start location randomly
	start_loc = T_maze.all_loc[np.random.choice(7)]
	T_maze.cur_state = State(start_loc, T_maze.tex_loc[start_loc]) 
	# Add trials to the set
	trialset.add_trial()

	#--------------------------------
	#### RUN THE CURRENT TRIAL
	#--------------------------------
	store_reward = np.zeros((trialset.trials[-1].num_events,1))
	store_return = np.zeros((trialset.trials[-1].num_events,1))
	store_value  = np.zeros((trialset.trials[-1].num_events,1))
	store_rv     = np.zeros((trialset.trials[-1].num_events,1))
	store_ev     = np.zeros((trialset.trials[-1].num_events,1))
	store_activity = []
	
	what_do = []
	# Run through the number of events specified per trial
	for j in range(trialset.current_trial.num_events):
		#____ FORWARD PASS THROUGH NETWORK ______
		# format state information as appropriate input vector 
		NN_input = make_state_vector(trialset.current_trial.events[j][1].loc, trialset.current_trial.events[j][1].tex, trialset.current_trial.events[j][2])
		network.forward_pass(NN_input)

		# convert action choice from 0-1 into corresponding state string
		action_choice = np.random.choice(T_maze.all_actions, 1, p=network.output_layer.selected_action.transpose()[0].tolist())[0]

		# Add events to the trial corresponding to action selections
		trialset.current_trial.add_event(action_choice)
		
		# store the necessary reward and value for gradient accumulation	
		store_reward[j] = T_maze.rwd
		store_value[j]  = network.value[0][0]

		# create a list of network activities for this timestep
		netactivity = []
		if network.num_hidden == 0:
			netactivity.append(network.output_layer.inputs)
		if network.num_hidden > 0:
			netactivity.append(network.hidden_layers[0].inputs)	
			for nl in range(network.num_hidden):
				netactivity.append(network.hidden_layers[nl].activity)
		netactivity.append([network.output_layer.selected_action.copy(), network.output_layer.soft_policy])

		# store the activity list for this time-step
		store_activity.append(netactivity)

		what_do.append((trialset.current_trial.events[j][1].loc,trialset.current_trial.events[j][2], T_maze.correct_reward))

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
		network.accumulate_gradients(store_rv[j], store_activity[j])

	if i % 10 == 0: 
		print "Total reward for trial # " + str(i) + ", was: " + str(np.sum(store_reward)) + ", with total prediction error: " + str(np.sum(store_ev))
	if np.sum(store_reward) == 0 and i > 1000: 
		F.write('\n{}'.format(what_do))

	network.update_weights()
	trial_reward[i] = np.sum(store_reward)
	trial_preder[i] = np.sum(store_ev)

	test_options(val_data, T_maze, network, value=True, policy=False)
F.close()


elapsed_time = time.time() - start_time
print "Total time for run with {} trials is {} minutes".format(trialset.tot_trials, elapsed_time/60 )

###plot_test_vals(val_data, trialset)
###
###grid_vals = np.asarray([[val_data[i][0][-1] for i in labels[0]]])
###plot_val_grid(grid_vals)
####val_plt = plt.imshow(grid_vals[:,5:], cmap='jet', interpolation='none')
####plt.xticks(np.arange(7), labels[0])
####plt.colorbar(val_plt, ticks=[-1, 1])
###
###plt.figure(2)
###plt.plot(trial_reward, 'b', label="Reward")
###plt.plot(trial_preder, 'r', label="Prediction Error")
###plt.legend(loc=0)
###plt.show()

def calc_centre(loc):
	if (loc=='W2'):
		x = 0 + 0.5
		y = 2 + 0.5
	elif (loc=='W1'):
		x = 1 + 0.5
		y = 2 + 0.5
	elif (loc=='HOME'):
		x = 2 + 0.5
		y = 2 + 0.5
	elif (loc=='E1'):
		x = 3 + 0.5
		y = 2 + 0.5
	elif (loc=='E2'):
		x = 4 + 0.5
		y = 2 + 0.5
	elif (loc=='S1'):
		x = 2 + 0.5
		y = 1 + 0.5
	elif (loc=='S2'):
		x = 2 + 0.5
		y = 0 + 0.5
	else:
		x=0
		y=0
		print "Location error"
	return (x,y)

def rew_loc(loc):
	if (loc=='W2'):
		marker_x = 0
		marker_y = 2.5 
	elif (loc=='E2'):
		marker_x = 5
		marker_y = 2.5 
	elif (loc=='S2'):
		marker_x = 2.5 
		marker_y = 0
	else: 
  		print "reward loc not found"

  	return (marker_x, marker_y)

x = np.arange(6)
y = np.arange(4)


fig = plt.figure(figsize=(10,6))
patches = []
ax = fig.gca()
ax.set_xticks(x)
ax.set_yticks(y)

agt = plt.Circle((2.5,2.5), 0.2, fc='#234756')
rwd = plt.Circle((0,0), 0.1, fc='r')

time_template = 'time = %.1fs'
time_text = ax.text(0.02, 0.05, '', transform=ax.transAxes, color= 'r')
def init():	
	time_text.set_text(time_template % (i))
	patches.append(mpatches.Rectangle((0, 0), 2, 2, color='k', ec="none"))
	patches.append(mpatches.Rectangle((3, 0), 2, 2, color='k', ec="none"))
	patches.append(mpatches.Rectangle((0, 3), 6, 2, color='k', ec="none"))
	collection = PatchCollection(patches, alpha=0.2)
	ax.add_collection(collection)
	print "Starting location is", what_do[0][0]
	print "Reward starts at ", T_maze.correct_reward
	agt.center = calc_centre(what_do[1][0])
	rwd.center = rew_loc(what_do[1][2])
	ax.add_patch(agt)
	ax.add_patch(rwd)
	return agt, rwd, time_text

def animate(i):
	xa, ya = agt.center
	xa, ya = calc_centre(what_do[i][0])
	agt.center = xa, ya
	if (what_do[i+1][1]=='poke'):
		time_text = ax.text((xa/5)-0.005, (ya/3)-0.01, '*', transform=ax.transAxes, color= 'r', fontsize=14, fontweight='bold')
	else:
		time_text = ax.text(xa/5, ya/3, '', transform=ax.transAxes, color= 'r')
	print "step {}, loc {}, action {}".format(i, what_do[i][0], what_do[i+1][1])
	
	xr, yr = rwd.center
	xr, yr = rew_loc(what_do[i][2])
	rwd.center = xr, yr 
	ax.set_title('Timestep = {}, Action Selected: {}'.format(i, what_do[i][1]))

	return agt, rwd, time_text,


anim = animation.FuncAnimation(fig, animate, 
								init_func=init, 
								frames=len(what_do)-1,
								interval=750,
								repeat=True, repeat_delay=1000, blit=True)
plt.grid()
plt.show()




if print_to_file:
	sys.stdout = orig_stdout
	fil.close()

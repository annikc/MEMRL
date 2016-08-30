### Outline for Behavioural Experiment MDP 
import numpy as np 
import matplotlib.pyplot as plt 
import pdb
import random
from MDP_def import *

''' 
----------------------
		NOTES 
----------------------
* Rewarding works, events[3] tracks cumulative reward  -- should we instead be tracking reward at a given step? 
* Commenting code 
* 


----------------------
		TO DO 
----------------------
* Initial pass -- start with getting some agent who selects actions properly -- write functions to plot current state/environment 
	- animated plots 
	- better way of generating grid of environment?
		- look into using TKinter 

* Have agent select actions based on reward maximization

'''




#------------------------------------------------------
""" 					Set up						"""
#------------------------------------------------------
# Create environment with default structure
T_maze = Environment()
# Generate trial set with specified number of trials
trialset = Trial_Set(1)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

#------------------------------------------------------
""" 		Generate Trial Run Visualization		"""
#------------------------------------------------------

# Create grid for plotting 
height = 3
width = 5
env_map = [[]]*height
env_labels =  [[]]*height
for i in range(len(env_map)):
	env_map[i] = [[]]*width
	env_labels[i] = [[]]*width

env_labels[0][0] = 'W2'

def plot_walk(agent_loc, step_no): 
	if agent_loc == 'W2':
		for i in range(len(env_map)):
			for j in range(len(env_map[i])):
				env_map[i][j] = (1,1,1)
		env_map[0][0] = (1,0,0)
	elif agent_loc == 'W1':
		for i in range(len(env_map)):
			for j in range(len(env_map[i])):
				env_map[i][j] = (1,1,1)
		env_map[0][1] = (1,0,0)
	elif agent_loc == 'HOME':
		for i in range(len(env_map)):
			for j in range(len(env_map[i])):
				env_map[i][j] = (1,1,1)
		env_map[0][2] = (1,0,0)
	elif agent_loc == 'E1': 
		for i in range(len(env_map)):
			for j in range(len(env_map[i])):
				env_map[i][j] = (1,1,1)
		env_map[0][3] = (1,0,0) 
	elif agent_loc == 'E2': 
		for i in range(len(env_map)):
			for j in range(len(env_map[i])):
				env_map[i][j] = (1,1,1)
		env_map[0][4] = (1,0,0)
	elif agent_loc == 'S1':
		for i in range(len(env_map)):
			for j in range(len(env_map[i])):
				env_map[i][j] = (1,1,1)
		env_map[1][2] = (1,0,0)
	elif agent_loc == 'S2':
		for i in range(len(env_map)):
			for j in range(len(env_map[i])):
				env_map[i][j] = (1,1,1)
		env_map[2][2] = (1,0,0)	

	#props = dict(facecolor='none', alpha=0.7)
	plt.text(-0.05, 0, '$W2$')
	plt.text(0.95, 0, '$W1$')
	plt.text(1.75, 0, '$HOME$')
	plt.text(2.95, 0, '$E1$')
	plt.text(3.95, 0, '$E2$')
	plt.text(1.95, 1, '$S1$')
	plt.text(1.95, 2, '$S2$')
	plt.text(3.95, -0.75, 't = {0}'.format(step_no))

	plt.imshow(env_map, interpolation="none", aspect=1)
	plt.show()

def plot_action_choices(choice): 
	plt.scatter(step_no, return_is)


#------------------------------------------------------
""" 				   Run Trial					"""
#------------------------------------------------------
## Try this out for "weighted" action selection
weighted_actions = [['N','E','S','W','stay'], ['poke']]
actions_list = trialset.working_env.all_actions
#____ WEIGHT MATRIX INITIALIZATIONS ______
W_0 = np.random.randn(50,17)
W_1 = np.random.randn(50,50)
W_A = np.random.randn(6,50)
W_V = np.random.randn(1,50)

b_0 = np.random.randn(50,1)
b_1 = np.random.randn(50,1)
b_A = np.random.randn(6,1)
b_V = np.random.randn(1,1)


for i in range(trialset.tot_trials):
	h_0 = np.zeros(50)
	h_1 = np.zeros(50)
	policy_calc = np.zeros(6)
	value_calc = np.zeros(1)
	# Add trials to the set
	trialset.add_trial()
	print "Agent's initial state (time = ", trialset.current_trial.time_is,") is ", trialset.current_trial.events[0][1].loc
	north = []
	east = [] 
	south = []
	west = []
	stay = []
	poke = [] 
	# Run through the number of events specified per trial
	for j in range(trialset.current_trial.num_events):
		print " ----- "
		print "The current state is ", trialset.current_trial.events[j][1].loc
		input_vec = np.zeros(17)
		# feed network 
		loc_var = trialset.current_trial.events[j][1].loc
		if loc_var == 'HOME':
			input_vec[0] = 1
		elif loc_var == 'W1':
			input_vec[1] = 1 
		elif loc_var == 'W2':
			input_vec[2] = 1 
		elif loc_var == 'E1':
			input_vec[3] = 1 
		elif loc_var == 'E2':
			input_vec[4] = 1
		elif loc_var == 'S1':
			input_vec[5] = 1
		elif loc_var == 'S2':
			input_vec[6] = 1

		tex_var = trialset.current_trial.events[j][1].tex
		if tex_var == 'null':
			input_vec[7] = 1
		elif tex_var == 'A':
			input_vec[8] = 1
		elif tex_var == 'B':
			input_vec[9] = 1
		elif tex_var == 'C':
			input_vec[10] = 1

		last_act = trialset.current_trial.events[j][2]
		if last_act == 'stay':
			input_vec[11] = 1
		elif last_act == 'poke':
			input_vec[12] = 1
		elif last_act == 'N':
			input_vec[13] = 1
		elif last_act == 'E':
			input_vec[14] = 1
		elif last_act == 'W':
			input_vec[15] = 1
		elif last_act == 'S':
			input_vec[16] = 1 

		#____ WEIGHT MATRICES ______
		# initialized above, updated in this block 
		# similarly with bias vectors 

		#____ RANDOM CHOICES ______
		# inputs/"state" vector
		s = np.transpose(input_vec)
		# intermediate matrix dot product of weights and inputs
		A = np.dot(W_0,s)
		# hidden layer 1 unit values
		for i in range(len(h_0)):
			h_0[i] = np.tanh(A[i] + b_0[i])
		# intermediate matrix dot product of weights and hidden layer 1
		B = np.dot(W_1,h_0)
		# hidden layer 2 unit values
		for i in range(len(h_1)):
			h_1[i] = np.tanh(B[i] + b_1[i])
		# intermediate matrix dot product of weights and hidden layer 2
		C = np.dot(W_A,h_1)
		# ##############  figure out what is going on with softmax in this layer ##############
		# intermediate calculation of policy units
		D = np.zeros(len(policy_calc))
		for i in range(len(policy_calc)):
			D[i] = C[i] + b_A[i]
		# policy units
		policy_calc = softmax(D)
		# intermediate matrix dot product of weights and hidden layer 2
		E = np.dot(W_V,h_1)
		for i in range(len(value_calc)):
			value_calc[i] = (E[i] + b_V[i])

		#act = T_maze.all_actions
		#choice = random.choice(act)
		# _____ PREFERENCE FOR POKING ____
		#act = weighted_actions
		#pre_choice = random.choice(act)
		#choice = random.choice(pre_choice)

		try_this = np.random.choice(actions_list, 1, p=list(policy_calc))
		print "attempt action", try_this[0]
		choice = try_this[0]	

		print "Agent chooses action ", choice

		# Add to tally of choices
		if choice == 'N':
			north.append(1)
		elif choice == 'E':
			east.append(1)
		elif choice == 'S':
			south.append(1)
		elif choice == 'W':
			west.append(1)
		elif choice == 'stay':
			stay.append(1)
		elif choice == 'poke':
			poke.append(1)
		else:
			print "Choice tally error"
		
		
#		print "Agent's return is ", trialset.current_trial.events[j][4]
		# Add events to the trial corresponding to action selections
		trialset.current_trial.add_event(choice)
		agent_loc = trialset.current_trial.events[j][1].loc
		print "agent loc", agent_loc
		step_no = trialset.current_trial.events[j][0]
		print "step no", step_no
		plot_walk(agent_loc, step_no)
		#if j == 0: 
		#	pass
		#elif j == 1:
		#	plt.scatter(step_no, trialset.current_trial.events[j][4], color='b', label="Return")
		#	plt.scatter(step_no, trialset.current_trial.events[j][3], color='r', label="Reward")
		#else:
		#	plt.scatter(step_no, trialset.current_trial.events[j][4], color='b')
		#	plt.scatter(step_no, trialset.current_trial.events[j][3], color='r')
	#plt.set_xlim = ([0, step_no + 5]) ## NOT WORKING? 
	#plt.set_ylim = ([-0.05, 2.5]) ## NOT WORKING?
	#plt.legend(loc=4)
	#plt.show()

	#ind = 1
	#wid = 1 
	#fig, ax = plt.subplots()
	#north_plot = ax.bar(ind, len(north), wid, color='blue')
	#east_plot = ax.bar(ind+1*wid, len(east), wid, color='blue')
	#south_plot = ax.bar(ind+2*wid, len(south), wid, color='blue')
	#west_plot = ax.bar(ind+3*wid, len(west), wid, color='blue')
	#stay_plot = ax.bar(ind+4*wid, len(stay), wid, color='black')
	#poke_plot = ax.bar(ind+5*wid, len(poke), wid, color='red')

	#plt.show()


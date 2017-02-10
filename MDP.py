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


colour = {1:(0.803921569,0,0.337254902), 
			2:(1,0.164705882,0.164705882), 
			3:(1,0.3725490196,0.27058823529), 
			4:(1,0.33725490196,0.41568627451), 
			5:(1,0.41176470588,0.164705882), 
			6:(0.88235294117,0.64705882352,0.48235294117), 
			7:(0.86274509803,0.8431372549,0.96470588235),
			8:(0.67058823529,0.74901960784,0.94117647058),
			9:(0,0.11372549019,0.30196078431),
			10:(0.92156862745,0.91372549019,0.87058823529),
			11:(0.97647058823,0.89803921568,0.47450980392),
			12:(1,0.87058823529,0.87058823529)}

#------------------------------------------------------
""" 					Set up						"""
#------------------------------------------------------
# Create environment with default structure
T_maze = Environment()
# Generate trial set with specified number of trials
trialset = Trial_Set(1)


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

for i in range(trialset.tot_trials):
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
		#____ RANDOM CHOICES ______
		#act = T_maze.all_actions
		#choice = random.choice(act)
		# _____ PREFERENCE FOR POKING ____ 
		act = weighted_actions
		pre_choice = random.choice(act)
		choice = random.choice(pre_choice)

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

	plt.show()

#------------------------------------------------------
""" 			   Accumulate gradients				"""
#------------------------------------------------------

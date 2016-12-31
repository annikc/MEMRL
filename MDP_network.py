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
* Rewarding works, events[3] tracks cumulative reward  -- should we instead be tracking reward at a given step? 
* Commenting code 
* I have put weight updates after each event -- sort of fixing issue of just selecting one method indiscriminantly? 


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

'''
----------------------
	  Derivatives 
----------------------
rtrn = trialset.working_env.rtrn
val_t = np.dot(W_V,h_1) - b_V 

rv = -(rtrn - val_t)
d_sig1 = np.zeros((l,l))
for j in range(l):
	for i in range(l):
		if i == j: 
			d_sig1[i,j] = h_1[j](1 - h_1[j])
		else:
			d_sig1[i,j] = 0 

B = np.zero((l,l,m))
for k in range(m):
	for j in range(l):
		for i in range(l):
			if i == j: 
			B[i,j,k] = h_0[k]
		else:
			B[i,j,k] = 0 

d_sig2 = np.zeros((m,m))
	for i in range(m):
		for j in range(m):
			if j == i: 
				d_sig2[i,j] = h_0[j](1 - h_0[j])
			else:
				d_sig2[i,j] = 0

D = np.zeros((m,m,n))
for k in range(n): 
	for j in range(m):
		for i inrange(m):
			if i == j: 
				D[i,j,k] = s[k]


u = np.dot(W_V,d_sig1)
E = np.dot(W_1,d_sig2)
v = np.dot(u,E)


dLv_dWv = np.zeros((1,l))
for i in range(l): 
	dLv_dWv[0][i] = rv*h_1[i]


dLv_dW1 = np.zeros((l,m))
for i in range(l): 
	for j in range(m):
		dLv_dW1[i][j] = rv*u[l]*h_0[m]

dLv_dW0 = np.zeros((m,n))
for i in range(m):
	for j in range(n):
		dLv_dW0[i][j] = rv*v[m]*input_vec[n]

dLv_dbv = rv

dLv_db1 = np.zeros(l)
for i in range(l): 
	dLv_db1[i] = rv*u[i]

dLv_db0 = np.zeros(m): 
for i in range(m):
	dLv_db0[i] = rv*v[i]

'''



#------------------------------------------------------
""" 					Set up						"""
#------------------------------------------------------
# Create environment with default structure
T_maze = Environment()
network = Network()
# Generate trial set with specified number of trials
trialset = Trial_Set(10, T_maze)

## Try this out for "weighted" action selection
weighted_actions = [['N','E','S','W','stay'], ['poke']]
actions_list = trialset.working_env.all_actions # ['N','E','S','W','stay','poke']

#____ LEARNING RATES _____
epsilon = np.array([0.00001,0.00001,0.000001,0.00001])

#____ WEIGHT MATRIX INITIALIZATIONS ______
k = 6
l = 50
m = 50
n = 17

w_scale = np.array([1.0, 1.0, 1.0, 1.0])

W_0 = w_scale[0]*np.random.randn(m,n)
W_1 = w_scale[1]*np.random.randn(l,m)
W_A = w_scale[2]*np.random.randn(k,l)
W_V = w_scale[3]*np.random.randn(1,l)

b_0 = w_scale[0]*np.random.randn(m,1)
b_1 = w_scale[1]*np.random.randn(l,1)
b_A = w_scale[2]*np.random.randn(k,1)
b_V = w_scale[3]*np.random.randn(1,1)

h_0 = np.zeros((m,1))
h_1 = np.zeros((l,1))
policy_calc = np.zeros((k,1))
value_calc = np.zeros((1,1))

store_return = []
store_reward = []
store_trialnum = []
store_ev = []
reward_test = 0
rand_vec = []

home_store = [[],[],[],[],[],[]]
E1_store = [[],[],[],[],[],[]]
E2_store = [[],[],[],[],[],[]]
W1_store = [[],[],[],[],[],[]]
W2_store = [[],[],[],[],[],[]]
S1_store = [[],[],[],[],[],[]]
S2_store = [[],[],[],[],[],[]]


for i in range(k):
	rand_vec.append((np.random.random(),))
#------------------------------------------------------
""" 				   Run Trial					"""
#------------------------------------------------------
for i in range(trialset.tot_trials):
	# Add trials to the set
	trialset.add_trial()
	trialset.current_trial.events[0][1].loc = 'HOME'
	###print "Agent's initial state (time = ", trialset.current_trial.time_is,") is ", trialset.current_trial.events[0][1].loc
	north = []
	east = [] 
	south = []
	west = []
	stay = []
	poke = []

	dLv_dWv = np.zeros((1,l))
	dLv_dW1 = np.zeros((l,m))
	dLv_dW0 = np.zeros((m,n))
	dLv_dbv = 0
	dLv_db1 = np.zeros((l,1))
	dLv_db0 = np.zeros((m,1))

	dLpi_dWa = np.zeros((k,l))
	dLpi_dW1 = np.zeros((l,m))
	dLpi_dW0 = np.zeros((m,n))
	dLpi_dba = np.zeros((k,1))
	dLpi_db1 = np.zeros((l,1))
	dLpi_db0 = np.zeros((m,1))

	dWv = np.zeros((1,l))
	dWa = np.zeros((k,l))
	dW1 = np.zeros((l,m))
	dW0 = np.zeros((m,n))
	dbv = 0
	dba = np.zeros((k,1))
	db1 = np.zeros((l,1))
	db0 = np.zeros((m,1))
	input_vec = 12345

	# variable to track value error
	ev_total = 0
	
	# Run through the number of events specified per trial
	for j in range(trialset.current_trial.num_events):
		###print " ----- "
		###print "The current state is ", trialset.current_trial.events[j][1].loc
		input_vec = np.zeros((17,1))
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

		#____ FORWARD PASS THROUGH NETWORK ______

		# inputs/"state" vector
		s = input_vec
		#print s.T
		
		# hidden layer 1 unit values
		pre_h_0 = np.dot(W_0,s)
		h_0 = sigmoid(pre_h_0 + b_0)

		# hidden layer 2 unit values
		pre_h_1 = np.dot(W_1,h_0)
		h_1 = sigmoid(pre_h_1 + b_1)

		# policy unit values
		pre_D = np.dot(W_A,s)
		D = pre_D + b_A
		#try_this_soln = [x/7 for x in D]
		# policy units
		#if i < 3: 
		#	pass
		#else: 
		#	pdb.set_trace()
		policy_calc = softmax(D)
		# intermediate matrix dot product of weights and hidden layer 2
		value_calc = (np.dot(W_V,h_1) + b_V)

		#____ CHOICE SELECTION ______
		try_this = np.random.choice(actions_list, 1, p=list(policy_calc[:,0]))

		choice = try_this[0]
		if print_statements:
			print "Agent chooses action ", choice
		else:
			pass

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

		# Add events to the trial corresponding to action selections
		trialset.current_trial.add_event(choice)
		agent_loc = trialset.current_trial.events[j][1].loc
		###print "agent loc", agent_loc
		step_no = trialset.current_trial.events[j][0]
		###print "step no", step_no
		#plot_walk(agent_loc, step_no)

		#____ CHOICE EVALUATION ______
		# Calculate return and value 
		rtrn = trialset.working_env.rtrn

		rv = (rtrn - value_calc)

		ev_total += (0.5*rv[0][0]**2)

		#____ WEIGHT UPDATES ______
		# Calculate values for derivative matrices
		d_h1 = h_1*(1-h_1)
		d_sig1 = np.diag(d_h1[:,0])

		d_h0 = h_0*(1-h_0)
		d_sig2 = np.diag(d_h0[:,0])

		#dgamma_dmu = np.eye(k)-np.transpose(np.reshape(np.repeat(policy_calc, k), [k,k]))


		u = np.dot(W_V,d_sig1)
		E = np.dot(W_1,d_sig2)
		v = np.dot(u,E)

		dLv_dWv = -rv*h_1[:,0]

		#dLv_dW1 = -rv*np.outer(np.transpose(u)[:,0],h_0[:,0])

		#dLv_dW0 = -rv*np.outer(np.transpose(v)[:,0],s[:,0])

		dLv_dbv = -rv

		#dLv_db1 = -rv*np.transpose(u)

		#dLv_db0 = -rv*np.transpose(v)

		dLpi_dWa = rv*(np.repeat(h_1.transpose(),k,axis=0)*(1.0 - k*np.repeat(policy_calc,l,axis=1)))# note: these are actually the sums of the gradients
		dLpi_dba = rv*(1.0 - k*policy_calc)

		dWv = dWv + dLv_dWv
		dWa = dWa + dLpi_dWa

		dbv = dbv + dLv_dbv
		dba = dba + dLpi_dba

		if j < (trialset.current_trial.num_events-8):
			pass
		elif j == (trialset.current_trial.num_events - 8): 
			trialset.current_trial.events[0][1].loc = 'HOME'
		elif j == (trialset.current_trial.num_events - 7):
			for it in range(len(policy_calc)):
				home_store[it].append(policy_calc[it][0])
			trialset.current_trial.events[0][1].loc = 'E1'
		elif j == trialset.current_trial.num_events - 6:
			for it in range(len(policy_calc)):
				E1_store[it].append(policy_calc[it][0]) 
			trialset.current_trial.events[0][1].loc = 'E2'
		elif j == trialset.current_trial.num_events - 5:
			for it in range(len(policy_calc)):
				E2_store[it].append(policy_calc[it][0])
			trialset.current_trial.events[0][1].loc = 'W1'
		elif j == trialset.current_trial.num_events - 4: 
			for it in range(len(policy_calc)):
				W1_store[it].append(policy_calc[it][0])
			trialset.current_trial.events[0][1].loc = 'W2'
		elif j == trialset.current_trial.num_events - 3: 
			for it in range(len(policy_calc)):
				W2_store[it].append(policy_calc[it][0])
			trialset.current_trial.events[0][1].loc = 'S1'
		elif j == trialset.current_trial.num_events - 2: 
			for it in range(len(policy_calc)):
				S1_store[it].append(policy_calc[it][0])
			trialset.current_trial.events[0][1].loc = 'S2'
		elif j == trialset.current_trial.num_events - 1: 
			for it in range(len(policy_calc)):
				S2_store[it].append(policy_calc[it][0])

		reward_test = reward_test + trialset.working_env.rwd

		if print_statements:
			print "cumulative reward is ", reward_test
		else:
			pass


	print("Total return for trial # ", i, ", was: ", rtrn)
	store_return.append(rtrn)
	store_reward.append(reward_test)
	store_trialnum.append(trialset.num_trials)
	store_ev.append(ev_total)

	# Update weight matrices
	'''
	Need to implement calculations for value and write in derivative functions for policy calculations
	'''
	W_0 = W_0 #- dLv_dW0 # - dLpi_dW0
	W_1 = W_1 #- dLv_dW1 # - dLpi_dW1
	W_A = W_A - np.multiply(epsilon[2], dWa)
	W_V = W_V - np.multiply(epsilon[3], dWv)

	b_0 = b_0 #- dLv_db0 # - dLpi_db0
	b_1 = b_1 #- dLv_db1 # - dLpi_db1
	b_A = b_A - np.multiply(epsilon[2], dba)
	b_V = b_V - np.multiply(epsilon[3], dbv) 
	
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



# row and column sharing
f, axarr = plt.subplots(3, 5, figsize=(12,8))
axarr[0, 0].plot(store_trialnum, W2_store[0], 'r', label='North')
axarr[0, 0].plot(store_trialnum, W2_store[1], 'k', label='East')
axarr[0, 0].plot(store_trialnum, W2_store[2], 'y', label='South')
axarr[0, 0].plot(store_trialnum, W2_store[3], 'g-o', label='West')
axarr[0, 0].plot(store_trialnum, W2_store[4], 'c', label='Stay')
axarr[0, 0].plot(store_trialnum, W2_store[5], 'b-o', label='Poke')
axarr[0, 0].set_title('W2')

axarr[0, 1].plot(store_trialnum, W1_store[0], 'r', label='North')
axarr[0, 1].plot(store_trialnum, W1_store[1], 'k', label='East')
axarr[0, 1].plot(store_trialnum, W1_store[2], 'y', label='South')
axarr[0, 1].plot(store_trialnum, W1_store[3], 'g-o', label='West')
axarr[0, 1].plot(store_trialnum, W1_store[4], 'c', label='Stay')
axarr[0, 1].plot(store_trialnum, W1_store[5], 'b-o', label='Poke')
axarr[0, 1].set_title('W1')

axarr[0, 2].plot(store_trialnum, home_store[0], 'r', label='North')
axarr[0, 2].plot(store_trialnum, home_store[1], 'k', label='East')
axarr[0, 2].plot(store_trialnum, home_store[2], 'y', label='South')
axarr[0, 2].plot(store_trialnum, home_store[3], 'g-o', label='West')
axarr[0, 2].plot(store_trialnum, home_store[4], 'c', label='Stay')
axarr[0, 2].plot(store_trialnum, home_store[5], 'b-o', label='Poke')
axarr[0, 2].set_title('HOME')

axarr[0, 3].plot(store_trialnum, E1_store[0], 'r', label='North')
axarr[0, 3].plot(store_trialnum, E1_store[1], 'k', label='East')
axarr[0, 3].plot(store_trialnum, E1_store[2], 'y', label='South')
axarr[0, 3].plot(store_trialnum, E1_store[3], 'g-o', label='West')
axarr[0, 3].plot(store_trialnum, E1_store[4], 'c', label='Stay')
axarr[0, 3].plot(store_trialnum, E1_store[5], 'b-o', label='Poke')
axarr[0, 3].set_title('E1')

axarr[0, 4].plot(store_trialnum, E2_store[0], 'r', label='North')
axarr[0, 4].plot(store_trialnum, E2_store[1], 'k', label='East')
axarr[0, 4].plot(store_trialnum, E2_store[2], 'y', label='South')
axarr[0, 4].plot(store_trialnum, E2_store[3], 'g-o', label='West')
axarr[0, 4].plot(store_trialnum, E2_store[4], 'c', label='Stay')
axarr[0, 4].plot(store_trialnum, E2_store[5], 'b-o', label='Poke')
axarr[0, 4].set_title('E2')

axarr[1, 2].plot(store_trialnum, S1_store[0], 'r', label='North')
axarr[1, 2].plot(store_trialnum, S1_store[1], 'k', label='East')
axarr[1, 2].plot(store_trialnum, S1_store[2], 'y', label='South')
axarr[1, 2].plot(store_trialnum, S1_store[3], 'g-o', label='West')
axarr[1, 2].plot(store_trialnum, S1_store[4], 'c', label='Stay')
axarr[1, 2].plot(store_trialnum, S1_store[5], 'b-o', label='Poke')
axarr[1, 2].set_title('S1')

axarr[2, 2].plot(store_trialnum, S2_store[0], 'r', label='North')
axarr[2, 2].plot(store_trialnum, S2_store[1], 'k', label='East')
axarr[2, 2].plot(store_trialnum, S2_store[2], 'y', label='South')
axarr[2, 2].plot(store_trialnum, S2_store[3], 'g-o', label='West')
axarr[2, 2].plot(store_trialnum, S2_store[4], 'c', label='Stay')
axarr[2, 2].plot(store_trialnum, S2_store[5], 'b-o', label='Poke')
axarr[2, 2].set_title('S2')

axarr[1,0].axis('off')
axarr[1,1].axis('off')
axarr[2,0].axis('off')
axarr[2,1].axis('off')
axarr[-2,-2].axis('off')
axarr[-1,-2].axis('off')
axarr[-2,-1].axis('off')
axarr[-1,-1].axis('off')

axarr[0,0].legend(loc='center left', bbox_to_anchor=(4.55, -1.5))

#plt.savefig('returnplot.png')
plt.show()


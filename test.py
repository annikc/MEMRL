import numpy as np 
import pdb 
import bisect as bs 

# basic parameters
T = 1 # trial length in time-steps
n = 30  # number of input units
m = 100 # number of hidden units
l = 6   # number of actions (output)
r = 1   # radius of RF 

# action consequences 
all_actions = ['N','E','S','W','stay','poke']
geometry = {'HOME':np.array([2,2]), 'W2':np.array([0,2]), 'W1':np.array([1,2]), 'S1': np.array([2,1]), 'S2':np.array([2,0]), 'E1':np.array([3,2]), 'E2':np.array([4,2])}
act_select = {'HOME':{'N':'HOME','E':'E1','S':'S1','W':'W1','stay':'HOME','poke':'HOME'},
				'W2':{'N':'W2', 'E':'W1', 'S':'W2', 'W':'W2', 'stay':'W2', 'poke':'W2'}, 
				'W1':{'N':'W1','E':'HOME','S':'W1','W':'W2','stay':'W1','poke':'W1'}, 
				'S1':{'N':'HOME','E':'S1','S':'S2','W':'S1','stay':'S1','poke':'S1'},
				'S2':{'N':'S1','E':'S2','S':'S2','W':'S2','stay':'S2','poke':'S2'},
				'E1':{'N':'E1','E':'E2','S':'E1','W':'HOME','stay':'E1','poke':'E1'},
				'E2':{'N':'E2','E':'E2','S':'E2','W':'E1','stay':'E2','poke':'E2'}}



# dynamic variables
cur_pos = 'HOME'
bias_0 = np.random.normal(0,0.1,(m,1))
bias_1a = np.random.normal(0,0.1,(l,1))
bias_1v = np.random.uniform(0,0.1,(1,1))

# synaptic weights 
W0 = np.random.normal(0,0.1,(m,n))
W1a = np.random.normal(0,0.1,(l,m))
W1v = np.random.uniform(0,0.1,(1,m))

# specify input unit receptive fields
x_centers = np.random.uniform(0,5,(n,1))
y_centers = np.random.uniform(0,3,(n,1))

# define functions 
def sigm(x):
	return 1/(1+np.exp(-x))

def softmax(x):
	return np.exp(x)/(np.sum(np.exp(x)))



# implement a feed-forward pass

# loop over time-steps
for t in range(T):

	# set the input unit activities
	units_in = np.exp(-((geometry[cur_pos][0] - x_centers)**2 + (geometry[cur_pos][1] - y_centers)**2)/(2*r**2))
	
	# prop to hidden layer 
	units_hid = sigm(np.dot(W0, units_in) + bias_0)

	# prop to output layer 
	units_a = softmax(np.dot(W1a, units_hid) + bias_1a)
	print units_a
	unit_v = np.dot(W1v, units_hid) + bias_1v
	print unit_v

	# update position
	cumprobs = np.cumsum(units_a)
	rand = np.random.random()
	selection = bs.bisect(cumprobs,rand)
	cur_action = all_actions[selection]
	print "current position is ", cur_pos, "action is", cur_action
	cur_pos = act_select[cur_pos][cur_action]
	print "new position is", cur_pos


	
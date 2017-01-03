# MDP
PhD Thesis work -- computational model of learning and memory in decision making in reinforcement learning tasks

## Using files (as of Jan 3, 2017)
singletrial.py is currently the most up to date file using the network outlined in MDP_def.py. Currently this is a shallow network with an input layer of 17 units, an action output layer of 6 units, and a single value output unit. The number of layers (and units therein) is given in singletrial where the network object is created (line 70). 

singletrial runs through trials and will plot the accumulated gradients for each step in the last trial in the set. Currently we only look at a set with 1 trial. 

#!/usr/bin/python2

#Shruti's changes: 
#allows user to set environment (instead of using default game) and number of reps (instead of using 10000)
#range object assigned to index_array converted to list so it could be shuffled
#implemented checks for positive integers for _ntrain and _ntest arguments
#tested if index_array is actually shuffled (it hasn't)


#Questions/To-Do: 
#should index_array should be storing shuffled version?
#have to implement check to see if valid game ID from availible game environments
#where should _ntest be used? 
#update to store in bin file instead of HDF5?
#make requirements.txt so all dependencies can be installed automatically? 

import os
import gym
import torch
import torchvision
import numpy as np
import argparse as ap
import tables as tb
from random import shuffle
import pdb

#########################################################################
class OneStepDataset(torch.utils.data.Dataset):
	"""
	A class for datasets to train the one-step prediction model. The
	data is stored in a .hdf5 file in a specified directory.
	"""

	#################################	
	def __init__(self, game, datadir, fname, nsamp=10000, nframes=100, net=None, 
	                           filters=tb.Filters(complib='blosc', complevel=5)):
		"""
		Args:
			game (str)   : the OpenAI game to use
			datadir (str): the directory to store the dataset
			fname (str)  : the name for the data file
			nsamp (int)  : the number of samples to draw
			                                 (default = 10000)
			nframes (int): the number of input frames to store 
			                                   (default = 100)
			net (Network): the network for generating actions
                                      (default = None, i.e. random policy)
		"""
	
		# make sure the data directory exists
		if not os.path.isdir(datadir):
			raise NotADirectoryError("%s is not a directory" %(datadir))

		# create the environment
		self.env = gym.make(game)

		# store the number of frames
		self.nframes = nframes

		# an internal class for the sequence set table
		class sets_table_desc(tb.IsDescription):
			s_t  = tb.UInt8Col(shape=self.env.observation_space.shape+(self.nframes,))
			s_t1 = tb.UInt8Col(shape=self.env.observation_space.shape)
			a    = tb.Int32Col(self.nframes)
			r    = tb.Float64Col(self.nframes)
		self.sets_desc = sets_table_desc
	
		# store the total number of samples and current number
		self.nsamp = nsamp
		self.csamp = 0

		# set the network
		self.net = net

		# store the filters
		self.filters = filters

		# store the name of the file
		self.file_name = '%s/%s_%s.hdf5' %(datadir,game,fname)
	
		# FOR LATER: MAKE IT FLEXIBLE FOR DIFFERENT ACTION SPACES Example:
		#			if type(env.action_space) is gym.spaces.discrete.Discrete:
		#				a = Int32Col(self.nframes)
		#			else:
		#				a = Float32Col(self.nframes)

	#################################	
	def __len__(self):
		"""
		Get the number of training samples that have currently been generated.
		"""
		return self.csamp

	#################################	
	def __getitem__(self, idx):
		"""
		Get a training sample (s_t, s_t1, a, r), return as a tuple of pytorch tensors,
		where:
		  - s_t  = environment state at time t
		  - s_t1 = environment state at time t+1
		  - a    = action at time t
		  - r    = reward at time 

		Args:
		  - idx (int): the index of the sample to retrieve
		"""

		# open the file for reading
		try:
			self.file_handle = tb.open_file(self.file_name, mode='r')
		except FileNotFoundError:
			print("The file %s does not exist. Has it been written to yet?" %(self.file_name))
		except:
			print("Problem opening %s for reading." %(self.file_name))
			
		# pointer to the table holding the training sets
		sets = self.file_handle.root.Training.sets
		
		# get the tuple for return
		rettup = (sets[idx]['s_t'], sets[idx]['s_t1'], sets[idx]['a'], sets[idx]['r'])

		# close the file
		self.file_handle.close()

		return rettup 

	#################################	
	def set_network(self, net):
		"""
		Set the neural network used for generating actions.

		Args:
			net (torch.nn.Module): the network for generating actions
		"""
		self.net = net

	#################################	
	def act(self):
		"""
		Select sets of nframe actions until done.
		"""
		
		# reset the environment
		self.env.reset()
		done = False
	
		# initialize holders
		s_t     = np.empty(self.env.observation_space.shape+(self.nframes,),dtype=np.uint8)
		s_t[:]  = np.nan
		s_t1    = np.empty(self.env.observation_space.shape,dtype=np.uint8)
		s_t1[:] = np.nan
		a       = np.empty((self.nframes,),dtype=np.int32)
		a[:]    = np.nan
		r       = np.empty((self.nframes,),dtype=np.float64)
		r[:]    = np.nan
		
		# reset the environment
		s_t[:,:,:,0] = self.env.reset()
		done = False

		# play for nframes or until done...
		for f in range(self.nframes):
			if self.net is None:
				a[f] = self.env.action_space.sample()
			else:
				a[f] = self.net(s_t)
			if f < (self.nframes-1):
				s_t[:,:,:,f+1], r[f], done, info = self.env.step(a[f])
			elif f == self.nframes-1:
				s_t1[:,:,:], r[f], done, info = self.env.step(a[f])
			if done:
				break

		# get a pointer to the currently writeable table row (for
		current_set         = self.sets.row
		current_set['s_t']  = s_t
		current_set['s_t1'] = s_t1
		current_set['a']    = a
		current_set['r']    = r

		# now actually add to the table
		current_set.append()

		# flush the table buffers (updates description on disk)
		self.sets.flush()
		
	#################################	
	def write(self):
		"""
		Write a training set and close the file.
		"""

		# open the file for writing
		try:
			self.file_handle = tb.open_file(self.file_name, mode='w')
		except PermissionError:
			print("Permission denied when opening %s for writing." %(self.file_name))
		except:
			print("Couldn't open %s for writing." %(self.file_name))

		# set the current number of samples to 0
		self.csamp = 0

		# set the root of the file
		self.file_root   = self.file_handle.root

		# create the HDF5 group where the data will reside
		self.training_group = self.file_handle.create_group(self.file_root, "Training")

		# create the sequence set table
		self.sets = self.file_handle.create_table('/Training', "sets", self.sets_desc, 
											 "training sets table", self.filters, self.nsamp)
		
		# store nsamp samples
		self.csamp = 0
		for s in range(self.nsamp):
			self.act()
			self.csamp += 1
	
		# close the file
		self.file_handle.close()
	
##########################################################################################
class OneStepNetwork(nn.Module):
	"""
	A class for generating a one-step prediction model on a reinforcement learning task.
	The model is always structured as a convolutional network with two layers of 
	convolution/pooling, followed by three fully connected layers, then two inversion
	layers of transpose-convolution/unpooling. The state of the task at time t is fed
	into the input layer, the action and reward are fed into the third fully connected
	layer, and the state at time t+1 is the target at the output layer. The middle
	fully connected layer is intended to be used as a state representation for a RL
	system.

	Appropriate data for this network is obtained with the OneStepDataset class. An 
	instance of this class must be fed into the constructor in order for the network
	to build itself appropriately.
	
	"""
	
	#################################	
	def __init__(self, dataset, convp, poolp, fullp):
		"""
		Args:
			- dataset (OneStepDataset): the dataset object that will be used to
		                               to generate training data for this model
			- convp	(dictionary): the parameters controlling the convolution layers
								   keys/valaues as follows:
									- padding (2-tuple of ints, one entry for each layer, same both dims)
									- rfsize (2-tuple of ints, one entry for each layer, square rfs assumed)
									- stride (2-tuple of ints, one entry for each layer, same both dims)
									- channels (2-tuple of ints, one entry for each layer)
			- poolp	(dictionary): the parameters controlling the pooling layers
								   keys as follows:
									- rfsize (2-tuple of ints, one entry for each layer, square rfs assumed)
									- stride (2-tuple of ints, one entry for each layer, same both dims)
			- fullp	(dictionary): the parameters controlling the fully connected layers
								   keys as follows:
									- nunits (2-tuple of ints, onr entry for each layer)
			- latvp	(dictionary): the parameters controlling the latent variable layers
								   keys as follows:
									- mu (2-tuple of ints, onr entry for each layer)
		"""

		# call the super init
		super(OneStepNetwork, self).__init__()

		# get the shape of the observation and action spaces
		self.input_shape = dataset.env.observation_space.shape
		self.act_shape   = (dataset.env.action_space.n,)

		# determine the shapes of the various output stages
		self.c1shape = (np.floor((self.input_shape[0] + 2*convp['padding'][0] - convp['rfsize'][0])/convp['stride'][0]) + 1,
		                np.floor((self.input_shape[1] + 2*convp['padding'][0] - convp['rfsize'][0])/convp['stride'][0]) + 1,
		                                                                                               convp['channels'][0])
		self.p1shape = (np.floor((self.c1shape[0] - poolp['rfsize'][0])/poolp['stride'][0]) + 1,
		                np.floor((self.c1shape[1] - poolp['rfsize'][0])/convp['stride'][0]) + 1,
		                                                                   convp['channels'][0])
		self.c2shape = (np.floor((self.p1shape[0] + 2*convp['padding'][1] - convp['rfsize'][1])/convp['stride'][1]) + 1,
		                np.floor((self.p1shape[1] + 2*convp['padding'][1] - convp['rfsize'][1])/convp['stride'][1]) + 1,
		                                                                                           convp['channels'][1])
		self.p2shape = (np.floor((self.c2shape[0] - poolp['rfsize'][1])/poolp['stride'][1]) + 1,
		                np.floor((self.c2shape[1] - poolp['rfsize'][1])/convp['stride'][1]) + 1,
		                                                                   convp['channels'][1])

		# build the encoding layers
		self.conv1 = torch.nn.Conv2d(self.input_shape[2], convp['channels'][0], convp['rfsize'][0], 
		                                    stride=convp['stride'][0], padding=convp['padding'][1])
		self.pool1 = torch.nn.MaxPool2d(poolp['rfsize'][0], poolp['stride'][1], return_indices=True)

		self.conv2 = torch.nn.Conv2d(convp['channels'][0], convp['channels'][1], convp['rfsize'][1], 
		                                     stride=convp['stride'][1], padding=convp['padding'][1])
		self.pool2 = torch.nn.MaxPool2d(poolp['rfsize'][1], poolp['stride'][1], return_indices=True)

		self.fc1   = torch.nn.Linear(np.prod(self.p2shape), fullp['nunits'])
		self.mu    = torch.nn.Linear(np.prod(self.p2shape), fullp['nmu'])
		self.logv  = torch.nn.Linear(np.prod(self.p2shape), fullp['nlogv'])

		# build the decoding layers
		self.fc2   = torch.nn.Linear(np.prod(self.p2shape), fullp['nunits'][3])
		# TODO NEED TO CORRECT DECODING PARAMETERS
		self.u1shape = (np.floor((self.input_shape[0] + 2*convp['padding'][0] - convp['rfsize'][0])/convp['stride'][0]) + 1,
		                np.floor((self.input_shape[1] + 2*convp['padding'][0] - convp['rfsize'][0])/convp['stride'][0]) + 1,
		                                                                                               convp['channels'][0])
		self.d1shape = (np.floor((self.c1shape[0] - poolp['rfsize'][0])/poolp['stride'][0]) + 1,
		                np.floor((self.c1shape[1] - poolp['rfsize'][0])/convp['stride'][0]) + 1,
		                                                                   convp['channels'][0])
		self.u2shape = (np.floor((self.p1shape[0] + 2*convp['padding'][1] - convp['rfsize'][1])/convp['stride'][1]) + 1,
		                np.floor((self.p1shape[1] + 2*convp['padding'][1] - convp['rfsize'][1])/convp['stride'][1]) + 1,
		                                                                                           convp['channels'][1])
		self.d2shape = (np.floor((self.c2shape[0] - poolp['rfsize'][1])/poolp['stride'][1]) + 1,
		                np.floor((self.c2shape[1] - poolp['rfsize'][1])/convp['stride'][1]) + 1,
		                                                                   convp['channels'][1])

	def encode(self, x):

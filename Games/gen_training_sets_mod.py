#!/usr/bin/python2

# Shruti's changes:
# allows user to set environment (instead of using default game) and number of reps (instead of using 10000)
# range object assigned to index_array converted to list so it could be shuffled
# implemented checks for positive integers for _ntrain and _ntest arguments
# tested if index_array is actually shuffled (it hasn't)


# Questions/To-Do:
# should index_array should be storing shuffled version?
# have to implement check to see if valid game ID from available game environments
# where should _ntest be used?
# update to store in bin file instead of HDF5?
# make requirements.txt so all dependencies can be installed automatically?

import gym
import argparse
from tables import *
from random import shuffle
import os.path
import numpy as np

# default_game = 'SpaceInvaders-v4'
default_game = 'MsPacman-v4'
# enable compression in PyTables - significantly reduced file size
default_filters = Filters(complib='blosc', complevel=5)
# default_filters = None

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def get_game_info(game, nreps):
    """Gets all the game information from virtual environment written to data table rows

    Args:
        game: the game environment to be working with
        nreps: the number of sets of game frame info

    Returns:
        A two dimensional array containing sets of s_t, a, s_t_1 info
    """

    env = gym.make(game)
    # get the first observation, s(t), initialize the game environment
    s_t = env.reset()
    # initializes list to store all steps
    reps = []
    # action chosen based on random policy
    # TODO: action given by sampling from model free
    for ns in range(nreps):
        # retrieves next state for action and stores into a list
        action_probs = softmax(np.zeros(env.action_space.n))
        #action = env.action_space.sample()
        action = np.random.choice(np.arange(env.action_space.n), 1, p=list(action_probs))

        s_t_1, reward, done, info = env.step(action)
        row = [s_t, action, s_t_1]
        if done:
            # game over'd in some way, reset s(t) and continue data collection
            s_t = env.reset()
        else:
            s_t = s_t_1
        # adds to reps list
        reps.append(row)

    return reps


def gen_training_set(game, nreps):
    """Generate a HDF5 file that contains rows with training data.

    Args:
        game: the game environment to be working with
        nreps: the number of sets of game frame info

    """

    # open the output file, creating it if it does not exist
    #  look into specifying chunk size once established probable size of dataset necessary
    file_handle = open_file('%s.hdf5' % game, mode='w')
    root = file_handle.root
    # create the HDF5 group where the data will reside
    file_handle.create_group(root, "Training")
    # create the sets table
    # first define the Class that will model the table
    # ...N.B. this could also be defined as a dict or numpy dtype

    reps = get_game_info(game, nreps)

    class SetsTableDesc(IsDescription):
        # multidimensional byte array
        # reps[0][0] and reps[0][1] are used to set shape of column to st and st_1
        st = UInt8Col(shape=reps[0][0].shape)
        a = Int32Col()
        st1 = UInt8Col(shape=reps[0][2].shape)

    sets = file_handle.create_table('/Training', "sets", SetsTableDesc, "training sets table", default_filters, nreps)
    print('generating training set for %s\n' % game)

    # pointer to currently writeable table row
    for rep in reps:
        # Retrieves data from array and puts writes into file
        current_set = sets.row
        current_set['st'] = rep[0]
        current_set['a'] = rep[1]
        current_set['st1'] = rep[2]
        # now actually add to the table
        current_set.append()

    # flush the table buffers (updates description on disk)
    sets.flush()
    print('generated %d sets\n' % sets.shape[0])
    # finally, close the file
    file_handle.close()
    # env.render(close=True)


def shuffle_training_set(game):
    # open an existing file
    file_handle = open_file('%s.hdf5' % game, mode='a')
    root = file_handle.root
    # pointer to the table holding the training sets, /Training/sets
    # can slice into like usual in python, e.g. sets[:]['st'] returns all instances of st
    sets = root.Training.sets
    #sets.shape[0] only returns the number of rows
    #range(sets.shape[0]) creates a range from 0-(number of rows-1)
    #need to convert range object to list to iterate over indices
    index_array = list(range(sets.shape[0]))
    shuffle(index_array)
    for i in range(sets.shape[0]):
	    out_row = sets.row
	    rand_index = index_array[i]
	    current_set = sets[rand_index]
	    out_row['st'] = current_set['st']
	    out_row['a'] = current_set['a']
	    out_row['st1'] = current_set['st1']
	    out_row.append()

    # toss the unrandomized rows
    sets.remove_rows(0,len(index_array))
    sets.flush()
    print(sets.shape[0])



def positive_int(value):
    """Checks if the value for the _ntrain or _ntest parameters are positive integers.

    Args:
        value: either the _ntrain or _ntest value passed in from command line

    Returns:
        the input value parsed into an integer
    """
    val = int(value)
    if val <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return val

# other customizable things


def main():
    parser = argparse.ArgumentParser(description='Generate example sets from Atari games to train a CNN')
    parser.add_argument('game_name', help='the Atari game from which sets will be generated')
    parser.add_argument('_ntrain', help='how many training sets to generate', type=positive_int)
    #parser.add_argument('_ntest', help='how many test sets to generate', type=positive_int)
    args = parser.parse_args()
    print(args)

    # makes data sub-folder if it doesn't already exist
    if not os.path.isdir("./data"):
        os.mkdir("data")

    # stores current directory before changing
    prev_dir = os.getcwd()

    # change process to write HDF5 file to data sub-folder
    os.chdir("./data")

    gen_training_set(args.game_name, args._ntrain)
    print('now shuffling the data')
    shuffle_training_set(args.game_name)

    # change back to parent folder
    os.chdir(prev_dir)


if __name__ == "__main__":
        main()

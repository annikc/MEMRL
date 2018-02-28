'''
Take game file name as argument so it only runs for the game you want
'''


import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from tables import *
import os

parser = argparse.ArgumentParser(description='Plot data from game file.')
parser.add_argument('file_name', help='the HDF5 file to read')
args = parser.parse_args()
#if the data folder exists
if os.path.isdir("./data"):
    os.chdir("./data")
    file_handle = open_file(args.file_name, mode='a')
    root = file_handle.root
    sets = root.Training.sets
    index_array = list(range(sets.shape[0]))
    #pulls up st and st+1 images for 5 random rows
    for x in range (0,5):
        random_num = random.randint(0, sets.shape[0]-1)
        print("Round: "+ str(x+1))
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.imshow(sets[random_num]['st'])
        ax2.imshow(sets[random_num]['st1'])
        ax1.set_title('Example # {}'.format(random_num))
        plt.show()
else:
    print("Sorry, no game files found!")

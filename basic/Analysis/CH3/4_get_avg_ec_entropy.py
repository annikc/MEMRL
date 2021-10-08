## uses functions from CH2/compare_ec_pol.py
import gym
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
import matplotlib as mpl

import matplotlib.colors as colors
import matplotlib.cm as cmx

from modules.Utils.gridworld_plotting import plot_pref_pol, plot_polmap
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents.RepresentationLearning.learned_representations import sr, onehot
from Analysis.analysis_utils import analysis_specs,linc_coolwarm,make_env_graph,compute_graph_distance_matrix, LINCLAB_COLS, plot_specs, fade
from random_funcs import make_arrows, get_KLD, get_avg_incidence_of_memories, plot_dist_v_entropy, plot_avg_entropy, plot_all_entropy,\
    plot_entropy_hisogram, test_avg_POLmaps, get_mem_maps, plot_memory_maps, plot_avg_laplace
from scipy.special import rel_entr
from scipy.stats import entropy


# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'ec_track_pols.csv')
groups_to_split = ['env_name','representation','EC_cache_limit']
gb = df.groupby(groups_to_split)["save_id"]

rep_dict = {'analytic successor':sr, 'onehot':onehot}
cache_limits = analysis_specs['cache_limits']
env_name = 'gridworld:gridworld-v41'
reps = ['onehot', 'analytic successor']
color_map = {'onehot':LINCLAB_COLS['blue'], 'analytic successor':LINCLAB_COLS['red']}
rep = reps[1]

env = gym.make(env_name)
plt.close()

state_reps, _, __, ___ = rep_dict[rep](env)


G = make_env_graph(env)
gd = compute_graph_distance_matrix(G,env)
dist_in_state_space = gd[env.twoD2oneD((14,14))]


env_name = 'gridworld:gridworld-v41'
rep = 'analytic successor'
pct = 50

plot_entropy_hisogram(env_name, ['analytic successor','onehot'],pcts_to_plot=[100,75,50,25])
plot_all_entropy(env_name, ['analytic successor','onehot'],pcts_to_plot=[100,75,50,25],type='violin')
plot_avg_laplace(env_name,pcts_to_plot=[100,75,50,25],reps_to_plot=['analytic successor','onehot'])
test_avg_POLmaps(env_name, rep)





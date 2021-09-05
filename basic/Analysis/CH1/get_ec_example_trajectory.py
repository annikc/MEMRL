import gym
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import laplace
from modules.Utils.gridworld_plotting import plot_pref_pol, plot_polmap
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents.RepresentationLearning.learned_representations import sr, onehot
from Analysis.analysis_utils import analysis_specs,linc_coolwarm,make_env_graph,compute_graph_distance_matrix, LINCLAB_COLS
from scipy.special import rel_entr
from scipy.stats import entropy
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection as LC

# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'ec_track_pols.csv')
groups_to_split = ['env_name','representation','EC_cache_limit']
gb = df.groupby(groups_to_split)["save_id"]

env_name = 'gridworld:gridworld-v41'
env = gym.make(env_name)
plt.close()
reps_to_plot = ['onehot','analytic successor']
pcts_to_plot = [100,75,50,25]
cache_limits = analysis_specs['cache_limits']

#save_processed_data(env_name,cache_limits,pcts_to_plot,reps_to_plot,save='xy')

def sample_from_ec_pol(state_reps, ec_dict,**kwargs):
    blank_mem = Memory(cache_limit=400, entry_size=4)
    blank_mem.cache_list = ec_dict
    start_state = kwargs.get('start_state',np.random.choice(list(state_reps.keys())))
    trajectory = []
    env.set_state(start_state)
    state = start_state
    trajectory.append(env.oneD2twoD(state))
    for i in range(250):
        policy = blank_mem.recall_mem(state_reps[state])
        action = np.random.choice(np.arange(4),p=policy)
        next_state, reward, done, info = env.step(action)
        state = next_state
        trajectory.append(env.oneD2twoD(state))
        if reward ==10.:
            break
    return trajectory


def plot_example_trajectories(env,reps_to_plot, pcts_to_plot):
    run_colors = [LINCLAB_COLS['red'],LINCLAB_COLS['blue'],LINCLAB_COLS['green']]
    run_labels = {'onehot': 'Unstructured','analytic successor':'Structured'}
    start_locations = [105,114,285]
    fig, ax = plt.subplots(len(reps_to_plot),len(pcts_to_plot))
    for r,rep in enumerate(reps_to_plot):
        for p, pct in enumerate(pcts_to_plot):

            run_id = list(gb.get_group((env_name,rep,cache_limits[env_name][pct])))[-1]
            print(rep, pct, run_id)

            with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
                data = pickle.load(f)

            rep_dict = {'analytic successor': sr, 'onehot':onehot}
            state_reps, _, __, ___ = rep_dict[rep](env)
            if rep == 'analytic successor':
                for s1 in env.obstacle:
                    state_reps.pop(s1)

            all_the_eps = data['ec_dicts']
            rwd_colrow=(14,14)
            rect = plt.Rectangle(rwd_colrow, 1, 1, facecolor='gray', alpha=0.3,edgecolor='k')
            ax[r,p].pcolor(env.grid,cmap='bone_r',edgecolors='k', linewidths=0.1)
            ax[r,p].axis(xmin=0, xmax=20, ymin=0,ymax=20)
            ax[r,p].set_aspect('equal')
            ax[r,p].add_patch(rect)
            ax[r,p].set_yticks([])
            ax[r,p].get_xaxis().set_visible(False)
            #ax[r,p].get_yaxis().set_visible(False)
            ax[r,p].invert_yaxis()

            for xx in range(len(run_colors)):
                trajectory = sample_from_ec_pol(state_reps,all_the_eps[-1],start_state=start_locations[xx])
                print(trajectory)

                lines = []
                for i in range(len(trajectory)-1):
                    # need to reverse r-c from state2d to be x-y for line collection
                    state = trajectory[i]
                    next_state = trajectory[i+1]
                    lines.append([(state[1]+0.5,state[0]+0.5),(next_state[1]+0.5,next_state[0]+0.5)])
                ax[r,p].add_patch(plt.Circle(lines[0][0],radius=0.3,color=run_colors[xx]))
                lc = LC(lines, colors=run_colors[xx], linestyle="--",linewidths=0.85,alpha=0.75)
                ax[r,p].add_collection(lc)
        ax[r,0].set_ylabel(f"{run_labels[rep]}")
    plt.savefig('../figures/CH2/example_trajectories1.svg')
    plt.show()
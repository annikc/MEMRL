import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pickle
import sys
import gym

sys.path.append('../../modules/')
from Analysis.analysis_utils import get_avg_std, get_grids

# import csv data summary
parent_path = '../../Data/'
cos_df = pd.read_csv(parent_path+'throttled_ec_allreps_chebyshev.csv')

# parse data by relevant columns
gb_cos = cos_df.groupby(['env_name','representation','EC_cache_limit'])["save_id"]

# get cache limit sizes for the restriction conditions -- different for each environment
cache_limits = {'gridworld:gridworld-v11':{100:400, 75:300, 50:200, 25:100},
                'gridworld:gridworld-v31':{100:365, 75:273, 50:182, 25:91},
                'gridworld:gridworld-v41':{100:384, 75:288, 50:192, 25:96},
                'gridworld:gridworld-v51':{100:286, 75:214, 50:143, 25:71}}

# theoretical optimal performance (average number of steps penalization to get to reward state)
avg_max_rwd = {'11':9.87, '31':9.85, '41':9.84, '51':9.86}

labels_for_plot = {'analytic successor':'SR', 'onehot':'onehot', 'random':'random','place_cell':'PC','conv_latents':'latent'}

convert_rep_to_color = {'analytic successor':'C0',
                        'onehot':'C1',
                        'random':'C2',
                        'place_cell':'C3',
                        'conv_latents':'C4'}

envs_to_plot = ['gridworld:gridworld-v11']
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['random', 'onehot','place_cell','analytic successor',]#, 'conv_latents'] # df.representation.unique()
rep_labels = [labels_for_plot[x] for x in reps_to_plot]
env = envs_to_plot[0]
tmp_env_obj = gym.make(env)
plt.close()
e_grid = tmp_env_obj.grid

def plot_throttled_performance():
    fig, ax = plt.subplots(1,2,figsize=(15,3),gridspec_kw={'width_ratios': [1, 3]})
    if env[-2:] == '51':
        rwd_colrow= (16,9)
    else:
        rwd_colrow=(14,14)

    rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
    ax[0].pcolor(e_grid,cmap='bone_r',edgecolors='k', linewidths=0.1)
    ax[0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
    ax[0].set_aspect('equal')
    ax[0].add_patch(rect)
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].invert_yaxis()

    for r, rep in enumerate(reps_to_plot):
        # ax[0,j] plot average performance with error bars
        # ax[1,j] plot variance of differnt rep types
        for j, pct in enumerate(pcts_to_plot):
            v_list = list(gb_cos.get_group((env, rep, cache_limits[env][pct])))
            print(env, rep, pct, v_list)
            avg_, std_ = get_avg_std(v_list,cutoff=5000, smoothing=100)
            avg_cos, std_cos = np.mean(avg_), np.mean(std_)
            ax[1].bar(2*r+0.35*j,avg_cos, yerr=std_cos,width=0.35, color=convert_rep_to_color[rep], alpha=pct/100)

            ax[1].set_ylim([0,10.5])

            ax[1].set_xticks(np.arange(0,2*len(reps_to_plot),2)+0.5)
            ax[1].set_xticklabels(rep_labels,rotation=0)
    #ax[1].axhline(y=avg_max_rwd[env[-2:]], color='r', linestyle='--')
    p100 = mpatches.Patch(color='gray',alpha=1, label='100')
    p75 = mpatches.Patch(color='gray',alpha=0.75, label='75')
    p50 = mpatches.Patch(color='gray',alpha=.5, label='50')
    p25 = mpatches.Patch(color='gray',alpha=.25, label='25')
    plt.legend(handles=[p100,p75,p50,p25], bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1, title='Episodic Memory\nCapacity (%)')

    plt.show()


plot_throttled_performance()
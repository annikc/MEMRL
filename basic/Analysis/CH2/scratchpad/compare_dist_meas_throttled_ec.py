import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import sys
import gym

sys.path.append('../../modules/')
from Analysis.analysis_utils import get_avg_std, get_grids
from Utils import running_mean as rm

# import csv data summary
parent_path = '../../Data/'
cos_df = pd.read_csv(parent_path+'throttled_ec_allreps_cosine.csv')
euc_df = pd.read_csv(parent_path+'throttled_ec_allreps_euclidean.csv')

gb_cos = cos_df.groupby(['env_name','representation','EC_cache_limit'])["save_id"]#.apply(list)
gb_euc = euc_df.groupby(['env_name','representation','EC_cache_limit'])["save_id"]


env_string = 'gridworld:gridworld-v31'
cache_limits = {'gridworld:gridworld-v11':{100:400, 75:300, 50:200, 25:100},
                'gridworld:gridworld-v31':{100:365, 75:273, 50:182, 25:91},
                'gridworld:gridworld-v41':{100:384, 75:288, 50:192, 25:96},
                'gridworld:gridworld-v51':{100:286, 75:214, 50:143, 25:71}}


convert_rep_to_color = {'analytic successor':'C0', 'onehot':'C1', 'random':'C2','place_cell':'C3','conv_latents':'k'}
labels_for_plot = {'analytic successor':'SR', 'onehot':'onehot', 'random':'random','place_cell':'PC','conv_latents':'latent'}

# plots
# environments = diff figures
# columns = restricted size
# rows = performance and average variance in signal
envs_to_plot = ['gridworld:gridworld-v41']
pcts_to_plot = [25,50,75,100]
reps_to_plot = ['analytic successor','place_cell', 'random', 'onehot']#, 'conv_latents'] # df.representation.unique()
rep_labels = [labels_for_plot[x] for x in reps_to_plot]
env = envs_to_plot[0]
tmp_env_obj = gym.make(env)
plt.close()
e_grid = tmp_env_obj.grid

def plot_throttled_performance():
    fig, ax = plt.subplots(2,len(pcts_to_plot)+1)
    if env[-2:] == '51':
        rwd_colrow= (16,9)
    else:
        rwd_colrow=(14,14)

    rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
    ax[0,0].pcolor(e_grid,cmap='bone_r',edgecolors='k', linewidths=0.1)
    ax[0,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
    ax[0,0].set_aspect('equal')
    ax[0,0].add_patch(rect)
    ax[0,0].get_xaxis().set_visible(False)
    ax[0,0].get_yaxis().set_visible(False)
    ax[0,0].invert_yaxis()

    for r, rep in enumerate(reps_to_plot):
        avg_rwd_var =[[],[]]
        # ax[0,j] plot average performance with error bars
        # ax[1,j] plot variance of differnt rep types
        for j, pct in enumerate(pcts_to_plot):
            v_list = list(gb_cos.get_group((env, rep, cache_limits[env][pct])))
            print(env, rep, pct, v_list)
            avg_, std_ = get_avg_std(v_list,cutoff=5000, smoothing=100)
            avg_cos, std_cos = np.mean(avg_), np.mean(std_)
            ax[0,j+1].bar(2*r+0.5,avg_cos, yerr=std_cos,width=0.5, color=convert_rep_to_color[rep], alpha=0.5)

            v_list = list(gb_euc.get_group((env, rep, cache_limits[env][pct])))
            print(env, rep, pct, v_list)
            avg_, std_ = get_avg_std(v_list,cutoff=5000, smoothing=100)
            avg_euc, std_euc = np.mean(avg_), np.mean(std_)
            ax[0,j+1].bar(2*r,avg_euc, yerr=std_euc,width=0.5,color=convert_rep_to_color[rep], alpha=1)

            ax[0,j+1].set_ylim([0,15])
            ax[0,j+1].set_title(f'{pct}')
            ax[0,j+1].set_xticks([0,2,4,6])
            ax[0,j+1].set_xticklabels(rep_labels,rotation=315)

            ax[1,j+1].scatter(avg_cos, std_cos, color=convert_rep_to_color[rep], alpha=0.5)
            ax[1,j+1].scatter(avg_euc, std_euc, color=convert_rep_to_color[rep], alpha=1)
            ax[1,j+1].set_xlim([0,10])
            ax[1,j+1].set_ylim([0,6])

    ax[1,0].axis('off')
    ax[1,-1].legend(bbox_to_anchor =(1.1,-.05), ncol=len(reps_to_plot))

    plt.show()

def plot_3d_avg_var():
    for i, env in enumerate(envs_to_plot):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for r, rep in enumerate(reps_to_plot):
            avg_rwd_var =[[],[]]
            # ax[0,j] plot performance curves
            # ax[1,j] plot variance of differnt rep types
            for j, pct in enumerate(pcts_to_plot):
                v_list = list(gb.get_group((env, rep, cache_limits[env][pct])))
                print(env, rep, pct, v_list)
                avg_, std_ = get_avg_std(v_list,cutoff=5000, smoothing=100)
                avg_rwd_var[0].append(np.mean(avg_))
                avg_rwd_var[1].append(np.mean(std_))

            print(avg_rwd_var[0],avg_rwd_var[1], 'avg vs std')
            ax.plot3D(pcts_to_plot,avg_rwd_var[1], avg_rwd_var[0],'o-',color=convert_rep_to_color[rep])
    plt.show()

plot_throttled_performance()
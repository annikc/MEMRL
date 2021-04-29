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
df = pd.read_csv(parent_path+'throttled_ec_allreps_cosine.csv')

gb = df.groupby(['env_name','representation','EC_cache_limit'])["save_id"]#.apply(list)

for key, item in gb:
    print(key, list(item))

env_string = 'gridworld:gridworld-v31'
rep_string = 'random'
cache_pct  = 100
cache_limits = {'gridworld:gridworld-v11':{100:400, 75:300, 50:200, 25:100},
                'gridworld:gridworld-v31':{100:365, 75:273, 50:182, 25:91},
                'gridworld:gridworld-v41':{100:384, 75:288, 50:192, 25:96},
                'gridworld:gridworld-v51':{100:286, 75:214, 50:143, 25:71}}
cache_size_for_env = cache_limits[env_string][cache_pct]
print(cache_size_for_env)

v_list = list(gb.get_group((env_string, rep_string, cache_size_for_env)))
#grids = get_grids([11,31,41,51])

convert_rep_to_color = {'analytic successor':'C0', 'onehot':'C1', 'random':'C2','state-centred pc f0.05':'C3','conv_latents':'k'}
labels_for_plot = {'analytic successor':'SR', 'onehot':'onehot', 'random':'random','state-centred pc f0.05':'PC','conv_latents':'latent'}

# plots
# environments = diff figures
# columns = restricted size
# rows = performance and average variance in signal
envs_to_plot = ['gridworld:gridworld-v51']
pcts_to_plot = [25,50,75,100]
reps_to_plot = ['analytic successor','state-centred pc f0.05', 'random', 'onehot']#, 'conv_latents'] # df.representation.unique()
print(reps_to_plot)
def plot_throttled_performance():
    for i, env in enumerate(envs_to_plot):
        fig, ax = plt.subplots(2,len(pcts_to_plot)+1)
        tmp_env_obj = gym.make(env)
        e_grid = tmp_env_obj.grid
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
            # ax[0,j] plot performance curves
            # ax[1,j] plot variance of differnt rep types
            for j, pct in enumerate(pcts_to_plot):
                v_list = list(gb.get_group((env, rep, cache_limits[env][pct])))
                print(env, rep, pct, v_list)
                avg_, std_ = get_avg_std(v_list,cutoff=5000, smoothing=100)
                ax[0,j+1].plot(avg_, label=f'{labels_for_plot[rep]}',color=convert_rep_to_color[rep])
                ax[0,j+1].fill_between(np.arange(len(avg_)),avg_-std_, avg_+std_, color=convert_rep_to_color[rep],alpha=0.3)
                ax[0,j+1].set_ylim([-4,12])
                ax[0,j+1].set_title(f'{pct}')
                ax[1,j+1].bar(r, np.mean(std_),label=f'{labels_for_plot[rep]}', color=convert_rep_to_color[rep])
                ax[1,j+1].set_ylim([0,6])
                avg_rwd_var[0].append(np.mean(avg_))
                avg_rwd_var[1].append(np.mean(std_))

            print(avg_rwd_var[0],avg_rwd_var[1], 'avg vs std')
            ax[1,0].plot(avg_rwd_var[1], avg_rwd_var[0],'o-',color=convert_rep_to_color[rep])
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

plot_3d_avg_var()
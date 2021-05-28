import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import gym

sys.path.append('../../modules/')
from Analysis.analysis_utils import get_avg_std

# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'throttled_ec_allreps_chebyshev.csv')

gb = df.groupby(['env_name','representation','EC_cache_limit'])["save_id"]

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
                        'place_cell':'C4',
                        'conv_latents':'C3'}

envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['conv_latents','place_cell', 'analytic successor'] # df.representation.unique()
rep_labels = [labels_for_plot[x] for x in reps_to_plot]
grids=[]
for env in envs_to_plot:
    tmp_env_obj = gym.make(env)
    plt.close()
    grids.append(tmp_env_obj.grid)

def plot_throttled_performance(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,save=False):
    fig, ax = plt.subplots(len(envs_to_plot),2,figsize=(12,3*len(envs_to_plot)), sharex='col', gridspec_kw={'width_ratios': [1, 2]})
    for i, env in enumerate(envs_to_plot):
        if env[-2:] == '51':
            rwd_colrow= (16,9)
        else:
            rwd_colrow=(14,14)

        rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
        ax[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[i,0].set_aspect('equal')
        ax[i,0].add_patch(rect)
        ax[i,0].get_xaxis().set_visible(False)
        ax[i,0].get_yaxis().set_visible(False)
        ax[i,0].invert_yaxis()

        for r, rep in enumerate(reps_to_plot):
            # ax[0,j] plot average performance with error bars
            # ax[1,j] plot variance of differnt rep types
            for j, pct in enumerate(pcts_to_plot):
                v_list = list(gb.get_group((env, rep, cache_limits[env][pct])))
                print(env, rep, pct, v_list)
                normalization_factor = avg_max_rwd[env[-2:]]
                avg_, std_ = get_avg_std(v_list,normalization_factor=normalization_factor,cutoff=5000, smoothing=100)
                avg_cos, std_cos = np.mean(avg_), np.mean(std_)
                ax[i,1].bar(2*r+0.35*j,avg_cos, yerr=std_cos,width=0.35,color=convert_rep_to_color[rep], alpha=pct/100)

        ax[i,1].set_ylim([0,1.2])
        ax[i,1].set_yticks(np.arange(0,1.5,0.25))
        ax[i,1].set_yticklabels([0,'',50,'',100,''])
        ax[i,1].set_ylabel('% Optimal Performance')

    ax[i,1].set_xticks(np.arange(0,2*len(reps_to_plot),2)+0.5)
    ax[i,1].set_xticklabels(rep_labels,rotation=0)

    #ax[1].axhline(y=avg_max_rwd[env[-2:]], color='r', linestyle='--')
    p100 = mpatches.Patch(color='gray',alpha=1, label='100')
    p75 = mpatches.Patch(color='gray',alpha=0.75, label='75')
    p50 = mpatches.Patch(color='gray',alpha=.5, label='50')
    p25 = mpatches.Patch(color='gray',alpha=.25, label='25')
    #plt.legend(handles=[p100,p75,p50,p25], bbox_to_anchor=(0.5, len(envs_to_plot)*1.1), loc='lower center', ncol=4, title='Episodic Memory Capacity (%)')
    ax[0,1].set_title("Limited Capacity with Structured Representations")
    if save:
        format = 'svg'
        plt.savefig(f'../figures/CH2/forgetting_structured.{format}', format=format)
    plt.show()


plot_throttled_performance(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,save=True)
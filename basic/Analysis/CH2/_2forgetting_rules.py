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
oldest_df = pd.read_csv(parent_path+'throttled_ec_allreps_chebyshev.csv')
random_df = pd.read_csv(parent_path+'random_forget_ec.csv')
LRA_df = pd.read_csv(parent_path+'forget_least_recently_accessed_mem.csv')

gb_old = oldest_df.groupby(['env_name','representation','EC_cache_limit'])["save_id"]
gb_rnd = random_df.groupby(['env_name','representation','EC_cache_limit'])["save_id"]
gb_lra = LRA_df.groupby(['env_name','representation','EC_cache_limit'])["save_id"]

gbs = [gb_rnd, gb_old]#, gb_lra]

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



def plot_throttled_performance(save=False):
    fig, ax = plt.subplots(len(reps_to_plot),2,figsize=(10,len(reps_to_plot)*2), sharey='col',sharex='col',gridspec_kw={'width_ratios': [1, 3]})
    width=0.45
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

    norm = avg_max_rwd[env[-2:]]
    for r, rep in enumerate(reps_to_plot):
        if r ==0:
            pass
        else:
            ax[r,0].axis('off')
        # ax[0,j] plot average performance with error bars
        # ax[1,j] plot variance of differnt rep types
        for j, pct in enumerate(pcts_to_plot):
            for g, gb in enumerate(gbs):
                v_list = list(gb.get_group((env, rep, cache_limits[env][pct])))
                print(env, rep, pct, v_list)
                avg_, std_ = get_avg_std(v_list,cutoff=5000, smoothing=100,normalization_factor=norm)
                avg_cos, std_cos = np.mean(avg_), np.mean(std_)
                if g ==0:
                    ax[r,1].bar(j+width*g,avg_cos, yerr=std_cos,width=width, edgecolor=convert_rep_to_color[rep],fill=False,hatch='//', alpha=pct/100)
                else:
                    ax[r,1].bar(j+width*g,avg_cos, yerr=std_cos,width=width, color=convert_rep_to_color[rep], alpha=pct/100)

        ax[r,1].set_ylim([0,1.2])
        ax[r,1].set_yticks(np.arange(0,1.5,0.25))
        right = 1
        top = 0.98
        ax[r,1].text(right, top, f'{labels_for_plot[rep]}', horizontalalignment='right', verticalalignment='top', transform=ax[r,1].transAxes)
        ax[r,1].set_yticklabels([0,'',50,'',100,''])
        ax[r,1].set_ylabel(f'Performance \n(% Optimal)')

    ax[r,1].set_xticks(np.arange(len(pcts_to_plot))+(width/2))
    ax[r,1].set_xticklabels(pcts_to_plot)
    ax[r,1].set_xlabel('Memory Capacity (%)')

    p_rand = mpatches.Patch(fill=False,edgecolor='gray',alpha=1, hatch='///',label='Random Entry')
    p_old = mpatches.Patch(color='gray',alpha=1, label='Oldest Entry')
    plt.legend(handles=[p_rand,p_old], bbox_to_anchor=(0.5, len(reps_to_plot)*1.16), loc='lower center', ncol=2, title='Forgetting Rule')
    if save:
        format = 'svg'
        plt.savefig(f'../figures/CH2/compare_rand_forgetting_{version}.{format}', format=format)
    plt.show()


#plot_throttled_performance(save=True)

def compare_against_random(env, e_grid, gb, gb_rf, reps_to_plot, pcts_to_plot, save=False):
    fig, ax = plt.subplots(len(reps_to_plot),len(pcts_to_plot)+1,figsize=(15,6), sharex='col')
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

    ax[1,0].axis('off')

    norm = avg_max_rwd[env[-2:]]

    for r, rep in enumerate(reps_to_plot):
        # ax[0,j] plot average performance with error bars
        # ax[1,j] plot variance of differnt rep types
        for j, pct in enumerate(pcts_to_plot):
            v_list = list(gb.get_group((env, rep, cache_limits[env][pct])))
            print(env, rep, pct, v_list)
            avg_, std_ = get_avg_std(v_list,cutoff=5000, smoothing=100,normalization_factor=norm)
            avg_cos, std_cos = np.mean(avg_), np.mean(std_)
            ax[r,j+1].bar([0],avg_cos, yerr=std_cos,width=0.35, color=convert_rep_to_color[rep], alpha=pct/100)

            v_list = list(gb_rf.get_group((env, rep, cache_limits[env][pct])))
            print(env, rep, pct, v_list)
            avg_, std_ = get_avg_std(v_list,cutoff=5000, smoothing=100,normalization_factor=norm)
            avg_cos, std_cos = np.mean(avg_), np.mean(std_)
            ax[r,j+1].bar([0.5],avg_cos, yerr=std_cos,width=0.35, color=convert_rep_to_color[rep], alpha=pct/100)

    #ax[0,1].set_ylim([0,12])
    #ax[1,1].set_ylim([0,12])
    #ax[0,1].set_xticks(np.arange(0,2*len(reps_to_plot),2)+0.5)
    #ax[0,1].set_xticklabels(rep_labels,rotation=0)

    #ax[1,1].set_xticks(np.arange(0,2*len(reps_to_plot),2)+0.5)
    #ax[1,1].set_xticklabels(rep_labels,rotation=0)

    #ax[0,1].set_ylabel('Average Reward per Trial')
    #ax[1,1].set_ylabel('Average Reward per Trial')
    #ax[0,1].set_title('Forget Oldest Entry')
    #ax[1,1].set_title('Forget Random Entry')
    #ax[1].axhline(y=avg_max_rwd[env[-2:]], color='r', linestyle='--')
    p100 = mpatches.Patch(color='gray',alpha=1, label='100')
    p75 = mpatches.Patch(color='gray',alpha=0.75, label='75')
    p50 = mpatches.Patch(color='gray',alpha=.5, label='50')
    p25 = mpatches.Patch(color='gray',alpha=.25, label='25')
    plt.legend(handles=[p100,p75,p50,p25], bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1, title='Episodic Memory\nCapacity (%)')
    if save:
        format = 'svg'
        plt.savefig(f'../figures/CH2/compare_rand_forgetting_{version}.{format}', format=format)
    plt.show()


version = 5
envs_to_plot = [f'gridworld:gridworld-v{version}1']
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['random', 'onehot','conv_latents','place_cell','analytic successor'] # df.representation.unique()
rep_labels = [labels_for_plot[x] for x in reps_to_plot]
env = envs_to_plot[0]
tmp_env_obj = gym.make(env)
plt.close()
e_grid = tmp_env_obj.grid
#compare_against_random(env, e_grid, gb_old, gb_rnd, reps_to_plot, pcts_to_plot, save=False)
plot_throttled_performance(save=True)
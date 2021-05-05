## plot effect of cache limit size on episodic control
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import gym
import pandas as pd

sys.path.append('../../modules/')
from Analysis.analysis_utils import get_detailed_avg_std, get_id_dict, get_grids, plot_each
from modules.Utils import running_mean as rm

data_dir = '../../Data/'
df = pd.read_csv(data_dir+'ec_throttled_latents_emptyhead.csv')
ref = pd.read_csv(data_dir+'ec_empty_test.csv')

envs = df.env_name.unique()
size = df.EC_cache_limit.unique()
print('#####', )
ref_dict = get_id_dict(ref,reps=['conv_latents'])


cache_limits = {'gridworld:gridworld-v11':{1:400, 0.75:300, 0.5:200, 0.25:100},
                'gridworld:gridworld-v31':{1:365, 0.75:273, 0.5:182, 0.25:91},
                'gridworld:gridworld-v41':{1:384, 0.75:288, 0.5:192, 0.25:96},
                'gridworld:gridworld-v51':{1:286, 0.75:214, 0.5:143, 0.25:71}}
def get_cache_size_id_dict(df):
    master_dict = {}
    envs = df.env_name.unique()
    cache_size = df.EC_cache_limit.unique()
    for env in envs:
        master_dict[env] = {}
        for size in cache_limits[env]:
            print(size)
            id_list = list(df.loc[(df['env_name']==env)
             & (df['EC_cache_limit']==cache_limits[env][size])]['save_id'])

            master_dict[env][size]=id_list
    return master_dict

master_dict = get_cache_size_id_dict(df)
for k in ref_dict.keys():
    master_dict[k][1] = ref_dict[k]['conv_latents']
    print(k, master_dict[k].keys())


grids = get_grids([x[-2:] for x in envs])

convert_rep_to_color = {0.25:'C2', 0.5:'C9', 0.75:'C4', 1:'C3'}

def plot_all(save=False, cutoff=25000, smoothing=5000):
    fig, axs = plt.subplots(4, 4, figsize=(15,10), sharex='col')
    for ind, name in enumerate(envs):
        if name[-2:] == '51':
            rwd_colrow= (16,9)
        else:
            rwd_colrow=(14,14)

        rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
        axs[ind,0].pcolor(grids[ind],cmap='bone_r',edgecolors='k', linewidths=0.1)
        axs[ind,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        axs[ind,0].set_aspect('equal')
        axs[ind,0].add_patch(rect)
        axs[ind,0].get_xaxis().set_visible(False)
        axs[ind,0].get_yaxis().set_visible(False)
        axs[ind,0].invert_yaxis()



        for jnd, cache_size in enumerate(master_dict[name].keys()):
            print(f'cachesize {cache_size} for env {name}')
            v_list = master_dict[name][cache_size]
            avg_, std_, a_s, s_s = get_detailed_avg_std(v_list,cutoff=cutoff, smoothing=smoothing)
            axs[ind,1].plot(avg_, label=f'{cache_size}', color=convert_rep_to_color[cache_size]) # label=f'n={len(v_list)}'
            axs[ind,1].fill_between(np.arange(len(avg_)),avg_-std_, avg_+std_, alpha=0.1,color=convert_rep_to_color[cache_size])
            axs[ind,1].set_ylim([-4,12])
            axs[ind,2].bar(jnd, np.mean(std_), color=convert_rep_to_color[cache_size])

            for xx in range(len(a_s)):
                axs[ind,3].scatter(a_s[xx],s_s[xx], color=convert_rep_to_color[cache_size],alpha=0.2)
            axs[ind,3].scatter(np.mean(avg_),np.mean(std_), color=convert_rep_to_color[cache_size])
        axs[ind,2].set_ylim([0,6])
        axs[ind,3].set_ylim([0,1])
        axs[ind,3].set_xlim([9,10])
        #axs[ind,1].legend(loc=0)
        if ind == len(envs)-1:
            axs[ind,1].set_xlabel('Episodes')
            #axs[ind,1].set_ylabel('Cumulative \nReward')
    axs[0,1].legend(loc='upper center', ncol=4, bbox_to_anchor=(0.2,1.2))
    if save:
        plt.savefig('../figures/CH1/ec_throttled1.svg',format='svg')

    plt.show()

plot_all(cutoff=5000,smoothing=50,save=True)

#plot_each(master_dict[env][rep], data_dir)
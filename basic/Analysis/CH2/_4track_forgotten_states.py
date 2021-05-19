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
df = pd.read_csv(parent_path+'track_forgotten_states.csv')
#df = pd.read_csv(parent_path+'forget_least_recently_accessed_mem.csv')
# parse data by relevant columns
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
                        'place_cell':'C3',
                        'conv_latents':'C4'}

envs_to_plot = ['gridworld:gridworld-v41']
pcts_to_plot = [75,50,25]
reps_to_plot = ['random', 'onehot','conv_latents','place_cell','analytic successor',]#, 'conv_latents'] # df.representation.unique()
rep_labels = [labels_for_plot[x] for x in reps_to_plot]
env = envs_to_plot[0]
tmp_env_obj = gym.make(env)
plt.close()
e_grid = tmp_env_obj.grid

def plot_forgetting_distr():
    fig, ax = plt.subplots(len(pcts_to_plot),len(reps_to_plot),figsize=(15,9),sharex='col',sharey='col')
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.1, hspace=0.05)
    maxes = [10,50,100]
    if env[-2:] == '51':
        rwd_colrow= (16,9)
    else:
        rwd_colrow=(14,14)
    cb_axes = []
    for p, pct in enumerate(pcts_to_plot):
        for r, rep in enumerate(reps_to_plot):
            run_id = list(gb.get_group((env,rep,cache_limits[env][pct])))[0]
            with open(parent_path+f'results/{run_id}_data.p', 'rb') as f:
                data = pickle.load(f)['forgotten_states']
            barriers = np.where(e_grid==1)
            for k in range(len(barriers[0])):
                row,col = barriers[0][k], barriers[1][k]
                data[row,col] =np.nan
            im = ax[p,r].imshow(data, vmin=0, vmax=maxes[p])
            ax[p,r].set_aspect('equal')
            ax[p,r].add_patch(plt.Rectangle(np.add(rwd_colrow,(-0.5,-0.5)),1,1,fill=False,edgecolor='w'))
            ax[p,r].set_yticklabels([])
            ax[p,r].set_xticklabels([])
            #ax[p,r].get_yaxis().set_visible(False)
            if p==0:
                ax[p,r].set_title(f'{labels_for_plot[rep]}')
            if r==0:
                ax[p,r].set_ylabel(f'{pct}')
        pos = ax[p,r].get_position()
        print(pos)
        cb_axes.append(fig.add_axes([0.81, pos.y0, 0.02, 0.2197])) # left, bottom, width, height
        cbar = fig.colorbar(im, cax=cb_axes[-1])




    plt.show()

plot_forgetting_distr()

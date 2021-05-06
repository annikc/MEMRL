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

envs_to_plot = ['gridworld:gridworld-v51']
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['random', 'onehot','conv_latents','place_cell','analytic successor',]#, 'conv_latents'] # df.representation.unique()
rep_labels = [labels_for_plot[x] for x in reps_to_plot]
env = envs_to_plot[0]
tmp_env_obj = gym.make(env)
plt.close()
e_grid = tmp_env_obj.grid

def plot_forgetting_distr():
    fig, ax = plt.subplots(len(reps_to_plot),len(pcts_to_plot)+1,figsize=(15,3),sharex='col',sharey='col')
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
        if r ==0:
            pass
        else:
            ax[r,0].axis('off')
        for p, pct in enumerate(pcts_to_plot):
            run_id = list(gb.get_group((env,rep,cache_limits[env][pct])))[0]
            with open(parent_path+f'results/{run_id}_data.p', 'rb') as f:
                data = pickle.load(f)['forgotten_states']
            a=ax[r,p+1].imshow(data/1000)

            ax[r,p+1].set_aspect('equal')
            ax[r,p+1].add_patch(plt.Rectangle(np.add(rwd_colrow,(-0.5,-0.5)),1,1,fill=False,edgecolor='w'))
            #ax[r,p+1].invert_yaxis()
            if r==0:
                ax[r,p+1].set_title(f'{pct}')
            if p == 0:
                ax[r,p+1].set_ylabel(f'{labels_for_plot[rep]}')
    plt.show()

plot_forgetting_distr()

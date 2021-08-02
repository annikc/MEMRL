import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import pickle
import pandas as pd
from Analysis.analysis_utils import plot_specs, analysis_specs, LINCLAB_COLS, colorFader
from Analysis.analysis_utils import get_grids, structured_unstructured

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'track_forgotten_states.csv')
#df['representation'] = df['representation'].apply(structured_unstructured)

gb = df.groupby(['env_name','representation','EC_cache_limit','forgetting_rule'])["save_id"]


convert_rep_to_color = {'structured':LINCLAB_COLS['red'], 'unstructured':LINCLAB_COLS['blue']}
labels_for_plot = {'structured':'structured','unstructured':'unstructured'}
cache_limits = analysis_specs['cache_limits']


envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
reps_to_plot = ['analytic successor','onehot']
grids = get_grids(envs_to_plot)


env = envs_to_plot[1]
rep = reps_to_plot[1]
pct = 25
forgetting_rule = 'oldest'

### diverging colormap
low ='#6e00c1' #purple
mid = '#dbdbdb' #gray
high = '#ff8000' #orange

n=500
fade_1 = []
fade_2 = []
for i in range(n):
    fade_1.append(colorFader(low,mid,i/n))
    fade_2.append(colorFader(mid,high,i/n))
#fade = ListedColormap(fade_1 + fade_2 + fade_3 + fade_4)
#plt.imshow([np.arange(n*4)],cmap=fade,aspect='auto')
fade_cm = colors.ListedColormap(np.vstack(fade_1 + fade_2))

def get_avg_array(idlist):
    objs = []
    for n in range(len(idlist)):
        run_id = idlist[n]

        with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
            data = pickle.load(f)

        num_forgets = np.nansum(data['forgotten_states'])
        objs.append(data['forgotten_states']/num_forgets)

    return np.nanmean(objs, axis=0)

def plot_forgetting_frequency(envs_to_plot, reps_to_plot, **kwargs):
    pct = kwargs.get('pct',75)

    fig, ax = plt.subplots(len(envs_to_plot),6)
    for e, env in enumerate(envs_to_plot):
        for r, rep in enumerate(reps_to_plot):
            objs = []
            for i, forgetting_rule in enumerate(['oldest','random']):

                idlist = list(gb.get_group((env, rep, int(cache_limits[env][100]*(pct/100)),forgetting_rule)))
                print(env,rep,pct,forgetting_rule,len(idlist))

                array = get_avg_array(idlist)

                objs.append(array)

                a = ax[e,3*r+i].imshow(array/0.005)
                fig.colorbar(a,ax=ax[e,3*r+i])
                ax[0,3*r+i].set_title(f'{rep}\n{forgetting_rule}')
            a = ax[e,3*r+i+1].imshow(((objs[0]-objs[1])+0.002)/0.004,cmap=fade_cm,vmin=0,vmax=1)
            fig.colorbar(a,ax=ax[e,3*r+i])
    plt.show()

def plot_forgetting_frequency_diff_only(envs_to_plot, reps_to_plot, **kwargs):
    pct = kwargs.get('pct',75)

    fig, ax = plt.subplots(len(envs_to_plot),2)
    for e, env in enumerate(envs_to_plot):
        for r, rep in enumerate(reps_to_plot):
            objs = []
            for i, forgetting_rule in enumerate(['oldest','random']):

                idlist = list(gb.get_group((env, rep, int(cache_limits[env][100]*(pct/100)),forgetting_rule)))
                print(env,rep,pct,forgetting_rule,len(idlist))

                array = get_avg_array(idlist)

                objs.append(array)

                #a = ax[e,3*r+i].imshow(array/0.005)
                #fig.colorbar(a,ax=ax[e,3*r+i])
                #ax[0,3*r+i].set_title(f'{rep}\n{forgetting_rule}')
            diff = objs[0]-objs[1]
            max_diff = np.nanmax(diff)
            min_diff = np.nanmin(diff)
            a = ax[e,r].imshow((diff-min_diff)/(max_diff-min_diff) ,cmap=fade_cm,vmin=0,vmax=1)
            fig.colorbar(a,ax=ax[e,r])
            if env[-2:] == '51':
                rwd_colrow= (15.5,8.5)
            else:
                rwd_colrow=(13.5,13.5)
            rect = plt.Rectangle(rwd_colrow, 1, 1, fill=False,edgecolor='k')
            ax[e,r].add_patch(rect)
            ax[e,r].get_xaxis().set_visible(False)
            ax[e,r].get_yaxis().set_visible(False)
            ax[0,r].set_title(f'{rep}')
    plt.savefig(f'../figures/CH2/forgetting_map_{pct}.svg')
    plt.show()
plot_forgetting_frequency_diff_only(envs_to_plot,reps_to_plot,pct=25)


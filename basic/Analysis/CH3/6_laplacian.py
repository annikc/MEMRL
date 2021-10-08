## uses functions from CH2/compare_ec_pol.py
import gym
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
import matplotlib as mpl

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import matplotlib.cm as cmx
from modules.Utils import running_mean as rm
from modules.Utils.gridworld_plotting import plot_pref_pol, plot_polmap
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents.RepresentationLearning.learned_representations import sr, onehot
from Analysis.analysis_utils import analysis_specs,linc_coolwarm,make_env_graph,compute_graph_distance_matrix, LINCLAB_COLS, plot_specs
from scipy.special import rel_entr
from scipy.stats import entropy


# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'ec_track_pols.csv')
groups_to_split = ['env_name','representation','EC_cache_limit']
gb = df.groupby(groups_to_split)["save_id"]


def plot_avg_laplace(env_name, pcts_to_plot,reps_to_plot):
    fig, ax = plt.subplots(len(reps_to_plot)*2,len(pcts_to_plot))
    for p, pct in enumerate(pcts_to_plot):
        for r, rep in enumerate(reps_to_plot):
            run_id = list(gb.get_group((env_name,rep,cache_limits[env_name][pct])))[0]
            print(run_id)

            with open(f'../../Data/ec_dicts/lifetime_dicts/{run_id}_polarcoord.p', 'rb') as f:
                polar_array = pickle.load(f)

            lpc = []
            for i in range(polar_array.shape[0]):
                lpc.append(laplace(polar_array[i,:]))
            mean_polar = np.mean(polar_array,axis=0)
            mean_laplace = np.mean(np.asarray(lpc),axis=0)
            ax[r*2+0,p].imshow(mean_polar,cmap=fade)
            print(rep, pct, mean_polar[15,2])
            a = ax[r*(2)+1,p].imshow(mean_laplace, cmap=fade_cm,vmin=-1000,vmax=1000)
            ax[r,p].get_xaxis().set_visible(False)
            #ax[r,p].get_yaxis().set_visible(False)
            ax[r,p].set_yticklabels([])
            if r ==0:
                ax[r,p].set_title(pct)
    for r in range(2):
        ax[r,0].set_ylabel(reps_to_plot[r])
        fig.colorbar(a,ax=ax[r,-1])


    plt.show()


#################
nums = {25:'0c9cf76e-8994-4052-b2e7-3fa59d53a6c5',
        50:'0219483c-54b9-4879-82e9-a7f05879262a',
        75:'6c7df218-8f3a-47a3-b9c2-00c3b6820e2b',
        100:'6d41b20a-056d-4286-b631-17f81896b83f'
        }

id_num = nums[50]
with open(f'../../Data/results/{id_num}_data.p', 'rb') as f:
    data = pickle.load(f)

pol_records = np.asarray(data['P_snap'])
print(pol_records.shape)

test = [np.asarray(list(x)) for x in pol_records[0:3, 5,5]]
print(np.asarray(test).shape)




# convert array to 4-d
def convert_array(arr,save=False):
    dxdy = np.array([(0,-1),(0,1),(1,0),(-1,0)])
    expanded_arr = np.empty((arr.shape[0],20,20))
    example_polar_coord = np.zeros(400)
    example_polar_coord[:] = np.nan
    laplace_array = np.empty((arr.shape[0],20,20))
    for ep in range(arr.shape[0]):
        if ep%500==0:
            print(ep)
        pol_map = arr[ep].flatten()
        for ind, x in enumerate(pol_map):
            p = np.asarray(list(x))
            pref_dir = np.dot(p,dxdy)
            example_polar_coord[ind] = np.degrees(np.arctan2(pref_dir[1],pref_dir[0]))%360

        expanded_arr[ep] = example_polar_coord.reshape(20,20)
    #a = plt.imshow(example_polar_coord.reshape(20,20),cmap=fade,vmin=0, vmax=360)
    #plt.colorbar(a)
    #plt.show()

    if save:
        with open(f'policy_arrays/{id_num}_array.p', 'wb') as savedata:
            pickle.dump(expanded_arr, savedata)



#convert_array(pol_records,save=True)
def load_array(id_num):
    with open(f'./policy_arrays/{id_num}_array.p', 'rb') as f:
        data = pickle.load(f)

    return(data)

arr = load_array(id_num)

def rough_laplace(arr, coord):
    lpc = []
    for x in range(0,15000,1):
        target = arr[x,coord[0],coord[1]]
        neighbours =[arr[x,coord[0]-1,coord[1]],
                     arr[x,coord[0]+1,coord[1]],
                     arr[x,coord[0],coord[1]-1],
                     arr[x,coord[0],coord[1]+1]]
        mean_ = np.nanmean(neighbours)
        diffy = target-mean_
        lpc.append(diffy)
    return lpc
lpc = rough_laplace(arr,(13,14))
plt.plot(lpc)
plt.show()
'''
plt.plot(rm(lpc[:,6,5],200), color='r',label='(5,5)')
plt.plot(rm(lpc[:,13,14],200), color='b',label='(14,14)')
plt.legend(loc=0)
plt.show()

plot_pref_pol(gym.make('gridworld:gridworld-v41'),pol_records[0])
dxdy = np.array([(0,-1),(0,1),(1,0),(-1,0)])
example_polar_coord = np.zeros(400)
example_polar_coord[:] = np.nan
for ind, x in enumerate(pol_records[0].flatten()):
    p = np.asarray(list(x))
    pref_dir = np.dot(p,dxdy)
    example_polar_coord[ind] = np.degrees(np.arctan2(pref_dir[1],pref_dir[0]))%360

print(example_polar_coord.reshape(20,20))
fig, ax = plt.subplots(1,2)
a = ax[0].imshow(example_polar_coord.reshape(20,20),cmap=fade,vmin=0, vmax=360)
plt.colorbar(a, ax=ax[0])
lpc = laplace(example_polar_coord.reshape(20,20))
b = ax[1].imshow(lpc)
plt.colorbar(b, ax=ax[1])
plt.show()

print(lpc[5,5])
'''
# matrix multiplication with direction vector

# conversion from cartesian to polar

# compute scalar laplacian

# get laplacian for (5,5) and (14,14) over 15k


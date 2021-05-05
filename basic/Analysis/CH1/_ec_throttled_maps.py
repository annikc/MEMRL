## plot effect of cache limit size on episodic control
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.cm as cmx

import pickle
import sys
import gym
import pandas as pd

sys.path.append('../../modules/')
from modules.Agents.RepresentationLearning.learned_representations import load_saved_latents
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from Analysis.analysis_utils import get_avg_std, get_id_dict, get_grids, plot_each
from modules.Utils import running_mean as rm
from modules.Utils import softmax
from scipy.stats import entropy


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

def make_arrows(action, probability):
    '''
    alternate style:
        def make_arrows(action):
        offsets = [(0,0.25),(0,-0.25),(0.25,0),(-0.25,0),(0,0),(0.1,0.1) ] # D U R L J P
        dx,dy = offsets[action]
        head_w, head_l = 0.1, 0.1
        return dx, dy, head_w, head_l
    :param action:
    :param probability:
    :return:
    '''
    if probability == 0:
        dx, dy = 0, 0
        head_w, head_l = 0, 0
    else:
        dxdy = [(0.0, 0.25),  # D
                (0.0, -0.25),  # U
                (0.25, 0.0),  # R
                (-0.25, 0.0),  # L
                (0.1, -0.1),  # points right and up #J
                (-0.1, 0.1),  # points left and down # P
                ]
        dx, dy = dxdy[action]  #dxdy[(action-1)%len(dxdy)] ## use if action-space remapping

        head_w, head_l = 0.1, 0.1

    return dx, dy, head_w, head_l






grids = get_grids([x[-2:] for x in envs])

convert_rep_to_color = {100:'C2', 200:'C9', 300:'C4', 400:'C3'}

def get_ec_maps():
    ec_maps = {}
    example_env = gym.make(envs[0])
    plt.close()
    for ind, name in enumerate(envs):
        ec_maps[name] = {}
        for jnd, cache_size in enumerate(master_dict[name].keys()):
            print(name, cache_size)
            v_list = master_dict[name][cache_size]
            policy_map = np.zeros(example_env.shape, dtype=[(x, 'f8') for x in example_env.action_list])
            # load_ec dict
            with open(data_dir+f'ec_dicts/{v_list[0]}_EC.p', 'rb') as f:
                cache_list = pickle.load(f)
            for key in list(cache_list.keys()):
                coord = example_env.oneD2twoD(cache_list[key][2])
                policy_map[coord] = tuple(softmax(np.nan_to_num(cache_list[key][0][:,0])))
            ec_maps[name][cache_size] = policy_map
    return ec_maps

def get_mem_maps():
    ec_maps = {}
    for key in master_dict.keys():
        env = gym.make(key[:-1])
        print(env.rewards.keys(), "reward at ")
        plt.close()
        print(key, len(env.useable))
        ec_maps[key] = {}
        latents, _, __, ___ = load_saved_latents(env)

        for j, cache_size in enumerate(master_dict[key].keys()):
            print(j,cache_size)
            v_list = master_dict[key][cache_size]
            policy_map = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])
            # load_ec dict
            with open(data_dir+f'ec_dicts/{v_list[0]}_EC.p', 'rb') as f:
                cache_list = pickle.load(f)

            mem = Memory(entry_size=env.action_space.n, cache_limit=400)
            mem.cache_list = cache_list

            for state2d in env.useable:
                state1d = env.twoD2oneD(state2d)
                state_rep = latents[state1d]

                policy_map[state2d] = tuple(mem.recall_mem(tuple(state_rep)))

            ec_maps[key][cache_size] = policy_map
    return ec_maps


def plot_all_maxpol(save=False):
    fig, axs = plt.subplots(4, 5, sharex='col')
    cmap = plt.cm.Spectral_r
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    chance_threshold = 0.2

    for ind, name in enumerate(ec_maps.keys()):
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

        for jnd, cache_size in enumerate(list(ec_maps[name].keys())):
            policy_array = ec_maps[name][cache_size]
            axs[ind,jnd+1].pcolor(grids[ind], vmin=0, vmax=1, cmap='bone')
            axs[ind,jnd+1].add_patch(plt.Rectangle(rwd_colrow,1,1,edgecolor='w',fill=False))
            for i in range(20):
                for j in range(20):
                    action = np.argmax(tuple(policy_array[i][j]))
                    prob = max(policy_array[i][j])

                    dx1, dy1, head_w, head_l = make_arrows(action, prob)
                    if prob > chance_threshold:
                        if (dx1, dy1) == (0, 0):
                            pass
                        else:
                            colorVal1 = scalarMap.to_rgba(prob)
                            axs[ind,jnd+1].arrow(j + 0.5, i + 0.5, dx1, dy1, head_width=0.3, head_length=0.2, color=colorVal1)
                    else:
                        pass
            axs[ind,jnd+1].set_aspect('equal')
            axs[ind,jnd+1].invert_yaxis()
            axs[ind,jnd+1].get_xaxis().set_visible(False)
            axs[ind,jnd+1].get_yaxis().set_visible(False)
    if save:
        plt.savefig('../figures/CH1/ec_throttled_maps.svg',format='svg')

    plt.show()

def plot_all_prefpol(ec_maps, save=False):
    fig, axs = plt.subplots(4, 5, sharex='col')
    cmap = plt.cm.Spectral_r
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    chance_threshold = 0.2

    for ind, name in enumerate(list(ec_maps.keys())):
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

        for jnd, cache_size in enumerate(list(ec_maps[name].keys())):
            print("coord:", ind,jnd+1)
            policy_array = ec_maps[name][cache_size]
            axs[ind,jnd+1].pcolor(grids[ind], vmin=0, vmax=1, cmap='bone')
            axs[ind,jnd+1].add_patch(plt.Rectangle(rwd_colrow,1,1,edgecolor='w',fill=False))
            for i in range(20):
                for j in range(20):
                    policy = tuple(policy_array[i, j])
                    dx, dy = 0.0, 0.0
                    for pind, k in enumerate(policy):
                        action = pind
                        prob = k
                        if prob < 0.01:
                            pass
                        else:
                            dx1, dy1, head_w, head_l = make_arrows(action, prob)
                            dx += dx1*prob
                            dy += dy1*prob
                    if dx ==0.0 and dy == 0.0:
                        pass
                    else:
                        colorVal1 = scalarMap.to_rgba(entropy(policy))
                        axs[ind,jnd+1].arrow(j + 0.5, i + 0.5, dx, dy, head_width=0.3, head_length=0.2, color=colorVal1)

            axs[ind,jnd+1].set_aspect('equal')
            axs[ind,jnd+1].invert_yaxis()
            axs[ind,jnd+1].get_xaxis().set_visible(False)
            axs[ind,jnd+1].get_yaxis().set_visible(False)
    if save:
        plt.savefig('../figures/CH1/ec_throttled_maps.svg',format='svg')

    plt.show()

def plot_one_prefpol(ec_maps, map_index, save=False):
    fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex='col')
    cmap = plt.cm.Spectral_r
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    chance_threshold = 0.2

    if map_index == 3:
            rwd_colrow= (16,9)
    else:
        rwd_colrow=(14,14)

    for ind, ax in enumerate(axes.flatten()):
        sizes = list(ec_maps[list(ec_maps.keys())[map_index]].keys())
        print(sizes)
        env_ec_maps = ec_maps[list(ec_maps.keys())[map_index]]
        cache_size = list(env_ec_maps.keys())[ind]

        policy_array = env_ec_maps[cache_size]
        ax.pcolor(grids[map_index], vmin=0, vmax=1, cmap='bone')
        ax.add_patch(plt.Rectangle(rwd_colrow,1,1,edgecolor='w',fill=False))
        for i in range(20):
            for j in range(20):
                policy = tuple(policy_array[i, j])
                dx, dy = 0.0, 0.0
                for pind, k in enumerate(policy):
                    action = pind
                    prob = k
                    if prob < 0.01:
                        pass
                    else:
                        dx1, dy1, head_w, head_l = make_arrows(action, prob)
                        dx += dx1*prob
                        dy += dy1*prob
                if dx ==0.0 and dy == 0.0:
                    pass
                else:
                    colorVal1 = scalarMap.to_rgba(entropy(policy))
                    ax.arrow(j + 0.5, i + 0.5, dx, dy, head_width=0.3, head_length=0.2, color=colorVal1)

        ax.set_aspect('equal')
        ax.set_title(f'{cache_size}')
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    if save:
        save_format = 'svg'
        plt.savefig(f'../figures/CH1/mem_map_{map_index}.{save_format}',format=save_format)

    plt.show()



#plot_all_maxpol()
#plot_all_prefpol()
ec_maps = get_ec_maps()
mem_maps = get_mem_maps()
for i in range(4):
    plot_one_prefpol(mem_maps,i,save=True)
#plot_all(cutoff=5000,smoothing=50,save=True)

#plot_each(master_dict[env][rep], data_dir)
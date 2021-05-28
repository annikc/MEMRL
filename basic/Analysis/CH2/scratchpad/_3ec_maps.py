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
from modules.Agents.RepresentationLearning.learned_representations import onehot, random, place_cell, sr, latents
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from Analysis.analysis_utils import get_avg_std, get_grids, plot_each
from modules.Utils.gridworld_plotting import make_arrows
from modules.Utils import softmax
from scipy.stats import entropy


parent_path = '../../Data/'

df = pd.read_csv(parent_path+'throttled_ec_allreps_chebyshev.csv')
print(df.env_name.unique())
gb = df.groupby(['env_name','representation','EC_cache_limit'])["save_id"]

cache_limits = {'gridworld:gridworld-v11':{100:400, 75:300, 50:200, 25:100},
                'gridworld:gridworld-v31':{100:365, 75:273, 50:182, 25:91},
                'gridworld:gridworld-v41':{100:384, 75:288, 50:192, 25:96},
                'gridworld:gridworld-v51':{100:286, 75:214, 50:143, 25:71}}

def get_ec_map(gb, env_name, rep, pct, ind=0):
    example_env = gym.make(env_name)
    plt.close()

    run_list = list(gb.get_group((env_name, rep, cache_limits[env_name][pct])))
    run_id = run_list[ind]
    policy_map = np.zeros(example_env.shape, dtype=[(x, 'f8') for x in example_env.action_list])

    with open(parent_path+f'ec_dicts/{run_id}_EC.p', 'rb') as f:
        cache_list = pickle.load(f)

    for key in cache_list.keys():
        coord = example_env.oneD2twoD(cache_list[key][2])
        policy_map[coord] = tuple(softmax(np.nan_to_num(cache_list[key][0][:,0])))

    return policy_map

def get_mem_map(gb,env_name,rep,pct,ind=0):
    example_env = gym.make(env_name)
    plt.close()

    rep_types = {'onehot':onehot, 'random':random, 'place_cell':place_cell, 'analytic successor':sr}
    if rep == 'latents':
        conv_ids = {'gridworld:gridworld-v1':'c34544ac-45ed-492c-b2eb-4431b403a3a8',
                    'gridworld:gridworld-v3':'32301262-cd74-4116-b776-57354831c484',
                    'gridworld:gridworld-v4':'b50926a2-0186-4bb9-81ec-77063cac6861',
                    'gridworld:gridworld-v5':'15b5e27b-444f-4fc8-bf25-5b7807df4c7f'}
        load_id = conv_ids[f'{env_name[:-1]}']
        agent_path = parent_path+f'agents/{load_id}.pt'
        state_reps, representation_name, input_dims, _ = latents(example_env, agent_path)
    else:
        state_reps, representation_name, input_dims, _ = rep_types[rep](example_env)

    run_id = list(gb.get_group((env_name, rep, cache_limits[env_name][pct])))[ind]
    policy_map = np.zeros(example_env.shape, dtype=[(x, 'f8') for x in example_env.action_list])

    with open(parent_path+f'ec_dicts/{run_id}_EC.p', 'rb') as f:
        cache_list = pickle.load(f)

    mem = Memory(entry_size=example_env.action_space.n, cache_limit=400) # cache limit doesn't matter since we are only using for recall
    mem.cache_list = cache_list

    for state2d in example_env.useable:
        state1d = example_env.twoD2oneD(state2d)
        state_rep = tuple(state_reps[state1d])
        #print(state_rep in cache_list.keys())

        policy_map[state2d] = tuple(mem.recall_mem(state_rep))

    return policy_map


def plt_prefpol(env_num, env_grid, maps_dict,plot_name,save=False):
    '''
    :param env_name: int specifying environment type
    :param maps_dict: dictionary containing policy maps
    :param save: whether to save generated plot
    :return: nothing
    '''

    fig, axes = plt.subplots(1, len(list(maps_dict.keys())), figsize =(2*len(list(maps_dict.keys()))+2, 3))
    cmap = plt.cm.Spectral_r
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    chance_threshold = 0.2

    if env_num == 5:
        rwd_colrow= (16,9)
    else:
        rwd_colrow=(14,14)

    for ind, ax in enumerate(axes.flatten()):
        # lay down base map
        ax.pcolor(env_grid, vmin=0, vmax=1, cmap='bone')
        ax.add_patch(plt.Rectangle(rwd_colrow,1,1,edgecolor='w',fill=False))

        #index corresponds to one of the elements of the maps dict
        cache_size = list(maps_dict.keys())[ind]
        policy_array = maps_dict[cache_size]
        rows, columns = env_grid.shape
        for i in range(rows):
            for j in range(columns):
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
    plt.suptitle(f'{plot_name}')
    if save:
        save_format = 'svg'
        plt.savefig(f'../figures/CH2/mem_map_{env_num}.{save_format}',format=save_format)

    #plt.show()

def plt_maxpol(env_num, env_grid, maps_dict,plot_name, save=False):
    '''
    :param env_name: int specifying environment type
    :param maps_dict: dictionary containing policy maps
    :param save: whether to save generated plot
    :return: nothing
    '''

    fig, axes = plt.subplots(1, len(list(maps_dict.keys())), figsize =(2*len(list(maps_dict.keys()))+2, 3))
    cmap = plt.cm.Spectral_r
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    chance_threshold = 0.2

    if env_num == 5:
        rwd_colrow= (16,9)
    else:
        rwd_colrow=(14,14)

    for ind, ax in enumerate(axes.flatten()):
        # lay down base map
        ax.pcolor(env_grid, vmin=0, vmax=1, cmap='bone')
        ax.add_patch(plt.Rectangle(rwd_colrow,1,1,edgecolor='w',fill=False))

        #index corresponds to one of the elements of the maps dict
        cache_size = list(maps_dict.keys())[ind]
        policy_array = maps_dict[cache_size]
        rows, columns = env_grid.shape
        for i in range(rows):
            for j in range(columns):
                action = np.argmax(tuple(policy_array[i][j]))
                prob = max(policy_array[i][j])

                dx1, dy1, head_w, head_l = make_arrows(action, prob)
                if prob > chance_threshold:
                    if (dx1, dy1) == (0, 0):
                        pass
                    else:
                        colorVal1 = scalarMap.to_rgba(prob)
                        ax.arrow(j + 0.5, i + 0.5, dx1, dy1, head_width=0.3, head_length=0.2, color=colorVal1)
                else:
                    pass

        ax.set_aspect('equal')
        ax.set_title(f'{cache_size}')
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(f'{plot_name}')
    plt.tight_layout()
    if save:
        save_format = 'svg'
        plt.savefig(f'../figures/CH2/mem_map_{env_num}.{save_format}',format=save_format)

    #plt.show()

version = 1
env_name = f'gridworld:gridworld-v{version}1'
grid = get_grids(env_name[-2:])[0]
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['random','analytic successor','onehot','place_cell']
for rep in reps_to_plot:
    ind=0
    maps_dict = {}
    for pct in pcts_to_plot:
        mm = get_mem_map(gb,env_name,rep,pct,ind)
        maps_dict[pct] = mm
    plt_prefpol(version,grid,maps_dict,plot_name=rep)
plt.show()


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


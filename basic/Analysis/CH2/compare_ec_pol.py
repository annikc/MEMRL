import gym
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from modules.Utils.gridworld_plotting import plot_pref_pol, plot_polmap
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents.RepresentationLearning.learned_representations import sr, onehot
from Analysis.analysis_utils import analysis_specs,linc_coolwarm,make_env_graph,compute_graph_distance_matrix, LINCLAB_COLS
from scipy.special import rel_entr
from scipy.stats import entropy


# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'ec_track_pols.csv')
groups_to_split = ['env_name','representation','EC_cache_limit']
gb = df.groupby(groups_to_split)["save_id"]

def get_avg_of_memories(data):
    n_visits = np.zeros(env.shape)
    ec_pol_grid = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])
    for i in range(len(data['ec_dicts'])):
        print(i)
        blank_mem = Memory(cache_limit=400, entry_size=4)
        blank_mem.cache_list = data['ec_dicts'][i]
        states = []
        for k, key in enumerate(blank_mem.cache_list.keys()):
            twoD = env.oneD2twoD(blank_mem.cache_list[key][2])
            old_policy = ec_pol_grid[twoD]
            current_policy = blank_mem.recall_mem(key)

            average = []
            for x,y in zip(old_policy, current_policy):
                z = x + (y-x)/(k+1)
                average.append(z)
            ec_pol_grid[twoD] = tuple(average)

            n_visits[twoD]+=1


def get_KLD(data,probe_state,trial_num):
    probe_rep = state_reps[probe_state]

    KLD_array = np.zeros(env.shape)
    KLD_array[:] = np.nan
    entropy_array = np.zeros(env.shape)
    entropy_array[:] = np.nan
    ec_pol_grid = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])

    blank_mem = Memory(cache_limit=400, entry_size=4)
    blank_mem.cache_list = data['ec_dicts'][trial_num]
    probe_pol = blank_mem.recall_mem(probe_rep)
    for k in state_reps.keys():
        sr_rep = state_reps[k]
        pol = blank_mem.recall_mem(sr_rep)
        twoD = env.oneD2twoD(k)
        KLD_array[twoD] = sum(rel_entr(list(probe_pol),list(pol)))
        ec_pol_grid[twoD] = tuple(pol)
        entropy_array[twoD] = entropy(pol,base=2)

    return KLD_array,ec_pol_grid,entropy_array


rep_dict = {'analytic successor':sr, 'onehot':onehot}
cache_limits = analysis_specs['cache_limits']
env_name = 'gridworld:gridworld-v41'
reps = ['onehot', 'analytic successor']
color_map = {'onehot':LINCLAB_COLS['blue'], 'analytic successor':LINCLAB_COLS['red']}
rep = reps[0]

env = gym.make(env_name)
plt.close()

state_reps, _, __, ___ = rep_dict[rep](env)
if rep == 'analytic successor':
    for s1 in env.obstacle:
        state_reps.pop(s1)

G = make_env_graph(env)
gd = compute_graph_distance_matrix(G,env)
dist_in_state_space = gd[env.twoD2oneD((14,14))]

def plot_dist_v_entropy(kld_ent = 'ent'):
    probe_state = (13,14)
    fig, ax = plt.subplots(1,4,figsize=(8,2))

    E = []
    for i, pct in enumerate([100,75]):
        run_id = list(gb.get_group((env_name,rep,cache_limits[env_name][pct])))[0]
        print(run_id)

        with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
            data = pickle.load(f)

        K = []
        if pct ==100:
            start=999
        else:
            start=899
        for x in range(start,1000):
            print(x)
            kld_, ec_pols,entropy_ = get_KLD(data, env.twoD2oneD(probe_state), x)
            if kld_ent =='ent':
                K.append(entropy_)
            elif kld_ent == 'kld':
                K.append(kld_)
            E.append(ec_pols)
        kld = np.mean(K, axis=0)

        avg_entropy = kld.reshape(1,400)

        ax[i].scatter(dist_in_state_space,avg_entropy,color=color_map[rep])
        ax[i].set_ylim([-0.1,2.1])
        ax[i].set_xlim([-1,30])

        #a = ax[i].imshow(kld,vmin=0,vmax=2,cmap=linc_coolwarm)
        #ax[i].add_patch(plt.Rectangle(np.add((14,14),(-0.5,-0.5)),1,1, fill=False,edgecolor='w'))
        #ax[i].set_title(f'{pct}')
        #ax[i].get_xaxis().set_visible(False)
        #ax[i].get_yaxis().set_visible(False)
    plt.suptitle(f'{rep}')
    format ='svg'
    plt.savefig(f'../figures/CH2/dist_v_{kld_ent}{env_name[-2:]}_{rep}.{format}',format=format)
    plt.show()

def plot_maps(kld_ent = 'ent'):
    probe_state = (13,14)
    fig, ax = plt.subplots(1,4)

    E = []
    for i, pct in enumerate([100,75,50,25]):
        run_id = list(gb.get_group((env_name,rep,cache_limits[env_name][pct])))[0]
        print(run_id)

        with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
            data = pickle.load(f)

        K = []
        if pct ==100:
            start=999
        else:
            start=899
        for x in range(start,1000):
            print(x)
            kld_, ec_pols,entropy_ = get_KLD(data, env.twoD2oneD(probe_state), x)
            if kld_ent =='ent':
                K.append(entropy_)
            elif kld_ent == 'kld':
                K.append(kld_)
            E.append(ec_pols)
        kld = np.mean(K, axis=0)

        a = ax[i].imshow(kld,vmin=0,vmax=2,cmap=linc_coolwarm)
        ax[i].add_patch(plt.Rectangle(np.add((14,14),(-0.5,-0.5)),1,1, fill=False,edgecolor='w'))
        ax[i].set_title(f'{pct}')
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
    plt.suptitle(f'{rep}')
    plt.colorbar(a)
    format ='svg'
    plt.savefig(f'../figures/CH2/singletrial_{kld_ent}{env_name[-2:]}_{rep}.{format}',format=format)
    plt.show()

plot_dist_v_entropy('ent')
def plot_square():
    # get id of sim
    pct = 100
    run_id = list(gb.get_group((env_name,rep,cache_limits[env_name][pct])))[0]
    print(run_id)
    with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
        data = pickle.load(f)

    trial_number = 999
    ec_dict = data['ec_dicts'][trial_number]
    blank_mem = Memory(cache_limit=400, entry_size=4)
    blank_mem.cache_list = ec_dict
    probs = np.array((4,20,20))

    probe_state = (14,5)
    state = env.twoD2oneD(probe_state)
    west  = env.twoD2oneD((probe_state[0],   probe_state[1]-1))
    north = env.twoD2oneD((probe_state[0]-1, probe_state[1]))
    east  = env.twoD2oneD((probe_state[0],   probe_state[1]+1))
    south = env.twoD2oneD((probe_state[0]+1, probe_state[1]))
    print(state,west,north,east,south)
    pi_0 = blank_mem.recall_mem(state_reps[state])
    pi_w = blank_mem.recall_mem(state_reps[west])
    pi_n = blank_mem.recall_mem(state_reps[north])
    pi_e = blank_mem.recall_mem(state_reps[east])
    pi_s = blank_mem.recall_mem(state_reps[south])

    print(pi_0,pi_w)
    ks = np.zeros((3,3))
    ks[:] =np.nan

    ks[0,1] = sum(rel_entr(pi_0,pi_n))
    ks[1,0] = sum(rel_entr(pi_0,pi_w))
    ks[1,1] = sum(rel_entr(pi_0,pi_0))
    ks[1,2] = sum(rel_entr(pi_0,pi_e))
    ks[2,1] = sum(rel_entr(pi_0,pi_s))

    a = plt.imshow(ks)
    plt.colorbar(a)
    plt.show()
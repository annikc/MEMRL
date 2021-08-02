import gym
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
import matplotlib as mpl

import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.cm as cmx

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

rep_dict = {'analytic successor':sr, 'onehot':onehot}
cache_limits = analysis_specs['cache_limits']
env_name = 'gridworld:gridworld-v41'
reps = ['onehot', 'analytic successor']
color_map = {'onehot':LINCLAB_COLS['blue'], 'analytic successor':LINCLAB_COLS['red']}
rep = reps[1]

env = gym.make(env_name)
plt.close()

state_reps, _, __, ___ = rep_dict[rep](env)
if rep == 'analytic successor':
    for s1 in env.obstacle:
        state_reps.pop(s1)

G = make_env_graph(env)
gd = compute_graph_distance_matrix(G,env)
dist_in_state_space = gd[env.twoD2oneD((14,14))]

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
                ]
        dx, dy = dxdy[action]  #dxdy[(action-1)%len(dxdy)] ## use if action-space remapping

        head_w, head_l = 0.1, 0.1

    return dx, dy, head_w, head_l

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


def get_avg_incidence_of_memories(data):
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

def plot_dist_v_entropy(kld_ent = 'ent'):
    probe_state = (13,14)
    fig, ax = plt.subplots(1,4,figsize=(14,3))

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

        avg_entropy = kld.reshape(1,400)

        ax[i].scatter(dist_in_state_space,avg_entropy,color=color_map[rep])
        ax[i].set_ylim([-0.1,2.1])
        ax[i].set_xlim([-1,max(dist_in_state_space)+1])

        #a = ax[i].imshow(kld,vmin=0,vmax=2,cmap=linc_coolwarm)
        #ax[i].add_patch(plt.Rectangle(np.add((14,14),(-0.5,-0.5)),1,1, fill=False,edgecolor='w'))
        #ax[i].set_title(f'{pct}')
        #ax[i].get_xaxis().set_visible(False)
        #ax[i].get_yaxis().set_visible(False)
    plt.suptitle(f'{rep}')
    format ='svg'
    plt.savefig(f'../figures/CH2/dist_v_{kld_ent}{env_name[-2:]}_{rep}.{format}',format=format)
    plt.show()

def plot_maps(env_name, rep, kld_ent = 'ent'):
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
            start=999
        for x in range(start,1000):
            print(x)
            kld_, ec_pols,entropy_ = get_KLD(data, env.twoD2oneD(probe_state), x)
            if kld_ent =='ent':
                K.append(entropy_)
            elif kld_ent == 'kld':
                K.append(kld_)
            E.append(ec_pols)
        kld = np.mean(K, axis=0)

        a = ax[i].imshow(kld,vmin=0,vmax=2.,cmap='viridis')
        print(np.nanmax(kld))
        ax[i].add_patch(plt.Rectangle(np.add((14,14),(-0.5,-0.5)),1,1, fill=False,edgecolor='w'))
        ax[i].set_title(f'{pct}')
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
    plt.suptitle(f'{rep}')
    plt.colorbar(a)
    format ='svg'
    plt.savefig(f'../figures/CH2/viridis_longrun_{kld_ent}{env_name[-2:]}_{rep}.{format}',format=format)
    plt.show()

def get_mem_maps(data,trial_num=-1,full_mem=True):
    blank_mem = Memory(cache_limit=400, entry_size=4)
    blank_mem.cache_list = data['ec_dicts'][trial_num]

    ec_pol_grid = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])
    if full_mem:
        for key, value in state_reps.items():
            twoD = env.oneD2twoD(key)
            sr_rep = value
            pol = blank_mem.recall_mem(sr_rep)

            ec_pol_grid[twoD] = tuple(pol)
    else:
        for ec_key in blank_mem.cache_list.keys():
            twoD = env.oneD2twoD(blank_mem.cache_list[ec_key][2])
            pol  = blank_mem.recall_mem(ec_key)

            ec_pol_grid[twoD] = tuple(pol)

    return ec_pol_grid




def plot_memory_maps(env_name,rep,pcts_to_plot,full_mem=True):
    env = gym.make(env_name)
    plt.close()

    state_reps, _, __, ___ = rep_dict[rep](env)
    if rep == 'analytic successor':
        for s1 in env.obstacle:
            state_reps.pop(s1)

    fig, ax = plt.subplots(1,len(pcts_to_plot),figsize=(18,4))
    cmap = linc_coolwarm
    cNorm = colors.Normalize(vmin=0, vmax=2)
    for i, pct in enumerate(pcts_to_plot):
        run_id = list(gb.get_group((env_name,rep,cache_limits[env_name][pct])))[0]
        print(run_id)

        with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
            data = pickle.load(f)

        pol_array = get_mem_maps(data,full_mem=full_mem) # for alt use trial #800

        # set base plot
        ax[i].pcolor(env.grid,vmin=0,vmax=1,cmap='bone')

        for rwd_loc in env.rewards:
            rwd_r, rwd_c = rwd_loc
            ax[i].add_patch(plt.Rectangle((rwd_c, rwd_r), width=0.99, height=1, linewidth=1, ec='white', fill=False))
        ax[i].set_aspect('equal')
        ax[i].set_title(pct)
        ax[i].invert_yaxis()
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)

        for r in range(env.r):
            for c in range(env.c):
                policy = tuple(pol_array[r, c])

                dx, dy = 0.0, 0.0
                for ind, k in enumerate(policy):
                    action = ind
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
                    colorVal1 = cmap(entropy(policy))
                    if entropy(policy) > 2:
                        pass
                    else:
                        ax[i].arrow(c + 0.5, r + 0.5, dx, dy, head_width=0.3, head_length=0.3, color=colorVal1)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,wspace=0.02, hspace=0.02)
    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(cmx.ScalarMappable(norm=cNorm, cmap=cmap),ax=cb_ax)
    cbar.set_label('Policy Entropy')
    format = 'svg'
    plt.savefig(f'../figures/CH2/example_memory_maps_{rep}_{env_name[-2:]}_alt_trial.{format}',format=format)
    plt.show()

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return (1-mix)*c1 + mix*c2 #mpl.colors.to_hex()#

north='#1f77b4' #blue "#50a2d5"
east = "#4bb900" #"#76bb4b" #green
south= '#ffe200' # yellow
west = "#eb3920"# red
n=90
fade_1 = []
fade_2 = []
fade_3 = []
fade_4 = []
for i in range(n):
    fade_1.append(colorFader(north,east,i/n))
    fade_2.append(colorFader(east,south,i/n))
    fade_3.append(colorFader(south,west,i/n))
    fade_4.append(colorFader(west,north,i/n))
#fade = ListedColormap(fade_1 + fade_2 + fade_3 + fade_4)
#plt.imshow([np.arange(n*4)],cmap=fade,aspect='auto')

fade = colors.ListedColormap(np.vstack(fade_1 + fade_2 + fade_3 + fade_4))

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

green = "#4bb900"


env_name = 'gridworld:gridworld-v41'
rep = 'analytic successor'
pct = 50
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

#plot_avg_laplace(env_name,pcts_to_plot=[100,75,50,25],reps_to_plot=['analytic successor','onehot'])

for rep in ['analytic successor', 'onehot']:
    plot_maps(env_name, rep)




### Junkyard
def plot_kld_neighbours():
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



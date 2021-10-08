## uses functions from CH2/compare_ec_pol.py
import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

import matplotlib.colors as colors
import matplotlib.cm as cmx

from modules.Utils.gridworld_plotting import plot_pref_pol, plot_polmap
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from Analysis.analysis_utils import analysis_specs,linc_coolwarm,make_env_graph,compute_graph_distance_matrix, LINCLAB_COLS, plot_specs, fade
from scipy.special import rel_entr
from scipy.stats import entropy

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
    ec_pol_grid = np.zeros((*env.shape,4))#np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])

    blank_mem = Memory(cache_limit=400, entry_size=4)
    blank_mem.cache_list = data['ec_dicts'][trial_num]
    probe_pol = blank_mem.recall_mem(probe_rep)

    #for k in state_reps.keys():
        #sr_rep = state_reps[k]
    for sr_rep in blank_mem.cache_list.keys():
        k = blank_mem.cache_list[sr_rep][2]
        pol = blank_mem.recall_mem(sr_rep)
        twoD = env.oneD2twoD(k)
        KLD_array[twoD] = sum(rel_entr(list(probe_pol),list(pol)))
        ec_pol_grid[twoD][:] = pol
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

def plot_avg_entropy(env_name, reps_to_plot, pcts_to_plot =[100,75,50,25], kld_ent = 'ent',**kwargs):
    plottype = kwargs.get('type','bar')
    convert_rep_to_color = kwargs.get('colors',color_map)
    reps_to_titles = {'onehot':'Unstructured', 'analytic successor':'Structured'}
    probe_state = (13,14)
    E = []
    avg_entropy = {}
    std_entropy = {}
    for rep in reps_to_plot:
        avg_entropy[rep] = []
        std_entropy[rep] = []
        for i, pct in enumerate(pcts_to_plot):
            run_id = list(gb.get_group((env_name,rep,cache_limits[env_name][pct])))[0]
            print(run_id)

            with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
                data = pickle.load(f)

            K = []
            if pct ==100:
                start=999
            else:
                start=895
            for x in range(start,1000):
                print(x)
                kld_, ec_pols,entropy_ = get_KLD(data, env.twoD2oneD(probe_state), x)
                K.append(entropy_.flatten())

            mean_ent = np.nanmean(K, axis=0)

            #std = np.nanstd(K)
            #mean_ent = np.nanmean(K)
            #print(mean_ent)
            avg_entropy[rep].append(np.nan_to_num(mean_ent))
            #std_entropy[rep].append(std)
    if plottype == 'bar':
        for r, rep in enumerate(reps_to_plot):
            barwidth=0.3
            plt.bar(np.arange(len(pcts_to_plot))+(barwidth*r), avg_entropy[rep], yerr= std_entropy[rep], width=barwidth, color= color_map[rep], alpha=1)
        plt.xticks(np.arange(len(pcts_to_plot))+(barwidth/2), labels=pcts_to_plot)
    elif plottype == 'violin':
        fig, ax = plt.subplots(2,1, sharex=True)
        for r, rep in enumerate(reps_to_plot):
            body = ax[r].violinplot(positions=np.asarray(pcts_to_plot), dataset=avg_entropy[rep], vert=True, widths=15)

            for violinkey in body.keys():
                if violinkey == 'bodies':
                    for b in body['bodies']:
                        b.set_color(convert_rep_to_color[rep])
                        b.set_alpha(pct/100)
                else:
                    body[violinkey].set_color(convert_rep_to_color[rep])
            ax[r].set_xticks(pcts_to_plot)
            ax[r].set_xlim([110, 15])
            ax[r].set_ylabel(f'{reps_to_titles[rep]}')
            ax[r].set_yticks([0,.5,1,1.5,2])
    plt.savefig('../figures/CH3/distribution_entropy.svg')
    plt.show()

def plot_all_entropy(env_name, reps_to_plot, pcts_to_plot =[100,75,50,25], kld_ent = 'ent',**kwargs):
    plottype = kwargs.get('type','bar')
    convert_rep_to_color = kwargs.get('colors',color_map)
    reps_to_titles = {'onehot':'Unstructured', 'analytic successor':'Structured'}
    probe_state = (13,14)
    E = []
    avg_entropy = {}
    std_entropy = {}
    for rep in reps_to_plot:
        avg_entropy[rep] = []
        std_entropy[rep] = []
        for i, pct in enumerate(pcts_to_plot):
            run_id = list(gb.get_group((env_name,rep,cache_limits[env_name][pct])))[0]
            print(run_id)

            with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
                data = pickle.load(f)

            K = []
            if pct ==100:
                start=999
            else:
                start=795
            for x in range(start,1000):
                print(x)
                kld_, ec_pols,entropy_ = get_KLD(data, env.twoD2oneD(probe_state), x)
                K.append(entropy_.flatten())

            mean_ent = np.nanmean(K, axis=0)

            #std = np.nanstd(K)
            #mean_ent = np.nanmean(K)
            #print(mean_ent)
            avg_entropy[rep].append(np.nan_to_num(np.asarray(K).flatten()))
            #std_entropy[rep].append(std)
    if plottype == 'bar':
        for r, rep in enumerate(reps_to_plot):
            barwidth=0.3
            plt.bar(np.arange(len(pcts_to_plot))+(barwidth*r), avg_entropy[rep], yerr= std_entropy[rep], width=barwidth, color= color_map[rep], alpha=1)
        plt.xticks(np.arange(len(pcts_to_plot))+(barwidth/2), labels=pcts_to_plot)
    elif plottype == 'violin':
        fig, ax = plt.subplots(2,1, sharex=True)
        for r, rep in enumerate(reps_to_plot):
            body = ax[r].violinplot(positions=np.asarray(pcts_to_plot), dataset=avg_entropy[rep], vert=True, widths=15)

            for violinkey in body.keys():
                if violinkey == 'bodies':
                    for b in body['bodies']:
                        b.set_color(convert_rep_to_color[rep])
                        b.set_alpha(75/100)
                else:
                    body[violinkey].set_color(convert_rep_to_color[rep])
            ax[r].set_xticks(pcts_to_plot)
            ax[r].set_xlim([110, 15])
            ax[r].set_ylabel(f'{reps_to_titles[rep]}')
            ax[r].set_yticks([0,.5,1,1.5,2])
    plt.savefig('../figures/CH3/distribution_entropy_allvisits.svg')
    plt.show()

def plot_entropy_hisogram(env_name, reps_to_plot, pcts_to_plot =[100,75,50,25], kld_ent = 'ent',**kwargs):
    plottype = kwargs.get('type','bar')
    convert_rep_to_color = kwargs.get('colors',color_map)
    reps_to_titles = {'onehot':'Unstructured', 'analytic successor':'Structured'}
    probe_state = (13,14)
    E = []
    avg_entropy = {}
    std_entropy = {}
    for rep in reps_to_plot:
        avg_entropy[rep] = []
        std_entropy[rep] = []
        for i, pct in enumerate(pcts_to_plot):
            run_id = list(gb.get_group((env_name,rep,cache_limits[env_name][pct])))[0]
            print(run_id)

            with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
                data = pickle.load(f)

            K = []
            if pct ==100:
                start=999
            else:
                start=795
            for x in range(start,1000):
                print(x)
                kld_, ec_pols,entropy_ = get_KLD(data, env.twoD2oneD(probe_state), x)
                K.append(entropy_.flatten())

            avg_entropy[rep].append(np.nan_to_num(np.asarray(K).flatten()))
    fig, ax = plt.subplots(len(reps_to_plot),len(pcts_to_plot), sharex=True)
    for r, rep in enumerate(reps_to_plot):
        for p, pct in enumerate(pcts_to_plot):
            #body = ax[r].violinplot(positions=np.asarray(pcts_to_plot), dataset=avg_entropy[rep], vert=True, widths=15)
            ax[r,p].hist(avg_entropy[rep][p],color=convert_rep_to_color[rep])

    plt.savefig('../figures/CH3/histo_entropy_allvisits.svg')
    plt.show()

def test_avg_POLmaps(env_name, rep, kld_ent = 'ent'):
    probe_state = (13,14)

    E = []
    for i, pct in enumerate([100,75,50,25]):
        run_id = list(gb.get_group((env_name,rep,cache_limits[env_name][pct])))[0]
        print(run_id)

        with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
            data = pickle.load(f)

        K = []
        count = 0
        start=998
        E = np.zeros((20,20,4))
        for x in range(start,1000):
            print(x)
            kld_, ec_pols,entropy_ = get_KLD(data, env.twoD2oneD(probe_state), x)
            E += ec_pols
            count+= 1

        avg = E/count
        print((avg[0,0,:]))
        plot_pref_pol(env,avg)


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







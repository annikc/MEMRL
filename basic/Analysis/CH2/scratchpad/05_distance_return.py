import numpy as np
import gym
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import pandas as pd
from matplotlib import cm
from scipy.stats import pearsonr
from Analysis.analysis_utils import plot_specs, analysis_specs, get_grids

# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'ec_avg_dist_rtn.csv')
c = analysis_specs
gb = df.groupby(['env_name','representation','EC_cache_limit'])["save_id"]

convert_rep_to_color = plot_specs['rep_colors']
labels_for_plot = plot_specs['labels']
cache_limits = analysis_specs['cache_limits']

def d_r_success_fail(list_of_ids):
    success, fail = [[],[]],[[],[]]
    for j in range(len(list_of_ids)):
        run_id = list_of_ids[j]
        with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
            data = pickle.load(f)

        n_trials = len(data['total_reward'])
        for i in range(100,n_trials):
            dist_returns = data['dist_rtn'][i]

            states_              = dist_returns[0]
            reconstructed_states = dist_returns[1]
            ec_distances         = dist_returns[2]
            computed_returns     = dist_returns[3]


            avg_dist = np.mean(ec_distances)
            avg_rtrn = np.mean(computed_returns)

            if data['total_reward'][i] < -2.49:
                fail[0].append(avg_rtrn)
                fail[1].append(avg_dist)
            else:
                success[0].append(avg_rtrn)
                success[1].append(avg_dist)

    return success, fail

version = 4
env = f'gridworld:gridworld-v{version}1'
envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
pcts_to_plot = [75,50,25]
reps_to_plot = ['random', 'onehot','conv_latents','place_cell','analytic successor']
grids        = get_grids(envs_to_plot)

def plot_success_failure_bars():
    fig, ax = plt.subplots(len(envs_to_plot), 2,sharex='col', gridspec_kw={'width_ratios': [1, 4]})
    for i, env in enumerate(envs_to_plot):
        if env[-2:] == '51':
            rwd_colrow= (16,9)
        else:
            rwd_colrow=(14,14)

        rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
        ax[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[i,0].set_aspect('equal')
        ax[i,0].add_patch(rect)
        ax[i,0].get_xaxis().set_visible(False)
        ax[i,0].get_yaxis().set_visible(False)
        ax[i,0].invert_yaxis()
        for j, rep in enumerate(reps_to_plot):
            for k, pct in enumerate(pcts_to_plot):
                list_of_ids = list(gb.get_group((env, rep, cache_limits[env][pct])))
                s, f = d_r_success_fail(list_of_ids)
                ax[i,1].bar(j+k*0.25,np.mean(s[1]),yerr=np.std(s[1])/np.sqrt(len(s[1])),width=0.1,color=convert_rep_to_color[rep], alpha=pct/100)
                ax[i,1].bar(j+k*0.25+0.1,np.mean(f[1]),yerr=np.std(f[1])/np.sqrt(len(f[1])),width=0.1,color=convert_rep_to_color[rep],alpha=pct/100,hatch='///')
    plt.show()

def plot_success_failure_line():
    fig, ax = plt.subplots(len(envs_to_plot), 2,sharex='col', gridspec_kw={'width_ratios': [1, 4]})
    for i, env in enumerate(envs_to_plot):
        if env[-2:] == '51':
            rwd_colrow= (16,9)
        else:
            rwd_colrow=(14,14)

        rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
        ax[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[i,0].set_aspect('equal')
        ax[i,0].add_patch(rect)
        ax[i,0].get_xaxis().set_visible(False)
        ax[i,0].get_yaxis().set_visible(False)
        ax[i,0].invert_yaxis()
        for j, rep in enumerate(reps_to_plot):
            s_avg, f_avg = [],[]
            s_sem, f_sem = [],[]
            for k, pct in enumerate(pcts_to_plot):
                list_of_ids = list(gb.get_group((env, rep, cache_limits[env][pct])))
                s, f = d_r_success_fail(list_of_ids)
                s_avg.append(np.mean(s[1]))
                s_sem.append(np.std(s[1])/np.sqrt(len(s[1])))
                f_avg.append(np.mean(f[1]))
                f_sem.append(np.std(f[1])/np.sqrt(len(f[1])))

            ax[i,1].errorbar(pcts_to_plot,s_avg,yerr=s_sem,c=convert_rep_to_color[rep],marker='o')
            ax[i,1].errorbar(pcts_to_plot,f_avg,yerr=f_sem,c=convert_rep_to_color[rep],marker='s',linestyle=':')
    plt.show()

plot_success_failure_bars()

def plot_per_rep_one_env():
    fig, ax = plt.subplots(len(reps_to_plot),len(pcts_to_plot),sharey=True,sharex=True)
    for r, rep in enumerate(reps_to_plot):
        for p, pct in enumerate(pcts_to_plot):
            list_of_ids = list(gb.get_group((env, rep, cache_limits[env][pct])))
            s, f = d_r_success_fail(list_of_ids)

            ax[r,p].bar([0],np.mean(s[1]),yerr=np.std(s[1]),width=0.75,color=convert_rep_to_color[rep])
            ax[r,p].bar([1],np.mean(f[1]),yerr=np.std(f[1]),width=0.75,color=convert_rep_to_color[rep],hatch='//')
            if r == 0:
                ax[r,p].set_title(f'{pct}')
            if r == len(reps_to_plot)-1:
                ax[r,p].set_xticks([0,1])
                ax[r,p].set_xticklabels(['Success', 'Fail'])
        ax[r,0].set_ylabel('Average \nRecall \nDistance')

    plt.show()

def d_r_over_pct_rep():
    fig, ax = plt.subplots(len(pcts_to_plot),len(reps_to_plot),sharex=True,sharey=True)
    # row 0 = over pcts
    # row 1 = over reps
    for j, rep in enumerate(reps_to_plot):

        for p, pct in enumerate(pcts_to_plot):
            run_id = list(gb.get_group((env_name, rep, cache_limits[env_name][pct])))[0]
            with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
                data = pickle.load(f)

            n_trials = len(data['total_reward'])
            d, r = [], []
            for i in range(100,n_trials):
                dist_returns = data['dist_rtn'][i]

                states_              = dist_returns[0]
                reconstructed_states = dist_returns[1]
                ec_distances         = dist_returns[2]
                computed_returns     = dist_returns[3]


                avg_dist = np.mean(ec_distances)
                avg_rtrn = np.mean(computed_returns)
                d.append(avg_dist)
                r.append(avg_rtrn)


            m,b = np.polyfit(d,r,1)
            pr_, pv_ = pearsonr(d,r)

            ax[p,j].scatter(d,r,c=convert_rep_to_color[rep], alpha=0.2)
            ax[p,j].plot(d, m*np.asarray(d)+b,label=f'{labels_for_plot[rep]}',c=convert_rep_to_color[rep])
            ax[p,j].annotate(f'{pr_:.2f},{pv_:.2f}',(1,10))
            #ax[1,j].scatter(d,r,c=convert_rep_to_color[rep])
            ax[p,0].set_ylabel(f'{pct}')
        ax[0,j].set_title(f'{labels_for_plot[rep]}')
        #ax[0,j].set_xlim([0,1])
        ax[0,j].set_ylim([0,10])
    plt.show()

def scatter_hist(x, y, ax, ax_histx, ax_histy, color):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y,color=color,alpha=0.2)
    ax.set_xlim([min(x),max(x)])
    ax.set_ylim([min(y),max(y)])

    # now determine nice limits by hand:
    binwidth = 0.2
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins,color=color, alpha=0.2)
    ax_histx.set_title('R')
    ax_histy.set_title('D')
    ax_histy.hist(y, bins=bins, orientation='horizontal',color=color,alpha=0.2)

def make_env_graph(env):
    action_cols = ['orange','red','green','blue']
    G = nx.DiGraph() # why directed/undirected graph?

    for action in range(env.action_space.n):
        # down, up, right, left
        for state2d in env.useable:
            state1d = env.twoD2oneD(state2d)
            next_state = np.where(env.P[action,state1d,:]==1)[0]
            if len(next_state) ==0:
                pass
            else:
                for sprime in next_state:
                    edge_weight = env.P[action,state1d,sprime]
                    G.add_edge(state1d, sprime,color=action_cols[action],weight=edge_weight)

    return G

def compute_distance_matrix(G, env):
    x = nx.shortest_path(G)
    useable_1d = [env.twoD2oneD(x) for x in env.useable]
    shortest_dist_array = np.zeros((env.nstates,env.nstates))
    shortest_dist_array[:]=np.nan

    for start in useable_1d:
        for target in list(x[start].keys()):
            shortest_dist_array[start][target]= len(x[start][target])-1

    return shortest_dist_array

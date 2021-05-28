import numpy as np
import gym
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import pandas as pd
from matplotlib import cm
from scipy.stats import pearsonr

# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'ec_avg_dist_rtn.csv')

gb = df.groupby(['env_name','representation','EC_cache_limit'])["save_id"]

labels_for_plot = {'analytic successor':'SR', 'onehot':'onehot', 'random':'random','place_cell':'PC','conv_latents':'latent'}

convert_rep_to_color = {'analytic successor':'C0',
                        'onehot':'C1',
                        'random':'C2',
                        'place_cell':'C4',
                        'conv_latents':'C3'}


# get cache limit sizes for the restriction conditions -- different for each environment
cache_limits = {'gridworld:gridworld-v11':{100:400, 75:300, 50:200, 25:100},
                'gridworld:gridworld-v31':{100:365, 75:273, 50:182, 25:91},
                'gridworld:gridworld-v41':{100:384, 75:288, 50:192, 25:96},
                'gridworld:gridworld-v51':{100:286, 75:214, 50:143, 25:71}}


version = 3
env_name = f'gridworld:gridworld-v{version}1'

pcts_to_plot = [100,75,50,25]
reps_to_plot = ['random', 'onehot','conv_latents','place_cell','analytic successor'] # df.representation.unique()
rep_labels = [labels_for_plot[x] for x in reps_to_plot]
tmp_env_obj = gym.make(env_name)
plt.close()
e_grid = tmp_env_obj.grid

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

convert_rep_to_color = {'analytic successor':'C0',
                        'onehot':'C1',
                        'random':'C2',
                        'place_cell':'C4',
                        'conv_latents':'C3'}

version = 3
env_name = f'gridworld:gridworld-v{version}1'
pct = 50

plt.figure()
for rep in ['conv_latents', 'analytic successor', 'onehot']:

    run_id = list(gb.get_group((env_name, rep, cache_limits[env_name][pct])))[0]
    with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
        data = pickle.load(f)

    n_trials = len(data['total_reward'])

    avg_d, avg_r = [],[]
    for i in range(100,n_trials):
        dist_returns = data['dist_rtn'][i]

        states_              = dist_returns[0]
        reconstructed_states = dist_returns[1]
        ec_distances         = dist_returns[2]
        computed_returns     = dist_returns[3]
        '''
        if i == 0:
            d = list(ec_distances)
            r = list(computed_returns)
        else:
            d += list(ec_distances)
            r += list(computed_returns)
        '''

        avg_d.append(np.mean(ec_distances))
        avg_r.append(np.mean(computed_returns))


    plt.scatter(avg_d,avg_r,c=convert_rep_to_color[rep], alpha=0.5)

plt.show()
'''
m,b = np.polyfit(d,r,1)
pr_, pv_ = pearsonr(d,r)
y = np.asarray(d)
x = np.asarray(r)
fig, ax = plt.subplots(2,1)
#plt.scatter(x,y)
ax[0].hist(y, bins=6)
ax[1].hist(x, bins=10)'''
plt.show()


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

'''
# start with a square Figure
fig = plt.figure(figsize=(8, 8))

# Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0],sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1],sharey=ax)
for rep in [reps_to_plot[0]]:
    print(rep)
    run_id = list(gb.get_group((env_name, rep, cache_limits[env_name][pct])))[0]
    with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
        data = pickle.load(f)

    n_trials = len(data['total_reward'])

    for i in range(100,n_trials):
        dist_returns = data['dist_rtn'][i]

        states_              = dist_returns[0]
        reconstructed_states = dist_returns[1]
        ec_distances         = dist_returns[2]
        computed_returns     = dist_returns[3]


        #avg_dist = np.mean(ec_distances)
        #avg_rtrn = np.mean(computed_returns)
        #d.append(avg_dist)
        #r.append(avg_rtrn)
        if i == 100:
            d = list(ec_distances)
            r = list(computed_returns)
        else:
            d += list(ec_distances)
            r += list(computed_returns)


    m,b = np.polyfit(d,r,1)
    pr_, pv_ = pearsonr(d,r)
    y = np.asarray(d)
    x = np.asarray(r)

    # use the previously defined function
    scatter_hist(x, y, ax, ax_histx, ax_histy, color=convert_rep_to_color[rep])

plt.show()







env = gym.make(f'gridworld:gridworld-v{version}1')
G= make_env_graph(env)
shortest_dist_array = compute_distance_matrix(G,env)
graphdist = nx.shortest_path(G)

n_trials = len(data['total_reward'])
d, r = [], []
viridis = cm.get_cmap('viridis', n_trials)
nums = np.linspace(0,1,n_trials)
cs = [viridis(n) for n in nums]

n_trials = len(data['total_reward'])
d, r = [], []
for i in range(n_trials):
    dist_returns = data['dist_rtn'][i]

    states_              = dist_returns[0]
    reconstructed_states = dist_returns[1]
    ec_distances         = dist_returns[2]
    computed_returns     = dist_returns[3]


    avg_dist = np.mean(ec_distances)
    avg_rtrn = np.mean(computed_returns)
    d.append(avg_dist)
    r.append(avg_rtrn)


plt.figure()
ax1 = plt.scatter(d,r,c=cs)
plt.colorbar(ax1)
plt.show()'''

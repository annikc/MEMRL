import numpy as np
import gym
import networkx as nx
import matplotlib.pyplot as plt
from modules.Utils.gridworld_plotting import opt_pol_map, plot_pref_pol
from Analysis.analysis_utils import make_env_graph, compute_graph_distance_matrix

def plot_world(world, **kwargs):
    scale = kwargs.get('scale', 0.35)
    title = kwargs.get('title', 'Grid World')
    ax_labels = kwargs.get('ax_labels', False)
    state_labels = kwargs.get('states', False)
    invert_ = kwargs.get('invert', False)
    if invert_:
        cmap = 'bone'
    else:
        cmap = 'bone_r'
    r,c = world.shape

    G = make_env_graph(env)
    sp = nx.shortest_path(G)
    gd = compute_graph_distance_matrix(G, world)

    fig = plt.figure(figsize=(c*scale, r*scale))
    ax = fig.add_subplot(1,1,1)

    gridMat = np.zeros(world.shape)
    for i, j in world.obstacle2D:
        gridMat[i, j] = 1.0
    for i, j in world.terminal2D:
        gridMat[i, j] = 0.2
    ax.pcolor(world.grid, edgecolors='k', linewidths=0.75, cmap=cmap, vmin=0, vmax=1)

    U = np.zeros((r, c))
    V = np.zeros((r, c))
    U[:] = np.nan
    V[:] = np.nan

    if len(world.action_list) >4 :
        if world.jump is not None:
            for (a, b) in world.jump.keys():
                (a2, b2) = world.jump[(a, b)]
                U[a, b] = (b2 - b)
                V[a, b] = (a - a2)

    C, R = np.meshgrid(np.arange(0, c) + 0.5, np.arange(0, r) + 0.5)
    ax.quiver(C, R, U, V, scale=1, units='xy')

    for rwd_loc in world.rewards.keys():
        rwd_r, rwd_c = rwd_loc
        if world.rewards[rwd_loc] < 0:
            colorcode = 'red'
        else:
            colorcode = 'darkgreen'
        ax.add_patch(plt.Rectangle((rwd_c, rwd_r), width=1, height=1, linewidth=2, facecolor=colorcode, alpha=0.5))

    if state_labels:
        for (i,j) in world.useable:
            # i = row, j = col
            oneD = world.twoD2oneD((i,j))
            #ax.text(j+0.5,i+0.7, s=f'{oneD}', ha='center')
            ax.text(j+0.5,i+0.7, s=f'{gd[oneD,105]}', ha='center')


    #ax.set_xticks([np.arange(c) + 0.5, np.arange(c)])
    #ax.set_yticks([np.arange(r) + 0.5, np.arange(r)])
    ax.invert_yaxis()
    ax.set_aspect('equal')
    if not ax_labels:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    ax.set_title(title)

    return fig, ax

def softmax(x, T=1):
	e_x = np.exp((x - np.nanmax(x))/T)
	return np.round(e_x / np.nansum(e_x,axis=0),8)

env_name = 'gridworld:gridworld-v3'
env = gym.make(env_name)
plt.close()
'''
fig, ax = plot_world(env, states=True)
plt.show()

G = make_env_graph(env)
sp = nx.shortest_path(G)
gd = compute_graph_distance_matrix(G, env)

num_steps_to_rwd = np.zeros((20,20))
num_steps_to_rwd[:] = np.nan
for (i,j) in env.useable:
    # i = row, j = col
    oneD = env.twoD2oneD((i,j))
    num_steps_to_rwd[i,j] = gd[oneD,105]

plt.imshow(num_steps_to_rwd)
plt.show()
'''

def attempt_opt_pol(env):
    rwd_loc = env.twoD2oneD(list(env.rewards.keys())[0])
    G = make_env_graph(env)
    sp = nx.shortest_path(G)
    gd = compute_graph_distance_matrix(G, env)

    num_steps_to_rwd = np.zeros((20,20))
    num_steps_to_rwd[:] = np.nan
    for (i,j) in env.useable:
        # i = row, j = col
        oneD = env.twoD2oneD((i,j))
        num_steps_to_rwd[i,j] = gd[oneD,rwd_loc]

    opt_pol_matrix = np.zeros((20,20,4))
    opt_pol_matrix[:] = np.nan
    for (r,c) in env.useable:
        index = env.twoD2oneD((r,c))
        steps_from_index = num_steps_to_rwd[r,c]

        if r+1 < 20:
            state_down = (r+1,c)
            steps_down = num_steps_to_rwd[state_down]
            if steps_down < steps_from_index:
                opt_pol_matrix[r,c,0] = 1

        if r-1 >= 0:
            state_up = (r-1,c)
            index_up = env.twoD2oneD(state_up)
            steps_up = num_steps_to_rwd[state_up]
            if steps_up < steps_from_index:
                opt_pol_matrix[r,c,1] = 1

        if c+1 <20:
            state_right = (r,c+1)
            steps_right = num_steps_to_rwd[state_right]
            if steps_right < steps_from_index:
                opt_pol_matrix[r,c,2] = 1

        if (c-1)>=0:
            state_left =(r,c-1)
            steps_left = num_steps_to_rwd[state_left]
            if steps_left < steps_from_index:
                opt_pol_matrix[r,c,3] = 1

        opt_pol_matrix[r,c,:] = np.nan_to_num(softmax(opt_pol_matrix[r,c,:]))

    return  opt_pol_matrix
#opt_pol = attempt_opt_pol(env)

#op = opt_pol_map(gym.make('gridworld:gridworld-v1'))
#op[9:,:] = [0., 0.,0,0]
#plot_pref_pol(env, opt_pol)
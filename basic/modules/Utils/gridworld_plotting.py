'''
Set up functions for plotting gridworld environment
TODO: Interactive policy plotting -- mpld3?

Author: Annik Carson
-- July 2020
'''

# =====================================
#           IMPORT MODULES            #
# =====================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.cm as cmx

from ..Utils import softmax
from scipy.stats import entropy

# =====================================
#              FUNCTIONS
# =====================================
def plot_world(world, plotNow=False, current_state = False, **kwargs):
    scale = kwargs.get('scale', 1)
    title = kwargs.get('title', 'Grid World')
    ax_labels = kwargs.get('ax_labels', False)
    state_labels = kwargs.get('states', False)
    invert_ = kwargs.get('invert', False)
    if invert_:
        cmap = 'bone'
    else:
        cmap = 'bone_r'
    r,c = world.shape

    fig = plt.figure(figsize=(c*scale, r*scale))

    gridMat = np.zeros(world.shape)
    for i, j in world.obstacle2D:
        gridMat[i, j] = 1.0
    for i, j in world.terminal2D:
        gridMat[i, j] = 0.2
    plt.pcolor(world.grid, edgecolors='k', linewidths=0.75, cmap=cmap, vmin=0, vmax=1)

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
    plt.quiver(C, R, U, V, scale=1, units='xy')

    for rwd_loc in world.rewards.keys():
        rwd_r, rwd_c = rwd_loc
        if world.rewards[rwd_loc] < 0:
            colorcode = 'red'
        else:
            colorcode = 'darkgreen'
        plt.gca().add_patch(plt.Rectangle((rwd_c, rwd_r), width=1, height=1, linewidth=2, facecolor=colorcode, alpha=0.5))

    if current_state:
        agent_r, agent_c = world.oneD2twoD(world.state)
        agent_dot = plt.Circle((agent_c + .5, agent_r + .5), 0.35, fc='b') ## plot functions use x,y we use row(y), col(x)
        plt.gca().add_patch(agent_dot)

    if state_labels:
        for (i,j) in world.useable:
            # i = row, j = col
            plt.annotate(f'{world.twoD2oneD((i,j))}', (j+0.3,i+0.7))


    plt.xticks(np.arange(c) + 0.5, np.arange(c))
    plt.yticks(np.arange(r) + 0.5, np.arange(r))
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    if not ax_labels:
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
    plt.title(title)
    if plotNow:
        plt.show()
    return fig

def plot_valmap(maze, value_array, save=False, **kwargs):
    '''
    :param maze: the environment object
    :param value_array: array of state values
    :param save: bool. save figure in current directory
    :return: None
    '''
    show = kwargs.get('show', True)
    directory = kwargs.get('directory', './figures/')
    title = kwargs.get('title', 'State Value Estimates')
    filetype = kwargs.get('filetype', 'svg')
    vals = value_array.copy()
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_axes([0, 0, 0.85, 0.85])
    axc = fig.add_axes([0.75, 0, 0.05, 0.85])
    vmin, vmax = kwargs.get('v_range', [0, 1])
    cmap = plt.cm.Spectral_r
    cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
    for i in maze.obstacles_list:
        vals[i] = np.nan
    cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
    ax1.pcolor(vals, cmap=cmap, vmin=vmin, vmax=vmax)

    # add patch for reward location/s (red)
    for rwd_loc in maze.rewards:
        rwd_y, rwd_x = rwd_loc
        ax1.add_patch(plt.Rectangle((rwd_y, rwd_x), width=0.99, height=1, linewidth=1, ec='white', fill=False))

    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    ax1.set_title(title)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    if save:
        plt.savefig(f'{directory}v_{title}.{filetype}', format=f'{filetype}', bbox_inches='tight')
    if show:
        plt.show()

    plt.close()

def plot_polmap(maze, policy_array, save=False, **kwargs):
    '''
    :param maze: the environment object
    :param save: bool. save figure in current directory
    :return: None
    '''
    show = kwargs.get('show', True)
    directory = kwargs.get('directory', './figures/')
    title = kwargs.get('title', 'Most Likely Action from Policy')
    filetype = kwargs.get('filetype', 'svg')
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_axes([0, 0, 0.85, 0.85])
    axc = fig.add_axes([0.75, 0, 0.05, 0.85])

    cmap = plt.cm.Spectral_r
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    # make base grid
    ax1.pcolor(maze.grid, vmin=0, vmax=1, cmap='bone')
    # add patch for reward location/s (red)
    for rwd_loc in maze.rewards:
        rwd_y, rwd_x = rwd_loc
        ax1.add_patch(plt.Rectangle((rwd_y, rwd_x), width=0.99, height=1, linewidth=1, ec='white', fill=False))

    chance_threshold = kwargs.get('threshold', 0.18)  # np.round(1 / len(maze.actionlist), 6)

    cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
    for i in range(maze.c):
        for j in range(maze.r):
            action = np.argmax(tuple(policy_array[i][j]))
            prob = max(policy_array[i][j])

            dx1, dy1, head_w, head_l = make_arrows(action, prob)
            if prob > chance_threshold:
                if (dx1, dy1) == (0, 0):
                    pass
                else:
                    colorVal1 = scalarMap.to_rgba(prob)
                    ax1.arrow(j + 0.5, i + 0.5, dx1, dy1, head_width=0.3, head_length=0.2, color=colorVal1)
            else:
                pass
    ax1.set_aspect('equal')
    ax1.set_title(title)
    ax1.invert_yaxis()
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    if save:
        plt.savefig(f'{directory}p_{title}.{filetype}', format=f'{filetype}', bbox_inches='tight')
    if show:
        plt.show()
    #plt.close()

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

def plot_softmax(x, T=1):
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].bar(np.arange(len(x)), x)
    y = softmax(x, T)
    axarr[1].bar(np.arange(len(x)), y)
    plt.show()

def opt_pol_map(env):
    optimal_policy = np.zeros((env.y, env.x, len(env.actionlist)))
    for location in env.useable:
        xdim, ydim = location
        xrwd, yrwd = env.rwd_loc[0]

        if xdim < xrwd:
            optimal_policy[ydim, xdim][1] = 1
            if ydim < yrwd:
                optimal_policy[ydim, xdim][3] = 1
            elif ydim > yrwd:
                optimal_policy[ydim, xdim][0] = 1

        elif xdim > xrwd:
            optimal_policy[ydim, xdim][2] = 1
            if ydim < yrwd:
                optimal_policy[ydim, xdim][3] = 1
            elif ydim > yrwd:
                optimal_policy[ydim, xdim][0] = 1
        else:
            if ydim < yrwd:
                optimal_policy[ydim, xdim][3] = 1
            elif ydim > yrwd:
                optimal_policy[ydim, xdim][0] = 1
            else:
                optimal_policy[ydim, xdim][5] = 1

        optimal_policy[ydim, xdim] = softmax(optimal_policy[ydim, xdim], T=0.01)

    return optimal_policy


######################
def plot_pref_pol(maze, policy_array, save=False, **kwargs):
    '''
        :param maze: the environment object
        :param save: bool. save figure in current directory
        :return: None
        '''
    show = kwargs.get('show', True)
    title = kwargs.get('title', 'Policy Entropy')
    directory = kwargs.get('directory', './figures/')
    filetype = kwargs.get('filetype', 'svg')
    vmax = kwargs.get('upperbound', 2)
    rewards = kwargs.get('rwds', maze.rewards)
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_axes([0, 0, 0.85, 0.85])
    axc = fig.add_axes([0.75, 0, 0.05, 0.85])

    cmap = plt.cm.Spectral_r
    cNorm = colors.Normalize(vmin=0, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
    # make base grid
    ax1.pcolor(maze.grid, vmin=0, vmax=vmax, cmap='bone')
    # add patch for reward location/s (red)
    for rwd_loc in rewards:
        rwd_r, rwd_c = rwd_loc
        ax1.add_patch(plt.Rectangle((rwd_c, rwd_r), width=0.99, height=1, linewidth=1, ec='white', fill=False))

    chance_threshold = kwargs.get('threshold', 0.18)  # np.round(1 / len(maze.actionlist), 6)

    for i in range(maze.r):
        for j in range(maze.c):
            policy = tuple(policy_array[i, j])

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
                colorVal1 = scalarMap.to_rgba(entropy(policy))
                if entropy(policy) > 1.2:
                    pass
                else:
                    ax1.arrow(j + 0.5, i + 0.5, dx, dy, head_width=0.3, head_length=0.5, color=colorVal1)


    ax1.set_aspect('equal')
    ax1.set_title(title)
    ax1.invert_yaxis()
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    if save:
        plt.savefig(f'{directory}pref_{title}.{filetype}', format=f'{filetype}', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_optimal(maze, policy_array, save=False, **kwargs):
    '''
    :param maze: the environment object
    :param save: bool. save figure in current directory
    :return: None
    '''
    show = kwargs.get('show', True)
    title = kwargs.get('title', 'Most Likely Action from Policy')
    directory = kwargs.get('directory', '../data/figures/')
    filetype = kwargs.get('filetype', 'png')
    rewards = kwargs.get('rwds', maze.rewards)
    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_axes([0, 0, 0.85, 0.85])
    axc = fig.add_axes([0.75, 0, 0.05, 0.85])

    cmap = plt.cm.Spectral_r
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
    # make base grid
    ax1.pcolor(maze.grid, vmin=0, vmax=1, cmap='bone')
    # add patch for reward location/s (red)
    for rwd_loc in rewards:
        rwd_r, rwd_c = rwd_loc
        ax1.add_patch(plt.Rectangle((rwd_c, rwd_r), width=0.99, height=1, linewidth=1, ec='white', fill=False))

    chance_threshold = kwargs.get('threshold',0.18)  #np.round(1 / len(maze.actionlist), 6)


    for i in range(maze.r):
        for j in range(maze.c):
            policy = tuple(policy_array[i,j])

            dx, dy = 0,0
            for ind, k in enumerate(policy):
                if i == 0 and j ==0:
                    print(ind,k)
                action = ind
                prob = k
                if prob < 0.01:
                    pass
                else:
                    dx1, dy1, head_w, head_l = make_arrows(action, prob)
                    dx += dx1
                    dy += dy1
            colorVal1 = scalarMap.to_rgba(entropy(policy))
            ax1.arrow(j+0.5, i+0.5, dx/2, dy/2, head_width=0.25, head_length=0.5, color=colorVal1)

    ax1.set_aspect('equal')
    ax1.set_title(title)
    ax1.invert_yaxis()

    if save:
        plt.savefig(f'{directory}{title}.{filetype}', format=f'{filetype}', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

env = gym.make('gridworld:gridworld-v4')

def compute_return(reward_vec,gamma):
    running_add = 0
    returns = []
    for t in reversed(range(len(reward_vec))):
        running_add = running_add*gamma + reward_vec[t]
        returns.insert(0, running_add)

    return returns

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
            ax.annotate(f'{world.twoD2oneD((i,j))}', (j+0.3,i+0.7))


    #ax.set_xticks([np.arange(c) + 0.5, np.arange(c)])
    #ax.set_yticks([np.arange(r) + 0.5, np.arange(r)])
    ax.invert_yaxis()
    ax.set_aspect('equal')
    if not ax_labels:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    ax.set_title(title)

    return fig, ax

fig, ax = plot_world(env)
num_steps  = [(2,2),(2,3),(2,4),(2,5),(3,5),(4,5),(5,5)]
for i, step in enumerate(num_steps):
    c,r = step
    ax.add_patch(patches.Circle((c + .5, r + .5), 0.35, fc='b', alpha =i/len(num_steps)))

plt.show()

rewards = -0.01*np.ones(7)
rewards[-1] = 10
print(rewards)
print(compute_return(rewards,gamma=0.98))

from modules.Utils import softmax
print(softmax((10,8.76,0,9.2),T=1))
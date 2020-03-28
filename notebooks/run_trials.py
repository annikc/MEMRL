from __future__ import division
from modules import *
fig_savedir = '../data/figures/'

grid_params = {
    'y_height': 100,
    'x_width': 100,
    'walls': False,
    'rho': 0.0,
    'maze_type': 'none',
    'barheight': 4,
    'port_shift': 'none',
    'step_penaliz': -0.01

}

# make environment
maze = eu.gridworld(grid_params)
agent_params = {
    'load_model': False,
    'load_dir': '../data/outputs/gridworld/openfield{}{}.pt'.format(grid_params['x_width'], grid_params['y_height']),
    'rwd_placement': 'training_loc',

    'state_type': 'conv',
    'lin_dims': 500,
    'rfsize': 20,
    'action_dims': len(maze.actionlist),
    'temperature': 1,

    'batch_size': 1,
    'gamma': 0.98,  # discount factor
    'eta': 5e-4,

    'use_EC': True,
    'cachelim': 300,  # memory limit should be ~75% of #actions x #states
    'EC': {},
    'mem_temp': 0.3

}
run_dict = ac.reset_agt(maze, agent_params)

if agent_params['use_EC']:
    # agent_params['cachelim'] = int(0.5*np.prod(maze.grid.shape))
    agent_params['EC'] = ec.ep_mem(run_dict['agent'], agent_params['cachelim'])

#gp.plot_env(run_dict['environment'])
#run_dict = ac.reset_agt(run_dict['environment'], agent_params)
#expt.run(run_dict,full=False, rec_mem=False, use_EC=False) ## by default runs truncated trials with MF only
#ac.torch.save(run_dict['agent'],agent_params['load_dir'])

def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    return img

# get weights from conv layer and normalize them for visualization
weights = normalize_output(run_dict['agent'].hidden[0].weight.data)
print(weights.shape)

n_filters, ix = , 1
for i in range(n_filters):
    # get filter:
    f = weights[i,:,:,:]
    for j in range(3):
        ax = plt.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[j,:,:], cmap ='gray')
        ix+=1
plt.show()



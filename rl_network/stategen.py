'''
Plot functions used for AC Agent in RL gridworld task
Author: Annik Carson 
-- June 2018
'''

# =====================================
#           IMPORT MODULES            #
# =====================================
from __future__ import division, print_function
import actorcritic as ac
import sys
sys.path.insert(0,'../environments/'); import gridworld_plotting as gp


import numpy as np
np.random.seed(12345)

class PlaceCells(object):
    def __init__(self, num_cells, grid, **kwargs):
        self.num_cells = num_cells
        self.gx, self.gy = grid.x, grid.y
        self.field_width = kwargs.get('fwhm', 0.25)

        self.x = np.random.uniform(0,1,(self.num_cells,))
        self.y = np.random.uniform(0,1,(self.num_cells,))

    def activity(self, position):
        x0 = (position[0]/self.gx)+float(1/(2*self.gx))
        y0 = (position[1]/self.gy)+float(1/(2*self.gy))

        pc_activities = (np.exp(-((self.x-x0)**2 + (self.y-y0)**2) / self.field_width**2))
        state_vec = pc_activities.round(decimals = 10)
        return np.reshape(state_vec, (1,len(state_vec)))


class SceneFrame(object):
    def __init__(self, grid, agent, reward, **kwargs):
        self.grid = grid
        self.agent_pos = agent
        self.rwd_pos = reward



def get_frame(maze, **kwargs):
    agent_location  = kwargs.get('agtlocation', maze.cur_state)
    reward_location = kwargs.get('rwdlocation', maze.rwd_loc[0])
    state_type      = kwargs.get('state_type', 'conv')
    #grid
    grid = maze.grid
    #location of reward
    rwd_position = np.zeros_like(maze.grid)
    rwd_position[reward_location[1], reward_location[0]] = 1
    #location of agent
    agt_position = np.zeros_like(maze.grid)
    agt_position[agent_location[1], agent_location[0]] = 1

    if state_type == 'pcs':
        return np.array((grid, rwd_position, agt_position)) #np.transpose(np.array((grid, rwd_position, agt_position)),axes=[1,2,0])
    else:
        return np.array([(grid, rwd_position, agt_position)])

def gen_input(maze, agt_dictionary, **kwargs):
    state_type = kwargs.get('state_type', agt_dictionary['state_type'])
    if state_type == 'pcs':
        # place cell parameters
        num_pc = 1000
        fwhm = 0.05
        pcs = PlaceCells(num_cells=num_pc, grid=maze, fwhm=fwhm)
        gp.make_env_plots(maze,env=True,pc_map=True,pcs=pcs, save=False)

        agt_dictionary['pcs'] = pcs
        agt_dictionary['input_dims'] = num_pc
        agt_dictionary['hid_types']  = ['linear']
        agt_dictionary['hid_dims']   = [500]


    elif state_type == 'conv':
        #gp.make_env_plots(maze,env=True,save=True)
        num_channels = 3
        agt_dictionary['num_channels'] = num_channels
        if maze.bound:
            agt_dictionary['input_dims'] = (maze.y+2, maze.x+2, agt_dictionary['num_channels'])
        else:
            agt_dictionary['input_dims'] = (maze.y, maze.x, agt_dictionary['num_channels'])


            hidden_layer_types = kwargs.get('hid_types', ['conv', 'pool', 'linear'])
        agt_dictionary['hid_types'] = hidden_layer_types
        for ind, i in enumerate(hidden_layer_types):
            if ind == 0:
                agt_dictionary['hid_dims'] = [ac.conv_output(agt_dictionary['input_dims'])]
            else:
                if i == 'conv' or i == 'pool':
                    agt_dictionary['hid_dims'].append(ac.conv_output(agt_dictionary['hid_dims'][ind-1]))
                elif i == 'linear':
                    agt_dictionary['hid_dims'].append(agt_dictionary['lin_dims'])

    agt_dictionary['maze'] = maze

    return agt_dictionary
	    
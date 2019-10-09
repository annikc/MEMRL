'''
Set up object classes for gridworld environment
gridworld class defines the environment and available actions, reward function, etc.

gymworld makes step function more consistent with the OpenAI gymworld

Author: Annik Carson
-- Oct 2019
'''

# =====================================
#           IMPORT MODULES            #
# =====================================
from __future__ import division, print_function
import numpy as np
np.random.seed(12345)

class gridworld(object): 
    def __init__(self, grid_params, **kwargs):
        self.y 				= grid_params['y_height']
        self.x 				= grid_params['x_width']
        self.rho 			= grid_params['rho']
        self.bound 			= grid_params['walls']
        self.maze_type 		= grid_params['maze_type']
        self.port_shift		= grid_params['port_shift']

        self.actionlist 	= kwargs.get('actionlist', ['N', 'E', 'W', 'S', 'stay', 'poke'])
        self.rwd_action 	= kwargs.get('rewarded_action', 'poke')

        self.step_penalization = kwargs.get('pen', 0)

        self.barheight 		= kwargs.get('barheight',11)

        self.grid, self.useable, self.obstacles = self.grid_maker()

        if self.maze_type == 'tmaze':
            self.rwd_loc 		= [self.useable[0]]

        if self.maze_type == 'triple_reward':
            self.rwd_loc 		= [(self.x-1, 0), (self.x-1, self.y-1), (0, self.y-1)]
            self.orig_rwd_loc 	= [(self.x-1, 0), (self.x-1, self.y-1), (0, self.y-1)]
            self.starter 		= kwargs.get('t_r_start', (0,0))

        else:
            rwd_choice 			= np.random.choice(len(self.useable))
            self.rwd_loc 		= [self.useable[rwd_choice]]
            self.orig_rwd_loc 	= []

        start_choice 	= np.random.choice(len(self.useable))
        self.start_loc 	= self.loc_picker()
        print(self.start_loc) #self.useable[start_choice]
        if self.maze_type=='triple_reward':
            self.start_loc = self.starter

        self.reset()

        self.empty_map = self.make_map(self.grid, False)

        ## OpenAI gym bits
        self.action_space = action_wrapper(self.actionlist)

    def grid_maker(self):
        '''
        Default grid is empty -- all squares == 0
        If env_type given, set some squares to == 1
                (agent cannot occupy these squares)
        In the case of the T maze, set all squares =1
                and just rewrite which squares the agent may
                occupy (i.e. grid = 0 at these points)
        '''

        env_types = ['none', 'bar','room','tmaze', 'triple_reward']
        if self.maze_type not in env_types:
            print("Environment Type '{0}' Not Recognized. \nOptions are: {1} \nDefault is Open Field (maze_type = 'none')".format(self.maze_type, env_types))

        grid = np.zeros((self.y,self.x), dtype=int)

        if self.maze_type == 'bar':
            self.rho = 0
            barheight = self.barheight
            for i in range(self.x-4):
                grid[barheight][i+2] = 1

        elif self.maze_type == 'room':
            self.rho = 0
            vwall = int(self.x/2)
            hwall = int(self.y/2)

            #make walls
            for i in range(self.x):
                grid[vwall][i] = 1
            for i in range(self.y):
                grid[i][hwall] = 1

            # make doors
            grid[vwall][np.random.choice(np.arange(0,vwall))] = 0
            grid[vwall][np.random.choice(np.arange(vwall+1, self.x))] = 0

            grid[np.random.choice(np.arange(0,hwall))][hwall] = 0
            grid[np.random.choice(np.arange(hwall+1, self.y))][hwall] = 0

        elif self.maze_type == 'tmaze':
            self.rho = 0
            self.possible_ports = []
            grid = np.ones((self.y, self.x), dtype=int)
            h1, v1 = int(self.x/2), 0
            if h1%2==0:
                for i in range(self.x):
                    grid[v1][i] = 0
                    if i == 0:
                        self.possible_ports.append((i,v1))
                    elif i == self.x-1:
                        self.possible_ports.append((i,v1))
            else:
                for i in range(self.x):
                    grid[v1][i] = 0
                    if i == 0:
                        self.possible_ports.append((i,v1))
                    elif i == self.x-1:
                        self.possible_ports.append((i,v1))

            if self.y > int(self.x/2):
                for i in range(self.y):
                    grid[i][h1] = 0
                    if i == self.y-1:
                        self.possible_ports.append((h1,i))
            else:
                for i in range(self.y):
                    grid[i][h1] = 0
                    if i == self.y-1:
                        self.possible_ports.append((h1,i))

        # for env_type = none, can set randomized obstacles on the grid, specified by density rho
        if self.rho != 0:
            maze = np.vstack([[np.random.choice([0,1], p = [1-self.rho, self.rho]) for _ in range(self.x)] for _ in range(self.y)])
            grid = grid + maze

        if self.bound:
            grid_bound = np.ones((self.y+2, self.x+2), dtype=int)
            grid_bound[1:-1][:,1:-1] = grid
            grid = grid_bound


        # lists of tuples storing locations of open grid space and obstacles (unusable grid space)
        useable_grid = list(zip(np.where(grid==0)[1], np.where(grid==0)[0]))
        obstacles = list(zip(np.where(grid==1)[1], np.where(grid==1)[0]))

        return grid, useable_grid, obstacles

    def make_map(self, grid, pol=False):
        '''
        Set up a map for the agent to record its policy and value
            estimates as it navigates the grid
        '''
        if pol:
            pv_map = np.zeros(grid.shape, dtype=[('N', 'f8'), ('E', 'f8'),('W', 'f8'), ('S', 'f8'),('stay', 'f8'), ('poke', 'f8')])
            pv_map[grid == 1] = (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)

        else:
            pv_map = np.zeros(grid.shape)
            pv_map[grid == 1] = np.nan
        return pv_map

    def loc_picker(self):
        start_buffer = 5
        a1 = np.random.choice([-1,1])
        start_x = self.rwd_loc[0][0] + np.random.choice([-1, 1])*np.random.choice(start_buffer)
        if start_x < 0:
            start_x = 0
        elif start_x > self.grid.shape[1] - 1:
            start_x = self.grid.shape[1] - 1
        start_y = self.rwd_loc[0][1] + np.random.choice([-1, 1])*np.random.choice(start_buffer)
        if start_y < 0:
            start_y = 0
        elif start_y > self.grid.shape[1] - 1:
            start_y = self.grid.shape[1] - 1

        return (start_x, start_y)

    def reset(self):
        #start_choice = np.random.choice(len(self.useable))
        #self.start_loc = self.useable[start_choice]
        self.start_loc = self.loc_picker()


        self.cur_state = self.start_loc
        self.last_action = 'NA'
        self.rwd = 0

        self.done = False
        if self.maze_type == 'triple_reward':
            self.start_loc = self.starter
        self.reward_tally = {}
        for i in self.orig_rwd_loc:
            self.reward_tally[i] = 0

    def move(self, action):
        if action == 'N':
            next_state = (self.cur_state[0], self.cur_state[1]-1)
            if next_state in self.useable:
                self.cur_state = next_state
        elif action == 'E':
            next_state = (self.cur_state[0]+1, self.cur_state[1])
            if next_state in self.useable:
                self.cur_state = next_state
        elif action == 'W':
            next_state = (self.cur_state[0]-1, self.cur_state[1])
            if next_state in self.useable:
                self.cur_state = next_state
        elif action == 'S':
            next_state = (self.cur_state[0], self.cur_state[1]+1)
            if next_state in self.useable:
                self.cur_state = next_state

        self.get_reward(action)
        self.last_action = action
        return self.cur_state

    def get_reward(self, action):
        if (action == 'poke') & (self.cur_state in self.rwd_loc):
            self.rwd = 1
            self.done = True
            if self.maze_type == 'tmaze':
                if self.port_shift in ['equal', 'left', 'right']:
                    self.shift_rwd(self.port_shift)
            self.reward_tally[self.cur_state] += 1

        else:
            self.rwd = self.step_penalization
            self.done = False

    def set_rwd(self, rwd_loc):
        if not isinstance(rwd_loc, list):
            print("must be list of tuples")
        self.rwd_loc = rwd_loc
        if self.maze_type is not 'triple_reward':
            for i in self.rwd_loc:
                self.orig_rwd_loc.append(i)

    def shift_rwd(self,shift):
        port_rwd_probabilities = [0.333, 0.333, 0.333]
        current_rewarded_port = self.rwd_loc[0]

        if shift == 'equal':
            dir_prob = [0.5, 0.5]

        elif shift == 'left':
            dir_prob = [0.95, 0.05]

        elif shift == 'right':
            dir_prob = [0.05, 0.95]

        if self.rwd_loc[0] in self.possible_ports:
            poked_port = self.possible_ports.index(self.rwd_loc[0])
            right_port = (self.possible_ports.index(self.rwd_loc[0])+1)%3
            left_port = (self.possible_ports.index(self.rwd_loc[0])+2)%3

            port_rwd_probabilities[poked_port] = 0
            port_rwd_probabilities[left_port] = dir_prob[0]*port_rwd_probabilities[left_port]
            port_rwd_probabilities[right_port] = dir_prob[1]*port_rwd_probabilities[right_port]

            port_rwd_probabilities = [rp/sum(port_rwd_probabilities) for rp in port_rwd_probabilities]
            new_rwd_loc = np.random.choice(3,1,p=port_rwd_probabilities)
            self.rwd_loc = [self.possible_ports[new_rwd_loc[0]]]
        else:
            print('is not works good')

    def step(self,action):
        action_string = self.actionlist[action]
        observation = self.move(action_string)
        self.cur_state = observation
        done = False
        info = None
        return observation, self.rwd, done, info

class action_wrapper(object):
    def __init__(self, actionlist):
        self.n = len(actionlist)
        self.actionlist = actionlist

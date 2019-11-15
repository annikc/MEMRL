'''
Set up object classes for gridworld environment
gridworld class defines the environment and available actions, reward function, etc.

gymworld makes step function more consistent with the OpenAI gymworld

Author: Annik Carson
-- November 2019
'''

# =====================================
#           IMPORT MODULES            #
# =====================================
from __future__ import division, print_function
import numpy as np
from gym import spaces

np.random.seed(12345)
class gridworld(object):
    '''
    State representation defined in     __init__(); loc_picker()
    Action representation defined in    move()
    Reward function defined in          get_reward(); set_rwd(); shift_rwd()
    Step function defined in            step()

    '''
    def __init__(self, grid_params, **kwargs):
        '''
        :param grid_params: dictionary of inputs for specifying environment
        required arguments
            y_height, x_width (int, int):   size of gridworld environment
            maze_type (str):  specify type of gridworld task
                                possible env_types = ['none', 'bar','room','tmaze', 'triple_reward']
                                    none: open field task. can also include rho option in grid_params to specify density
                                                of randomly distributed obstacles in the environment through which the agent
                                                cannot pass
                                    bar: a solid line of obstacles running horizontally through the environment with a
                                                single space to pass through on either side
                                    room: space is divided into quadrants which can be accessed only through a single
                                               space from an adjacent quadrant
                                    tmaze: standard tmaze task
                                    triple_reward: gridworld task with presence of multiple rewards
                                                        *** need to check this works with all environments // should it be removed?
        optional arguments
            rho (float, [0,1) ): density of obstacles randomly distributed in environment.
            walls (bool):        whether gridworld has external perimeter
            port_shift (str):    possible port shift types = ['none', 'equal', 'left', 'right']
                                    none: reward location does not change
                                    equal: equal probability of reward moving to other possible locations
                                    left: proportionally greater probability of reward moving leftward upon change
                                    right: proportionally greater probability of reward moving rightward upon change

        other kwargs not specified in grid_params dictionary
            barheight (int):     row of grid to be obstructed
                                    only used if maze_type == 'bar'

            actionlist (list): what actions are available for the agent to take
                                by default these are designed after the tripoke task in which 'stay' (no change of state)
                                and 'poke' (necessary action to obtain reward) are included. can be modified to only have
                                4 actions: up/down/right/left as in simpler tasks
            rwd_action (str, element of actionlist): which action when taken in the reward state will yield a reward
            pen (float): penalization for steps taken without rewarded action
        '''

        # State Representation
        self.x 				= grid_params['x_width']
        self.y 				= grid_params['y_height']
        self.maze_type 		= grid_params['maze_type']
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.y, self.x, 3))
        # optional additional args for state representation
        ### ************************* is there a more pythonic way to do this??? ******************************
        if 'rho' in grid_params.keys():
            self.rho 		= grid_params['rho']
        else:
            self.rho        = 0
        if 'walls' in grid_params.keys():
            self.bound 		= grid_params['walls']
        else:
            self.bound      = False
        if 'port_shift' in grid_params.keys():
            self.port_shift	= grid_params['port_shift']
        else:
            self.port_shift = 'none'

        if self.maze_type == 'bar':
            self.barheight 	= kwargs.get('barheight',11)

        # Generate the map of permissible occupancy locations
        self.grid, self.useable, self.obstacles = self.grid_maker()

        # Actions Representation
        self.actionlist 	= kwargs.get('actionlist', ['N', 'E', 'W', 'S', 'stay', 'poke'])
        self.action_space = spaces.Discrete(len(self.actionlist))
        # Reward Representation
        self.rwd_action 	= kwargs.get('rewarded_action', 'poke')
        self.rwd_mag        = kwargs.get('reward_size', 1)
        self.step_penalization = kwargs.get('pen', 0)


        ## ************* is there a better way to do this? ******************************
        if self.maze_type == 'tmaze':
            self.rwd_loc 		= [self.useable[0]]

        if self.maze_type == 'triple_reward':
            self.rwd_loc 		= [(self.x-1, 0), (self.x-1, self.y-1), (0, self.y-1)]
            self.orig_rwd_loc 	= [(self.x-1, 0), (self.x-1, self.y-1), (0, self.y-1)]
            self.starter 		= kwargs.get('t_r_start', (0,0))
            self.start_loc      = self.starter

        else:
            rwd_choice 			= np.random.choice(len(self.useable))
            self.rwd_loc 		= [self.useable[rwd_choice]]
            self.orig_rwd_loc 	= []

        # agent is initialized in a random location from the available states
        around_reward = kwargs.get('around_reward', False)
        self.reset(around_reward=around_reward) # <-- agent's starting location function is called within reset()

        ## OpenAI gym bits
        self.action_space = action_wrapper(self.actionlist)

    def grid_maker(self):
        '''
        Default grid is empty -- all squares == 0
        If env_type given, set some squares to == 1 (agent cannot occupy these squares)
        In the case of the T maze, set all squares =1 and just rewrite which squares the agent may occupy
        (i.e. grid = 0 at these points)
        '''

        env_types = ['none', 'bar','room','tmaze', 'triple_reward']
        # Check that maze type is valid
        if self.maze_type not in env_types:
            print(f"Environment Type '{self.maze_type}' Not Recognized. \nOptions are: {env_types} \nDefault is Open Field (maze_type = 'none')")

        # initialize empty grid
        grid = np.zeros((self.y,self.x), dtype=int)

        # set up obstables for different grid types
        if self.maze_type == 'bar':
            self.rho = 0
            for i in range(self.x-4):
                grid[self.barheight][i+2] = 1

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

    def loc_picker(self, around_reward = False):
        if around_reward:
            # pick starting location for agent in radius around reward location
            start_buffer = 5  # radius around reward
            get_start_loc = True
            while get_start_loc:
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

                if (start_x, start_y) in self.useable:
                    get_start_loc = False
        else: # pick a random starting location for agent within the useable spaces
            get_start = np.random.choice(len(self.useable))
            start_x = self.useable[get_start][0]
            start_y = self.useable[get_start][1]

        return (start_x, start_y)

    def set_rwd(self, rwd_loc):
        if not isinstance(rwd_loc, list):
            print("must be list of tuples")
        self.rwd_loc = rwd_loc
        if self.maze_type is not 'triple_reward':
            for i in self.rwd_loc:
                self.orig_rwd_loc.append(i)

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

    def reset(self, **kwargs):
        around_reward = kwargs.get('around_reward', False)
        self.start_loc = self.loc_picker(around_reward=around_reward)
        self.cur_state = self.start_loc
        self.observation = self.get_frame()

        self.last_action = 'NA'
        self.rwd = 0

        self.done = False
        if self.maze_type == 'triple_reward':
            self.start_loc = self.starter
        self.reward_tally = {}
        for i in self.orig_rwd_loc:
            self.reward_tally[i] = 0

    def get_reward(self, action):
        if (action == 'poke') & (self.cur_state in self.rwd_loc):
            self.rwd = self.rwd_mag
            self.done = True
            if self.maze_type == 'tmaze':
                if self.port_shift in ['equal', 'left', 'right']:
                    self.shift_rwd(self.port_shift)
            self.reward_tally[self.cur_state] += 1

        else:
            self.rwd = self.step_penalization
            self.done = False

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
        self.cur_state = self.move(action_string)
        self.observation = self.get_frame()
        done = False
        info = None
        return self.observation, self.rwd, done, info

    def get_frame(self, **kwargs):
        agent_location  = kwargs.get('agtlocation', self.cur_state)
        reward_location = kwargs.get('rwdlocation', self.rwd_loc[0])
        state_type      = kwargs.get('state_type', 'conv')
        #grid
        grid = self.grid
        #location of reward
        rwd_position = np.zeros_like(grid)
        rwd_position[reward_location[1], reward_location[0]] = 1
        #location of agent
        agt_position = np.zeros_like(grid)
        agt_position[agent_location[1], agent_location[0]] = 1

        if state_type == 'pcs':
            return np.array((grid, rwd_position, agt_position)) #np.transpose(np.array((grid, rwd_position, agt_position)),axes=[1,2,0])
        else:
            return np.array([(grid, rwd_position, agt_position)])

class action_wrapper(object):
    def __init__(self, actionlist):
        self.n = len(actionlist)
        self.actionlist = actionlist

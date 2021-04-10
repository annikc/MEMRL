# =====================================
#           IMPORT MODULES            #
# =====================================
import gym
from gym import spaces, error, utils
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .gw_master import GridWorld, plot_world

GridWorld6 = GridWorld

class GridWorld4(GridWorld):
    def __init__(self):
        self.action_list = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None
        super().__init__(actionlist=self.action_list, rewarded_action=self.rewarded_action)
class GridWorld4_movedR(GridWorld):
    def __init__(self):
        self.action_list = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None
        self.rewards = {(14,14):10}
        super().__init__(actionlist=self.action_list, rewarded_action=self.rewarded_action, rewards=self.rewards)

class GridWorld4_random_obstacle(GridWorld):
    def __init__(self):
        self.action_list = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None
        # obstacles list corresponds to one instance of rho = 0.1
        # using list of obstacles instead so that they are the same each instantiation
        # using rho generates new obstacles each time
        self.obstacles_list = [(0, 11), (0, 14), (3, 1), (3, 19), (4, 4), (4, 11), (4, 15), (4, 17), (6, 4), (6, 18), (7, 6), (8, 1), (8, 11), (9, 0), (9, 8), (9, 14), (10, 13), (11, 4), (11, 16), (12, 5), (12, 18), (12, 19), (13, 2), (13, 5), (13, 15), (14, 6), (14, 9), (14, 19), (15, 4), (15, 7), (15, 19), (16, 7), (17, 0), (17, 2), (17, 11), (18, 1), (19, 5), (19, 7), (19, 11)]
        super().__init__(actionlist=self.action_list, rewarded_action=self.rewarded_action,obstacles=self.obstacles_list)
class GridWorld4_random_obstacle_movedR(GridWorld):
    def __init__(self):
        self.action_list = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None
        self.rewards = {(14,14):10}
        # obstacles list corresponds to one instance of rho = 0.1
        # using list of obstacles instead so that they are the same each instantiation
        # using rho generates new obstacles each time
        self.obstacles_list = [(0, 11), (0, 14), (3, 1), (3, 19), (4, 4), (4, 11), (4, 15), (4, 17), (6, 4), (6, 18), (7, 6), (8, 1), (8, 11), (9, 0), (9, 8), (9, 14), (10, 13), (11, 4), (11, 16), (12, 5), (12, 18), (12, 19), (13, 2), (13, 5), (13, 15), (14, 6), (14, 9), (14, 19), (15, 4), (15, 7), (15, 19), (16, 7), (17, 0), (17, 2), (17, 11), (18, 1), (19, 5), (19, 7), (19, 11)]
        super().__init__(actionlist=self.action_list, rewarded_action=self.rewarded_action,obstacles=self.obstacles_list, rewards=self.rewards)

class GridWorld4_rooms(GridWorld):
    def __init__(self):
        self.action_list = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None
        self.obstacles_list = [(0, 10), (1, 10), (2, 10), (3, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 0), (10, 1), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14), (10, 16), (10, 17), (10, 18), (10, 19), (11, 10), (12, 10), (14, 10), (15, 10), (16, 10), (17, 10), (18, 10), (19, 10)]
        super().__init__(actionlist=self.action_list, rewarded_action=self.rewarded_action,obstacles=self.obstacles_list)
class GridWorld4_rooms_movedR(GridWorld):
    def __init__(self):
        self.action_list = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None
        self.rewards = {(14,14):10}
        self.obstacles_list = [(0, 10), (1, 10), (2, 10), (3, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 0), (10, 1), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14), (10, 16), (10, 17), (10, 18), (10, 19), (11, 10), (12, 10), (14, 10), (15, 10), (16, 10), (17, 10), (18, 10), (19, 10)]
        super().__init__(actionlist=self.action_list, rewarded_action=self.rewarded_action,obstacles=self.obstacles_list,rewards=self.rewards)

class GridWorld4_bar(GridWorld):
    def __init__(self):
        self.action_list = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None
        self.maze_type = 'bar'
        self.barheight = 9
        super().__init__(actionlist=self.action_list, rewarded_action=self.rewarded_action,env_type=self.maze_type, barheight=self.barheight)
class GridWorld4_bar_movedR(GridWorld):
    def __init__(self):
        self.action_list = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None
        self.rewards = {(14,14):10}
        self.maze_type = 'bar'
        self.barheight = 9
        super().__init__(actionlist=self.action_list, rewarded_action=self.rewarded_action,env_type=self.maze_type, barheight=self.barheight, rewards=self.rewards)


class GridWorld4_tunnel(GridWorld):
    def __init__(self):
        self.action_list = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None
        # obstacles list corresponds to one instance of rho = 0.1
        # using list of obstacles instead so that they are the same each instantiation
        # using rho generates new obstacles each time
        tunnel_blocks = []
        for i in range(20):
            for j in range(7,13):
                if i == 9:
                    pass
                else:
                    tunnel_blocks.append((i,j))

        self.obstacles_list = tunnel_blocks
        self.rewards = {(9,3):10}
        super().__init__(cols=20,rows=20,actionlist=self.action_list, rewards=self.rewards,rewarded_action=self.rewarded_action,obstacles=self.obstacles_list)
class GridWorld4_tunnel_movedR(GridWorld):
    def __init__(self):
        self.action_list = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None
        # obstacles list corresponds to one instance of rho = 0.1
        # using list of obstacles instead so that they are the same each instantiation
        # using rho generates new obstacles each time
        tunnel_blocks = []
        for i in range(20):
            for j in range(7,13):
                if i == 9:
                    pass
                else:
                    tunnel_blocks.append((i,j))

        self.obstacles_list = tunnel_blocks
        self.rewards = {(9,16):10}
        super().__init__(cols=20,rows=20,actionlist=self.action_list, rewards=self.rewards,rewarded_action=self.rewarded_action,obstacles=self.obstacles_list)


class GridWorld4_hairpin(GridWorld):
    def __init__(self):
        self.action_list = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None
        # obstacles list corresponds to one instance of rho = 0.1
        # using list of obstacles instead so that they are the same each instantiation
        # using rho generates new obstacles each time
        hairpin_blocks = []
        ncols = 9
        nrows = 10
        for i in range(ncols):
            for j in range(nrows):
                if i%2==0:
                    pass
                else:
                    hairpin_blocks.append((j,i))
        for i in range(ncols):
            if i%2==0:
                pass
            elif i%4==1:
                hairpin_blocks.remove((nrows-1,i))
            elif i%4==3:
                hairpin_blocks.remove((0, i))

        self.obstacles_list = hairpin_blocks
        self.rewards = {(0,0):10}
        super().__init__(cols=ncols,rows=nrows, actionlist=self.action_list, rewards=self.rewards,
                         rewarded_action=self.rewarded_action, obstacles=self.obstacles_list, random_start=True)
class GridWorld4_hairpin_movedR(GridWorld):
    def __init__(self):
        self.action_list = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None
        # obstacles list corresponds to one instance of rho = 0.1
        # using list of obstacles instead so that they are the same each instantiation
        # using rho generates new obstacles each time
        hairpin_blocks = []
        ncols = 9
        nrows = 10
        for i in range(ncols):
            for j in range(nrows):
                if i%2==0:
                    pass
                else:
                    hairpin_blocks.append((j,i))
        for i in range(ncols):
            if i%2==0:
                pass
            elif i%4==1:
                hairpin_blocks.remove((nrows-1,i))
            elif i%4==3:
                hairpin_blocks.remove((0, i))

        self.obstacles_list = hairpin_blocks
        self.rewards = {(nrows-1,ncols-1):10}
        super().__init__(cols=ncols,rows=nrows, actionlist=self.action_list, rewards=self.rewards,
                         rewarded_action=self.rewarded_action, obstacles=self.obstacles_list, random_start=True)

class LinearTrack(GridWorld):
    def __init__(self):
        self.action_list = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None
        track_length = 20
        self.rewards = {(0,track_length-1):10}
        super().__init__(rows=1, cols=track_length, actionlist=self.action_list, rewarded_action=self.rewarded_action, rewards=self.rewards)

class LinearTrack_1(GridWorld):
    # difference from Linear track = only has left-right transitions
    def __init__(self):
        self.action_list = ['Right', 'Left']
        self.rewarded_action = None
        self.rewards = {(0,19):5}
        super().__init__(rows=1, cols=20, actionlist=self.action_list, rewarded_action=self.rewarded_action, rewards=self.rewards)

    def buildTransitionMatrix(self):
        # initialize
        self.P = np.zeros((len(self.action_list), self.nstates, self.nstates))  # down, up, right, left, jump, poke

        self.P[0, list(range(0, self.nstates-1)), list(range(1, self.nstates))] = 1  							# right
        self.P[1, list(range(1, self.nstates)), list(range(0, self.nstates-1))] = 1  							# left

class MiniGrid(GridWorld):
    def __init__(self):
        self.action_list = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None
        super().__init__(rows=7, cols=7, actionlist=self.action_list, rewarded_action=self.rewarded_action)

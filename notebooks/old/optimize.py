from __future__ import division
from modules import *
from run_trials import run_trials
import uuid
import pymysql
from influxdb import InfluxDBClient

fig_savedir = '../data/figures/'

conn = pymysql.connect(host='jenkins.c2g09w2ghsye.us-east-1.rds.amazonaws.com',
                       user='jeremyforan',
                       passwd='nafaweM3',
                       db='shinjo',
                       port=3306,
                       autocommit=True)
#cur = conn.cursor()

grid_params = {
    'y_height': 20,
    'x_width': 20,
    'walls': False,
    'rho': 0,
    'maze_type': 'none',
    'port_shift': 'none'
}

agent_params = {
    'load_model': False,
    'load_dir': '../data/outputs/gridworld/MF{}{}training.pt'.format(grid_params['x_width'],
                                                                     grid_params['y_height']),
    'action_dims': 6,  # =len(maze.actionlist)
    'batch_size': 1,
    'gamma': 0.98,  # discount factor
    'eta': 5e-4,
    'temperature': 1,
    'use_EC': False,
    'cachelim': 300,  # int(0.75*np.prod(maze.grid.shape)) # memory limit should be ~75% of #actions x #states
    'state_type': 'conv'
}

run_dict = {
    'NUM_EVENTS': 100,
    'NUM_TRIALS': 3000,
    'print_freq': 1 / 10,
    'track_events': False,
    'total_loss': [[], []],
    'total_reward': [],
    'conn' : conn
    #'val_maps': [],
    #'policies': [{}, {}],
    #'deltas': [],
    #'spots': [],
    #'vls': []
}



def alpha_variation():
    for alpha in range(0,1):
        for i in range(0,10):
            astar = 0.3#np.round(0.1*(alpha + 1),2)
            print(f"Running w alpha = {astar}")
            a1 = run_trials(run_dict, agent_params, grid_params,save=False, alpha= astar)
            file = open('record_uuid.txt', 'a+')
            file.write(f"{astar},{a1} \n")
            file.close()

def multilayer():
    a1 = run_trials(run_dict, agent_params, grid_params, use_EC=False, save=False)
    file = open('multilayer_uuid.txt', 'a+')
    file.write(f"{a1} \n")
    file.close()


# make environment
maze = eu.gridworld(grid_params)
maze.set_rwd([(int(grid_params['y_height'] / 2), int(grid_params['x_width'] / 2))])

# update agent params dictionary with layer sizes appropriate for environment
agent_params = sg.gen_input(maze,agent_params, hid_types=['conv', 'pool', 'linear', 'linear'])

run_trials(run_dict, agent_params, grid_params)




#ac.torch.save(MF,agent_params['load_dir'])



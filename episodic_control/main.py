import sys
import actorcritic as ac
import environment as gw
import memory as ec

sys.modules['gw'] = gw
sys.modules['ac'] = ac
# TODO -- need to do this for all imported modules????

import run_experiment as expt

import pickle
import argparse
import pandas as pd
import uuid
from analysis import DataFilter
## remove after july 24
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def main(experiment_type, env_type, rho, arch, use_pvals, mem_temp, mem_envelope, alpha, beta, RECORD=True):
    parent_dir = './data/'
    log_file = 'cottage_scinet_logs.csv'
    # set environment parameters
    rows, columns = 20, 20
    penalty = -0.01
    if experiment_type == 0:
        reward_location = (5, 5)
    else:
        reward_location = (15, 15)
    print(experiment_type, reward_location, "arg")

    # generate environment object
    env = gw.GridWorld(rows=rows, cols=columns, env_type=env_type,
                       rewards={reward_location: 1},
                       step_penalization=penalty,
                       rho=rho,
                       actionlist=['Down', 'Up', 'Right', 'Left'],
                       rewarded_action=None)

    # agent parameters
    training = {
        'load_model': False,
        'load_dir': '',

        'architecture': arch,
        'input_dims': env.observation.shape,
        'action_dims': len(env.action_list),
        'hidden_types': ['conv', 'pool', 'conv', 'pool', 'linear', 'linear'],
        'hidden_dims': [None, None, None, None, 100, 200],

        'freeze_w': False,

        'rfsize': 5,

        'gamma': 0.98,
        'eta': 5e-4,

        'use_EC': False
    }

    testing_1 = training.copy()
    testing_1.update({'load_model': True, 'freeze_w': True})

    testing_2 = testing_1.copy()
    testing_2.update({'use_EC': True})

    testing_4 = testing_1.copy()
    testing_4.update({'freeze_w': False})

    testing_5 = testing_4.copy()
    testing_5.update({'use_EC': True})

    params = [training, testing_1, testing_2, testing_2, testing_4, testing_5, testing_5]

    NUM_TRIALS = 100
    NUM_EVENTS = 50

    # mixing parameters for MF-EC control
    # alpha = 0.01 # MF confidence boost for reward
    # beta = 100 # MF confidence decay - number of steps to decay to 1%

    # create an agent with parameters for a given experiment type
    agent_params = params[experiment_type]
    load_id = ' '
    if experiment_type != 0:
        # read csv file - get id tag for matching conditions
        df = pd.read_csv(parent_dir + log_file)
        filter = DataFilter(df,
                            expt_type=[0],
                            env_type=[str(env_type)],
                            dims=[str(env.shape)],
                            rho=[float(env.rho)],
                            arch=[arch])

        load_id = list(filter.ids)[0]
        agent_params['load_dir'] = parent_dir + f'agent_weights/{load_id}.pt'
        print(load_id)
        env = pickle.load(open(parent_dir + f'environments/{load_id}_env.p', 'rb'))
        
        # set new reward location and update env.R accordingly 
        env.rewards = {reward_location: 1}
        env.buildRewardFunction()
        env.finish_after_first_reward = False

    # create agent from parameters for given experiment type
    agent = ac.make_agent(agent_params)

    if experiment_type in [2, 3, 5, 6]:
        # create memory module
        mem = ec.EpisodicMemory(cache_limit=0.75 * env.nstates,
                                entry_size=agent.action_dims,
                                mem_temp=mem_temp,
                                mem_envelope=mem_envelope,
                                pvals=use_pvals)
    else:
        mem = None
    # create run_id
    run_id = uuid.uuid4()
    # create experiment object
    ex = expt.Experiment(agent, env, use_mem=agent_params['use_EC'], mem=mem)

    # run experiment
    ex.run(NUM_TRIALS, NUM_EVENTS, alpha=alpha, beta=beta)

    # log data
    if RECORD:
        expt.data_log(run_id, experiment_type, ex, load_from=load_id)

    #### trying some craziness -- delete later (Jul 24)
    plt.figure(0)
    x = ex.data['confidence_selection'][0] # MFCS
    y = ex.data['confidence_selection'][1] # policy_choice
    viridis = cm.get_cmap('inferno', len(x)).colors
    plt.scatter(x, y, c = viridis, alpha=0.3)
    plt.yticks([0,1], ['MF', 'EC'])
    plt.ylim([-0.5, 1.5])
    plt.show()
    plt.close()


    plt.figure(1)
    y = ex.data['confidence_selection'][0] # MFCS
    x = np.arange(len(x))
    cols = ['b', 'g'] #mf , ec
    z = ex.data['confidence_selection'][1] # policy_choice
    col = [cols[i] for i in z]
    plt.scatter(x, y, c = col, alpha=0.3)
    plt.show()
    plt.close()



def process_command_line_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--expt',
                        choices=[0, 1, 2, 3, 4, 5, 6],
                        default=0,
                        type=int,
                        help="Experiment type")  # TODO describe expt types
    parser.add_argument('--env',
                        choices=['None', 'room'],
                        nargs='?',
                        default='None',
                        help='Environment type: None, room')
    parser.add_argument('--rho',
                        type=float,
                        nargs='?',
                        default=0.0,
                        help='Obstacle density value ')

    parser.add_argument('--arch',
                        type=str,
                        choices=['A', 'B'],
                        nargs='?',
                        default='B',
                        help='Agent architecture to use: A = no successor representation, B = with successor representation')

    parser.add_argument('--pvals',
                        type=bool,
                        nargs='?',
                        default=False,
                        help='Whether to decay memory relative to encoding time')

    parser.add_argument('--memtemp',
                        type=float,
                        nargs='?',
                        default=0.1,
                        help='Temperature value for EC softmax function')

    parser.add_argument('--memenv',
                        type=int,
                        nargs='?',
                        default=50,
                        help='Memory decay parameter (speed of forgetting)'
                        )

    parser.add_argument('--alpha',
                        type=float,
                        nargs='?',
                        default=1,
                        help='MF-EC control mixing bump')

    parser.add_argument('--beta',
                        nargs='?',
                        default=10,
                        type=int,
                        help='MF-EC control mixing decay')



    args = parser.parse_args()

    return args


args = process_command_line_arguments()

if args.env == 'None':
    env_type = None
else:
    env_type = args.env

if __name__ == '__main__':
    main(args.expt, env_type, args.rho, args.arch, args.pvals, args.memtemp, args.memenv, args.alpha, args.beta, RECORD=False)

import sys
import argparse
import pandas as pd
import analysisE2 as e2

sys.path.insert(0,'../rl_network/'); import ac
sys.path.insert(0,'../memory/'); import episodic as ec
sys.path.insert(0,'../environments/'); import gw

import experiment as expt
import torch
import uuid
import pickle
import csv

def data_log(run_id, experiment_type, experiment, **kwargs):
    load_from = kwargs.get('load_from',' ')
    write = kwargs.get('write', True)
    expt_log = [
    'save_id',          #uuid
    'experiment_type',  #int
    'load_from',        #str
    'num_trials',       #int
    'num_events',       #int
    'ENVIRONMENT',      #str
    'shape',            #tuple
    'rho',              #float
    'rewards',          #dict
    'action_list',      #list
    'rwd_action',       #str
    'step_penalization',#float
    'useable',          #list
    'obstacle2D',       #list
    'terminal2D',       #list
    'jump',             #list or NoneType
    'random_start',     #bool
    'AGENT',            #arch
    'use_SR',           #bool
    'freeze_weights',   #bool
    'layers',           #list
    'hidden_types',     #list
    'gamma',            #float
    'eta',              #float
    'optimizer',        #torch optim. class
    'MEMORY', #         # string*
    'cache_limit',      #int
    'use_pvals',        #bool
    'memory_envelope',  #int
    'mem_temp',         #float
    'alpha',            #float   # memory mixing parameters
    'beta'              #int
    ]

    log_jam = [
        run_id,
        experiment_type,
        load_from,
        experiment.num_trials,
        experiment.num_events,

        str(experiment.env.maze_type), # 'ENVIRONMENT'
        experiment.env.shape,
        float(experiment.env.rho),
        experiment.env.rewards,
        experiment.env.action_list,
        str(experiment.env.rwd_action),
        experiment.env.step_penalization,
        experiment.env.useable,
        experiment.env.obstacle2D,
        experiment.env.terminal2D,
        experiment.env.jump,
        experiment.env.random_start,

        experiment.agent_architecture, #'AGENT'
        experiment.agent.use_SR,
        experiment.agent.optimizer.param_groups[0]['lr'] == 0.0, # evaluates true if frozen weights
        experiment.agent.layers,
        experiment.agent.hidden_types,
        experiment.agent.gamma,
        experiment.agent.eta,
        experiment.agent.optimizer,

        str(experiment.episodic)
    ]

    if experiment.episodic != None:
        epi_log = [
            experiment.episodic.cache_limit,
            experiment.episodic.use_pvals,
            experiment.episodic.mem_temp,
            experiment.alpha,
            experiment.beta
        ]
        log_jam += epi_log
    if write:
        parent_folder = '../data/outputs/gridworld/E2/'
        # write to logger
        with open(parent_folder+'/experiments_log.csv', 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log_jam)

        # save environment
        if experiment_type == 0:
            pickle.dump(experiment.env, open(f'{parent_folder}environments/{run_id}_env.p', 'wb'))

        # save agent
        torch.save(experiment.agent, f'{parent_folder}agent_weights/{run_id}.pt')
        # save data
        pickle.dump(experiment.data, open(f'{parent_folder}results/{run_id}_data.p', 'wb'))
        # save memory
        if experiment.episodic != None:
            pickle.dump(experiment.episodic, open(f'{parent_folder}episodic_memory/{run_id}_EC.p', 'wb'))

def main(experiment_type, env_type, rho, arch, use_pvals, mem_temp, mem_envelope, alpha, beta):
    # set environment parameters
    rows, columns = 20,20
    penalty = -0.01
    if experiment_type == 0:
        reward_location = (5,5)
    else:
        reward_location = (15,15)

    # generate environment object
    env = gw.GridWorld(rows=rows,cols=columns,env_type=env_type,
                       rewards = {reward_location:1},
                       step_penalization=penalty,
                       rho=rho,
                       actionlist = ['Down', 'Up', 'Right', 'Left'],
                       rewarded_action=None)

    # agent parameters
    training = {
        'load_model':  False,
        'load_dir':    '',

        'architecture': arch,
        'input_dims':  env.observation.shape,
        'action_dims': len(env.action_list),
        'hidden_types':['conv','pool','conv', 'pool', 'linear','linear'],
        'hidden_dims': [None, None, None, None, 100, 200],

        'freeze_w':    False,

        'rfsize':      5,

        'gamma':       0.98,
        'eta':         5e-4,

        'use_EC':      False
    }

    testing_1 = training.copy()
    testing_1.update({'load_model':True, 'freeze_w':True})

    testing_2 = testing_1.copy()
    testing_2.update({'use_EC':True})

    testing_4 = testing_1.copy()
    testing_4.update({'freeze_w':False})

    testing_5 = testing_4.copy()
    testing_5.update({'use_EC':True})

    params = [training, testing_1, testing_2, testing_2, testing_4, testing_5, testing_5]

    NUM_TRIALS = 5000
    NUM_EVENTS = 250

    # mixing parameters for MF-EC control
    #alpha = 0.01 # MF confidence boost for reward
    #beta = 100 # MF confidence decay - number of steps to decay to 1%

    # create an agent with parameters for a given experiment type
    agent_params = params[experiment_type]
    load_id = ' '
    if experiment_type != 0:
        # read csv file - get id tag for matching conditions
        df = pd.read_csv('../data/outputs/gridworld/E2/experiments_log.csv')
        filter = e2.data_filter(df,
                                expt_type = [0],
                                env_type=[str(env_type)],
                                dims=[str(env.shape)],
                                rho =[float(env.rho)],
                                arch=[arch])

        load_id = list(filter.ids)[0]
        agent_params['load_dir'] = f'../data/outputs/gridworld/E2/agent_weights/{load_id}.pt'

        env.finish_after_first_reward = False
    # create agent from parameters for given experiment type
    agent = ac.make_agent(agent_params)

    if experiment_type in [2,3,5,6]:
        # create memory module
        mem = ec.ep_mem(cache_limit = 0.75 * env.nstates,
                        entry_size=agent.action_dims,
                        mem_temp=mem_temp,
                        mem_envelope=mem_envelope,
                        pvals=use_pvals)
    else:
        mem = None
    # create run_id
    run_id = uuid.uuid4()
    # create experiment object
    ex = expt.test_expt(agent, env, use_mem=agent_params['use_EC'], mem=mem)

    # run experiment
    ex.run(NUM_TRIALS, NUM_EVENTS, alpha=alpha,beta=beta)

    # log data
    print('run data log')
    data_log(run_id, experiment_type, ex, load_from=load_id)
    print(f'run {run_id} logged')
def process_command_line_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--expt',
                        choices=[0,1,2,3,4,5,6],
                        default=0,
                        type=int,
                        help="Experiment type") # TODO describe expt types
    parser.add_argument('--env',
                        choices=[None, 'room'],
                        nargs='?',
                        default=None,
                        help='Environment type: None, room')
    parser.add_argument('--rho',
                        type=float,
                        nargs='?',
                        default=0.0,
                        help='Obstacle density value ')

    parser.add_argument('--arch',
                        type=str,
                        choices = ['A', 'B'],
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
                        default=0.05,
                        help='Temperature value for EC softmax function')

    parser.add_argument('--memenv',
                        type=int,
                        nargs='?',
                        default=50,
                        help='Memory decay parameter (speed of forgetting)'
                        )

    parser.add_argument('--alpha',
                        nargs = '?',
                        default = 1,
                        type=int,
                        help='MF-EC control mixing bump')

    parser.add_argument('--beta',
                        nargs = '?',
                        default = 10000,
                        type=int,
                        help='MF-EC control mixing decay')
    args = parser.parse_args()

    return args
args = process_command_line_arguments()

main(args.expt, args.env, args.rho, args.arch, args.pvals, args.memtemp, args.memenv, args.alpha, args.beta)

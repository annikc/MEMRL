# write experiment class
# expt class should take agent and environment
# functions for stepping through events/trials, updating,
# collecting data, writing data
# Annik Carson Oct 28, 2020

# Experiment Object Class and Related Functions
# Written and maintained by Annik Carson
# Last updated: July 2020
#
# =====================================
#           IMPORT MODULES            #
# =====================================
import numpy as np
import time
import pickle, csv


class Experiment(object):
    def __init__(self, agent, environment, **kwargs):
        self.env = environment

        self.agent = agent
        # self.rep_learner = rep_learner  #TODO add in later
        self.data = self.reset_data_logs()

    def reset_data_logs(self):
        data_log = {'total_reward': [],
                    'loss': [[], []],
                    'trial_length': []
                    }
        return data_log

    def policy_arbitration(self):
        # TODO
        ## set whether to use agent.MFC or agent.EC
        pass

    def representation_learning(self):
        # TODO
        # to be run before experiment to learn representations of states
        pass

    def run(self, NUM_TRIALS, NUM_EVENTS, **kwargs):
        print_freq = kwargs.get('printfreq', 100)
        self.render = kwargs.get('render', False)

        self.reset_data_logs()
        t = time.time()
        for trial in range(NUM_TRIALS):
            if self.render:
                self.env.figure[0].canvas.set_window_title(f'Trial {trial}')
            self.env.reset()
            self.reward_sum = 0

            for event in range(NUM_EVENTS):
                # get observation from environment
                state = self.env.state  ## for record keeping only
                readable = 0 # self.env.oneD2twoD(self.env.state)  ## for record keeping only
                mem_state = np.zeros(self.env.nstates)
                mem_state[state] = 1

                # get observation from environment
                obs = self.env.get_observation()

                # get action from agent
                action, log_prob, expected_value = self.agent.get_action(np.expand_dims(obs, axis=0), tuple(mem_state))  ## expand dims to make batch size =1
                # take step in environment
                next_state, reward, done, info = self.env.step(action)

                # end of event
                target_value = 0
                self.reward_sum += reward

                self.agent.log_event(episode=trial, event=event,
                                     state=tuple(mem_state), action=action, reward=reward, next_state=next_state,
                                     log_prob=log_prob, expected_value=expected_value, target_value=target_value,
                                     done=done, readable_state=readable)

                if self.render:
                    self.env.render()
                if done:
                    break

            p, v = self.agent.finish_()

            self.data['total_reward'].append(self.reward_sum)
            self.data['loss'][0].append(p)
            self.data['loss'][1].append(v)

            if trial == 0:
                running_rwdavg = self.reward_sum
            else:
                running_rwdavg = ((trial) * running_rwdavg + self.reward_sum) / (trial + 2)

            if trial % print_freq == 0:
                print(f"Episode: {trial}, Score: {self.reward_sum} (Running Avg:{running_rwdavg}) [{time.time()-t}s]")
                t = time.time()


## JUNKYARD == NOV 9, 2020
'''
class training(Experiment):
    def __init__(self, ep=basic_env_params()):
        self.env_params = ep
        # generate environment object
        env = gw.GridWorld(rows=ep.rows, cols=ep.columns, env_type=ep.env_type,
                           rewards={ep.reward_location: ep.reward_mag},
                           step_penalization=ep.penalty,
                           rho=ep.rho,
                           actionlist=ep.actionlist,
                           rewarded_action=ep.rewarded_action)

        # generate agent object
        self.agent_params = basic_agent_params(env)
        agent = ac.make_agent(self.agent_params.__dict__)
        super().__init__(agent, env)
        self.run = super().runMF


class testing(Experiment):
    def __init__(self, expt_type, env_params=basic_env_params()):
        parent_dir = './data/'
        log_file = 'sep2020.csv'

        # load items that match criteria
        df = pd.read_csv(parent_dir + log_file)
        filtered_d = DataFilter(df,
                                expt_type=[0],
                                env_type=[str(env_params.env_type)],
                                dims=[str((env_params.shape))],
                                rho=[env_params.rho],
                                action_list=[str(env_params.actionlist)],
                                rewards=[f'{{(5, 5): {env_params.reward_mag}}}'],
                                rewarded_action=[str(env_params.rewarded_action)],
                                arch=['B'],  ## default
                                )
        load_id = list(filtered_d.ids)[0]
        # load saved environment
        with open(parent_dir + f'environments/{load_id}_env.p', 'rb') as f:
            env = pickle.load(f)

        if expt_type in [1, 2, 3, 4, 5, 6]:
            # rshift
            env_params.reward_location = (15, 15)  # for expt_type <10
            env.rewards = {env_params.reward_location: env_params.reward_mag}
            env.buildRewardFunction()
            env.finish_after_first_reward = True  ### possibly don't need
        if expt_type in [11, 12, 13, 14, 15, 16]:
            # pshift
            env.remapTransitionMatrix()

        agent_params = basic_agent_params(env)
        agent_params.load_model = True
        agent_params.load_dir = parent_dir + f'agent_weights/{load_id}.pt'

        if expt_type in [1, 2]:
            agent_params.freeze_w = True

        agent = ac.make_agent(agent_params.__dict__)

        if expt_type in [2, 3, 5, 6, 12, 15]:
            mem_params = basic_mem_params()
            mem = ec.EpisodicMemory(cache_limit=mem_params.cache_limit,
                                    entry_size=agent.action_dims,
                                    mem_temp=mem_params.mem_temp,
                                    mem_envelope=mem_params.mem_envelope,
                                    pvals=mem_params.use_pvals)
            super().__init__(agent, env, use_mem=True, mem=mem)
            self.run = super().runEC
        else:
            super().__init__(agent, env)
            self.run = super().runMF


def get_snapshot(sample_obs, env, agent):  # TODO: update for general agent
    # get sample observations from all useable spaces in environment
    samples, states = sample_obs

    # forward pass through network
    if agent.use_SR:
        pols, vals, _, __ = agent(torch.Tensor(samples))
    else:
        pols, vals, _ = agent(torch.Tensor(samples))

    # initialize empty data frames
    pol_grid = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])
    val_grid = np.empty(env.shape)
    # populate with data from network
    for s, p, v in zip(states, pols, vals):
        pol_grid[s] = tuple(p.data.numpy())
        val_grid[s] = v.item()

    return pol_grid, val_grid


def data_log(log_file, run_id, experiment_type, experiment, **kwargs):
    parent_folder = kwargs.get('write_to_dir', './data/')
    load_from = kwargs.get('load_from', ' ')
    write = kwargs.get('write', True)
    expt_log = [
        'save_id',  # uuid
        'experiment_type',  # int
        'load_from',  # str
        'num_trials',  # int
        'num_events',  # int
        'ENVIRONMENT',  # str
        'shape',  # tuple
        'rho',  # float
        'rewards',  # dict
        'action_list',  # list
        'rwd_action',  # str
        'step_penalization',  # float
        'useable',  # list
        'obstacle2D',  # list
        'terminal2D',  # list
        'jump',  # list or NoneType
        'random_start',  # bool
        'AGENT',  # arch
        'use_SR',  # bool
        'freeze_weights',  # bool
        'layers',  # list
        'hidden_types',  # list
        'gamma',  # float
        'eta',  # float
        'optimizer',  # torch optim. class
        'MEMORY',  # # string*
        'cache_limit',  # int
        'use_pvals',  # bool
        'memory_envelope',  # int
        'mem_temp',  # float
        'alpha',  # float   # memory mixing parameters
        'beta'  # int
    ]

    log_jam = [
        run_id,
        experiment_type,
        load_from,
        experiment.num_trials,
        experiment.num_events,

        str(experiment.env.maze_type),  # 'ENVIRONMENT'
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

        experiment.agent_architecture,  # 'AGENT'
        experiment.agent.use_SR,
        experiment.agent.optimizer.param_groups[0]['lr'] == 0.0,  # evaluates true if frozen weights
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
            experiment.episodic.memory_envelope,
            experiment.episodic.mem_temp,
            experiment.alpha,
            experiment.beta
        ]
    else:
        epi_log = [
            'None', 'None', 'None', 'None', 'None', 'None']
    log_jam += epi_log
    print('writing to file')
    if write:
        # save environment
        if experiment_type == 0:
            pickle.dump(experiment.env, open(f'{parent_folder}environments/{run_id}_env.p', 'wb'))
        # save agent
        torch.save(experiment.agent, f'{parent_folder}agent_weights/{run_id}.pt')
        # save data
        pickle.dump(experiment.data, open(f'{parent_folder}results/{run_id}_data.p', 'wb'))
        # save memory
        if experiment.episodic is not None:
            pickle.dump(experiment.episodic, open(f'{parent_folder}episodic_memory/{run_id}_EC.p', 'wb'))
        print(f'{run_id} recorded')

        # write to logger
        with open(parent_folder + log_file, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log_jam)

    else:
        print_dict = {}
        for i in range(len(log_jam)):
            print_dict[expt_log[i]] = log_jam[i]
        print(print_dict)
'''
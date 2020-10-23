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
import torch ## is there another way to do it without needing to make a torch tensor?
import environment as gw
import actorcritic as ac
import memory as ec
import pandas as pd
from analysis import DataFilter

class basic_agent_params():
    def __init__(self, env):
        self.load_model = False
        self.load_dir   = ''
        self.architecture = 'B'
        self.input_dims = env.observation.shape
        self.action_dims = len(env.action_list)
        self.hidden_types = ['conv', 'pool', 'conv', 'pool', 'linear', 'linear']
        self.hidden_dims = [None, None, None, None, 1000, 1000]
        self.freeze_w = False
        self.rfsize = 5
        self.gamma = 0.98
        self.eta = 5e-4
        self.use_EC = False

class basic_env_params():
    def __init__(self):
        self.rows, self.columns = 20,20
        self.shape = (self.rows,self.columns)
        self.env_type = None
        self.rho = 0.0
        self.penalty = -0.01
        self.reward_location = (5,5)
        self.reward_mag = 10
        self.actionlist = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None

class basic_mem_params():
    def __init__(self):
        self.cache_limit = 400
        self.mem_temp = 1
        self.mem_envelope = 50
        self.use_pvals = False

class Experiment(object):
    def __init__(self, agent, environment, **kwargs):
        self.env = environment

        self.agent = agent
        if self.agent.use_SR:
            self.agent_architecture = 'B'
        else:
            self.agent_architecture = 'A'

        self.episodic = kwargs.get('mem', None)
        if self.episodic != None:
            self.mem_size = self.episodic.cache_limit

        self.data = self.reset_data_logs()

    def reset_data_logs(self):
        data_log = {'total_reward': [],
                    'loss': [[],[],[]],
                    'trial_length': [],
                    'pol_tracking':[],
                    'val_tracking':[],
                    'ec_tracking': [],
                    }
        return data_log

    def policy_arbitration(self,reward_tminus1):
        self.alpha = 0
        self.beta = 0
        self.decay = 0

        confidence_in_model_free =(self.decay*self.MF_cs) + self.alpha*reward_tminus1

        if confidence_in_model_free > 1.0:
            self.MF_cs = 1.0
        elif confidence_in_model_free < 0.0:
            self.MF_cs = 0.0
        else:
            self.MF_cs = confidence_in_model_free

    def trial_reset(self, trial):
        # reset environment, choose new starting location for agent
        self.env.resetEnvironment()
        # clear hidden layer cache if using lstm or gru cells
        self.agent.reinit_hid()
        self.reward_sum = 0

        if self.episodic != None:
            self.memory_buffer = [[], [], [], [], trial]  # [timestamp, state_t, a_t, readable_state, trial]

    def action_selection(self, policy_, value_, lin_act=None):

        if self.use_memory:
            episodic_memory = self.episodic.recall_mem(key=lin_act, timestep=self.timestamp, env=self.recency_env)
            episodic_pol = torch.from_numpy(episodic_memory)
            choice = self.agent.select_ec_action(policy_, value_, episodic_pol)
            policy_ = episodic_pol

            '''
            # compute MFCS
            self.policy_arbitration(self.last_reward)
            #self.MF_cs = 0.01
            cstar = np.random.choice([0,1], p=[self.MF_cs, 1 - self.MF_cs])
            pol_choice = ['mf', 'ec'][cstar]
            #cstar = 1

            if pol_choice == 'ec':
                self.eccount +=1
                episodic_memory = self.episodic.recall_mem(key=lin_act, timestep=self.timestamp, env=self.recency_env)
                episodic_pol = torch.from_numpy(episodic_memory)
                choice = self.agent.select_ec_action(policy_, value_, episodic_pol)
                policy_ = episodic_pol
            else:
                self.mfcount +=1
                choice = self.agent.select_action(policy_, value_)

            self.data['confidence_selection'][0].append(self.MF_cs)
            self.data['confidence_selection'][1].append(cstar)
            '''
        else:
            choice = self.agent.select_action(policy_, value_)

        return choice, policy_

    def save_to_mem(self, timestamp, lin_act, choice, current_state, trial):
        self.memory_buffer[0].append(timestamp)
        self.memory_buffer[1].append(lin_act)
        self.memory_buffer[2].append(choice)
        self.memory_buffer[3].append(current_state)
        self.memory_buffer[4] = trial


    def runEC(self, NUM_TRIALS, NUM_EVENTS, **kwargs):
        print('running episodic control')

        print_freq         = kwargs.get('printfreq', 100)
        get_samples        = kwargs.get('get_samples', False)
        self.reset_data_logs()

        sample_observations = self.env.get_sample_obs()

        self.recency_env = self.episodic.calc_envelope(halfmax=20)
        self.timestamp = 0
        t = time.time()
        for trial in range(NUM_TRIALS):
            self.trial_reset(trial)

            for event in range(NUM_EVENTS):
                # observation from environment
                observation = torch.Tensor(np.expand_dims(self.env.get_observation(), axis=0))

                # observation to agent
                if self.agent.use_SR:
                    policy_, value_, phi_, psi_ = self.agent(observation)
                    self.agent.saved_phi.append(phi_)
                    self.agent.saved_psi.append(psi_)
                else:
                    policy_, value_, psi_ = self.agent(observation)

                # linear activity
                lin_act = tuple(np.round(phi_.data[0].numpy(), 4)) ##### CHANGED TO PHI
                choice, poli = self.action_selection(policy_, value_,lin_act)

                # select action from policy
                action = self.env.action_list[choice][0]

                self.save_to_mem(self.timestamp, lin_act, choice, self.env.oneD2twoD(self.env.state), trial)

                # take a step in the environment
                s_1d, reward, isdone = self.env.move(action)

                self.agent.saved_rewards.append(reward)
                self.reward_sum += reward
                self.last_reward  = reward ### new with policy arbitration
                self.timestamp += 1

                if isdone:
                    if self.agent.use_SR:
                        self.agent.saved_phi.append(phi_)
                        self.agent.saved_psi.append(psi_)
                    encountered_reward = True
                    break

            if self.agent.use_SR:
                p_loss, v_loss, psi_loss = self.agent.finish_trial_EC(cache=self.episodic, buffer=self.memory_buffer)
            else:
                p_loss, v_loss  = self.agent.finish_trial_EC(cache=self.episodic, buffer=self.memory_buffer)

            self.data['trial_length'].append(event+1)
            self.data['total_reward'].append(self.reward_sum)
            self.data['loss'][0].append(p_loss.item())
            self.data['loss'][1].append(v_loss.item())
            if self.agent.use_SR:
                self.data['loss'][2].append(psi_loss.item())
            self.data['trials_run_to_date'] += 1
            if get_samples:
                pol_grid, val_grid = get_snapshot(sample_observations, self.env, self.agent)
                self.data['pol_tracking'].append(pol_grid)
                self.data['val_tracking'].append(val_grid)
                self.data['t'].append(trial)

            if trial == 0:
                running_rwdavg = self.reward_sum
            else:
                running_rwdavg = ((trial) * running_rwdavg + self.reward_sum) / (trial + 2)

            if trial == 0 or trial%print_freq==0 or trial == NUM_TRIALS - 1:
                print(f"{trial}: {self.reward_sum} ({time.time() - t}s / Running Av: {running_rwdavg}")
                t = time.time()

    def runMF(self, NUM_TRIALS, NUM_EVENTS, **kwargs):
        print('running model free control')
        self.num_trials = NUM_TRIALS
        self.num_events = NUM_EVENTS
        self.use_memory = False

        print_freq         = kwargs.get('printfreq', 100)
        get_samples        = kwargs.get('get_samples', False)

        reset_data = kwargs.get('reset_data',True)
        if reset_data:
            self.data = self.reset_data_logs()

        self.timestamp = 0
        self.starts = kwargs.get('starts', None)
        sample_observations = self.env.get_sample_obs()
        t = time.time()

        for trial in range(NUM_TRIALS):
            self.trial_reset(trial)

            for event in range(NUM_EVENTS):
                # get state observation
                observation = torch.Tensor(np.expand_dims(self.env.get_observation(), axis=0))
                # pass observation through network
                if self.agent.use_SR:
                    policy_, value_, phi_, psi_ = self.agent(observation)
                    self.agent.saved_phi.append(phi_)
                    self.agent.saved_psi.append(psi_)
                else:
                    policy_, value_, psi_ = self.agent(observation)

                # linear activity
                choice, poli = self.action_selection(policy_, value_)

                # select action from policy
                action = self.env.action_list[choice][0]

                # take a step in the environment
                s_1d, reward, isdone = self.env.move(action)

                self.agent.saved_rewards.append(reward)
                self.reward_sum += reward

                self.timestamp += 1

                if isdone:
                    if self.agent.use_SR:
                        self.agent.saved_phi.append(phi_)
                        self.agent.saved_psi.append(psi_)
                    encountered_reward = True
                    break

            if self.agent.use_SR:
                p_loss, v_loss, psi_loss = self.agent.finish_trial()
            else:
                p_loss, v_loss = self.agent.finish_trial()

            self.data['trial_length'].append(event+1)
            self.data['total_reward'].append(self.reward_sum)
            self.data['loss'][0].append(p_loss.item())
            self.data['loss'][1].append(v_loss.item())
            if self.agent.use_SR:
                self.data['loss'][2].append(psi_loss.item())

            if get_samples:
                pol_grid, val_grid = get_snapshot(sample_observations, self.env, self.agent)
                self.data['pol_tracking'].append(pol_grid)
                self.data['val_tracking'].append(val_grid)
            if trial == 0:
                running_rwdavg = self.reward_sum
            else:
                running_rwdavg = ((trial)*running_rwdavg + self.reward_sum)/(trial+2)

            if trial == 0 or trial%print_freq==0 or trial == NUM_TRIALS - 1:
                print(f"{trial}: {self.reward_sum} ({time.time() - t}s / Running Av: {running_rwdavg}")
                t = time.time()



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
                                arch = ['B'], ## default
                                )
        load_id = list(filtered_d.ids)[0]
        # load saved environment
        with open(parent_dir + f'environments/{load_id}_env.p', 'rb') as f:
            env = pickle.load(f)

        if expt_type in [1,2,3,4,5,6]:
            #rshift
            env_params.reward_location = (15,15)   # for expt_type <10
            env.rewards = {env_params.reward_location: env_params.reward_mag}
            env.buildRewardFunction()
            env.finish_after_first_reward = True ### possibly don't need
        if expt_type in [11,12,13,14,15,16]:
            #pshift
            env.remapTransitionMatrix()

        agent_params = basic_agent_params(env)
        agent_params.load_model = True
        agent_params.load_dir = parent_dir + f'agent_weights/{load_id}.pt'


        if expt_type in [1,2]:
            agent_params.freeze_w = True

        agent = ac.make_agent(agent_params.__dict__)

        if expt_type in [2,3,5,6,12,15]:
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



def get_snapshot(sample_obs, env, agent):
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


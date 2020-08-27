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


#### temp delete after aug 21
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.cm as cmx
import matplotlib.patches as patches

from os import listdir
from os.path import isfile, join
from scipy.stats import entropy
def make_arrows(action, probability):
    '''
    alternate style:
        def make_arrows(action):
        offsets = [(0,0.25),(0,-0.25),(0.25,0),(-0.25,0),(0,0),(0.1,0.1) ] # D U R L J P
        dx,dy = offsets[action]
        head_w, head_l = 0.1, 0.1
        return dx, dy, head_w, head_l
    :param action:
    :param probability:
    :return:
    '''
    if probability == 0:
        dx, dy = 0, 0
        head_w, head_l = 0, 0
    else:
        dxdy = [(0.0, 0.25),  # D
                (0.0, -0.25),  # U
                (0.25, 0.0),  # R
                (-0.25, 0.0),  # L
                (0.1, -0.1),  # points right and up #J
                (-0.1, 0.1),  # points left and down # P
                ]
        dx, dy = dxdy[action]

        head_w, head_l = 0.1, 0.1

    return dx, dy, head_w, head_l


def plot_pref_pol(maze, policy_array, save=False, **kwargs):
    '''
        :param maze: the environment object
        :param save: bool. save figure in current directory
        :return: None
        '''
    show = kwargs.get('show', True)
    title = kwargs.get('title', 'Policy Entropy')
    directory = kwargs.get('directory', '../data/figures/')
    filetype = kwargs.get('filetype', 'png')
    vmax = kwargs.get('upperbound', 2)
    rewards = kwargs.get('rwds', maze.rewards)
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_axes([0, 0, 0.85, 0.85])
    axc = fig.add_axes([0.75, 0, 0.05, 0.85])

    cmap = plt.cm.Spectral_r
    cNorm = colors.Normalize(vmin=0, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
    # make base grid
    ax1.pcolor(maze.grid, vmin=0, vmax=vmax, cmap='bone')
    # add patch for reward location/s (red)
    for rwd_loc in rewards:
        rwd_r, rwd_c = rwd_loc
        ax1.add_patch(plt.Rectangle((rwd_c, rwd_r), width=0.99, height=1, linewidth=1, ec='white', fill=False))

    chance_threshold = kwargs.get('threshold', 0.18)  # np.round(1 / len(maze.actionlist), 6)

    for i in range(maze.r):
        for j in range(maze.c):
            policy = tuple(policy_array[i, j])

            dx, dy = 0.0, 0.0
            for ind, k in enumerate(policy):
                action = ind
                prob = k
                if prob < 0.01:
                    pass
                else:
                    dx1, dy1, head_w, head_l = make_arrows(action, prob)
                    dx += dx1*prob
                    dy += dy1*prob
            if dx ==0.0 and dy == 0.0:
                pass
            else:
                colorVal1 = scalarMap.to_rgba(entropy(policy))
                ax1.arrow(j + 0.5, i + 0.5, dx, dy, head_width=0.3, head_length=0.5, color=colorVal1)


    ax1.set_aspect('equal')
    ax1.set_title(title)
    ax1.invert_yaxis()

    if save:
        plt.savefig(f'{directory}{title}.{filetype}', format=f'{filetype}', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_polmap(maze, policy_array, save=False, **kwargs):
    '''
    :param maze: the environment object
    :param save: bool. save figure in current directory
    :return: None
    '''
    show = kwargs.get('show', True)
    title = kwargs.get('title', 'Most Likely Action from Policy')
    filetype = kwargs.get('filetype', 'png')
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_axes([0, 0, 0.85, 0.85])
    axc = fig.add_axes([0.75, 0, 0.05, 0.85])

    cmap = plt.cm.Spectral_r
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    # make base grid
    ax1.pcolor(maze.grid, vmin=0, vmax=1, cmap='bone')
    # add patch for reward location/s (red)
    for rwd_loc in maze.rewards:
        rwd_y, rwd_x = rwd_loc
        ax1.add_patch(plt.Rectangle((rwd_y, rwd_x), width=0.99, height=1, linewidth=1, ec='white', fill=False))

    chance_threshold = kwargs.get('threshold', 0.18)  # np.round(1 / len(maze.actionlist), 6)

    cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
    for i in range(maze.c):
        for j in range(maze.r):
            action = np.argmax(tuple(policy_array[i][j]))
            prob = max(policy_array[i][j])

            dx1, dy1, head_w, head_l = make_arrows(action, prob)
            if prob > chance_threshold:
                if (dx1, dy1) == (0, 0):
                    pass
                else:
                    colorVal1 = scalarMap.to_rgba(prob)
                    ax1.arrow(j + 0.5, i + 0.5, dx1, dy1, head_width=0.3, head_length=0.2, color=colorVal1)
            else:
                pass
    ax1.set_aspect('equal')
    ax1.set_title(title)
    ax1.invert_yaxis()

    if save:
        plt.savefig(f'../data/figures/p_{title}.{filetype}', format=f'{filetype}', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
#### / temp

class Experiment(object):
    def __init__(self, agent, environment, **kwargs):
        self.agent = agent
        if self.agent.use_SR:
            self.agent_architecture = 'B'
        else:
            self.agent_architecture = 'A'
        self.env = environment

        self.data = self.reset_data_logs()

        self.use_memory = kwargs.get('use_mem', False)
        self.rec_memory = kwargs.get('rec_mem', False)
        self.mem_size = kwargs.get('mem_size', 0.75 * environment.nstates)
        self.episodic = kwargs.get('mem', None)

        self.alpha = kwargs.get('alpha', 1)
        self.beta = kwargs.get('beta', 10000)
        threshold = 0.01
        self.decay = np.power(threshold, 1 / self.beta)
        self.MF_cs = 0 ## temp aug 13

        self.wo_pen = kwargs.get('wopen', False) # if false, step penalization factors into MFCS arbitration


    def reset_data_logs(self):
        data_log = {'total_reward': [],
                    'loss': [[],[],[]],
                    'trial_length': [],
                    'trials_run_to_date':0,
                    'pol_tracking':[],
                    'val_tracking':[],
                    'ec_tracking': [],
                    'starts': [],
                    't': [],
                    'mfcs':[],
                    'arbitration_count': [[],[],[]],
                    'confidence_selection': [[],[]], #mfcs, whether ec or mf chosen
                    'sar': [],
                    'visited_states': []
                   }
        return data_log

    def policy_arbitration(self,reward_tminus1):
        if self.wo_pen:
            if reward_tminus1 < 0:
                reward_tminus1 = 0
        confidence_in_model_free =(self.decay*self.MF_cs) + self.alpha*reward_tminus1

        if confidence_in_model_free > 1.0:
            self.MF_cs = 1.0
        elif confidence_in_model_free < 0.0:
            self.MF_cs = 0.0
        else:
            self.MF_cs = confidence_in_model_free


    def trial_reset(self, trial):
        # reset environment, choose new starting location for agent
        self.env.resetEnvironment(around_rwd=self.around_reward, radius=self.start_radius)
        # clear hidden layer cache if using lstm or gru cells
        self.agent.reinit_hid()
        self.reward_sum = 0

        self.memory_buffer = [[], [], [], [], trial]  # [timestamp, state_t, a_t, readable_state, trial]
        if self.use_memory:
            if trial == 0 :
                self.MF_cs = 0.0
            self.last_reward = 0
        else:
            self.MF_cs = 1.0
        self.mfcount= 0
        self.eccount =0

    def action_selection(self, policy_, value_, lin_act=None):
        '''
        if self.use_memory:
            # compute MFCS
            #self.policy_arbitration(self.last_reward)
            self.MF_cs = 0.01
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

            #self.data['confidence_selection'][0].append(self.MF_cs)
            #self.data['confidence_selection'][1].append(cstar)
            #print(self.MF_cs, cstar)
        else:
            choice = self.agent.select_action(policy_, value_)

        return choice, policy_
        '''
        #self.eccount += 1
        episodic_memory = self.episodic.recall_mem(key=lin_act, timestep=self.timestamp, env=self.recency_env)
        episodic_pol = torch.from_numpy(episodic_memory)
        choice = self.agent.select_ec_action(policy_, value_, episodic_pol)
        policy_ = episodic_pol

        return choice, policy_

    def save_to_mem(self, timestamp, lin_act, choice, current_state, trial):
        self.memory_buffer[0].append(timestamp)
        self.memory_buffer[1].append(lin_act)
        self.memory_buffer[2].append(choice)
        self.memory_buffer[3].append(current_state)
        self.memory_buffer[4] = trial

    def run(self, NUM_TRIALS, NUM_EVENTS, **kwargs):
        self.num_trials = NUM_TRIALS
        self.num_events = NUM_EVENTS
        print_freq         = kwargs.get('printfreq', 100)
        get_samples        = kwargs.get('get_samples', False)
        self.around_reward = kwargs.get('around_reward', True)
        self.start_radius  = kwargs.get('radius', 5)

        reset_data = kwargs.get('reset_data',True)
        if reset_data:
            self.data = self.reset_data_logs()

        #self.ploss_scale   = 0  # this is equivalent to calculating MF_confidence = sech(0) = 1
        #self.mfc_env = ec.calc_envelope(halfmax=3.12)  # 1.04 was the calculated standard deviation of policy loss after learning on open field gridworld task **** may need to change for different tasks
        if self.episodic != None:
            self.recency_env = self.episodic.calc_envelope(halfmax=20)

        self.timestamp = 0

        self.starts = kwargs.get('starts', None)

        #if get_samples: ### aug21 temp change back
        #    sample_observations = self.env.get_sample_obs()
        sample_observations = self.env.get_sample_obs()

        t = time.time()
        encountered_reward = False
        print(f"running alpha {self.alpha}, beta: {self.beta}")
        for trial in range(NUM_TRIALS):
            self.data['sar'].append([])
            self.trial_reset(trial)
            if self.starts is not None:
                start_ = self.starts[np.random.choice(np.arange(4))]
                self.env.set_state(self.env.twoD2oneD(start_))
            self.data['starts'].append(self.env.oneD2twoD(self.env.state))
            ec_trace  = 0
            ec_events = []
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
                if self.rec_memory or self.use_memory:
                    lin_act = tuple(np.round(psi_.data[0].numpy(), 4))
                    choice, poli = self.action_selection(policy_, value_,lin_act)
                else:
                    choice, poli = self.action_selection(policy_, value_)

                if self.eccount > ec_trace:
                    ec_events.append(event)
                    ec_trace = self.eccount

                # select action from policy
                action = self.env.action_list[choice][0]

                if self.rec_memory or self.use_memory:
                    self.save_to_mem(self.timestamp, lin_act, choice, self.env.oneD2twoD(self.env.state), trial)

                # take a step in the environment
                s_1d, reward, isdone = self.env.move(action)
                #self.data['sar'][trial].append((self.env.oneD2twoD(self.env.state), poli.detach().numpy(), action, reward))### try
                #temp
                #self.data['visited_states'].append(self.env.oneD2twoD(self.env.state))
                #/temp

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

            if self.rec_memory or self.use_memory:
                if self.agent.use_SR:
                    p_loss, v_loss, psi_loss = self.agent.finish_trial_EC(cache=self.episodic, buffer=self.memory_buffer)
                else:
                    p_loss, v_loss  = self.agent.finish_trial_EC(cache=self.episodic, buffer=self.memory_buffer)
            else:
                if self.agent.use_SR:
                    p_loss, v_loss, psi_loss = self.agent.finish_trial()
                else:
                    p_loss, v_loss = self.agent.finish_trial()

            #self.data['arbitration_count'][0].append(self.mfcount)
            #self.data['arbitration_count'][1].append(self.eccount)
            #self.data['arbitration_count'][2].append(ec_events)
            #print(f'MF:{self.mfcount}/EC:{self.eccount} || R:{self.reward_sum} ')
            #self.ploss_scale = abs(p_loss.item())
            #print(np.vstack(self.data['sar'][trial]))
            # temp
            #mpol_array = np.zeros(self.env.grid.shape, dtype=[(x, 'f8') for x in self.env.action_list])
            #mem = self.episodic
            #for i in mem.cache_list.keys():
            #    values = mem.cache_list[i]
            #    row, col = values[2]
            #    pol = mem.recall_mem(i, timestep=0)
            #    mpol_array[row, col] = tuple(pol)
            #plot_polmap(self.env, mpol_array, title='argmax pol EC')
            #plot_pref_pol(self.env, mpol_array, title='pref pol EC')
            #/temp

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
            if trial == 0 or trial%print_freq==0 or trial == NUM_TRIALS - 1:
                print(f"{trial}: {self.reward_sum} ({time.time() - t}s / {event} steps - MF selected {self.mfcount} times)")
                t = time.time()
            ### temp aug 21
            #pol_grid, val_grid = get_snapshot(sample_observations, self.env, self.agent)

            #plot_polmap(self.env, pol_grid, title='argmax pol MF')
            #plot_pref_pol(self.env, pol_grid,  title='Preferred pol MF')

            ### /temp


            if self.around_reward and trial > 0 and trial == int(NUM_TRIALS / 2):  # np.mean(data['trial_length'][-20:])< 2*start_radius:
                print(trial)
                self.around_reward = False

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

def data_log(run_id, experiment_type, experiment, **kwargs):
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
        # write to logger
        with open(parent_folder+'experiments_log.csv', 'a+', newline='') as file:
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
        if experiment.episodic is not None:
            pickle.dump(experiment.episodic, open(f'{parent_folder}episodic_memory/{run_id}_EC.p', 'wb'))
        print(f'{run_id} recorded')
    else:
        print_dict = {}
        for i in range(len(log_jam)):
            print_dict[expt_log[i]] = log_jam[i]
        print(print_dict)
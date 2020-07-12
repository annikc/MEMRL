### =============================================================================
### Updated June 2020
### =============================================================================
from __future__ import division
import numpy as np
import time
import torch
import pickle
import csv
import uuid
import sys
# import modules from other folders in the tree
sys.path.insert(0,'../rl_network/'); import ac;  import stategen as sg
sys.path.insert(0,'../memory/'); import episodic as ec
sys.path.insert(0,'../environments/'); import gw; import gridworld_plotting as gp

####################################################
def log_experiments(save_id, experiment_type, env, agent, data, mem=None, **kwargs):
    arch_type = kwargs.get('arch', 'B')
    load = kwargs.get('load', ' ')
    alpha = kwargs.get('alpha', -1)
    beta = kwargs.get('beta', -1)

    save_flag = kwargs.get('save_flag', False)
    if experiment_type == 0 or save_flag:
        save = f'../data/outputs/gridworld/weights/{save_id}.pt'
        ac.torch.save(agent, save)
    else:
        save = kwargs.get('save', ' ')

    pvals = kwargs.get('pvals', False)

    expt_log = [save_id, experiment_type, load, save]
    # add environment parameters
    if env.maze_type == None:
        maze_type = 'Openfield'
    else:
        maze_type = env.maze_type
    expt_log.append(maze_type) # type of environment
    expt_log.append(env.shape) # dimensions
    expt_log.append(len(env.action_list)) # number of actions
    if env.rwd_action == None:
        rwd_action = 'None'
    else:
        rwd_action = env.rwd_action
    expt_log.append(rwd_action)
    expt_log.append(env.rewards)
    expt_log.append(env.step_penalization)
    expt_log.append(env.rho)

    # add agent parameters
    expt_log.append(arch_type)

    if mem is not None:
        expt_log.append(mem.mem_temp)
        expt_log.append(pvals)
        expt_log.append(mem.memory_envelope)
        pickle.dump(mem.cache_list, open(f'../data/outputs/gridworld/episodic_cache/{save_id}_EC.p', 'wb'))
    else:
        expt_log.append(-1)  # mem_temp = EC_entropy
        expt_log.append(pvals)
        expt_log.append(-1)  # Mem_decay

    # switching parameters
    expt_log.append(alpha)
    expt_log.append(beta)

    with open('../data/outputs/gridworld/experiments.csv', 'a+', newline='') as file:
        writer = csv.writer(file)
        if experiment_type is not None:
            writer.writerow(expt_log)
            experiment_type = None
        else:
            raise Exception('enter experiment type ')
    pickle.dump(data, open(f'../data/outputs/gridworld/results/{save_id}_data.p', 'wb'))

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

class test_expt(object):
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
                    'mfcs':[]
                   }
        return data_log

    def policy_arbitration(self,reward_tminus1):
        threshold = 0.01
        decay = np.power(threshold,1/self.beta)
        self.MF_cs = decay*self.MF_cs + self.alpha*reward_tminus1
        if self.MF_cs > 1.0:
            self.MF_cs = 1.0
        elif self.MF_cs < 0.0:
            self.MF_cs = 0.0

    def trial_reset(self, trial):
        # reset environment, choose new starting location for agent
        self.env.resetEnvironment(around_rwd=self.around_reward, radius=self.start_radius)
        # clear hidden layer cache if using lstm or gru cells
        self.agent.reinit_hid()
        self.reward_sum = 0

        self.memory_buffer = [[], [], [], [], trial]  # [timestamp, state_t, a_t, readable_state, trial]
        if self.use_memory:
            self.MF_cs = 0.0
            self.last_reward = 0
        else:
            self.MF_cs = 1.0

    def action_selection(self, policy_, value_, lin_act=None):
        policy = policy_.data[0]
        value = value_.item()
        if self.use_memory:
            # compute MFCS
            self.policy_arbitration(self.last_reward)
            pol_choice = np.random.choice(['mf', 'ec'], p=[self.MF_cs, 1 - self.MF_cs])
            #print(self.MF_cs, pol_choice, self.last_reward)
            if pol_choice == 'ec':
                episodic_memory = self.episodic.recall_mem(key=lin_act, timestep=self.timestamp, env=self.recency_env)
                episodic_pol = torch.from_numpy(episodic_memory)
                choice = self.agent.select_ec_action(policy_, value_, episodic_pol)
            else:
                choice = self.agent.select_action(policy_, value_)
        else:
            choice = self.agent.select_action(policy_, value_)

        return choice

    def save_to_mem(self, timestamp, lin_act, choice, current_state, trial):
        self.memory_buffer[0].append(timestamp)
        self.memory_buffer[1].append(lin_act)
        self.memory_buffer[2].append(choice)
        self.memory_buffer[3].append(current_state)
        self.memory_buffer[4] = trial

    def run(self, NUM_TRIALS, NUM_EVENTS, **kwargs):
        self.num_trials = NUM_TRIALS
        self.num_events = NUM_EVENTS
        print_freq         = kwargs.get('printfreq', 0.1)
        get_samples        = kwargs.get('get_samples', False)
        self.around_reward = kwargs.get('around_reward', False)
        self.start_radius  = kwargs.get('radius', 5)

        self.alpha = kwargs.get('alpha', 1)
        self.beta =  kwargs.get('beta', 10000) # full model free control
        reset_data = kwargs.get('reset_data', False)
        if reset_data:
            self.data = self.reset_data_logs()

        #self.ploss_scale   = 0  # this is equivalent to calculating MF_confidence = sech(0) = 1
        #self.mfc_env = ec.calc_envelope(halfmax=3.12)  # 1.04 was the calculated standard deviation of policy loss after learning on open field gridworld task **** may need to change for different tasks

        self.recency_env = ec.calc_envelope(halfmax=20)

        self.timestamp = 0

        self.starts = kwargs.get('starts', None)

        if get_samples:
            sample_observations = self.env.get_sample_obs()

        t = time.time()
        encountered_reward = False
        for trial in range(NUM_TRIALS):
            self.trial_reset(trial)
            if self.starts is not None:
                start_ = self.starts[np.random.choice(np.arange(4))]
                self.env.set_state(self.env.twoD2oneD(start_))
            self.data['starts'].append(self.env.oneD2twoD(self.env.state))
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
                    choice = self.action_selection(policy_, value_,lin_act)
                else:
                    choice = self.action_selection(policy_, value_)
                # select action from policy
                action = self.env.action_list[choice][0]

                if self.rec_memory or self.use_memory:
                    self.save_to_mem(self.timestamp, lin_act, choice, self.env.oneD2twoD(self.env.state), trial)

                #print(self.env.oneD2twoD(self.env.state),action)
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

            #self.ploss_scale = abs(p_loss.item())
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

            if trial == 0 or trial % int(print_freq * NUM_TRIALS) == 0 or trial == NUM_TRIALS - 1:
                print(f"{trial}: {self.reward_sum} ({time.time() - t}s)")
                t = time.time()

            if self.around_reward and trial > 0 and trial == int(NUM_TRIALS / 2):  # np.mean(data['trial_length'][-20:])< 2*start_radius:
                print(trial)
                self.around_reward = False
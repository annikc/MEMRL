### =============================================================================
### Updated Feb 2020
### =============================================================================
from __future__ import division
import numpy as np
import time
import torch
import sys
# import modules from other folders in the tree
sys.path.insert(0,'../rl_network/'); import ac;  import stategen as sg
sys.path.insert(0,'../memory/'); import episodic as ec
sys.path.insert(0,'../environments/'); import gw; import gridworld_plotting as gp
from scipy.stats import entropy
# temp
import matplotlib.pyplot as plt
import time
#/
####################################################
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
        self.env = environment
        self.use_memory = kwargs.get('use_mem', False)
        self.rec_memory = kwargs.get('rec_mem', False)
        if self.rec_memory or self.use_memory:
            self.mem_size = 0.75 * environment.nstates
            self.episodic = ec.ep_mem(self.agent, cache_limit=self.mem_size)

    def trial_reset(self, trial):
        # reset environment, choose new starting location for agent
        self.env.resetEnvironment(around_rwd=self.around_reward, radius=self.start_radius)
        # clear hidden layer cache if using lstm or gru cells
        self.agent.reinit_hid()
        self.reward_sum = 0

        #if self.record_memory: #self.episodic.reset_cache()  ## why did I have this here?
        self.memory_buffer = [[], [], [], [], trial]  # [timestamp, state_t, a_t, readable_state, trial]
        if self.use_memory:
            pass
            #self.MF_cs = self.episodic.make_pvals(self.ploss_scale, envelope=self.mfc_env)
        else:
            self.MF_cs = 1

    def action_selection(self, policy_, value_, lin_act=None):
        policy = policy_.data[0]
        value = value_.item()
        if self.use_memory:
            pol_choice = 'ec' #np.random.choice(['mf', 'ec'], p=[self.MF_cs, 1 - self.MF_cs])
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

    def run(self, NUM_TRIALS, NUM_EVENTS, data, **kwargs):
        print_freq         = kwargs.get('printfreq', 0.1)
        get_samples        = kwargs.get('get_samples', False)
        self.around_reward = kwargs.get('around_reward', False)
        self.start_radius  = kwargs.get('radius', 5)

        self.ploss_scale   = 0  # this is equivalent to calculating MF_confidence = sech(0) = 1
        self.mfc_env = ec.calc_env(halfmax=3.12)  # 1.04 was the calculated standard deviation of policy loss after learning on open field gridworld task **** may need to change for different tasks
        self.recency_env = ec.calc_env(halfmax=20)

        self.timestamp = 0

        if get_samples:
            sample_observations = self.env.get_sample_obs()

        t = time.time()
        encountered_reward = False
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
                if self.rec_memory or self.use_memory:
                    lin_act = tuple(np.round(psi_.data[0].numpy(), 4))
                    choice = self.action_selection(policy_, value_,lin_act)
                else:
                    choice = self.action_selection(policy_, value_)
                # select action from policy
                action = self.env.action_list[choice][0]

                if self.rec_memory or self.use_memory:
                    self.save_to_mem(self.timestamp, lin_act, choice, self.env.oneD2twoD(self.env.state), trial)

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

            self.ploss_scale = abs(p_loss.item())
            data['trial_length'].append(event)
            data['total_reward'].append(self.reward_sum)
            data['loss'][0].append(p_loss.item())
            data['loss'][1].append(v_loss.item())
            if self.agent.use_SR:
                data['loss'][2].append(psi_loss.item())
            data['trials_run_to_date'] += 1
            if get_samples:
                pol_grid, val_grid = get_snapshot(sample_observations, self.env, self.agent)
                data['pol_tracking'].append(pol_grid)
                data['val_tracking'].append(val_grid)
                data['t'].append(trial)

            if trial == 0 or trial % int(print_freq * NUM_TRIALS) == 0 or trial == NUM_TRIALS - 1:
                print(f"{trial}: {self.reward_sum} ({time.time() - t}s)")
                t = time.time()

            if self.around_reward and trial > 0 and trial == int(NUM_TRIALS / 2):  # np.mean(data['trial_length'][-20:])< 2*start_radius:
                print(trial)
                self.around_reward = False
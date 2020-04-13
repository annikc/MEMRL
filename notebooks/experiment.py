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
    pols, vals = agent(torch.Tensor(samples))

    # initialize empty data frames
    pol_grid = np.zeros(env.shape, dtype=[('N', 'f8'), ('E', 'f8'), ('W', 'f8'), ('S', 'f8'), ('stay', 'f8'), ('poke', 'f8')])
    val_grid = np.empty(env.shape)

    # populate with data from network
    for s, p, v in zip(states, pols, vals):
        pol_grid[s] = tuple(p.data.numpy())
        val_grid[s] = v.item()

    return pol_grid, val_grid


######################################################
def run(run_dict, full=False, use_EC = False, **kwargs):
    '''
    :param run_dict: dictionary storing data frames and run parameters
    :param full: boolean -- True: trials run for entire NUM_EVENTS, False: trials run until first reward collected
    :param use_EC: boolean, whether to use episodic system or not
    '''
    KLD = False
    if KLD:
        run_dict['kld'] = []
        maze= run_dict['environment']
        reward_c, reward_r = maze.rwd_loc[0]
        pol_array = np.zeros(maze.grid.shape,
                             dtype=[('N', 'f8'), ('E', 'f8'), ('W', 'f8'), ('S', 'f8'), ('stay', 'f8'), ('poke', 'f8')])
        mag = 100
        for i in range(pol_array.shape[0]):  # rows
            for j in range(pol_array.shape[1]):  # columns
                # D: N
                # U: S
                # L: W
                # R: E
                if i < reward_r:
                    D = 0
                    U = mag
                elif i > reward_r:
                    D = mag
                    U = 0
                else:
                    D = 0
                    U = 0
                if j < reward_c:
                    R = mag
                    L = 0
                elif j > reward_c:
                    R = 0
                    L = mag
                else:
                    R = 0
                    L = 0
                stay = 0
                poke = 0
                if i == reward_r and j == reward_c:
                    poke = mag

                actions = [D, R, L, U, stay, poke]
                policy = ac.softmax(actions)
                pol_array[i][j] = tuple(policy)



    # get run parameters from run_dict
    NUM_TRIALS  = run_dict['NUM_TRIALS']
    NUM_EVENTS  = run_dict['NUM_EVENTS']
    # specify environment
    maze        = run_dict['environment']
    # specify agent
    agent_params= run_dict['agt_param']
    MF          = run_dict['agent']
    opt         = run_dict['optimizer']

    # initialize data frames
    run_dict['total_loss']  = [[], []]
    run_dict['total_reward']= []
    run_dict['track_cs']    = [[], []]
    run_dict['rpe']         = np.zeros(maze.grid.shape)

    rec_mem     = kwargs.get('rec_mem', False)
    print_freq  = kwargs.get('print_freq', 100)
    save_data   = kwargs.get('save', True)
    saveplots   = kwargs.get('plots', False)

    if not full:
        run_dict['trial_length'] = []
    if rec_mem:
        EC = agent_params['EC']
        EC.reset_cache()

    ploss_scale = 0                           # this is equivalent to calculating MF_confidence = sech(0) = 1
    mfc_env     = ec.calc_env(halfmax = 3.12) # 1.04 was the calculated standard deviation of policy loss after learning on open field gridworld task **** may need to change for different tasks
    recency_env = ec.calc_env(halfmax = 20)
    reward      = 0
    timestamp   = 0
    is_done     = False
    blocktime   = time.time()

    vvs = []
    ### RUN THE TRIALS
    for trial in range(NUM_TRIALS):
        if is_done == True:
            break

        # reset
        maze.reset()
        MF.reinit_hid()
        if use_EC:
            MF_cs =  EC.make_pvals(ploss_scale, envelope=mfc_env, shift =0)

        memory_buffer = [[], [], [], [], trial]  # [timestamp, state_t, a_t, readable_state, trial]
        reward_sum = 0

        for event in range(NUM_EVENTS):
            state = torch.Tensor(maze.observation) # tensorize state -- can do this better

            policy_, value_, lin_act_ = MF(state) # pass through AC network to get MF policy / value
            lin_act = tuple(np.round(lin_act_.data[0].numpy(), 4))   # get activity of linear layer for EC dict key

            if use_EC:
                pol_choice = np.random.choice([0,1], p=[MF_cs, 1-MF_cs])
                if pol_choice:
                    # get policy from EC
                    pol = torch.from_numpy(EC.recall_mem(lin_act, timestamp, env=recency_env, mem_temp=agent_params['mem_temp'])) ## check this env parameter
                    choice, policy, value = ac.select_ec_action(MF, policy_, value_, pol)
                else:
                    choice, policy, value = ac.select_action(MF,policy_, value_)
            else:
                choice, policy, value = ac.select_action(MF,policy_, value_)

            if rec_mem:
                # save data to memory buffer
                memory_buffer[0].append(timestamp)
                memory_buffer[1].append(lin_act)
                memory_buffer[2].append(choice)
                memory_buffer[3].append(maze.cur_state)
                memory_buffer[4] = trial

            if event < NUM_EVENTS:
                next_state, reward, done, info = maze.step(choice)

            MF.rewards.append(reward)
            reward_sum += reward
            timestamp += 1

            if reward == maze.rwd_mag:
                if not full:
                    run_dict['trial_length'].append(event)
                    is_done = False
                    break
            if not full:
                if event == NUM_EVENTS-1:
                    run_dict['trial_length'].append(event)

        ### some policy bullshit at reward location
        #check_policy.append(MF(torch.Tensor(maze.get_frame(agtlocation=(12, 12))))[0].data[0])
        #if trial == 2500:
        #vv, pp = ac.snapshot(agent=run_dict['agent'], maze=run_dict['environment'])
        #vvs.append(vv)
        #    gp.plot_polmap(run_dict['environment'], pp)
        #    gp.plot_valmap(run_dict['environment'], vv, v_range=[0, 1])
        #    run_dict['environment'].rwd_loc = [(12,12)]
        if rec_mem:
            p_loss, v_loss = ac.finish_trial(MF,agent_params['gamma'],opt,cache=EC, buffer=memory_buffer)
        else:
            p_loss, v_loss = ac.finish_trial(MF,agent_params['gamma'],opt)

        ploss_scale = abs(p_loss.item())
        if save_data:
            run_dict['total_loss'][0].append(p_loss.item())
            run_dict['total_loss'][1].append(v_loss.item())
            run_dict['total_reward'].append(reward_sum)

        if KLD:
            ec_policies = ac.mem_snapshot(run_dict['environment'], agent_params['EC'], trial_timestamp=50, decay=recency_env, mem_temp=0.3)
            # __, mf_pols = ac.snapshot(agent=run_dict['agent'], maze=run_dict['environment'])
            mf_policies = pol_array #np.vstack(mf_pols)

            kld = np.zeros(ec_policies.shape)
            for i in range(ec_policies.shape[0]):
                for j in range(ec_policies.shape[1]):
                    if sum([e for e in ec_policies[i][j]]) == 0.0:
                        kld[i][j] = np.nan
                    else:
                        mf_pol = ac.softmax([m for m in mf_policies[i][j]])
                        ec_pol = ac.softmax([e for e in ec_policies[i][j]])
                        kld[i][j] = entropy(ec_pol, mf_pol)
            run_dict['kld'].append(kld)
        run_dict['pp']= vvs


        if saveplots:
            vv, pp = ac.snapshot(agent=run_dict['agent'], maze =run_dict['environment'])

            gp.plot_polmap(run_dict['environment'], pp, save=True, show=False, title=f"{trial}")
            gp.plot_valmap(run_dict['environment'], vv, save=True, show=False, title=f"{trial}")

            abcd = ac.mem_snapshot(run_dict['environment'], agent_params['EC'], trial_timestamp=trial, decay=recency_env, mem_temp=agent_params['mem_temp'], get_vals=False)
            gp.plot_polmap(run_dict['environment'], abcd, threshold=0.22, save=True, show=False, title=f"EC_{trial}")

        if trial ==0 or trial%print_freq==0 or trial == NUM_TRIALS-1:
            print("[{0}]  Trial {1} TotRew = {2} ({3:.3f}s)".format(time.strftime("%H:%M:%S", time.localtime()), trial+1, reward_sum,time.time()-blocktime)) #print("[{0}]  Trial {1} total reward = {2} (Avg {3:.3f})".format(time.strftime("%H:%M:%S", time.localtime()), trial, reward_sum, float(reward_sum)/float(NUM_EVENTS)), "Block took {0:.3f}".format(time.time()-blocktime))
            blocktime = time.time()




class Experiment(object):
    def __init__(self, environment, agent, learning_rate=5e-4, discount_factor=0.98, **kwargs):
        self.env = environment
        self.agent = agent
        self.eta = learning_rate
        self.gamma = discount_factor

        # set up memory
        self.record_memory = kwargs.get('rec_mem', False)
        self.use_memory = kwargs.get('use_mem', False)

        if self.record_memory or self.use_memory:
            self.mem_size = 0.75 * environment.nstates * len(environment.action_list)
            self.episodic = ec.ep_mem(self.agent, cache_limit=self.mem_size)

            self.key_layer = len(self.agent.hidden) - 2
            self.mem_policy_entropy = kwargs.get('EC_entropy', 0.3)

        self.full = kwargs.get('full', False)
        if not self.full:
            self.trial_length = []

        self.around_reward = kwargs.get('around_rwd', False)
        self.reward_radius = kwargs.get('radius', self.env.shape[0])

        self.total_policy_loss = []
        self.total_value_loss = []
        self.total_reward = []

        self.print_freq = kwargs.get('print_freq', 100)
        self.save_data = kwargs.get('save', True)
        self.saveplots = kwargs.get('plots', False)

    def reset(self, trial):
        self.env.resetEnvironment(around_rwd=self.around_reward, radius=self.reward_radius)
        self.agent.reinit_hid()

        if self.record_memory:
            self.episodic.reset_cache()

        if self.use_memory:
            self.MF_cs = self.episodic.make_pvals(self.ploss_scale, envelope=self.mfc_env)
        else:
            self.MF_cs = 1

        self.memory_buffer = [[], [], [], [], trial]  # [timestamp, state_t, a_t, readable_state, trial]
        self.reward_sum = 0

    def save_to_mem(self, timestamp, lin_act, choice, current_state, trial):
        self.memory_buffer[0].append(timestamp)
        self.memory_buffer[1].append(lin_act)
        self.memory_buffer[2].append(choice)
        self.memory_buffer[3].append(current_state)
        self.memory_buffer[4] = trial

    def action_selection(self, policy_, value_, lin_act=None):
        policy = policy_.data[0]
        value = value_.item()
        if self.use_memory:
            pol_choice = np.random.choice(['mf', 'ec'], p=[self.MF_cs, 1 - self.MF_cs])
            if pol_choice == 'ec':
                episodic_memory = self.episodic.recall_mem(lin_act, self.timestamp, env=self.recency_env, mem_temp=self.mem_policy_entropy)
                episodic_pol = torch.from_numpy(episodic_memory)
                choice = self.agent.select_ec_action(policy_, value_, episodic_pol)
            else:
                choice = self.agent.select_action(policy_, value_)
        else:
            choice = self.agent.select_action(policy_, value_)

        return choice, policy, value

    def forward_pass(self, state):
        if self.record_memory or self.use_memory:
            mf_policy, mf_value, lin_act_ = self.agent(state, lin_act=self.key_layer)
            lin_act = tuple(np.round(lin_act_.data[0].numpy(), 4))
        else:
            mf_policy, mf_value = self.agent(state)
            lin_act = None

        return mf_policy, mf_value, lin_act

    def run(self, NUM_TRIALS, NUM_EVENTS, data_storage, **kwargs):
        print_freq = kwargs.get('print_freq', self.print_freq)

        self.ploss_scale = 0  # this is equivalent to calculating MF_confidence = sech(0) = 1
        self.mfc_env = ec.calc_env(halfmax=3.12)  # 1.04 was the calculated standard deviation of policy loss after learning on open field gridworld task **** may need to change for different tasks
        self.recency_env = ec.calc_env(halfmax=20)

        reward = 0
        self.timestamp = 0

        blocktime = time.time()

        for trial in range(NUM_TRIALS):
            self.reset(trial)

            for event in range(NUM_EVENTS):
                state = torch.Tensor(np.expand_dims(self.env.observation, axis=0))

                mf_policy, mf_value, lin_act = self.forward_pass(state)

                choice, policy, value = self.action_selection(mf_policy, mf_value, lin_act)
                action = self.env.action_list[choice][0]
                if self.record_memory:
                    self.save_to_mem(self.timestamp, lin_act, choice, self.env.oneD2twoD(self.env.state), trial)

                if event < NUM_EVENTS:
                    next_state, reward, done = self.env.move(action)

                self.agent.saved_rewards.append(reward)
                self.reward_sum += reward
                self.timestamp += 1

                if done:
                    data_storage['trial_length'].append(event)
                    break

            if self.record_memory:
                p_loss, v_loss = self.agent.finish_trial_EC(cache=self.episodic, buffer=self.memory_buffer)
            else:
                p_loss, v_loss = self.agent.finish_trial()

            self.ploss_scale = abs(p_loss.item())
            data_storage['trial_length'].append(event)
            data_storage['total_reward'].append(self.reward_sum)
            data_storage['loss'][0].append(p_loss.item())
            data_storage['loss'][1].append(v_loss.item())

            # if saveplots:
            # vv, pp = ac.snapshot(agent=run_dict['agent'], maze =run_dict['environment'])

            # gp.plot_polmap(run_dict['environment'], pp, save=True, show=False, title=f"{trial}")
            # gp.plot_valmap(run_dict['environment'], vv, save=True, show=False, title=f"{trial}")

            # abcd = ac.mem_snapshot(run_dict['environment'], agent_params['EC'], trial_timestamp=trial, decay=recency_env, mem_temp=agent_params['mem_temp'], get_vals=False)
            # gp.plot_polmap(run_dict['environment'], abcd, threshold=0.4, save=True, show=False, title=f"EC_{trial}")

            if trial == 0 or trial % print_freq == 0 or trial == NUM_TRIALS - 1:
                print("[{0}]  Trial {1} TotRew = {2} ({3:.3f}s)".format(time.strftime("%H:%M:%S", time.localtime()),
                                                                        trial + 1, self.reward_sum,
                                                                        time.time() - blocktime))  # print("[{0}]  Trial {1} total reward = {2} (Avg {3:.3f})".format(time.strftime("%H:%M:%S", time.localtime()), trial, reward_sum, float(reward_sum)/float(NUM_EVENTS)), "Block took {0:.3f}".format(time.time()-blocktime))
                blocktime = time.time()



# =========================================================================
'''
rows, columns = 20, 20
env = gw.GridWorld(rows=rows,cols=columns,
                   rewards = {(5,5):1},#rewards={(int(rows/2),int(columns/2)):1},#
                   step_penalization=-0.01,
                   rho=0.0)
agent_params = {
    'load_model':  True,
    'load_dir':     f'../data/outputs/gridworld/openfield{rows}{columns}.pt',
    'freeze_w':    False,

    'input_dims':  env.observation.shape,
    'action_dims': len(env.action_list),
    'hidden_types':['conv','pool','conv','pool','linear','linear'],
    'hidden_dims': [None, None, None, None, 500, 200],

    'rfsize':      5,
    'stride':      1,
    'padding':     1,
    'dilation':    1,

    'gamma':       0.98,
    'eta':         5e-4,

    'use_EC':      True,
    'EC':          {},
    'cachelim':    300,
    'mem_temp':    0.3
}
agent = ac.make_agent(agent_params)
data = {'total_reward': [],
        'loss': [[],[]],
        'trial_length': [],
        'trials_run_to_date':0}
expt = Experiment(env, agent, use_ec=True)
expt.run(NUM_TRIALS=10, NUM_EVENTS=200, data_storage=data)
'''
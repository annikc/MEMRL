# write experiment class
# expt class should take agent and environment
# functions for stepping through events/trials, updating,
# collecting data, writing data
# Annik Carson Nov 20, 2020
# TODO
# =====================================
#           IMPORT MODULES            #
# =====================================
import numpy as np
import time
import pickle, csv
import uuid


class Experiment(object):
    def __init__(self, agent, environment, **kwargs):
        self.env = environment
        self.agent = agent
        # self.rep_learner = rep_learner  #TODO add in later
        self.data = self.reset_data_logs()
        self.agent.counter = 0

        # temp
        # only for gridworld environment
        self.sample_obs, self.sample_states = self.env.get_sample_obs()
        self.sample_reps = self.get_reps()
        # / temp

    def reset_data_logs(self):
        data_log = {'total_reward': [],
                    'loss': [[], []],
                    'trial_length': [],
                    'EC_snap': [],
                    'P_snap': [],
                    'V_snap': []
                    }
        return data_log

    def get_reps(self):
        reps = []
        for i in self.sample_states:
            j = self.env.twoD2oneD(i)
            r = np.zeros(self.env.nstates)
            r[j] = 1
            reps.append(r)
        return reps

    def representation_learning(self):
        # TODO
        # to be run before experiment to learn representations of states
        pass

    def get_representation(self):
        # TODO
        # use self.representation_network
        # pass observation from environment
        # output representation to be used for self.agent input
        pass

    def snapshot(self):
        # initialize empty data frames
        pol_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])
        val_grid = np.empty(self.env.shape)

        mem_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])

        # forward pass through network
        pols, vals = self.agent.MFC(self.sample_obs)

        # populate with data from network
        for s, p, v in zip(self.sample_states, pols, vals):
            pol_grid[s] = tuple(p.data.numpy())
            val_grid[s] = v.item()

        for ind, rep in enumerate(self.sample_reps):
            mem_pol = self.agent.EC.recall_mem(tuple(rep))
            state = self.sample_states[ind]
            mem_grid[state] = tuple(mem_pol)

        return pol_grid, val_grid, mem_grid


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
                action, log_prob, expected_value = self.agent.get_action(np.expand_dims(obs, axis=0))  ## expand dims to make batch size =1

                mem_state = tuple(mem_state)# tuple(self.agent.MFC.h_act.detach().numpy()[0])

                # take step in environment
                next_state, reward, done, info = self.env.step(action)

                # end of event
                target_value = 0
                self.reward_sum += reward

                self.agent.log_event(episode=trial, event=self.agent.counter,
                                     state=mem_state, action=action, reward=reward, next_state=next_state,
                                     log_prob=log_prob, expected_value=expected_value, target_value=target_value,
                                     done=done, readable_state=readable)
                self.agent.counter += 1
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
                snaps = self.snapshot()
                self.data['P_snap'].append(snaps[0])
                self.data['V_snap'].append(snaps[1])
                self.data['EC_snap'].append(snaps[2])
                print(f"Episode: {trial}, Score: {self.reward_sum} (Running Avg:{running_rwdavg}) [{time.time()-t}s]")
                t = time.time()

    def record_log(self, **kwargs): ## TODO -- set up logging
        parent_folder = kwargs.get('dir', './data/')
        log_name     = kwargs.get('file', 'records_Dec2020.csv')
        load_from = kwargs.get('load_from', ' ')

        save_id = uuid.uuid4()
        timestamp = time.asctime(time.localtime())

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


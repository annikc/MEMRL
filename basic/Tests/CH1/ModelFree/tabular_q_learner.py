import torch
import gym
import numpy as np
import modules.Agents.Networks as nets
import modules.Agents.EpisodicMemory as Memory
from modules.Agents import Agent
from modules.Experiments import expt as expt
import matplotlib.pyplot as plt
from modules.Utils import running_mean as rm
import time
import uuid
import csv
import pickle


'''
Q learner needs to encounter reward in order to make useful updates to Q table; therefore run(NUM_TRIALS) but no 
num_events argument   
'''



env_name   = 'gym_grid:gridworld-v1'

# create environment
env = gym.make(env_name)
plt.close()

class Tabular_Q_Agent(object):
    def __init__(self, env, n_eps, learning_rate=0.1, discount=0.98):
        self.action_space= np.arange(env.action_space.n)
        self.q_table = np.random.uniform(low=-1, high=1, size=(env.nstates, env.action_space.n))
        self.LEARNING_RATE=learning_rate
        self.DISCOUNT=discount
        self.epsilon      = 0.4
        self.start_eps_decay = 1
        num_episodes = n_eps #TODO clean up
        self.end_eps_decay= num_episodes//2
        self.eps_decay_val= self.epsilon/(self.end_eps_decay-self.start_eps_decay)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            action= np.argmax(self.q_table[state,:])
        else:
            action = np.random.randint(len(self.q_table[state,:]))

        return action

    def q_update(self, current_state, current_action, reward, new_state, done):
        if not done:
            current_q = self.q_table[ current_state, current_action]
            max_future_q = np.max(self.q_table[new_state,:])
            new_q = (1-self.LEARNING_RATE)*current_q + self.LEARNING_RATE*(reward + self.DISCOUNT*max_future_q)
            self.q_table[current_state, current_action] = new_q
        else:
            self.q_table[current_state, current_action] = 0

class Q_Expt(expt):
    def __init__(self, agent, environment):
        super(Q_Expt,self).__init__(agent, environment)

    def record_log(self, expt_type, env_name, n_trials, n_steps, **kwargs): ## TODO -- set up logging
        parent_folder = kwargs.get('dir', './Data/')
        log_name     = kwargs.get('file', 'test_bootstrap.csv')
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
        extra_info = kwargs.get('extra', [])

        log_jam = [timestamp, save_id, env_name, expt_type, n_trials, n_steps, self.agent.LEARNING_RATE, self.agent.DISCOUNT] + extra_info

        # write to logger
        with open(parent_folder + log_name, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log_jam)

        # save data
        with open(f'{parent_folder}results/{save_id}_data.p', 'wb') as savedata:
            pickle.dump(self.data, savedata)
        # save agent Q table
        with open(f'{parent_folder}agents/{save_id}_Qtable.p','wb') as saveQ:
            pickle.dump(agent.q_table, saveQ)
        # save episodic dictionary
        #if self.agent.EC != None:
        #    with open(f'{parent_folder}ec_dicts/{save_id}_EC.p', 'wb') as saveec:
        #        pickle.dump(self.agent.EC.cache_list, saveec)
        print(f'Logged with ID {save_id}')

    def single_step(self,trial):
        state = self.state
        # get action from agent
        action = self.agent.choose_action(state)

        # take step in environment
        next_state, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.agent.q_update(state,action,reward,next_state,done)

        self.state = next_state

        return done

    def end_of_trial(self, trial):
        self.data['total_reward'].append(self.reward_sum) # removed for bootstrap expts
        self.data['P_snap'].append(self.agent.q_table.copy())
        if self.agent.end_eps_decay >= trial >= self.agent.start_eps_decay:
            self.agent.epsilon -= self.agent.eps_decay_val

        if trial <= 10:
            self.running_rwdavg = np.mean(self.data['total_reward'])
        else:
            self.running_rwdavg = np.mean(self.data['total_reward'][-10:-1])

        if trial % self.print_freq == 0:
            print(f"Episode: {trial}, Score: {self.reward_sum} (Running Avg:{self.running_rwdavg}) [{time.time() - self.t}s]")
            ep_reward = self.data['total_reward']

            self.average_reward = sum(ep_reward[-self.print_freq:])/len(ep_reward[-self.print_freq:])
            self.agg_ep_rewards['ep'].append(trial)
            self.agg_ep_rewards['avg'].append(self.average_reward)
            self.agg_ep_rewards['min'].append(min(ep_reward[-self.print_freq:]))
            self.agg_ep_rewards['max'].append(max(ep_reward[-self.print_freq:]))
            print(f'Ep {trial}: Avg {self.average_reward}; Min {min(ep_reward[-self.print_freq:])}; Max {max(ep_reward[-self.print_freq:])}')

            self.t = time.time()

    def run(self, NUM_TRIALS, **kwargs):
        self.print_freq = kwargs.get('printfreq', 100)
        self.reset_data_logs()
        self.t = time.time()
        self.agg_ep_rewards = {'ep':[], 'avg':[], 'min':[], 'max':[], 'qs':[]}


        for trial in range(NUM_TRIALS):
            self.state = self.env.reset()
            self.reward_sum = 0
            done = False

            while not done:
                done = self.single_step(trial)

            self.end_of_trial(trial)




# generate agent
ntrials = 100000
agent = Tabular_Q_Agent(env,n_eps=ntrials)
run = Q_Expt(agent, env)


run.run(NUM_TRIALS=ntrials, printfreq=1000)
run.record_log(dir='../../../Data/',file='Qlearning.csv',expt_type='MF_Q',env_name=env_name,n_trials=ntrials,n_steps='Inf')


def pref_Q_action(qtable):
    action_table = np.zeros(env.shape)
    for state in range(env.nstates):
        state2d = env.oneD2twoD(state)
        action_table[state2d] = np.argmax(qtable[state,:])

    return action_table

pref_action = pref_Q_action(run.agent.q_table)


fig,ax = plt.subplots(2,1,sharex=False)
agg_ep_rewards = run.agg_ep_rewards
ax[0].plot(agg_ep_rewards['ep'], agg_ep_rewards['avg'], label='avg')
ax[0].plot(agg_ep_rewards['ep'], agg_ep_rewards['min'], label='min')
ax[0].plot(agg_ep_rewards['ep'], agg_ep_rewards['max'], label='max')
ax[0].legend(loc=0)
a = ax[1].imshow(pref_action, interpolation='none')
cbar = fig.colorbar(a, ax=ax[1],ticks=[0,1,2,3])
cbar.ax.set_yticklabels(['Down','Up','Right','Left'])


plt.show()

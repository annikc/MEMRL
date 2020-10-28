import numpy as np
import pickle
import pandas as pd

class DataFilter(object):
    def __init__(self, df, **kwargs):
        self.print       = kwargs.get('print', False)
        self.parent_dir  = kwargs.get('parent_dir', './data/')
        self.load_from   = kwargs.get('load_from',       df['load_from'].unique())
        self.exp_type    = kwargs.get('expt_type',       df['experiment_type'].unique()) #list of strings
        self.env_type    = kwargs.get('env_type',        df['ENVIRONMENT'].unique()) #string
        self.dims        = kwargs.get('dims',            df['dims'].unique()) #string
        self.rho         = kwargs.get('rho',             df['rho'].unique()) # float

        self.action_list = kwargs.get('action_list',     df['action_list'].unique()) # int  [str(['Down', 'Up', 'Right', 'Left'])]) #
        self.rewards     = kwargs.get('rewards',         df['rewards'].unique())  # string
        self.rwd_action  = kwargs.get('rewarded_action', df['rwd_action'].unique()) # string
        self.arch        = kwargs.get('arch',            df['AGENT'].unique()) # list of strings
        self.freeze_w    = kwargs.get('freeze_w',        df['freeze_weights'].unique())
        self.pvals       = kwargs.get('use_pvals',       df['use_pvals'].unique()) # bool
        self.ec_entropy  = kwargs.get('mem_temp',        df['mem_temp'].unique()) #string ?!?!?!
        self.mem_envelope= kwargs.get('mem_envelope',    df['memory_envelope'].unique())
        self.alpha       = kwargs.get('alpha',           df['alpha'].unique())
        self.beta        = kwargs.get('beta',            df['beta'].unique())
        print_deets = kwargs.get('print_deets', False)
        if print_deets:
            vars = [self.exp_type,
                    self.env_type,
                    self.dims,
                    self.rho,
                    self.action_list,
                    self.rewards,
                    self.rwd_action,
                    self.arch,
                    self.pvals,
                    self.ec_entropy,
                    self.mem_envelope,]
            keys = ['experiment_type',
                    'ENVIRONMENT',
                    'dims',
                    'rho',
                    'action_list',
                    'rewards',
                    'rwd_action',
                    'AGENT',
                    'use_pvals',
                    'mem_temp',
                    'memory_envelope']

            for var, key in zip(vars, keys):
                self.print_fields(df, var, key)

            print(f'expt_type: {self.exp_type} at {np.where(df["experiment_type"].isin(self.exp_type))}')

        self.info = self.filter_df(df)

        print(f'{self.info.shape[0]}/{df.shape[0]} entries match criteria')
        if self.info.shape[0] == 0:
            print('No data to collect')
            #raise Exception("No Data to Collect")
        self.ids = self.info['save_id']

        #self.data = self.get_data()
        #self.normalized_rwd = self.get_normalized_reward()
        #self.reward_avg, self.reward_sd = self.average('total_reward')
        
    def print_fields(self, df, variable, key):
            print(f'{key}: {variable} at {np.where(df[key].isin(variable))}')
    def filter_df(self, df):
        idx = np.where((df['experiment_type'].isin(self.exp_type))
                       & (df['ENVIRONMENT'].isin(self.env_type))
                       & (df['load_from'].isin(self.load_from))
                       & (df['dims'].isin(self.dims))
                       & (df['rho'].isin(self.rho))
                       & (df['action_list'].isin(self.action_list))
                       & (df['rewards'].isin(self.rewards))
                       & (df['rwd_action'].isin(self.rwd_action))
                       & (df['AGENT'].isin(self.arch))
                       & (df['freeze_weights'].isin(self.freeze_w))
                       & (df['use_pvals'].isin(self.pvals))
                       & (df['mem_temp'].isin(self.ec_entropy))
                       & (df['memory_envelope'].isin(self.mem_envelope))
                       & (df['alpha'].isin(self.alpha))
                       & (df['beta'].isin(self.beta))
                      )
        vals = df.loc[idx].values
        cols = df.columns
        filtered_info = pd.DataFrame(vals, columns=cols)
        return filtered_info

    def record_info(self, df, record_id):
        idx = np.where((df['save_id'].isin(record_id)))
        vals = df.loc[idx].values
        cols = df.columns
        filtered_info = pd.DataFrame(vals, columns=cols)
        return filtered_info


    def get_data(self):
        data = []
        for ii, id_ in enumerate(self.ids):
            data.append(pickle.load(open(self.parent_dir + f'results/{id_}_data.p', 'rb')))
        return data

    def get_normalized_reward(self):
        nd = []
        for ii, dat in enumerate(self.data):
            expt_type  = self.info.loc[ii]['experiment_type']
            reward_mag = float(self.info.loc[ii]['rewards'].strip('{}').split(':')[1])
            num_events = float(self.info.loc[ii]['num_events'])
            step_pen   = float(self.info.loc[ii]['step_penalization'])
            lower_bound = num_events * step_pen
            if expt_type == 0:
                upper_bound = reward_mag
            else:
                if self.info.loc[ii]['rwd_action'] == 'None':
                    upper_bound = reward_mag*num_events/2
                else:
                    upper_bound = reward_mag*num_events
            temp_ = (np.array(dat['total_reward']) - lower_bound) / (upper_bound-lower_bound)
            nd.append(temp_)

        return nd
    def get_expt_data(self, expt_type):
        d = []
        df_ = self.info
        idx = df_.index[df_['experiment_type']==expt_type].tolist()
        for i in idx:
            d.append(self.data[i]) # double check this index is correct in self.data too
        return d

    def average(self, data_type='total_reward', **kwargs):
        start_ind = kwargs.get('start_ind', 0)
        end_ind   = kwargs.get('end_ind', -1)

        data_to_average = []
        array_lengths = [len(i[data_type]) for i in self.data]
        masked_array = np.ma.empty((len(self.data),np.max(array_lengths)))
        masked_array.mask = True

        for idx, i in enumerate(self.data):
            masked_array[idx,:len(i[data_type])] = i[data_type]

        return masked_array.mean(axis=0), masked_array.std(axis=0)


class RewardFilter(object):
    def __init__(self, df, **kwargs):
        self.print = kwargs.get('print', False)
        self.parent_dir = kwargs.get('parent_dir', './data/')
        self.load_from = kwargs.get('load_from', df['load_from'].unique())
        self.exp_type = kwargs.get('expt_type', df['experiment_type'].unique())  # list of strings
        self.env_type = kwargs.get('env_type', df['ENVIRONMENT'].unique())  # string
        self.dims = kwargs.get('dims', df['dims'].unique())  # string
        self.rho = kwargs.get('rho', df['rho'].unique())  # float

        self.action_list = kwargs.get('action_list',
                                      df['action_list'].unique())  # int  [str(['Down', 'Up', 'Right', 'Left'])]) #
        self.rewards = kwargs.get('rewards', df['rewards'].unique())  # string
        self.rwd_action = kwargs.get('rewarded_action', df['rwd_action'].unique())  # string
        self.arch = kwargs.get('arch', df['AGENT'].unique())  # list of strings
        self.freeze_w = kwargs.get('freeze_w', df['freeze_weights'].unique())
        self.pvals = kwargs.get('use_pvals', df['use_pvals'].unique())  # bool
        self.ec_entropy = kwargs.get('mem_temp', df['mem_temp'].unique())  # string ?!?!?!
        self.mem_envelope = kwargs.get('mem_envelope', df['memory_envelope'].unique())
        self.alpha = kwargs.get('alpha', df['alpha'].unique())
        self.beta = kwargs.get('beta', df['beta'].unique())
        print_deets = False
        if print_deets:
            print(df['experiment_type'].unique(), self.exp_type)
            print(df['ENVIRONMENT'].unique(), self.env_type)
            print(df['dims'].unique(), self.dims)
            print(df['rho'].unique(), self.rho)
            print(df['rewards'].unique(), self.rewards)
            print(df['action_list'].unique(), self.action_list)
            print(df['rwd_action'].unique(), self.rwd_action)
            print(df['AGENT'].unique(), self.arch)
            print(df['freeze_weights'].unique(), self.freeze_w)
            print(df['use_pvals'].unique(), self.pvals)
            print(df['mem_temp'].unique(), self.ec_entropy)
            print(df['memory_envelope'].unique(), self.mem_envelope)
            print(df['alpha'].unique(), self.alpha)
            print(df['beta'].unique(), self.beta)

        self.info = self.filter_df(df)

        print(f'{self.info.shape[0]}/{df.shape[0]} entries match criteria')
        if self.info.shape[0] == 0:
            raise Exception("No Data to Collect")
        self.ids = self.info['save_id']
        self.num_samples = kwargs.get('num_samples', len(self.ids))
        if self.num_samples > len(self.ids):
            print(f'Requested {self.num_samples}: only {len(self.ids)} samples available')
            self.num_samples = len(self.ids)

        self.data = self.get_reward_data()
        # self.normalized_rwd = self.get_normalized_reward()
        # self.reward_avg, self.reward_sd = self.average('total_reward')

    def filter_df(self, df):
        idx = np.where((df['experiment_type'].isin(self.exp_type))
                       & (df['ENVIRONMENT'].isin(self.env_type))
                       & (df['load_from'].isin(self.load_from))
                       & (df['dims'].isin(self.dims))
                       & (df['rho'].isin(self.rho))
                       & (df['action_list'].isin(self.action_list))
                       & (df['rewards'].isin(self.rewards))
                       & (df['rwd_action'].isin(self.rwd_action))
                       & (df['AGENT'].isin(self.arch))
                       & (df['freeze_weights'].isin(self.freeze_w))
                       & (df['use_pvals'].isin(self.pvals))
                       & (df['mem_temp'].isin(self.ec_entropy))
                       & (df['memory_envelope'].isin(self.mem_envelope))
                       & (df['alpha'].isin(self.alpha))
                       & (df['beta'].isin(self.beta))
                       )
        vals = df.loc[idx].values
        cols = df.columns
        filtered_info = pd.DataFrame(vals, columns=cols)
        return filtered_info

    def get_reward_data(self):
        data = []
        for ii, id_ in enumerate(self.ids[0:self.num_samples]):
            print(ii)
            with open(self.parent_dir + f'results/{id_}_data.p', 'rb') as f:
                kosher_pickle = pickle.load(f)
                data.append(kosher_pickle['total_reward'])
        return data


# filter for all expts w same architecture
def arch_filter(df, arch, **kwargs):
    expts = kwargs.get('expts', np.arange(7))
    arch_filt = []
    for i in expts:
        if i == 2 or i == 5:
            filtered  = DataFilter(df, expt_type=[i], arch=[arch])
        elif i == 3 or i ==6:
            filtered = None
        else:
            filtered  = DataFilter(df, expt_type=[i], arch=[arch])

        arch_filt.append(filtered)

    return arch_filt

def smoothed_reward(data, smoothing):
    smoothed_reward = []
    for x in data:
        if x is None:
            smoothed_reward.append(None)
        else:
            smoothed_reward.append(running_mean(x.reward_avg, smoothing))

    return smoothed_reward

def ec_recall(EC, key):
    # cos sim
    key_array = np.asarray(list(EC.keys()))
    entry = np.asarray(key)
    mqt = np.dot(key_array,entry)
    norm = np.linalg.norm(key_array, axis =1) * np.linalg.norm(entry)
    cos_sim = mqt/norm
    lin_act = tuple(key_array[np.argmax(cos_sim)])
    maxcos_sim = max(cos_sim)

    # recall
    memory = np.nan_to_num(EC[lin_act][0])
    deltas = memory[:,0]
    policy = ac.softmax(maxcos_sim * deltas, T=0.05)
    return policy


def ec_policies(maze, EC, trial_timestamp,**kwargs):
    envelope = kwargs.get('decay', 50)
    mem_temp = kwargs.get('mem_temp', 1)
    mpol_array = np.zeros(maze.grid.shape, dtype=[(x, 'f8') for x in maze.action_list])


    # cycle through readable states
    for key in EC.keys():
        row, col = EC[key][2]
        pol = ec_recall(EC, key)
        mpol_array[row,col] = tuple(pol)

    return mpol_array

def plot_from_id(run_id, mf=False):
    if mf:
        agent_dir = f'../data/outputs/gridworld/weights/{run_id}.pt'
        agent = torch.load(agent_dir)
    mem_dir = f'../data/outputs/gridworld/episodic_cache/{run_id}_EC.p'
    mem = pickle.load(open(mem_dir, 'rb'))

    ## to do: build grid from csv specs similar to rwd_loc

    rwd_loc = df.loc[df['Run_ID'] == run_id, 'Rewards'].item()
    r_x = int(rwd_loc[rwd_loc.index('(')+1 : rwd_loc.index(',')])
    r_y = int(rwd_loc[rwd_loc.index(',')+2 : rwd_loc.index(')')])

    reward_location = (r_x,r_y)

    openfield = gw.GridWorld(rows=20, cols=20, env_type=None, rewards={reward_location:1}, step_penalization=-0.01,
                            actionlist=['Down','Up','Right','Left'], rewarded_action=None)

    if mf:
        obs = openfield.get_sample_obs()
        mf_pol, mf_val = expt.get_snapshot(obs, openfield, agent)
        gp.plot_pref_pol(openfield,mf_pol, threshold=0.1)


    abcd = ec_policies(openfield,mem,trial_timestamp = 0, mem_temp = 0.5)
    gp.plot_pref_pol(openfield, abcd, threshold = 0.1, title='ec_pol_new_rwd',upperbound=2)


def arbitration(x, alpha, beta):
    #alpha = bump size
    # beta = number of steps to decay to 0.01
    threshold = 0.01
    decay = np.power(threshold,1/beta)
    a = np.empty_like(x, dtype=float)

    for ind, i in enumerate(x):
        if ind == 0:
            a[ind] = i*alpha
        else:
            calc = decay*a[ind-1] + i*alpha
            if calc > 1:
                calc = 1
            a[ind] = calc
    return a
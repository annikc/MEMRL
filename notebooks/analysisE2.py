import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import torch
import sys
sys.path.insert(0,'../rl_network/'); import ac
sys.path.insert(0,'../memory/'); import episodic as ec
sys.path.insert(0,'../environments/'); import gw; import gridworld_plotting as gp

import experiment as expt

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_traces(datas, smoothing=1, **kwargs):
    plt.figure()
    sds = kwargs.get('sd', None)
    labels = kwargs.get('labels', [None]*len(datas))
    for i in range(len(datas)):
        data = datas[i]
        smoothed_data = running_mean(data,smoothing)
        plt.plot(smoothed_data, label=labels[i])

        if sds is not None:
            sd = sds[i]
            error = running_mean(sd,smoothing)
            plt.fill_between(np.arange(len(smoothed_data)), smoothed_data-error, smoothed_data+error, alpha=0.25)
    if labels[0] is not None:
        plt.legend(bbox_to_anchor=(1.01,1), loc='upper left', borderaxespad=0.0)
    plt.ylim([-3, 1.5])
    plt.show()
    plt.close()

class data_filter(object):
    def __init__(self, df, **kwargs):
        self.exp_type    = kwargs.get('expt_type',       df['experiment_type'].unique()) #list of strings
        self.env_type    = kwargs.get('env_type',        df['ENVIRONMENT'].unique()) #string
        self.dims        = kwargs.get('dims',            df['dims'].unique()) #string
        self.rho         = kwargs.get('rho',             df['rho'].unique()) # float

        self.action_list = kwargs.get('action_list',     df['action_list'].unique()) # int  [str(['Down', 'Up', 'Right', 'Left'])]) #
        self.rwd_action  = kwargs.get('rewarded_action', df['rwd_action'].unique()) # df['rwd_action'].unique()) # string

        self.arch        = kwargs.get('arch',            df['AGENT'].unique()) # list of strings
        self.freeze_w    = kwargs.get('freeze_w',        df['freeze_weights'].unique())
        self.pvals       = kwargs.get('use_pvals',       df['use_pvals'].unique()) # bool
        self.ec_entropy  = kwargs.get('mem_temp',        df['mem_temp'].unique()) #string ?!?!?!
        self.mem_envelope= kwargs.get('mem_envelope',    df['memory_envelope'].unique())
        self.alpha       = kwargs.get('alpha',           df['alpha'].unique())
        self.beta        = kwargs.get('beta',            df['beta'].unique())

        print(df['experiment_type'].unique(), self.exp_type)
        print(df['ENVIRONMENT'].unique(), self.env_type)
        print(df['dims'].unique(), self.dims)
        print(df['rho'].unique(), self.rho)
        print(df['action_list'].unique(), self.action_list)
        print(df['rwd_action'].unique(), self.rwd_action)
        print(df['AGENT'].unique(), self.arch)
        print(df['freeze_weights'].unique(), self.freeze_w)
        print(df['use_pvals'].unique(), self.pvals)
        print(df['mem_temp'].unique(), self.ec_entropy)
        print(df['memory_envelope'].unique(), self.mem_envelope)
        print(df['alpha'].unique(), type(self.alpha))
        print(df['beta'].unique(), self.beta)

        self.info = self.filter_df(df)
        print(f'{self.info.shape[0]}/{df.shape[0]} entries match criteria')
        if self.info.shape[0] == 0:
            raise Exception("No Data to Collect")
        self.ids = self.info['save_id']
        self.data = self.get_data()

        self.reward_avg, self.reward_sd = self.average('total_reward')

    def filter_df(self, df):
        idx = np.where((df['experiment_type'].isin(self.exp_type))
                       & (df['ENVIRONMENT'].isin(self.env_type))
                       & (df['dims'].isin(self.dims))
                       & (df['rho'].isin(self.rho))
                       & (df['action_list'].isin(self.action_list))
                       & (df['rwd_action'].isin(self.rwd_action))
                       & (df['AGENT'].isin(self.arch))
                       & (df['freeze_weights'].isin(self.freeze_w))
                       & (df['use_pvals'].isin(self.pvals))
                       & (df['mem_temp'].isin(self.ec_entropy))
                       & (df['memory_envelope'].isin(self.mem_envelope))
                       & (df['alpha'].isin(self.alpha))
                       & (df['beta'].isin(self.beta))
                      )
        filtered_info = df.loc[idx]

        return filtered_info

    def get_data(self):
        data = []
        for ii, id_ in enumerate(self.ids):
            data.append(pickle.load(open(f'../data/outputs/gridworld/E2/results/{id_}_data.p', 'rb')))
        return data

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


# filter for all expts w same architecture
def arch_filter(df, arch, **kwargs):
    expts = kwargs.get('expts', np.arange(7))
    arch_filt = []
    for i in expts:
        if i == 2 or i == 5:
            filtered  = data_filter(df, expt_type=[i], arch=[arch])
        elif i == 3 or i ==6:
            filtered = None
        else:
            filtered  = data_filter(df, expt_type=[i], arch=[arch])

        arch_filt.append(filtered)

    return arch_filt

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from basic.modules.Utils import running_mean as rm
from basic.modules.Utils.gridworld_plotting import plot_polmap, plot_pref_pol

df = pd.read_csv('../../Data/test_environments.csv')


r_data = [[], []]
env_types = {'open':1, 'obstacles':2, '4rooms':3, 'bar':4}
env = 'bar'
controllers = ['MF', 'EC']


for i, cont in enumerate(controllers):
    idx = np.where( (df['environment']==f'gym_grid:gridworld-v{env_types[env]}')
                  & (df['controller']==cont)
                  & (df['n_trials']==5000))


    for value in df.loc[idx]['id']:
        with open(f'../../Data/results/{value}_data.p', 'rb') as f:
            d = pickle.load(f)
            r = d['total_reward']
            print(value, len(r))
            r_data[i].append(d['total_reward'])

smoothing = 30

mf_r_avg = rm(np.mean(np.vstack(r_data[0]), axis=0),smoothing)
mf_r_std = rm(np.std(np.vstack(r_data[0]), axis=0),smoothing)

ec_r_avg = rm(np.mean(np.vstack(r_data[1]), axis=0),smoothing)
ec_r_std = rm(np.std(np.vstack(r_data[1]), axis=0),smoothing)

plt.plot(mf_r_avg)
plt.fill_between(np.arange(len(mf_r_std)), mf_r_avg-mf_r_std, mf_r_avg+mf_r_std, alpha=0.5)

plt.plot(ec_r_avg)
plt.fill_between(np.arange(len(ec_r_std)), ec_r_avg-ec_r_std, ec_r_avg+ec_r_std, alpha=0.5)

plt.show()

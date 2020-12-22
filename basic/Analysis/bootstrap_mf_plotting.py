import pickle
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from basic.modules.Utils import running_mean as rm
from basic.modules.Utils.gridworld_plotting import plot_polmap, plot_pref_pol

id_data = pd.read_csv('../Data/test_bootstrap.csv')
print(id_data['run_id'])
bootstrap_data = []
modelfree_data =[]

for i in range(len(id_data['run_id'])):
    run_id = id_data.iloc[[i]]['run_id'].item()
    type = id_data.iloc[[i]]['type'].item()
    print(type)
    if type == 'bootstrap':
        with open(f'../Data/results/{run_id}_data.p', 'rb') as f:
            bootstrap_data.append(pickle.load(f))

    elif type == 'mf_only':
        with open(f'../Data/results/{run_id}_data.p', 'rb') as f:
            modelfree_data.append(pickle.load(f))

pre_e_reward = []
pre_b_reward = []
for x in range(len(bootstrap_data)):
    print(x)
    pre_e_reward.append(bootstrap_data[x]['total_reward'])
    pre_b_reward.append(bootstrap_data[x]['bootstrap_reward'])

pre_m_reward = []
for x in range(len(modelfree_data)):
    pre_m_reward.append(modelfree_data[x]['total_reward'])


e_reward = np.mean(np.vstack(pre_e_reward), axis=0)
e_std    = np.std(np.vstack(pre_e_reward), axis=0)

b_reward = np.mean(np.vstack(pre_b_reward), axis=0)
b_std    = np.std(np.vstack(pre_b_reward), axis=0)

m_reward = np.mean(np.vstack(pre_m_reward), axis=0)
m_std    = np.std(np.vstack(pre_m_reward), axis=0)

smoothing = 100
smooth_e = rm(e_reward, smoothing)
smooth_e_std = rm(e_std, smoothing)

smooth_m = rm(m_reward, smoothing)
smooth_m_std = rm(m_std, smoothing)

smooth_b = rm(b_reward, smoothing)
smooth_b_std = rm(b_std, smoothing)


plt.figure(0 , figsize=(5,5))
plt.plot(smooth_m, label='model free', color='gray')
plt.fill_between(np.arange(len(smooth_m)), smooth_m-smooth_m_std,smooth_m+smooth_m_std, alpha = 0.5, color = 'gray')

plt.plot(smooth_e, label='episodic')
plt.fill_between(np.arange(len(smooth_e)), smooth_e-smooth_e_std,smooth_e+smooth_e_std, alpha = 0.5)

plt.plot(smooth_b, label='episodic learning MF')
plt.fill_between(np.arange(len(smooth_b)), smooth_b-smooth_b_std,smooth_b+smooth_b_std, alpha = 0.5)

plt.legend(loc='lower right')
plt.savefig('bootstrapping.svg',format='svg')
plt.show()


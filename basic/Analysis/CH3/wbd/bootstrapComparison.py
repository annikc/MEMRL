import pickle
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from basic.modules.Utils import running_mean as rm
from basic.modules.Utils.gridworld_plotting import plot_polmap, plot_pref_pol

id_data = pd.read_csv('../Data/scinetJan10_2021.csv')
print(id_data['run_id'])
novel_R = {'MF':[], 'EC':[], 'bootstrap':[]}
moved_R = {'MF':[], 'EC':[], 'bootstrap':[]}

for i in range(len(id_data['run_id'])):
    run_id = id_data.iloc[[i]]['run_id'].item()
    env_type = id_data.iloc[[i]]['env_type'].item()
    expt_type = id_data.iloc[[i]]['expt_type'].item()

    with open(f'../Data/results/{run_id}_data.p', 'rb') as f:
        data = pickle.load(f)

    if env_type == 'gym_grid:gridworld-v1': #reward at 5,5
        novel_R[expt_type].append(data)
    elif env_type == 'gym_grid:gridworld-v11': # reward at 10,10, MF trained on 5,5
        moved_R[expt_type].append(data)
    else:
        raise Exception('Env Type Error')



for x in novel_R['MF']:
    dat = rm(x['total_reward'], 50)
    plt.plot(dat, alpha = 0.5)

plt.show()

'''
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
'''

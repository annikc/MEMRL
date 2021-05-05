import numpy as np
import matplotlib.pyplot as plt
from modules.Utils import running_mean as rm
import pickle

import os

print(os.getcwd())

data_dir = '../Data/'

ids = {'oh_train':'d80ea92c-422c-436a-b0ff-84673d43a30d',
       'oh_test' :'513a8f80-97f0-454e-b4c7-30aa731e5d0b',
       'oh_test_ec': 'd860ec5b-6bee-416e-925c-0ee8f1939e69',
       #'oh_test_ec2': '461803f6-bac1-4b24-a272-f5cdbcfc54a0',
       'pc_train':'b6f51c73-ebc0-467a-b5e5-5b51a5a3208d',
       'pc_test' :'c5c55bf5-b827-4a96-ba29-f345f4257429',
       'pc_test_ec':'dc126211-0af0-4fc1-8788-3f1b8567cdc2',
       #'pc_test_ec2':'7156b8bd-68f2-4b11-b446-8498aae04133'
       }


total_reward = {}
for key, value in ids.items():
    if value == '':
        pass
    else:
        load_id = value
        with open(data_dir+ f'results/{load_id}_data.p', 'rb') as f:
            total_reward[key] = pickle.load(f)['total_reward']

smoothing = 100

plt.figure()
plt.plot(rm(total_reward['oh_test'],smoothing),':',c='C0',label='model free control (onehot)')
plt.plot(rm(total_reward['oh_test_ec'],smoothing), c='C0',label='episodic control (onehot)')
plt.plot(rm(total_reward['pc_test'],smoothing),':',c='C1',label='model free control (place cell)')
plt.plot(rm(total_reward['pc_test_ec'],smoothing), c='C1',label='episodic control (place cell)')

plt.xlim([0,1900])
plt.legend(loc='upper center', bbox_to_anchor =(0.5, 1.1), ncol=2)
plt.show()








fig, ax = plt.subplots(2,1)
for str in ids.keys():
    if str[3:] == 'train':
        f = 0
    elif str[3:] == 'test':
        f = 1
    if str[0:2] == 'oh':
        label = 'onehot'
    else:
        label = 'place cell'
    ax[f].plot(rm(total_reward[str], smoothing), label=label)
ax[0].set_xlim([0,1000])
ax[1].set_xlim([0,5000])
ax[1].set_xlabel('Trial')
plt.ylabel('Average Reward / 100 Trials')
ax[0].set_ylim([-2.6,10.1])
ax[1].set_ylim([-2.6,10.1])
ax[0].legend(loc =4)
plt.show()
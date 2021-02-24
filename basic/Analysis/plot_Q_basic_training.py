import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import gym



env_name   = 'gym_grid:gridworld-v1'

# create environment
env = gym.make(env_name)
plt.close()



def pref_Q_action(qtable):
    action_table = np.zeros(env.shape)
    for state in range(env.nstates):
        state2d = env.oneD2twoD(state)
        action_table[state2d] = np.argmax(qtable[state,:])

    return action_table

#filename = '../../../Data/Qlearning.csv'
#df = pd.read_csv(filename)

run_id = 'b9cfe145-d857-40e0-af22-d6f7b496c9ac'

with open(f'../Data/results/{run_id}_data.p', 'rb') as f:
    print('hello')
    data = pickle.load(f)

total_reward = data['total_reward']
Q_tables = data['P_snap']

print_freq = 1000
agg_data = {'ep':[], 'avg':[], 'min':[], 'max':[]}
for j in range(0, len(total_reward),print_freq):
    average_reward = sum(total_reward[j:j+print_freq])/len(total_reward[j:j+print_freq])
    min_reward = min(total_reward[j:j+print_freq])
    max_reward = max(total_reward[j:j+print_freq])

    agg_data['ep'].append(j)
    agg_data['avg'].append(average_reward)
    agg_data['min'].append(min_reward)
    agg_data['max'].append(max_reward)

save_fig_dir = 'figures/CH1/Qlearning/'
format = 'svg'
save = True

plt.figure()
plt.plot(agg_data['ep'],agg_data['avg'], label='avg')
plt.plot(agg_data['ep'],agg_data['min'], label='min')
plt.plot(agg_data['ep'],agg_data['max'], label='max')
plt.ylim([0,10.1])
plt.legend(loc=0)
if save:
    plt.savefig(f'{save_fig_dir}total_reward.{format}',format=f'{format}')
plt.show()

for index in [0,1000,10000,50000,100000-1]:
    plt.figure()
    pref_action = pref_Q_action(qtable=Q_tables[index])
    cmap = cm.get_cmap('viridis', 4)    # 11 discrete colors
    a = plt.imshow(pref_action, interpolation='none',cmap=cmap)
    cbar = colorbar(a,ticks=[0,1,2,3])
    cbar.ax.set_yticklabels(['Down','Up','Right','Left'])
    plt.title(f'Preferred Action After {index} Trials')

    plt.savefig(f'{save_fig_dir}Q_actions{index}.{format}', format=f'{format}')
    plt.close()

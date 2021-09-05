import numpy as np
import pandas as pd
import gym
from Analysis.analysis_utils import structured_unstructured, LINCLAB_COLS, plot_specs, analysis_specs
from Analysis.analysis_utils import get_grids
from modules.Utils.gridworld_plotting import plot_polmap, plot_valmap, plot_pref_pol
from modules.Utils import running_mean as rm
import scipy.special as sc
from optimal_p_v_maps import attempt_opt_pol
import pickle
import matplotlib.pyplot as plt
cache_limits = analysis_specs['cache_limits']

# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'shallowAC_withPVmaps.csv')

df['representation'] = df['representation'].apply(structured_unstructured)
groups_to_split = ['env_name','representation']
df_gb = df.groupby(groups_to_split)["save_id"]

envs_to_plot = ['gridworld:gridworld-v1','gridworld:gridworld-v4','gridworld:gridworld-v3','gridworld:gridworld-v5']
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['unstructured','structured']
grids = get_grids(envs_to_plot)

env = envs_to_plot[1]
env_obj = gym.make(env)
plt.close()
pct = 100
rep = 'unstructured'

id_list = list(df_gb.get_group((env,rep)))
print(env, rep, len(id_list))
for i, id_num in enumerate(id_list):
    with open(parent_path+ f'results/{id_num}_data.p', 'rb') as f:
        dats = pickle.load(f)
        p_maps = dats['P_snap']
        v_maps = dats['V_snap']
        print(len(p_maps), len(dats['total_reward']))

#plt.figure()
#plt.plot(rm(dats['total_reward'],200))
opt_pol = attempt_opt_pol(env_obj)
KLD_ = np.zeros((20,20))
test = [x for x in p_maps[49][0,0]]
def my_own_kld(p,q):
    k = 0
    for x in range(len(p)):
        if q[x] == 0:
            q_val = 1e-18
        else:
            q_val = q[x]
        k += p[x] * np.log(p[x]/q_val)
    return k
print(test, opt_pol[0,0], my_own_kld(test,opt_pol[0,0]))
for r in range(20):
    for c in range(20):
        t = [x for x in p_maps[49][r,c]]
        KLD_[r,c] = my_own_kld(t,opt_pol[r,c])

plot_pref_pol(env_obj, p_maps[49])
plot_pref_pol(env_obj,opt_pol)

a = plt.imshow(KLD_, vmin=0,vmax=20)
plt.colorbar(a)
plt.show()


#plot_valmap(env_obj, v_maps[49],v_range=[0,10])
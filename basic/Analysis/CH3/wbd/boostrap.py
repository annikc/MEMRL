import pandas as pd
from Analysis.analysis_utils import structured_unstructured, LINCLAB_COLS, plot_specs, analysis_specs
from Analysis.analysis_utils import get_grids
from modules.Utils import running_mean as rm
from Analysis.analysis_utils import avg_performance_over_envs, avg_perf_over_envs_lines, avg_performance_over_envs_violins
import pickle
import matplotlib.pyplot as plt

# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'bootstrapped_perceptron_mf.csv')
df['representation'] = df['representation'].apply(structured_unstructured)
groups_to_split = ['env_name','representation','EC_cache_limit','num_trials']
gb = df.groupby(groups_to_split)["save_id"]

base_ = pd.read_csv(parent_path+'train_perceptron_AC.csv')
base_['representation'] = base_['representation'].apply(structured_unstructured)
groups_to_split = ['env_name','representation']
gb_base = base_.groupby(groups_to_split)["save_id"]

envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['unstructured','structured']
grids = get_grids(envs_to_plot)

cache_limits = analysis_specs['cache_limits']

env = envs_to_plot[3]
pct = 100
rep = 'unstructured'

id_list = gb.get_group((env,rep,int(cache_limits[env][100]*(pct/100)),1000))
print(id_list)

data_dir='../../Data/results/'
ec_results = []
mf_results = []
colors = [LINCLAB_COLS['red'], LINCLAB_COLS['blue'], LINCLAB_COLS['green'],LINCLAB_COLS['purple'], LINCLAB_COLS['orange'], LINCLAB_COLS['grey']]
fig, ax = plt.subplots(2,4,sharey=True)
for r, rep in enumerate(reps_to_plot):
    for p, pct in enumerate(pcts_to_plot):
        print(rep, pct)
        try:
            id_list = gb.get_group((env,rep,int(cache_limits[env][100]*(pct/100)),5000))
            for i, id_num in enumerate(id_list):
                if i ==0:
                    with open(data_dir+ f'{id_num}_data.p', 'rb') as f:
                        dats = pickle.load(f)
                        ax[r,p].plot(rm(dats['total_reward'],200), linestyle=':', color=colors[i])
                        ax[r,p].plot(rm(dats['bootstrap_reward'],200), color='k')
            baseline_id = list(gb_base.get_group((env,rep)))[0]
            with open(data_dir+f'{baseline_id}_data.p','rb') as f:
                dats = pickle.load(f)
                filt = rm(dats['total_reward'][0:5000],200)
                ax[r,p].plot(filt, color='cyan')
        except:
            print('no data')
ax[0,0].set_ylim([-2.5,10])
plt.show()
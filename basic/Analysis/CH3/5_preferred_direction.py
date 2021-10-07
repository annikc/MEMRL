## import some things
import numpy as np
import pandas as pd
from Analysis.analysis_utils import structured_unstructured, LINCLAB_COLS, plot_specs, analysis_specs
from Analysis.analysis_utils import get_grids
from modules.Utils import running_mean as rm
from Analysis.analysis_utils import avg_performance_over_envs, avg_perf_over_envs_lines, avg_performance_over_envs_violins
import pickle
import matplotlib.pyplot as plt

# import csv data summary
parent_path = '../../Data/'

# get baseline (MF retrain alone)
base_ = pd.read_csv(parent_path+'train_test_shallowAC.csv')
base_['representation'] = base_['representation'].apply(structured_unstructured)

# get EC with bootstrapped MF
df = pd.read_csv(parent_path+'bootstrapped_retrain_shallow_AC.csv')
df['representation'] = df['representation'].apply(structured_unstructured)

groups_to_split = ['env_name','representation']

gb_base = base_.groupby(groups_to_split+['num_trials'])["save_id"]
gb = df.groupby(groups_to_split+['EC_cache_limit','num_trials'])["save_id"]

colors = {100:LINCLAB_COLS['red'], 75: LINCLAB_COLS['orange'], 50:LINCLAB_COLS['green'], 25:LINCLAB_COLS['purple']}

def occupancy_plot(id_list):
    fig, ax = plt.subplots(1,2)
    max = 0.025
    EC = []
    MF = []
    for id_num in id_list:
        with open(parent_path+f'results/{id_num}_data.p','rb') as f:
            dats = pickle.load(f)
        EC_map = dats['EC_occupancy']
        EC_visits = np.nansum(EC_map)
        EC.append(EC_map/EC_visits)

        MF_map = dats['MF_occupancy']
        MF_visits = np.nansum(MF_map)
        MF.append(MF_map/MF_visits)

    EC_occ = np.nanmean(EC,axis=0).reshape(20,20)
    print(np.nanmax(EC_occ))
    MF_occ = np.nanmean(MF,axis=0).reshape(20,20)
    a= ax[0].imshow(EC_occ, vmin=0,vmax=np.nanmax(EC_occ))
    ax[0].set_title('EC')
    plt.colorbar(a, ax = ax[0])

    b = ax[1].imshow(MF_occ,vmin=0, vmax=np.nanmax(EC_occ))
    ax[1].set_title('MF')
    plt.colorbar(b, ax = ax[1])
    plt.show()


####
envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
pcts_to_plot = [100,75,50,25]
grids = get_grids(envs_to_plot)
cache_limits = analysis_specs['cache_limits']
#plot_single_retraining(envs_to_plot[1],[25,50,75,100],'structured',index=1)
#plot_all_retraining(envs_to_plot[1],[25,50,75,100],['structured','unstructured'])

env = envs_to_plot[0]
rep = 'structured'
pct = 50
ec_group = df.groupby(groups_to_split+['EC_cache_limit','extra_info'])["save_id"]
id_list = list(ec_group.get_group((env,rep,int(cache_limits[env][100]*(pct/100)),'occ_map')))
print(len(id_list))
id_num = id_list[0]
with open(parent_path+f'results/{id_num}_data.p','rb') as f:
    dats = pickle.load(f)

for i in dats.keys():
    print(i, len(dats[i]))
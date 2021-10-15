## import some things
import numpy as np
import pandas as pd
from Analysis.analysis_utils import structured_unstructured, LINCLAB_COLS, plot_specs, analysis_specs
from Analysis.analysis_utils import get_grids
from modules.Utils import running_mean as rm
import pickle
import matplotlib.pyplot as plt
import gym

# import csv data summary
parent_path = '../../Data/'

# get baseline (MF retrain alone)
base_ = pd.read_csv(parent_path+'train_test_shallowAC.csv')
base_['representation'] = base_['representation'].apply(structured_unstructured)

# get EC with bootstrapped MF
df = pd.read_csv(parent_path+'bootstrapped_retrain_shallow_AC.csv')
df['representation'] = df['representation'].apply(structured_unstructured)

groups_to_split = ['env_name','representation']

gb_base = base_.groupby(groups_to_split)["save_id"]
gb = df.groupby(groups_to_split)["save_id"]

colors = {100:LINCLAB_COLS['red'], 75: LINCLAB_COLS['orange'], 50:LINCLAB_COLS['green'], 25:LINCLAB_COLS['purple']}

env = 'gridworld:gridworld-v51'
e = gym.make(env)
plt.close()
rep = 'structured'

id_list = {'gridworld:gridworld-v11':'a886a36b-77af-4845-b950-71e64506190c',
           'gridworld:gridworld-v31':'22d9a5cc-13e4-4fe3-9e92-9f25c5ba9b18',
           'gridworld:gridworld-v41':'5f0f1b3f-db3e-4a19-9d19-cf817e9aeee3',
           'gridworld:gridworld-v51':'ac9a6807-9ecb-405c-b31e-3ff4ccaa2bfd'}

id_num =id_list[env]
with open(parent_path+f'results/{id_num}_data.p','rb') as f:
    dats = pickle.load(f)
    print(id_num, [(x, len(dats[x])) for x in dats.keys()])

start_ind = 0
end_ind = -1
MF_map = np.nansum(np.asarray(dats['MF_occupancy'][start_ind:end_ind]),axis=0)

MF_visits = np.nansum(np.asarray(MF_map))
MF=MF_map/MF_visits

MF_occ = np.log(MF.reshape(20,20)/(1/len(e.useable)))
fig,ax = plt.subplots(2,1)
ax[0].plot(rm(dats['total_reward'][start_ind:end_ind],200))
ax[1].imshow(MF_occ,cmap='RdBu_r', vmin=-4,vmax=4)
plt.savefig(f'../figures/CH3/mf_only_{env[-2:]}_state_occ.svg')
plt.show()
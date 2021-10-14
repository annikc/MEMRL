## import some things
import numpy as np
import pandas as pd
from Analysis.analysis_utils import structured_unstructured, LINCLAB_COLS, plot_specs, analysis_specs
from Analysis.analysis_utils import get_grids
from modules.Utils import running_mean as rm
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

gb_base = base_.groupby(groups_to_split)["save_id"]
gb = df.groupby(groups_to_split)["save_id"]

colors = {100:LINCLAB_COLS['red'], 75: LINCLAB_COLS['orange'], 50:LINCLAB_COLS['green'], 25:LINCLAB_COLS['purple']}

env = 'gridworld:gridworld-v41'
rep = 'structured'
id_list = gb.get_group((env,rep))
id_list = ['252cb42a-aad2-4fcc-bfa2-46d430835bf3']
for id_num in id_list:
    with open(parent_path+f'results/{id_num}_data.p','rb') as f:
        dats = pickle.load(f)
        print(id_num, [(x, len(dats[x])) for x in dats.keys()])
        print(dats['MF_occupancy'])

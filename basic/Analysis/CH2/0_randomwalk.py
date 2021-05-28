import pandas as pd
import pickle
import numpy as np
from Analysis.analysis_utils import analysis_specs

# import csv data summary
parent_path = '../../Data/'
ref_df = pd.read_csv(parent_path+'random_walk.csv')
gb_ref = ref_df.groupby(['env_name'])['save_id']

envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']

for env in envs_to_plot:
    normalization_factor = analysis_specs['avg_max_rwd'][env]
    results = []
    list_of_ids = (list(gb_ref.get_group(env)))
    for id_num in list_of_ids:
        with open(parent_path+ f'results/{id_num}_data.p', 'rb') as f:
            dats = pickle.load(f)
            reward_info = dats['total_reward']
            results.append(reward_info)

    raw_results = np.vstack(results)

    scaled_results = (raw_results+2.5)/(normalization_factor+2.5)

    sample_avgs = np.mean(scaled_results,axis=1)

    avg_ = np.mean(sample_avgs)
    std_ = np.std(sample_avgs)
    print(f'for env: {env}, avg:{avg_}/std:{std_}')




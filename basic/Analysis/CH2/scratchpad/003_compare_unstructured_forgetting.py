import pandas as pd
import sys
sys.path.append('../../modules/')
from Analysis.analysis_utils import get_grids, compare_avg_performance_lineplot

# import csv data summary
parent_path = '../../Data/'
oldest_df = pd.read_csv(parent_path+'throttled_ec_allreps_chebyshev.csv')
random_df = pd.read_csv(parent_path+'random_forget_ec.csv')

groups_to_split = ['env_name','representation','EC_cache_limit']
gbs = [oldest_df.groupby(groups_to_split)["save_id"], random_df.groupby(groups_to_split)["save_id"]]


envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['random', 'onehot','conv_latents','place_cell','analytic successor'] # df.representation.unique()
grids        = get_grids(envs_to_plot)

#avg_performance_over_envs(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,legend='pcts',savename='random_forgetting',save=True)
for i, env in enumerate(envs_to_plot):
    compare_avg_performance_lineplot(gbs[0], gbs[1],env,reps_to_plot,pcts_to_plot,grids[i],save=True,savename=f'compare_rand_forgetting_v{env[-2:]}',plot_title='',legend=False)
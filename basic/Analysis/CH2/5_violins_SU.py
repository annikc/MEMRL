import pandas as pd
from Analysis.analysis_utils import structured_unstructured, LINCLAB_COLS
from Analysis.analysis_utils import get_grids, avg_performance_over_envs, each_performance_over_envs_violins

# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'throttled_ec_allreps_chebyshev.csv')

df['representation'] = df['representation'].apply(structured_unstructured)
groups_to_split = ['env_name','representation','EC_cache_limit']
gb = df.groupby(groups_to_split)["save_id"]

color_map = {'structured':LINCLAB_COLS['red'], 'unstructured':LINCLAB_COLS['blue']}
labels    = {'structured':'structured','unstructured':'unstructured'}
envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
reps_to_plot = ['unstructured','structured']
pcts_to_plot = [100,75,50,25]
grids = get_grids(envs_to_plot)
each_performance_over_envs_violins(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,colors=color_map,labels=labels,legend='pcts',savename='restricted_SU_violin',save=True)
#avg_perf_over_envs_lines(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,colors=color_map,labels=labels,legend='reps',savename='restricted_SU_lines',compare_chance=True,save=True)
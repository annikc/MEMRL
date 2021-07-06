import pandas as pd
from Analysis.analysis_utils import structured_unstructured, LINCLAB_COLS, plot_specs
from Analysis.analysis_utils import get_grids
from Analysis.analysis_utils import avg_performance_over_envs, avg_perf_over_envs_lines, avg_performance_over_envs_violins

# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'throttled_ec_allreps_chebyshev.csv')

split_type = 'SU'
if split_type == 'SU':
    df['representation'] = df['representation'].apply(structured_unstructured)

groups_to_split = ['env_name','representation','EC_cache_limit']
gb = df.groupby(groups_to_split)["save_id"]

if split_type == 'SU':
    color_map = {'structured':LINCLAB_COLS['red'], 'unstructured':LINCLAB_COLS['blue']}
    labels    = {'structured':'structured','unstructured':'unstructured'}
    reps_to_plot = ['structured','unstructured']

elif split_type == 'allreps':
    color_map = plot_specs['rep_colors']
    labels    = plot_specs['labels']
    reps_to_plot = ['random','onehot','conv_latents','place_cell','analytic successor']

envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
pcts_to_plot = [90,80,70,60,50,40,30,20,10]
grids = get_grids(envs_to_plot)
#avg_performance_over_envs(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,colors=color_map,labels=labels,legend='pcts',savename=f'restricted_{split_type}_bars',save=True)
avg_perf_over_envs_lines(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,colors=color_map,labels=labels,legend='reps',save=True,savename=f'restricted_{split_type}_lines',compare_chance=True)
#avg_performance_over_envs_violins(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,colors=color_map,labels=labels,legend='reps',savename=f'restricted_{split_type}_violins_inset',compare_chance=False,save=True)





















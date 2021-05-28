import numpy as np
import pandas as pd
import sys
sys.path.append('../../modules/')

from Analysis.analysis_utils import get_grids, analysis_specs, plot_specs, avg_performance_over_envs
from Analysis.analysis_utils import no_rm_avg_std
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# import csv data summary
parent_path = '../../Data/'
#df = pd.read_csv(parent_path+'random_forget_ec.csv')
df = pd.read_csv(parent_path+'throttled_ec_allreps_chebyshev.csv')

def structured_unstructured(df_element):
    map = {'analytic successor':'structured',
           'place_cell':'structured',
           'onehot':'unstructured',
           'random':'unstructured',
           'conv_latents':'structured'}
    new_element = map[df_element]
    return new_element

df['representation'] = df['representation'].apply(structured_unstructured)
groups_to_split = ['env_name','representation','EC_cache_limit']
gb = df.groupby(groups_to_split)["save_id"]

LINCLAB_COLS = {"blue"  : "#50a2d5", # Linclab blue
                "red"   : "#eb3920", # Linclab red
                "grey"  : "#969696", # Linclab grey
                "green" : "#76bb4b", # Linclab green
                "purple": "#9370db",
                "orange": "#ff8c00",
                "pink"  : "#bb4b76",
                "yellow": "#e0b424",
                "brown" : "#b04900",
                }

color_map = {'structured':LINCLAB_COLS['red'], 'unstructured':LINCLAB_COLS['blue']}
labels    = {'structured':'structured','unstructured':'unstructured'}
envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
reps_to_plot = ['unstructured','structured']
pcts_to_plot = [100,75,50,25]
grids = get_grids(envs_to_plot)
avg_performance_over_envs(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,colors=color_map,labels=labels, savename='str_unstr_relative_to_unrestricted',save=True)
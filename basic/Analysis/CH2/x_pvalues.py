import numpy as np
import scipy.stats as sp
import pickle
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from modules.Utils import running_mean as rm

import pandas as pd
from Analysis.analysis_utils import structured_unstructured, analysis_specs,welchs_pval
from Analysis.analysis_utils import get_grids, avg_performance_over_envs, avg_perf_over_envs_lines

avg_max_rwd = analysis_specs['avg_max_rwd']
cache_limits = analysis_specs['cache_limits']

def data_sample(list_of_ids, normalization_factor=10, cutoff=5000):
    # default normalization factor - max possible reward
    data_dir='../../Data/results/'
    results = []
    for id_num in list_of_ids:
        with open(data_dir+ f'{id_num}_data.p', 'rb') as f:
            dats = pickle.load(f)
            reward_info = dats['total_reward'][0:cutoff]
            results.append(reward_info)

    raw_results = np.vstack(results)
    scaled_results = (raw_results+2.5)/(normalization_factor+2.5)
    sample_avgs = np.mean(scaled_results,axis=1)

    return sample_avgs


# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'throttled_ec_allreps_chebyshev.csv')

df['representation'] = df['representation'].apply(structured_unstructured)
groups_to_split = ['env_name','representation','EC_cache_limit']
gb = df.groupby(groups_to_split)["save_id"]


envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
reps_to_plot = ['unstructured','structured']
pcts_to_plot = [75,50,25]

e = envs_to_plot[1]
prob = data_sample(list(gb.get_group((e,'structured',cache_limits[e][75]))),normalization_factor=avg_max_rwd[e])
quer = data_sample(list(gb.get_group((e,'structured',cache_limits[e][50]))),normalization_factor=avg_max_rwd[e])

print(sp.ttest_ind(prob,quer).pvalue)

t,p = welchs_pval(prob,quer)
print(p)
for e, env in enumerate(envs_to_plot):
    normalization_factor = avg_max_rwd[env]
    for r, rep in enumerate(reps_to_plot):
            ref_sample = data_sample(list(gb.get_group((env, rep, cache_limits[env][100]))),normalization_factor=normalization_factor,cutoff=5000 )
            mean_ref = np.mean(ref_sample)
            sd_ref   = np.std(ref_sample)
            n_ref    = len(ref_sample)

            s_ref = (sd_ref**2)/n_ref

            for j, pct in enumerate(pcts_to_plot):
                v_list = list(gb.get_group((env, rep, cache_limits[env][pct])))
                #print(env, rep, pct, len(v_list))
                lifetime_avgs = data_sample(v_list,normalization_factor=normalization_factor,cutoff=5000)

                mean_q   = np.mean(lifetime_avgs)
                sd_q     = np.std(lifetime_avgs)
                n_q      = len(lifetime_avgs)
                s_q      = (sd_q**2)/n_q

                dif_of_means  = mean_ref - mean_q
                s_delta       = np.sqrt(s_ref +  s_q)

                dof           = (s_ref + s_q)**2 /( ((s_ref**2)/(n_ref-1)) + ((s_q**2)/(n_q-1)) )
                dof1          = n_ref + n_q - 2
                t_statistic   = dif_of_means / s_delta

                p_value = (1.0 - sp.t.cdf(abs(t_statistic),dof))
                pval1 = (1.0 - sp.t.cdf(abs(t_statistic),dof1))
                pval2 = sp.ttest_ind(ref_sample,lifetime_avgs)
                print(f"Env:{env[-2:]}, rep:{rep[0:3]}   comparing 100:{pct} (mean:{mean_q}, std:{sd_q})//\ndof:{dof:.2} or {n_ref+n_q-2} // T stat: {t_statistic:.4}, Pvalue: {p_value} or {pval1} or {pval2}\n------")
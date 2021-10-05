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
def plot_compare_conv_retraining(envs_to_plot, pcts_to_plot, rep):
    fig, ax = plt.subplots(len(envs_to_plot),3, figsize=(10,12))
    for e, env in enumerate(envs_to_plot):
        if env[-1] == '5':
            rwd_colrow0 = (3,9)
            rwd_colrow1= (16,9)
        else:
            rwd_colrow0 = (5,5)
            rwd_colrow1=(14,14)

        rect0 = plt.Rectangle(rwd_colrow0, 1, 1, facecolor='gray',edgecolor=None, alpha=0.6)
        rect1 = plt.Rectangle(rwd_colrow1, 1, 1, facecolor='g', edgecolor=None,alpha=0.3)
        ax[e,0].pcolor(grids[envs_to_plot.index(env)],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[e,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[e,0].set_aspect('equal')
        #ax[e,0].add_patch(rect0)
        ax[e,0].add_patch(rect1)
        ax[e,0].get_xaxis().set_visible(False)
        ax[e,0].get_yaxis().set_visible(False)
        ax[e,0].invert_yaxis()


        id_list = gb_base.get_group((env[0:22],rep))
        mf_retrain = []
        print('MF data')
        for id_num in id_list[0:1]:
            with open(parent_path+f'results/{id_num}_data.p','rb') as f:
                dats = pickle.load(f)
                raw_score = dats['total_reward'][5000:15000]
                normalization = analysis_specs['avg_max_rwd'][env[0:22]]
                transformed = rm((np.asarray(raw_score)+2.5)/(normalization +2.5) , 200)
                mf_retrain.append(transformed)
        means = np.nanmean(mf_retrain,axis=0)
        maxes = means+(np.nanstd(mf_retrain,axis=0)/np.sqrt(len(mf_retrain)))
        mins  = means-(np.nanstd(mf_retrain,axis=0)/np.sqrt(len(mf_retrain)))
        ax[e,2].plot(means, 'k', alpha=0.7)
        ax[e,2].fill_between(np.arange(len(means)),mins,maxes, color='k', alpha=0.2)

        print('EC bootstrapped data')
        for p, pct in enumerate(pcts_to_plot):
            print(pct)
            ec_performance = []
            mf_bootstrap = []
            try:
                id_list = gb.get_group((env,rep,int(cache_limits[env][100]*(pct/100)),15000))
                print(env,pct, len(id_list))
                for i, id_num in enumerate(id_list):
                    with open(parent_path+f'results/{id_num}_data.p','rb') as f:
                        dats = pickle.load(f)
                        raw_score = dats['bootstrap_reward'][0:15000]
                        normalization = analysis_specs['avg_max_rwd'][env[0:22]]
                        transformed = rm((np.asarray(raw_score)+2.5)/(normalization +2.5) , 200)
                        mf_bootstrap.append(transformed)

                        raw_score = dats['total_reward'][0:15000]
                        normalization = analysis_specs['avg_max_rwd'][env[0:22]]
                        transformed = rm((np.asarray(raw_score)+2.5)/(normalization +2.5) , 200)
                        ec_performance.append(transformed)

                means = np.nanmean(ec_performance,axis=0)
                ax[e,1].plot(means, label=f'{pct}')

                means = np.nanmean(mf_bootstrap,axis=0)
                ax[e,2].plot(means, label=f'{pct}')
            except:
                print(f'no data for EC{env}{rep}{int(cache_limits[env][100]*(pct/100))}')
        ax[e,1].legend(loc=0)
        ax[e,2].legend(loc=0)
        ax[0,1].set_title('EC perf')
        ax[0,2].set_title('Bootstrap Perf')
        ax[e,1].set_ylim(0,1.1)
        ax[e,2].set_ylim(0,1.1)
    plt.savefig(f'../figures/CH3/example_bootstrap.svg')
    plt.show()

def occupancy_plot(id_num):
    with open(parent_path+f'results/{id_num}_data.p','rb') as f:
        dats = pickle.load(f)
    occ_map = dats['occupancy']
    all_visits = np.nansum(occ_map)
    occ = occ_map.reshape(20,20)
    plt.imshow(occ/all_visits)
    plt.show()



def plot_all_retraining(env, pcts_to_plot, rep):
    fig, ax = plt.subplots(2,2, figsize=(10,12))

    ## get MF only -- baseline
    id_list = gb_base.get_group((env[0:22],rep,30000))
    mf_retrain = []
    for id_num in id_list:
        with open(parent_path+f'results/{id_num}_data.p','rb') as f:
            dats = pickle.load(f)
            raw_score = dats['total_reward'][5000:20000]
            normalization = analysis_specs['avg_max_rwd'][env[0:22]]
            transformed = rm((np.asarray(raw_score)+2.5)/(normalization +2.5) , 200)
            mf_retrain.append(transformed)
    means = np.nanmean(mf_retrain,axis=0)
    maxes = means+(np.nanstd(mf_retrain,axis=0)/np.sqrt(len(mf_retrain)))
    mins  = means-(np.nanstd(mf_retrain,axis=0)/np.sqrt(len(mf_retrain)))
    ax[0,1].plot(means, 'k', alpha=0.7)
    ax[0,1].fill_between(np.arange(len(means)),mins,maxes, color='k', alpha=0.2)

    print('EC bootstrapped data')
    for p, pct in enumerate(pcts_to_plot):
        print(pct)
        ec_performance = []
        mf_bootstrap = []
        try:
            id_list = gb.get_group((env,rep,int(cache_limits[env][100]*(pct/100)),15000))
            print(env,pct, len(id_list))
            for i, id_num in enumerate(id_list):
                with open(parent_path+f'results/{id_num}_data.p','rb') as f:
                    dats = pickle.load(f)
                    raw_score = dats['bootstrap_reward'][0:15000]
                    normalization = analysis_specs['avg_max_rwd'][env[0:22]]
                    transformed = rm((np.asarray(raw_score)+2.5)/(normalization +2.5) , 200)
                    mf_bootstrap.append(transformed)

                    raw_score = dats['total_reward'][0:15000]
                    normalization = analysis_specs['avg_max_rwd'][env[0:22]]
                    transformed = rm((np.asarray(raw_score)+2.5)/(normalization +2.5) , 200)
                    ec_performance.append(transformed)

            means = np.nanmean(ec_performance,axis=0)
            ax[0,0].plot(means, label=f'{pct}', color=colors[pct])

            means = np.nanmean(mf_bootstrap,axis=0)
            maxes = means+(np.nanstd(mf_bootstrap,axis=0)/np.sqrt(len(mf_bootstrap)))
            mins  = means-(np.nanstd(mf_bootstrap,axis=0)/np.sqrt(len(mf_bootstrap)))
            ax[0,1].plot(means, label=f'{pct}', color=colors[pct])
            ax[0,1].fill_between(np.arange(len(means)),mins,maxes, color=colors[pct], alpha=0.2)

        except:
            print(f'no data for EC{env}{rep}{int(cache_limits[env][100]*(pct/100))}')
    ax[0,0].legend(loc=0)
    ax[0,1].legend(loc=0)
    ax[0,0].set_title('EC perf')
    ax[0,1].set_title('Bootstrap Perf')
    ax[0,0].set_ylim(0,1.1)
    ax[0,1].set_ylim(0,1.1)
    #plt.savefig(f'../figures/CH3/example_bootstrap.svg')
    plt.show()


def plot_single_retraining(env, pcts_to_plot, rep,index):
    fig, ax = plt.subplots(2,2, figsize=(10,12))

    ## get MF only -- baseline
    id_list = gb_base.get_group((env[0:22],rep))
    filler = np.zeros(5000)
    filler[:] = np.nan
    mf_retrain = []
    for id_num in id_list[0:3]:
        print(id_num)
        with open(parent_path+f'results/{id_num}_data.p','rb') as f:
            dats = pickle.load(f)
            raw_score = dats['total_reward'][5000:20000]
            normalization = analysis_specs['avg_max_rwd'][env[0:22]]
            transformed = rm((np.asarray(raw_score)+2.5)/(normalization +2.5) , 200)
            if len(transformed) == 9801:
                transformed = np.concatenate((transformed,filler))

            mf_retrain.append(transformed)
    lens = [len(x) for x in mf_retrain]
    print(lens, len(mf_retrain), len(mf_retrain[0]))
    means = np.nanmean(np.asarray(mf_retrain),axis=0)
    maxes = means+(np.nanstd(mf_retrain,axis=0)/np.sqrt(len(mf_retrain)))
    mins  = means-(np.nanstd(mf_retrain,axis=0)/np.sqrt(len(mf_retrain)))
    ax[0,1].plot(means, 'k', alpha=0.7)
    ax[0,1].fill_between(np.arange(len(means)),mins,maxes, color='k', alpha=0.2)

    print('EC bootstrapped data')
    for p, pct in enumerate(pcts_to_plot):
        print(pct)
        ec_performance = []
        mf_bootstrap = []
        try:
            current_id_list = gb.get_group((env,rep,int(cache_limits[env][100]*(pct/100)),15000))
            print(env,pct, len(current_id_list), 'helloooooo')
            print(current_id_list)
            id_num = list(current_id_list)[index]
            print(id_num)
            with open(parent_path+f'results/{id_num}_data.p','rb') as f:
                dats = pickle.load(f)
                raw_score = dats['bootstrap_reward'][0:15000]
                normalization = analysis_specs['avg_max_rwd'][env[0:22]]
                transformed = rm((np.asarray(raw_score)+2.5)/(normalization +2.5) , 200)
                mf_bootstrap.append(transformed)

                raw_score = dats['total_reward'][0:15000]
                normalization = analysis_specs['avg_max_rwd'][env[0:22]]
                transformed = rm((np.asarray(raw_score)+2.5)/(normalization +2.5) , 200)
                ec_performance.append(transformed)

            ax[0,0].plot(ec_performance[0], label=f'{pct}', color=colors[pct])
            ax[0,1].plot(mf_bootstrap[0], label=f'{pct}', color=colors[pct])

        except:
            print(f'no data for EC{env}{rep}{int(cache_limits[env][100]*(pct/100))}')
    ax[0,0].legend(loc=0)
    ax[0,1].legend(loc=0)
    ax[0,0].set_title('EC perf')
    ax[0,1].set_title('Bootstrap Perf')
    ax[0,0].set_ylim(0,1.1)
    ax[0,1].set_ylim(0,1.1)
    #plt.savefig(f'../figures/CH3/example_bootstrap.svg')
    plt.show()


####
envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
pcts_to_plot = [100,75,50,25]
grids = get_grids(envs_to_plot)
cache_limits = analysis_specs['cache_limits']
#plot_single_retraining(envs_to_plot[1],[25,50,75,100],'structured',index=1)
#plot_all_retraining(envs_to_plot[1],[25,50,75,100],'structured')

env = envs_to_plot[1]
rep = 'structured'
pct = 50
id_list = list(gb.get_group((env,rep,int(cache_limits[env][100]*(pct/100)),15000)))
print(id_list[0])

occupancy_plot(id_list[2])
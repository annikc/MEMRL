import numpy as np
import scipy.stats as sp
import pickle
import gym
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from modules.Utils import running_mean as rm
import networkx as nx
from matplotlib.colors import ListedColormap

### plotting settings
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

analysis_specs = {
    'cache_limits':{'gridworld:gridworld-v11':{100:400, 75:300, 50:200, 25:100},
                    'gridworld:gridworld-v31':{100:365, 75:273, 50:182, 25:91},
                    'gridworld:gridworld-v41':{100:384, 75:288, 50:192, 25:96},
                    'gridworld:gridworld-v51':{100:286, 75:214, 50:143, 25:71}
                    },
    'avg_max_rwd':{'gridworld:gridworld-v11':9.87,
                   'gridworld:gridworld-v31':9.85,
                   'gridworld:gridworld-v41':9.84,
                   'gridworld:gridworld-v51':9.86
                   },
    'chance_perf':{'gridworld:gridworld-v11':[0.2216007853100826,0.005266129262900299], #chance performance calculated as average of 7 runs in each environment
                   'gridworld:gridworld-v31':[0.1987820242914986,0.002717778942716886],# data scaled to 0 1 interval using avg_max_rwd values for each environment
                   'gridworld:gridworld-v41':[0.2187621440148188,0.004092214797714025], # raw_data-(-2.5) / (max-(-2.5))
                   'gridworld:gridworld-v51':[0.2720107720758212,0.006256527951845687]
                   }
}
plot_specs = {
    'labels':{'analytic successor':'SR',
              'onehot':'onehot',
              'random':'random',
              'place_cell':'PC',
              'conv_latents':'latent'},
    'rep_colors':{'analytic successor':'C0',
                  'onehot':'C1',
                  'random':'C2',
                  'place_cell':'C4',
                  'conv_latents':'C3'}


}

N = 256
blu_vals = np.ones((N, 4))
blu_vals[:, 0] = np.linspace(39/256,1, N)
blu_vals[:, 1] = np.linspace(102/256,1, N)
blu_vals[:, 2] = np.linspace(199/256,1, N)
bluecmp = ListedColormap(blu_vals)
bluecmp.set_bad(color='w')

red_vals = np.ones((N, 4))
red_vals[:, 0] = np.linspace(235/256,1,N)
red_vals[:, 1] = np.linspace(77/256,1,N)
red_vals[:, 2] = np.linspace(57/256,1,N)
redcmp = ListedColormap(red_vals)
redcmp.set_bad(color='w')

blu_r_vals = np.ones((N, 4))
blu_r_vals[:, 0] = np.linspace(1,39/256, N) #np.linspace(1,80/256, N)
blu_r_vals[:, 1] = np.linspace(1,102/256, N) #np.linspace(1,162/256,N)
blu_r_vals[:, 2] = np.linspace(1,199/256, N) #np.linspace(1,213/256,N)
bluecmp_r = ListedColormap(blu_r_vals)
bluecmp_r.set_bad(color='w')

red_r_vals = np.ones((N, 4))
red_r_vals[:, 0] = np.linspace(1,235/256,N)
red_r_vals[:, 1] = np.linspace(1,77/256,N)
red_r_vals[:, 2] = np.linspace(1,57/256,N)
redcmp_r = ListedColormap(red_r_vals)
redcmp_r.set_bad(color='w')

linc_coolwarm = ListedColormap(np.concatenate((blu_vals,red_r_vals)))
linc_coolwarm_r = ListedColormap(np.concatenate((red_vals, blu_r_vals)))

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return (1-mix)*c1 + mix*c2 #mpl.colors.to_hex()#

def welchs_pval(ref_sample, query_sample):
    mean_ref = np.mean(ref_sample)
    sd_ref   = np.std(ref_sample)
    n_ref    = len(ref_sample)
    s_ref    = (sd_ref**2)/n_ref

    mean_q   = np.mean(query_sample)
    sd_q     = np.std(query_sample)
    n_q      = len(query_sample)
    s_q      = (sd_q**2)/n_q

    dif_of_means  = mean_ref - mean_q
    s_delta       = np.sqrt(s_ref +  s_q)

    dof           = (s_ref + s_q)**2 /( ((s_ref**2)/(n_ref-1)) + ((s_q**2)/(n_q-1)) )
    t_statistic   = dif_of_means / s_delta

    p_value = (1.0 - sp.t.cdf(abs(t_statistic),dof))

    return t_statistic, p_value

def structured_unstructured(df_element):
    map = {'analytic successor':'structured',
           'place_cell':'structured',
           'state-centred pc f0.05':'',
           'onehot':'unstructured',
           'random':'',
           'conv_latents':''}
    new_element = map[df_element]
    return new_element

def data_avg_std(list_of_ids, normalization_factor=10, cutoff=5000):
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

    avg_ = np.mean(sample_avgs)
    std_ = np.std(sample_avgs)

    return avg_, std_

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

def make_env_graph(env):
    action_cols = ['orange','red','green','blue']
    G = nx.DiGraph() # why directed/undirected graph?

    for action in range(env.action_space.n):
        # down, up, right, left
        for state2d in env.useable:
            state1d = env.twoD2oneD(state2d)
            next_state = np.where(env.P[action,state1d,:]==1)[0]
            if len(next_state) ==0:
                pass
            else:
                for sprime in next_state:
                    edge_weight = env.P[action,state1d,sprime]
                    G.add_edge(state1d, sprime,color=action_cols[action],weight=edge_weight)

    return G

def compute_graph_distance_matrix(G, env):
    x = nx.shortest_path(G)
    useable_1d = [env.twoD2oneD(x) for x in env.useable]
    shortest_dist_array = np.zeros((env.nstates,env.nstates))
    shortest_dist_array[:]=np.nan

    for start in useable_1d:
        for target in list(x[start].keys()):
            shortest_dist_array[start][target]= len(x[start][target])-1

    return shortest_dist_array


def get_grids(env_names):
    grids = []
    for ind, environment_to_plot in enumerate(env_names):
        env = gym.make(environment_to_plot)
        plt.close()
        grids.append(env.grid)

    return grids

def plot_each(list_of_ids,data_dir,cutoff=25000, smoothing=500):
    plt.figure()
    for id_num in list_of_ids:
        with open(data_dir+ f'results/{id_num}_data.p', 'rb') as f:
            dats = pickle.load(f)
            reward_info = dats['total_reward'][0:cutoff]
        processed_rwd = rm(reward_info, smoothing)
        plt.plot(processed_rwd, label=id_num[0:8])
    plt.legend(loc='upper center', bbox_to_anchor=(0.1,1.1))
    plt.ylim([-4,12])
    plt.show()

def avg_performance_over_envs(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,save=False,savename='',plot_title='',legend=False,compare_chance=False, **kwargs):
    if save:
        if savename=='':
            raise Exception('Must pass argument to savename to specify title to save plot')

    cache_limits = analysis_specs['cache_limits']
    avg_max_rwd  = analysis_specs['avg_max_rwd']

    convert_rep_to_color = kwargs.get('colors',plot_specs['rep_colors'])
    labels_for_plot      = kwargs.get('labels',plot_specs['labels'])

    fig, ax = plt.subplots(len(envs_to_plot),2,figsize=(2*(len(reps_to_plot)+2),3*len(envs_to_plot)), sharex='col', gridspec_kw={'width_ratios': [1, 1]})
    ## rows = different environments
    ## column 0 = env grid
    ## column 1 = performance comparison across representations; cache sizes
    bar_width = 0.75
    for i, env in enumerate(envs_to_plot):
        if env[-2:] == '51':
            rwd_colrow= (16,9)
        else:
            rwd_colrow=(14,14)

        rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
        ax[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[i,0].set_aspect('equal')
        ax[i,0].add_patch(rect)
        ax[i,0].get_xaxis().set_visible(False)
        ax[i,0].get_yaxis().set_visible(False)
        ax[i,0].invert_yaxis()

        normalization_factor = avg_max_rwd[env]

        for r, rep in enumerate(reps_to_plot):
            ref_sample = data_sample(list(gb.get_group((env, rep, cache_limits[env][100]))),normalization_factor=normalization_factor,cutoff=5000)
            for j, pct in enumerate(pcts_to_plot):
                v_list = list(gb.get_group((env, rep, cache_limits[env][pct])))
                print(env, rep, pct, len(v_list))
                sample_avgs = data_sample(v_list,normalization_factor=normalization_factor,cutoff=5000)
                avg_, std_ = np.mean(sample_avgs), np.std(sample_avgs)
                ax[i,1].bar(r*len(pcts_to_plot)+bar_width*j,avg_,yerr=std_,width=bar_width, color=convert_rep_to_color[rep], alpha=50/100,capsize=2)
                xpoints = (bar_width/10)*np.random.randn(len(sample_avgs))+(r*len(pcts_to_plot)+bar_width*j)
                ax[i,1].scatter(xpoints,sample_avgs, facecolor=convert_rep_to_color[rep],alpha=1,edgecolor=None,s=12,zorder=10)
                if pct != 100:
                    t, p = welchs_pval(ref_sample,sample_avgs)
                    if p<0.001 and t<0:
                        x1, x2 = r*len(pcts_to_plot), r*len(pcts_to_plot)+bar_width*j
                        ax[i,1].plot([x1,x2],[1.0+0.05*j,1.0+0.05*j],color='k')
                        ax[i,1].text(x1+(x2-x1)/2,1.0+0.05*j,f'p={p:.4}',fontsize=5,ha='center')

        ax[i,1].set_ylim([0.75,1.])
        ax[i,1].set_yticks(np.arange(0.75,1.,0.05))
        #ax[i,1].set_yticklabels([0,'',50,'',100,])
        ax[i,1].set_ylabel('Performance \n(% Optimal)')

        if compare_chance:
            ax[i,1].axhline(analysis_specs['chance_perf'][env][0],c='gray',linestyle=':',alpha=0.5)

    ax[i,1].set_xticks(np.arange(0,len(pcts_to_plot)*len(reps_to_plot),len(pcts_to_plot))+bar_width*0.5*j)
    ax[i,1].set_xticklabels([labels_for_plot[x] for x in reps_to_plot],rotation=0)
    ax[i,1].set_xlabel('State Encoding')

    if legend=='reps':
        legend_patch_list = []
        for rep in reps_to_plot:
            legend_patch_list.append(mpatches.Patch(color=convert_rep_to_color[rep], label=labels_for_plot[rep]))
        plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.1), loc='lower center', ncol=len(legend_patch_list),title='State Representation')
    elif legend=='pcts':
        legend_patch_list = []
        for pct in pcts_to_plot:
            legend_patch_list.append(mpatches.Patch(color='gray',label=f'{pct}',alpha=pct/100))
            plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.16), loc='lower center', ncol=len(legend_patch_list), title='Episodic Memory Capacity (%)')
    else:
        if plot_title=='':
            print('No title passed to arg plot_title')
        ax[0,1].set_title(plot_title)
    if save:
        format = kwargs.get('format','svg')
        plt.savefig(f'../figures/CH2/{savename}.{format}', format=format)
    plt.show()


def avg_performance_over_envs_violins(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,save=False,savename='',plot_title='',legend=False,compare_chance=False, **kwargs):
    if save:
        if savename=='':
            raise Exception('Must pass argument to savename to specify title to save plot')

    cache_limits = analysis_specs['cache_limits']
    avg_max_rwd  = analysis_specs['avg_max_rwd']

    convert_rep_to_color = kwargs.get('colors',plot_specs['rep_colors'])
    labels_for_plot      = kwargs.get('labels',plot_specs['labels'])

    fig, ax = plt.subplots(len(envs_to_plot),len(reps_to_plot)+1,figsize=(2*(len(reps_to_plot)+2)+4,3*len(envs_to_plot)), sharex='col', sharey=False,gridspec_kw={'width_ratios': np.ones(1+len(reps_to_plot))})
    ## rows = different environments
    ## column 0 = env grid
    ## column 1 = performance comparison across representations; cache sizes
    bar_width = 0.75
    minmax = [(.94,1.02), (.80,1.05), (.80,1.05), (.74,1.05)]
    for i, env in enumerate(envs_to_plot):
        if env[-2:] == '51':
            rwd_colrow= (16,9)
        else:
            rwd_colrow=(14,14)

        rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
        ax[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[i,0].set_aspect('equal')
        ax[i,0].add_patch(rect)
        ax[i,0].get_xaxis().set_visible(False)
        ax[i,0].get_yaxis().set_visible(False)
        ax[i,0].invert_yaxis()

        normalization_factor = avg_max_rwd[env]

        for r, rep in enumerate(reps_to_plot):
            ref_sample = data_sample(list(gb.get_group((env, rep, cache_limits[env][100]))),normalization_factor=normalization_factor,cutoff=5000)
            dats = []
            for j, pct in enumerate(pcts_to_plot):
                v_list = list(gb.get_group((env, rep, cache_limits[env][pct])))
                print(env, rep, pct, len(v_list))
                sample_avgs = data_sample(v_list,normalization_factor=normalization_factor,cutoff=5000)
                dats.append(sample_avgs)
                if pct != 100:
                    t, p = welchs_pval(ref_sample,sample_avgs)
                    if p<0.001 and t<0:
                        x1, x2 = 100, pct
                        ax[i,1].plot([x1,x2],[1.0+0.01*(j-1),1.0+0.01*(j-1)],color='k')
                        ax[i,1].text(x1+(x2-x1)/2,1.0+0.01*(j-1),f'p={p:.4}',fontsize=5,ha='center')
            body = ax[i,r+1].violinplot(positions=pcts_to_plot,dataset=dats,vert=True, widths=10,showmeans=True)
            ax[i,r+1].set_ylim(minmax[i])


            for violinkey in body.keys():
                if violinkey == 'bodies':
                    for b in body['bodies']:
                        b.set_color(convert_rep_to_color[rep])
                        b.set_alpha(pct/100)
                else:
                    body[violinkey].set_color(convert_rep_to_color[rep])

        ax[i,1].set_ylabel('Performance \n(% Optimal)')

    for r, rep in enumerate(reps_to_plot):
        if compare_chance:
            for i, env in enumerate(envs_to_plot):
                ax[i,r+1].axhline(analysis_specs['chance_perf'][env][0],c='gray',linestyle=':',alpha=0.5)
        ax[i,r+1].set_xlim([110,40])
        ax[i,r+1].set_xticks(pcts_to_plot)
        #ax[i,r+1].set_yticks(np.arange(0.75,1.1,0.05))
        #ax[i,r+1].set_yticklabels(['',80,'',90,'',100,'',''])
        ax[i,r+1].set_xlabel('Memory Capacity (%)')

        '''ax[i,r+1].set_ylim([0,1.2])
        ax[i,r+1].set_xlim([110,15])
        ax[i,r+1].set_xticks(pcts_to_plot)
        ax[i,r+1].set_ylim([0.25,1.2])
        ax[i,r+1].set_yticks(np.arange(0,1.25,0.25))
        ax[i,r+1].set_yticklabels([0,'',50,'',100,])
        ax[i,r+1].set_xlabel('Memory Capacity (%)')'''


    if legend=='reps':
        legend_patch_list = []
        for rep in reps_to_plot:
            legend_patch_list.append(mpatches.Patch(color=convert_rep_to_color[rep], label=labels_for_plot[rep]))
        plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.1), loc='lower center', ncol=len(legend_patch_list),title='State Representation')
    elif legend=='pcts':
        legend_patch_list = []
        for pct in pcts_to_plot:
            legend_patch_list.append(mpatches.Patch(color='gray',label=f'{pct}',alpha=pct/100))
            plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.16), loc='lower center', ncol=len(legend_patch_list), title='Episodic Memory Capacity (%)')
    else:
        if plot_title=='':
            print('No title passed to arg plot_title')
        ax[0,1].set_title(plot_title)
    if save:
        format = kwargs.get('format','svg')
        plt.savefig(f'../figures/CH2/{savename}.{format}', format=format)
    plt.show()

def avg_performance_over_envs_violins_sidebyside(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,save=False,savename='',plot_title='',legend=False,compare_chance=False, **kwargs):
    if save:
        if savename=='':
            raise Exception('Must pass argument to savename to specify title to save plot')

    cache_limits = analysis_specs['cache_limits']
    avg_max_rwd  = analysis_specs['avg_max_rwd']

    convert_rep_to_color = kwargs.get('colors',plot_specs['rep_colors'])
    labels_for_plot      = kwargs.get('labels',plot_specs['labels'])

    fig, ax = plt.subplots(len(envs_to_plot),2,figsize=(6,3*len(envs_to_plot)), sharex='col', sharey='col',gridspec_kw={'width_ratios': [1,1]})
    ## rows = different environments
    ## column 0 = env grid
    ## column 1 = performance comparison across representations; cache sizes
    bar_width = 0.75
    for i, env in enumerate(envs_to_plot):
        if env[-2:] == '51':
            rwd_colrow= (16,9)
        else:
            rwd_colrow=(14,14)

        rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
        ax[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[i,0].set_aspect('equal')
        ax[i,0].add_patch(rect)
        ax[i,0].get_xaxis().set_visible(False)
        ax[i,0].get_yaxis().set_visible(False)
        ax[i,0].invert_yaxis()

        normalization_factor = avg_max_rwd[env]

        for r, rep in enumerate(reps_to_plot):
            ref_sample = data_sample(list(gb.get_group((env, rep, cache_limits[env][100]))),normalization_factor=normalization_factor,cutoff=5000)
            dats = []
            for j, pct in enumerate(pcts_to_plot):
                v_list = list(gb.get_group((env, rep, int(cache_limits[env][100]*(pct/100)))))
                print(env, rep, pct, len(v_list))
                sample_avgs = data_sample(v_list,normalization_factor=normalization_factor,cutoff=5000)
                dats.append(sample_avgs)
            body = ax[i,1].violinplot(positions=[r],dataset=dats,vert=True, widths=0.6, showmeans=True)
            #xpoints = (1/10)*np.random.randn(len(sample_avgs))+(r*len(pcts_to_plot)+j)
            #ax[i,1].scatter(xpoints,sample_avgs, facecolor='gray',alpha=1,edgecolor=None,s=12,zorder=10)
            for violinkey in body.keys():
                if violinkey == 'bodies':
                    for b in body['bodies']:
                        b.set_color(convert_rep_to_color[rep])
                        b.set_alpha(50/100)
                else:
                    body[violinkey].set_color(convert_rep_to_color[rep])

        ax[i,1].set_ylabel('Performance \n(% Optimal)')


    if compare_chance:
        for i, env in enumerate(envs_to_plot):
            ax[i,1].axhline(analysis_specs['chance_perf'][env][0],c='gray',linestyle=':',alpha=0.5)
    ax[i,1].set_ylim([0,1.])
    ax[i,1].set_ylim([0.75,1.])
    ax[i,1].set_yticks(np.arange(0.75,1.,0.05))


    if legend=='reps':
        legend_patch_list = []
        for rep in reps_to_plot:
            legend_patch_list.append(mpatches.Patch(color=convert_rep_to_color[rep], label=labels_for_plot[rep]))
        plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.1), loc='lower center', ncol=len(legend_patch_list),title='State Representation')
    elif legend=='pcts':
        legend_patch_list = []
        for pct in pcts_to_plot:
            legend_patch_list.append(mpatches.Patch(color='gray',label=f'{pct}',alpha=pct/100))
            plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.16), loc='lower center', ncol=len(legend_patch_list), title='Episodic Memory Capacity (%)')
    else:
        if plot_title=='':
            print('No title passed to arg plot_title')
        ax[0,1].set_title(plot_title)
    if save:
        format = kwargs.get('format','svg')
        plt.savefig(f'../figures/CH2/{savename}.{format}', format=format)
    plt.show()




def avg_perf_over_envs_lines(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,save=False,savename='',plot_title='',legend=False,compare_chance=False, **kwargs):
    if save:
        if savename=='':
            raise Exception('Must pass argument to savename to specify title to save plot')

    cache_limits = analysis_specs['cache_limits']
    avg_max_rwd  = analysis_specs['avg_max_rwd']

    convert_rep_to_color = kwargs.get('colors',plot_specs['rep_colors'])
    labels_for_plot      = kwargs.get('labels',plot_specs['labels'])

    fig, ax = plt.subplots(len(envs_to_plot),2,figsize=(10,3*len(envs_to_plot)), sharex='col', gridspec_kw={'width_ratios': [1, 2]})
    ## rows = different environments
    ## column 0 = env grid
    ## column 1 = performance comparison across representations; cache sizes
    for i, env in enumerate(envs_to_plot):
        if env[-2:] == '51':
            rwd_colrow= (16,9)
        else:
            rwd_colrow=(14,14)

        rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
        ax[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[i,0].set_aspect('equal')
        ax[i,0].add_patch(rect)
        ax[i,0].get_xaxis().set_visible(False)
        ax[i,0].get_yaxis().set_visible(False)
        ax[i,0].invert_yaxis()

        normalization_factor = avg_max_rwd[env]

        for r, rep in enumerate(reps_to_plot):
            linevalues_avgs = []
            linevalues_stds = []
            for j, pct in enumerate(pcts_to_plot):
                v_list = list(gb.get_group((env, rep, int(cache_limits[env][100]*(pct/100)))))
                print(env, rep, pct, len(v_list))
                avgs,stds = data_avg_std(v_list,normalization_factor=normalization_factor,cutoff=5000)
                linevalues_avgs.append(avgs)
                linevalues_stds.append(stds)
                if pct == 100 and rep=='structured':
                    ax[i,1].axhline(avgs,c='gray',linestyle=':',alpha=0.5)

            ax[i,1].errorbar(pcts_to_plot, linevalues_avgs, yerr=linevalues_stds,color=convert_rep_to_color[rep],marker='o',capsize=6)
                #ax[i,1].bar(r*len(pcts_to_plot)+bar_width*j,avg_cos, yerr=sem_cos,width=bar_width, color=convert_rep_to_color[rep], alpha=pct/100)

        ax[i,1].set_ylim([0,1.2])
        ax[i,1].set_yticks(np.arange(0,1.25,0.25))
        ax[i,1].set_yticklabels([0,'',50,'',100,])
        ax[i,1].set_ylabel('Performance \n(% Optimal)')

        if compare_chance:
            #ax[i,1].axhline(1,c='gray',linestyle='--',alpha=0.5)
            ax[i,1].axhline(analysis_specs['chance_perf'][env][0],c='gray',linestyle=':',alpha=0.5)

    ax[i,1].set_xlabel('Memory Capacity (%)')
    ax[i,1].set_xticks(pcts_to_plot)
    ax[i,1].set_xticklabels(pcts_to_plot)
    buffer=5
    ax[i,1].set_xlim([max(pcts_to_plot)+buffer,min(pcts_to_plot)-buffer])

    if legend=='reps':
        legend_patch_list = []
        for rep in reps_to_plot:
            legend_patch_list.append(mpatches.Patch(color=convert_rep_to_color[rep], label=labels_for_plot[rep]))
        plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.16), loc='lower center', ncol=len(legend_patch_list),title='State Representation')
    elif legend=='pcts':
        legend_patch_list = []
        for pct in pcts_to_plot:
            legend_patch_list.append(mpatches.Patch(color='gray',label=f'{pct}',alpha=pct/100))
            plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.16), loc='lower center', ncol=len(legend_patch_list), title='Episodic Memory Capacity (%)')
    else:
        if plot_title=='':
            print('No title passed to arg plot_title')
        ax[0,1].set_title(plot_title)
    if save:
        format = kwargs.get('format','svg')
        plt.savefig(f'../figures/CH2/{savename}.{format}', format=format)
    plt.show()

def compare_perf_over_envs_lines(gb_probe,gb_ref,envs_to_plot,reps_to_plot,pcts_to_plot,grids,save=False,savename='',plot_title='',legend=False,compare_chance=False, **kwargs):
    gbs = [gb_probe, gb_ref]
    if save:
        if savename=='':
            raise Exception('Must pass argument to savename to specify title to save plot')

    cache_limits = analysis_specs['cache_limits']
    avg_max_rwd  = analysis_specs['avg_max_rwd']

    convert_rep_to_color = kwargs.get('colors',plot_specs['rep_colors'])
    convert_rep_to_linestyle = kwargs.get('linestyles',{0:'-', 1:':'})
    labels_for_plot      = kwargs.get('labels',plot_specs['labels'])

    fig, ax = plt.subplots(len(envs_to_plot),2,figsize=(14,3*len(envs_to_plot)), sharex='col', sharey='col', gridspec_kw={'width_ratios': [1, 2]})
    ## rows = different environments
    ## column 0 = env grid
    ## column 1 = performance comparison across representations; cache sizes
    for i, env in enumerate(envs_to_plot):
        if env[-2:] == '51':
            rwd_colrow= (16,9)
        else:
            rwd_colrow=(14,14)

        rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
        ax[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[i,0].set_aspect('equal')
        ax[i,0].add_patch(rect)
        ax[i,0].get_xaxis().set_visible(False)
        ax[i,0].get_yaxis().set_visible(False)
        ax[i,0].invert_yaxis()

        normalization_factor = avg_max_rwd[env]
        for g, gb in enumerate(gbs):
            for r, rep in enumerate(reps_to_plot):
                linevalues_avgs = []
                linevalues_stds = []
                for j, pct in enumerate(pcts_to_plot):
                    v_list = list(gb.get_group((env, rep, cache_limits[env][pct])))
                    print(env, rep, pct, len(v_list))
                    avgs,stds = data_avg_std(v_list,normalization_factor=normalization_factor,cutoff=5000)
                    linevalues_avgs.append(avgs)
                    linevalues_stds.append(stds)
                alf = [1,0.5]
                ax[i,1].errorbar(pcts_to_plot, linevalues_avgs, yerr=linevalues_stds,color=convert_rep_to_color[rep],marker='o',capsize=4, linestyle=convert_rep_to_linestyle[g],alpha=alf[g])
                #ax[i,1].bar(r*len(pcts_to_plot)+bar_width*j,avg_cos, yerr=sem_cos,width=bar_width, color=convert_rep_to_color[rep], alpha=pct/100)

            ax[i,1].set_ylim([0,1.1])
            ax[i,1].set_yticks(np.arange(0,1.25,0.25))
            ax[i,1].set_yticklabels([0,'',50,'',100,])
            if g==0:
                ax[i,1].set_ylabel('Performance \n(% Optimal)')

            if compare_chance:
                ax[i,1].axhline(analysis_specs['chance_perf'][env][0],c='gray',linestyle=':',alpha=0.5)


            ax[i,1].set_xticks(pcts_to_plot)
            ax[i,1].set_xticklabels(pcts_to_plot)
            ax[i,1].set_xlim([105,20])

    ax[i,1].set_xlabel('Memory Capacity (%)')

    if legend=='reps':
        legend_patch_list = []
        for rep in reps_to_plot:
            legend_patch_list.append(mpatches.Patch(color=convert_rep_to_color[rep], label=labels_for_plot[rep]))
        plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower center', ncol=len(legend_patch_list),title='State Encoding')
    elif legend=='pcts':
        legend_patch_list = []
        for pct in pcts_to_plot:
            legend_patch_list.append(mpatches.Patch(color='gray',label=f'{pct}',alpha=pct/100))
            plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower center', ncol=len(legend_patch_list), title='Episodic Memory Capacity (%)')
    else:
        if plot_title=='':
            print('No title passed to arg plot_title')
        ax[0,1].set_title(plot_title)
    if save:
        format = kwargs.get('format','svg')
        plt.savefig(f'../figures/CH2/{savename}.{format}', format=format)
    plt.show()

def compare_perf_over_envs_lines_separated(gb_probe,gb_ref,envs_to_plot,reps_to_plot,pcts_to_plot,grids,save=False,savename='',plot_title='',legend=False,compare_chance=False, **kwargs):
    gbs = [gb_probe, gb_ref]
    if save:
        if savename=='':
            raise Exception('Must pass argument to savename to specify title to save plot')

    cache_limits = analysis_specs['cache_limits']
    avg_max_rwd  = analysis_specs['avg_max_rwd']

    convert_rep_to_color = kwargs.get('colors',plot_specs['rep_colors'])
    convert_rep_to_linestyle = kwargs.get('linestyles',{0:'-', 1:':'})
    labels_for_plot      = kwargs.get('labels',plot_specs['labels'])

    fig, ax = plt.subplots(len(envs_to_plot),3,figsize=(14,2*len(envs_to_plot)), sharex='col', sharey='col', gridspec_kw={'width_ratios': [1, 1, 1]})
    ## rows = different environments
    ## column 0 = env grid
    ## column 1 = performance comparison across representations; cache sizes
    for i, env in enumerate(envs_to_plot):
        if env[-2:] == '51':
            rwd_colrow= (16,9)
        else:
            rwd_colrow=(14,14)

        rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
        ax[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[i,0].set_aspect('equal')
        ax[i,0].add_patch(rect)
        ax[i,0].get_xaxis().set_visible(False)
        ax[i,0].get_yaxis().set_visible(False)
        ax[i,0].invert_yaxis()

        normalization_factor = avg_max_rwd[env]
        for g, gb in enumerate(gbs):
            for r, rep in enumerate(reps_to_plot):
                linevalues_avgs = []
                linevalues_stds = []
                for j, pct in enumerate(pcts_to_plot):
                    v_list = list(gb.get_group((env, rep, cache_limits[env][pct])))
                    print(env, rep, pct, len(v_list))
                    avgs,stds = data_avg_std(v_list,normalization_factor=normalization_factor,cutoff=5000)
                    linevalues_avgs.append(avgs)
                    linevalues_stds.append(stds)
                linestyles = ['-',':']
                alphas = [1,.5]
                ax[i,r+1].errorbar(pcts_to_plot, linevalues_avgs, yerr=linevalues_stds,color=convert_rep_to_color[rep],marker='o',linestyle=linestyles[g],capsize=4,alpha=alphas[g])
                #ax[i,1].bar(r*len(pcts_to_plot)+bar_width*j,avg_cos, yerr=sem_cos,width=bar_width, color=convert_rep_to_color[rep], alpha=pct/100)

            ax[i,g+1].set_ylim([0,1.1])
            ax[i,g+1].set_yticks(np.arange(0,1.25,0.25))
            ax[i,g+1].set_yticklabels([0,'',50,'',100,])
            if g==0:
                ax[i,1].set_ylabel('Performance \n(% Optimal)')

            if compare_chance:
                ax[i,g+1].axhline(analysis_specs['chance_perf'][env][0],c='gray',linestyle=':',alpha=0.5)


            ax[i,g+1].set_xticks(pcts_to_plot)
            ax[i,g+1].set_xticklabels(pcts_to_plot)
            ax[i,g+1].set_xlim([105,20])

    ax[i,g+1].set_xlabel('Memory Capacity (%)')
    ax[0,1].set_title('Forget Oldest Entry')
    ax[0,2].set_title('Forget Random Entry')

    if legend=='reps':
        legend_patch_list = []
        for rep in reps_to_plot:
            legend_patch_list.append(mpatches.Patch(color=convert_rep_to_color[rep], label=labels_for_plot[rep]))
        plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower center', ncol=len(legend_patch_list),title='State Representation')
    elif legend=='pcts':
        legend_patch_list = []
        for pct in pcts_to_plot:
            legend_patch_list.append(mpatches.Patch(color='gray',label=f'{pct}',alpha=pct/100))
            plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower center', ncol=len(legend_patch_list), title='Episodic Memory Capacity (%)')
    else:
        if plot_title=='':
            print('No title passed to arg plot_title')
        ax[0,1].set_title(plot_title)
    if save:
        format = kwargs.get('format','svg')
        plt.savefig(f'../figures/CH2/{savename}.{format}', format=format)
    plt.show()

def compare_perf_over_envs_violins_sidebyside(gb_probe,gb_ref,envs_to_plot,reps_to_plot,pcts_to_plot,grids,save=False,savename='',plot_title='',legend=False,compare_chance=False, **kwargs):
    gbs = [gb_probe, gb_ref]
    if save:
        if savename=='':
            raise Exception('Must pass argument to savename to specify title to save plot')

    cache_limits = analysis_specs['cache_limits']
    avg_max_rwd  = analysis_specs['avg_max_rwd']

    convert_rep_to_color = kwargs.get('colors',plot_specs['rep_colors'])
    convert_rep_to_linestyle = kwargs.get('linestyles',{0:'-', 1:':'})
    labels_for_plot      = kwargs.get('labels',plot_specs['labels'])

    fig, ax = plt.subplots(len(envs_to_plot),3,figsize=(14,3*len(envs_to_plot)), sharex='col', sharey='col', gridspec_kw={'width_ratios': [1, 1, 1]})
    ## rows = different environments
    ## column 0 = env grid
    ## column 1 = performance comparison across representations; cache sizes
    for i, env in enumerate(envs_to_plot):
        if env[-2:] == '51':
            rwd_colrow= (16,9)
        else:
            rwd_colrow=(14,14)

        rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
        ax[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[i,0].set_aspect('equal')
        ax[i,0].add_patch(rect)
        ax[i,0].get_xaxis().set_visible(False)
        ax[i,0].get_yaxis().set_visible(False)
        ax[i,0].invert_yaxis()

        normalization_factor = avg_max_rwd[env]
        for g, gb in enumerate(gbs):
            for r, rep in enumerate(reps_to_plot):
                dats =[]
                for j, pct in enumerate(pcts_to_plot):
                    v_list = list(gb.get_group((env, rep, cache_limits[env][pct])))
                    print(env, rep, pct, len(v_list))
                    sample_avgs = data_sample(v_list,normalization_factor=normalization_factor,cutoff=5000)
                    dats.append(sample_avgs)

                violin_width = 7
                positions = np.asarray(pcts_to_plot)+violin_width*(1/2-g)

                body = ax[i,r+1].violinplot(positions=positions,dataset=dats,vert=True, widths=violin_width,showmeans=False)
                darkness = [75,25]
                for violinkey in body.keys():
                        if violinkey == 'bodies':
                            for b in body['bodies']:
                                b.set_color(convert_rep_to_color[rep])
                                b.set_alpha(darkness[g]/100)
                        else:
                            body[violinkey].set_color(convert_rep_to_color[rep])

            ax[i,g+1].set_ylim([0,1.1])
            ax[i,g+1].set_yticks(np.arange(0,1.25,0.25))
            ax[i,g+1].set_yticklabels([0,'',50,'',100,])
            if g==0:
                ax[i,1].set_ylabel('Performance \n(% Optimal)')

            if compare_chance:
                ax[i,g+1].axhline(analysis_specs['chance_perf'][env][0],c='gray',linestyle=':',alpha=0.5)


            ax[i,g+1].set_xticks(pcts_to_plot)
            ax[i,g+1].set_xticklabels(pcts_to_plot)
            ax[i,g+1].set_xlim([75+violin_width+2, 25-violin_width-2])

    ax[i,g+1].set_xlabel('Memory Capacity (%)')
    ax[0,1].set_title('Unstructured State Encoding')
    ax[0,2].set_title('Structured State Encoding')

    if legend=='reps':
        legend_patch_list = []
        for rep in reps_to_plot:
            legend_patch_list.append(mpatches.Patch(color=convert_rep_to_color[rep], label=labels_for_plot[rep]))
        plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower center', ncol=len(legend_patch_list),title='State Encoding')
    elif legend=='pcts':
        legend_patch_list = []
        fgt = ['Oldest Entry', 'Random Entry']
        for i, pct in enumerate(darkness):
            legend_patch_list.append(mpatches.Patch(color='gray',label=f'{fgt[i]}',alpha=pct/100))
            plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower center', ncol=len(legend_patch_list), title='Forgetting Rule')
    else:
        if plot_title=='':
            print('No title passed to arg plot_title')
        ax[0,1].set_title(plot_title)
    if save:
        format = kwargs.get('format','svg')
        plt.savefig(f'../figures/CH2/{savename}.{format}', format=format)
    plt.show()


def compare_avg_performance_against_random(probe_gb, rand_gb,env,reps_to_plot,pcts_to_plot,grid,save=False,savename='',plot_title='',legend=False):
    gbs = [rand_gb,probe_gb]
    version = env[-2:-1]
    if save:
        if savename=='':
            raise Exception('Must pass argument to savename to specify title to save plot')

    cache_limits = analysis_specs['cache_limits']
    avg_max_rwd  = analysis_specs['avg_max_rwd']

    convert_rep_to_color = plot_specs['rep_colors']
    labels_for_plot      = plot_specs['labels']

    fig, ax = plt.subplots(len(reps_to_plot),2,figsize=(10,len(reps_to_plot)*2), sharey='col',sharex='col',gridspec_kw={'width_ratios': [1, 3]})
    width=0.45
    if env[-2:] == '51':
        rwd_colrow= (16,9)
    else:
        rwd_colrow=(14,14)

    rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
    ax[0,0].pcolor(grid,cmap='bone_r',edgecolors='k', linewidths=0.1)
    ax[0,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
    ax[0,0].set_aspect('equal')
    ax[0,0].add_patch(rect)
    ax[0,0].get_xaxis().set_visible(False)
    ax[0,0].get_yaxis().set_visible(False)
    ax[0,0].invert_yaxis()

    norm = avg_max_rwd[env]
    for r, rep in enumerate(reps_to_plot):
        if r ==0:
            pass
        else:
            ax[r,0].axis('off')
        # ax[0,j] plot average performance with error bars
        # ax[1,j] plot variance of differnt rep types
        for j, pct in enumerate(pcts_to_plot):
            for g, gb in enumerate(gbs):
                v_list = list(gb.get_group((env, rep, cache_limits[env][pct])))
                print(env, rep, pct, len(v_list))
                avg_, std_ = data_avg_std(v_list,cutoff=5000,normalization_factor=norm)
                avg_cos, std_cos = np.mean(avg_), np.mean(std_)
                if g ==0:
                    ax[r,1].bar(j+width*g,avg_cos, yerr=std_cos,width=width, edgecolor=convert_rep_to_color[rep],fill=False,hatch='//', alpha=pct/100)
                else:
                    ax[r,1].bar(j+width*g,avg_cos, yerr=std_cos,width=width, color=convert_rep_to_color[rep], alpha=pct/100)

        ax[r,1].set_ylim([0,1.2])
        ax[r,1].set_yticks(np.arange(0,1.5,0.25))
        right = 1
        top = 0.98
        ax[r,1].text(right, top, f'{labels_for_plot[rep]}', horizontalalignment='right', verticalalignment='top', transform=ax[r,1].transAxes)
        ax[r,1].set_yticklabels([0,'',50,'',100,''])
        ax[r,1].set_ylabel(f'Performance \n(% Optimal)')

    ax[r,1].set_xticks(np.arange(len(pcts_to_plot))+(width/2))
    ax[r,1].set_xticklabels(pcts_to_plot)
    ax[r,1].set_xlabel('Memory Capacity (%)')

    p_rand = mpatches.Patch(fill=False,edgecolor='gray',alpha=1, hatch='///',label='Random Entry')
    p_old = mpatches.Patch(color='gray',alpha=1, label='Oldest Entry')
    plt.legend(handles=[p_rand,p_old], bbox_to_anchor=(0.5, len(reps_to_plot)*1.16), loc='lower center', ncol=2, title='Forgetting Rule')
    if save:
        format = 'svg'
        plt.savefig(f'../figures/CH2/compare_rand_forgetting_{version}.{format}', format=format)
    plt.show()




# JUNKYARD
def get_avg_std(list_of_ids, normalization_factor=1, cutoff=5000, smoothing=500):
    data_dir='../../Data/results/'
    results = []
    for id_num in list_of_ids:
        with open(data_dir+ f'{id_num}_data.p', 'rb') as f:
            dats = pickle.load(f)
            reward_info = dats['total_reward'][0:cutoff]
            results.append(reward_info)

    pp = np.vstack(results)/normalization_factor
    avg_ = rm(np.mean(pp,axis=0),smoothing)
    std_ = rm(np.std(pp, axis=0), smoothing)

    return avg_, std_

def get_detailed_avg_std(list_of_ids, cutoff=5000, smoothing=500):
    data_dir='../../Data/results/'
    results = []
    for id_num in list_of_ids:
        with open(data_dir+ f'{id_num}_data.p', 'rb') as f:
            dats = pickle.load(f)
            reward_info = dats['total_reward'][0:cutoff]
            results.append(reward_info)

    pp = np.vstack(results)
    avg_ = rm(np.mean(pp,axis=0),smoothing)
    std_ = rm(np.std(pp, axis=0), smoothing)
    a_s, s_s = [], []
    for xx in range(len(pp)):
        rr = pp[xx]
        smoothed_rr = rm(rr, smoothing)
        a_s.append(np.mean(smoothed_rr))
        s_s.append(np.std(smoothed_rr))

    return avg_, std_, np.asarray(a_s), np.asarray(s_s)


def avg_performance_over_envs_relative(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,save=False,savename='',plot_title='',legend=False,ref_gb=None, **kwargs):
    if save:
        if savename=='':
            raise Exception('Must pass argument to savename to specify title to save plot')

    cache_limits = analysis_specs['cache_limits']
    avg_max_rwd  = analysis_specs['avg_max_rwd']

    convert_rep_to_color = kwargs.get('colors',plot_specs['rep_colors'])
    labels_for_plot      = kwargs.get('labels',plot_specs['labels'])

    fig, ax = plt.subplots(len(envs_to_plot),2,figsize=(2*(len(reps_to_plot)+2),3*len(envs_to_plot)), sharex='col', gridspec_kw={'width_ratios': [1, 1]})
    ## rows = different environments
    ## column 0 = env grid
    ## column 1 = performance comparison across representations; cache sizes
    bar_width = 0.75
    for i, env in enumerate(envs_to_plot):
        if env[-2:] == '51':
            rwd_colrow= (16,9)
        else:
            rwd_colrow=(14,14)

        rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
        ax[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[i,0].set_aspect('equal')
        ax[i,0].add_patch(rect)
        ax[i,0].get_xaxis().set_visible(False)
        ax[i,0].get_yaxis().set_visible(False)
        ax[i,0].invert_yaxis()

        normalization_factor = avg_max_rwd[env]

        for r, rep in enumerate(reps_to_plot):
            v_list = list(gb.get_group((env, rep, cache_limits[env][100])))
            hundo_p_avg,hundo_p_sem  = data_avg_std(v_list,normalization_factor=normalization_factor,cutoff=5000)

            for j, pct in enumerate(pcts_to_plot):
                v_list = list(gb.get_group((env, rep, cache_limits[env][pct])))
                print(env, rep, pct, len(v_list))
                avgs,stds = data_avg_std(v_list,normalization_factor=normalization_factor,cutoff=5000)
                avg_cos = np.mean(avgs)
                sem_cos = np.mean(stds)#/np.sqrt(len(stds))

                ax[i,1].bar(r*len(pcts_to_plot)+bar_width*j,avg_cos-hundo_p_avg,width=bar_width, color=convert_rep_to_color[rep], alpha=pct/100)

        #ax[i,1].set_ylim([0,1.])
        #ax[i,1].set_yticks(np.arange(0,1.25,0.25))
        #ax[i,1].set_yticklabels([0,'',50,'',100,])
        ax[i,1].set_ylabel('Performance \n(% Optimal)')

        if ref_gb is not None:
            v_list = list(ref_gb.get_group(env))
            avg, std = data_avg_std(v_list,normalization_factor=normalization_factor)
            ax[i,1].axhline(avg,c='gray',linestyle=':',alpha=0.5)

    ax[i,1].set_xticks(np.arange(0,len(pcts_to_plot)*len(reps_to_plot),len(pcts_to_plot))+bar_width*0.5*j)
    ax[i,1].set_xticklabels([labels_for_plot[x] for x in reps_to_plot],rotation=0)
    ax[i,1].set_xlabel('State Encoding')

    if legend=='reps':
        legend_patch_list = []
        for rep in reps_to_plot:
            legend_patch_list.append(mpatches.Patch(color=convert_rep_to_color[rep], label=labels_for_plot[rep]))
        plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.1), loc='lower center', ncol=len(legend_patch_list),title='State Representation')
    elif legend=='pcts':
        legend_patch_list = []
        for pct in pcts_to_plot:
            legend_patch_list.append(mpatches.Patch(color='gray',label=f'{pct}',alpha=pct/100))
            plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.16), loc='lower center', ncol=len(legend_patch_list), title='Episodic Memory Capacity (%)')
    else:
        if plot_title=='':
            print('No title passed to arg plot_title')
        ax[0,1].set_title(plot_title)
    if save:
        format = kwargs.get('format','svg')
        plt.savefig(f'../figures/CH2/{savename}.{format}', format=format)
    plt.show()

def compare_avg_performance_lineplot(probe_gb,rand_gb,env,reps_to_plot,pcts_to_plot,grid,save=False,savename='',plot_title='',legend=False):
    gbs = [rand_gb,probe_gb]
    version = env[-2:-1]
    if save:
        if savename=='':
            raise Exception('Must pass argument to savename to specify title to save plot')

    cache_limits = analysis_specs['cache_limits']
    avg_max_rwd  = analysis_specs['avg_max_rwd']

    convert_rep_to_color = plot_specs['rep_colors']
    labels_for_plot      = plot_specs['labels']

    fig, ax = plt.subplots(len(reps_to_plot),2,figsize=(10,len(reps_to_plot)*2), sharey='col',sharex='col',gridspec_kw={'width_ratios': [1, 3]})
    width=0.45
    if env[-2:] == '51':
        rwd_colrow= (16,9)
    else:
        rwd_colrow=(14,14)

    rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
    ax[0,0].pcolor(grid,cmap='bone_r',edgecolors='k', linewidths=0.1)
    ax[0,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
    ax[0,0].set_aspect('equal')
    ax[0,0].add_patch(rect)
    ax[0,0].get_xaxis().set_visible(False)
    ax[0,0].get_yaxis().set_visible(False)
    ax[0,0].invert_yaxis()

    norm = avg_max_rwd[env]
    data_dict ={}

    for r, rep in enumerate(reps_to_plot):
        if r ==0:
            pass
        else:
            ax[r,0].axis('off')
        # ax[0,j] plot average performance with error bars
        # ax[1,j] plot variance of differnt rep types
        data_dict[rep] = {'rand':[[],[]], 'oldest':[[],[]]}
        for j, pct in enumerate(pcts_to_plot):
            for g, gb in enumerate(gbs):
                v_list = list(gb.get_group((env, rep, cache_limits[env][pct])))
                print(env, rep, pct, v_list)
                avg_, std_ = data_avg_std(v_list,cutoff=5000,normalization_factor=norm)
                avg_cos, std_cos = np.mean(avg_), np.mean(std_)
                if g ==0:
                    data_dict[rep]['rand'][0].append(avg_cos)
                    data_dict[rep]['rand'][1].append(std_cos)
                else:
                    data_dict[rep]['oldest'][0].append(avg_cos)
                    data_dict[rep]['oldest'][1].append(std_cos)

        xs = pcts_to_plot
        ax[r,1].errorbar(xs, data_dict[rep]['oldest'][0],yerr=data_dict[rep]['oldest'][1],color=convert_rep_to_color[rep],linestyle='-',capsize=8)
        ax[r,1].errorbar(xs, data_dict[rep]['rand'][0],yerr=data_dict[rep]['rand'][1],color='gray',alpha=0.5,linestyle=':',capsize=8)
        ax[r,1].set_xlim([105,20])

        ax[r,1].set_ylim([0,1.2])
        ax[r,1].set_yticks(np.arange(0,1.5,0.25))
        right = 1
        top = 0.98
        ax[r,1].text(right, top, f'{labels_for_plot[rep]}', horizontalalignment='right', verticalalignment='top', transform=ax[r,1].transAxes)
        ax[r,1].set_yticklabels([0,'',50,'',100,''])
        ax[r,1].set_ylabel(f'Performance \n(% Optimal)')

    ax[r,1].set_xlabel('Memory Capacity (%)')

    p_rand = Line2D([0], [0], color='gray', lw=2, alpha=0.5, linestyle=":",label='Random Entry') # mpatches.Patch(fill=False,edgecolor='gray',alpha=1, hatch='///',label='Random Entry')
    p_old  = Line2D([0], [0], color='black', lw=2,linestyle="-",label='Oldest Entry')#mpatches.Patch(color='gray',alpha=1, label='Oldest Entry')
    plt.legend(handles=[p_rand,p_old], bbox_to_anchor=(0.5, len(reps_to_plot)*1.16), loc='lower center', ncol=2, title='Forgetting Rule')
    if save:
        format = 'svg'
        plt.savefig(f'../figures/CH2/compare_rand_forgetting_{version}.{format}', format=format)
    plt.show()

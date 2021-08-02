import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from Analysis.analysis_utils import plot_specs, analysis_specs, LINCLAB_COLS
from Analysis.analysis_utils import get_grids, structured_unstructured

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def d_r_success_fail(list_of_ids):
    success, fail = [[],[],[]],[[],[],[]]
    for j in range(len(list_of_ids)):
        su, fa = [[],[]], [[],[]]
        run_id = list_of_ids[j]
        with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
            data = pickle.load(f)

        n_trials = len(data['total_reward'])
        for i in range(1,n_trials):
            dist_returns = data['dist_rtn'][i]

            states_              = dist_returns[0]
            reconstructed_states = dist_returns[1]
            ec_distances         = dist_returns[2]
            computed_returns     = dist_returns[3]


            avg_dist = np.mean(ec_distances)
            avg_rtrn = np.mean(computed_returns)

            if data['total_reward'][i] < -2.49:
                fa[0].append(avg_rtrn)
                fa[1].append(avg_dist)
            else:
                su[0].append(avg_rtrn)
                su[1].append(avg_dist)
        if len(su[0])==0:
            pass
        else:
            success[0].append(np.mean(su[0]))
            success[1].append(np.mean(su[1]))
            success[2].append(len(su[1]))
        if len(fa[0])==0:
            pass
        else:
            fail[0].append(np.mean(fa[0]))
            fail[1].append(np.mean(fa[1]))
            fail[2].append(len(fa[1]))

    return success, fail

def success_fail_plus(list_of_ids):
    success, fail = [[],[],[]],[[],[],[]]
    for j in range(len(list_of_ids)):
        su, fa = [[],[]], [[],[]]
        run_id = list_of_ids[j]
        with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
            data = pickle.load(f)

        n_trials = len(data['total_reward'])
        for i in range(1,n_trials):
            dist_returns = data['dist_rtn'][i]

            states_              = dist_returns[0]
            reconstructed_states = dist_returns[1]
            ec_distances         = dist_returns[2]
            computed_returns     = dist_returns[3]


            avg_dist = np.mean(ec_distances)
            avg_rtrn = np.mean(computed_returns)
            prop_state_matches = sum(states_==reconstructed_states) / len(states_)
            non_match_distances = ec_distances[states_!=reconstructed_states]
            if len(non_match_distances)==0:
                non_match_dist = np.nan
            else:
                non_match_dist = np.mean(non_match_distances)

            if data['total_reward'][i] < -2.49:
                fa[0].append(prop_state_matches)
                fa[1].append(non_match_dist)
            else:
                su[0].append(prop_state_matches)
                su[1].append(non_match_dist)
        if len(su[0])==0:
            pass
        else:
            success[0].append(np.mean(su[0]))
            success[1].append(np.nanmean(su[1]))
            success[2].append(len(su[1]))
        if len(fa[0])==0:
            pass
        else:
            fail[0].append(np.mean(fa[0]))
            fail[1].append(np.nanmean(fa[1]))
            fail[2].append(len(fa[1]))

    return success, fail

def plot_success_failure_bars(gb,envs_to_plot,reps_to_plot,pcts_to_plot,save=False,savename='',plot_title='',legend=False,**kwargs):
    if save:
        if savename=='':
            raise Exception('Must pass argument to savename to specify title to save plot')

    convert_rep_to_color = kwargs.get('colors',plot_specs['rep_colors'])
    bar_width = 0.3
    fig, ax = plt.subplots(len(envs_to_plot), 3,figsize=(12,15),sharex='col',sharey='col', gridspec_kw={'height_ratios': [1, 1, 1, 1]})
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
        for j, rep in enumerate(reps_to_plot):
            for k, pct in enumerate(pcts_to_plot):
                cache_size = int(cache_limits[env][100]*(pct/100))
                print(cache_size/cache_limits[env][100])
                list_of_ids = list(gb.get_group((env, rep, cache_size)))
                print(list_of_ids)
                s, f = d_r_success_fail(list_of_ids)
                print(env, rep,pct, len(s[1]), len(f[1]), len(s[1])+len(f[1]))
                ax[i,1].bar(k+j*bar_width,np.mean(s[1]),yerr=np.std(s[1]),width=bar_width,color=convert_rep_to_color[rep])
                ax[i,2].bar(k+j*bar_width,np.mean(f[1]),yerr=np.std(f[1]),width=bar_width,color=convert_rep_to_color[rep])
                #ax[i,1].text(k+j*bar_width,0.5,f'{len(s[1])}',fontsize=6, rotation=90)
                #ax[i,2].text(k+j*bar_width,0.5,f'{len(f[1])}',fontsize=6, rotation=90)
                ax[i,2].set_yticklabels([])

        ax[i,1].set_ylabel('Average Distance\n of Closest Memory')

    for f in range(2):
        ax[i,f+1].set_ylim([0,1])
        ax[i,f+1].set_xticks(np.arange(3)+bar_width/2)
        ax[i,f+1].set_xticklabels([75,50,25])
        ax[i,f+1].set_xlabel('Memory Capacity (%)')

    ax[0,1].set_title('Successful Trials',fontsize=12)
    ax[0,2].set_title('Failed Trials',fontsize=12)

    if legend=='reps':
        legend_patch_list = []
        for rep in reps_to_plot:
            legend_patch_list.append(mpatches.Patch(color=convert_rep_to_color[rep], label=labels_for_plot[rep]))
        plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower right', ncol=len(legend_patch_list),title='State Representation')
    elif legend=='pcts':
        legend_patch_list = []
        for pct in pcts_to_plot:
            legend_patch_list.append(mpatches.Patch(color='gray',label=f'{pct}',alpha=pct/100))
            plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower right', ncol=len(legend_patch_list), title='Episodic Memory Capacity (%)')
    if save:
        format = kwargs.get('format','svg')
        plt.savefig(f'../figures/CH2/{savename}.{format}', format=format)
    plt.show()

def plot_success_failure_violins(gb,envs_to_plot,reps_to_plot,pcts_to_plot,save=False,savename='',plot_title='',legend=False,**kwargs):
    if save:
        if savename=='':
            raise Exception('Must pass argument to savename to specify title to save plot')

    convert_rep_to_color = kwargs.get('colors',plot_specs['rep_colors'])
    bar_width = 0.3
    fig, ax = plt.subplots(len(envs_to_plot), len(reps_to_plot)+1,figsize=(12,15),sharex='col',sharey='col', gridspec_kw={'height_ratios': np.ones(len(envs_to_plot))})
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
        violin_width=5
        for j, rep in enumerate(reps_to_plot):
            dats_s = []
            dats_f = []
            for k, pct in enumerate(pcts_to_plot):
                cache_size = int(cache_limits[env][100]*(pct/100))
                print(cache_size/cache_limits[env][100])
                list_of_ids = list(gb.get_group((env, rep, cache_size)))
                print(list_of_ids)
                s, f = d_r_success_fail(list_of_ids)
                dats_s.append(s[1])
                dats_f.append(f[1])
                print(env, rep,pct, len(s[1]), len(f[1]), len(s[1])+len(f[1]))

            s_body = ax[i,1].violinplot(positions=np.asarray(pcts_to_plot)+violin_width*(1/2-j),dataset=dats_s,vert=True, widths=violin_width,showmeans=False)
            f_body = ax[i,2].violinplot(positions=np.asarray(pcts_to_plot)+violin_width*(1/2-j),dataset=dats_f,vert=True, widths=violin_width,showmeans=False)
            for violinkey in s_body.keys():
                if violinkey == 'bodies':
                    for b in s_body['bodies']:
                        b.set_color(convert_rep_to_color[rep])
                        b.set_alpha(75/100)
                else:
                    s_body[violinkey].set_color(convert_rep_to_color[rep])
            for violinkey in f_body.keys():
                if violinkey == 'bodies':
                    for b in f_body['bodies']:
                        b.set_color(convert_rep_to_color[rep])
                        b.set_alpha(25/100)
                else:
                    f_body[violinkey].set_color(convert_rep_to_color[rep])

        ax[i,1].set_ylabel('Average Distance\n of Closest Memory')

    for f in range(2):
        ax[i,f+1].set_ylim([0,1])
        ax[i,f+1].set_xticks([75,50,25])
        ax[i,f+1].set_xlim([90+violin_width+2, 25-violin_width-2])
        ax[i,f+1].set_xticklabels([75,50,25])
        ax[i,f+1].set_xlabel('Memory Capacity (%)')

    ax[0,1].set_title('Successful Trial',fontsize=12)
    ax[0,2].set_title('Failed Trial',fontsize=12)

    if legend=='reps':
        legend_patch_list = []
        for rep in reps_to_plot:
            legend_patch_list.append(mpatches.Patch(color=convert_rep_to_color[rep], label=labels_for_plot[rep]))
        plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower right', ncol=len(legend_patch_list),title='State Representation')
    elif legend=='pcts':
        legend_patch_list = []
        for pct in pcts_to_plot:
            legend_patch_list.append(mpatches.Patch(color='gray',label=f'{pct}',alpha=pct/100))
            plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower right', ncol=len(legend_patch_list), title='Episodic Memory Capacity (%)')
    elif legend=='SF':
        legend_patch_list = []
        legend_patch_list.append(mpatches.Patch(color='gray',label=f'Successful Trial',alpha=75/100))
        legend_patch_list.append(mpatches.Patch(color='gray',label=f'Failed Trial',alpha=25/100))
        plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower right', ncol=len(legend_patch_list))

    if save:
        format = kwargs.get('format','svg')
        plt.savefig(f'../figures/CH2/{savename}.{format}', format=format)
    plt.show()

def plot_success_failure_lines(gb,envs_to_plot,reps_to_plot,pcts_to_plot,save=False,savename='',plot_title='',legend=False,**kwargs):
    if save:
        if savename=='':
            raise Exception('Must pass argument to savename to specify title to save plot')

    convert_rep_to_color = kwargs.get('colors',plot_specs['rep_colors'])
    bar_width = 0.3
    fig, ax = plt.subplots(len(envs_to_plot), len(reps_to_plot)+1,figsize=(12,15),sharex='col',sharey='col', gridspec_kw={'height_ratios': np.ones(len(envs_to_plot))})
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
        violin_width=5
        for j, rep in enumerate(reps_to_plot):
            dats_s = []
            dats_f = []
            for k, pct in enumerate(pcts_to_plot):
                cache_size = int(cache_limits[env][100]*(pct/100))
                print(cache_size/cache_limits[env][100])
                list_of_ids = list(gb.get_group((env, rep, cache_size)))
                print(list_of_ids)
                s, f = d_r_success_fail(list_of_ids)
                dats_s.append(np.mean(s[1]))
                dats_f.append(np.mean(f[1]))
                print(env, rep,pct, len(s[1]), len(f[1]), len(s[1])+len(f[1]))

            ax[i,1].errorbar(pcts_to_plot,dats_s,marker='o',color=convert_rep_to_color[rep])
            ax[i,2].errorbar(pcts_to_plot,dats_f,marker='o',color=convert_rep_to_color[rep])


        ax[i,1].set_ylabel('Average Distance\n of Closest Memory')

    for f in range(2):
        ax[i,f+1].set_ylim([0,1])
        ax[i,f+1].set_xticks([90,80,70,60,50,40,30,20,10])
        ax[i,f+1].set_xlim([100, 0])
        ax[i,f+1].set_xticklabels([90,80,70,60,50,40,30,20,10])
        ax[i,f+1].set_xlabel('Memory Capacity (%)')

    ax[0,1].set_title('Successful Trial',fontsize=12)
    ax[0,2].set_title('Failed Trial',fontsize=12)

    if legend=='reps':
        legend_patch_list = []
        for rep in reps_to_plot:
            legend_patch_list.append(mpatches.Patch(color=convert_rep_to_color[rep], label=labels_for_plot[rep]))
        plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower right', ncol=len(legend_patch_list),title='State Representation')
    elif legend=='pcts':
        legend_patch_list = []
        for pct in pcts_to_plot:
            legend_patch_list.append(mpatches.Patch(color='gray',label=f'{pct}',alpha=pct/100))
            plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower right', ncol=len(legend_patch_list), title='Episodic Memory Capacity (%)')
    elif legend=='SF':
        legend_patch_list = []
        legend_patch_list.append(mpatches.Patch(color='gray',label=f'Successful Trial',alpha=75/100))
        legend_patch_list.append(mpatches.Patch(color='gray',label=f'Failed Trial',alpha=25/100))
        plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower right', ncol=len(legend_patch_list))

    if save:
        format = kwargs.get('format','svg')
        plt.savefig(f'../figures/CH2/{savename}.{format}', format=format)
    plt.show()

def plot_sf_difference_lines(gb,envs_to_plot,reps_to_plot,pcts_to_plot,save=False,savename='',plot_title='',legend=False,**kwargs):
    if save:
        if savename=='':
            raise Exception('Must pass argument to savename to specify title to save plot')

    convert_rep_to_color = kwargs.get('colors',plot_specs['rep_colors'])
    bar_width = 0.3
    fig, ax = plt.subplots(len(envs_to_plot), 2,figsize=(10,15),sharex='col',sharey='col', gridspec_kw={'height_ratios': [1, 1, 1, 1]})
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
        for j, rep in enumerate(reps_to_plot):
            difs_to_plot = []
            errs_to_plot = []
            for k, pct in enumerate(pcts_to_plot):
                cache_size = int(cache_limits[env][100]*(pct/100))
                print(cache_size/cache_limits[env][100])
                list_of_ids = list(gb.get_group((env, rep, cache_size)))
                if env[3:]=='v11' and rep=='structured' and pct=='75':
                    print('hello')

                s, f = d_r_success_fail(list_of_ids)

                dif_of_means = np.mean(f[1]) - np.mean(s[1])
                rat_of_stds  = (np.std(f[1])/np.sqrt(len(f[1])))/(np.std(s[1])/np.sqrt(len(s[1])))

                print(env[-3:], rep,pct, f'avg d fail: {np.mean(f[1])}, avg d succ: {np.mean(s[1])} diff of means: {dif_of_means}')

                difs_to_plot.append(dif_of_means)
                errs_to_plot.append(rat_of_stds)
            print(env[-3:],rep,pcts_to_plot, difs_to_plot)
            ax[i,1].errorbar(pcts_to_plot,difs_to_plot,marker='o',color=convert_rep_to_color[rep])
            #ax[i,1].set_yticklabels([])

        ax[i,1].set_ylabel('Average Distance\n of Closest Memory')

    ax[i,1].set_ylim([0,1])
    ax[i,1].set_xlim([80,20])
    ax[i,1].set_xticks([75,50,25])
    ax[i,1].set_xlabel('Memory Capacity (%)')

    ax[0,1].set_title('Difference (fail-success) Trials',fontsize=12)

    if legend=='reps':
        legend_patch_list = []
        for rep in reps_to_plot:
            legend_patch_list.append(mpatches.Patch(color=convert_rep_to_color[rep], label=labels_for_plot[rep]))
        plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower right', ncol=len(legend_patch_list),title='State Representation')
    elif legend=='pcts':
        legend_patch_list = []
        for pct in pcts_to_plot:
            legend_patch_list.append(mpatches.Patch(color='gray',label=f'{pct}',alpha=pct/100))
            plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower right', ncol=len(legend_patch_list), title='Episodic Memory Capacity (%)')
    if save:
        format = kwargs.get('format','svg')
        plt.savefig(f'../figures/CH2/{savename}.{format}', format=format)
    plt.show()

def plot_success_failure_frequency(gb,envs_to_plot,reps_to_plot,pcts_to_plot,save=False,savename='',plot_title='',legend=False,**kwargs):
    if save:
        if savename=='':
            raise Exception('Must pass argument to savename to specify title to save plot')

    convert_rep_to_color = kwargs.get('colors',plot_specs['rep_colors'])
    bar_width = 0.3
    fig, ax = plt.subplots(len(envs_to_plot), 4,figsize=(22,15),sharex='col',sharey='col', gridspec_kw={'height_ratios': np.ones(len(envs_to_plot))})
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
        for j, rep in enumerate(reps_to_plot):
            success_incidence = []
            prop_matches = [[],[]]
            errors = [[],[]]

            dist_meas = [[],[]]
            for k, pct in enumerate(pcts_to_plot):
                print(env, rep,pct)

                cache_size = int(cache_limits[env][100]*(pct/100))
                list_of_ids = list(gb.get_group((env, rep, cache_size,'chebyshev')))
                print(len(list_of_ids), cache_size)
                s, f = success_fail_plus(list_of_ids)

                # total incidence of successful trials
                num_success = np.mean(s[2])
                total_num   = np.mean(s[2])+np.mean(f[2])
                success_incidence.append(num_success/total_num)

                # proportion of memory events for which state is in memory
                prop_matches[0].append(np.mean(s[0]))
                errors[0].append(np.std(s[0]))
                prop_matches[1].append(np.mean(f[0]))
                errors[1].append(np.std(f[0]))

                # average distance
                dist_meas[0].append(np.mean(s[1]))
                dist_meas[1].append(np.mean(f[1]))


            ax[i,1].plot(pcts_to_plot,success_incidence,marker='o',color=convert_rep_to_color[rep])
            ax[i,2].errorbar(pcts_to_plot,prop_matches[0],marker='o',linestyle='-',color=convert_rep_to_color[rep])#yerr=errors[0],
            ax[i,2].errorbar(pcts_to_plot,prop_matches[1],marker='o',linestyle=':',color=convert_rep_to_color[rep],alpha=0.5)#yerr=errors[1],
            ax[i,3].plot(pcts_to_plot,dist_meas[0],marker='o',linestyle='-',color=convert_rep_to_color[rep])
            ax[i,3].plot(pcts_to_plot,dist_meas[1],marker='o',linestyle=':',color=convert_rep_to_color[rep],alpha=0.5)

        ax[i,1].set_ylabel('% of All Episodes')
        ax[i,2].set_ylabel('% of All Events')
        ax[i,3].set_ylabel('Average Distance')

    for p in range(2):
        ax[i,p+1].set_ylim([0,1.1])
        ax[i,p+1].set_yticks(np.arange(0,1.25,.25))
        ax[i,p+1].set_yticklabels([0,25,50,75,100])
        ax[i,p+1].set_xlim([100,0])
        ax[i,p+1].set_xlabel('Memory Capacity (%)')

    ax[i,3].set_ylim([0.5,1.0])
    ax[i,3].set_xlim([100,0])
    ax[i,3].set_xlabel('Memory Capacity (%)')

    ax[0,1].set_title('Successful Episodes',fontsize=10)
    ax[0,2].set_title('Events with Exact Match in Memory',fontsize=10)
    ax[0,3].set_title('Nearest Non-Match Memory to Probe State',fontsize=8)

    if legend=='reps':
        legend_patch_list = []
        for rep in reps_to_plot:
            legend_patch_list.append(mpatches.Patch(color=convert_rep_to_color[rep], label=labels_for_plot[rep]))

        legend_patch_list.append(Line2D([0], [0], color='gray', lw=2, alpha=0.5, linestyle=":",label='Failed Trial'))
        legend_patch_list.append(Line2D([0], [0], color='black', lw=2, alpha=0.5, linestyle="-",label='Successful Trial'))

        plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower right', ncol=len(legend_patch_list),title='State Representation')
    elif legend=='pcts':
        legend_patch_list = []
        for pct in pcts_to_plot:
            legend_patch_list.append(mpatches.Patch(color='gray',label=f'{pct}',alpha=pct/100))
            plt.legend(handles=legend_patch_list, bbox_to_anchor=(0.5, len(envs_to_plot)*1.18), loc='lower right', ncol=len(legend_patch_list), title='Episodic Memory Capacity (%)')
    if save:
        format = kwargs.get('format','svg')
        plt.savefig(f'../figures/CH2/{savename}.{format}', format=format)
    plt.show()

def plot_success_failure_line():
    fig, ax = plt.subplots(len(envs_to_plot), 2,sharex='col', gridspec_kw={'width_ratios': [1, 4]})
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
        for j, rep in enumerate(reps_to_plot):
            s_avg, f_avg = [],[]
            s_sem, f_sem = [],[]
            for k, pct in enumerate(pcts_to_plot):
                cache_size = int(cache_limits[env][100]*(pct/100))
                print(cache_size/cache_limits[env][100])
                list_of_ids = list(gb.get_group((env, rep, cache_size,'chebyshev')))

                s, f = d_r_success_fail(list_of_ids)
                s_avg.append(np.mean(s[1]))
                s_sem.append(np.std(s[1])/np.sqrt(len(s[1])))
                f_avg.append(np.mean(f[1]))
                f_sem.append(np.std(f[1])/np.sqrt(len(f[1])))

            ax[i,1].errorbar(pcts_to_plot,s_avg,yerr=s_sem,c=convert_rep_to_color[rep],marker='o')
            ax[i,1].errorbar(pcts_to_plot,f_avg,yerr=f_sem,c=convert_rep_to_color[rep],marker='s',linestyle=':')
        ax[i,1].set_xlim([80,20])
        ax[i,1].set_xticks(pcts_to_plot)
    plt.show()


# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'ec_avg_dist_rtn.csv')
df['representation'] = df['representation'].apply(structured_unstructured)

gb = df.groupby(['env_name','representation','EC_cache_limit','EC_similarity_meas'])["save_id"]

convert_rep_to_color = {'structured':LINCLAB_COLS['red'], 'unstructured':LINCLAB_COLS['blue']}
labels_for_plot = {'structured':'structured','unstructured':'unstructured'}
cache_limits = analysis_specs['cache_limits']


envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
reps_to_plot = ['unstructured','structured']
pcts_to_plot = [90,80,70,60,50,40,30,20,10]
grids = get_grids(envs_to_plot)

plot_success_failure_frequency(gb,envs_to_plot,reps_to_plot,pcts_to_plot,colors=convert_rep_to_color,legend='reps',save=True, savename='success_failure_nonmatchdist')
#plot_success_failure_lines(gb, envs_to_plot, reps_to_plot, pcts_to_plot,legend='reps',colors=convert_rep_to_color,save=True,savename='success_failure_SU_lines')
#plot_success_failure_violins(gb, envs_to_plot, reps_to_plot, pcts_to_plot,legend='reps',colors=convert_rep_to_color,save=True,savename='success_failure_SU_violins_alt')

## sandbox
## gw41 - 60% capacity
## unstructured: '9de09039-a9c0-43b6-aa95-b98d60886ff2'
## structured: '2b857409-e5d4-427c-92fa-b04437725952'

run_id = '2b857409-e5d4-427c-92fa-b04437725952'

with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
    data = pickle.load(f)

n_trials = len(data['total_reward'])
for i in range(100,103):
    dist_returns = data['dist_rtn'][i]
    states_              = dist_returns[0]
    reconstructed_states = dist_returns[1]
    ec_distances         = dist_returns[2]
    computed_returns     = dist_returns[3]

    print(ec_distances[states_!=reconstructed_states])


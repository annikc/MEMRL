import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gym
from modules.Agents.RepresentationLearning.learned_representations import onehot, random, place_cell, sr, latents
from scipy.spatial.distance import pdist, cdist,squareform
from Analysis.analysis_utils import get_grids
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from modules.Utils import softmax
from Analysis.analysis_utils import make_env_graph, compute_graph_distance_matrix, LINCLAB_COLS

rep_type = 'random'

envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
grids = get_grids(envs_to_plot)

color_map = {'place_cell':LINCLAB_COLS['red'],'sr':LINCLAB_COLS['red'], 'unstructured':LINCLAB_COLS['blue']}

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

cmaps_mappings = {'random':bluecmp, 'onehot':bluecmp, 'place_cell':redcmp, 'sr':redcmp, 'latents':redcmp}
cmaps_mappings_r = {'random':bluecmp_r, 'onehot':bluecmp_r, 'place_cell':redcmp_r, 'sr':redcmp_r, 'latents':redcmp_r}

def plot_squares(what_to_plot='all_state_rep',current_cmap = 'viridis_r'):
    for test_env_name in [envs_to_plot[1]]:
        sim_ind = 169
        envno = envs_to_plot.index(test_env_name)
        #test_env_name = envs_to_plot[env_id]
        # make new env to run test in
        env = gym.make(test_env_name)
        plt.close()
        #plot_world(env,plotNow=True,scale=0.4,states=True)


        rep_types = {'random':random,'onehot':onehot, 'place_cell':place_cell, 'sr':sr}#, 'latents':latents}
        fig, ax = plt.subplots(1, len(list(rep_types.items()))+1, figsize=(14,2))

        if test_env_name[-2:] == '51':
            rwd_colrow= (16,9)
        else:
            rwd_colrow=(14,14)

        rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
        agt_colrow = (env.oneD2twoD(sim_ind)[1]+0.5,env.oneD2twoD(sim_ind)[0]+0.5)
        circ = plt.Circle(agt_colrow,radius=0.3,color='blue')
        ax[0].pcolor(grids[envno],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[0].set_aspect('equal')
        ax[0].add_patch(rect)
        ax[0].add_patch(circ)
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[0].invert_yaxis()

        for j, rep_type in enumerate(rep_types.keys()):
            relative_path_to_data = '../../Data/' # from within Tests/CH1
            if rep_type == 'latents':
                conv_ids = {'gridworld:gridworld-v1':'c34544ac-45ed-492c-b2eb-4431b403a3a8',
                            'gridworld:gridworld-v3':'32301262-cd74-4116-b776-57354831c484',
                            'gridworld:gridworld-v4':'b50926a2-0186-4bb9-81ec-77063cac6861',
                            'gridworld:gridworld-v5':'15b5e27b-444f-4fc8-bf25-5b7807df4c7f'}
                run_id = conv_ids[test_env_name[:-1]]
                agent_path = relative_path_to_data+f'agents/{run_id}.pt'
                state_reps, representation_name, input_dims, _ = latents(env, agent_path)
            else:
                state_reps, representation_name, input_dims, _ = rep_types[rep_type](env)
            reps_as_matrix = np.zeros((400,400))
            reps_as_matrix[:]  = np.nan

            for ind, (k,v) in enumerate(state_reps.items()):
                reps_as_matrix[k] = v
                if k in env.obstacle:
                    reps_as_matrix[k,:] = np.nan


            RS = squareform(pdist(reps_as_matrix,metric='chebyshev'))
            for state2d in env.obstacle:
                RS[state2d,:] = np.nan
                RS[:,state2d] = np.nan


            #plot representation of a single state
            if what_to_plot == 'single_state_rep':
                #current_cmap = cmaps_mappings_r[rep_type]
                a = ax[j+1].imshow((reps_as_matrix[sim_ind]/np.nanmax(reps_as_matrix)).reshape(env.shape),vmin=0, vmax=1, cmap=current_cmap)

            elif what_to_plot == 'all_state_rep':
                #current_cmap = cmaps_mappings_r[rep_type]
                a = ax[j+1].imshow(reps_as_matrix/np.nanmax(reps_as_matrix),vmin=0, vmax=1, cmap=current_cmap)

            elif what_to_plot == 'all_state_sim_sliced':
                #current_cmap = cmaps_mappings_r[rep_type]
                sliced = RS.copy()

                print(np.argwhere(np.isnan(sliced)))
                for state1d in reversed(env.obstacle):
                    sliced = np.delete(sliced,state1d,0)
                    sliced = np.delete(sliced,state1d,1)

                a = ax[j+1].imshow(sliced/np.nanmax(sliced), vmin=0, vmax=1,  cmap=current_cmap)

            elif what_to_plot == 'single_state_sim':
                #current_cmap = cmaps_mappings[rep_type]
                a = ax[j+1].imshow(RS[sim_ind].reshape(env.shape)/np.nanmax(RS),vmin=0, vmax=1,  cmap=current_cmap)

            elif what_to_plot == 'all_state_sim':
                #current_cmap = cmaps_mappings[rep_type]
                a = ax[j+1].imshow(RS/np.nanmax(RS), vmin=0, vmax=1,  cmap=current_cmap)

            if j == len(list(rep_types.keys()))-1:
                divider = make_axes_locatable(ax[j+1])
                cax = divider.append_axes('right',size='5%',pad=0.05)
                plt.colorbar(a,cax=cax)
            ax[j+1].get_xaxis().set_visible(False)
            ax[j+1].get_yaxis().set_visible(False)
        plt.savefig(f'../figures/CH2/{test_env_name[-3:]}_{what_to_plot}.svg')
        plt.show()

def plot_vectors(current_cmap='viridis', save=False):
    for test_env_name in [envs_to_plot[1]]:
        what_to_plot = 'all_state_rep'
        sim_ind = 169
        envno = envs_to_plot.index(test_env_name)
        #test_env_name = envs_to_plot[env_id]
        # make new env to run test in
        env = gym.make(test_env_name)
        plt.close()
        #plot_world(env,plotNow=True,scale=0.4,states=True)


        rep_types = {'random':random,'onehot':onehot, 'place_cell':place_cell, 'sr':sr}
        fig, ax = plt.subplots(1, len(rep_types.keys())+1, figsize=(14,10))

        if test_env_name[-2:] == '51':
            rwd_colrow= (16,9)
        else:
            rwd_colrow=(14,14)

        rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
        agt_colrow = (env.oneD2twoD(sim_ind)[1]+0.5,env.oneD2twoD(sim_ind)[0]+0.5)
        circ = plt.Circle(agt_colrow,radius=0.3,color='blue')
        ax[0].pcolor(grids[envno],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[0].set_aspect('equal')
        ax[0].add_patch(rect)
        ax[0].add_patch(circ)
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[0].invert_yaxis()

        for j, rep_type in enumerate(rep_types.keys()):
            relative_path_to_data = '../../Data/' # from within Tests/CH1
            if rep_type == 'latents':
                conv_ids = {'gridworld:gridworld-v1':'c34544ac-45ed-492c-b2eb-4431b403a3a8',
                            'gridworld:gridworld-v3':'32301262-cd74-4116-b776-57354831c484',
                            'gridworld:gridworld-v4':'b50926a2-0186-4bb9-81ec-77063cac6861',
                            'gridworld:gridworld-v5':'15b5e27b-444f-4fc8-bf25-5b7807df4c7f'}
                run_id = conv_ids[test_env_name[:-1]]
                agent_path = relative_path_to_data+f'agents/{run_id}.pt'
                state_reps, representation_name, input_dims, _ = latents(env, agent_path)
            else:
                state_reps, representation_name, input_dims, _ = rep_types[rep_type](env)
            reps_as_matrix = np.zeros((400,400))
            reps_as_matrix[:]  = np.nan

            for ind, (k,v) in enumerate(state_reps.items()):
                reps_as_matrix[k] = v

            RS = squareform(pdist(reps_as_matrix,metric='chebyshev'))
            for state2d in env.obstacle:
                RS[state2d,:] = np.nan
                RS[:,state2d] = np.nan


            #plot representation of a single state
            #current_cmap = cmaps_mappings_r[rep_type]

            #a = ax[j+1].imshow((reps_as_matrix[sim_ind]/np.nanmax(reps_as_matrix[sim_ind])).reshape(400,1),vmin=0, vmax=1, cmap=current_cmap, aspect='auto')
            a = ax[j+1].imshow((reps_as_matrix[sim_ind]/np.nanmax(reps_as_matrix[sim_ind])).reshape(400,1),vmin=0, vmax=1, cmap=current_cmap, aspect='auto')

            if j == len(list(rep_types.keys()))-1:
                divider = make_axes_locatable(ax[j+1])
                cax = divider.append_axes('right',size='5%',pad=0.05)
                plt.colorbar(a,cax=cax)
            ax[j+1].get_xaxis().set_visible(False)
            ax[j+1].get_yaxis().set_visible(False)
        if save:
            plt.savefig(f'../figures/CH2/{test_env_name[-3:]}_vectors.svg')
        plt.show()

#plot_squares(what_to_plot='all_state_sim')

def get_graph_dist_from_state(envs_to_plot, sim_ind):
    graph_distances = []
    for i,test_env_name in enumerate(envs_to_plot):
        env=gym.make(test_env_name)
        plt.close()
        G = make_env_graph(env)
        gd = compute_graph_distance_matrix(G,env)
        dist_in_state_space = gd[sim_ind]
        graph_distances.append(dist_in_state_space)
    return graph_distances

def plot_dist_to_neighbours():
    for test_env_name in [envs_to_plot[1]]:
        sim_ind = 169
        envno = envs_to_plot.index(test_env_name)

        # make new env to run test in
        env = gym.make(test_env_name)
        plt.close()

        # make graph of env states
        G = make_env_graph(env)
        gd= compute_graph_distance_matrix(G,env)
        dist_in_state_space = np.delete(gd[sim_ind],sim_ind) #distance from sim ind to all other states

        rep_types = {'random':random, 'onehot':onehot, 'place_cell':place_cell, 'sr':sr}#, 'latents':latents}
        fig, ax = plt.subplots(1, len(list(rep_types.items()))+1, figsize=(14,2))

        if test_env_name[-2:] == '51':
            rwd_colrow= (16,9)
        else:
            rwd_colrow=(14,14)

        rect = plt.Rectangle(rwd_colrow, 1, 1, color='g', alpha=0.3)
        agt_colrow = (env.oneD2twoD(sim_ind)[1]+0.5,env.oneD2twoD(sim_ind)[0]+0.5)
        circ = plt.Circle(agt_colrow,radius=0.3,color='blue')
        ax[0].pcolor(grids[envno],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[0].set_aspect('equal')
        ax[0].add_patch(rect)
        ax[0].add_patch(circ)
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[0].invert_yaxis()

        cmap = linc_coolwarm_r #cm.get_cmap('coolwarm')



        for j, rep_type in enumerate(rep_types.keys()):
            relative_path_to_data = '../../Data/' # from within Tests/CH1
            if rep_type == 'latents':
                conv_ids = {'gridworld:gridworld-v1':'c34544ac-45ed-492c-b2eb-4431b403a3a8',
                            'gridworld:gridworld-v3':'32301262-cd74-4116-b776-57354831c484',
                            'gridworld:gridworld-v4':'b50926a2-0186-4bb9-81ec-77063cac6861',
                            'gridworld:gridworld-v5':'15b5e27b-444f-4fc8-bf25-5b7807df4c7f'}
                run_id = conv_ids[test_env_name[:-1]]
                agent_path = relative_path_to_data+f'agents/{run_id}.pt'
                state_reps, representation_name, input_dims, _ = latents(env, agent_path)
            else:
                state_reps, representation_name, input_dims, _ = rep_types[rep_type](env)
            reps_as_matrix = np.zeros((400,400))
            reps_as_matrix[:]  = np.nan

            for ind, (k,v) in enumerate(state_reps.items()):
                reps_as_matrix[k] = v
                if k in env.obstacle:
                    reps_as_matrix[k,:] = np.nan


            RS = squareform(pdist(reps_as_matrix,metric='chebyshev'))
            for state2d in env.obstacle:
                RS[state2d,:] = np.nan
                RS[:,state2d] = np.nan

            dist_in_rep_space = np.delete(RS[sim_ind],sim_ind)
            print(type(dist_in_rep_space))
            rgba = [cmap(x) for x in dist_in_rep_space/np.nanmax(dist_in_rep_space)]
            ax[j+1].scatter(dist_in_state_space,dist_in_rep_space,color='#b40426',alpha=0.2,linewidths=0.5 )
            ax[j+1].set_xlabel("D(s,s')")
            ax[j+1].set_ylim([0.4,1.1])
            if j != 0:
                ax[j+1].set_yticklabels([])
            else:
                ax[j+1].set_ylabel("D(R(s), R(s'))")

        #a = ax[j+2].imshow((dist_in_state_space/np.nanmax(dist_in_state_space)).reshape(env.shape), cmap=cmap)
        #plt.colorbar(a, ax=ax[j+2])
        plt.savefig('../figures/CH2/distance.svg')
        plt.show()

#plot_vectors(linc_coolwarm,save=True)
plot_dist_to_neighbours()
#plot_squares(what_to_plot='single_state_sim',current_cmap=linc_coolwarm_r)
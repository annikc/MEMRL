import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.cm as cmx
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

import os
from os import listdir
from os.path import isfile, join


import imageio



# =====================================
#        DEFINE PLOT FUNCTIONS        #
# =====================================

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def softmax(x,T=1):
	 """Compute softmax values for each sets of scores in x."""
	 e_x = np.exp((x - np.max(x))/T)
	 return e_x / e_x.sum(axis=0) # only difference

def plot_softmax(x, T=1):
	f, axarr = plt.subplots(2, sharex=True)
	axarr[0].bar(np.arange(len(x)), x)
	y = softmax(x, T)    
	axarr[1].bar(np.arange(len(x)), y) 
	plt.show()


def make_env_plots(maze, env=True, pc_map=False, pcs=None, pc_vec=False, save=False):
	grid = maze.grid
	useable_grid = maze.useable
	agent_loc = maze.cur_state
	
	if env: 
		# plot maze -- agent (blue) and reward (red)
		fig = plt.figure(0)

		axis  = fig.add_axes([0.05, 0.05, .9, .9]) # [left, bottom, width, height]

		ax = fig.gca()
		axis.pcolor(grid, cmap = 'bone', vmax =1, vmin = 0)

		for rwd_loc in maze.rwd_loc:
			rwd_v, rwd_h = rwd_loc
			ax.add_patch(plt.Circle((rwd_v+.5, rwd_h+.5), 0.35, fc='r'))
		
		agent_v, agent_h = agent_loc
		ax.add_patch(plt.Circle((agent_v+.5, agent_h+.5), 0.35, fc='b'))
		if maze.maze_type == 'tmaze':
			pass
			#for port in maze.possible_ports:
			#	ax.add_patch(plt.Circle(np.add(port, (0.5,0.5)), 0.25, fc='g'))

		#ax.invert_yaxis()
		#plt.colorbar()
		#if maze.y == maze.x: 
			#plt.axis('tight')
		ax.set_aspect('equal')
		if save:
			plt.savefig('{}environment.svg'.format(maze.maze_type), format='svg', pad_inches =2)
		
	if pc_map:
		cmap = plt.cm.jet
		vmin = 0
		vmax = max(pcs.activity(maze.cur_state)[0])
		cNorm  = colors.Normalize(vmin=0, vmax=vmax)
		scalarMap = cmx.ScalarMappable(norm = cNorm, cmap=cmap)
		# plot place cells 
		# circle radius given by fwhm of place cells (???)
		fig = plt.figure(1)
		ax  = fig.add_axes([0, 0.1, 0.6, 0.85]) # [left, bottom, width, height]
		axc = fig.add_axes([0.63, 0.1, 0.03, 0.85])

	
		for i in range(len(pcs.x)):

			colorVal = scalarMap.to_rgba(np.squeeze(pcs.activity(maze.cur_state))[i])
			ax.add_patch(patches.Circle((pcs.x[i], pcs.y[i]), 0.025, fc=colorVal, ec='none', alpha=0.5))
			ax.set_ylim([1,0])
		cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
		ax.axis('equal')
		if maze.y == maze.x:
			ax.axis('tight')
		#ax.grid('on')
		plt.show()

	if pc_vec:
		cmap = plt.cm.jet
		vmin = 0
		vmax = max(pcs.activity(maze.cur_state)[0])
		cNorm  = colors.Normalize(vmin=0, vmax=vmax)
		scalarMap = cmx.ScalarMappable(norm = cNorm, cmap=cmap)

		# plot input vector
		fig = plt.figure(2)
		ax  = fig.add_axes([0, 0.25, 0.7, 0.15]) # [left, bottom, width, height]
		axc = fig.add_axes([0, 0.1, 0.7, 0.07])

		ax.pcolor(pcs.activity(maze.cur_state), vmin=0, vmax=1, cmap=cmap)
		ax.set_yticklabels([''])
		cb2 = colorbar.ColorbarBase(axc, cmap = cmap, norm = cNorm, orientation='horizontal')
		plt.show()

def make_gif(mypath, mazetype):
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	intfiles = [int(f) for f in onlyfiles]
	intfiles.sort()
	if mazetype =='none':
		gifname = './valplots/gifs/grid{}.gif'.format(obs_rho)
	else: 
		gifname = './valplots/gifs/{}.gif'.format(mazetype)

	with imageio.get_writer(gifname, mode='I', duration=0.5) as writer:
				for filename in intfiles:
					image = imageio.imread(mypath+str(filename))
					writer.append_data(image)
	print( "Gif file saved at ", gifname)


def make_arrows(action, probability):
	if probability == 0: 
		dx, dy = 0, 0
		head_w, head_l = 0,0
	
	else: 
		if action == 0: #N
			dx = 0
			dy = -.25
		elif action == 1: #E
			dx = .25
			dy = 0
		elif action == 2: #W
			dx = -.25
			dy = 0
		elif action == 3: #S
			dx = 0
			dy = .25
		elif action == 4: #stay
			dx = -.1
			dy = -.1
		elif action ==5: #poke 
			dx = .1
			dy = .1
		
		head_w, head_l = 0.1, 0.1
		
	return dx,dy, head_w, head_l



class artist_instance:
	def __init__(self, xy=None, rad=None):
		self.xy=xy if xy is not None else (1,1)
		self.radius = rad if rad is not None else 0.1
		self.fc = 'w'
		self.ec = 'r'
	def art(self):
		return getattr(patches,'Circle')(xy=self.xy,radius=self.radius,fc=self.fc,ec=self.ec)

def print_value_maps(maze,val_maps,**kwargs):
	maptype = kwargs.get('type', 'value')
	value_min, value_max = kwargs.get('val_range', (np.nanmin(val_maps),np.nanmax(val_maps)))
	mazetype = maze.maze_type
	obs_rho = maze.rho
	#rwd_loc = maze.rwd_loc[0]
	maps = kwargs.get('maps', 'all')
	title = kwargs.get('title',None)
	save_dir = kwargs.get('save_dir', None)
	if title == None:
		if mazetype == 'none':
			if obs_rho != 0.0:
				save_string = './figures/grid_obs{}_{}map.svg'.format(obs_rho,maptype)
			else:
				save_string = './figures/{}.svg'.format(title.replace(" ",""))
		else: 
			save_string = './figures/{}_{}map.svg'.format(mazetype,maptype)
	else:
		if mazetype == 'none':
			if obs_rho !=0.0:
				save_string = './figures/grid_obs{}_{}.svg'.format(obs_rho,title.replace(" ",""))
			else:
				save_string = './figures/{}.svg'.format(title.replace(" ",""))
		else: 
			save_string = './figures/{}_{}map.svg'.format(mazetype,title.replace(" ", ""))
	
	if save_dir != None:
		save_string = save_string.replace("./figures/", save_dir)
	
	if maps == 'all':
		plotrows = 4
		plotcols = 5
		fig, axes = plt.subplots(nrows=plotrows, ncols=plotcols, sharex=True, sharey =True)
		items = np.linspace(0, len(val_maps)-1, plotrows*plotcols)
		rp_s = []
		for rwd_loc in maze.rwd_loc:
			rp_s.append(artist_instance(xy=np.add(rwd_loc,(0.5,0.5)), rad=.2))

		for i, ax in enumerate(axes.flat):
			data = val_maps[int(items[i])]
			im = ax.pcolor(data, cmap= 'Spectral_r', vmin=value_min, vmax=value_max)
			im.cmap.set_under('w', 1.0)
			im.cmap.set_over('r', 1.0)
			im.cmap.set_bad('w', 1.0)
			ax.axis('off')
			ax.set_aspect('equal')
			for rwd_patch in rp_s:
				ax.add_patch(rwd_patch.art())
			ax.set_title('{}'.format(int(items[i])))

		if title != None:
			plt.suptitle(title)

		axes[0,0].invert_yaxis()

		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		fig.colorbar(im, cax=cbar_ax)
		plt.savefig(save_string, format='svg')
		plt.show()

	elif type(maps) == list:
		plotrows = 4
		plotcols = 5
		fig, axes = plt.subplots(nrows=plotrows, ncols=plotcols, sharex=True, sharey =True)
		items = np.floor(np.linspace(maps[0], maps[-1], plotrows*plotcols))
		rp_s = []
		for rwd_loc in maze.rwd_loc:
			rp_s.append(artist_instance(xy=np.add(rwd_loc,(0.5,0.5)), rad=.2))

		for i, ax in enumerate(axes.flat):
			data = val_maps[int(items[i])]
			im = ax.pcolor(data, cmap= 'Spectral_r', vmin=value_min, vmax=value_max)
			im.cmap.set_under('w', 1.0)
			im.cmap.set_over('r', 1.0)
			im.cmap.set_bad('w', 1.0)
			ax.axis('off')
			ax.set_aspect('equal')
			for rwd_patch in rp_s:
				ax.add_patch(rwd_patch.art())
			ax.set_title('{}'.format(int(items[i])))

		if title != None:
			plt.suptitle(title)

		axes[0,0].invert_yaxis()

		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		fig.colorbar(im, cax=cbar_ax)
		plt.savefig(save_string, format='svg')
		plt.show()


	else:
		if type(maps) == int:
			data = val_maps[maps]
			trial = maps%len(val_maps)
			fig = plt.figure(1)
			ax = plt.subplots(1)
			im = plt.pcolor(data, vmin=np.nanmin(data), vmax=np.nanmax(data), cmap='Spectral_r')
			rp_s = []
			for rwd_loc in maze.rwd_loc:
				rp_s.append(artist_instance(rwd_loc, rad=.2))
			im.cmap.set_bad('w', 1.0)
			#for rwd_patch1 in rp_s:
			#	im.add_patch(rwd_patch1.art())
			#plt.imshow(data, vmin=1.32, vmax=1.5, cmap='jet', interpolation='none')
			plt.title('Trial {}'.format(trial))
			plt.colorbar()
			plt.savefig('valuemap.svg', format = 'svg')
			plt.show()
		else:
			print("Must specify which map to print (integer value) else specify 'all' ")

def policy_plot(maze, policies, **kwargs):
	pol_source = kwargs.get('polsource', 'MF')
	chance_threshold = kwargs.get('chance_threshold', np.round(1/len(maze.actionlist), 6))
	fig = plt.figure()

	cmap = plt.cm.Spectral_r
	cNorm  = colors.Normalize(vmin=0, vmax=1)
	scalarMap = cmx.ScalarMappable(norm = cNorm, cmap=cmap)


	ax1  = fig.add_axes([0.04, 0, 0.85, 0.85]) # [left, bottom, width, height]
	axc = fig.add_axes([0.89, 0.125, 0.05, 0.6])

	cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)

	ax1.imshow(maze.grid, vmin=0, vmax=1, cmap='bone', interpolation='none')
	ax1.add_patch(patches.Circle(maze.rwd_loc[0], 0.35, fc='w'))

	for i in range(0, maze.grid.shape[1]):
	    for j in range(0, maze.grid.shape[0]):
	        action = np.argmax(tuple(policies[i][j]))
	        prob = max(policies[i][j])
	        
	        dx1,dy1,head_w,head_l = make_arrows(action, prob)
	        if prob > chance_threshold:
	            if (dx1, dy1) == (0,0):
	                pass
	            else:
	                colorVal1 = scalarMap.to_rgba(prob)
	                ax1.arrow(j, i, dx1, dy1, head_width =0.3, head_length =0.2, color=colorVal1)
	        else:
	            pass

	ax1.set_title("{} policy".format(pol_source))
	plt.show()

def make_dual_policy_plots(maze, EC_policies, MF_policies, **kwargs):
	chance_threshold = kwargs.get('chance_threshold', np.round(1/len(maze.actionlist), 6))
	visited_locs = kwargs.get('visited_locs', [])

	fig 		= plt.figure()
	cmap 		= plt.cm.Spectral_r
	cNorm  		= colors.Normalize(vmin=0, vmax=1)
	scalarMap 	= cmx.ScalarMappable(norm = cNorm, cmap=cmap)

	ax1  		= fig.add_axes([0.04, 0, 0.4, 0.85]) # [left, bottom, width, height]
	ax2   		= fig.add_axes([0.47, 0, 0.4, 0.85]) # [left, bottom, width, height]
	axc 		= fig.add_axes([0.89, 0.125, 0.05, 0.6])

	cb1 		= colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)

	ax1.pcolor(maze.grid, vmin=0, vmax=1, cmap='bone')#, interpolation='none')
	ax1.add_patch(patches.Circle(maze.rwd_loc[0], 0.35, fc='w'))
	ax1.add_patch(patches.Circle((10,10), 0.35, fc='k', ec='w'))
	ax1.invert_yaxis()
	ax1.set_aspect('equal')

	
	ax2.pcolor(maze.grid, vmin=0, vmax=1, cmap='bone')#, interpolation='none')
	ax2.add_patch(patches.Circle(np.add(maze.rwd_loc[0], (0.5,0.5)), 0.35, fc='w'))
	#ax2.add_patch(patches.Circle((10,10), 0.35, fc='k', ec='w'))
	ax2.invert_yaxis()
	ax2.set_aspect('equal')

	# p_field indicies
	# 0 - choice, 
	# 1 - list(tfprob)[choice], 
	# 2 - list(tfprob).index(max(list(tfprob))),
	# 3 - max(list(tfprob)), 
	# 4 - i)

	for i in range(0, maze.grid.shape[1]):
	    for j in range(0, maze.grid.shape[0]):
	        EC_action = np.argmax(tuple(EC_policies[i][j]))
	        EC_prob = max(EC_policies[i][j])
	        
	        dx1,dy1,head_w,head_l = make_arrows(EC_action, EC_prob)
	        if EC_prob > chance_threshold:
	            if (dx1, dy1) == (0,0):
	                pass
	            else:
	                colorVal1 = scalarMap.to_rgba(EC_prob)
	                ax1.arrow(j+0.5, i+0.5, dx1, dy1, head_width =0.3, head_length =0.2, color=colorVal1)
	        else:
	            pass
	        
	        MF_action = np.argmax(tuple(MF_policies[i][j]))
	        MF_prob = max(MF_policies[i][j])
	        dx2,dy2,head_w,head_l = make_arrows(MF_action, MF_prob)
	        
	        if MF_prob > chance_threshold:
	            if (dx2, dy2) == (0,0):
	                pass
	            else:
	                colorVal1 = scalarMap.to_rgba(MF_prob)
	                ax2.arrow(j+0.5, i+0.5, dx2, dy2, head_width =0.3, head_length =0.2, color=colorVal1)
	        else: 
	            pass
	ax1.set_title("EC_policy")
	ax2.set_title("MF_policy")
	savefig = kwargs.get('savedir', None)
	if savefig is not None:
		print("Saving Figure at {}".format(savefig))
		plt.savefig(savefig, format='svg')

	plt.show()


class KLD_holder(object):
	def __init__(self,gridworld,**kwargs):
		self.flag = kwargs.get('track', 'KLD')
		self.y 		 = gridworld.y
		self.x 		 = gridworld.x
		self.num_act = len(gridworld.actionlist)
		if self.flag == 'KLD':
			self.map = np.zeros((self.y, self.x))
			self.map.fill(np.nan)
			self.op 	= opt_pol_map(gridworld)
		elif self.flag == 'policy':
			self.map 	 = np.zeros((self.y, self.x, self.num_act))
		else:
			print("Flag error. Track 'KLD' (default) or 'policy' as keyword")
		

	def update(self,state, policy):
		#policy must be list or array
		if self.flag == 'policy':
			self.map[state[1], state[0]] = policy

		elif self.flag =='KLD':
			optimal = self.op[state[1], state[0]]
			self.map[state[1], state[0]] = st.entropy(optimal,policy)
			
	def reset(self):
		if self.flag == 'policy':
			self.map = np.zeros((self.y, self.x, self.num_act))
		elif self.flag == 'KLD':
			self.map = np.zeros((self.y, self.x))
		

def opt_pol_map(gridworld):
	optimal_policy = np.zeros((gridworld.y, gridworld.x, len(gridworld.actionlist)))

	for location in gridworld.useable:
		xdim,ydim=location
		xrwd,yrwd=gridworld.rwd_loc[0]
		
		if xdim<xrwd:
			optimal_policy[ydim,xdim][1] = 1
			if ydim<yrwd:
				optimal_policy[ydim,xdim][3] = 1
			elif ydim>yrwd:
				optimal_policy[ydim,xdim][0] = 1
				
		elif xdim>xrwd:
			optimal_policy[ydim,xdim][2] = 1
			if ydim<yrwd:
				optimal_policy[ydim,xdim][3] = 1
			elif ydim>yrwd:
				optimal_policy[ydim,xdim][0] = 1
		else:
			if ydim<yrwd:
				optimal_policy[ydim,xdim][3] = 1
			elif ydim>yrwd:
				optimal_policy[ydim,xdim][0] = 1
			else:
				optimal_policy[ydim,xdim][5] = 1
		
		optimal_policy[ydim,xdim] = softmax(optimal_policy[ydim,xdim],T=0.01)

	return optimal_policy


def save_value_map(vm, maze, trial, savedir):
    data = vm
    fig = plt.figure()
    im = plt.imshow(data, vmin = 0, vmax = 40, cmap='Spectral_r', interpolation ='none')
    rp_s = []
    for reward_loc in maze.rwd_loc:
        rp_s.append(eu.artist_instance(xy=np.add(reward_loc,(0,0)), rad = 0.2))
    im.cmap.set_bad('w', 1.0)
    for rp1 in rp_s:
        fig.axes[0].add_patch(rp1.art())
    plt.colorbar()
    plt.title('Trial {}'.format(trial))
    plt.savefig(savedir+'trial_{}'.format(trial))
    plt.close()
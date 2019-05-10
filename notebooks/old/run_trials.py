from modules import * 


def run_trials(NUM_TRIALS, NUM_EVENTS, trial_params):
	MF 			= trial_params['MF']
	use_EC		= trial_params['use_EC']
	if use_EC:
		EC 		= trial_params['EC']
		EC.reset_cache()
		EC.reward_unseen = True 


	timestamp 	= trial_params['timestamp'] # 0 
	thresh_suc 	= trial_params['success_threshold'] # 0.9 # average success before employing EC or removing most rewarded port   !!!! (NUM_EVENTS -((maze.y-1)+(maze.x-1)))/NUM_EVENTS

	maze 		= trial_params['maze']

	total_loss 	= trial_params['total_loss']
	total_reward= trial_params['total_reward'] 
	val_maps 	= trial_params['val_maps']
	policies 	= trial_params['policies']
	add_mem_dict= trial_params['add_mem_dict']

	deltas 		= trial_params['deltas']
	spots 		= trial_params['spots']
	vls 		= trial_params['vls']


	blocktime = time.time()
	for trial in range(NUM_TRIALS):
		track_deltas 	= []
		track_spots 	= []
		visited_locs 	= []
		
		reward_sum 		= 0
		v_last      	= 0
		trialstart_stamp = timestamp
		
		#assuming only using convolutional inputs
		env.reset()
		# because we need to include batch size of 1 
		frame = np.expand_dims(sg.get_frame(maze), axis=0)
		state = ac.Variable(ac.torch.FloatTensor(frame))

		MF.reinit_hid()

		for event in range(NUM_EVENTS):
			if use_EC:
				policy_, value_, lin_act_ 	= MF(state,temperature)
				add_mem_dict['state'] 		= maze.cur_state

				visited_locs.append(maze.cur_state)

				choice, policy, value = ac.select_action(MF,policy_, value_)
				if event < NUM_EVENTS: 
					next_state, reward, done, info = env.step(choice)

				MF.rewards.append(reward)
				delta = reward + discount_factor*value - v_last  #compute eligibility trace/rpe approximation
				
				add_mem_dict['activity']  = tuple(lin_act_.view(-1).data)
				add_mem_dict['action']    = choice
				add_mem_dict['delta']     = delta
				add_mem_dict['timestamp'] = timestamp            
				EC.add_mem(add_mem_dict, keep_hist = True)             #add event to memory cache
				if reward != 0:
					EC.reward_update(trialstart_stamp, timestamp, reward)
				#EC.reward_update(trialstart_stamp, timestamp, delta[0])
				track_deltas.append(delta[0])
				track_spots.append(maze.cur_state)
			

			else: 
				policy_, value_ = MF(state, temperature)[0:2]
				choice, policy, value = ac.select_action(MF,policy_, value_)
				if event < NUM_EVENTS: 
					next_state, reward, done, info = env.step(choice)

				MF.rewards.append(reward)
				delta = reward + discount_factor*value - v_last  #compute eligibility trace/rpe approximation
			
				
			# because we need to include batch size of 1 
			frame = np.expand_dims(sg.get_frame(maze), axis = 0)
			state = ac.Variable(ac.torch.FloatTensor(frame))
			reward_sum += reward
		
			v_last = value
			timestamp += 1
		
		p_loss, v_loss = ac.finish_trial(MF, discount_factor,opt)
		
		total_loss[0].append(p_loss.data[0])
		total_loss[1].append(v_loss.data[0])
		total_reward.append(reward_sum)
		
		value_map = ac.generate_values(maze,MF)
		val_maps.append(value_map.copy())
		
				
		deltas.append(track_deltas)
		spots.append(track_spots)
		vls.append(visited_locs)
		if trial ==0 or trial%100==0 or trial == NUM_TRIALS-1:
			EC_policies, MF_policies = ac.generate_values(maze, MF,EC=EC)
			policies[0]['{}'.format(trial)] = EC_policies
			policies[1]['{}'.format(trial)] = MF_policies
			#print("[{0}]  Trial {1} total reward = {2} (Avg {3:.3f})".format(time.strftime("%H:%M:%S", time.localtime()), trial, reward_sum, float(reward_sum)/float(NUM_EVENTS)), "Block took {0:.3f}".format(time.time()-blocktime))
			print("[{0}]  Trial {1} TotRew = {2} ({3:.3f}s)".format(time.strftime("%H:%M:%S", time.localtime()), trial, reward_sum,time.time()-blocktime))
			blocktime = time.time()





	# flags to be changed mid run
	add_episodic_cache = False  ## Possibly unnecessary now 
	if add_episodic_cache:
		rwd_threshold = True
	midrun_rwd_removal = False

	if midrun_rwd_removal: 
		reward_tally = {}
		for _ in maze.rwd_loc: 
			reward_tally[_] = []
		trial_rwd_switch = 0





	blocktime = time.time()
	for trial in range(NUM_TRIALS):
		trialstart_stamp = timestamp
		
		reward_sum = 0
		v_last      = 0
		track_deltas = []
		track_spots = []
		visited_locs = []
		
		if state_type == 'pcs':
			get_pcs = pcs.activity(env.reset())
			state = ac.Variable(ac.torch.FloatTensor(get_pcs))
		elif state_type == 'conv':
			env.reset()
			# because we need to include batch size of 1 
			frame = np.expand_dims(sg.get_frame(maze), axis=0)
			state = ac.Variable(ac.torch.FloatTensor(frame))
			
		MF.reinit_hid()
		for event in range(NUM_EVENTS):
			# pass state through EC module
			if use_EC:
				policy_, value_, lin_act_ = MF(state,temperature)
				add_mem_dict['state'] = maze.cur_state
				visited_locs.append(maze.cur_state)
			else: 
				policy_, value_ = MF(state, temperature)[0:2]
			
			choice, policy, value = ac.select_action(MF,policy_, value_)
			if event < NUM_EVENTS: 
				next_state, reward, done, info = env.step(choice)

			MF.rewards.append(reward)
			delta = reward + discount_factor*value - v_last  #compute eligibility trace/rpe approximation

			
			if use_EC:
				add_mem_dict['activity']  = tuple(lin_act_.view(-1).data)
				add_mem_dict['action']    = choice
				add_mem_dict['delta']     = delta
				add_mem_dict['timestamp'] = timestamp            
				EC.add_mem(add_mem_dict, keep_hist = True)             #add event to memory cache
				if reward != 0:
					EC.reward_update(trialstart_stamp, timestamp, reward)
				#EC.reward_update(trialstart_stamp, timestamp, delta[0])
				track_deltas.append(delta[0])
				track_spots.append(maze.cur_state)
			
			if state_type == 'pcs':
				state = ac.Variable(ac.torch.FloatTensor(pcs.activity(next_state)))       # update state
			elif state_type == 'conv':
				# because we need to include batch size of 1 
				frame = np.expand_dims(sg.get_frame(maze), axis = 0)
				state = ac.Variable(ac.torch.FloatTensor(frame))
			reward_sum += reward
		
			v_last = value
			timestamp += 1
		
		
		if add_episodic_cache:
			if (np.array(total_reward[-50:]).mean() > thresh_suc*NUM_EVENTS):
				if rwd_threshold:
					print(" \t Started Memory at Trial ", trial)
					if midrun_rwd_removal:
						maxsums = {}
						for item in reward_tally.items():
							maxsums[item[0]] = sum(item[1])
						most_rewarded_location = max(maxsums.iteritems(), key=operator.itemgetter(1))[0] 
						maze.rwd_loc.remove(most_rewarded_location)
						trial_rwd_switch = trial
						print("removed reward at ", most_rewarded_location)

					rwd_threshold = False
					use_EC = True
		
		if midrun_rwd_removal:
			if (trial_rwd_switch!=0) and (trial == trial_rwd_switch + 1000):
				maze.rwd_loc.append(most_rewarded_location)

		p_loss, v_loss = ac.finish_trial(MF, discount_factor,opt)
		
		total_loss[0].append(p_loss.data[0])
		total_loss[1].append(v_loss.data[0])
		total_reward.append(reward_sum)
		
		if state_type == 'pcs':
			value_map = ac.generate_values(maze,MF,pcs=pcs)
		else:
			value_map = ac.generate_values(maze,MF)
		val_maps.append(value_map.copy())
		
		if midrun_rwd_removal:
			for item in maze.reward_tally.items():
				reward_tally[item[0]].append(item[1])
				
		deltas.append(track_deltas)
		spots.append(track_spots)
		vls.append(visited_locs)
		if trial ==0 or trial%100==0 or trial == NUM_TRIALS-1:
			EC_policies, MF_policies = ac.generate_values(maze, MF,EC=EC)
			policies[0]['{}'.format(trial)] = EC_policies
			policies[1]['{}'.format(trial)] = MF_policies
			#print("[{0}]  Trial {1} total reward = {2} (Avg {3:.3f})".format(time.strftime("%H:%M:%S", time.localtime()), trial, reward_sum, float(reward_sum)/float(NUM_EVENTS)), "Block took {0:.3f}".format(time.time()-blocktime))
			print("[{0}]  Trial {1} TotRew = {2} ({3:.3f}s)".format(time.strftime("%H:%M:%S", time.localtime()), trial, reward_sum,time.time()-blocktime))
			blocktime = time.time()

return total_loss, total_reward, deltas, spots, vls, val_maps, 
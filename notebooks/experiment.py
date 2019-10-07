
# write a function for runs with episodic mem and without -- take use_EC as a param
# assume just for conv inputs
import numpy as np
import time
import torch
import sys
sys.path.insert(0,'../rl_network/'); import actorcritic as ac;  import stategen as sg
sys.path.insert(0,'../memory/'); import episodic as ec

def run_full_trials(run_dict, use_EC = False, **kwargs):
    NUM_TRIALS = run_dict['NUM_TRIALS']
    NUM_EVENTS = run_dict['NUM_EVENTS']

    maze         = run_dict['environment']
    MF           = run_dict['agent']
    opt          = run_dict['optimizer']
    agent_params = run_dict['agt_param']
    penalization = kwargs.get('pen', 0)

    #use_EC     = False# agent_params['use_EC']
    #EC         = agent_params['EC']

    save_data  = kwargs.get('save', True)



    blocktime = time.time()

    if use_EC:
        #EC = agent_params['EC']
        EC.reset_cache()
        memory_buffer = [[],[],[],[], 0] # [timestamp, state_t, a_t, readable_state, trial]
        run_dict['total_loss']   = [[],[]]
        run_dict['total_reward'] = []
        run_dict['track_cs']     = [[],[]]
        run_dict['rpe']          = np.zeros(maze.grid.shape)

        reward    = 0
        timestamp = 0
        tslr      = np.nan_to_num(np.inf)

        compare_mfec = {}

        for trial in range(NUM_TRIALS):
            trialstart_stamp = timestamp

            reward_sum   = 0
            v_last       = 0

            maze.reset()

            state = torch.Tensor(sg.get_frame(maze))
            MF.reinit_hid() #reinit recurrent hidden layers

            for event in range(NUM_EVENTS):
                if trial is not 0:
                    #compute confidence in MFC
                    if event in [0,1]:
                        MF_cs = EC.make_pvals(tslr,envelope=10)
                    else:
                        MF_cs = EC.make_pvals(tslr,envelope=10, pol_id = pol_flag, mfc=MF_cs)
                    # pass state through EC module
                else:
                    MF_cs = EC.make_pvals(tslr,envelope=10)
                policy_, value_, lin_act_ = MF(state, temperature = 1)
                lin_act = tuple(np.round(lin_act_.data[0].numpy(),4))

                if trial == 0:
                    choice, policy, value = ac.select_action(MF,policy_, value_)
                else:
                    if event is not 0:
                        ec_pol = torch.from_numpy(EC.recall_mem(lin_act, timestamp, env=150))
                        candidate_policies = [policy_, ec_pol]
                        pol_choice = np.random.choice([0,1], p=[MF_cs, 1-MF_cs])
                        pol = candidate_policies[pol_choice]
                        if pol_choice == 0:
                            pol_flag = 'MF'
                        else:
                            pol_flag = 'EC'

                        choice, policy, value = ac.select_ec_action(MF, policy_, value_, pol)
                    else:
                        choice, policy, value = ac.select_action(MF,policy_, value_)

                memory_buffer[3].append(maze.cur_state)

                compare_mfec[maze.cur_state] = [policy.numpy(), choice, value]

                if event < NUM_EVENTS:
                    next_state, reward, done, info = maze.step(choice)

                if event is not 0:
                    if reward == 1:
                        tslr = 0
                    else:
                        tslr += 1

                    run_dict['track_cs'][0].append(tslr)
                    run_dict['track_cs'][1].append(MF_cs)

                MF.rewards.append(reward)

                memory_buffer[0].append(timestamp)
                memory_buffer[1].append(lin_act)
                memory_buffer[2].append(choice)
                memory_buffer[4] = trial

                # because we need to include batch size of 1
                state = torch.Tensor(sg.get_frame(maze))
                reward_sum += reward

                v_last = value
                timestamp += 1

            p_loss, v_loss = ac.finish_trial(MF,agent_params['gamma'],opt,cache=EC, buffer=memory_buffer)

            if save_data:
                #value_map = ac.generate_values(maze,MF)
                #run_dict['total_loss'][0].append(p_loss.data[0])
                #run_dict['total_loss'][1].append(v_loss.data[0])
                run_dict['total_reward'].append(reward_sum)
                #run_dict['val_maps'].append(value_map.copy())
                #run_dict['deltas'].append(track_deltas)
                #run_dict['spots'].append(track_spots)
                #run_dict['vls'].append(visited_locs)

            if trial ==0 or trial%10==0 or trial == NUM_TRIALS-1:
                print("[{0}]  Trial {1} TotRew = {2} ({3:.3f}s)".format(time.strftime("%H:%M:%S", time.localtime()), trial+1, reward_sum,time.time()-blocktime)) #print("[{0}]  Trial {1} total reward = {2} (Avg {3:.3f})".format(time.strftime("%H:%M:%S", time.localtime()), trial, reward_sum, float(reward_sum)/float(NUM_EVENTS)), "Block took {0:.3f}".format(time.time()-blocktime))
                blocktime = time.time()
            run_dict['comp_mfec'] = compare_mfec

    else:
        run_dict['rpe'] = np.zeros(maze.grid.shape)
        run_dict['total_loss'] =  [[],[]]
        run_dict['total_reward'] = []
        compare_mfec = {}

        for trial in range(NUM_TRIALS):
            reward_sum   = 0
            v_last       = 0
            track_deltas = []
            track_spots  = []
            visited_locs = []

            maze.reset()
            state = torch.Tensor(sg.get_frame(maze))
            MF.reinit_hid() #reinit recurrent hidden layers
            for event in range(NUM_EVENTS):
                policy_, value_ = MF(state, agent_params['temperature'])[0:2]
                choice, policy, value = ac.select_action(MF,policy_, value_)

                compare_mfec[maze.cur_state] = [policy.numpy(), choice, value]

                if event < NUM_EVENTS:
                    next_state, reward, done, info = maze.step(choice)

                if reward != 1:
                    reward = penalization

                MF.rewards.append(reward)


                #compute eligibility trace/rpe approximation
                delta = reward + agent_params['gamma']*value - v_last
                state = torch.Tensor(sg.get_frame(maze))
                run_dict['rpe'][maze.cur_state[1]][maze.cur_state[0]] = delta

                reward_sum += reward
                v_last = value

            p_loss, v_loss = ac.finish_trial(MF,agent_params['gamma'],opt)

            if save_data:
                #value_map = ac.generate_values(maze,MF)
                run_dict['total_loss'][0].append(p_loss.item())
                run_dict['total_loss'][1].append(v_loss.item())
                run_dict['total_reward'].append(reward_sum)
                #run_dict['val_maps'].append(value_map.copy())
                #run_dict['deltas'].append(track_deltas)
                #run_dict['spots'].append(track_spots)
                #run_dict['vls'].append(visited_locs)

            if trial ==0 or trial%100==0 or trial == NUM_TRIALS-1:
                print("[{0}]  Trial {1} TotRew = {2} ({3:.3f}s)".format(time.strftime("%H:%M:%S", time.localtime()), trial+1, reward_sum,time.time()-blocktime)) #print("[{0}]  Trial {1} total reward = {2} (Avg {3:.3f})".format(time.strftime("%H:%M:%S", time.localtime()), trial, reward_sum, float(reward_sum)/float(NUM_EVENTS)), "Block took {0:.3f}".format(time.time()-blocktime))
                blocktime = time.time()
            run_dict['comp_mfec'] = compare_mfec


# write a function for runs with episodic mem and without -- take use_EC as a param
# assume just for conv inputs
def run_truncated_trials(run_dict, use_EC=False, **kwargs):
    NUM_TRIALS = run_dict['NUM_TRIALS']
    NUM_EVENTS = run_dict['NUM_EVENTS']

    maze         = run_dict['environment']
    MF           = run_dict['agent']
    opt          = run_dict['optimizer']
    agent_params = run_dict['agt_param']

    use_EC      = False# agent_params['use_EC']
    #EC         = agent_params['EC']

    penalization = kwargs.get('pen', 0)

    save_data  = kwargs.get('save', True)



    blocktime = time.time()

    if use_EC:
        EC = agent_params['EC']
        EC.reset_cache()
        memory_buffer = [[],[],[],[], 0] # [timestamp, state_t, a_t, readable_state, trial]
        run_dict['total_loss']   = [[],[]]
        run_dict['total_reward'] = []
        run_dict['track_cs']     = [[],[]]
        run_dict['rpe']          = np.zeros(maze.grid.shape)
        run_dict['trial_length'] = []

        reward    = 0
        timestamp = 0
        tslr      = np.nan_to_num(np.inf)

        compare_mfec = {}

        for trial in range(NUM_TRIALS):
            trialstart_stamp = timestamp

            reward_sum   = 0
            v_last       = 0

            maze.reset()

            state = torch.Tensor(sg.get_frame(maze))
            MF.reinit_hid() #reinit recurrent hidden layers
            print(maze.cur_state)
            for event in range(NUM_EVENTS):
                if trial is not 0:
                    #compute confidence in MFC
                    if event in [0,1]:
                        MF_cs = EC.make_pvals(tslr,envelope=10)
                    else:
                        MF_cs = EC.make_pvals(tslr,envelope=10, pol_id = pol_flag, mfc=MF_cs)
                    # pass state through EC module
                else:
                    MF_cs = EC.make_pvals(tslr,envelope=10)
                policy_, value_, lin_act_ = MF(state, temperature = 1)
                lin_act = tuple(np.round(lin_act_.data[0].numpy(),4))

                if trial == 0:
                    choice, policy, value = ac.select_action(MF,policy_, value_)
                else:
                    if event is not 0:
                        ec_pol = torch.from_numpy(EC.recall_mem(lin_act, timestamp, env=150))
                        candidate_policies = [policy_, ec_pol]
                        pol_choice = np.random.choice([0,1], p=[MF_cs, 1-MF_cs])
                        pol = candidate_policies[pol_choice]
                        if pol_choice == 0:
                            pol_flag = 'MF'
                        else:
                            pol_flag = 'EC'
                        choice, policy, value = ac.select_ec_action(MF, policy_, value_, pol)
                    else:
                        choice, policy, value = ac.select_action(MF,policy_, value_)

                memory_buffer[3].append(maze.cur_state)

                compare_mfec[maze.cur_state] = [policy.numpy(), choice, value]

                if event < NUM_EVENTS:
                    next_state, reward, done, info = maze.step(choice)

                if event is not 0:
                    if reward == 1:
                        tslr = 0

                    else:
                        tslr += 1
                        reward = -0.01
                    run_dict['track_cs'][0].append(tslr)
                    run_dict['track_cs'][1].append(MF_cs)

                MF.rewards.append(reward)

                memory_buffer[0].append(timestamp)
                memory_buffer[1].append(lin_act)
                memory_buffer[2].append(choice)
                memory_buffer[4] = trial

                # because we need to include batch size of 1
                state = torch.Tensor(sg.get_frame(maze))
                reward_sum += reward

                v_last = value
                timestamp += 1
                if reward == 1:
                    run_dict['trial_length'].append(event)
                    #print(f'{trial}:{event}')
                    break
                if event == NUM_EVENTS-1:
                    run_dict['trial_length'].append(event)


            p_loss, v_loss = ac.finish_trial(MF,agent_params['gamma'],opt,cache=EC, buffer=memory_buffer)

            if save_data:
                #value_map = ac.generate_values(maze,MF)
                #run_dict['total_loss'][0].append(p_loss.data[0])
                #run_dict['total_loss'][1].append(v_loss.data[0])
                run_dict['total_reward'].append(reward_sum)
                #run_dict['val_maps'].append(value_map.copy())
                #run_dict['deltas'].append(track_deltas)
                #run_dict['spots'].append(track_spots)
                #run_dict['vls'].append(visited_locs)
            #if reward ==1:
            #    break

            if trial ==0 or trial%10==0 or trial == NUM_TRIALS-1:
                print("[{0}]  Trial {1} TotRew = {2} ({3:.3f}s)".format(time.strftime("%H:%M:%S", time.localtime()), trial+1, reward_sum,time.time()-blocktime)) #print("[{0}]  Trial {1} total reward = {2} (Avg {3:.3f})".format(time.strftime("%H:%M:%S", time.localtime()), trial, reward_sum, float(reward_sum)/float(NUM_EVENTS)), "Block took {0:.3f}".format(time.time()-blocktime))
                blocktime = time.time()
            run_dict['comp_mfec'] = compare_mfec

    else:
        run_dict['rpe'] = np.zeros(maze.grid.shape)
        run_dict['total_loss']   = [[],[]]
        run_dict['total_reward'] = []
        run_dict['trial_length'] = []
        compare_mfec = {} #[[],[],[],[]]

        for trial in range(NUM_TRIALS):
            reward_sum   = 0
            v_last       = 0
            track_deltas = []
            track_spots  = []
            visited_locs = []

            maze.reset()
            state = torch.Tensor(sg.get_frame(maze))
            MF.reinit_hid() #reinit recurrent hidden layers
            for event in range(NUM_EVENTS):
                policy_, value_ = MF(state, agent_params['temperature'])[0:2]
                choice, policy, value = ac.select_action(MF,policy_, value_)

                compare_mfec[maze.cur_state] = [policy.numpy(), choice, value]

                if event < NUM_EVENTS:
                    next_state, reward, done, info = maze.step(choice)
                if reward != 1:
                    reward = penalization
                MF.rewards.append(reward)

                #compute eligibility trace/rpe approximation
                delta = reward + agent_params['gamma']*value - v_last
                state = torch.Tensor(sg.get_frame(maze))
                run_dict['rpe'][maze.cur_state[1]][maze.cur_state[0]] = delta

                reward_sum += reward
                v_last = value

                if reward == 1:
                    run_dict['trial_length'].append(event)
                    #print(f'{trial}:{event}')
                    break
                if event == NUM_EVENTS-1:
                    run_dict['trial_length'].append(event)
            p_loss, v_loss = ac.finish_trial(MF,agent_params['gamma'],opt)

            if save_data:
                #value_map = ac.generate_values(maze,MF)
                run_dict['total_loss'][0].append(p_loss.item())
                run_dict['total_loss'][1].append(v_loss.item())
                run_dict['total_reward'].append(reward_sum)
                #run_dict['val_maps'].append(value_map.copy())
                #run_dict['deltas'].append(track_deltas)
                #run_dict['spots'].append(track_spots)
                #run_dict['vls'].append(visited_locs)

            if trial ==0 or trial%100==0 or trial == NUM_TRIALS-1:
                print("[{0}]  Trial {1} TotRew = {2} ({3:.3f}s)".format(time.strftime("%H:%M:%S", time.localtime()), trial+1, reward_sum,time.time()-blocktime)) #print("[{0}]  Trial {1} total reward = {2} (Avg {3:.3f})".format(time.strftime("%H:%M:%S", time.localtime()), trial, reward_sum, float(reward_sum)/float(NUM_EVENTS)), "Block took {0:.3f}".format(time.time()-blocktime))
                blocktime = time.time()
            run_dict['comp_mfec'] = compare_mfec

'''
def run_full_trials(run_dict, use_EC = False, **kwargs):
    NUM_TRIALS = run_dict['NUM_TRIALS']
    NUM_EVENTS = run_dict['NUM_EVENTS']

    maze         = run_dict['environment']
    MF           = run_dict['agent']
    opt          = run_dict['optimizer']
    agent_params = run_dict['agt_param']
    penalization = kwargs.get('pen', 0)

    use_EC     = False# agent_params['use_EC']
    #EC         = agent_params['EC']

    save_data  = kwargs.get('save', True)



    blocktime = time.time()

    if use_EC:
        #EC = agent_params['EC']
        EC.reset_cache()
        memory_buffer = [[],[],[],[], 0] # [timestamp, state_t, a_t, readable_state, trial]
        run_dict['total_loss']   = [[],[]]
        run_dict['total_reward'] = []
        run_dict['track_cs']     = [[],[]]
        run_dict['rpe']          = np.zeros(maze.grid.shape)

        reward    = 0
        timestamp = 0
        tslr      = np.nan_to_num(np.inf)

        compare_mfec = {}

        for trial in range(NUM_TRIALS):
            trialstart_stamp = timestamp

            reward_sum   = 0
            v_last       = 0

            maze.reset()

            state = torch.Tensor(sg.get_frame(maze))
            MF.reinit_hid() #reinit recurrent hidden layers

            for event in range(NUM_EVENTS):
                if trial is not 0:
                    #compute confidence in MFC
                    if event in [0,1]:
                        MF_cs = EC.make_pvals(tslr,envelope=10)
                    else:
                        MF_cs = EC.make_pvals(tslr,envelope=10, pol_id = pol_flag, mfc=MF_cs)
                    # pass state through EC module
                else:
                    MF_cs = EC.make_pvals(tslr,envelope=10)
                policy_, value_, lin_act_ = MF(state, temperature = 1)
                lin_act = tuple(np.round(lin_act_.data[0].numpy(),4))

                if trial == 0:
                    choice, policy, value = ac.select_action(MF,policy_, value_)
                else:
                    if event is not 0:
                        ec_pol = torch.from_numpy(EC.recall_mem(lin_act, timestamp, env=150))
                        candidate_policies = [policy_, ec_pol]
                        pol_choice = np.random.choice([0,1], p=[MF_cs, 1-MF_cs])
                        pol = candidate_policies[pol_choice]
                        if pol_choice == 0:
                            pol_flag = 'MF'
                        else:
                            pol_flag = 'EC'

                        choice, policy, value = ac.select_ec_action(MF, policy_, value_, pol)
                    else:
                        choice, policy, value = ac.select_action(MF,policy_, value_)

                memory_buffer[3].append(maze.cur_state)

                compare_mfec[maze.cur_state] = [policy.numpy(), choice, value]

                if event < NUM_EVENTS:
                    next_state, reward, done, info = maze.step(choice)

                if event is not 0:
                    if reward == 1:
                        tslr = 0
                    else:
                        tslr += 1

                    run_dict['track_cs'][0].append(tslr)
                    run_dict['track_cs'][1].append(MF_cs)

                MF.rewards.append(reward)

                memory_buffer[0].append(timestamp)
                memory_buffer[1].append(lin_act)
                memory_buffer[2].append(choice)
                memory_buffer[4] = trial

                # because we need to include batch size of 1
                state = torch.Tensor(sg.get_frame(maze))
                reward_sum += reward

                v_last = value
                timestamp += 1

            p_loss, v_loss = ac.finish_trial(MF,agent_params['gamma'],opt,cache=EC, buffer=memory_buffer)

            if save_data:
                #value_map = ac.generate_values(maze,MF)
                #run_dict['total_loss'][0].append(p_loss.data[0])
                #run_dict['total_loss'][1].append(v_loss.data[0])
                run_dict['total_reward'].append(reward_sum)
                #run_dict['val_maps'].append(value_map.copy())
                #run_dict['deltas'].append(track_deltas)
                #run_dict['spots'].append(track_spots)
                #run_dict['vls'].append(visited_locs)

            if trial ==0 or trial%10==0 or trial == NUM_TRIALS-1:
                print("[{0}]  Trial {1} TotRew = {2} ({3:.3f}s)".format(time.strftime("%H:%M:%S", time.localtime()), trial+1, reward_sum,time.time()-blocktime)) #print("[{0}]  Trial {1} total reward = {2} (Avg {3:.3f})".format(time.strftime("%H:%M:%S", time.localtime()), trial, reward_sum, float(reward_sum)/float(NUM_EVENTS)), "Block took {0:.3f}".format(time.time()-blocktime))
                blocktime = time.time()
            run_dict['comp_mfec'] = compare_mfec

'''
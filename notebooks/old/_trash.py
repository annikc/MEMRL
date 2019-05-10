from modules import *

# Data Collection Parameters
fig_savedir = '../data/figures/'
print_freq  = 1/10

# Environment Parameters
y_height   = 20
x_width    = 20
walls 	   = False
rho 	   = 0
maze_type  = 'none'
port_shift = 'none'

# Model Parameters
load_model = True
state_type = 'conv' # input type 
action_dims = len(maze.actionlist) #policy output layer size
batch_size = 1 
discount_factor = 0.98 
eta = 5e-4 #gradient descent learning rate
temperature = 1

# Run Parameters
NUM_EVENTS = 100
NUM_TRIALS = 1

use_EC = False


#make environment
maze = eu.gridworld([y_height, x_width], rho = rho, maze_type = maze_type, port_shift = port_shift, walls = walls, barheight=int(y_height/2 -1))
#set reward location
maze.rwd_loc = [(5,15)] #[(int(y_height/2),int(x_width/2))]


if maze_type is not 'triple_reward':
    for i in maze.rwd_loc: 
        maze.orig_rwd_loc.append(i)

if state_type == 'pcs':
    # place cell parameters
    num_pc = 1000
    fwhm = 0.05
    pcs = sg.PlaceCells(num_cells=num_pc, grid=maze, fwhm=fwhm)

    #show environment
    eu.make_env_plots(maze,env=True,pc_map=True,pcs=pcs, save=False)
else: 
    eu.make_env_plots(maze,env=True,save=True)

## test out gridworld wrapper. 
env = eu.gymworld(maze)

if state_type == 'conv':
    num_channels = 3
    if maze.bound:
        input_dims = (y_height+2, x_width+2, num_channels)
    else:
        input_dims = (y_height, x_width, num_channels)
    hid_types = ['conv', 'pool', 'linear']
    conv_dims = ac.conv_output(input_dims)
    pool_dims = ac.conv_output(conv_dims)
    hid_dims = [conv_dims, pool_dims, 500]

elif state_type == 'pcs':
    input_dims = 1000
    hid_types = ['linear']
    hid_dims = [500]


if load_model:
    # # load previously saved model: 
    MF = ac.torch.load('../data/outputs/gridworld/MF{}{}training.pt'.format(maze.x, maze.y))
else:
    MF = ac.AC_Net(input_dims, action_dims, batch_size, hid_types, hid_dims)


opt = ac.optim.Adam(MF.parameters(), lr = eta)
    
cachelim = int(0.75*np.prod(maze.grid.shape)) # memory limit should be ~75% of #actions x #states
EC = ec.ep_mem(MF,cachelim) 


# --------------------------------
# empty data frames for recording
# --------------------------------
#data frames
total_loss = [[],[]]
total_reward = []
val_maps = []
policies = [{},{}]
EC.reset_cache()
EC.reward_unseen = True 
add_mem_dict = {} #dictionary of items which get put into memory cache
timestamp = 0
success_benchmark = 0.9 # average success before employing EC or removing most rewarded port

# flags to be changed mid run
add_episodic_cache = False  ## Possibly unnecessary now 
if add_episodic_cache:
    rwd_threshold = True
midrun_rwd_removal = False
success_benchmark = (NUM_EVENTS -((maze.y-1)+(maze.x-1)))/NUM_EVENTS
print(success_benchmark)

if midrun_rwd_removal: 
    reward_tally = {}
    for _ in maze.rwd_loc: 
        reward_tally[_] = []
    trial_rwd_switch = 0
    
deltas = []
spots = []

blocktime = time.time()
vls = []
#==================================
# Run Trial
#==================================
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
        if (np.array(total_reward[-50:]).mean() > success_benchmark*NUM_EVENTS):
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

plt.figure(0)
plt.plot(total_loss[0], label='p')
plt.plot(total_loss[1], label='v')
plt.legend(loc=0)

plt.figure(1)
plt.plot(total_reward, label='r', color='r', marker='')

#plt.figure(2)
#delta_of_interest = deltas[-1]
#plt.plot(deltas[-1])
#plt.annotate('{}'.format(spots[-1][np.argmax(delta_of_interest)]), xy=(np.argmax(delta_of_interest), max(delta_of_interest)) )
#print(maze.rwd_loc[0])
if maze_type == 'triple_reward':
    plt.figure(2)
    plt.plot(reward_tally[reward_tally.keys()[0]], label='{}'.format(reward_tally.keys()[0]))
    plt.plot(reward_tally[reward_tally.keys()[1]], label='{}'.format(reward_tally.keys()[1]))
    plt.plot(reward_tally[reward_tally.keys()[2]], label='{}'.format(reward_tally.keys()[2]))
    plt.legend(loc=0)

eu.print_value_maps(maze,
                    val_maps,
                    maps=0,#list(np.arange(850, 1050)), 
                    val_range=(-1,50),
                    save_dir=fig_savedir,
                    title='Value Map')

EC_policy = np.zeros((maze.y, maze.x), dtype=[('N', 'f8'), ('E', 'f8'),('W', 'f8'), ('S', 'f8'),('stay', 'f8'), ('poke', 'f8')])
ex = EC_policy[0][0]
for i in EC.cache_list.keys():
    loc = EC.cache_list[i][2]
    pol = EC.cache_list[i][0]
    t = EC.cache_list[i][1]
    if max(tuple(eu.softmax(pol))) > max(EC_policy[loc[1]][loc[0]]):
        EC_policy[loc[1]][loc[0]] = tuple(eu.softmax(pol))
    #    previous_pol = np.asarray(EC_policy[loc[1]][loc[0]])
    #    print( previous_pol,np.asarray(eu.softmax(pol) ))
    #    new_pol = tuple(eu.softmax(np.add(np.asarray(eu.softmax(pol)),previous_pol)))
    #    EC_policy[loc[1]][loc[0]] = tuple(eu.softmax(pol))

test1 = [int(po) for po in policies[1].keys()]
epoch = max(test1)
eu.make_dual_policy_plots(maze, EC_policy, policies[1][str(epoch)], savedir='PolMaps.svg')#EC_policies, MF_policies)
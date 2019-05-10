from modules import *

# Data Collection & Output Parameters
fig_savedir = '../data/figures/'
print_freq  = 1/10

# Environment Parameters
grid_params = {
    'y_height':   20,
    'x_width':    20,
    'walls':      False,
    'rho':        0,
    'maze_type':  'none',
    'port_shift': 'none'
}
#make environment
maze = eu.gridworld(grid_params)
# set custom reward location 
# maze.set_rwd([(5,15)])


# Model Parameters
agent_params = {
    'state_type':   'conv', 
    'load_model':   True
    'action_dims':  6 #=len(maze.actionlist)
    'batch_size':   1
    'gamma':        0.98 #discount factor
    'eta':          5e-4
    'temperature':  1
    'NUM_EVENTS':   100
    'NUM_TRIALS':   1
    'use_EC':       False
}

# add environment-specific parameters to dictionary 
agent_params = sg.gen_input(maze, agent_params) 

# use OpenAI gym wrapper
env = eu.gymworld(maze)



if agent_params['load_model']:
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
trial_params = {
    'maze':             maze,
    'MF':               MF,
    'use_EC':           use_EC,
    'EC':               EC,
    'cachelim':         int(0.75*np.prod(maze.grid.shape)) # memory limit should be ~75% of #actions x #states   
    'total_loss':       [[],[]], 
    'total_reward':     [],
    'val_maps':         [],
    'policies':         [{},{}], 
    'add_mem_dict':     {},
    'deltas':           [],
    'spots':            [],
    'vls':              [],
    'success_benchmark':(NUM_EVENTS -((maze.y-1)+(maze.x-1)))/NUM_EVENTS,
    'timestamp':        0
    }



# flags to be changed mid run
add_episodic_cache = False  ## Possibly unnecessary now 
if add_episodic_cache:
    rwd_threshold = True
midrun_rwd_removal = False
success_benchmark = (NUM_EVENTS -((maze.y-1)+(maze.x-1)))/NUM_EVENTS

if midrun_rwd_removal: 
    reward_tally = {}
    for _ in maze.rwd_loc: 
        reward_tally[_] = []
    trial_rwd_switch = 0
    


## run trials
run_trials(NUM_TRIALS, NUM_EVENTS, trial_params)
### 



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
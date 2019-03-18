from __future__ import division
from importlib import reload
from modules import *
fig_savedir = '../data/figures/'

grid_params = {
    'y_height':   20,
    'x_width':    20,
    'walls':      False,
    'rho':        0,
    'maze_type':  'none',
    'port_shift': 'none'
}

agent_params = {
    'load_model':   False,
    'load_dir':     '../data/outputs/gridworld/MF{}{}training.pt'.format(grid_params['x_width'],grid_params['y_height']),
    'action_dims':  6, #=len(maze.actionlist)
    'batch_size':   1,
    'gamma':        0.98, #discount factor
    'eta':          5e-4,
    'temperature':  1,
    'use_EC':       False,
    'cachelim':     300, #int(0.75*np.prod(maze.grid.shape)) # memory limit should be ~75% of #actions x #states
    'state_type':   'conv'
}

run_dict = {
    'NUM_EVENTS':   150,
    'NUM_TRIALS':   5000,
    'print_freq':   1/10,
    'total_loss':   [[],[]],
    'total_reward': [],
    'val_maps':     [],
    'policies':     [{},{}],
    'deltas':       [],
    'spots':        [],
    'vls':          []
}

#make environment
maze = eu.gridworld(grid_params)
maze.set_rwd([(int(grid_params['y_height']/2),int(grid_params['x_width']/2))])
env = eu.gymworld(maze) # openAI-like wrapper

#update agent params dictionary with layer sizes appropriate for environment
agent_params = sg.gen_input(maze, agent_params)

MF,opt = ac.make_agent(agent_params)

EC = ec.ep_mem(MF,agent_params['cachelim'])


# write a function for runs with episodic mem and without -- take use_EC as a param
# assume just for conv inputs
def run_trials(run_dict, use_EC, **kwargs):
    save_data = kwargs.get('save', True)
    NUM_TRIALS = run_dict['NUM_TRIALS']
    NUM_EVENTS = run_dict['NUM_EVENTS']

    if not use_EC:
        blocktime = time.time()
        # ==================================
        # Run Trial
        # ==================================
        for trial in range(NUM_TRIALS):
            reward_sum = 0
            v_last = 0
            track_deltas = []
            track_spots = []
            visited_locs = []

            env.reset()
            state = ac.Variable(ac.torch.FloatTensor(sg.get_frame(maze)))
            MF.reinit_hid()  # reinit recurrent hidden layers

            for event in range(NUM_EVENTS):
                policy_, value_ = MF(state, agent_params['temperature'])[0:2]
                choice, policy, value = ac.select_action(MF, policy_, value_)

                if event < NUM_EVENTS:
                    next_state, reward, done, info = env.step(choice)

                MF.rewards.append(reward)
                delta = reward + agent_params['gamma'] * value - v_last  # compute eligibility trace/rpe approximation
                state = ac.Variable(ac.torch.FloatTensor(sg.get_frame(maze)))

                reward_sum += reward
                v_last = value

            p_loss, v_loss = ac.finish_trial(MF, agent_params['gamma'], opt)

            if save_data:
                # value_map = ac.generate_values(maze,MF)
                run_dict['total_loss'][0].append(p_loss.data[0])
                run_dict['total_loss'][1].append(v_loss.data[0])
                run_dict['total_reward'].append(reward_sum)
                # run_dict['val_maps'].append(value_map.copy())
                # run_dict['deltas'].append(track_deltas)
                # run_dict['spots'].append(track_spots)
                # run_dict['vls'].append(visited_locs)

            if trial == 0 or trial % 100 == 0 or trial == NUM_TRIALS - 1:
                # EC_policies, MF_policies = ac.generate_values(maze, MF,EC=EC)
                # run_dict['policies'][0]['{}'.format(trial)] = EC_policies
                # run_dict['policies'][1]['{}'.format(trial)] = MF_policies
                print("[{0}]  Trial {1} TotRew = {2} ({3:.3f}s)".format(time.strftime("%H:%M:%S", time.localtime()),
                                                                        trial + 1, reward_sum,
                                                                        time.time() - blocktime))  # print("[{0}]  Trial {1} total reward = {2} (Avg {3:.3f})".format(time.strftime("%H:%M:%S", time.localtime()), trial, reward_sum, float(reward_sum)/float(NUM_EVENTS)), "Block took {0:.3f}".format(time.time()-blocktime))
                blocktime = time.time()


a1 = run_trials(run_dict, False)

plt.plot(run_dict['total_reward'])
plt.show()

ac.torch.save(MF,agent_params['load_dir'])
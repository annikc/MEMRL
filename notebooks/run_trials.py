from __future__ import division
from modules import *
import uuid
import pymysql
from influxdb import InfluxDBClient

fig_savedir = '../data/figures/'

# write a function for runs with episodic mem and without -- take use_EC as a param
# assume just for conv inputs
def run_trials(run_dict, agent_params, grid_params, **kwargs):
    save_data  = kwargs.get('save', True)
    alpha      = kwargs.get('alpha', None)  ## parameter for relative contribution of policy loss and value loss
    conn       = kwargs.get('conn', run_dict['conn'])
    NUM_TRIALS = run_dict['NUM_TRIALS']
    NUM_EVENTS = run_dict['NUM_EVENTS']
    maze       = agent_params['maze']
    env        = eu.gymworld(maze)

    if run_dict['track_events']:
        cur = conn.cursor()
        # make id for storing in database
        db_id = uuid.uuid4()
        print(f"Run id is {db_id}")
        db_name = time.strftime("%Y%b%d_%H%M%S", time.localtime())
        client = InfluxDBClient('localhost', 8086, 'root', 'root', 'mdp')
        event_track = InfluxDBClient('localhost', 8086, 'root', 'root', 'sarv')
        db_data = [{
            "measurement": "run",
            "tags": {
                "id": f"{db_id}",
                "environment": "gridworld",
            },
            "fields": {
                "tot_rwd": 0,
                "pol_loss": 0,
                "val_loss": 0,
                "tr_no": 0
            }
        }
        ]

        # change x_width etc to be from appropriate dictionary
        insert = f" INSERT INTO mdp (`report_uuid`,`x_width`, `y_height`, `walls`, `rho`, `maze_type`, `port_shift`, `gamma`, `eta`, `num_events`, `num_trials`) VALUES ('{db_id}', {grid_params['x_width']}, {grid_params['y_height']}, {grid_params['walls']}, {grid_params['rho']}, '{grid_params['maze_type']}', '{grid_params['port_shift']}', {agent_params['gamma']}, {agent_params['eta']}, {run_dict['NUM_EVENTS']}, {run_dict['NUM_TRIALS']});"
        cur.execute(insert)

    MF, opt = ac.make_agent(agent_params)

    blocktime = time.time()

    if agent_params['use_EC']:
        EC = ec.ep_mem(MF, agent_params['cachelim'])

        add_mem_dict = {}  # dictionary of items which get put into memory cache
        timestamp = 0

        for trial in range(NUM_TRIALS):
            trialstart_stamp = timestamp

            reward_sum = 0
            v_last = 0

            env.reset()
            state = ac.Variable(ac.torch.FloatTensor(np.expand_dims(sg.get_frame(maze), axis=0)))
            MF.reinit_hid()  # reinit recurrent hidden layers

            db_event_data = []

            for event in range(NUM_EVENTS):
                # pass state through EC module
                policy_, value_ = MF(state, agent_params['temperature'])[0:2]
                add_mem_dict['state'] = maze.cur_state

                '''
                NEED TO USE CONFIDENCE SCORE HERE
                - track time since last reward
                - 

                '''

                choice, policy, value = ac.select_action(MF, policy_, value_)

                if event < NUM_EVENTS:
                    next_state, reward, done, info = env.step(choice)

                if run_dict['track_events']:
                    db_event_data.append({
                        "measurement": "run",
                        "tags": {
                            "id": f"{db_id}",
                            "environment": "gridworld",
                            "tr_no": trial,
                            "event_no": event
                        },
                        "fields": {
                            "state_y": maze.cur_state[0],
                            "state_x": maze.cur_state[1],
                            "action": choice,
                            "reward": reward,
                            "value": value[0]
                        }
                    })

                MF.rewards.append(reward)
                delta = reward + agent_params['gamma'] * value - v_last  # compute eligibility trace/rpe approximation

                add_mem_dict['activity'] = tuple(lin_act_.view(-1).data)
                add_mem_dict['action'] = choice
                add_mem_dict['delta'] = delta
                add_mem_dict['timestamp'] = timestamp
                EC.add_mem(add_mem_dict, keep_hist=True)  # add event to memory cache

                if reward != 0:
                    EC.reward_update(trialstart_stamp, timestamp, reward)
                # EC.reward_update(trialstart_stamp, timestamp, delta[0])

                state = ac.Variable(ac.torch.FloatTensor(sg.get_frame(maze)))

                reward_sum += reward
                v_last = value

                timestamp += 1

            p_loss, v_loss = ac.finish_trial(MF, agent_params['gamma'], opt)

            if save_data:
                # value_map = ac.generate_values(maze,MF)
                run_dict['total_loss'][0].append(p_loss.item())
                run_dict['total_loss'][1].append(v_loss.item())
                run_dict['total_reward'].append(reward_sum)
                # run_dict['val_maps'].append(value_map.copy())
                # run_dict['deltas'].append(track_deltas)
                # run_dict['spots'].append(track_spots)
                # run_dict['vls'].append(visited_locs)

            if run_dict['track_events']:
                db_data[0]["fields"]["tot_rwd"] = reward_sum
                db_data[0]["fields"]["pol_loss"] = p_loss.data[0]
                db_data[0]["fields"]["val_loss"] = v_loss.data[0]
                db_data[0]["fields"]["tr_no"] = trial

                client.write_points(db_data)

            if trial == 0 or trial % 100 == 0 or trial == NUM_TRIALS - 1:
                print("[{0}]  Trial {1} TotRew = {2} ({3:.3f}s)".format(time.strftime("%H:%M:%S", time.localtime()),
                                                                        trial + 1, reward_sum,
                                                                        time.time() - blocktime))  # print("[{0}]  Trial {1} total reward = {2} (Avg {3:.3f})".format(time.strftime("%H:%M:%S", time.localtime()), trial, reward_sum, float(reward_sum)/float(NUM_EVENTS)), "Block took {0:.3f}".format(time.time()-blocktime))
                blocktime = time.time()

    else:
        for trial in range(NUM_TRIALS):
            reward_sum = 0
            v_last = 0

            env.reset()
            state = ac.Variable(ac.torch.FloatTensor(sg.get_frame(maze)))
            MF.reinit_hid()  # reinit recurrent hidden layers

            db_event_data = []

            for event in range(NUM_EVENTS):
                policy_, value_ = MF(state, agent_params['temperature'])[0:2]
                choice, policy, value = ac.select_action(MF, policy_, value_)

                if event < NUM_EVENTS:
                    next_state, reward, done, info = env.step(choice)

                if run_dict['track_events']:
                    db_event_data.append({
                        "measurement": "run",
                        "tags": {
                            "id": f"{db_id}",
                            "environment": "gridworld",
                            "tr_no": trial,
                            "event_no": event
                        },
                        "fields": {
                            "state_y": maze.cur_state[0],
                            "state_x": maze.cur_state[1],
                            "action": choice,
                            "reward": reward,
                            "value": value[0]
                        }
                    })

                MF.rewards.append(reward)
                delta = reward + agent_params['gamma'] * value - v_last  # compute eligibility trace/rpe approximation
                state = ac.Variable(ac.torch.FloatTensor(sg.get_frame(maze)))

                reward_sum += reward
                v_last = value

            p_loss, v_loss = ac.finish_trial(MF, agent_params['gamma'], opt, alpha=alpha)

            if save_data:
                # value_map = ac.generate_values(maze,MF)
                run_dict['total_loss'][0].append(p_loss.item())
                run_dict['total_loss'][1].append(v_loss.item())
                run_dict['total_reward'].append(reward_sum)
                # run_dict['val_maps'].append(value_map.copy())
                # run_dict['deltas'].append(track_deltas)
                # run_dict['spots'].append(track_spots)
                # run_dict['vls'].append(visited_locs)

            if run_dict['track_events']:
                db_data[0]["fields"]["tot_rwd"] = reward_sum
                db_data[0]["fields"]["pol_loss"] = p_loss.data[0]
                db_data[0]["fields"]["val_loss"] = v_loss.data[0]
                db_data[0]["fields"]["tr_no"] = trial

                client.write_points(db_data)

            # if run_dict['track_events']:
            #    event_track.write_points(db_event_data)

            if trial == 0 or trial % 100 == 0 or trial == NUM_TRIALS - 1:
                print(f"[{time.strftime('%H:%M:%S',time.localtime())}]  Trial {trial + 1} TotRew = {reward_sum} ({time.time() - blocktime:.3f}s)")
                blocktime = time.time()

    return db_id
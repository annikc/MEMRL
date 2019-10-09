dicts are stored with the following commands 

grid_dict = {}
for prop, propval in vars(maze).iteritems():
    grid_dict[prop]= propval
store_data= {}
store_data['total_reward'] = total_reward
store_data['total_loss'] = total_loss
store_data['val_maps'] = val_maps
store_data['EC'] = EC.cache_list
store_data['deltas'] = deltas
store_data['policies'] = [MF_policies, EC_policies]
store_data['params'] = {'grid':grid_dict,
                       'state_type':state_type,
                       'discount_factor': discount_factor,
                       'eta':eta, 
                       'runtime':[NUM_EVENTS, NUM_TRIALS]}

datestamp = time.strftime("%y%m%d_%H%M", time.localtime())
np.save('../data/outputs/gridworld/pydicts/{}.npy'.format(datestamp), store_data)


to load use: 
read_dictionary = np.load('my_file.npy').item()
import os
from modules.Agents.Networks.load_network_weights import convert_agent_to_weight_dict

path_from_here = '../../Data/network_objs/'
destination_path = '../../Data/agents/'

items = os.listdir(path_from_here)
for x in items[0:10]:
    convert_agent_to_weight_dict(path_from_here+x, destination_path=destination_path+x)
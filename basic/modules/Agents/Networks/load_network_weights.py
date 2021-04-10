import numpy as np
import torch


def load_saved_head_weights(AC_head_agent, path_to_saved_agent):
    # load in agent weights for full conv network
    saved_network = torch.load(path_to_saved_agent)

    new_state_dict = {}
    for key in saved_network.state_dict().keys():
        if key[0:6] == 'output':
            if key[7] == '0':
                new_key = 'pol'+key[8:]
                new_state_dict[new_key] = saved_network.state_dict()[key]
            elif key[7] == '1':
                new_key = 'val'+key[8:]
                new_state_dict[new_key] = saved_network.state_dict()[key]

    AC_head_agent.load_state_dict(new_state_dict)
    return AC_head_agent

from .annik_ac import ActorCritic

from .cnn import CNN_AC
from .cnn_2n import CNN_2N

from .fcx2 import FullyConnected_AC as FC
from .fcx2_2n import FullyConnected_2N as FC2N

## write general network class which takes standard input

class Network(object):
    def __init__(self):
        pass
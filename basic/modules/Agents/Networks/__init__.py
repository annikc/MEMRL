import gym
from .annik_ac import perceptron_AC as flat_ActorCritic
from .annik_ac import shallow_AC_network as shallow_ActorCritic
from .annik_ac import fully_connected_AC_network as fc_ActorCritic
from .annik_ac import flex_ActorCritic

from .DQN import DQN

# for actor critic agent
class conv_FO_params(object): # fully observable = input tensor includes channel for reward information
    def __init__(self, env):
        self.input_dims = (3,*env.shape)
        self.action_dims = env.action_space.n
        self.hidden_types = ['conv', 'pool', 'conv', 'pool', 'linear', 'linear']
        self.hidden_dims = [None, None, None, None, 400, 200] #first fc layer was 600
        self.lr = 5e-4

    def get_input_dims(self,env):
        if isinstance(env.observation_space, gym.spaces.box.Box):
            print(f'box: {env.observation_space.shape}')
            input_dims = env.observation_space.shape[0]
        elif isinstance(env.observation_space, gym.spaces.discrete.Discrete):
            print(f'discrete: {env.observation_space.n}')
            input_dims = env.observation_space.n
        else:
            input_dims = env.observation_space.shape

        return input_dims

# for actor critic agent
class conv_PO_params(object): # partially observable = input tensor contains env.obstacle and state channels only
    def __init__(self, env):
        self.input_dims = (2,*env.shape)
        self.action_dims = env.action_space.n
        self.hidden_types = ['conv', 'pool', 'conv', 'pool', 'linear', 'linear']
        self.hidden_dims = [None, None, None, None, 400, 200] # fc layers were 600, 400
        self.lr = 5e-4

    def get_input_dims(self,env):
        if isinstance(env.observation_space, gym.spaces.box.Box):
            print(f'box: {env.observation_space.shape}')
            input_dims = env.observation_space.shape[0]
        elif isinstance(env.observation_space, gym.spaces.discrete.Discrete):
            print(f'discrete: {env.observation_space.n}')
            input_dims = env.observation_space.n
        else:
            input_dims = env.observation_space.shape

        return input_dims

class fc_params(conv_PO_params):
    def __init__(self, env):
        super().__init__(env)
        self.input_dims = env.nstates
        self.hidden_types = ['linear', 'linear']
        self.hidden_dims = [200, 200]
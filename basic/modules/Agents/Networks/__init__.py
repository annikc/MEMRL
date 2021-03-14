import gym

from .annik_ac import fully_connected_AC_network as ActorCritic
from .annik_ac import flex_ActorCritic

from .DQN import DQN

# for actor critic agent
class params(object):
    def __init__(self, env):
        self.input_dims = self.get_input_dims(env)
        self.action_dims = env.action_space.n
        self.hidden_types = ['conv', 'pool', 'conv', 'pool', 'linear', 'linear']
        self.hidden_dims = [None, None, None, None, 1000, 1000]
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

class fc_params(params):
    def __init__(self, env):
        super(fc_params, self).__init__(env)
        self.input_dims = env.nstates
        self.hidden_types = ['linear', 'linear']
        self.hidden_dims = [200, 200]
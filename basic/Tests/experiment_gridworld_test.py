from basic.Experiments import Experiment as ex
import gym
from basic.Agents.Networks.annik_ac import ActorCritic as Network
from basic.Agents.EpisodicMemory import EpisodicMemory as Memory
from basic.Agents import Agent


class basic_agent_params():
    def __init__(self, env):
        self.load_model = False
        self.load_dir   = ''
        self.architecture = 'A'
        self.input_dims = env.observation.shape
        self.action_dims = 4
        self.hidden_types = ['conv', 'pool', 'conv', 'pool', 'linear', 'linear']
        self.hidden_dims = [None, None, None, None, 1000, 1000]
        self.freeze_w = False
        self.rfsize = 5
        self.gamma = 0.98
        self.eta = 5e-4

env = gym.make('gym_grid:gridworld-v1')

agent_params = basic_agent_params(env)
agent = Agent(network=Network(agent_params.__dict__) )#, memory=Memory(entry_size=env.action_space.n, cache_limit=env.nstates))

run = ex(agent,env)
run.run(100,100, printfreq=10)
# TODO: loading parameters with yaml file?
# TODO: logging data
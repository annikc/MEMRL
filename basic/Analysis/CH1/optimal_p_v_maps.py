import numpy as np
import gym
import networkx as nx
from modules.Utils.gridworld_plotting import opt_pol_map, plot_pref_pol
from Analysis.analysis_utils import make_env_graph, compute_graph_distance_matrix

env_name = 'gridworld:gridworld-v4'
env = gym.make(env_name)

G = make_env_graph(env)
sp = nx.shortest_path(G)
gd = compute_graph_distance_matrix(G, env)

print(sp[0][105])
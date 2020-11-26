import numpy as np
import matplotlib.pyplot as plt


# for actor critic agent
class basic_agent_params(object):
    def __init__(self, env):
        self.input_dims = env.observation_space.shape
        self.action_dims = env.action_space.n
        self.hidden_types = ['conv', 'pool', 'conv', 'pool', 'linear', 'linear']
        self.hidden_dims = [None, None, None, None, 1000, 1000]
        self.lr = 5e-4


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def softmax(x, T=1):
	e_x = np.exp((x - np.max(x))/T)
	return np.round(e_x / e_x.sum(axis=0),8)

def discount_rwds(r, gamma = 0.99):
	disc_rwds = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		running_add = running_add*gamma + r[t]
		disc_rwds[t] = running_add
	return disc_rwds

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)




def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-10):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 10 scores')
    plt.savefig(figure_file)
    
    
    


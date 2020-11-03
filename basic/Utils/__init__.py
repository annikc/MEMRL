import numpy as np

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
    
    
    


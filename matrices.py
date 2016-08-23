import numpy as np 

fat = np.zeros(shape=(3,3))

for i in range(fat.shape[0]):
	for j in range(fat.shape[1]):
		fat[i,j] = np.random.rand(1)
	fat[i] = fat[i]/sum(fat[i])
	# this gives a Dirichlet distribution; each row is an independent Dirichlet distr. w/ alpha_i = 1 
# can also use np.
print fat

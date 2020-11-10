import basic.Agents.Networks as nets
lr = 0.01
input_dim = 4
fc1_dims = 30
fc2_dims = 30
n_actions = 4

test = nets.FC(lr, input_dim, fc1_dims, fc2_dims, n_actions)
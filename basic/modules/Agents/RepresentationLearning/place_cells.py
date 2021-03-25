import numpy as np
import matplotlib.pyplot as plt

class place_cells(object):
    def __init__(self, env_shape, num_cells, field_size, **kwargs):
        self.env_shape = env_shape
        self.num_cells = num_cells
        self.field_size = field_size
        self.cell_centres = kwargs.get('cell_centres',self.get_cell_centres())

    def get_cell_centres(self):
        cell_centres = []
        for i in range(self.num_cells):
            x, y = np.random.random(2)
            cell_centres.append((x,y))
        return np.asarray(cell_centres)

    # define normalized 2D gaussian
    def gaus2d(self, state):
        x  = state[0]
        y  = state[1]
        mx = self.cell_centres[:,1]
        my = self.cell_centres[:,0]
        sx = sy = self.field_size
        return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

    def get_activities(self, states):
        # TODO - array operation
        activities = []
        for state in states:
            xy_state = (state[1]/self.env_shape[1], state[0]/self.env_shape[0]) # input state as (x, y) in [0,1] i.e. portion of total grid area
            place_cell_activity = self.gaus2d(xy_state)
            activities.append(place_cell_activity)
        return np.asarray(activities)

    def plot_placefields(self, env_states_to_map, num_pc_view=9):
        states = np.asarray(env_states_to_map)
        gridworld_pc_reps = self.get_activities(env_states_to_map)
        get_rand_cells = np.random.choice(self.env_shape[0]*self.env_shape[1],num_pc_view,replace=False)

        num_rows = int(np.ceil(np.sqrt(num_pc_view)))
        num_col = int(np.ceil(np.sqrt(num_pc_view)))
        fig, axes = plt.subplots(num_rows,num_col)

        for i, ax in enumerate(axes.flat):
            if i >= num_pc_view:
                ax.axis('off')
            else:
                # for every state, get what the place cell activity is
                ax.scatter(states[:,1],states[:,0], c =gridworld_pc_reps[:,get_rand_cells[i]])
                cell_center = np.round(np.multiply(self.cell_centres[get_rand_cells[i]],self.env_shape),1)
                ax.set_title(f'{cell_center}')
                ax.set_yticks([])
                ax.set_xticks([])
                ax.invert_yaxis()
                ax.set_aspect('equal')
        #plt.show()
        return get_rand_cells


'''
# make an imaginary environment to generate place cells over
env_shape = (10,10)

# get place cells
num_place_cells = 20
pc = place_cells(env_shape=env_shape,num_cells=num_place_cells, field_size=0.2)


# generate some states to get place cell activity for
num_states = 300
r_coords = np.random.sample(size=num_states)*env_shape[0]
c_coords = np.random.sample(size=num_states)*env_shape[1]
states = np.asarray([(r,c) for r,c in zip(r_coords,c_coords)])

# get place cell activities for those states
activities = pc.get_activities(states)
print(f'state {states[0]}, PC activity {activities}')
print(activities.shape)

num_pc_view = 9
get_rand_cells = np.random.choice(num_place_cells,num_pc_view,replace=False)
print(get_rand_cells)
fig, axes = plt.subplots(3,3)
for i, ax in enumerate(axes.flat):
    ax.scatter(states[:,0],states[:,1], c =activities[:,get_rand_cells[i]])
    cell_center = np.round(np.multiply(pc.cell_centres[get_rand_cells[i]],env_shape),2)
    print(cell_center)
    ax.set_title(f'{cell_center}')
    ax.set_yticks([])
    ax.set_xticks([])
plt.show()

'''







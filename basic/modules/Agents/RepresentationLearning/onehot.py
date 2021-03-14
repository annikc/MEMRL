import numpy as np
import matplotlib.pyplot as plt

class onehot_cells(object):
    def __init__(self, num_cells):
        self.num_cells = num_cells

    def get_activities(self, states):
        activities = []
        for state in states:
            vec = np.zeros(self.num_cells)
            vec[state] = 1
            activities.append(vec)

        return np.asarray(activities)

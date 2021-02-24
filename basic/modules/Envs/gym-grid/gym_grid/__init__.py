from gym.envs.registration import register

register(
    id='gridworld-v0',
    entry_point='gym_grid.envs:GridWorld',
)

register(
    id='gridworld-v1',
    entry_point='gym_grid.envs:GridWorld4',
)

register(
    id='gridworld-v11',
    entry_point='gym_grid.envs:GridWorld4_movedR',
)

register(
    id='gridworld-v111',
    entry_point='gym_grid.envs:MiniGrid',
)

register(
    id='gridworld-v2',
    entry_point='gym_grid.envs:GridWorld4_random_obstacle',
)

register(
    id='gridworld-v3',
    entry_point='gym_grid.envs:GridWorld4_rooms',
)

register(
    id='gridworld-v4',
    entry_point='gym_grid.envs:GridWorld4_bar',
)
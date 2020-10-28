from gym.envs.registration import register

register(
    id='gridworld-v0',
    entry_point='gym_grid.envs:GridWorld',
)

register(
    id='gridworld-v1',
    entry_point='gym_grid.envs:GridWorld4',
)
from gym.envs.registration import register

register(
    id='gridworld-v0',
    entry_point='gridworld.environments:GridWorld',
)

register(
    id='gridworld-v1',
    entry_point='gridworld.environments:GridWorld4',
)

register(
    id='gridworld-v2',
    entry_point='gridworld.environments:GridWorld4_random_obstacle',
)

register(
    id='gridworld-v3',
    entry_point='gridworld.environments:GridWorld4_rooms',
)

register(
    id='gridworld-v4',
    entry_point='gridworld.environments:GridWorld4_bar',
)

register(
    id='gridworld-v5',
    entry_point='gridworld.environments:GridWorld4_tunnel',
)

register(
    id='gridworld-v6',
    entry_point='gridworld.environments:GridWorld4_hairpin',
)

### variations on gridworld V1
register(
    id='gridworld-v11',
    entry_point='gridworld.environments:GridWorld4_movedR',
)

register(
    id='gridworld-v111',
    entry_point='gridworld.environments:MiniGrid',
)
register(
    id='gridworld-v112',
    entry_point='gridworld.environments:LinearTrack',
)
register(
    id='gridworld-v1122',
    entry_point='gridworld.environments:LinearTrack_1',
)

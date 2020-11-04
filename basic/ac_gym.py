# =====================================
#         Runs OpenAI Gyms
# =====================================

import gym
from Agents.Networks import cnn, fcx2, fcx2_2n
from Agents import DualStream as Agent

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # =====================================
    #       Environment Listings
    # =====================================
    # CartPole-v1 = 195, MontainCar = -110 LunarLander-v2 = 200, BipedalWalker-v2/Hardcore = 300, CarRacing-v0 = 900
    # Classic Control
    CartPole = ['CartPole-v1', 'fc', 195] # learning
    MountainCar = ['MountainCar-v0', 'fc', -110]  # not learning
    Pendulum = ['Pendulum-v0', 'fc', None]

    # Box2D
    LunarLander = ['LunarLander-v2', 'fc', 200] # learning
    BipedalWalker = ['BipedalWalker-v2', 'fc', 300]
    BipedalWalkerHardCore = ['BipedalWalkerHardCore-v2', 'fc', 300]
    CarRacing = ['CarRacing-v0', 'fc', 900]

    # Atari
    pong = ['PongNoFrameskip-v4', 'cnn', None]

    # =====================================
    #   Choose Environment From Listings
    # =====================================
    use_environment = LunarLander
    beat_environment = True # if true will continue until avg_reward from last 100 episodes >= use_environemnt[2]
    n_episodes = 1000
    # creates environment
    if use_environment[1] == 'fc':
        env = gym.make(use_environment[0])
    else:
        env = make_env(use_environment[0])

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    # setup networks
    # Note: for agent_mc value_network output_dims = 1, for agent_td value_network output_dims = n_actions

    policy_value_network = fcx2.Network(lr=0.001, input_dim=state_shape, fc1_dims=100, fc2_dims=100, n_actions=n_actions)

    # setup networks
    policy_network = fcx2_2n.Network(lr=0.006, input_dim=state_shape, fc1_dims=50, fc2_dims=30, output_dims=n_actions)
    value_network = fcx2_2n.Network(lr=0.01, input_dim=state_shape, fc1_dims=50, fc2_dims=30, output_dims=1)


    # Setup Agent
    #agent = Agent(network = policy_value_network)
    agent = Agent(policy_network=policy_network, value_network=value_network)

    # Training parameters
    score_history = []
    ploss = []
    vloss = []
    n_episodes = 100

    # create named tuple for storing transitions

    for episode in range(n_episodes):
        done = False
        score = 0
        state = env.reset()
        transition_num = 1
        while not done:
            action, log_prob, expected_value = agent.get_action(state) # choose action to take
            next_state, reward, done, info = env.step(action) # get info from taking that action
            score += reward

            target_value = 0

            # Transitions are stored in named tuples
            agent.log_event(episode,transition_num,
                            state, action, reward, next_state,
                            log_prob, expected_value, target_value,
                            done, readable_state=(0,0))
            state = next_state

            transition_num += 1

        if not agent.TD:
            p, v = agent.finish_()

        print(f"Episode: {episode}, Score: {score}")
        score_history.append(score)
        ploss.append(p)
        vloss.append(v)

    # Output Data
    plot_name = 'cartpole'
    figure_file = 'Data/plots/' + plot_name
    x = [i+1 for i in range(n_episodes)]
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(score_history)
    ax[1].plot(ploss, label='p')
    ax[1].plot(vloss, label='v')
    ax[1].legend(bbox_to_anchor=(1.05, 0.95))

    plt.show()
    #plot_learning_curve(x, score_history, figure_file)

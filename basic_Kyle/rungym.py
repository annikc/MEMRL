# =====================================
#         Runs OpenAI Gyms
# =====================================

import gym
from Agents.agent_mc import Agent_MC
from Agents.agent_td import Agent_TD
from Agents.agent_mc_2n import Agent_MC_2N
from Utils.utils import plot_learning_curve
from Utils.make_envs import make_env
from Agents.Transition_Cache.transition_cache import Transition_Cache
from Agents.Networks import fcx2, fcx2_2n, cnn, cnn_2n
from collections import namedtuple


if __name__ == '__main__':
    Transition = namedtuple('Transition', 'episode, transition, state, action, reward, \
                                next_state, log_prob, expected_value, target_value, done')
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
    
    # =====================================
    #              Setup Networks
    # =====================================
    #--- Networks for single network agents ---
    # fully connected
    if use_environment[1] == 'fc':
        policy_value_network = fcx2.Network(lr=0.06, input_dim=state_shape, fc1_dims=30, fc2_dims=30, n_actions=n_actions)
        policy_network = fcx2_2n.Network(lr=0.006, input_dim=state_shape, fc1_dims=30, fc2_dims=30, output_dims=n_actions)
        value_network = fcx2_2n.Network(lr=0.01, input_dim=state_shape, fc1_dims=30, fc2_dims=30, output_dims=1)
    else:
        policy_value_network = cnn.Network(lr=0.06, input_dims=state_shape, fc1_dims=30, n_actions=n_actions)
        policy_network = cnn_2n.Network(lr=0.006, input_dims=state_shape, fc1_dims=30, output_dims=n_actions)
        value_network = cnn_2n.Network(lr=0.01, input_dims=state_shape, fc1_dims=30, output_dims=1)
        
    # =====================================
    #              Setup Agents
    # =====================================
    agent_1n = Agent_MC(policy_value_network=policy_value_network, trans_cache_size = 100000, gamma = 0.99, TD = False)
    agent_2n = Agent_MC_2N(policy_network=policy_network, value_network=value_network, 
                    trans_cache_size = 100000, gamma = 0.99, TD = False)

    # =====================================
    #      Choose Agent For Training
    # =====================================
    agent = agent_2n

    score_history = []
    def learning_loop(episode):
        done = False
        score = 0 
        state = env.reset()
        transition_num = 1
        while not done:
            action, log_prob, expected_value = agent.get_action(state) # choose action to take
            next_state, reward, done, info = env.step(action) # get info from taking that action
            score += reward
            target_value = 0
            transition = Transition(episode=episode, transition=transition_num, state=state, action=action, 
                                                    reward=reward, next_state=next_state, log_prob=log_prob, 
                                                    expected_value=expected_value, target_value=target_value, done=done)
            agent.store_transition(transition) 
            state = next_state
            if agent.TD:
                agent.learn()

            transition_num += 1
        
        if not agent.TD:
            agent.learn()
            agent.clear_transition_cache()

        return score
        
    if not beat_environment:
        for episode in range(n_episodes):
            score = learning_loop(episode)
            score_history.append(score)
            if len(score_history) < 100:
                average = sum(score_history[-100:]) / len(score_history)
                print(f"Episode: {episode}, Score: {score}, Avg: {average}")
            else:
                average = sum(score_history[-100:]) / 100
                beat_environment = average <= use_environment[2]
                print(f"Episode: {episode}, Score: {score}, 100Avg: {average}")
            
    else:
        while beat_environment:
            episode = len(score_history) + 1
            score = learning_loop(episode)
            score_history.append(score)
            if len(score_history) >= 100:
                average = sum(score_history[-100:]) / 100
                beat_environment = average <= use_environment[2]
                print(f"Episode: {episode}, Score: {score}, 100Avg: {average}")
            else:
                average = sum(score_history[-100:]) / len(score_history)
                print(f"Episode: {episode}, Score: {score}, Avg: {average}")
                 # when false is returned the environment is beat

    env.render()

    # plot training data
    figure_file = 'Data/plots/' + use_environment[0]
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    
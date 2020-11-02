# =====================================
#         Runs OpenAI Gyms
# =====================================

import gym
from Agents.agent_mc import Agent_MC
from Utils.utils import plot_learning_curve
from Utils.make_envs import make_env
from Agents.Transition_Cache.transition_cache import Transition_Cache
from Agents.Networks import cnn, fcx2
from collections import namedtuple


if __name__ == '__main__':
    # Setup gym environment
    env = gym.make('CartPole-v1')  # gym.make('LunarLander-v2'), make_env('PongNoFrameskip-v4')
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    # setup networks
    policy_network = fcx2.Network(lr=0.006, input_dim=state_shape, fc1_dims=50, fc2_dims=30, output_dims=n_actions)
    value_network = fcx2.Network(lr=0.01, input_dim=state_shape, fc1_dims=50, fc2_dims=30, output_dims=1)

    # Setup Agent
    agent = Agent_MC(policy_network=policy_network, value_network=value_network, 
                    trans_cache_size = 100000, gamma = 0.99, TD = False)

    # Training parameters 
    score_history = []
    n_episodes = 200

    # create named tuple for storing transitions
    Transition = namedtuple('Transition', 'episode, transition, state, action, reward, \
                                new_state, log_prob, expected_value, target_value, done')

    for episode in range(n_episodes):
        done = False
        score = 0 
        state = env.reset()
        transition_num = 1
        while not done:
            action, log_prob, expected_value = agent.get_action(state) # choose action to take
            new_state, reward, done, info = env.step(action) # get info from taking that action
            score += reward
            if agent.TD:
                expected_value = agent.get_expected_value(new_state)
                agent.learn()

            target_value = 0
            agent.current_transition = Transition(episode=episode, transition=transition_num, state=state, action=action, 
                                                    reward=reward, new_state=new_state, log_prob=log_prob, 
                                                    expected_value=expected_value, target_value=target_value, done=done)
            agent.store_transition() # encode information about the step
            state = new_state
            
            if agent.TD:
                agent.learn()

            transition_num += 1
        
        if not agent.TD:
            agent.learn()
            agent.clear_transition_cache()

        print(f"Episode: {episode}, Score: {score}")
        score_history.append(score)

    # Output Data
    plot_name = 'cartpole'
    figure_file = 'Data/plots/' + plot_name
    x = [i+1 for i in range(n_episodes)]
    plot_learning_curve(x, score_history, figure_file)
    
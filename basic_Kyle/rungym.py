# =====================================
#         Runs OpenAI Gyms
# =====================================

import gym
from Agents.Agent import Agent
from Utils.utils import plot_learning_curve
from Utils.make_envs import make_env
from Agents.Memory.transition_memory import Transition_Memory
from Agents.Networks import networks
from Agents.Learning import learing_algorithms
from Agents.ActionSelection import action_selection


if __name__ == '__main__':
    # Setup gym environment
    env = gym.make('CartPole-v1')  # gym.make('LunarLander-v2'), make_env('PongNoFrameskip-v4')
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    # setup networks
    policy_network = networks.FCx2(lr=0.006, input_dim=state_shape, fc1_dims=50, fc2_dims=50, n_actions=n_actions)
    value_network = networks.FCx2(lr=0.01, input_dim=state_shape, fc1_dims=50, fc2_dims=50, n_actions=1)

    #setup action_selection algorith
    action_selection = action_selection.SoftMax()

    # setup learning algorithm 
    learning_algorithm = learing_algorithms.Monte_Carlo_Learning(gamma=0.99)
    TD_learning = False
    MC_learning = True
    
    # setup memory systems
    transition_memory = Transition_Memory(mem_size=100000, input_dims=state_shape)

    # Setup Agent
    agent = Agent(policy_network=policy_network, value_network=value_network, action_selection=action_selection, 
                    transition_memory=transition_memory, learning_algorithm=learning_algorithm)

    # Training parameters 
    score_history = []
    n_episodes = 5

    for episode in range(n_episodes):
        done = False
        score = 0 
        state = env.reset()
        transition_num = 0
        while not done:
            transition_num += 1
            action, log_prob, model_value = agent.get_action(state) # choose action to take
            new_state, reward, done, info = env.step(action) # get info from taking that action
            score += reward
            agent.store_transition(state, action, reward, new_state, log_prob, model_value, done) # encode information about the step
            state = new_state
            
            if TD_learning:
                agent.learn()
        
        if MC_learning:
            agent.learn()
            agent.clear_transition_memory()

        print(f"Episode: {episode}, Score: {score}")
        score_history.append(score)

    # Output Data
    plot_name = 'cartpole'
    figure_file = 'Data/plots/' + plot_name
    x = [i+1 for i in range(n_episodes)]
    plot_learning_curve(x, score_history, figure_file)
    
# =====================================
#         Runs OpenAI Gyms
# =====================================

import gym
from Agents.acmc import Agent_fcx2
from Agents.acmc_cnn import Agent_cnn
from Utils.utils import plot_learning_curve
from Utils.make_envs import make_env


if __name__ == '__main__':
    # env = gym.make('LunarLander-v2')  # For testing Agent_fcx2
    env = gym.make('CartPole-v1')  # For testing Agent_fcx2
    # env = make_env('PongNoFrameskip-v4')
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    
    agent = Agent_fcx2(alpha=0.006, beta=0.01, input_dims=state_shape, gamma=0.99, n_actions=n_actions, l1_size=50, l2_size=50)
    #agent = Agent_cnn(alpha=0.006, beta=0.01, input_dims=(state_shape), gamma=0.99, n_actions=n_actions, l1_size=50, l2_size=512)

    filename = 'Pong'
    figure_file = 'plots/' + filename

    score_history = []
    n_episodes = 5

    for episode in range(n_episodes):
        done = False
        score = 0 
        state = env.reset()
        while not done:
            action, log_prob, critic_value = agent.get_action_log_value(state) # choose action to take
            state_, reward, done, info = env.step(action) # get info from taking that action
            score += reward # add reward to the score for the episode 
            agent.MC.store_transition(log_prob, critic_value, reward, done) # encode information about the step
            state = state_
        print(f"Episode: {episode}, Score: {score}")
        score_history.append(score)
        agent.mc_learn()
        agent.MC.clear_buffer()


    # filename = 'cartpole.png'
    # x = [i+1 for i in range(n_episodes)]
    # plot_learning_curve(x, score_history, figure_file)
    
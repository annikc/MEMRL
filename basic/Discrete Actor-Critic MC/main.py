# =====================================
#         Runs OpenAI Gyms
# =====================================

import gym
from mc_agent import MC_Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    # first we create our agent
    agent = MC_Agent(alpha=0.00001, beta=0.0005, input_dims=state_shape, gamma=0.99, n_actions=n_actions, l1_size=32, l2_size=32)

    filename = 'cartpole'
    figure_file = 'plots/' + filename

    env = gym.make('CartPole-v1')
    score_history = []
    n_episodes = 10000

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


    filename = 'cartpole.png'
    x = [i+1 for i in range(n_episodes)]
    plot_learning_curve(x, score_history, figure_file)
    
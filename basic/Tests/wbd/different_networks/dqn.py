import gym
from basic.modules.Agents.Networks.DQN import DQN, DQ_agent
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = DQ_agent(gamma=0.99, epsilon=1.0,batch_size=64, n_actions=4,
                     eps_end=0.01, input_dims=[8], lr=0.003)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done  = False
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action,reward,observation_,done)
            agent.learn()
            observation = observation_

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print(f'episode:{i}, score: {score}, avg:{ avg_score}, epsilon: {agent.epsilon}')

    # plot learning
    x = [i+1 for i in range(n_games)]
    plt.plot(x, scores, label='score')
    plt.plot(x,eps_history,label='epsilon')
    plt.legend(loc=0)
    plt.show()


main()
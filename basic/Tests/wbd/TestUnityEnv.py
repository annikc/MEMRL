# =======================================
# Runs Unity Environments w/ Gym Wrapper
# =======================================
import gym
from Agents import Agent as Agent_MC
from Utils import plot_learning_curve
import Agents.Networks as nets
from Agents.Networks import params

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

## make environment
#env = gym.make('MountainCar-v0')
#env = gym.make('CartPole-v0')
#env = gym.make('gym_grid:gridworld-v1')

## make parameters based on environment

params = params(env)
params.hidden_types = ['linear', 'linear']
params.hidden_dims = [10, 20]

## instantiate network with params
ac_test     = nets.ActorCritic(params)
print(ac_test)
#cnn_test    = nets.CNN_AC(params)
#cnn_2n_test = nets.CNN_2N(params)
#fcx2        = nets.FC(params)
#fcx2_2n     = nets.FC2N(params)



'''
if __name__ == '__main__':

    # =====================================
    #       Make Unity Environment
    # =====================================
    # for windows
    #envU = UnityEnvironment("C:/Users/KySquared/Documents/GitHub/MEMRL/basic/Envs/Unity/FirstExperiment/Windows/FirstExperiment.exe", seed=1, no_graphics=True)
    # for linux
    envU = UnityEnvironment("Envs/Unity/FirstExperiment/LinuxDev/FirstExperiment.x86_64", seed=1, no_graphics=True)

    env = UnityToGymWrapper(envU, allow_multiple_obs=False)
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape[0]

    
    # =====================================
    #              Setup Networks
    # =====================================
    params = basic_agent_params(env)
    params.input_dims = state_shape
    params.hidden_dims = [50,50]
    params.hidden_types =['linear', 'linear']
    policy_network = fcx2.FullyConnected_AC(params)
  
    # =====================================
    #              Setup Agent
    # =====================================
    agent = Agent_MC(policy_network)

    # =====================================
    #          Training Parameters
    # =====================================
    n_episodes = 100
    plot_name = "Basic_GridWorld"
    score_history = [] 
    
    for episode in range(n_episodes):
        done = False
        score = 0 
        state = env.reset()
        transition_num = 1
        while not done:
            action, log_prob, value = agent.MF_action(state)
            next_state, reward, done, _ = env.step(action) # get info from taking that action
            score += reward
            agent.log_event(episode = 0, event = transition_num, state = state, action = action, reward = reward, next_state = next_state, log_prob = log_prob, expected_value = value, target_value=0,
                                     done=done, readable_state=0)
            state = next_state
            transition_num += 1
        agent.finish_()

        print(f"Episode: {episode}, Score: {score}")
        score_history.append(score)

    # plot training data
    figure_file = 'Data/plots/' + plot_name
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
'''
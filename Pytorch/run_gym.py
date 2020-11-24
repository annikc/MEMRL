# =======================================
# Runs Unity Environments w/ Gym Wrapper
# =======================================

from Agents.agent_mc import Agent_MC
from Utils.utils import plot_learning_curve
from Agents.Networks import fcx2
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

if __name__ == '__main__':

    # =====================================
    #       Make Unity Environment
    # =====================================
    envU = UnityEnvironment("Unity_Envs/SimpleGridWorld/PyTorch_Testing.exe", seed=1)
    env = UnityToGymWrapper(envU, allow_multiple_obs=False)
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    # =====================================
    #              Setup Networks
    # =====================================
    policy_network = fcx2.Network(lr=0.006, input_dims=state_shape, fc1_dims=30, fc2_dims=30, n_actions=n_actions)
    value_network = fcx2.Network(lr=0.01, input_dims=state_shape, fc1_dims=30, fc2_dims=30, n_actions=1)
  
    # =====================================
    #              Setup Agent
    # =====================================
    agent = Agent_MC(network=policy_network, value_network=value_network)

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
            action = agent.MF_action(state)
            next_state, reward, done, _ = env.step(action) # get info from taking that action
            score += reward
            agent.store_transition(episode=episode, transition=transition_num,
                                    reward=reward, next_state=next_state, done=done)
            state = next_state
            if agent.TD:
                agent.learn()

            transition_num += 1
        
        print(f"Episode: {episode}, Score: {score}")
        score_history.append(score)

    # plot training data
    figure_file = 'Data/plots/' + plot_name
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    
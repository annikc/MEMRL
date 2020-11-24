# =======================================
# Runs Unity Environments w/o Gym Wrapper
# =======================================

from Agents.agent_mc import Agent_MC
from Utils.utils import plot_learning_curve
from Agents.Networks import fcx2
from mlagents_envs.environment import UnityEnvironment
import numpy as np
# from gym_unity.envs import UnityToGymWrapper

if __name__ == '__main__':

    # =====================================
    #       Make Unity Environment
    # =====================================
    env = UnityEnvironment("Unity_Envs/SimpleGridWorld/PyTorch_Testing.exe", seed=1)
    env.reset() # take a single step so that brain info will be sent over
    behavior_name = list(env.behavior_specs.keys())[0]# since there's only one behavior in this env
    n_actions = env.behavior_specs[behavior_name].action_spec.continuous_size if env.behavior_specs[behavior_name].action_spec.continuous_size else env.behavior_specs[behavior_name].action_spec.discrete_branches[0] # assuming not a multidiscrete action space
    state_shapes = env.behavior_specs[behavior_name].observation_shapes # a list of the shape of each observation, in the same order as in DecisionSteps/TerminalSteps
    state_shape = (np.sum(state_shapes),) # total number of values received as observations, formatted as a tuple due to architecture of Network
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
        
        env.reset()
        step, _ = env.get_steps(behavior_name)
        n_agents = len(step)
        state = np.array([])
        for agent_id in step.agent_id: # there should only be one agent here in this case, so just concatenate everything into one state
            for obs in step[agent_id].obs:
                state = np.append(state, obs)

        transition_num = 1
        while not done:
            # action = np.array(agent.MF_action(state)).reshape(1, n_actions)
            action = agent.MF_action(state)
            action = np.array(action).reshape((n_agents, 1)) # and not a multidiscrete action space (ie. only one type of action)
            # print("state", state)
            # print("action", action)
            env.set_actions(behavior_name, action)
            
            env.step()
            decision_step, terminal_step = env.get_steps(behavior_name)
            
            # since there's only one agent, the sim is going to be either done or not. If there were multiple agents, then we'd have to choose the corresponding step for each agent that required an action (either terminal_step if that agent was done, or decision_step)
            # it's possible for an agent to be in both terminal_step and decision_step, if it ended one simulation and started another one in between calls to env.step()
            done = len(terminal_step) != 0

            if done:
                step = terminal_step
            else:
                step = decision_step

            reward = 0 # since there's only one agent, there's only one reward

            next_state = np.array([])
            for agent_id in step.agent_id: # there should only be one agent here in this case, so just concatenate everything into one state
                for obs in step[agent_id].obs:
                    next_state = np.append(next_state, obs)
                reward += step[agent_id].reward
            
            agent.store_transition(episode=episode, transition=transition_num,
                                    reward=reward, next_state=next_state, done=done)
            
            if agent.TD:
                agent.learn()

            state = next_state
            score += reward
            transition_num += 1
        
        print(f"Episode: {episode}, Score: {score}")
        score_history.append(score)

    # plot training data
    figure_file = 'Data/plots/' + plot_name
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    
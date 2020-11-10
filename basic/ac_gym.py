# =====================================
# Test to get the whole mess working
# =====================================
import numpy as np
import gym
from Agents.Networks.annik_ac import ActorCritic as Network
from Agents.EpisodicMemory import EpisodicMemory as Memory
from Agents import Agent
import matplotlib.pyplot as plt

class basic_agent_params():
    def __init__(self, env):
        self.load_model = False
        self.load_dir   = ''
        self.architecture = 'A'
        self.input_dims = env.observation_space.shape # env.observation.shape # for gridworld
        self.action_dims = env.action_space.n
        self.hidden_types = ['conv', 'pool', 'conv', 'pool', 'linear', 'linear']
        self.hidden_dims = [None, None, None, None, 1000, 1000]
        self.freeze_w = False
        self.rfsize = 5
        self.gamma = 0.98
        self.eta = 5e-4



env = gym.make('gym_grid:gridworld-v1')

agent_params = basic_agent_params(env)
network=Network(agent_params.__dict__) # gridworld
#network = Network(input_dim=agent_params.input_dims,fc1_dims=40, fc2_dims=40,n_actions=agent_params.action_dims,lr=0.05) # network for cartpole
agent = Agent(network=network )#, memory=Memory(entry_size=env.action_space.n, cache_limit=env.nstates))

ntrials = 5000
nevents = 250

track_reward = []
track_p_loss = []
track_v_loss = []


for trial in range(ntrials):
    env.reset()
    score = 0
    for event in range(nevents):
        #get observation from environment
        state = env.state ## for record keeping only
        readable = 0 #env.oneD2twoD(env.state) ## for record keeping only

        # get observation from environment
        obs = env.get_observation() # gridworld
        # get action from agent
        action, log_prob, expected_value = agent.get_action(np.expand_dims(obs, axis=0))  ## expand dims to make batch size =1
        # take step in environment
        next_state, reward, done, info = env.step(action)

        # end of event
        target_value = 0
        score += reward


        agent.log_event(episode=trial,event=event,
                        state=state, action=action, reward=reward, next_state=next_state,
                        log_prob=log_prob, expected_value=expected_value, target_value=target_value,
                        done=done, readable_state=readable)

        if done:
            break

    p,v = agent.finish_()

    track_reward.append(score)
    track_p_loss.append(p)
    track_v_loss.append(v)
    if trial%10==0:
        print(f"Episode: {trial}, Score: {score}")

fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(track_reward)
ax[1].plot(track_p_loss, label='p')
ax[1].plot(track_v_loss, label='v')
plt.show()
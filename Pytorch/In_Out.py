from mlagents_envs.base_env import DecisionStep
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from random import seed
from random import randint


envU = UnityEnvironment("Unity_Envs/SimpleGridWorld/PyTorch_Testing.exe", seed=1)
env = UnityToGymWrapper(envU, allow_multiple_obs=False)
print(env.action_space.n)
print(env.observation_space.shape)

num_episodes = 10

for episode in range(num_episodes):
    done = False
    state = env.reset()
    reward = 0
    while not done:
       print(state, reward)
       next_state, reward, done, info = env.step(randint(0,3))
       state = next_state


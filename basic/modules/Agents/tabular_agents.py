import numpy as np

class Q_Agent(object):
    def __init__(self, env, learning_rate=0.1, discount=0.95, epsilon=1.0):
        self.num_actions = env.action_space.n
        self.action_space = np.arange(self.num_actions)

        # this agent selects actions from a table of state,action values which we initalize randomly
        self.q_table = np.random.uniform(low=-2, high=0, size=(env.nstates, env.action_space.n))

        # parameters for learning
        self.epsilon       = epsilon
        self.learning_rate = learning_rate
        self.discount      = discount

    def choose_action(self, state):
        # this agent uses epsilon-greedy action selection, meaning that it selects
        # the greedy (highest value) action most of the time, but with epsilon probability
        # it will select a random action -- this helps encourage the agent to explore
        # unseen trajectories
        if np.random.random()>self.epsilon:
          # take the action which corresponds to the highest value in the q table at that row (state)
          action = np.argmax(self.q_table[state])
        else:
          action=np.random.choice(self.action_space)
        return action


    def update_q_table(self, current_state, current_action, reward, new_state):
        # this function describes how the Q table gets updated so the agent can make
        # better choices based on what it has experienced from the environment
        current_q = self.q_table[ current_state, current_action]
        max_future_q = np.max(self.q_table[new_state,:])

        new_q = (1-self.learning_rate)*current_q + self.learning_rate*(reward + self.discount*max_future_q)
        self.q_table[current_state, current_action] = new_q


#SARSA agent class
class SARSA_Agent(Q_Agent):
    def __init__(self, env, learning_rate=0.1, discount=0.95, epsilon=1.0):
        super().__init__(env, learning_rate=learning_rate , discount=discount, epsilon=epsilon)

    #updates q values after each step
    # similar to Q update but with one key difference -- we use the action we *actually* took
    # rather than guessing we took the max value action
    # with epsilon probability, we will have actually taken a random action, so SARSA wants to account for that
    def update_q_table(self,current_state, current_action, reward, next_state, next_action):
        current_q = self.q_table[ current_state, current_action]
        future_q = self.q_table[ next_state, next_action] # np.max(self.q_table[new_state,:])

        new_q = (1-self.learning_rate)*current_q + self.learning_rate*(reward + self.discount*future_q)
        self.q_table[current_state, current_action] = new_q


def q_navigate(env, q_agent, num_episodes, random_start=False, start=0):
    # set how we will decay the randomness of action selection over the course of training
    start_eps_decay = 1
    end_eps_decay = num_episodes//2
    epsilon_decay_value = q_agent.epsilon/(end_eps_decay-start_eps_decay)

    # initialize empty list for keeping track of rewards achieved per episode
    reward_tracking=[]
    max_steps= 1000

    for episode in range(num_episodes):

        print(episode)
        done = False
        # initalize reward counter
        total_reward=0

        # get first state and action
        if random_start:
            state=np.random.choice(env.nstates)
        else:
            state=start

        for step in range(max_steps):
            action = q_agent.choose_action(state)

            # take a step in the environment
            next_state, reward, done, __ = env.step(action)

            print(f'{state}/{action}/{reward}/{next_state}')

            total_reward+=reward

            if not done:
                q_agent.update_q_table(state, action, reward, next_state)
            else:
                q_agent.q_table[state, action] = 0
                break
            state=next_state

        reward_tracking.append(total_reward)

        if end_eps_decay >= episode >= start_eps_decay:
            q_agent.epsilon -= epsilon_decay_value

    return reward_tracking

def sarsa_navigate(env, sarsa_agent, num_episodes, random_start=False, start=0): # takes sarsa_agent as input
#-- will not work w Q_agent bc takes additional argument of next_state in update_q_table function
  # set how we will decay the randomness of action selection over the course of training
  start_eps_decay = 1
  end_eps_decay = num_episodes//2
  epsilon_decay_value = sarsa_agent.epsilon/(end_eps_decay-start_eps_decay)

  # initialize empty list for keeping track of rewards achieved per episode
  reward_tracking=[]
  max_steps= 1000

  for episode in range(num_episodes):
    # initalize reward counter
    total_reward=0

    # get first state and action
    if random_start:
      state=np.random.choice(env.nstates)
    else:
      state=start
    action = sarsa_agent.choose_action(state)

    for step in range(max_steps):
      # take a step in the environment
      next_state, reward, done = env.move(state, action)

      total_reward+=reward

      if not done:
        next_action = sarsa_agent.choose_action(next_state)
        sarsa_agent.update_q_table(state, action, reward, next_state, next_action)
      else:
        sarsa_agent.q_table[state, action] = 0
        break
      state=next_state
      action=next_action

    reward_tracking.append(total_reward)

    if end_eps_decay >= episode >= start_eps_decay:
      sarsa_agent.epsilon -= epsilon_decay_value

  return reward_tracking
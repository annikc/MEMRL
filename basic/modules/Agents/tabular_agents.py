import numpy as np

class TabularV_SR_Agent(object):
    def __init__(self, env, gamma, learning_rate, epsilon=1):
        self.env = env
        self.state_space = env.nstates
        self.action_space = env.action_space.n
        self.M          = np.zeros((self.state_space, self.state_space))
        self.reward_est = np.zeros(self.state_space)
        self.v_table    = np.zeros((self.state_space)) ## come back to this

        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        self.transitions = []


class Tabular_SR_Agent(object):
    def __init__(self, env, gamma, learning_rate, epsilon=1):
        self.env = env
        self.state_space = env.nstates
        self.action_space = env.action_space.n
        self.M          = np.zeros((self.action_space, self.state_space, self.state_space))
        self.reward_est = np.zeros(self.state_space)
        self.q_table    = np.zeros((self.state_space, self.action_space)) ## come back to this

        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        self.transitions = []

    def onehot(self, state_index):
        vec = np.zeros(self.state_space)
        vec[state_index] = 1
        return vec

    def choose_action(self, state):
        if np.random.random()>self.epsilon:
            action = np.argmax(self.get_q_estimate(state))
        else:
            action=np.random.choice(self.action_space)
        return action

    def update_q_table(self, current_state, current_action, reward, new_state):
        # this function describes how the Q table gets updated so the agent can make
        # better choices based on what it has experienced from the environment
        current_q = self.q_table[current_state, current_action]
        max_future_q = np.max(self.q_table[new_state,:])

        new_q = (1-self.lr)*current_q + self.lr*(reward + self.gamma*max_future_q)
        self.q_table[current_state, current_action] = new_q

    def get_q_estimate(self, state):
        reward_est = self.reward_est
        # given my current state, get successor states for all actions
        # multiply with onehot reward estimate to get column with M values across actions
        q_est = np.matmul(self.M[:,state,:], reward_est)

        return q_est

    def update(self):
        s, a, s_, r, done = self.transitions[-2]
        a_                = self.transitions[-1][1]

        # update successor matrix self.M
        if done:
            sr_error = self.onehot(s) + self.gamma*self.onehot(s_) - self.M[a,s,:]
        else:
            sr_error = self.onehot(s) + self.gamma*self.M[a_,s_,:] - self.M[a,s,:]

        self.M[a,s,:] += self.lr * sr_error

        #update reward estimate self.reward_est
        reward_error = r - self.reward_est[s_]
        self.reward_est[s_] += self.lr*reward_error

        return sr_error, reward_error

    def step_in_env(self,state):
        action = self.choose_action(state)
        next_state, reward, done, _ = self.env.step(action)
        self.transitions.append([state,action,next_state,reward,done])
        return next_state

class Tabular_Q_Agent(object):
    def __init__(self, env, end_eps_decay, learning_rate=0.1, discount=0.98):
        self.num_actions = env.action_space.n
        self.action_space = np.arange(self.num_actions)

        self.q_table = np.random.uniform(low=-1, high=1, size=(env.nstates, env.action_space.n))

        self.LEARNING_RATE=learning_rate
        self.DISCOUNT=discount
        self.epsilon      = 0.4
        self.start_eps_decay = 1
        self.end_eps_decay= end_eps_decay
        self.eps_decay_val= self.epsilon/(self.end_eps_decay-self.start_eps_decay)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            action= np.argmax(self.q_table[state,:])
        else:
            action = np.random.randint(len(self.q_table[state,:]))

        return action

    def q_update(self, current_state, current_action, reward, new_state, done):
        if not done:
            current_q = self.q_table[ current_state, current_action]
            max_future_q = np.max(self.q_table[new_state,:])
            new_q = (1-self.LEARNING_RATE)*current_q + self.LEARNING_RATE*(reward + self.DISCOUNT*max_future_q)
            self.q_table[current_state, current_action] = new_q
        else:
            self.q_table[current_state, current_action] = 0



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

            #print(f'{state}/{action}/{reward}/{next_state}')

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
            print(q_agent.epsilon, "epsilon val ")

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


### JUNKYARD
'''

'''
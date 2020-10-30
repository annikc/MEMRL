# =====================================
#     Creates MC Actor-Critc Agent
# =====================================
#       Function Descriptions
# =====================================
# get_action_log_value = takes observation (state) and returns the action from actor network (i.e. policy), log_value from actor, and value from critic networ 
# mc_learn = Uses stored transition from the episode to calculate losses and updates the actor and critic networks

class Agent():
    def __init__(self, policy_network, value_network, action_selection, transition_memory, learning_algorithm): #alpha/beta = agent/critic lrs
        self.log_probs = None
        # set up AC networks and MC buffer
        self.policy_network = policy_network # actor network output is action dimensions
        self.value_network = value_network  # critic netowrk outputs is value estimate 
        self.transition_memory = transition_memory 
        self.action_selection = action_selection
        self.learning = learning_algorithm

    def get_action(self, observation):
        action, log_prob, network_value = self.action_selection.choose_action(observation, self.policy_network, self.value_network)
        return action, log_prob, network_value

    def store_transition(self, state, action, reward, new_state, log_prob, model_value, done):
        self.transition_memory.store_transition(state, action, reward, new_state, log_prob, model_value, done)

    def clear_transition_memory(self):
        self.transition_memory.clear_memory()

    # learn
    def learn(self, MC=True):
        self.policy_network, self.value_network = self.learning.learn(self.policy_network, self.value_network, self.transition_memory)
        
        


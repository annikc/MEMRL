import torch as T
import torch.nn.functional as F

def action_softmax(self, state):
    probabilities = F.softmax(state)
    action_probs = T.distributions.Categorical(probabilities)
    action = action_probs.sample()
    log_prob = action_probs.log_prob(action)
    # get the value produced by the critic (value) network
    # then we want to return our action as a integer using .item() for OpenAI 
    # return the log_probs and critic value as well so we can save into episode buffer
    return action.item(), log_prob

def action_egreedy(self):
    return

def discount_rwds(self, transitions, gamma = 0.99):
    running_add = 0
    for transition in reversed(range(transitions)):
        running_add = running_add*gamma + transition.reward
        transition.target_value = running_add
    
    return transitions

def episode_losses(self, transitions):
    policy_losses = 0
    value_losses = 0
    for transition in range(len(transitions)): 
    # delta = reward + gamma*critic_value*(1-done)
        delta = transition.target_value - transition.expected_value
        log_prob = transition.log_prob  # This is the delta variable (i.e target_value*self.gamma + observed_value)
        policy_loss = (-log_prob * delta)
        value_loss = (delta**2)
        policy_losses += policy_loss
        value_losses += value_loss

    return policy_losses, value_losses
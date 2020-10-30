import torch as T
import torch.nn.functional as F


class SoftMax(): 
    def choose_action(self, observation, policy_network, value_network):
        probabilities = F.softmax(policy_network.forward(observation))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        # get the value produced by the critic (value) network
        if value_network != None:
            value = value_network.forward(observation)
            network_value = value.view(-1)
        else:
            network_value = None
        # then we want to return our action as a integer using .item() for OpenAI 
        # return the log_probs and critic value as well so we can save into episode buffer
        return action.item(), log_prob, network_value.item()
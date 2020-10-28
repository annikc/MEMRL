# =====================================
#     Creates MC Actor-Critc Agent
# =====================================
#       Function Descriptions
# =====================================
# get_action_log_value = takes observation (state) and returns the action from actor network (i.e. policy), log_value from actor, and value from critic networ 
# mc_learn = Uses stored transition from the episode to calculate losses and updates the actor and critic networks

from network import Network
import torch as T
import torch.nn.functional as F
from mc_buffer import MCBuffer
from torch.autograd import Variable

class MC_Agent():
    def __init__(self, alpha, beta, input_dims, gamma=0.99, l1_size = 256, l2_size = 256, n_actions = []): #alpha/beta = agent/critic lrs
        self.log_probs = None
        self.n_actions = n_actions
        # set up AC networks and MC buffer
        self.actor = Network(alpha, input_dims, l1_size, l2_size, n_actions)  # actor network output is action dimensions
        self.critic = Network(beta, input_dims, l1_size, l2_size, n_actions=1)  # critic netowrk outputs is value estimate 
        self.MC = MCBuffer(gamma) 
    
    def get_action_log_value(self, observation):
        # get the action produced by the actor (policy) network
        probabilities = F.softmax(self.actor.forward(observation))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        # get the value produced by the critic (value) network
        critic = self.critic.forward(observation)
        critic_value = critic.view(-1)
        # then we want to return our action as a integer using .item() for OpenAI 
        # return the log_probs and critic value as well so we can save into episode buffer
        return action.item(), log_prob, critic_value

    # Monte Carlo Method - Update at the end of the episode
    def mc_learn(self):
        
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        
        actor_losses = []
        critic_losses = []
        # Gets information stored in buffer
        logs_values_rewards = self.MC.get_buffer()
        # target_values are calucated using rewards-to-go backward iteration method (discounted_rwds function in mc_buffer)
        for (log_prob, observed_value), target_value in logs_values_rewards:  
            # delta = reward + gamma*critic_value*(1-done)
            delta = target_value - observed_value.item()  # This is the delta variable (i.e target_value*self.gamma + observed_value)
            actor_losses.append(-log_prob * delta)
            #print(actor_losses)
            return_bs = Variable(T.Tensor([[target_value]])).unsqueeze(-1) # used to make the shape work out
            print(observed_value.shape, return_bs.shape) # shape error - you can pass batch 
            critic_losses.append(F.smooth_l1_loss(observed_value, target_value))
            #print(critic_losses) # critic losses is updated differently than Anniks model not sure if this is correct (?)

        # back props losses through the network
        (T.cat(actor_losses) + t.cat(critic_losses)).backward()

        # then we optimize the networks
        self.actor.optimizer.step()
        self.critic.optimizer.step()

        # clear the buffer - could add save-to-memory functionality for EC 
        self.MC.clear_buffer()

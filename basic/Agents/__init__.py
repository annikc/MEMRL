# Write Agent Class Here
# Should take arg for network and for memory
# Annik Carson Oct 28,2020
from torch.distributions import Categorical
from Utils import discount_rwds #TODO

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Agent(object):
    def __init__(self, network, memory=None)
        self.MFC = network
        self.EC = memory

        self.saved_actions = []
        self.saved_rewards = []
        self.returns       = []
        self.saved_state_reps = [] ## unsure if this is necessary

        self.gamma = 0.98 # discount factor for return computation

        self.eta = 5e-4 # learning rate -- set in agent or in network?
        self.optimizer = torch.optim.Adam(self.MFC.parameters(), lr=self.eta)
        #TODO: update functions
        self.action_selection = self.MF_action
        self.network_update = self.update_MC

    def MF_action(self, state_observation):
        policy, value = self.MFC(state_observation)

        a = Categorical(policy)

        action = a.sample()

        self.saved_actions.append(SavedAction(a.log_prob(action), value))

        return action.item()

    def EC_action(self, state_observation):
        MF_policy, value = self.MFC(state_observation)
        ## here state observation should be state representation??
        EC_policy = self.EC.recall_mem(state_observation)

        a = Categorical(EC_policy)
        b = Categorical(MF_policy)

        action = a.sample() # select action using episodic

        self.saved_actions.append(SavedAction(b.log_prob(action), value)) # for model free weight updates, save log of model free prob for selected action

        return action.item()

    def update_MC(self):
        #compute monte carlo return
        self.returns = torch.Tensor(discount_rwds(np.asarray(self.saved_rewards), gamma=self.gamma)) # compute returns
        for (log_prob, value), r in zip(self.saved_actions, self.returns):
            rpe = r - value.item()
            policy_losses.append(-log_prob * rpe)
            return_bs = Variable(torch.Tensor([[r]])).unsqueeze(-1) # used to make the shape work out
            value_losses.append(F.smooth_l1_loss(value, return_bs))
        self.optimizer.zero_grad()
        p_loss, v_loss = torch.cat(policy_losses).sum(), torch.cat(value_losses).sum()
        total_loss = p_loss + v_loss
        total_loss.backward(retain_graph=False)
        self.optimizer.step()

        del self.saved_rewards[:]
        del self.saved_actions[:]
        return p_loss, v_loss

    def update_TD(self, state, reward, new_state, done):
        # for all pytorch programs you want to zero out the gradients for the optimizer
        # at the beginning of your learning function. we do this for both actor and critic
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        # next we get the value of the state and the value of the next state
        # from our critic network
        critic_value = self.critic.forward(state)
        critic_value_ = self.critic.forward(new_state)

        # next we calculate the TD error (i.e. delta)
        # we augment calculation with 1-int(done) so that we don't update
        # when episode is done
        delta = ((reward + self.gamma*critic_value*(1-int(done))) - critic_value_)
        # we use delta to calculate both the actor and critic losses
        # the modifies the action probabilities in the direction that maximizes
        # future reward
        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        # we back propogate the sum of the losses through the network
        (actor_loss + critic_loss).backward()

        # then we optimize
        self.actor.optimizer.step()
        self.critic.optimizer.step()

    def EC_storage(self, buffer):
        mem_dict = {}
        ## what info should be contained in buffer?
        # state / action / {reward/return/rpe} / timestamp / trial info
        returns = self.returns

        timesteps, states, actions, readable, trial = buffer
        for s, a, r, event, rdbl in zip(states,actions,returns,timesteps,readable):
            mem_dict['activity']  = s
            mem_dict['action']    = a
            mem_dict['delta']     = r
            mem_dict['timestamp'] = event
            mem_dict['readable']  = rdbl
            mem_dict['trial']     = trial

            self.EC.add_mem(mem_dict)

        del self.returns[:]
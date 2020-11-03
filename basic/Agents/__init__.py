# Write Agent Class Here
# Should take arg for network and for memory
# Annik Carson Oct 28,2020
from torch.distributions import Categorical
from basic.Utils import discount_rwds
from Agents.Transition_Cache.transition_cache import Transition_Cache

class Agent(object):
    def __init__(self, network, memory=None, **kwargs):
        self.MFC = network
        self.EC = memory
        self.optimizer = network.optimizer

        self.episode_record = self.reset_buffer()

        self.gamma = kwargs.get('discount',0.98) # discount factor for return computation
        self.transition_cache = Transition_Cache(cache_size=10000)
        self.TD = kwargs.get('td_learn', False)
        self.get_action = self.MF_action
        if self.TD:
            self.update = self.update_TD
        else:
            self.update = self.update_MC

    def reset_buffer(self):
        episode_record = {'states': [],
                          'next_state': None,
                          'actions': [],
                          'log_probs': [],
                          'values': [],
                          'rewards': [],
                          'returns': [],
                          'event_ts': [],
                          'readable_states': [],
                          'trial': None, ## must set trial in the beginning of each trial in experiment
                          'done': None}
        return episode_record

    def MF_action(self, state_observation):
        policy, value = self.MFC(state_observation)

        a = Categorical(policy)

        action = a.sample()

        self.episode_record['states'].append(state_observation)
        self.episode_record['actions'].append(action)
        self.episode_record['log_probs'].append(a.log_prob(action))
        self.episode_record['values'].append(value)

        #self.saved_actions.append(SavedAction(a.log_prob(action), value))

        return action.item(), a.log_prob(action), value.view(-1) ##TODO: why view instead of item

    def EC_action(self, state_observation):
        MF_policy, value = self.MFC(state_observation)
        ## here state observation should be state representation??
        EC_policy = self.EC.recall_mem(state_observation)

        a = Categorical(EC_policy)
        b = Categorical(MF_policy)

        action = a.sample() # select action using episodic

        self.episode_record['states'].append(state_observation)
        self.episode_record['actions'].append(action)
        self.episode_record['log_probs'].append(b.log_prob(action))
        self.episode_record['values'].append(value)

        #self.saved_actions.append(SavedAction(b.log_prob(action), value)) # for model free weight updates, save log of model free prob for selected action

        return action.item()

    def log_event(self, reward, next_state, done, event, readable_state):
        # records feedback from environment after action is taken
        self.episode_record['next_state'] = next_state
        self.episode_record['rewards'].append(reward)
        self.episode_record['event_ts'].append(event)
        self.episode_record['readable_states'].append(readable_state)
        self.episode_record['done'] = done

    def update_MC(self):
        #compute monte carlo return
        rewards = np.asarray(self.episode_record['rewards'])
        self.episode_record['returns'] = torch.Tensor(discount_rwds(rewards, gamma=self.gamma)) # compute returns

        for log_prob, value, r in zip(self.episode_record['log_probs'], self.episode_record['values'], self.episode_record['returns']):
            rpe = r - value.item()
            policy_losses.append(-log_prob * rpe)
            return_bs = Variable(torch.Tensor([[r]])).unsqueeze(-1) # used to make the shape work out
            value_losses.append(F.smooth_l1_loss(value, return_bs))

        self.optimizer.zero_grad()
        p_loss, v_loss = torch.cat(policy_losses).sum(), torch.cat(value_losses).sum()
        total_loss = p_loss + v_loss
        total_loss.backward(retain_graph=False)
        self.optimizer.step()

        return p_loss, v_loss

    def update_TD(self):
        # for all pytorch programs you want to zero out the gradients for the optimizer
        # at the beginning of your learning function. we do this for both actor and critic
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        # next we get the value of the state and the value of the next state
        # from our critic network
        v_current = self.episode_record['values'][-1]
        v_next = self.critic(self.episode_record['next_state']) # compute value for next state
        reward = self.rewards[-1]

        done = self.episode_record['done']

        # next we calculate the TD error (i.e. delta)
        # we augment calculation with 1-int(done) so that we don't update
        # when episode is done
        delta = ((reward + self.gamma*v_current*(1-int(done))) - v_next)
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

    def EC_storage(self):
        mem_dict = {}
        states = self.episode_record['states']
        actions = self.episode_record['actions']
        returns = self.episode_record['returns']
        timesteps = self.episode_record['event_ts']
        readable = self.episode_record['readable_states']
        trial = self.episode_record['trial']

        for s, a, r, event, rdbl in zip(states,actions,returns,timesteps,readable):
            mem_dict['activity']  = s
            mem_dict['action']    = a
            mem_dict['delta']     = r
            mem_dict['timestamp'] = event
            mem_dict['readable']  = rdbl
            mem_dict['trial']     = trial

            self.EC.add_mem(mem_dict)

    def finish_(self):
        ## if monte carlo, call at end of trial
        ## if TD, call at end of event
        self.update()
        if self.EC != None:
            self.EC_storage()

        self.episode_record = self.reset_buffer()

    ## from Kyle's agent_MC
    def learn(self):
        policy_losses = 0
        value_losses = 0

        # Calculates the discounted rewards (target_values) and updates transition with them
        # Note: passing transitions here isn't necessary but allows discount_rwds method to be in
        # seperate module if that makes more sense
        self.transition_cache.transition_cache = self.discount_rwds(self.transition_cache.transition_cache)

        # Gets policy and value losses
        # Note: passing transitions here isn't necessray but allows episode_losses method to be in
        # seperate module if that makes more sense
        policy_losses, value_losses = self.episode_losses(self.transition_cache.transition_cache)

        self.policy_value_network.optimizer.zero_grad()

        (policy_losses + value_losses).backward()
        self.policy_value_network.optimizer.step()

    def discount_rwds(self, transitions):
        running_add = 0
        for t in reversed(range(len(transitions))):
            running_add = running_add*self.gamma + transitions[t].reward
            transitions[t] = transitions[t]._replace(target_value = running_add)
            #print (transitions[t].target_value)

        return transitions

    # agents transition Cache
    def store_transition(self, transition):
        self.transition_cache.store_transition(transition)

    def clear_transition_cache(self):
        self.transition_cache.clear_cache()

    # Calculates the policy and value losses and returns them seperately
    def episode_losses(self, transitions):
        policy_losses = 0
        value_losses = 0

        for transition in transitions:

            target_value = transition.target_value
            expected_value = transition.expected_value
            delta = target_value - expected_value.item()

            log_prob = transition.log_prob  # This is the delta variable (i.e target_value*self.gamma + observed_value)
            policy_loss = (-log_prob * delta)
            return_bs = Variable(T.Tensor([[target_value]])).unsqueeze(-1) # used to make the shape work out

            value_loss = (F.smooth_l1_loss(expected_value, return_bs))
            policy_losses += policy_loss
            value_losses += value_loss

        return policy_losses, value_losses

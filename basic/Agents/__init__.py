## Agent Class Definitions
## Basic Agent Super Class -- expects a single (actor-critic) network
# DualNetwork inherits all functions from Agent base class and provides
# new update function which does a backward pass on both networks

import numpy as np
from collections import namedtuple
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from basic.Agents.Transition_Cache import Transition_Cache


Transition = namedtuple('Transition', 'episode, transition, state, action, reward, \
                                next_state, log_prob, expected_value, target_value, done, readable_state')

class Agent(object):
    def __init__(self, network, memory=None, **kwargs):
        self.MFC = network
        self.EC = memory
        self.transition_cache = Transition_Cache(cache_size=10000)

        self.optimizer = network.optimizer

        self.gamma = kwargs.get('discount',0.98) # discount factor for return computation

        self.get_action = self.MF_action

        self.TD = kwargs.get('td_learn', False)
        
        self.calc_loss = self.TD_loss if self.TD else self.MC_loss


    def MF_action(self, state_observation):
        policy, value = self.MFC(state_observation)

        a = Categorical(policy)

        action = a.sample()

        return action.item(), a.log_prob(action), value.view(-1) ##TODO: why view instead of item

    def EC_action(self, state_observation):
        MF_policy, value = self.MFC(state_observation)

        mem_state = tuple(self.MFC.h_act.detach().numpy()[0])

        EC_policy = torch.Tensor(self.EC.recall_mem(mem_state))

        a = Categorical(EC_policy)
        b = Categorical(MF_policy)

        action = a.sample() # select action using episodic

        return action.item(), b.log_prob(action), value.view(-1)

    def log_event(self, episode, event, state, action, reward, next_state, log_prob, expected_value, target_value, done, readable_state):
        # episode = trial
        # event = one step in the environment
        transition = Transition(episode=episode,
                                transition=event,
                                state=state,
                                action=action,
                                reward=reward,
                                next_state=next_state,
                                log_prob=log_prob,
                                expected_value=expected_value,
                                target_value=target_value,
                                done=done,
                                readable_state=readable_state
                                )
        self.transition_cache.store_transition(transition)

    def EC_storage(self):
        mem_dict = {}
        buffer = np.vstack(self.transition_cache.transition_cache)

        states    = buffer[:,2]
        actions   = buffer[:,3]
        returns   = buffer[:,8]
        timesteps = buffer[:,1]
        readable  = buffer[:,10]
        trial     = buffer[-1,0]

        for s, a, r, event, rdbl in zip(states,actions,returns,timesteps,readable):
            mem_dict['activity']  = s
            mem_dict['action']    = a
            mem_dict['delta']     = r
            mem_dict['timestamp'] = event
            mem_dict['readable']  = rdbl
            mem_dict['trial']     = trial

            self.EC.add_mem(mem_dict)

    def discount_rwds(self):
        transitions = self.transition_cache.transition_cache

        running_add = 0
        for t in reversed(range(len(transitions))):
            running_add = running_add*self.gamma + transitions[t].reward
            transitions[t] = transitions[t]._replace(target_value = running_add)
        # update transition cache with computed return values
        self.transition_cache.transition_cache = transitions

    def MC_loss(self):
        #compute monte carlo return
        self.discount_rwds()

        pol_loss = 0
        val_loss = 0
        for transition in self.transition_cache.transition_cache:
            G_t = transition.target_value
            V_t = transition.expected_value
            delta = G_t - V_t.item()

            log_prob = transition.log_prob

            pol_loss += -log_prob*delta
            G_t = torch.Tensor([G_t])
            v_loss = torch.nn.L1Loss()(V_t, G_t)
            val_loss += v_loss
        return pol_loss, val_loss

    def TD_loss(self):
        event = self.transition_cache.transition_cache[-1]
        V_current = event.expected_value
        next_state = event.next_state
        _, V_next = self.MFC(next_state)
        reward  = event.reward
        done = event.done

        # calculate the TD error (i.e. delta)
        # we augment calculation with 1-int(done) so that we don't update when episode is done
        delta = ((reward + self.gamma*V_current*(1-int(done))) - V_next)
        # we use delta to calculate both the actor and critic losses
        # the modifies the action probabilities in the direction that maximizes
        # future reward
        pol_loss = -self.log_probs * delta
        val_loss = delta**2

        return pol_loss, val_loss

    def update(self):
        pol_loss, val_loss = self.calc_loss()

        self.optimizer.zero_grad()
        total_loss = pol_loss + val_loss
        total_loss.backward()
        self.optimizer.step()

        return pol_loss, val_loss

    def finish_(self):
        ## if monte carlo, call at end of trial
        ## if TD, call at end of event
        p, v = self.update()
        if self.EC != None:
            self.EC_storage()

        self.transition_cache.clear_cache()
        return p,v

class DualNetwork(Agent):
    def __init__(self, policy_network, value_network, memory=None, **kwargs):

        self.policy_net = policy_network ## what happens if MFC is two separate networks ?
        self.value_net = value_network
        self.EC = memory
        self.transition_cache = Transition_Cache(cache_size=10000)

        self.policy_optimizer = self.policy_net.optimizer
        self.value_optimizer = self.value_net.optimizer

        self.gamma = kwargs.get('discount',0.98) # discount factor for return computation

        self.get_action = self.MF_action

        self.TD = kwargs.get('td_learn', False)
        if self.TD:
            self.calc_loss = self.TD_loss
        else:
            self.calc_loss = self.MC_loss


    def MF_action(self, state_observation):
        policy = F.softmax(self.policy_net(state_observation))
        value = self.value_net(state_observation)

        a = Categorical(policy)

        action = a.sample()

        return action.item(), a.log_prob(action), value.view(-1)

    def update(self):
        pol_loss, val_loss = self.calc_loss()

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss = pol_loss + val_loss
        total_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()

        return pol_loss, val_loss
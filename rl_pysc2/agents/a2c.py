""" AC2
"""
import numpy as np
import torch
from collections import namedtuple
from torch import nn
from torch.distributions.categorical import Categorical

from rl_pysc2.networks.deepmind_model import Encode, Output


class A2C(nn.Module):
    """
    """
    TransitionInfo = namedtuple("TransitionInfo", "log_prob entropy value")

    def __init__(self, network):
        super(A2C, self).__init__()
        self.network = network
        # Only for the discrete implementation
        self.dist = Categorical()
        # We decided to gather transitions inside A2C
        self.buffer = []

    def forward(self, state):
        """ Generate distribution of the policy. Log probability, entropy and
        action is taken from the distribution. In the training mode agent
        stores trainstions to be used in the gradient calculations.
            Arguments:
                - state: Torch tensor of observation
            Return:
                - action: Discrete action(int)
                - log_prob: Log probability of the action(torch tensor)
                - entropy: Entropy of the distribution(torch tensor)
                - value: Value of the state(torch tensor)
        """
        logit_act, value = self.network(state)
        dist = Categorical(logits=logit_act)
        action = dist.sample().item()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        if self.train:
            self.buffer.append(self.TransitionInfo(log_prob, entropy, value))

        return action

    def calculate_loss(self, next_state, is_terminal, gamma, tau, beta):
        """
        This function is inspired from ikostrikov's implementation.
        https://github.com/ikostrikov/pytorch-a3c
        Calculate actor and critic losses. Generalized advantage estimation is
        used in the actor loss.
        Arguments:
            - next_state: Next state of the environment
            - is_terminal: True if the next state is a terminal state
            - gamma: Discount rate
            - tau: Generalized advantage estimation coefficient
            - beta: Entropy regularization coefficient
        Return:
            - loss
        Raise:
            - ValueError: If the argument <next_state> is not a torch tensor
        """
        if not isinstance(next_state, torch.Tensor):
            raise ValueError("next_state argument must be torch tensor")
        value_loss = 0
        actor_loss = 0

        # N step return
        n_return = 0
        value_next = 0
        gae = 0
        if is_terminal is False:
            with torch.no_grad():
                _, value_next = self.network(next_state)
            n_return = value_next

        for log_prob, entropy, value in reversed(self.buffer):
            # Value loss
            n_return = n_return * gamma + reward
            value_loss += n_return - value

            # Actor loss
            delta = (value_next * gamma + reward - value).detach() 
            gae = gae * gamma * tau + delta

            policy_loss += log_prob * gae + entropy * beta
        
        self._clean_buffer()
        return value_loss - policy_loss

    def update_params(self, loss, optimizer):
        pass

    def _clean_buffer(self):
        self.buffer = []


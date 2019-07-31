import torch
import numpy as np
from collections import namedtuple


class VGP(torch.nn.Module):
    """
        - no nstep
        - no multi-env
    """

    Transition = namedtuple("Transition",
                            "reward done log_prob value next_value")

    def __init__(self, network, optimizer):
        super().__init__()
        self.network = network
        self.optimizer = optimizer

    def forward(self, state):
        logits, value = self.network(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # print(logits)
        return action, log_prob, value

    def update(self, trans, gamma):
        target = (trans.next_value.detach()*gamma*(1-trans.done) +
                  trans.reward)
        td = target - trans.value
        # print(trans.log_prob)
        policy_gain = -trans.log_prob * td.detach()
        value_loss = torch.nn.functional.mse_loss(trans.value, target)/2
        self.optimizer.zero_grad()
        loss = policy_gain + value_loss
        loss.mean().backward()
        # print(next(self.network.parameters()).grad)
        # print(self.optimizer.param_groups)
        self.optimizer.step()
        return value_loss.mean().item()

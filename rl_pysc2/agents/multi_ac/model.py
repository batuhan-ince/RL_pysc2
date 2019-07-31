import torch
import numpy as np
from collections import namedtuple


class MultiAC(torch.nn.Module):

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
        return action, log_prob, value

    def update(self, trans, gamma):
        target = (trans.next_value.detach()*gamma*(1-trans.done) +
                  trans.reward).detach()
        td = target - trans.value

        policy_gain = -trans.log_prob * td.detach()
        value_loss = torch.nn.functional.mse_loss(trans.value, target)/2
        self.optimizer.zero_grad()
        loss = policy_gain + value_loss
        loss.mean().backward()

        self.optimizer.step()
        return value_loss.mean().item()

    def update_per_env(self, trans, gamma):
        # target = (trans.next_value.detach()*gamma*(1-trans.done) +
        #           trans.reward).detach()
        # td = target - trans.value

        # policy_gain = -trans.log_prob * td.detach()
        # value_loss = torch.pow((trans.value - target), 2)/2

        # loss = []
        # for p_g, v_l in zip(torch.split(policy_gain, 1),
        #                     torch.split(value_loss, 1)):
        #     self.optimizer.zero_grad()
        #     (p_g + v_l).backward()
        #     self.optimizer.step()
        #     loss.append(v_l.mean().item())
        # return np.mean(loss)
        raise NotImplementedError

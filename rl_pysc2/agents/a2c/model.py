import torch
import numpy as np
from collections import namedtuple, deque


class A2C(torch.nn.Module):

    Transition = namedtuple("Transition",
                            "reward done log_prob value next_value entropy")

    def __init__(self, network, optimizer):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.queue = deque()

    def forward(self, state):
        logits, value = self.network(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy

    def update(self, gamma, beta=0.0):
        R = self.queue[-1].next_value.detach()
        gae = 0
        loss = 0
        while len(self.queue) > 0:
            reward, done, log_prob, value, next_value, entropy = self.queue.pop()
            R = R*(1 - done)*gamma + reward
            adv = R - value

            value_loss = adv.pow(2) / 2
            policy_gain = -log_prob * adv.detach()

            loss += (value_loss).mean().item()
            self.optimizer.zero_grad()
            (value_loss + policy_gain - entropy*beta).mean().backward()
            self.optimizer.step()
        return loss

    def add_trans(self, reward, done, log_prob, value, next_value, entropy):
        self.queue.append(self.Transition(reward, done,
                                          log_prob, value,
                                          next_value, entropy))

    def save_model(self, filename):
        states = {
            "network": self.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(states, open(filename, "wb"))

    def load_model(self, filename):
        states = torch.load(open(filename, "rb"))
        self.load_state_dict(states["network"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(states["optimizer"])

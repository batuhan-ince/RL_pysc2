import torch
import numpy as np
from collections import deque, namedtuple


class Reinforce(torch.nn.Module):

    Transition = namedtuple("Transition", "reward log_prob")

    def __init__(self, network, optimizer):
        super().__init__()
        self.network = network
        self.queue = deque()
        self.optimizer = optimizer

    def forward(self, state):
        if self.training:
            return self.step(state)
        else:
            with torch.no_grad():
                action, _ = self.step(state)
                return action

    def step(self, state):
        policy_logit = self.network(state)
        policy_dist = torch.distributions.Categorical(logits=policy_logit)
        action = policy_dist.sample()
        log_prob = policy_dist.log_prob(action)
        return action, log_prob

    def add_transitions(self, reward, log_prob):
        self.queue.append(self.Transition(reward, log_prob))

    def update(self, gamma):
        if len(self.queue) == 0:
            raise RuntimeError("No transition to update.")
        R = 0
        loss = 0
        for i in range(len(self.queue)):
            reward, log_prob = self.queue.pop()
            R = R*gamma + reward
            loss -= log_prob*R
        self.optimizer.zero_grad()
        loss.sum().backward()
        self.optimizer.step()
    
        return loss.sum().item()
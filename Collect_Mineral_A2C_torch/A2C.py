import math
import random

import gym
import numpy as np
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

#defining the constants necessary
N_STATES = 4
N_ACTIONS = 16
#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LRA = 0.001
LRC = 0.01


class Actor_Net(nn.Module):
    def __init__(self, ):
        super(Actor_Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 10)
        nn.init.xavier_uniform(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        self.fc2 = nn.Linear(10, 5)
        nn.init.xavier_uniform(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

        self.fc3 = nn.Linear(5, 5)
        nn.init.xavier_uniform(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)

        self.fc4 = nn.Linear(5, 5)
        nn.init.xavier_uniform(self.fc4.weight)
        self.fc4.bias.data.fill_(0.01)
        
        self.out = nn.Linear(5, N_ACTIONS)
        nn.init.xavier_uniform(self.out.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.out(x)
        actions_prob = F.softmax(x)
        return actions_prob


class Critic_Net(nn.Module):
    def __init__(self, ):
        super(Critic_Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 10)
        nn.init.xavier_uniform(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        self.fc2 = nn.Linear(10, 10)
        nn.init.xavier_uniform(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

        self.fc3 = nn.Linear(10, 10)
        nn.init.xavier_uniform(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)

        self.fc4 = nn.Linear(10, 10)
        nn.init.xavier_uniform(self.fc4.weight)
        self.fc4.bias.data.fill_(0.01)

        self.out = nn.Linear(10, N_ACTIONS)
        nn.init.xavier_uniform(self.out.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

#Combining Actor-Critic structure together
class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()
        self.critic, self.actor = Critic_Net(), Actor_Net()
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=LRA)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=LRC)

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        probs = self.actor.forward(x)
        dist = Categorical(probs)
        # return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel()), probs
        return dist, dist.sample(), probs

    def learn(self, R, X, action, GAMMA=0.99):
        X = Variable(torch.unsqueeze(torch.FloatTensor(X), 0))

        self.TD_ERROR = R + GAMMA * self.critic.forward(X)
        critic_loss = self.TD_ERROR ** 2
        dist, _, probs = self.choose_action(X)
        log_prob = dist.log_prob(action)

        # Critic Learning Part
        self.optimizer_critic.zero_grad()
        critic_loss.mean().backward(retain_graph=True)
        self.optimizer_critic.step()

        # Actor Learning Part
        actor_loss = -torch.mean(log_prob * self.TD_ERROR)
        self.optimizer_actor.zero_grad()
        actor_loss.mean().backward(retain_graph=True)
        self.optimizer_actor.step()

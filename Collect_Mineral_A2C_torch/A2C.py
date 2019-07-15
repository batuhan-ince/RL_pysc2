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

# defining the constants necessary
N_STATES = 16*16*16*2
N_ACTIONS = 64
# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LRA = 0.00001
LRC = 0.0001


class Actor_Net(nn.Module):
    def __init__(self, ):
        super(Actor_Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 100)
        nn.init.xavier_uniform(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        self.fc2 = nn.Linear(100, 50)
        nn.init.xavier_uniform(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

        self.fc3 = nn.Linear(50, 40)
        nn.init.xavier_uniform(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)

        self.fc4 = nn.Linear(40, 25)
        nn.init.xavier_uniform(self.fc4.weight)
        self.fc4.bias.data.fill_(0.01)

        self.out = nn.Linear(25, N_ACTIONS)
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

        self.fc1 = nn.Linear(N_STATES, 100)
        nn.init.xavier_uniform(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        self.fc2 = nn.Linear(100, 50)
        nn.init.xavier_uniform(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

        self.fc3 = nn.Linear(50, 40)
        nn.init.xavier_uniform(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)

        self.fc4 = nn.Linear(40, 25)
        nn.init.xavier_uniform(self.fc4.weight)
        self.fc4.bias.data.fill_(0.01)

        self.out = nn.Linear(25, N_ACTIONS)
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


# Combining Actor-Critic structure together
class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()
        self.critic_x, self.actor_x,self.critic_y, self.actor_y = Critic_Net(), Actor_Net(),Critic_Net(), Actor_Net()
        self.optimizer_actor_x = torch.optim.Adam(self.actor_x.parameters(), lr=LRA)
        self.optimizer_actor_y = torch.optim.Adam(self.actor_y.parameters(), lr=LRA)
        self.optimizer_critic_x = torch.optim.Adam(self.critic_x.parameters(), lr=LRC)
        self.optimizer_critic_y = torch.optim.Adam(self.critic_y.parameters(), lr=LRC)

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        x=torch.squeeze(x,0)
        prob_x,prob_y = self.actor_x.forward(x),self.actor_y.forward(x)
        dist_x,dist_y = Categorical(prob_x), Categorical(prob_y)
        return dist_x, dist_x.sample(), prob_x,dist_y,dist_y.sample(),prob_y

    def learn(self, R, X, z, action_x,action_y, GAMMA=0.99):
        X = Variable(torch.unsqueeze(torch.FloatTensor(X), 0))
        z = Variable(torch.unsqueeze(torch.FloatTensor(z), 0))
        X=torch.squeeze(X,0)
        z = torch.squeeze(z, 0)
        x_val2,y_val2=self.critic_x.forward(z),self.critic_y.forward(z)
        x_val1, y_val1 = self.critic_x.forward(X), self.critic_y.forward(X)

        self.TD_ERROR_X = R + GAMMA * x_val2-x_val1
        self.TD_ERROR_Y = R + GAMMA * y_val2-y_val1
        critic_loss_x = (self.TD_ERROR_X) ** 2
        critic_loss_y = (self.TD_ERROR_Y) ** 2

        # X = (torch.unsqueeze(X, 0))
        dist_x, _, prob_x,dist_y,___,prob_y = self.choose_action(X)
        log_prob_x = dist_x.log_prob(action_x)
        log_prob_y = dist_y.log_prob(action_y)

        # Critic Learning Part x
        self.optimizer_critic_x.zero_grad()
        critic_loss_x.mean().backward(retain_graph=True)
        self.optimizer_critic_x.step()

        # Critic Learning Part y
        self.optimizer_critic_y.zero_grad()
        critic_loss_y.mean().backward(retain_graph=True)
        self.optimizer_critic_y.step()

        # Actor Learning Part
        actor_loss_x = -torch.mean(log_prob_x * self.TD_ERROR_X)
        self.optimizer_actor_x.zero_grad()
        actor_loss_x.mean().backward(retain_graph=True)
        self.optimizer_actor_x.step()

        # Actor Learning Part
        actor_loss_y = -torch.mean(log_prob_y * self.TD_ERROR_Y)
        self.optimizer_actor_y.zero_grad()
        actor_loss_y.mean().backward(retain_graph=True)
        self.optimizer_actor_y.step()


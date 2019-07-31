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
# define constants
class Actor_Net(nn.Module):
   def __init__(self):
       super(Actor_Net, self).__init__()
       self.fc1 = nn.Linear(in_size, 300)
       nn.init.xavier_uniform(self.fc1.weight)
       self.fc1.bias.data.fill_(0.01)
       self.fc2 = nn.Linear(300, 150)
       nn.init.xavier_uniform(self.fc2.weight)
       self.fc2.bias.data.fill_(0.01)
       self.fc3 = nn.Linear(150, 75)
       nn.init.xavier_uniform(self.fc3.weight)
       self.fc3.bias.data.fill_(0.01)
       self.out = nn.Linear(75, out_size)
       nn.init.xavier_uniform(self.out.weight)
   def forward(self, x):
       x = self.fc1(x)
       x = F.relu(x)
       x = self.fc2(x)
       x = F.relu(x)
       x = self.fc3(x)
       x = F.relu(x)
       x = self.out(x)
       actions_prob = F.softmax(x)
       return actions_prob
class Critic_Net(nn.Module):
   def __init__(self):
       super(Critic_Net, self).__init__()
       self.fc1 = nn.Linear(in_size, 300)
       nn.init.xavier_uniform(self.fc1.weight)
       self.fc1.bias.data.fill_(0.01)
       self.fc2 = nn.Linear(300, 150)
       nn.init.xavier_uniform(self.fc2.weight)
       self.fc2.bias.data.fill_(0.01)
       self.fc3 = nn.Linear(150, 75)
       nn.init.xavier_uniform(self.fc3.weight)
       self.fc3.bias.data.fill_(0.01)
       self.out = nn.Linear(75, out_size)
       nn.init.xavier_uniform(self.out.weight)
   def forward(self, x):
       x = self.fc1(x)
       x = F.relu(x)
       x = self.fc2(x)
       x = F.relu(x)
       x = self.fc3(x)
       x = F.relu(x)
       actions_value = self.out(x)
       return actions_value
class A2C(nn.Module):
   def __init__(self):
       super(A2C, self).__init__()
       self.critic, self.actor = Critic_Net(), Actor_Net()
       self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=LR)
       self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=LR)
   def choose_action(self, x):
       x = Variable(torch.FloatTensor(x), 0)
       probs = self.actor.forward(x)
       dist = Categorical(probs)
       action = dist.sample()
       return dist, action.detach(), probs
   def learn(self, R, X, next_X, action, GAMMA=0.99):
       X = Variable(torch.FloatTensor(X), 0)
       next_X = Variable(torch.FloatTensor(next_X), 0)
       x_val = self.critic.forward(X)
       next_X_val = self.critic.forward(next_X)
       self.TD_ERROR = R + GAMMA * (next_X_val-x_val)
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
env = gym.make("CartPole-v0")
in_size = env.observation_space.shape[0]
out_size = env.action_space.n
LR = 0.0001
agent = A2C()
for i in range(10000):
   done = False
   obs = env.reset()
   eps_reward = 0
   while not done:
       # env.render()
       dist, action, probs = agent.choose_action(obs)
       next_obs, reward, done, _ = env.step(action.numpy())
       eps_reward += reward
       agent.learn(reward, obs, next_obs, action)
       obs = next_obs
   print("episode: ", i, "reward: ", eps_reward)
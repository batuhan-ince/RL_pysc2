import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym

BATCH_SIZE = 32
LR = 0.0005                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 10000
N_ACTIONS = 4
N_STATES = 4

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 30)
        nn.init.xavier_uniform(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        self.fc2 = nn.Linear(30, 30)
        nn.init.xavier_uniform(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        self.fc3 = nn.Linear(30, 30)
        nn.init.xavier_uniform(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)
        self.out = nn.Linear(30, N_ACTIONS)
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

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0)).to(device)
        # input only one sample

        if np.random.uniform() < self.epsilon:   # random
            action = np.random.randint(0, N_ACTIONS)
        else:   # greedy
            actions_value = self.eval_net.forward(x).to(device)
            action = torch.max(actions_value, 1)[1][0]

        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        print(self.epsilon)
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES])).to(device)
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))).to(device)
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])).to(device)
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:])).to(device)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(32,1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
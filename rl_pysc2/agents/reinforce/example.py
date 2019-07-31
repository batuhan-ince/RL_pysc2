import numpy as np
import torch
import gym
import matplotlib.pyplot as plt

from rl_pysc2.agents.reinforce.model import Reinforce


class Network(torch.nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.policynet = torch.nn.Sequential(
            torch.nn.Linear(in_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, out_size)
        )

        gain = torch.nn.init.calculate_gain("relu")

        def param_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain)
                torch.nn.init.zeros_(module.bias)
        self.apply(param_init)

    def forward(self, state):
        logits = self.policynet(state)
        return logits


if __name__ == "__main__":
    env_name = "CartPole-v0"
    gamma = 0.99

    env = gym.make(env_name)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.n
    network = Network(in_size, out_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    agent = Reinforce(network, optimizer)
    device = "cpu"

    env = gym.make(env_name)
    reward_list = []
    eps_count = 0

    for i in range(1000):
        eps_reward = 0
        done = False
        state = env.reset()
        while done is False:
            state = torch.from_numpy(state).float().to(device)
            action, log_prob = agent(state)
            state, reward, done, _ = env.step(action.item())
            eps_reward += reward
            agent.add_transitions(reward, log_prob)
        # Update
        eps_count += 1
        loss = agent.update(gamma)
        print(("Epsiode: {}, Reward: {}, Loss: {}")
              .format(eps_count, eps_reward, loss), end="\r")

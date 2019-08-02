import numpy as np
import torch
import gym
import matplotlib.pyplot as plt

from rl_pysc2.agents.reinforce.model import Reinforce
from rl_pysc2.networks.common_models import PolicyNet


if __name__ == "__main__":
    env_name = "CartPole-v0"
    gamma = 0.99

    env = gym.make(env_name)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.n
    network = PolicyNet(in_size, out_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
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

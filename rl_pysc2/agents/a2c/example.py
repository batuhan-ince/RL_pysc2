import numpy as np
import torch
import gym
import matplotlib.pyplot as plt

from rl_pysc2.agents.a2c.model import A2C
from rl_pysc2.utils.parallel_envs import ParallelEnv


class Model(torch.nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.policy = torch.nn.Linear(128, out_size)
        self.value = torch.nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        policy = self.policy(x)
        value = self.value(x)
        return policy, value


if __name__ == "__main__":
    env_name = "LunarLander-v2"
    n_step = 20
    n_env = 10
    gamma = 0.99
    beta = 0.01
    tau = 0.99

    env = gym.make(env_name)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.n
    model = Model(in_size, out_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    agent = A2C(model, n_step, optimizer)
    device = "cpu"

    p_env = ParallelEnv(n_env, lambda: gym.make(env_name))
    episode_rewards = np.zeros((n_env, 1))
    reward_list = []

    with p_env as states:
        for i in range(100000//n_step+1):
            # Step
            for j in range(n_step):
                states = torch.from_numpy(states).float().to(device)
                actions, log_probs, entropies, values = agent(states)
                next_states, rewards, dones = p_env.step(actions.to("cpu").numpy())
                # Reward Calculation
                episode_rewards += rewards
                for j, done in enumerate(dones.reshape(-1)):
                    if done == 1:
                        reward_list.append(episode_rewards[j].item())
                        episode_rewards[j] = 0.0
                # Append Transition
                rewards = torch.from_numpy(rewards).to(device).float()
                dones = torch.from_numpy(dones).to(device).float()
                agent.add_transition(values, rewards, dones, log_probs, entropies)
                states = next_states
            # Update
            v_l, p_g, e_g = agent.update(gamma, tau, beta)
            print(" "*80, end="\r")
            print("Episode: {}, Reward: {}, Value Loss: {:.3f}, Policy Gain: {:.3f}".format(
                len(reward_list), np.mean(reward_list[-100:]), v_l.mean().item(), p_g.mean().item()), end="\r")

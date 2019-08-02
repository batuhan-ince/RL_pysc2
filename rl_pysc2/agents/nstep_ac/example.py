import numpy as np
import torch
import gym

from rl_pysc2.agents.nstep_ac.model import NstepAC
from rl_pysc2.networks.common_models import DisjointNet


if __name__ == "__main__":
    env_name = "CartPole-v0"
    gamma = 0.99
    tau = 0.99
    nstep = 20

    env = gym.make(env_name)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.n
    network = DisjointNet(in_size, out_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    agent = NstepAC(network, optimizer)
    device = "cpu"

    env = gym.make(env_name)
    reward_list = []
    eps_count = 0

    def to_torch(array):
        return torch.from_numpy(array).to(device).float().view(1, -1)

    for i in range(1000):
        eps_reward = 0
        eps_loss = 0
        done = False
        state = env.reset()
        state = to_torch(state)
        while done is False:
            for i in range(nstep):
                action, log_prob, value = agent(state)
                next_state, reward, done, _ = env.step(action.item())
                next_state = to_torch(next_state)
                # with torch.no_grad():
                _, next_value = agent.network(next_state)
                agent.add_trans(reward, done,
                                log_prob, value, next_value)
                eps_reward += reward
                state = next_state
                if done is True:
                    break
            loss = agent.update(gamma, tau)
            eps_loss += loss
        # Update
        eps_count += 1
        print(("Episode: {}, Reward: {}, Loss: {}")
              .format(eps_count, eps_reward, loss), end="\r")

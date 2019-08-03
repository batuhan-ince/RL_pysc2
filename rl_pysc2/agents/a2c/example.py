import torch
import numpy as np
import gym

from rl_pysc2.agents.a2c.model import A2C
from rl_pysc2.utils.parallel_envs import ParallelEnv
from rl_pysc2.networks.common_models import DisjointNet


if __name__ == "__main__":

    env_name = "LunarLander-v2"
    gamma = 0.99
    nenv = 16   # For stability
    nstep = 20  # Speed

    env = gym.make(env_name)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.n
    network = DisjointNet(in_size, out_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    agent = A2C(network, optimizer)
    device = "cuda"
    agent.to(device)
    loss = 0
    del env

    penv = ParallelEnv(nenv, lambda: gym.make(env_name))
    eps_rewards = np.zeros((nenv, 1))
    reward_list = [0]

    def to_torch(array):
        if len(array.shape) == 4:
            array = np.transpose(array, (0, 3, 1, 2))
        return torch.from_numpy(array).to(device).float()

    with penv as state:
        state = to_torch(state)
        for i in range(100000):
            for j in range(nstep):
                action, log_prob, value = agent(state)
                action = action.unsqueeze(1).cpu().numpy()
                next_state, reward, done = penv.step(action)
                next_state = to_torch(next_state)
                with torch.no_grad():
                    _, next_value = agent.network(next_state)
                agent.add_trans(to_torch(reward), to_torch(done),
                                log_prob.unsqueeze(1), value,
                                next_value)
                state = next_state
                for j, d in enumerate(done.flatten()):
                    eps_rewards[j] += reward[j].item()
                    if d == 1:
                        reward_list.append(eps_rewards[j].item())
                        eps_rewards[j] = 0
                    print(("Epsiode: {}, Reward: {}, Loss: {}")
                          .format(len(reward_list)//nenv,
                                  np.mean(reward_list[-100:]), loss),
                          end="\r")
            loss = agent.update(gamma)

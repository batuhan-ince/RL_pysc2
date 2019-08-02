import gym
import sys
import numpy as np
from absl import flags
import torch
import argparse
import matplotlib.pyplot as plt

from rl_pysc2.gym_envs.base_env import SC2Env
from rl_pysc2.gym_envs.move_to_beacon import MoveToBeaconEnv
from rl_pysc2.gym_envs.collect_mineral_shards import CollectMineralEnv
from rl_pysc2.networks.starcraft_models import ScreenNet
from rl_pysc2.agents.a2c.model import A2C
from rl_pysc2.utils.parallel_envs import ParallelEnv


FLAGS = flags.FLAGS
flags.DEFINE_bool("render_sync", False, "Turn on sync rendering.")
FLAGS(sys.argv)


def train():
    env = MoveToBeaconEnv()
    # env.settings['visualize'] = True

    gamma = 0.99
    nenv = 2   # For stability
    nstep = 20  # Speed
    tau = 0.99
    n_timesteps = 100000

    env = MoveToBeaconEnv()
    in_channel = env.observation_space.shape[0]
    # action_size = screen_size**2
    screen_size = int(np.sqrt(env.action_space.n))
    network = ScreenNet(in_channel, screen_size)
    env.close()
    del env
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    agent = A2C(network, optimizer)
    device = "cuda"
    agent.to(device)
    loss = 0

    penv = ParallelEnv(nenv, MoveToBeaconEnv)
    eps_rewards = np.zeros((nenv, 1))
    reward_list = [0]

    def to_torch(array):
        return torch.from_numpy(array).to(device).float()

    with penv as state:
        state = to_torch(state)
        for i in range(n_timesteps//nstep):
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
            loss = agent.update(gamma, tau)
            if i % 10 == 0:
                agent.save_model("model_parameters.p")
        plt.plot([np.mean(reward_list[ind:ind+100])
                  for ind in range(len(reward_list)-100)])
        plt.savefig("reward.png")


if __name__ == "__main__":
    train()

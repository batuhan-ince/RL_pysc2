import gym
import sys
import numpy as np
from absl import flags
import torch

from rl_pysc2.gym_envs.base_env import SC2Env
from rl_pysc2.gym_envs.move_to_beacon import MoveToBeaconEnv
from rl_pysc2.gym_envs.collect_mineral_shards import CollectMineralEnv
from rl_pysc2.networks.deepmind_model import Encode, Output
from rl_pysc2.agents.a2c.model import A2C
from rl_pysc2.utils.parallel_envs import ParallelEnv


FLAGS = flags.FLAGS
flags.DEFINE_bool("render_sync", False, "Turn on sync rendering.")
FLAGS(sys.argv)


class Network(torch.nn.Module):

    def __init__(self, in_channel, out_size):
        super().__init__()
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, 64, 5, 1, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 5, 1, padding=2),
            torch.nn.ReLU(),
        )

        self.policy = torch.nn.Conv2d(32, 1, 5, 1, padding=2)
        self.value = torch.nn.Sequential(
            torch.nn.Linear(64*64*32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        
        gain = torch.nn.init.calculate_gain("relu")

        def param_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain)
                torch.nn.init.zeros_(module.bias)
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.dirac_(module.weight)
                torch.nn.init.zeros_(module.bias)                
        self.apply(param_init)

    def forward(self, state):
        encode = self.convnet(state)

        value = self.value(encode.reshape(-1, 64*64*32))
        logits = self.policy(encode).reshape(-1, 64*64)

        return logits, value


if __name__ == "__main__":
    env = MoveToBeaconEnv()
    # env.settings['visualize'] = True

    gamma = 0.99
    nenv = 2   # For stability
    nstep = 20  # Speed
    tau = 0.99

    env = MoveToBeaconEnv()
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.n
    network = Network(in_size, out_size)
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
            loss = agent.update(gamma, tau)

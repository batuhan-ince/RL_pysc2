import gym
import sys
import numpy as np
from absl import flags
import torch
import argparse
import logging
import lamp

from rl_pysc2.gym_envs.collect_mineral_shards import CollectMineralEnv
from rl_pysc2.networks.starcraft_models import ScreenNet
from rl_pysc2.agents.a2c.model import A2C
from rl_pysc2.utils.parallel_envs import ParallelEnv
from gymcolab.envs.simplemaze import SimpleMaze


class ScreenNet(torch.nn.Module):
    """ Model for movement based mini games in sc2.
    This network only takes screen input and only returns spatial outputs.
    Some of the example min games are MoveToBeacon and CollectMineralShards.
    Arguments:
        - in_channel: Number of feature layers in the screen input
        - screen_size: Screen size of the mini game. If 64 is given output
            size will be 64*64
    Note that output size depends on screen_size.
    """

    def __init__(self, in_channel, screen_size):
        super().__init__()
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, 64, 5, 1, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 5, 1, padding=2),
            torch.nn.ReLU(),
        )

        self.policy = torch.nn.Sequential(
            torch.nn.Linear(screen_size*screen_size*32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 5)
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(screen_size*screen_size*32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

        self.screen_size = screen_size
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

        value = self.value(
            encode.reshape(-1, self.screen_size*self.screen_size*32))
        logits = self.policy(
            encode.reshape(-1, self.screen_size*self.screen_size*32))

        return logits, value


def train():
    env = SimpleMaze()
    # env.settings['visualize'] = True
    logger = logger_config()

    hyperparams = dict(
        gamma=0.99,
        nenv=8,
        nstep=20,
        n_timesteps=100000,
        lr=0.0001,
    )

    in_channel = env.observation_space.shape[0]
    screen_size = env.observation_space.shape[1]
    network = ScreenNet(in_channel, screen_size)
    env.close()
    del env
    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=hyperparams["lr"])
    agent = A2C(network, optimizer)
    device = "cpu"
    agent.to(device)
    loss = 0

    penv = ParallelEnv(hyperparams["nenv"], SimpleMaze)
    eps_rewards = np.zeros((hyperparams["nenv"], 1))
    reward_list = [0]

    def to_torch(array):
        return torch.from_numpy(array).to(device).float()
    logger.hyperparameters(hyperparams, win="Hyperparameters")

    with penv as state:
        state = to_torch(state)
        for i in range(hyperparams["n_timesteps"]//hyperparams["nstep"]):
            for j in range(hyperparams["nstep"]):
                print(state.shape)
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
                        logger.scalar(np.mean(reward_list[-5:]),
                                      win="reward", trace="Last 5")
                        logger.scalar(np.mean(reward_list[-10:]),
                                      win="reward", trace="Last 10")
                        logger.scalar(loss, win="loss")
                    print(("Epsiode: {}, Reward: {}, Loss: {}")
                          .format(len(reward_list)//hyperparams["nenv"],
                                  np.mean(reward_list[-100:]), loss),
                          end="\r")
            loss = agent.update(hyperparams["gamma"])
            if i % 10 == 0:
                agent.save_model("model_parameters.p")


def logger_config():
    import yaml
    import logging.config
    with open('logger_config.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)
    return logger


if __name__ == "__main__":
    train()
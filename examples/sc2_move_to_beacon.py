import gym
import sys
import numpy as np
from absl import flags
import torch
import argparse
import logging
import lamp

from rl_pysc2.gym_envs.base_env import SC2Env
from rl_pysc2.gym_envs.move_to_beacon import MoveToBeaconEnv
from rl_pysc2.gym_envs.collect_mineral_shards import CollectMineralEnv
from rl_pysc2.networks.starcraft_models import ScreenNet
from rl_pysc2.agents.a2c.model import A2C
from rl_pysc2.utils.parallel_envs import ParallelEnv


FLAGS = flags.FLAGS
flags.DEFINE_bool("render_sync", False, "Turn on sync rendering.")
FLAGS(sys.argv)


class MultiCategoricalDist:
    def __init__(self, *logits):
        self.dists = [torch.distributions.Categorical(logits=logit)
                      for logit in logits]

    def sample(self):
        return [dist.sample() for dist in self.dists]

    def log_prob(self, *acts):
        return sum(dist.log_prob(act) for act, dist in zip(acts, self.dists))

    def entropy(self):
        product = 1
        for dist in self.dists:
            product *= dist.probs
        log_sum = sum(dist.logits for dist in self.dists)
        return -(log_sum*product).sum(-1)


class StarcraftAC2(A2C):
    def forward(self, state):
        logits, value = self.network(state)
        dist = MultiCategoricalDist(*logits)
        action = dist.sample()
        log_prob = dist.log_prob(*action)
        entropy = dist.entropy()
        action = action[0]*64 + action[1]

        return action, log_prob, value, entropy


def train(param_path, suffix, hyperparams):
    env = MoveToBeaconEnv()
    # env.settings['visualize'] = True
    logger = logger_config()

    env = MoveToBeaconEnv()
    in_channel = env.observation_space.shape[0]
    # action_size = screen_size**2
    screen_size = int(np.sqrt(env.action_space.n))
    network = ScreenNet(in_channel, screen_size)
    env.close()
    del env
    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=hyperparams["lr"])
    agent = StarcraftAC2(network, optimizer)
    device = "cuda"
    agent.to(device)
    loss = 0

    penv = ParallelEnv(hyperparams["nenv"], MoveToBeaconEnv)
    eps_rewards = np.zeros((hyperparams["nenv"], 1))
    reward_list = [0]

    def to_torch(array):
        return torch.from_numpy(array).to(device).float()
    logger.hyperparameters(hyperparams, win="Hyperparameters")

    with penv as state:
        state = to_torch(state)
        for i in range(hyperparams["n_timesteps"]//hyperparams["nstep"]):
            for j in range(hyperparams["nstep"]):
                action, log_prob, value, entropy = agent(state)
                action = action.unsqueeze(1).cpu().numpy()
                next_state, reward, done = penv.step(action)
                next_state = to_torch(next_state)
                with torch.no_grad():
                    _, next_value = agent.network(next_state)
                agent.add_trans(to_torch(reward), to_torch(done),
                                log_prob.unsqueeze(1), value,
                                next_value, entropy)
                state = next_state
                for j, d in enumerate(done.flatten()):
                    eps_rewards[j] += reward[j].item()
                    if d == 1:
                        reward_list.append(eps_rewards[j].item())
                        eps_rewards[j] = 0
                        logger.scalar(np.mean(reward_list[-10:]),
                                      win="reward" + suffix, trace="Last 10")
                        logger.scalar(np.mean(reward_list[-50:]),
                                      win="reward" + suffix, trace="Last 50")
                        logger.scalar(loss, win="loss" + suffix)
                    # print(("Epsiode: {}, Reward: {}, Loss: {}")
                    #       .format(len(reward_list)//hyperparams["nenv"],
                    #               np.mean(reward_list[-50:]), loss),
                    #       end="\r")
            loss = agent.update(hyperparams["gamma"], hyperparams["beta"])
            if i % 10 == 0:
                agent.save_model(param_path)


def logger_config():
    import yaml
    import logging.config
    with open('logger_config.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)
    return logger


def run():
    NUM_PROCESSES = 1
    PARAM_DIR = "models/A2C_vanilla/"
    HYPERPARAMS = dict(
        gamma=0.99,
        nenv=1,
        nstep=20,
        n_timesteps=1000000,
        lr=0.0001,
        beta=0.03,
    )

    processes = []
    for i in range(NUM_PROCESSES):
        suffix = str(i)
        param_dir = PARAM_DIR + suffix + "/model.b"
        process = torch.multiprocessing.Process(
            target=train,
            args=(param_dir, suffix, HYPERPARAMS)
        )
        processes.append(process)
        process.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    run()

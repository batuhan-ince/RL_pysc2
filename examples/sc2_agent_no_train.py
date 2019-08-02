import gym
import sys
import numpy as np
from absl import flags
import torch

from rl_pysc2.gym_envs.base_env import SC2Env
from rl_pysc2.gym_envs.move_to_beacon import MoveToBeaconEnv
from rl_pysc2.gym_envs.collect_mineral_shards import CollectMineralEnv
from rl_pysc2.networks.starcraft_models import DeepMindNet


FLAGS = flags.FLAGS
flags.DEFINE_bool("render_sync", False, "Turn on sync rendering.")
FLAGS(sys.argv)

env = CollectMineralEnv()
env.settings['visualize'] = True
obs = env.reset()

n_screen_features = [frame.shape[0] for frame in obs[0]]
n_nonspatial_features = obs[2].shape[0]
n_minimap_features = [frame.shape[0] for frame in obs[1]]

model = DeepMindNet(n_screen_features, n_minimap_features,
                    n_nonspatial_features, 64, 10)
obs = model.obs_to_torch(obs)
spatial, value, policy = model(obs)
for i in range(1000):
    done = False
    print(env.available_actions)
    obs = env.reset()
    while not done:
        action = np.random.randint(0, 64, size=2)
        obs, reward, done, info = env.step(action)
    print("Reward: ", env.episode_reward)

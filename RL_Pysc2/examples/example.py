import gym
import sys
import numpy as np
from absl import flags
import torch


from rl_pysc2.gym_envs.base_env import SC2Env
from rl_pysc2.gym_envs.move_to_beacon import MoveToBeaconEnv
from rl_pysc2.gym_envs.collect_mineral_shards import CollectMineralEnv


from rl_pysc2.networks.deepmind_model import Encode, Output



FLAGS = flags.FLAGS
flags.DEFINE_bool("render_sync", False, "Turn on sync rendering.")
FLAGS(sys.argv)

env = CollectMineralEnv()
env.settings['visualize'] = True
obs = env.reset()
encode = Encode(n_minimap_features=[frame.shape[0] for frame in obs[1]], n_nonspatial_features=obs[2].shape[0], n_screen_features=[frame.shape[0] for frame in obs[0]])
obs = encode.obs_to_torch(obs)
map=encode.forward(obs)
output = Output(map.shape[1],64,64,10)
spatial, value, policy = output(map)
for i in range(100):
    done=False
    print(env.available_actions)
    obs = env.reset()
    while not done:
        action = np.random.randint(0, 64, size=2)
        obs, reward, done, info = env.step(action)
    print("Reward: ",env.episode_reward)




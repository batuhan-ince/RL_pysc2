import numpy as np
import torch
from torch import nn

from rl_pysc2.networks.deepmind_model import Encode, Output


class A2C(nn.Module):
    def __init__(self, obs, n_actions, actor_learning_rate= 0.00001, critic_learning_rate= 0.0001, gamma=0.99):
        super(A2C, self).__init__()
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.obs = obs
        self.n_actions = n_actions

    def build(self):
        encode = Encode(n_screen_features=[frame.shape[0] for frame in self.obs[0]],
                        n_minimap_features=[frame.shape[0] for frame in self.obs[1]],
                        n_nonspatial_features=self.obs[2].shape[0])
        self.obs=encode.obs_to_torch(self.obs)
        feature_map=encode.forward(self.obs)
        output = Output(feature_map.shape[1], 64, 64,self.n_actions)
        spatial, value, policy = output(feature_map)
        spatial_log_prob
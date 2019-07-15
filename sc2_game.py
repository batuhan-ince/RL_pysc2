"""
   Copyright 2017 Islam Elnabarawy

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import gym
import sys
import numpy as np
from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions
from pysc2.lib import features
from absl import flags

FLAGS = flags.FLAGS
__author__ = 'Islam Elnabarawy'

_NO_OP = actions.FUNCTIONS.no_op.id


class SC2GameEnv(gym.Env):
    metadata = {'render.modes': [None, 'human']}
    default_settings = {'agent_interface_format': sc2_env.parse_agent_interface_format(
        feature_screen=64,
        feature_minimap=64,
    )}

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._kwargs = kwargs
        self._env = None

        self._episode = 0
        self._num_step = 0
        self._episode_reward = 0
        self._total_reward = 0

    def step(self, action):
        return self._safe_step(action)

    def _safe_step(self, action):
        self._num_step += 1
        if action[0] not in self.available_actions:
            raise ValueError("Invalid Action")
        obs = self._env.step([actions.FunctionCall(action[0], action[1:])])[0]
        self.available_actions = obs.observation['available_actions']
        reward = obs.reward
        self._episode_reward += reward
        self._total_reward += reward
        return self.preprocess_obs(obs), reward, obs.step_type == StepType.LAST, {}

    def reset(self):
        if self._env is None:
            self._init_env()
        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0
        obs = self._env.reset()[0]
        self.available_actions = obs.observation['available_actions']
        return self.preprocess_obs(obs)

    def save_replay(self, replay_dir):
        self._env.save_replay(replay_dir)

    def _init_env(self):
        args = {**self.default_settings, **self._kwargs}
        self._env = sc2_env.SC2Env(**args)

    def close(self):
        if self._env is not None:
            self._env.close()
        super().close()

    def preprocess_obs(self, obs):
        screen = obs.observation.feature_screen
        minimap = obs.observation.feature_minimap
        info = obs.observation.player
        screen = self.spatial_preprocess(screen,features.SCREEN_FEATURES)
        minimap = self.spatial_preprocess(minimap, features.MINIMAP_FEATURES)
        info = np.log(info + 1)
        return screen, minimap, info

    def spatial_preprocess(self, feature_map, feature_info):

        frames = []

        def _one_hot(input, limit):
            arange = np.arange(limit).reshape(-1, 1, 1)
            return (arange == input).astype(np.float32)
        for i, (_, _, _, _, scale, feature_type, _, _) in enumerate(feature_info):
            if feature_type == features.FeatureType.CATEGORICAL:
                frames.append(_one_hot(np.expand_dims(feature_map[i], 0), scale))
            elif feature_type == features.FeatureType.SCALAR:
                frames.append(np.log(np.expand_dims(feature_map[i], 0) + 1))
        return np.concatenate(frames, 0)

    @property
    def settings(self):
        return self._kwargs

    @property
    def action_spec(self):
        if self._env is None:
            self._init_env()
        return self._env.action_spec()

    @property
    def observation_spec(self):
        if self._env is None:
            self._init_env()
        return self._env.observation_spec()

    @property
    def episode(self):
        return self._episode

    @property
    def num_step(self):
        return self._num_step

    @property
    def episode_reward(self):
        return self._episode_reward

    @property
    def total_reward(self):
        return self._total_reward


if __name__ == "__main__":
    FLAGS(sys.argv)
    _MAP_NAME = 'MoveToBeacon'
    env = SC2GameEnv(map_name='MoveToBeacon')
    obs = env.reset()
    print([ob.shape for ob in obs])
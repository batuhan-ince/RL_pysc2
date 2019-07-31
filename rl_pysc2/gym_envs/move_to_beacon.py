"""
This module is adapted from Islam Elnabarawy
https://github.com/islamelnabarawy/sc2gym
MovetoBeacon
"""

from pysc2.lib import actions, features
import numpy as np
import gym.spaces as spaces

from rl_pysc2.gym_envs.base_env import SC2Env


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale

_NO_OP = actions.FUNCTIONS.no_op.id

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NOT_QUEUED = [0]

_MAP_NAME = 'MoveToBeacon'


class MoveToBeaconEnv(SC2Env):
    def __init__(self):
        super(MoveToBeaconEnv, self).__init__(map_name=_MAP_NAME)
        self.action_space = spaces.Discrete(64*64)
        high = np.ones((5, 64, 64), dtype=np.int32)
        low = np.zeros((5, 64, 64), dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high)

    def reset(self):
        super().reset()
        return self._post_reset()

    def _post_reset(self):
        obs, reward, done, info = self._safe_step([_SELECT_ARMY, _SELECT_ALL])
        return obs

    def step(self, action):
        y = action // 64
        x = action % 64
        action = [_MOVE_SCREEN, _NOT_QUEUED, [y, x]]
        obs, reward, done, info = self._safe_step(action)
        return obs, reward, done, info

    def preprocess_obs(self, obs):
        screen = obs.observation.feature_screen
        screen = self.spatial_preprocess([screen[_PLAYER_RELATIVE]],
                                         [features.SCREEN_FEATURES[_PLAYER_RELATIVE]])
        return screen[0]
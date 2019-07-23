"""
This module is adapted from Islam Elnabarawy
https://github.com/islamelnabarawy/sc2gym
Collect Mineral Shards
"""

from pysc2.lib import actions, features
import numpy as np

from rl_pysc2.gym_envs.base_env import SC2Env


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale

_NO_OP = actions.FUNCTIONS.no_op.id

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NOT_QUEUED = [0]

_MAP_NAME = 'CollectMineralShards'


class CollectMineralEnv(SC2Env):
    def __init__(self):
        super(CollectMineralEnv, self).__init__(map_name=_MAP_NAME)

    def reset(self):
        super().reset()
        return self._post_reset()

    def _post_reset(self):
        obs, reward, done, info = self._safe_step([_SELECT_ARMY, _SELECT_ALL])
        return obs

    def step(self, action):
        action = [_MOVE_SCREEN, _NOT_QUEUED, [action[0], action[1]]]
        obs, reward, done, info = self._safe_step(action)
        return obs, reward, done, info


import sys

from absl import flags

from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features
import numpy as np
import matplotlib.pyplot as plt
import math
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY
_SELECTED_UNIT = features.SCREEN_FEATURES.selected.index
FUNCTIONS = actions.FUNCTIONS

_SELECT_ALL = [0]
_NOT_QUEUED = [0]

def xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))

def positions(obs): #Returns the distance to the closest mineral and the state

    marine_map = (obs[0].observation.feature_screen.base[5] == 1)
    beacon_map = (obs[0].observation.feature_screen.base[5] == 3)
    state = np.dstack([marine_map, beacon_map]).reshape(16 * 16 * 16*2).astype(int)
    return  state



def calc_distance(observation):
    actual_obs = observation[0]
    scrn_player = actual_obs.observation.feature_screen.player_relative
    scrn_select = actual_obs.observation.feature_screen.selected
    scrn_density = actual_obs.observation.feature_screen.unit_density

    state_added = scrn_select + scrn_density

    marine_center = np.mean(xy_locs(scrn_player == 1), axis=0).round()

    # first step
    if np.sum(scrn_select) == 0:
        marine_center = np.mean(xy_locs(scrn_player == 1), axis=0).round()
        # marine behind beacon
        if isinstance(marine_center, float):
            marine_center = np.mean(xy_locs(state_added == 2), axis=0).round()
    else:
        # normal navigation
        marine_center = np.mean(xy_locs(state_added == 2), axis=0).round()
        if isinstance(marine_center, float):
            marine_center = np.mean(xy_locs(state_added == 3), axis=0).round()

    beacon_center = np.mean(xy_locs(scrn_player == 3), axis=0).round()

    distance = math.hypot(beacon_center[0] - marine_center[0],
                          beacon_center[1] - marine_center[1])

    state = np.dstack((marine_center, beacon_center)).reshape(4)

    return beacon_center, marine_center, distance, state

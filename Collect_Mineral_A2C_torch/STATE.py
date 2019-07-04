import sys

from absl import flags

from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features
import numpy as np
import matplotlib.pyplot as plt

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY
_SELECTED_UNIT = features.SCREEN_FEATURES.selected.index
FUNCTIONS = actions.FUNCTIONS

_SELECT_ALL = [0]
_NOT_QUEUED = [0]

def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))

def positions(obs): #Returns the distance to the closest mineral and the state
    player_relative = obs[0].observation.feature_screen.player_relative

    marines = _xy_locs(player_relative == _PLAYER_SELF)
    minerals = _xy_locs(player_relative == _PLAYER_NEUTRAL)
    marine_xy = np.mean(marines, axis=0).round()  # Average location.
    distances = np.linalg.norm(np.array(minerals)-marine_xy, axis=1)
    closest_mineral_xy = minerals[np.argmin(distances)]
    dist2 = np.linalg.norm(np.array(closest_mineral_xy) - marine_xy, axis=0)
    state = np.dstack((marine_xy, closest_mineral_xy)).reshape(4)
    return dist2, state


def distss(obs): #can be used for future reward modifies
    player_relative = obs[0].observation.feature_screen.player_relative
    minerals_y, minerals_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
    marines_y, marines_x = (player_relative == _PLAYER_SELF).nonzero()
    marines_x, marines_y, minerals_x, minerals_y = np.mean(marines_x), np.mean(marines_y), np.mean(minerals_x), np.mean(minerals_y)

    now_distance = ((marines_x/63 - minerals_x/63)**2 + (marines_y/63 - minerals_y/63)**2)

    return now_distance

def obs2done(obs): #Calculates mineral points
    collected_mineral = (obs[0].observation['player'][1])
    return collected_mineral
import sys
from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features
import matplotlib.pyplot as plt
import numpy as np
from STATE import calc_distance

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
friendly = 1
neutral = 3
_SELECTED_UNIT = features.SCREEN_FEATURES.selected.index

_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NO_OP = actions.FUNCTIONS.no_op.id
_RALLY_UNITS_SCREEN = actions.FUNCTIONS.Rally_Units_screen.id

_SELECT_ALL = [0]
_NOT_QUEUED = [0]


def no_operation(obs):
    action = actions.FunctionCall(_NO_OP, [])
    return action


def move_unit(obs, mode):
    _, marine_center, __, ___ = calc_distance(obs)
    target_x, target_y = marine_center

    if mode == 1:  # up
        dest_x, dest_y = np.clip(target_x, 0, 63), np.clip(target_y - 10, 0, 63)
    elif mode == 2:  # down
        dest_x, dest_y = np.clip(target_x, 0, 63), np.clip(target_y + 10, 0, 63)
    elif mode == 3:  # left
        dest_x, dest_y = np.clip(target_x - 10, 0, 63), np.clip(target_y, 0, 63)
    elif mode == 4:  # right
        dest_x, dest_y = np.clip(target_x + 10, 0, 63), np.clip(target_y, 0, 63)
    elif mode == 5:
        dest_x, dest_y = np.clip(target_x + 10, 0, 63), np.clip(target_y+10, 0, 63)
    elif mode == 6:
        dest_x, dest_y = np.clip(target_x - 10, 0, 63), np.clip(target_y-10, 0, 63)
    elif mode == 7:
        dest_x, dest_y = np.clip(target_x + 10, 0, 63), np.clip(target_y-10, 0, 63)
    elif mode == 8:
        dest_x, dest_y = np.clip(target_x - 10, 0, 63), np.clip(target_y+10, 0, 63)
    elif mode == 9:
        dest_x, dest_y = np.clip(target_x + 10, 0, 63), np.clip(target_y+5, 0, 63)
    elif mode == 10:
        dest_x, dest_y = np.clip(target_x + 5, 0, 63), np.clip(target_y+10, 0, 63)
    elif mode == 11:
        dest_x, dest_y = np.clip(target_x - 5, 0, 63), np.clip(target_y-10, 0, 63)
    elif mode == 12:
        dest_x, dest_y = np.clip(target_x - 10, 0, 63), np.clip(target_y-5, 0, 63)
    elif mode == 13:
        dest_x, dest_y = np.clip(target_x + 5, 0, 63), np.clip(target_y-10, 0, 63)
    elif mode == 14:
        dest_x, dest_y = np.clip(target_x + 10, 0, 63), np.clip(target_y-5, 0, 63)
    elif mode == 15:
        dest_x, dest_y = np.clip(target_x - 5, 0, 63), np.clip(target_y+10, 0, 63)
    elif mode == 16:
        dest_x, dest_y = np.clip(target_x - 10, 0, 63), np.clip(target_y+5, 0, 63)

    action = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [dest_x, dest_y]])

    return action


def actAgent2Pysc2(i, obs):
    if i == 0:
        action = move_unit(obs, 1)
    elif i == 1:
        action = move_unit(obs, 2)
    elif i == 2:
        action = move_unit(obs, 3)
    elif i == 3:
        action = move_unit(obs, 4)
    elif i == 4:
        action = move_unit(obs, 5)
    elif i == 5:
        action = move_unit(obs, 6)
    elif i == 6:
        action = move_unit(obs, 7)
    elif i == 7:
        action = move_unit(obs, 8)
    elif i == 8:
        action = move_unit(obs, 9)
    elif i == 9:
        action = move_unit(obs, 10)
    elif i == 10:
        action = move_unit(obs, 11)
    elif i == 11:
        action = move_unit(obs, 12)
    elif i == 12:
        action = move_unit(obs, 13)
    elif i == 13:
        action = move_unit(obs, 14)
    elif i == 14:
        action = move_unit(obs, 15)
    elif i == 15:
        action = move_unit(obs, 16)
    elif i == 100:
        action = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
    return action

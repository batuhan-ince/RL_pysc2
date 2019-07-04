import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import torch
import matplotlib.pyplot as plt
from ACTION import actAgent2Pysc2, no_operation
from STATE import obs2state, obs2distance
import numpy as np
import random
import tensorflow as tf
from collections import deque
import time
import math
from A2C import A2C


# Define the constant
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED_UNIT = features.SCREEN_FEATURES.selected.index
friendly = 1
neutral = 3
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NO_OP           = actions.FUNCTIONS.no_op.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_SELECT_ALL  = [0]
_NOT_QUEUED  = [0]
step_mul = 4
FLAGS = flags.FLAGS
EPISODES = 10000

def train():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(map_name="MoveToBeacon", step_mul=step_mul, visualize=True,
                        agent_interface_format=sc2_env.AgentInterfaceFormat(
                            feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64))) as env:
        a2c = A2C()
        rwd = []
        for episodes in range(EPISODES):
            done = False
            obs = env.reset()
            state = np.array(obs2state(obs))
            print('episode start')
            global_step = 0
            reward = 0
            cum_rew = 0
            first_distance = obs2distance(obs)
            while not done:
                global_step += 1
                time.sleep(0.2)
		#selecting marine
                while not 331 in obs[0].observation["available_actions"]:
                    actions = actAgent2Pysc2(100, obs)
                    obs = env.step(actions=[actions])
                _, action, __ = a2c.choose_action(state)
                actions = actAgent2Pysc2(action, obs)
                obs = env.step(actions=[actions])
                distance = obs2distance(obs)
                if global_step == 1:
                    pre_distance = distance
                next_state = obs2state(obs)
		#reward enginnering part
                reward = -(distance * 20)
                reward2 = -(distance - pre_distance) * 20 / first_distance
                if distance < 0.03:
                    reward = 10
                    print('+10 reward recieved')
                    done = True

                if obs[0].step_type == environment.StepType.LAST:
                    reward = -10
                    print('-10 reward recieved')
                    done = True

                a2c.learn(reward ,state, action)
                cum_rew = reward + cum_rew

                if distance < 0.03 or global_step == 200:
                    if cum_rew != 10 or cum_rew != 10.0:
                        rwd.append(cum_rew)
                    break
                state = next_state
                pre_distance = distance

            print("episode: ", episodes, "reward: ", cum_rew)
            if episodes%1000==1:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                nn_filename = "a2cAgent_Trained_Model_" + timestr + "  "+str(episodes)+ ".pth"
                torch.save(a2c.actor.state_dict(), nn_filename)

if __name__ == '__main__':
    train()


import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import matplotlib.pyplot as plt
from Collect_Mineral_A2C_torch.action import actAgent2Pysc2, no_operation
from STATE import distss, positions, obs2done
import numpy as np
import torch
import time
from A2C import A2C

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED_UNIT = features.SCREEN_FEATURES.selected.index
friendly = 1
neutral = 3
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NO_OP = actions.FUNCTIONS.no_op.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]
step_mul = 16
FLAGS = flags.FLAGS
EPISODES = (5+5)**4
BATCH_SIZE = 500

def train():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(map_name="CollectMineralShards", step_mul=step_mul, visualize=True,
                        agent_interface_format=sc2_env.AgentInterfaceFormat(
                            feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64))) as env:
        a2c = A2C()
        rwd = []
        for episodes in range(EPISODES):
            done = False
            obs = env.reset()

            _,state = positions(obs)
            global_step = 0
            reward = 0
            cum_rew = 0
            while not done:
                start = time.time()
                global_step += 1
                time.sleep(0.2)
                while not 331 in obs[0].observation["available_actions"]:
                    actions = actAgent2Pysc2(100, obs)
                    obs = env.step(actions=[actions])
                _, action, __ = a2c.choose_action(state)

                actions = actAgent2Pysc2(action, obs)
                obs = env.step(actions=[actions])
                dist, next_state = positions(obs)
                if global_step == 1:
                    pre_distance = dist
                reward= obs[0].reward
                if obs2done(obs) > 1900: #Collecting all the minerals ends the episode
                    done = True
                if obs[0].step_type == environment.StepType.LAST:#Ending  episode if it's the last step
                    done = True
                a2c.learn(reward ,state, action) #Learning
                cum_rew = reward + cum_rew
                state = next_state
                pre_distance = dist
            print('episode: ', episodes, 'reward: ', cum_rew)
            #Saving the net
            if episodes % 1000 == 1:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                nn_filename = "CMS_A2C_Agent_Trained_Model_" + timestr + "  " + str(episodes) + ".pth"
                torch.save(a2c, nn_filename)

if __name__ == '__main__':
    train()




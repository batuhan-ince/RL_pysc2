import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import torch
from ACTION import actAgent2Pysc2, no_operation
from STATES import calc_distance
import numpy as np
import time
from DQN import DQN, MEMORY_CAPACITY
from visdom import Visdom
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

# Define the constants
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
EPISODES = 1000


def test():
    # defining map
    FLAGS(sys.argv)
    with sc2_env.SC2Env(map_name="MoveToBeacon", step_mul=step_mul, visualize=True,
                        agent_interface_format=sc2_env.AgentInterfaceFormat(
                            feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64))) as env:
	# when train is complete, we'll use below code with saved model
        #dqn = torch.load("dqnAgent_Trained_Model_20190711-104115  10001.pth")
        scores = []
        for episodes in range(EPISODES):
            done = False
            obs = env.reset()
            _, __, ___, state = calc_distance(obs)
            print('episode start')
            global_step = 0
            reward = 0  # step reward
            cum_rew = 0  # episodic reward
            while not done:
                global_step += 1
                # selecting marines in while loop
                while not 331 in obs[0].observation["available_actions"]:
                    actions = actAgent2Pysc2(100, obs)
                    obs = env.step(actions=[actions])
                action = dqn.choose_action(state)  # choosing the action according to DQN
                actions = actAgent2Pysc2(action, obs)  # Calling action from ACTION.py
                obs = env.step(actions=[actions])
                _, __, distance, next_state = calc_distance(obs)
                if global_step == 1:
                    pre_distance = distance

                # Modifying reward to solve sparse reward issue
                reward = obs[0].reward

                    # Detecting the end of the episode
                if obs[0].step_type == environment.StepType.LAST:
                    done = True
                # dqn.store_transition(state, action, reward, next_state)
                # if done == True:
                #     if dqn.memory_counter > MEMORY_CAPACITY:
                #         dqn.learn()
                cum_rew += reward
                state = next_state
                pre_distance = distance
            scores.append(cum_rew)

            print("episode: ", episodes, "reward: ", cum_rew)
            # saving the net
            # if episodes % 1000 == 1:
            #     timestr = time.strftime("%Y%m%d-%H%M%S")
            #     nn_filename = "dqnAgent_Trained_Model_" + timestr + "  " + str(episodes) + ".pth"
            #     torch.save(dqn, nn_filename)
    return scores
if __name__ == '__main__':
    scores = test()
    plt.plot(list(range(len(scores))), scores)
    textstr = '\n'.join((
        r'$\mathrm{Gamma}=%.2f$' % (0.99,),
        r'$\mathrm{Learning Rate}=%.4f$' % (0.0005,),
        r'$\mathrm{Mean}=%.4f$' % (np.mean(scores),),
        r'$\mathrm{Max}=%.4f$' % (np.max(scores),)))
    plt.title("DQN MoveToBeacon")
    plt.xlabel("EPISODE")
    plt.ylabel("SCORE")
    plt.figtext(0.15, 0.75, textstr, wrap=True, horizontalalignment='left')
    plt.show()

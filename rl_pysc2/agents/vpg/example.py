import numpy as np
import torch
import gym

from rl_pysc2.agents.vpg.model import VGP


class Network(torch.nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.policynet = torch.nn.Sequential(
            torch.nn.Linear(in_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, out_size)
        )

        self.valuenet = torch.nn.Sequential(
            torch.nn.Linear(in_size, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
            # torch.nn.BatchNorm1d(128, affine=True)
        )

        gain = torch.nn.init.calculate_gain("relu")

        def param_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain)
                torch.nn.init.zeros_(module.bias)
        self.apply(param_init)

    def forward(self, state):
        value = self.valuenet(state)
        logits = self.policynet(state)

        return logits, value


if __name__ == "__main__":
    env_name = "CartPole-v0"
    gamma = 0.99

    env = gym.make(env_name)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.n
    network = Network(in_size, out_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    agent = VGP(network, optimizer)
    device = "cpu"

    env = gym.make(env_name)
    reward_list = []
    eps_count = 0

    def to_torch(array):
        return torch.from_numpy(array).to(device).float().view(1, -1)

    for i in range(1000):
        eps_reward = 0
        eps_loss = 0
        done = False
        state = env.reset()
        state = to_torch(state)
        while done is False:
            action, log_prob, value = agent(state)
            next_state, reward, done, _ = env.step(action.item())
            next_state = to_torch(next_state)
            # with torch.no_grad():
            _, next_value = agent.network(next_state)
            trans = agent.Transition(reward, done, log_prob, value, next_value)
            loss = agent.update(trans, gamma)
            eps_reward += reward
            eps_loss += loss
            state = next_state
        # Update
        eps_count += 1
        print(("Epsiode: {}, Reward: {}, Loss: {}")
              .format(eps_count, eps_reward, loss), end="\r")

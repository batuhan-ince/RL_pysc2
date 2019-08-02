import torch


class DisjointNet(torch.nn.Module):
    """ Seperate networks for value and policy.
    This model tends to work better at simpler environments (CartPole,
    LunarLander).
    Arguments
        - in_size: Observation size
        - out_size: Number of possible actions
    """
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


class PolicyNet(torch.nn.Module):
    """ Reinforce network which only outputs policy.
        Arguments
            - in_size: Observation size
            - out_size: Number of possible actions
    """
    def __init__(self, in_size, out_size):
        super().__init__()
        self.policynet = torch.nn.Sequential(
            torch.nn.Linear(in_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, out_size)
        )

        gain = torch.nn.init.calculate_gain("relu")

        def param_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain)
                torch.nn.init.zeros_(module.bias)
        self.apply(param_init)

    def forward(self, state):
        logits = self.policynet(state)
        return logits

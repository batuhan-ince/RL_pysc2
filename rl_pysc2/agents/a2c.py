import numpy as np
import torch
from torch import nn

from rl_pysc2.networks.deepmind_model import Encode, Output


class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()

"""
Deepmind FullyConv Agent Implementation from SC2LE paper.(https://deepmind.com/documents/110/sc2le.pdf)
"""


import torch
import torch.nn as nn
import numpy as np
from pysc2.lib import features
import torch.nn.functional as F


class EncodeSpatial(nn.Module):
    """...
        Args:
            n_features: List of #channel

    """
    def __init__(self, n_features):
        super().__init__()
        self.preconv_layers = [nn.Conv2d(n_f, 1, kernel_size=1, stride=1, padding=0) for n_f in n_features]

        self.conv1 = nn.Conv2d(len(n_features),16,kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        """

        :param input: List of 3d tensors of feature
        :return:
        """
        feature_maps = [F.relu(self.preconv_layers[i](x)) for i, x in enumerate(input)]
        x = torch.cat(feature_maps, dim=1)

        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        return x


class Encode(nn.Module):
    """
    ...
    Args:
        n_screen_features: List of feature sizes of screen
        n_minimap_features: List of feature sizes of minimap
        n_nonspatial_features: Non-spatial feature size
    """
    def __init__(self, n_screen_features, n_minimap_features, n_nonspatial_features):
        super().__init__()
        self.screen = EncodeSpatial(n_screen_features)
        self.minimap = EncodeSpatial(n_minimap_features)
        self.non_spatial_features = nn.Linear(n_nonspatial_features, 64)

    def forward(self, input):
        screen_map = self.screen(input[0])
        mini_map = self.minimap(input[1])
        assert screen_map.shape[-2:] == mini_map.shape[-2:], ("Spatial size of screen must"
                                                              " be the same with mini-map's")

        non_spatial = torch.tanh(self.non_spatial_features(input[2]))
        non_spatial = non_spatial.unsqueeze(-1).unsqueeze(-1)
        non_spatial = non_spatial.repeat(1, 1, *screen_map.shape[-2:])

        return torch.cat([screen_map, mini_map, non_spatial], dim=1)

    def obs_to_torch(self,input, device="cpu"):
        spatials = [[torch.from_numpy(np.expand_dims(feature_map, 0)).float().to(device) for feature_map in obs] for obs in input[:2]]
        return (*spatials, torch.from_numpy(np.expand_dims(input[2], 0)).float().to(device))

class Output(nn.Module):
    def __init__(self, input_channels, spatial_y, spatial_x, n_actions):
        super().__init__()
        self.spatial_conv = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.non_spatial_fc = nn.Linear(input_channels * spatial_y * spatial_x, 256)
        self.value = nn.Linear(256, 1)
        self.policy = nn.Linear(256, n_actions)

    def forward(self, input):
        spatial = self.spatial_conv(input)
        spatial = spatial.view(*spatial.shape[:2], spatial.shape[-1] * spatial.shape[-2])
        spatial = F.softmax(spatial, dim=-1)

        input = input.view(input.shape[0], -1)
        non_spatial = self.non_spatial_fc(input)
        value = self.value(non_spatial)
        policy = self.policy(non_spatial)

        return spatial, value, policy




""" Deepmind FullyConv Agent Implementation from SC2LE paper.
"""
import torch
import torch.nn as nn
import numpy as np
from pysc2.lib import features
import torch.nn.functional as F


class EncodeSpatial(nn.Module):
    """ Screen and minimap encoder
    """

    def __init__(self, n_features):
        super().__init__()
        self.preconv_layers = [
            nn.Conv2d(n_f, 1, kernel_size=1, stride=1, padding=0)
            for n_f in n_features]

        self.conv1 = nn.Conv2d(len(n_features), 16,
                               kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        feature_maps = [F.relu(self.preconv_layers[i](x))
                        for i, x in enumerate(input)]
        x = torch.cat(feature_maps, dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class Encode(nn.Module):
    """ Spatial and non spatial input encoder to be used in DeepMindNet.
    Input is expected to be list of screen, minimap and non-spatial
    observation where the both screen and minimap is list of 3D tensors.
    """

    def __init__(self, n_screen_features, n_minimap_features,
                 n_nonspatial_features):
        super().__init__()
        self.screen = EncodeSpatial(n_screen_features)
        self.minimap = EncodeSpatial(n_minimap_features)
        self.non_spatial_features = nn.Linear(n_nonspatial_features, 64)

    def forward(self, input):
        screen_map = self.screen(input[0])
        mini_map = self.minimap(input[1])
        assert screen_map.shape[-2:] == mini_map.shape[-2:], (
            "Spatial size of screen must"
            " be the same with mini-map's")

        non_spatial = torch.tanh(self.non_spatial_features(input[2]))
        non_spatial = non_spatial.unsqueeze(-1).unsqueeze(-1)
        non_spatial = non_spatial.repeat(1, 1, *screen_map.shape[-2:])

        return torch.cat([screen_map, mini_map, non_spatial], dim=1)

    def obs_to_torch(self, input, device="cpu"):
        spatials = [[torch.from_numpy(
            np.expand_dims(feature_map, 0)).float().to(device)
            for feature_map in obs] for obs in input[:2]]
        return (*spatials, torch.from_numpy(
            np.expand_dims(input[2], 0)).float().to(device))


class Output(nn.Module):
    """ Heads of the DeepMindNet including spatial action, non-spatial action
    and value.
    """

    def __init__(self, input_channels, spatial_y, spatial_x, n_actions):
        super().__init__()
        self.spatial_conv = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.non_spatial_fc = nn.Linear(
            input_channels * spatial_y * spatial_x, 256)
        self.value = nn.Linear(256, 1)
        self.policy = nn.Linear(256, n_actions)

    def forward(self, input):
        spatial = self.spatial_conv(input)
        spatial = spatial.view(
            *spatial.shape[:2], spatial.shape[-1] * spatial.shape[-2])
        spatial = F.softmax(spatial, dim=-1)

        input = input.view(input.shape[0], -1)
        non_spatial = self.non_spatial_fc(input)
        value = self.value(non_spatial)
        policy = self.policy(non_spatial)

        return spatial, value, policy


class DeepMindNet(torch.nn.Module):
    """ Network architecture proposed by DeepMind's SC2 paper
    (https://deepmind.com/documents/110/sc2le.pdf).
    Arguments:
        - n_screen_features: Number of screen features. Length of the first
            list in the input list.
        - n_minimap_features: Number of minimap features. Length of the second
            list in the input list.
        - n_nonspatial_features: NUmber of non-spatial features.
        - screen_size: Screen size of the SC2 environment.
        - n_actions: Number of actions to be returned.
    Model Input:
        - List
            - list [Tensor3D, Tensor3D, ...] Screen
            - list [Tensor3D, Tensor3D, ...] Minimap
            - Tensor2D Non-spatial
    Output:
        (spatial action, value ,non-spatial action)
    """

    def __init__(self, n_screen_features, n_minimap_features,
                 n_nonspatial_features, screen_size, n_actions):
        super().__init__()
        self.encoder = Encode(n_screen_features, n_minimap_features,
                              n_nonspatial_features)
        self.output = Output(128, screen_size, screen_size, n_actions)

        self.obs_to_torch = self.encoder.obs_to_torch

    def forward(self, input):
        embeded_feature_map = self.encoder(input)
        return self.output(embeded_feature_map)


class ScreenNet(torch.nn.Module):
    """ Model for movement based mini games in sc2.
    This network only takes screen input and only returns spatial outputs.
    Some of the example min games are MoveToBeacon and CollectMineralShards.
    Arguments:
        - in_channel: Number of feature layers in the screen input
        - screen_size: Screen size of the mini game. If 64 is given output
            size will be 64*64
    Note that output size depends on screen_size.
    """
    class ResidualConv(torch.nn.Module):
        def __init__(self, in_channel, **kwargs):
            super().__init__()
            assert kwargs["out_channels"] == in_channel, ("input channel must"
                                                          "be the same as out"
                                                          " channels")
            self.block = torch.nn.Sequential(
                torch.nn.Conv2d(**kwargs),
                torch.nn.InstanceNorm2d(in_channel),
                torch.nn.ReLU(),
                torch.nn.Conv2d(**kwargs),
                torch.nn.InstanceNorm2d(in_channel),
                torch.nn.ReLU(),
            )

        def forward(self, x):
            res_x = self.block(x)
            return res_x + x

    def __init__(self, in_channel, screen_size):
        super().__init__()
        res_kwargs = {
            "in_channels": 32,
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        }
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, 32, 3, 1, padding=1),
            self.ResidualConv(32, **res_kwargs),
            self.ResidualConv(32, **res_kwargs),
            self.ResidualConv(32, **res_kwargs),
            self.ResidualConv(32, **res_kwargs)
        )

        self.policy = torch.nn.Sequential(
            torch.nn.Linear(32*screen_size*screen_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2*screen_size)
        )

        self.value = torch.nn.Sequential(
            torch.nn.Linear(32*screen_size*screen_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

        self.screen_size = screen_size
        gain = torch.nn.init.calculate_gain("relu")

        def param_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain)
                torch.nn.init.zeros_(module.bias)
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.dirac_(module.weight)
                torch.nn.init.zeros_(module.bias)
        self.apply(param_init)

    def forward(self, state):
        encode = self.convnet(state)
        encode = encode.reshape(encode.shape[0], -1)

        value = self.value(encode)

        logits = self.policy(encode)

        return logits.split(self.screen_size, dim=-1), value

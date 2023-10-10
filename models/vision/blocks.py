import copy
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .operation import Conv1dNormActivation, SeparableConv1d, SqueezeExcitation


@dataclass
class BlockConfig:
    conv_op: nn.Module
    repeats: int
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    skip_op: str
    se_ratio: float


@dataclass
class BottleneckConfig:
    conv_op: nn.Module
    repeats: int
    stride: int
    groups: int
    base_width: int
    dilation: int
    input_channels: int
    out_channels: int
    skip_op: str
    se_ratio: float


class ConvBlock(nn.Module):
    def __init__(
        self,
        cnf: BlockConfig,
        norm_layer: Callable[..., nn.Module] = None,
        se_layer: Callable[..., nn.Module] = SqueezeExcitation
    ) -> None:
        super().__init__()
        
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = False
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        
        layers: List[nn.Module] = []
        activation_layer = nn.ReLU
        
        # Conv Layers
        for i in range(cnf.repeats):
            if i == 0:
                layers.append(
                Conv1dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )
            else:
                layers.append(
                    Conv1dNormActivation(
                        cnf.out_channels,
                        cnf.out_channels,
                        kernel_size=cnf.kernel,
                        stride=1,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer,
                    )
            )
        # squeeze and excitation
        squeeze_channels = max(1, int(cnf.out_channels * cnf.se_ratio))
        if 0 < cnf.se_ratio <= 1:
            layers.append(se_layer(cnf.out_channels, squeeze_channels)) 
        # skip op
        if cnf.skip_op == 'pool':
            # same padding is not implemented in torch
            layers.append(
                    nn.MaxPool1d(
                        kernel_size=3, 
                        stride=1, 
                        padding=1
                    )
                )
        elif cnf.skip_op == 'identity':
            self.use_res_connect = True
            self.identity = None
            if cnf.stride == 1 and cnf.input_channels != cnf.out_channels:
                self.identity = Conv1dNormActivation(
                                    cnf.input_channels,
                                    cnf.out_channels,
                                    kernel_size=cnf.kernel,
                                    stride=cnf.stride,
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer,
                                )
        
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        # 1. skip_op is none or pool
        result = self.block(input)
        # 2. same channels or diff channels
        if self.use_res_connect:
            if self.identity is None:
                result += input
            else:
                result += self.identity(input)
        return result


class SeparableConvBlock(nn.Module):
    def __init__(
        self,
        cnf: BlockConfig,
        norm_layer: Callable[..., nn.Module] = None,
        se_layer: Callable[..., nn.Module] = SqueezeExcitation
    ) -> None:
        super().__init__()
        
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = False
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        
        layers: List[nn.Module] = []
        activation_layer = nn.ReLU
        
        # Conv Layers
        for i in range(cnf.repeats):
            if i == 0:
                layers.append(
                SeparableConv1d(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    expand_ratio=1.0,
                    norm_layer=norm_layer,
                )
            )
            else:
                layers.append(
                    SeparableConv1d(
                        cnf.out_channels,
                        cnf.out_channels,
                        kernel_size=cnf.kernel,
                        stride=1,
                        expand_ratio=1.0,
                        norm_layer=norm_layer,
                    )
            )
        # squeeze and excitation
        squeeze_channels = max(1, int(cnf.out_channels * cnf.se_ratio))
        if 0 < cnf.se_ratio <= 1:
            layers.append(se_layer(cnf.out_channels, squeeze_channels)) 
        # skip op
        if cnf.skip_op == 'pool':
            # self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1) 
            # same padding is not implemented in torch
            layers.append(
                    nn.MaxPool1d(
                        kernel_size=3, 
                        stride=1, 
                        padding=1
                    )
                )
        elif cnf.skip_op == 'identity':
            self.use_res_connect = True
            self.identity = None
            if cnf.stride == 1 and cnf.input_channels != cnf.out_channels:
                self.identity = \
                    Conv1dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    )
        
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        # 1. skip_op is none or pool
        result = self.block(input)
        # 2. same channels or diff channels
        if self.use_res_connect:
            if self.identity is None:
                result += input
            else:
                result += self.identity(input)
        return result


class MBConvBlock(nn.Module):
    def __init__(
        self,
        cnf: BlockConfig,
    ) -> None:
        super().__init__()
        
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = False
        
        layers: List[nn.Module] = []
        
        # MBConv Layers
        for i in range(cnf.repeats):
            if i == 0:
                layers.append(
                MBConv(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    expand_ratio=1.0,
                    se_ratio=cnf.se_ratio,
                    drop_rate=0.2,
                    id_skip = True if cnf.skip_op == 'identity' else False
                )
            )
            else:
                layers.append(
                    MBConv(
                        cnf.out_channels,
                        cnf.out_channels,
                        kernel_size=cnf.kernel,
                        stride=1,
                        expand_ratio=1.0,
                        se_ratio=cnf.se_ratio,
                        drop_rate=0.2,
                        id_skip = True if cnf.skip_op == 'identity' else False
                    )
            )
        
        # skip op is pooling
        if cnf.skip_op == 'pool':
            layers.append(
                    nn.MaxPool1d(
                        kernel_size=3, 
                        stride=1, 
                        padding=1
                    )
                )
            
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        return result


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_ratio,
        se_ratio,
        drop_rate,
        id_skip,
        norm_layer: Callable[..., nn.Module] = None,
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.id_skip = id_skip
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        
        self.use_res_connect = False

        layers: List[nn.Module] = []
        activation_layer = nn.ReLU

        # expand
        expanded_channels = int(in_channels * expand_ratio)
        if expand_ratio != 1.0:
            layers.append(

                Conv1dNormActivation(
                    in_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv1dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, int(out_channels * se_ratio))
        if 0 < se_ratio <= 1:
            layers.append(se_layer(expanded_channels, squeeze_channels)) 
        
        # project
        layers.append(
            Conv1dNormActivation(
                expanded_channels, out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )
        
        # dropout
        if self.id_skip and stride == 1 and in_channels == out_channels:
            self.use_res_connect = True
            if drop_rate > 0:
                layers.append(nn.Dropout(drop_rate))

        self.block = nn.Sequential(*layers)
        self.out_channels = out_channels
        
    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result

# Shuffle Block
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, length = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, length)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, length)

    return x


class ShuffleBlock(nn.Module):
    def __init__(
        self,
        cnf: BlockConfig,
    ) -> None:
        super().__init__()
        
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = False
        
        layers: List[nn.Module] = []
        
        # ShuffleConv Layers
        for i in range(cnf.repeats):
            if i == 0:
                layers.append(
                    ShuffleConv(
                        cnf.input_channels,
                        cnf.out_channels,
                        stride=cnf.stride,
                        expand_ratio=1.0,
                        drop_rate=0.2,
                    )
                )
            else:
                layers.append(
                    ShuffleConv(
                        cnf.out_channels,
                        cnf.out_channels,
                        stride=1,
                        expand_ratio=1.0,
                        drop_rate=0.2,
                    )
            )
        
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        return result


class ShuffleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        expand_ratio = 1.0,
        drop_rate = 0.0,
        id_skip = 'identity',
        norm_layer: Callable[..., nn.Module] = None,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.id_skip = id_skip
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        
        self.downsample = None
        self.dropout = None

        branch1: List[nn.Module] = []
        branch2: List[nn.Module] = []
        activation_layer = nn.ReLU
        self.act_layer = activation_layer(inplace=True)

        branch_features = out_channels // 2
        if (self.stride == 1) and (in_channels != branch_features << 1):
            raise ValueError(
                f"Invalid combination of stride {stride}, inp {in_channels} and oup {out_channels} values. If stride == 1 then inp should be equal to oup // 2 << 1."
            )

        if self.stride > 1:
            branch1.extend((
                # depthwise
                Conv1dNormActivation(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=stride,
                    groups=in_channels,
                    norm_layer=norm_layer,
                    activation_layer=None,
                ),
                # conv1x1
                Conv1dNormActivation(
                    in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            ))
            
        branch2.extend((
            # conv1x1
            Conv1dNormActivation(
                in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            ),
            # depthwise
            Conv1dNormActivation(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=stride,
                groups=branch_features,
                norm_layer=norm_layer,
                activation_layer=None,
            ),
            # conv1x1
            Conv1dNormActivation(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        ))            
        # dropout
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)

        self.block1 = nn.Sequential(*branch1)
        self.block2 = nn.Sequential(*branch2)
        self.out_channels = branch_features
        
    def forward(self, input: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = input.chunk(2, dim=1)
            result = torch.cat((x1, self.block2(x2)), dim=1)
        else:
            result = torch.cat((self.block1(input), self.block2(input)), dim=1)
            
        result = channel_shuffle(result, 2)
        
        if self.dropout:
            result = self.dropout(result)
        
        return result


# ResNet Block
class ResConvBlock(nn.Module):
    def __init__(
        self,
        cnf: BlockConfig,
    ) -> None:
        super().__init__()
        
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = False
        
        layers: List[nn.Module] = []
        
        # ResConv Layers
        for i in range(cnf.repeats):
            if i == 0:
                layers.append(
                    ResConv(
                        cnf.input_channels,
                        cnf.out_channels,
                        stride=cnf.stride,
                        expand_ratio=1.0,
                        drop_rate=0.2,
                    )
                )
            else:
                layers.append(
                    ResConv(
                        cnf.out_channels,
                        cnf.out_channels,
                        stride=1,
                        expand_ratio=1.0,
                        drop_rate=0.2,
                    )
            )
        
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        return result


class ResConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        expand_ratio = 1.0,
        drop_rate = 0.0,
        id_skip = 'identity',
        norm_layer: Callable[..., nn.Module] = None,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.id_skip = id_skip
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        
        self.downsample = None

        layers: List[nn.Module] = []
        activation_layer = nn.ReLU
        self.act_layer = activation_layer(inplace=True)
        
        # conv3x1 w/ activation
        layers.append(
            Conv1dNormActivation(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                )
            )
        # conv3x1 w/o activation
        layers.append(
            Conv1dNormActivation(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=None,
                )
            )
        
        if in_channels != out_channels:
            self.downsample = Conv1dNormActivation(
                                in_channels, 
                                out_channels, 
                                kernel_size=1, 
                                stride=stride,
                                norm_layer=norm_layer, 
                                activation_layer=None
                            )
        # dropout
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))

        self.block = nn.Sequential(*layers)
        self.out_channels = out_channels
        
    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.downsample is not None:
            input = self.downsample(input)
            
        result += input
        result = self.act_layer(result)
        return result


# TODO: not implemented
# class BottleneckBlock(nn.Module):
#     def __init__(
#         self,
#         cnf: BottleneckConfig,  # BlockConfig
#     ) -> None:
#         super().__init__()
        
#         if not (1 <= cnf.stride <= 2):
#             raise ValueError("illegal stride value")

#         self.use_res_connect = False
        
#         layers: List[nn.Module] = []
        
#         # Bottleneck Layers
#         for i in range(cnf.repeats):
#             if i == 0:
#                 layers.append(
#                     Bottleneck(
#                         cnf.input_channels,
#                         cnf.out_channels,
#                         stride=cnf.stride,
#                         groups=cnf.groups, # 32
#                         base_width=cnf.base_width, # 4
#                         dilation=cnf.dilation,
#                         expand_ratio=4.0,
#                         drop_rate=0.2,
#                     )
#                 )
#                 out_channels = int(cnf.out_channels * 4.0)
#             else:
#                 layers.append(
#                     Bottleneck(
#                         out_channels,
#                         out_channels,
#                         stride=1,
#                         groups=cnf.groups, # 32
#                         base_width=cnf.base_width, # 4
#                         dilation=cnf.dilation,
#                         expand_ratio=4.0,
#                         drop_rate=0.2,
#                     )
#                 )
#                 out_channels = int(out_channels * 4.0)
        
#         self.block = nn.Sequential(*layers)
#         self.out_channels = out_channels

#     def forward(self, input: Tensor) -> Tensor:
#         result = self.block(input)
#         return result

# TODO: not implemented
# class Bottleneck(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         stride,
#         groups, # 32
#         base_width, # 4
#         dilation,
#         expand_ratio = 4.0,
#         drop_rate = 0.0,
#         id_skip = 'identity',
#         norm_layer: Callable[..., nn.Module] = None,
#     ) -> None:
#         super().__init__()
#         self.stride = stride
#         self.id_skip = id_skip
#         if stride not in [1, 2]:
#             raise ValueError(f"stride should be 1 or 2 insted of {stride}")
        
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm1d
#         width = int(in_channels * (base_width / 64.0)) * groups
        
#         self.downsample = None

#         layers: List[nn.Module] = []
#         activation_layer = nn.ReLU
#         self.act_layer = activation_layer(inplace=True)
        
#         # conv1x1 w/ activation
#         layers.append(
#             Conv1dNormActivation(
#                 in_channels,
#                 width,
#                 kernel_size=1,
#                 stride=1,
#                 norm_layer=norm_layer,
#                 activation_layer=activation_layer,
#                 )
#             )
#         # conv3x1 w/ activation
#         layers.append(
#             Conv1dNormActivation(
#                 width,
#                 width,
#                 kernel_size=3,
#                 stride=stride,
#                 groups=groups,
#                 # dilation=dilation,
#                 norm_layer=norm_layer,
#                 activation_layer=activation_layer,
#                 )
#             )
#         # conv1x1 w/o activation
#         expanded_channels = int(out_channels * expand_ratio)
#         layers.append(
#             Conv1dNormActivation(
#                 width,
#                 expanded_channels,
#                 kernel_size=1,
#                 stride=1,
#                 norm_layer=norm_layer,
#                 activation_layer=None,
#                 )
#             )
        
#         if in_channels != out_channels:
#             self.downsample = Conv1dNormActivation(
#                                 in_channels, 
#                                 expanded_channels, 
#                                 kernel_size=1, 
#                                 stride=stride,
#                                 norm_layer=norm_layer, 
#                                 activation_layer=None
#                             )
#         # dropout
#         if drop_rate > 0:
#             layers.append(nn.Dropout(drop_rate))

#         self.block = nn.Sequential(*layers)
#         self.out_channels = expanded_channels
        
#     def forward(self, input: Tensor) -> Tensor:
#         result = self.block(input)
#         if self.downsample is not None:
#             input = self.downsample(input)
            
#         result += input
#         result = self.act_layer(result)
#         return result
    
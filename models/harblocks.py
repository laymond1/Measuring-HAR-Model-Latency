import math
from sys import modules
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor

from .vision.operation import Conv1dNormActivation, SqueezeExcitation


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
class GTSResBlockConfig:
    conv_op: nn.Module
    repeats: int
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    n_groups: int
    first_grouped_conv: bool
    pool: bool
    skip_op: str
    se_ratio: float


# LSTM
class LSTMBlock(nn.Module):
    def __init__(
        self,
        cnf: BlockConfig,
    ) -> None:
        super().__init__()
        
        # LSTM Layers
        self.block = nn.LSTM(
                        input_size=cnf.input_channels, 
                        hidden_size=cnf.out_channels, 
                        num_layers=cnf.repeats, 
                        batch_first=True
                    )
        self.out_channels = cnf.out_channels

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x,(h, c) = self.block(x)
        h = h[-1,:,:]
        return h

# BiLSTM
class BiLSTMBlock(nn.Module):
    def __init__(
        self,
        cnf: BlockConfig,
    ) -> None:
        super().__init__()
        
        # LSTM Layers
        self.block = nn.LSTM(
                        input_size=cnf.input_channels, 
                        hidden_size=cnf.out_channels, 
                        num_layers=cnf.repeats, 
                        batch_first=True,
                        bidirectional=True
                    )
        self.out_channels = cnf.out_channels*2

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x,(h, c) = self.block(x)
        h = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
        return h

# GTSNet
class GTSResConvBlock(nn.Module):
    def __init__(
        self,
        cnf: GTSResBlockConfig,
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
                    GTSResConv(
                        cnf.input_channels,
                        cnf.out_channels,
                        n_groups=cnf.n_groups,
                        first_grouped_conv=cnf.first_grouped_conv,
                        pool=cnf.pool
                    )
                )
            else:
                layers.append(
                    GTSResConv(
                        cnf.out_channels,
                        cnf.out_channels,
                        n_groups=cnf.n_groups,
                        first_grouped_conv=cnf.first_grouped_conv,
                        pool=cnf.pool
                    )
                )
        
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        return result


class GTSResConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_groups,
        first_grouped_conv=True,
        pool = False,
        id_skip = 'identity',
        norm_layer: Callable[..., nn.Module] = None,
    ) -> None:
        super().__init__()
        self.id_skip = id_skip
        self.pool = pool
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        
        layers: List[nn.Module] = []
        activation_layer = nn.ReLU
        self.act_layer = activation_layer(inplace=True)
        
        # conv1
        layers.extend((
            GTSConvUnit(
                in_channels,
                out_channels,
                n_groups,
                first_grouped_conv
            ),
            activation_layer(inplace=True)
        ))
        # conv2
        layers.extend((
            GTSConvUnit(
                out_channels,
                out_channels,
                n_groups
            ),
            activation_layer(inplace=True)
        ))
        # conv3
        layers.append(
            GTSConvUnit(
                out_channels,
                out_channels,
                n_groups
            )
        )
        if in_channels != out_channels:
            self.shortcut = Conv1dNormActivation(
                                in_channels, 
                                out_channels, 
                                kernel_size=1, 
                                stride=1,
                                norm_layer=norm_layer, 
                                activation_layer=None
                            )
        else:
            self.shortcut = norm_layer(out_channels)
        
        self.block = nn.Sequential(*layers)
        self.out_channels = out_channels
        
    def forward(self, input: Tensor) -> Tensor:
        _, _, t = input.size()
        if self.pool:
            input = F.adaptive_max_pool1d(input, (t//2,)) # not supported in jit.script
        result = self.block(input)
        shortcut_y = self.shortcut(input)
        result += shortcut_y
        result = self.act_layer(result)
        return result
        

def channel_shuffle(x, groups):
    batchsize, num_channels, T = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, T)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, T)

    return x

class InplaceShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_groups):
        ctx.groups_ = n_groups
        n, c, t = input.size()
        slide = c // n_groups
        left_idx = torch.tensor([i*slide for i in range(n_groups)])
        right_idx = torch.tensor([i*slide+1 for i in range(n_groups)])

        buffer = input.data.new(n, n_groups, t).zero_()
        buffer[:, :, :-1] = input.data[:, left_idx, 1:] 
        input.data[:, left_idx] = buffer
        buffer.zero_()
        buffer[:, :, 1:] = input.data[:, right_idx, :-1]
        input.data[:, right_idx] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        n_groups = ctx.groups_
        n, c, t = grad_output.size()
        slide = c // n_groups
        left_idx = torch.tensor([i*slide for i in range(n_groups)])
        right_idx = torch.tensor([i*slide+1 for i in range(n_groups)])

        buffer = grad_output.data.new(n, left_idx,t).zero_()
        buffer[:, :, 1:] = grad_output.data[:, left_idx, :-1] # reverse
        grad_output.data[:, left_idx] = buffer
        buffer.zero_()
        buffer[:, :, :-1] = grad_output.data[:, right_idx, 1:]
        grad_output.data[:, right_idx] = buffer
        return grad_output, None

class GTSConv(nn.Module):
    def __init__(self, i_nc, n_groups):
        super(GTSConv, self).__init__()
        self.groups = n_groups
        self.conv = nn.Conv1d(i_nc, i_nc, kernel_size=1, padding=0, bias=False, groups=n_groups)
        self.bn = nn.BatchNorm1d(i_nc)
    def forward(self, x):
        out = InplaceShift.apply(x, self.groups)
        out = self.conv(x)
        out = self.bn(out)
        return out
    
class GTSConvUnit(nn.Module):
    '''
    Grouped Temporal Shift (GTS) module
    '''
    def __init__(self, i_nc, n_fs, n_groups, first_grouped_conv=True):
        super(GTSConvUnit, self).__init__()

        self.groups = n_groups
        self.grouped_conv = n_groups if first_grouped_conv else 1

        self.perm = nn.Sequential(
            nn.Conv1d(i_nc, n_fs, kernel_size=1, groups=self.grouped_conv, stride=1, bias=False),
            nn.BatchNorm1d(n_fs),
        )

        self.GTSConv = GTSConv(n_fs, n_groups)
        
    def forward(self, x):
        out = F.relu(self.perm(x))
        out = self.GTSConv(out)
        out = channel_shuffle(out, self.groups)
        return out

class GTSResBlock(nn.Module):
    def __init__(self, i_nc, n_fs, n_groups, first_grouped_conv=True, pool=False):
        super(GTSResBlock, self).__init__()

        self.i_nc = i_nc
        self.n_fs = n_fs
        self.pool = pool

        self.conv1 = GTSConvUnit(self.i_nc, self.n_fs, n_groups, first_grouped_conv)
        self.conv2 = GTSConvUnit(self.n_fs, self.n_fs, n_groups)
        self.conv3 = GTSConvUnit(self.n_fs, self.n_fs, n_groups)

        if i_nc == n_fs:
            self.shortcut = nn.BatchNorm1d(n_fs)
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(i_nc, n_fs, kernel_size=1, padding=0, stride=1, bias=False),
                nn.BatchNorm1d(n_fs),
            )

    def forward(self, x):
        _,_,t = x.size()
        if self.pool:
            x = F.adaptive_max_pool1d(x,t//2)
        residual = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        shortcut_y = self.shortcut(residual)
        out = out + shortcut_y
        return F.relu(out)
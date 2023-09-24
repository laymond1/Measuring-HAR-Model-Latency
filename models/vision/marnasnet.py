# MarNASNet-A model architecture from the `MarNASNets: Towards CNN model architectures specific to sensor-based human activity recognition <https://ieeexplore.ieee.org/document/10179199>`_ paper. 
# The original code for this paper can be found here: <https://github.com/Shakshi3104/tfmars>.
# This code has been re-implemented by Wonseon Lim using PyTorch.

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



__all__ = [
    "MarNASNet",
    "marnasnet_a",
    "marnasnet_b",
    "marnasnet_c",
    "marnasnet_d",
    "marnasnet_e"
]


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
            if input.size(1) == result.size(1):
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
            if input.size(1) == result.size(1):
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


class MarNASNet(nn.Module):
    def __init__(
        self,
        init_channels: int,
        blocks_setting: Sequence[BlockConfig],
        dropout: float = 0.2,
        num_classes: int = 6,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        
        """MarNASNet A to E main class
        
        Args:
            init_channels (int): The number of the input channels
            blocks_setting: Network structure
            dropout (float): The droupout probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        
        if not blocks_setting:
            raise ValueError("The blocks_setting should not be empty")
        elif not (
            isinstance(blocks_setting, Sequence)
            and all([isinstance(s, BlockConfig) for s in blocks_setting])
        ):
            raise TypeError("The blocks_setting should be List[BlockConfig]")
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
            
        layers: List[nn.Module] = []
        
        # building first layer
        firstconv_output_channels = blocks_setting[0].input_channels
        # fixed architecture in MarNASNets across all models.
        layers.append(
            Conv1dNormActivation(
                init_channels, 
                firstconv_output_channels, 
                kernel_size=3, 
                stride=1, 
                norm_layer=norm_layer, 
                activation_layer=nn.ReLU
            )
        )
        
        total_stage_blocks = sum(cnf.repeats for cnf in blocks_setting)
        for i, cnf in enumerate(blocks_setting):
            # copy to avoid modifications. shallow copy is enough
            block_cnf = copy.copy(cnf)
            layers.append(block_cnf.conv_op(cnf))
                
        # building last several layers
        lastconv_input_channels = blocks_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv1dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU,
            )
        )
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, num_classes),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)
                
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _marnasnet(
    init_channels: int,
    blocks_setting: Sequence[BlockConfig],
    dropout: float,
    last_channel: Optional[int],
    **kwargs: Any,
) -> MarNASNet:

    model = MarNASNet(init_channels, blocks_setting, dropout, last_channel=last_channel, **kwargs)

    return model


def _marnasnet_conf(
    arch: str,
    **kwargs: Any,
    ) -> Tuple[Sequence[BlockConfig], Optional[int]]:
    blocks_setting: Sequence[BlockConfig]
    # conv_op / repeats / kernel / stride / input_channels / out_channels / skip_op
    if arch.startswith("marnasnet_a"):
        block_conf = partial(BlockConfig, se_ratio=0.25)
        blocks_setting = [
            block_conf(MBConvBlock, 5, 5, 1, 32, 16, 'identity'),
            block_conf(ConvBlock,   2, 2, 1, 16, 32, 'none'),
            block_conf(ConvBlock,   2, 5, 1, 32, 64, 'identity'),
            block_conf(MBConvBlock, 5, 5, 2, 64, 128, 'identity'),
        ]
        last_channel = 1280
    elif arch.startswith("marnasnet_b"):
        block_conf = partial(BlockConfig, se_ratio=0.25)
        blocks_setting = [
            block_conf(MBConvBlock, 2, 3, 1, 32, 32, 'none'),
            block_conf(MBConvBlock, 5, 5, 2, 32, 32, 'identity'),
            block_conf(MBConvBlock, 2, 5, 2, 32, 32, 'identity'),
            block_conf(MBConvBlock, 5, 5, 2, 32, 32, 'identity'),
        ]
        last_channel = 1280
    elif arch.startswith("marnasnet_c"):
        block_conf = partial(BlockConfig, se_ratio=0.25)
        blocks_setting = [
            block_conf(MBConvBlock, 2, 5, 1, 32, 32, 'identity'),
            block_conf(ConvBlock,   4, 3, 1, 32, 32, 'identity'), # same padding is not implemented, k_size 2 -> 3
            block_conf(MBConvBlock, 2, 2, 2, 32, 192, 'none'),
            block_conf(MBConvBlock, 5, 5, 2, 192, 192, 'identity'),
        ]
        last_channel = 1280
    elif arch.startswith("marnasnet_d"):
        block_conf = partial(BlockConfig, se_ratio=0.25)
        blocks_setting = [
            block_conf(MBConvBlock, 5, 3, 1, 32, int(0.75 * 76), 'identity'),
            block_conf(MBConvBlock, 5, 5, 2, int(0.75 * 76), int(1.25 * 88), 'identity'),
            block_conf(MBConvBlock, 2, 2, 2, int(1.25 * 88), int(0.75 * 100), 'none'),
            block_conf(MBConvBlock, 2, 3, 2, int(0.75 * 100), int(0.75 * 112), 'identity'), # same padding is not implemented, k_size 2 -> 3
        ]
        last_channel = 1280
    elif arch.startswith("marnasnet_e"):
        block_conf = partial(BlockConfig, se_ratio=0.25)
        blocks_setting = [
            block_conf(MBConvBlock, 2, 3, 1, 32, 32, 'identity'), # same padding is not implemented, k_size 2 -> 3
            block_conf(MBConvBlock, 2, 5, 2, 32, 32, 'identity'), # same padding is not implemented, k_size 4 -> 5
            block_conf(ConvBlock,   5, 5, 1, 32, 32, 'identity'),
            block_conf(ConvBlock,   4, 3, 1, 32, 320, 'identity'), # same padding is not implemented, k_size 2 -> 3
            block_conf(MBConvBlock, 3, 5, 2, 320, 320, 'identity'),
        ]
        last_channel = 1280
    

    return blocks_setting, last_channel
                

def marnasnet_a(
    *, init_channels: int, **kwargs: Any
) -> MarNASNet:
    """ MarNASNet-A
    Args:
    
    """
    inverted_residual_setting, last_channel = _marnasnet_conf("marnasnet_a")
    return _marnasnet(init_channels, inverted_residual_setting, 0.2, last_channel, **kwargs)
    
    
def marnasnet_b(
    *, init_channels: int, **kwargs: Any
) -> MarNASNet:
    """ MarNASNet-B
    Args:
    
    """
    inverted_residual_setting, last_channel = _marnasnet_conf("marnasnet_b")
    return _marnasnet(init_channels, inverted_residual_setting, 0.2, last_channel, **kwargs)


def marnasnet_c(
    *, init_channels: int, **kwargs: Any
) -> MarNASNet:
    """ MarNASNet-C
    Args:
    
    """
    inverted_residual_setting, last_channel = _marnasnet_conf("marnasnet_c")
    return _marnasnet(init_channels, inverted_residual_setting, 0.2, last_channel, **kwargs)


def marnasnet_d(
    *, init_channels: int, **kwargs: Any
) -> MarNASNet:
    """ MarNASNet-D
    Args:
    
    """
    inverted_residual_setting, last_channel = _marnasnet_conf("marnasnet_d")
    return _marnasnet(init_channels, inverted_residual_setting, 0.2, last_channel, **kwargs)


def marnasnet_e(
    *, init_channels: int, **kwargs: Any
) -> MarNASNet:
    """ MarNASNet-E
    Args:
    
    """
    inverted_residual_setting, last_channel = _marnasnet_conf("marnasnet_e")
    return _marnasnet(init_channels, inverted_residual_setting, 0.2, last_channel, **kwargs)


if __name__ == "__main__":
    model = marnasnet_a(init_channels=3)
    input = torch.randn(1, 3, 256)
    output = model(input)
    output.shape
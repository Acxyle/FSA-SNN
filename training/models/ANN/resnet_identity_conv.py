#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 00:12:41 2023

@author: pytorch
@modified: acxyle
    
    - added resnext50_32x8/16/32d for 224^2 (eg. ImageNet)
    - added resnet20/38/56/110/164 for 32^2 (eg. CIFAR)
    
"""

from typing import Any, Callable, List, Optional, Type, Union, Literal

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "ResNet",
    
    "resnet18_identity_conv",
    "resnet34_identity_conv",
    "resnet50_identity_conv",
    "resnet101_identity_conv",
    "resnet152_identity_conv",
    
    "resnext50_32x4d_identity_conv",
    "resnext50_32x8d_identity_conv",
    "resnext50_32x16d_identity_conv",
    "resnext50_32x32d_identity_conv",
    
    "resnet20_identity_conv",
    "resnet38_identity_conv",
    "resnet56_identity_conv",
    "resnet110_identity_conv",
    "resnet164_identity_conv"
]


def conv3x3(in_planes: int, out_planes: int, stride: int=1, groups: int=1, dilation: int=1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv1x1_identity(in_planes: int, out_planes: int) -> nn.Conv2d:
    """non-parametric 1x1 convolution"""
    
    assert in_planes == out_planes
    
    conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
    
    conv1x1_channel = conv1x1.weight.size(0)
    with torch.no_grad():
        conv1x1.weight.copy_(torch.eye(conv1x1_channel).view(conv1x1_channel, conv1x1_channel, 1, 1))
    conv1x1.weight.requires_grad = False
    
    return conv1x1


def conv1x1_identity_kernel_check(weight: torch.Tensor) -> bool:
    
    assert weight.ndim == 4 and weight.shape[0] == weight.shape[1] and weight.shape[2] == 1 and weight.shape[3] == 1
    
    conv1x1_channel = weight.size(0)
    dummy_unit_matrix = torch.eye(conv1x1_channel).view(conv1x1_channel, conv1x1_channel, 1, 1).to('cuda:0')
    
    return torch.allclose(weight, dummy_unit_matrix.to(weight.dtype))



class BasicBlock_conv1x1_identity(nn.Module):
    
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock_conv1x1_identity only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock_conv1x1_identity")
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
        # ---
        if downsample is None and inplanes == planes:
            self.conv1x1_identity = conv1x1_identity(inplanes, planes)

    def forward(self, x: Tensor) -> Tensor:
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = self.conv1x1_identity(x)
            
            #if identity.shape == x.shape and conv1x1_identity_kernel_check(self.conv1x1_identity.weight.detach().clone()):
            #
            #    if self.conv1x1_identity.weight.dtype == torch.float32 and x.dtype == torch.float32:
            #        assert torch.allclose(identity, x, rtol=1e-03, atol=1e-08), 'self.conv1x1_identity() must be an identity conv operation'
            #    elif self.conv1x1_identity.weight.dtype == torch.float64 and x.dtype == torch.float64:
            #        assert torch.allclose(identity, x, rtol=1e-04, atol=1e-08), 'self.conv1x1_identity() must be an identity conv operation'
            #    else:
            #        raise RuntimeError('dtype not supported')
            #
            #else:     # when the conv kernel has been pruned, no longer an identity matrix (square matrix)
            #    
            #    ...
            
        out += identity
        out = self.relu(out)

        return out


class Bottleneck_conv1x1_identity(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        
        outplanes = planes * self.expansion
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # ---
        if downsample is None and inplanes == outplanes:
            self.conv1x1_identity = conv1x1_identity(inplanes, outplanes)

    def forward(self, x: Tensor) -> Tensor:

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = self.conv1x1_identity(x)
            assert torch.all(identity == x), 'self.conv1x1_identity() must be an identity conv operation'

        out += identity
        out = self.relu(out)

        return out


class ResNet_Base(nn.Module):
    
    def __init__(
        self,
        block: Type[Union[BasicBlock_conv1x1_identity, Bottleneck_conv1x1_identity]],
        layers: List[int],
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs
    ) -> None:
        
        super().__init__()
        
        # ---
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        # ---
        self.groups = groups
        self.base_width = width_per_group

    def _make_layer(
        self,
        block: Type[Union[BasicBlock_conv1x1_identity, Bottleneck_conv1x1_identity]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
    def _init_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad == False:
                    continue
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _zero_init_residual(self):
        
        for m in self.modules():
            if isinstance(m, Bottleneck_conv1x1_identity) and m.bn3.weight is not None:
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
            elif isinstance(m, BasicBlock_conv1x1_identity) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


class ResNet(ResNet_Base):
    
    def __init__(
        self,
        block: Type[Union[BasicBlock_conv1x1_identity, Bottleneck_conv1x1_identity]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        **kwargs
    ) -> None:
        
        self.inplanes = 64
        self.dilation = 1
        
        super().__init__(block=block, layers=layers, **kwargs)
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}")
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # ---
        self._init_weights()

        if zero_init_residual:
            self._zero_init_residual()

    def _forward_impl(self, x: Tensor) -> Tensor:
        
       x = self.conv1(x)
       x = self.bn1(x)
       x = self.relu(x)
       x = self.maxpool(x)

       x = self.layer1(x)
       x = self.layer2(x)
       x = self.layer3(x)
       x = self.layer4(x)

       x = self.avgpool(x)
       x = torch.flatten(x, 1)
       x = self.fc(x)

       return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

# ---
def _resnet(
    block: Type[Union[BasicBlock_conv1x1_identity, Bottleneck_conv1x1_identity]],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:
    
    model = ResNet(block, layers, **kwargs)

    return model


# ---
def resnet18_identity_conv(**kwargs: Any) -> ResNet:
    return _resnet(BasicBlock_conv1x1_identity, [2, 2, 2, 2], **kwargs)


def resnet34_identity_conv(**kwargs: Any) -> ResNet:
    return _resnet(BasicBlock_conv1x1_identity, [3, 4, 6, 3], **kwargs)


def resnet50_identity_conv(**kwargs: Any) -> ResNet:
    return _resnet(Bottleneck_conv1x1_identity, [3, 4, 6, 3], **kwargs)


def resnet101_identity_conv(**kwargs: Any) -> ResNet:
    return _resnet(Bottleneck_conv1x1_identity, [3, 4, 23, 3], **kwargs)


def resnet152_identity_conv(**kwargs: Any) -> ResNet:
    return _resnet(Bottleneck_conv1x1_identity, [3, 8, 36, 3], **kwargs)


# ---
def resnext50_32x4d_identity_conv(**kwargs: Any) -> ResNet:
 
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(Bottleneck_conv1x1_identity, [3, 4, 6, 3], **kwargs)


def resnext50_32x8d_identity_conv(**kwargs: Any) -> ResNet:
  
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(Bottleneck_conv1x1_identity, [3, 4, 6, 3], **kwargs)


def resnext50_32x16d_identity_conv(**kwargs: Any) -> ResNet:
  
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    return _resnet(Bottleneck_conv1x1_identity, [3, 4, 6, 3], **kwargs)


def resnext50_32x32d_identity_conv(**kwargs: Any) -> ResNet:
  
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 32
    return _resnet(Bottleneck_conv1x1_identity, [3, 4, 6, 3], **kwargs)


# -----
class ResNet_lite(ResNet_Base):
    
    def __init__(
        self,
        block: Type[Union[BasicBlock_conv1x1_identity, Bottleneck_conv1x1_identity]],
        layers: List[int],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        in_c: Literal[1, 3] = 3,
        **kwargs
    ) -> None:
        
        self.inplanes = 16
        self.dilation = 1
        
        super().__init__(block=block, layers=layers, **kwargs)

        self.conv1 = nn.Conv2d(in_c, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, self.inplanes, layers[0])
        self.layer2 = self._make_layer(block, self.inplanes*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.inplanes*2, layers[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.inplanes, num_classes)

        self._init_weights()

        if zero_init_residual:
            self._zero_init_residual()

    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc1(x)
        
        return y

    def forward(self, x):
        
        return self._forward_impl(x)

# ---
def _resnet_lite(
    block: Type[Union[BasicBlock_conv1x1_identity, Bottleneck_conv1x1_identity]],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:
    
    model = ResNet_lite(block, layers, **kwargs)

    return model


# ---
def resnet20_identity_conv(**kwargs: Any) -> ResNet:
    return _resnet_lite(BasicBlock_conv1x1_identity, [3, 3, 3], **kwargs)


def resnet38_identity_conv(**kwargs: Any) -> ResNet:
    return _resnet_lite(BasicBlock_conv1x1_identity, [6, 6, 6], **kwargs)


def resnet56_identity_conv(**kwargs: Any) -> ResNet:
    return _resnet_lite(BasicBlock_conv1x1_identity, [9, 9, 9], **kwargs)


def resnet110_identity_conv(**kwargs: Any) -> ResNet:
    return _resnet_lite(BasicBlock_conv1x1_identity, [18, 18, 18], **kwargs)


def resnet164_identity_conv(**kwargs: Any) -> ResNet:
    return _resnet_lite(BasicBlock_conv1x1_identity, [27, 27, 27], **kwargs)


if __name__ == '__main__':
    
    model = resnet56_identity_conv()
    x = torch.ones(32,3,32,32)
    y = model(x)
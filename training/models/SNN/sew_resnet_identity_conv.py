#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:25:28 2024

@author: fangwei123456
@modified: acxyle

    TODO:
            1. lite
            2. 

"""


import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import layer

from spikingjelly.activation_based import surrogate, neuron, functional

    
__all__ = ['SEWResNet', 
           
           'sew_resnet18', 
           'sew_resnet34', 
           'sew_resnet50', 
           'sew_resnet101', 
           'sew_resnet152',
           
           'sew_resnext50_32x4d', 
           'sew_resnext50_32x8d', 
           'sew_resnext50_32x16d', 
           'sew_resnext50_32x32d',
           
           'sew_resnext101_32x8d',
           'sew_wide_resnet50_2', 
           'sew_wide_resnet101_2']


def sew_function(x: torch.Tensor, y: torch.Tensor, cnf:str):
    if cnf == 'ADD':
        return x + y
    elif cnf == 'AND':
        return x * y
    elif cnf == 'IAND':
        return x * (1. - y)
    else:
        raise NotImplementedError


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv1x1_identity(in_planes: int, out_planes: int) -> layer.Conv2d:
    """non-parametric 1x1 convolution"""
    
    assert in_planes == out_planes
    
    conv1x1 = layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
    
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


class SEW_BasicBlock_conv1x1_identity(nn.Module):
    
    expansion = 1

    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1, 
                 downsample=None, 
                 groups=1,
                 base_width=64, 
                 dilation=1, 
                 norm_layer=None, 
                 cnf: str=None, 
                 spiking_neuron: callable=None, 
                 **kwargs) -> None:
        
        super().__init__()
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        
        if downsample is not None:
            self.downsample_sn = spiking_neuron(**deepcopy(kwargs))
            
        self.stride = stride
        self.cnf = cnf
        
        # ---
        if downsample is None and inplanes == planes:
            self.conv1x1_identity = conv1x1_identity(inplanes, planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))
        else:
            identity = self.conv1x1_identity(x)     # no 'downsample_sn' applied no matter spiking or sew
            
            #if identity.shape == x.shape and conv1x1_identity_kernel_check(self.conv1x1_identity.weight.detach().clone()):
            #
            #    if self.conv1x1_identity.weight.dtype == torch.float32 and x.dtype == torch.float32:
            #        assert torch.allclose(identity, x, rtol=1e-03, atol=1e-08), 'self.conv1x1_identity() must be an identity conv operation'
            #    elif self.conv1x1_identity.weight.dtype == torch.float64 and x.dtype == torch.float64:
            #        assert torch.allclose(identity, x, rtol=1e-04, atol=1e-08), 'self.conv1x1_identity() must be an identity conv operation'
            #    else:
            #        raise RuntimeError('dtype not supported')
            #
            #else:     # when the conv kernel has been pruned, this is no longer an identity matrix (square matrix)
            #    
            #    ...
    
        out = sew_function(identity, out, self.cnf)

        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


class SEW_Bottleneck_conv1x1_identity(nn.Module):
    
    expansion = 4

    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1, 
                 downsample=None, 
                 groups=1,
                 base_width=64, 
                 dilation=1, 
                 norm_layer=None, 
                 cnf:str=None, 
                 spiking_neuron:callable=None, 
                 **kwargs) -> None:
        
        super().__init__()
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        outplanes = planes * self.expansion
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.conv3 = conv1x1(width, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.sn3 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        
        if downsample is not None:
            self.downsample_sn = spiking_neuron(**deepcopy(kwargs))
            
        self.stride = stride
        self.cnf = cnf
        
        # ---
        if downsample is None and inplanes == outplanes:
            self.conv1x1_identity = conv1x1_identity(inplanes, outplanes)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sn3(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))
        else:
            identity = self.conv1x1_identity(x)
            
            #assert torch.all(identity == x), 'self.conv1x1_identity() must be an identity conv operation'

        out = sew_function(out, identity, self.cnf)

        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


class SEWResNet(nn.Module):
    
    def __init__(self, 
                 block, 
                 layers, 
                 num_classes=1000, 
                 zero_init_residual=False,
                 groups=1, 
                 width_per_group=64, 
                 replace_stride_with_dilation=None,
                 norm_layer=None, 
                 cnf:str=None, 
                 spiking_neuron:callable=None, 
                 **kwargs):
        
        super().__init__()
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                if m.weight.requires_grad == False:
                    continue
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, SEW_Bottleneck_conv1x1_identity):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, SEW_BasicBlock_conv1x1_identity):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cnf: str=None, spiking_neuron: callable = None, **kwargs):
        
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, cnf, spiking_neuron, **kwargs))
        
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _sew_resnet(arch, block, layers, pretrained, progress, cnf, spiking_neuron, **kwargs):
    model = SEWResNet(block, layers, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
    return model


def sew_resnet18(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable=None, **kwargs):

    return _sew_resnet('resnet18', SEW_BasicBlock_conv1x1_identity, [2, 2, 2, 2], pretrained, progress, cnf, spiking_neuron, **kwargs)


def sew_resnet34(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable=None, **kwargs):

    return _sew_resnet('resnet34', SEW_BasicBlock_conv1x1_identity, [3, 4, 6, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)


def sew_resnet50(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable=None, **kwargs):

    return _sew_resnet('resnet50', SEW_Bottleneck_conv1x1_identity, [3, 4, 6, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)


def sew_resnet101(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable=None, **kwargs):

    return _sew_resnet('resnet101', SEW_Bottleneck_conv1x1_identity, [3, 4, 23, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)


def sew_resnet152(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable=None, **kwargs):

    return _sew_resnet('resnet152', SEW_Bottleneck_conv1x1_identity, [3, 8, 36, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)


def sew_resnext50_32x4d(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable=None, **kwargs):

    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _sew_resnet('resnext50_32x4d', SEW_Bottleneck_conv1x1_identity, [3, 4, 6, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)


def sew_resnext50_32x8d(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable=None, **kwargs):

    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _sew_resnet(None, SEW_Bottleneck_conv1x1_identity, [3, 4, 6, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)

def sew_resnext50_32x16d(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable=None, **kwargs):

    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    return _sew_resnet(None, SEW_Bottleneck_conv1x1_identity, [3, 4, 6, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)

def sew_resnext50_32x32d(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable=None, **kwargs):

    kwargs['groups'] = 32
    kwargs['width_per_group'] = 32
    return _sew_resnet(None, SEW_Bottleneck_conv1x1_identity, [3, 4, 6, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)


def sew_resnext101_32x8d(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable=None, **kwargs):

    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _sew_resnet('resnext101_32x8d', SEW_Bottleneck_conv1x1_identity, [3, 4, 23, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)


def sew_wide_resnet50_2(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable=None, **kwargs):

    kwargs['width_per_group'] = 64 * 2
    return _sew_resnet('wide_resnet50_2', SEW_Bottleneck_conv1x1_identity, [3, 4, 6, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)

def sew_wide_resnet101_2(pretrained=False, progress=True, cnf: str = None, spiking_neuron: callable=None, **kwargs):

    kwargs['width_per_group'] = 64 * 2
    return _sew_resnet('wide_resnet101_2', SEW_Bottleneck_conv1x1_identity, [3, 4, 23, 3], pretrained, progress, cnf, spiking_neuron, **kwargs)


if __name__ == "__main__":
    
    model = sew_resnet18(
                        spiking_neuron=neuron.IFNode,
                        surrogate_function=surrogate.ATan(), 
                        detach_reset=True,
                        cnf='ADD')
    
    functional.set_step_mode(model, step_mode='m')
    x = torch.zeros(4,1,3,32,32)
    y = model(x)



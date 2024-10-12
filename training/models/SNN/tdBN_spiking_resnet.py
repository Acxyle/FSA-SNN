#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 21:51:06 2024

@author: fangwei123456, nguyenhongson1902

refer to: 
    https://github.com/fangwei123456/spikingjelly
    https://github.com/nguyenhongson1902/direct-training-snn/tree/main

@modified: acxyle

    This is a mixture of SpikingResnet (pkg:spikingjelly) and DeepSpikingResidualNetwork (Nguyenhongson)

"""

import torch
import torch.nn as nn

from copy import deepcopy
from spikingjelly.activation_based import layer


# ----------------------------------------------------------------------------------------------------------------------
__all__ = ['tdBN_SpikingResNet', 
           'tdbn_spiking_resnet18', 'tdbn_spiking_resnet34', 'tdbn_spiking_resnet50', 'tdbn_spiking_resnet101', 'tdbn_spiking_resnet152', 
           'tdbn_spiking_resnext50_32x4d', 'tdbn_spiking_resnext101_32x8d', 'tdbn_spiking_wide_resnet50_2', 'tdbn_spiking_wide_resnet101_2']

class tdBatchNorm2d(nn.BatchNorm2d):
    """
        Params and Input shape: [T, B, C, W, H]
    
    Args:
        alpha: 1/sqrt(n)*alpha
        v_threshold:
        
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1., v_threshold=1., affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        
        self.alpha = alpha
        self.v_threshold = v_threshold

        # Modify the shape of self.running_mean, self.running_var to be suitable with the mean and var shapes
        self.running_mean = torch.reshape(self.running_mean, (1, 1, -1, 1, 1))
        self.running_var = torch.reshape(self.running_var, (1, 1, -1, 1, 1))

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates (i.e. running mean and running var)
        if self.training: # In training mode, the current mean and variance are used
            mean = input.mean(dim=(0, 1, 3, 4), keepdim=True)
            var = input.var(dim=(0, 1, 3, 4), unbiased=False, keepdim=True) # compute variance via the biased estimator (i.e. unbiased=False)
            n = input.numel() / input.size(1)
            
            with torch.no_grad():
                self.running_mean = self.running_mean.view_as(mean)
                self.running_var = self.running_var.view_as(var)

                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var     # update running_var with unbiased var
        
        else: # In test mode, use mean and variance obtained by moving average
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * self.v_threshold * (input - mean) / (torch.sqrt(var + self.eps)) # x_k in paper https://arxiv.org/pdf/2011.05280
        if self.affine: # if True, we use the affine transformation (linear transformation)
            input = input * self.weight[None, None, -1, None, None] + self.bias[None, None, -1, None, None] # y_k in paper https://arxiv.org/pdf/2011.05280

        return input
    
    def extra_repr(self):
        return f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats}, v_threshold={self.v_threshold}, alpha={self.alpha:.2f}'

    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, spiking_neuron:callable=None, alpha=1., v_threshold=1., **kwargs):
        
        super(BasicBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = tdBatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = spiking_neuron(v_threshold=v_threshold, **deepcopy(kwargs))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, alpha=alpha*v_threshold/(2**0.5)) 
        self.sn2 = spiking_neuron(v_threshold=v_threshold, **deepcopy(kwargs))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)     # implemented tdBN as well

        out += identity
        out = self.sn2(out)
        
        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, spiking_neuron:callable=None, alpha=1., v_threshold=1., **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = spiking_neuron(v_threshold=v_threshold, **deepcopy(kwargs))
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = spiking_neuron(v_threshold=v_threshold, **deepcopy(kwargs))
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, alpha=alpha*v_threshold/(2**0.5))
        self.sn3 = spiking_neuron(v_threshold=v_threshold, **deepcopy(kwargs))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn3(out)

        return out


class tdBN_SpikingResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, spiking_neuron:callable=None, alpha=1., v_threshold=1., **kwargs):
        
        super(tdBN_SpikingResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = tdBatchNorm2d
            
        self._norm_layer = norm_layer

        self.inplanes = 64 # for self.conv1
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None, or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        # ----- 1. module 1st CNN
        self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = spiking_neuron(v_threshold=v_threshold, **deepcopy(kwargs))
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ----- 2. module feature
        # --- 2a. Standard Resnet 
        self.layer1 = self._make_layer(block, 64, layers[0], spiking_neuron=spiking_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], spiking_neuron=spiking_neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], spiking_neuron=spiking_neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], spiking_neuron=spiking_neuron, **kwargs)
        # --- 2b. DSRN
        # self.layer1 = self._make_layer(block, 128, layers[0], spiking_neuron=spiking_neuron, **kwargs)
        # self.layer2 = self._make_layer(block, 256, layers[1], stride=2, dilate=replace_stride_with_dilation[0], spiking_neuron=spiking_neuron, **kwargs)
        # self.layer3 = self._make_layer(block, 512, layers[2], stride=2, dilate=replace_stride_with_dilation[1], spiking_neuron=spiking_neuron, **kwargs)
        
        # ----- 3. module avgp
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        # -----
        
        # ----- 4. module classifier
        # --- 4a. standard Resnet
        self.fc = layer.Linear(512 * block.expansion, num_classes)
        # --- 4b. DSRN
        # self.fc1 = layer.Linear(512 * block.expansion, 256)
        # self.fc_s1 = spiking_neuron(v_threshold=v_threshold, **deepcopy(kwargs))
        # self.fc2 = layer.Linear(256, num_classes)
        # self.fc_s2 = spiking_neuron(v_threshold=v_threshold, **deepcopy(kwargs))
        
        # --- init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, spiking_neuron:callable=None, alpha=1., v_threshold=1., **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, alpha=alpha*v_threshold/(2**0.5)),
            )    
       
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, 
                            norm_layer, spiking_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, spiking_neuron=spiking_neuron, **kwargs))

        return nn.Sequential(*layers)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # --- standard Resnet
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
            
        # --- standard Resnet
        x = self.fc(x)
        # --- DSRN
        # x = self.fc1(x)
        # x = self.fc_s1(x)
        # x = self.fc2(x)
        # x = self.fc_s2(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


# --- standard Resnet
def _tdbn_spiking_resnet(arch, block, layers, progress, spiking_neuron, **kwargs):
    return tdBN_SpikingResNet(block, layers, spiking_neuron=spiking_neuron, **kwargs)

def tdbn_spiking_resnet18(progress=True, spiking_neuron: callable=None, **kwargs):
    return _tdbn_spiking_resnet('resnet18', BasicBlock, [2, 2, 2, 2], progress, spiking_neuron, **kwargs)     # Standard Resnet
    # return _tdbn_spiking_resnet('resnet18', BasicBlock, [3, 3, 2], progress, spiking_neuron, **kwargs)     # DSRN


def tdbn_spiking_resnet34(progress=True, spiking_neuron: callable=None, **kwargs):
    return _tdbn_spiking_resnet('resnet34', BasicBlock, [3, 4, 6, 3], progress, spiking_neuron, **kwargs)


def tdbn_spiking_resnet50(progress=True, spiking_neuron: callable=None, **kwargs):
    return _tdbn_spiking_resnet('resnet50', Bottleneck, [3, 4, 6, 3], progress, spiking_neuron, **kwargs)


def tdbn_spiking_resnet101(progress=True, spiking_neuron: callable=None, **kwargs):
    return _tdbn_spiking_resnet('resnet101', Bottleneck, [3, 4, 23, 3], progress, spiking_neuron, **kwargs)


def tdbn_spiking_resnet152(progress=True, spiking_neuron: callable=None, **kwargs):
    return _tdbn_spiking_resnet('resnet152', Bottleneck, [3, 8, 36, 3], progress, spiking_neuron, **kwargs)


def tdbn_spiking_resnext50_32x4d(progress=True, spiking_neuron: callable=None, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _tdbn_spiking_resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], progress, spiking_neuron, **kwargs)


def tdbn_spiking_resnext101_32x8d(progress=True, spiking_neuron: callable=None, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _tdbn_spiking_resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], progress, spiking_neuron, **kwargs)


def tdbn_spiking_wide_resnet50_2(progress=True, spiking_neuron: callable=None, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _tdbn_spiking_resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], progress, spiking_neuron, **kwargs)


def tdbn_spiking_wide_resnet101_2(progress=True, spiking_neuron: callable=None, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _tdbn_spiking_resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], progress, spiking_neuron, **kwargs)


# if __name__ == "__main__":
    
#     from spikingjelly.activation_based import surrogate, neuron, functional
    
#     model = tdbn_spiking_resnet50(num_classes=1000,
#                                 spiking_neuron=neuron.IFNode, 
#                                 surrogate_function=surrogate.ATan(), 
#                                 detach_reset=False, 
#                                 zero_init_residual=False)
#     functional.set_step_mode(model, step_mode='m')
    
#     print(model)
    
#     T = 4
#     x = torch.randn(T, 2, 3, 224, 224)
#     out = model(x)

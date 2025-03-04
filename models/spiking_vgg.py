#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 22:39:36 2022

@author: fangwei123456

@modified: acxyle

    this code is modified from the spiking VGG model from spikingjelly
    - weights access has been removed
    
"""

import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import functional, neuron, layer

__all__ = [
    'SpikingVGG',

    'spiking_vgg5','spiking_vgg5_bn',

    'spiking_vgg11','spiking_vgg11_bn',
    'spiking_vgg13','spiking_vgg13_bn',
    'spiking_vgg16','spiking_vgg16_bn',
    'spiking_vgg19','spiking_vgg19_bn',

    'spiking_vgg25','spiking_vgg25_bn',
    'spiking_vgg37','spiking_vgg37_bn',
    'spiking_vgg48','spiking_vgg48_bn',
]

class SpikingVGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, norm_layer=None, num_classes=1000, init_weights=True,
                 spiking_neuron: callable = None, **kwargs):
        
        super(SpikingVGG, self).__init__()
        
        self.features = self.make_layers(cfg=cfg, batch_norm=batch_norm, norm_layer=norm_layer, neuron=spiking_neuron, **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((7, 7))

        if len(self.features) == 8 or len(self.features) == 11:
                    
            self.classifier = nn.Sequential(
                layer.Linear(128 * 7 * 7, 1024),
                spiking_neuron(**deepcopy(kwargs)),
                layer.Dropout(),
                layer.Linear(1024, num_classes),
                )

        else:

            self.classifier = nn.Sequential(
                layer.Linear(512 * 7 * 7, 4096),
                spiking_neuron(**deepcopy(kwargs)),
                layer.Dropout(),
                layer.Linear(4096, 4096),
                spiking_neuron(**deepcopy(kwargs)),
                layer.Dropout(),
                layer.Linear(4096, num_classes),
            )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(cfg, batch_norm=False, norm_layer=None, neuron: callable = None, **kwargs):
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = layer.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, norm_layer(v), neuron(**deepcopy(kwargs))]
                else:
                    layers += [conv2d, neuron(**deepcopy(kwargs))]
                in_channels = v
        return nn.Sequential(*layers)


def sequential_forward(sequential, x_seq):
    assert isinstance(sequential, nn.Sequential)
    out = x_seq
    for i in range(len(sequential)):
        m = sequential[i]
        if isinstance(m, neuron.BaseNode):
            out = m(out)
        else:
            out = functional.seq_to_ann_forward(out, m)
    return out


cfgs = {
    'O': [64, 'M', 128, 128, 'M'],
    
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    
    "G": [64, 64, "M", 128, 128, "M", *[256]*6, "M", *[512]*6, "M", *[512]*6, "M"],
    "H": [64, 64, "M", 128, 128, "M", *[256]*10, "M", *[512]*10, "M", *[512]*10, "M"],
    "J": [64, 64, "M", 128, 128, "M", *[256]*15, "M", *[512]*15, "M", *[512]*11, "M"],
}


def _spiking_vgg(arch, cfg, batch_norm, pretrained, progress, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):
    
    if pretrained:
        kwargs['init_weights'] = False
        
    if batch_norm:
        norm_layer = norm_layer
    else:
        norm_layer = None
        
    model = SpikingVGG(cfg=cfgs[cfg], batch_norm=batch_norm, norm_layer=norm_layer, spiking_neuron=spiking_neuron, **kwargs)

    return model


# -----
def spiking_vgg5(pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs):

    return _spiking_vgg(None, 'O', False, pretrained, progress, None, spiking_neuron, **kwargs)


def spiking_vgg5_bn(pretrained=False, progress=True, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):

    return _spiking_vgg(None, 'O', True, pretrained, progress, norm_layer, spiking_neuron, **kwargs)


def spiking_vgg11(pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-11
        :rtype: torch.nn.Module

        A spiking version of VGG-11 model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg11', 'A', False, pretrained, progress, None, spiking_neuron, **kwargs)


def spiking_vgg11_bn(pretrained=False, progress=True, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):

    return _spiking_vgg('vgg11_bn', 'A', True, pretrained, progress, norm_layer, spiking_neuron, **kwargs)


def spiking_vgg13(pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs):

    return _spiking_vgg('vgg13', 'B', False, pretrained, progress, None, spiking_neuron, **kwargs)


def spiking_vgg13_bn(pretrained=False, progress=True, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):

    return _spiking_vgg('vgg13_bn', 'B', True, pretrained, progress, norm_layer, spiking_neuron, **kwargs)


def spiking_vgg16(pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs):

    return _spiking_vgg('vgg16', 'D', False, pretrained, progress, None, spiking_neuron, **kwargs)


def spiking_vgg16_bn(pretrained=False, progress=True, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):

    return _spiking_vgg('vgg16_bn', 'D', True, pretrained, progress, norm_layer, spiking_neuron, **kwargs)


def spiking_vgg19(pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs):

    return _spiking_vgg('vgg19', 'E', False, pretrained, progress, None, spiking_neuron, **kwargs)


def spiking_vgg19_bn(pretrained=False, progress=True, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):

    return _spiking_vgg('vgg19_bn', 'E', True, pretrained, progress, norm_layer, spiking_neuron, **kwargs)


def spiking_vgg25(pretrained=False, progress=True, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):
    
    return _spiking_vgg('vgg25', "G", False, pretrained, progress, norm_layer, spiking_neuron, **kwargs)


def spiking_vgg25_bn(pretrained=False, progress=True, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):
    
    return _spiking_vgg('vgg25_bn', "G", True, pretrained, progress, norm_layer, spiking_neuron, **kwargs)


def spiking_vgg37(pretrained=False, progress=True, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):
    
    return _spiking_vgg('vgg37', "H", False, pretrained, progress, norm_layer, spiking_neuron, **kwargs)


def spiking_vgg37_bn(pretrained=False, progress=True, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):
    
    return _spiking_vgg('vgg37_bn', "H", True, pretrained, progress, norm_layer, spiking_neuron, **kwargs)


def spiking_vgg48(pretrained=False, progress=True, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):
    
    return _spiking_vgg('vgg48', "J", False, pretrained, progress, norm_layer, spiking_neuron, **kwargs)


def spiking_vgg48_bn(pretrained=False, progress=True, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):
    
    return _spiking_vgg('vgg48_bn', "J", True, pretrained, progress, norm_layer, spiking_neuron, **kwargs)

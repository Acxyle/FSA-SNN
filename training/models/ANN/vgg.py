#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:10:25 2024

@author: pytorch
@modify: acxyle
    
    - added vgg(5/25/37/58)
    - added vgg(5/11/13/16/19/25/37/48)_lite for 32^2 (eg. CIFAR)
    - rewriten __str__():
        usage:
            import models_
            model = models_.vgg16_bn()
            str(model)
            #'vgg16_bn'
    
"""

from typing import Any, cast, Dict, List, Union

import torch
import torch.nn as nn


__all__ = [
    "VGG",
    
    "vgg5", "vgg5_bn",
    
    "vgg11", "vgg11_bn",
    "vgg13", "vgg13_bn",
    "vgg16", "vgg16_bn",
    "vgg19", "vgg19_bn",
    
    "vgg25", "vgg25_bn",
    "vgg37", "vgg37_bn",
    "vgg48", "vgg48_bn",
    
    "vgg5_lite", "vgg5_bn_lite",
    
    "vgg11_lite", "vgg11_bn_lite",
    "vgg13_lite", "vgg13_bn_lite",
    "vgg16_lite", "vgg16_bn_lite",
    "vgg19_lite", "vgg19_bn_lite",
    
    "vgg25_lite", "vgg25_bn_lite",
    "vgg37_lite", "vgg37_bn_lite",
    "vgg48_lite", "vgg48_bn_lite",
]


class VGG_Base(nn.Module):
    
    def __init__(self, cfgb:str, features:nn.Module, **kwargs)->None:
        
        super().__init__()
        
        self.cfgb = cfgb
        
        self.features = features

    def __str__(self):
        
        ...
        
    def _init_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# ---
cfgs: Dict[str, List[Union[str, int]]] = {
    'O': [64, 'M', *[128]*2, 'M'],
    
    'A': [64, 'M', 128, 'M', *[256]*2, 'M', *[512]*2, 'M', *[512]*2, 'M'],
    'B': [*[64]*2, 'M', *[128]*2, 'M', *[256]*2, 'M', *[512]*2, 'M', *[512]*2, 'M'],
    'D': [*[64]*2, 'M', *[128]*2, 'M', *[256]*3, 'M', *[512]*3, 'M', *[512]*3, 'M'],
    'E': [*[64]*2, 'M', *[128]*2, 'M', *[256]*4, 'M', *[512]*4, 'M', *[512]*4, 'M'],
    
    'G': [*[64]*2, "M", *[128]*2, "M", *[256]*6, "M", *[512]*6, "M", *[512]*6, "M"],
    'H': [*[64]*2, "M", *[128]*2, "M", *[256]*10, "M", *[512]*10, "M", *[512]*10, "M"],
    'J': [*[64]*2, "M", *[128]*2, "M", *[256]*15, "M", *[512]*15, "M", *[512]*11, "M"],
       }

cfgs_to_models: Dict[str, str] = {
    'O': 'vgg5',
    'A': 'vgg11',
    'B': 'vgg13',
    'D': 'vgg16',
    'E': 'vgg19',
    'G': 'vgg25',
    'H': 'vgg37',
    'J': 'vgg48'
    }


def make_layers(cfg: List[Union[str, int]], batch_norm: bool=False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(VGG_Base):
    
    def __init__(self, cfgb:str, features:nn.Module, num_classes:int=1000, init_weights:bool=True, dropout:float=0.5, **kwargs)->None:
        
        super().__init__(cfgb, features, **kwargs)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        if len(features) == 8 or len(features) == 11:     # for vgg5 and vgg5_bn
            
            self.classifier = nn.Sequential(
                                            nn.Linear(128 * 7 * 7, 1024),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=dropout),
                                            nn.Linear(1024, num_classes),
                                            )

        else:     # for others
        
            self.classifier = nn.Sequential(
                                            nn.Linear(512 * 7 * 7, 4096),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=dropout),
                                            nn.Linear(4096, 4096),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=dropout),
                                            nn.Linear(4096, num_classes),
                                            )
            
        if init_weights:
            self._init_weights()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def __str__(self):
        return cfgs_to_models[self.cfgb[0]] + ('_bn' if self.cfgb[1] else '')


# --- factory function for vgg
def _vgg(cfg:str, batch_norm:bool, **kwargs:Any)-> VGG:
    
    model = VGG((cfg, batch_norm), make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    
    return model


# ----- shallow vgg
def vgg5(**kwargs: Any) -> VGG:
    return _vgg("O", False, **kwargs)

def vgg5_bn(**kwargs: Any) -> VGG:
    return _vgg("O", True, **kwargs)


# ----- standard models
def vgg11(**kwargs: Any) -> VGG:
    return _vgg("A", False, **kwargs)

def vgg11_bn(**kwargs: Any) -> VGG:
    return _vgg("A", True, **kwargs)

def vgg13(**kwargs: Any) -> VGG:
    return _vgg("B", False, **kwargs)

def vgg13_bn(**kwargs: Any) -> VGG:
    return _vgg("B", True, **kwargs)

def vgg16(**kwargs: Any) -> VGG:
    return _vgg("D", False, **kwargs)

def vgg16_bn(**kwargs: Any) -> VGG:
    return _vgg("D", True, **kwargs)

def vgg19(**kwargs: Any) -> VGG:
    return _vgg("E", False, **kwargs)

def vgg19_bn(**kwargs: Any) -> VGG:
    return _vgg("E", True, **kwargs)


# ----- deep vggs
def vgg25(**kwargs: Any) -> VGG:
    return _vgg("G", False, **kwargs)

def vgg25_bn(**kwargs: Any) -> VGG:
    return _vgg("G", True, **kwargs)

def vgg37(**kwargs: Any) -> VGG:
    return _vgg("H", False, **kwargs)

def vgg37_bn(**kwargs: Any) -> VGG:
    return _vgg("H", True, **kwargs)

def vgg48(**kwargs: Any) -> VGG:
    return _vgg("J", False, **kwargs)

def vgg48_bn(**kwargs: Any) -> VGG:
    return _vgg("J", True, **kwargs)


# -----
class VGG_lite(VGG_Base):
    
    def __init__(self, cfgb:str, features:nn.Module, num_classes:int=10, init_weights:bool=True, dropout:float=0.5, **kwargs)->None:
        
        super().__init__(cfgb, features, **kwargs)
        
        if len(features) == 8 or len(features) == 11:     # for vgg5 and vgg5_bn
            
            self.classifier = nn.Sequential(
                                            nn.Linear(128 * 8 * 8, 256),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=dropout),
                                            nn.Linear(256, num_classes),
                                            )

        else:     # for others
        
            self.classifier = nn.Sequential(
                                            nn.Linear(512, 512),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=dropout),
                                            nn.Linear(512, num_classes),
                                            )
            
        if init_weights:
            self._init_weights()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def __str__(self):
        return cfgs_to_models[self.cfgb[0]] + ('_bn' if self.cfgb[1] else '') + '_lite'
    

def _vgg_lite(cfg:str, batch_norm:bool, **kwargs:Any)-> VGG:
    
    model = VGG_lite((cfg, batch_norm), make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    
    return model


# ----- shallow vgg
def vgg5_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("O", False, **kwargs)

def vgg5_bn_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("O", True, **kwargs)


# ----- standard models
def vgg11_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("A", False, **kwargs)

def vgg11_bn_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("A", True, **kwargs)

def vgg13_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("B", False, **kwargs)

def vgg13_bn_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("B", True, **kwargs)

def vgg16_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("D", False, **kwargs)

def vgg16_bn_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("D", True, **kwargs)

def vgg19_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("E", False, **kwargs)

def vgg19_bn_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("E", True, **kwargs)


# ----- deep vggs
def vgg25_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("G", False, **kwargs)

def vgg25_bn_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("G", True, **kwargs)

def vgg37_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("H", False, **kwargs)

def vgg37_bn_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("H", True, **kwargs)

def vgg48_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("J", False, **kwargs)

def vgg48_bn_lite(**kwargs: Any) -> VGG:
    return _vgg_lite("J", True, **kwargs)


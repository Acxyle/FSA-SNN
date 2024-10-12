#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 22:39:36 2024

@author: acxyle
    
    **DEMO**

    * no lite version model included
    * no layer-pruned model supported
    
    expected input: (1) model name; (2) model

    if the input is model name, then return standard layer names, units, and shapes. 
    if the input is model, then return standard layer names, (unique) units and shapes

"""

import torch
import torch.nn as nn
from typing import List, Literal
#import torchvision

from spikingjelly.activation_based import surrogate, neuron, functional, layer

from training import models

# ----------------------------------------------------------------------------------------------------------------------
class CNN_layers_base():
    
    def __init__(self, **kwargs):
        
        ...
        
    def get_layer_names_and_units_and_shapes(self, **kwargs) -> List:
        
        layers, units = self.get_layer_names_and_units(**kwargs)
        shapes = self.get_layer_shapes(**kwargs)
        
        return layers, units, shapes
    
    def get_layer_names_and_units(self, **kwargs) -> List:
        
        layers = self._layer_names(**kwargs)
        units = self._layer_units(**kwargs)
        
        return layers, units
    
    def get_layer_names(self, **kwargs) -> list:
        
        return self._layer_names(**kwargs)
    
    def get_layer_units(self, **kwargs) -> list:
        
        return self._layer_units(**kwargs)
    
    def get_layer_shapes(self, num_samples=500, **kwargs) -> list:
        
        layers, units = self.get_layer_names_and_units(**kwargs)
        
        return [_ for _ in zip([num_samples]*len(layers), units)]
    
    @property
    def target_layers(self, ) -> tuple:
        
        target_layers = (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear, 
                        nn.ReLU, 
                        
                        layer.Conv2d, layer.BatchNorm2d, layer.MaxPool2d, layer.AdaptiveAvgPool2d, layer.Linear, 
                        neuron.IFNode, neuron.LIFNode, neuron.ParametricLIFNode, neuron.QIFNode, neuron.EIFNode, neuron.IzhikevichNode)
       
        return target_layers


    def _layer_units(self, feature_shape:tuple[int]=(3,224,224), batch_size:int=1, T:int=4, **kwargs) -> list:
        """ ANN and STBP supported """
        # ---
        self.model.eval()
        
        features = []
        handles = []
        
        # --- 
        if 'spiking' in self.model_name or 'sew' in self.model_name:
            
            dummy_input = _preprocess_input(torch.zeros(batch_size, *feature_shape), T=T).to(next(self.model.parameters()).device)
            
            def _hook_fn(module, inputs, outputs):
                features.append(_postprocess_output(outputs.detach().cpu()))
        else:
            
            dummy_input = torch.zeros(batch_size, *feature_shape).to(next(self.model.parameters()).device)
            
            def _hook_fn(module, inputs, outputs):
                features.append(outputs.detach().cpu())

        # ---
        for idx, (_name, _module) in enumerate(self.model.named_modules()):
            if isinstance(_module, self.target_layers) and ('downsample' not in _name) and ('conv1x1' not in _name):
                handle = _module.register_forward_hook(_hook_fn)
                handles.append(handle)
                
        _ = self.model(dummy_input)
        functional.reset_net(self.model)
        
        for handle in handles:
            handle.remove()
        
        # --- 4. collect info
        units = []

        for idx, _ in enumerate(features):
            units.append(_.numel())
              
        functional.reset_net(self.model)
            
        return units


class VGG_layers_base(CNN_layers_base):
    
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
    
    
    def _layer_names(self, ) -> list:

        if 'spiking' in self.model_name:
            model_name = self.model_name.split('_', 1)[-1]
            act = 'sn'
        else:
            model_name = self.model_name
            act = 'an'
    
        # --- 0. init
        basic_block = ['Conv', 'BN', f'{act}'] if 'bn' in model_name else ['Conv', f'{act}']
        layers = [] 
        l_idx = 1
        b_idx = 1
        
        # --- 1. features
        for v in self.cfgs[model_name.split('_')[0]]:     # for each block
            if v == 'M':
                layers += [f'L{l_idx}_MaxPool']
                b_idx = 1
                l_idx += 1
            else:
                layers += [f'L{l_idx}_B{b_idx}_{_}' for _ in basic_block] 
                b_idx += 1
                
        layers += ['AvgPool']
        
        # --- 2. classifier
        if '5' in model_name:
            layers += ['fc_1', f'{act}_1', 'fc_2']  
        else:
            layers += ['fc_1', f'{act}_1', 'fc_2', f'{act}_2', 'fc_3']
        
        return layers
    
    
    @property
    def cfgs(self) -> dict:
        
        cfgs = {
            'vgg5': [64, 'M', *[128]*2, 'M'],
            
            'vgg11': [64, 'M', 128, 'M', *[256]*2, 'M', *[512]*2, 'M', *[512]*2, 'M'],
            'vgg13': [*[64]*2, 'M', *[128]*2, 'M', *[256]*2, 'M', *[512]*2, 'M', *[512]*2, 'M'],
            'vgg16': [*[64]*2, 'M', *[128]*2, 'M', *[256]*3, 'M', *[512]*3, 'M', *[512]*3, 'M'],
            'vgg19': [*[64]*2, 'M', *[128]*2, 'M', *[256]*4, 'M', *[512]*4, 'M', *[512]*4, 'M'],
            
            'vgg25': [*[64]*2, "M", *[128]*2, "M", *[256]*6, "M", *[512]*6, "M", *[512]*6, "M"],
            'vgg37': [*[64]*2, "M", *[128]*2, "M", *[256]*10, "M", *[512]*10, "M", *[512]*10, "M"],
            'vgg48': [*[64]*2, "M", *[128]*2, "M", *[256]*15, "M", *[512]*15, "M", *[512]*11, "M"],
               }
        
        return cfgs
    
 
class VGG_layers_info_generator(VGG_layers_base):
    
    def __init__(self, model:Literal[nn.Module, str], model_name=None, num_classes=50, **kwargs):
        
        super().__init__(**kwargs)
        
        if isinstance(model, nn.Module) and model_name is not None:
            
            self.model_name = model_name
            self.model = model
            
        elif isinstance(model, str):
            
            self.model_name = model.lower()
            self.model = models.vgg.__dict__[model](num_classes=num_classes)
        
        self.model.eval()
        
        super().__init__(**kwargs)
        
        if isinstance(model, nn.Module):
            
            self.model_name = str(model)     # model with __str__ rewritten
            self.model = model
            
        elif isinstance(model, str):
            
            self.model_name = model.lower()
            self.model = models.vgg.__dict__[model](num_classes=num_classes)
        
        
class SVGG_layers_info_generator(VGG_layers_base):
    
    def __init__(self, model:Literal[nn.Module, str], model_name=None, num_classes=50, **kwargs):
        
        super().__init__(**kwargs)
        
        if isinstance(model, nn.Module) and model_name is not None:
            
            self.model_name = model_name     # model with __str__ rewritten
            self.model = model
            
        elif isinstance(model, str):
            
            self.model_name = model.lower()
            self.model = models.spiking_vgg.__dict__[model](pretrained=False, 
                                                     num_classes=num_classes,
                                                     spiking_neuron=neuron.IFNode,
                                                     surrogate_function=surrogate.ATan(), 
                                                     detach_reset=True)
            
        functional.set_step_mode(self.model, step_mode='m')
        self.model.eval()
        

# ----------------------------------------------------------------------------------------------------------------------
class Resnet_layer_base(CNN_layers_base):
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
    def _layer_names(self, ) -> list:
        """
            residual, shortcut, and identity_mapping has been removed
        """
        
        model_dict = {}
        for prefix in ['', 'spiking_', 'sew_']:
            for k, v in self.cfgs.items():
                model_dict[prefix+k] = v.copy()
                
        # -----
        act = 'sn' if ('spiking' in self.model_name or 'sew' in self.model_name) else 'an'
        
        layers = ['Conv_0', 'BN_0', f'{act}_0', 'MaxPool_0']
        
        bottleneck = ['Conv_1', 'BN_1', f'{act}_1', 'Conv_2', 'BN_2', f'{act}_2', 'Conv_3', 'BN_3', f'{act}_3']
        basicblock = ['Conv_1', 'BN_1', f'{act}_1', 'Conv_2', 'BN_2', f'{act}_2']
        
        # ---
        target_blocks = basicblock if '18' in self.model_name or '34' in self.model_name else bottleneck
        for l_idx, blocks in enumerate(model_dict[self.model_name], start=1):   # each layer
            for b_idx in range(blocks):          # each block
                layers += [f'L{l_idx}_B{b_idx+1}_{_}' for _ in target_blocks]
        layers += ['AvgPool', 'fc']     
        
        return layers
        
    @property
    def cfgs(self, ):
        
        cfgs = {
            'resnet18': [2, 2, 2, 2], 
            'resnet34': [3, 4, 6, 3], 
            'resnet50': [3, 4, 6, 3], 
            'resnet101': [3, 4, 23, 3], 
            'resnet152': [3, 8, 36, 3],
            
            'resnext50_32x4d': [3, 4, 6, 3], 
            'resnext50_32x8d': [3, 4, 6, 3], 
            'resnext50_32x16d': [3, 4, 6, 3], 
            'resnext50_32x32d': [3, 4, 6, 3], 
        }
        
        return cfgs
    

class Resnet_layers_info_generator(Resnet_layer_base):
    
    def __init__(self, model:Literal[nn.Module, str], model_name=None, num_classes=50, **kwargs):
        
        super().__init__(**kwargs)
        
        if isinstance(model, nn.Module) and model_name is not None:
            
            self.model_name = model_name
            self.model = model
            
        elif isinstance(model, str):
            
            self.model_name = model.lower()
            self.model = models.resnet.__dict__[model](num_classes=num_classes)
        
        self.model.eval()
        

class SResnet_layers_info_generator(Resnet_layer_base):
    
    def __init__(self, model:Literal[nn.Module, str], model_name=None, num_classes=50, **kwargs):
        
        super().__init__(**kwargs)
        
        if isinstance(model, nn.Module) and model_name is not None:
            
            self.model_name = model_name
            self.model = model
            
        elif isinstance(model, str):
            
            self.model_name = model.lower()
            self.model = models.spiking_resnet.__dict__[model](
                                                        pretrained=False, 
                                                        num_classes=num_classes,
                                                        spiking_neuron=neuron.IFNode,
                                                        surrogate_function=surrogate.ATan(), 
                                                        detach_reset=True,
                                                        )
        
        else:
            
            raise AssertionError
        
        functional.set_step_mode(self.model, step_mode='m')
        self.model.eval()
        functional.reset_net(self.model)    


class SEWResnet_layers_info_generator(Resnet_layer_base):
    
    def __init__(self, model:Literal[nn.Module, str], model_name=None, num_classes=50, **kwargs):
        
        super().__init__(**kwargs)
        
        if isinstance(model, nn.Module) and model_name is not None:
            
            self.model_name = model_name
            self.model = model
            
        elif isinstance(model, str):
            
            self.model_name = model.lower()
            self.model = models.sew_resnet.__dict__[model](
                                                        pretrained=False, 
                                                        num_classes=num_classes,
                                                        spiking_neuron=neuron.IFNode,
                                                        surrogate_function=surrogate.ATan(), 
                                                        detach_reset=True,
                                                        cnf='ADD'
                                                        )
        
        else:
            
            raise AssertionError
        
        functional.set_step_mode(self.model, step_mode='m')
        self.model.eval()
        functional.reset_net(self.model)
    

# ----------------------------------------------------------------------------------------------------------------------
def _preprocess_input(input, T=4):
    return input.unsqueeze(0).repeat(T, 1, 1, 1, 1)

def _postprocess_output(output):
    return output.mean(0)



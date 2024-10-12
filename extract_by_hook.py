#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 20:10:50 2023

@author: acxyle

    - hook based feature extraction
    - usage: python extract_by_hook.py <universal args> NN <NN extracting args>
    
    *** this script only provided 'model_name' as entrance, not any model
    *** check legacy code for forward-based feature extraction

"""

# --- python
import os
import sys
import argparse
from tqdm import tqdm
from functools import partial

# --- pytorch
import torch

# --- spikingjelly
from spikingjelly.activation_based import neuron, functional

# --- local
from training import training_utils
from training.training_lite import training_strategy_parser, SP_Trainer_ANN, SP_Trainer_SNN

import utils_


# ----------------------------------------------------------------------------------------------------------------------
def universal_extracting_parser(parser):
    
    # --- env
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("-j", "--workers", default=1, type=int, metavar="N", help="number of data loading workers")

    parser.add_argument("--data_path", type=str, default='/home/acxyle-workstation/Dataset/CelebA50')
    
    parser.add_argument("--hierarchy", type=str, default='cls')
    parser.add_argument("--split_ratio", type=float, default=0.)
    
    # --- dummy inference
    parser.add_argument("--batch_size", type=int, default=1)
    
    return parser
    

def ANN_extracting_parser(parser_ANN):
    
    parser_ANN.add_argument("-m", "--model", type=str, default='resnet18')
    parser_ANN.add_argument("--num_classes", type=int, default=50)
    parser_ANN.add_argument("--model_weight", default='logs_ft_C50_Resnet18_C2k_fold_0/checkpoint_max_test_acc1.pth')
    
    return parser_ANN
    
    
def SNN_extracting_parser(parser_SNN, _neuron='LIF', _surrogate='ATan', T=4):
    
    parser_SNN.add_argument("-m", "--model", type=str, default='spiking_vgg16_bn')
    parser_SNN.add_argument("--num_classes", type=int, default=50)
    parser_SNN.add_argument("--model_weight", default='pth_c50/checkpoint_max_test_acc1.pth')
    
    parser_SNN.add_argument("--step_mode", type=str, default='m')
    parser_SNN.add_argument('--neuron', type=str, default=f'{_neuron}')
    parser_SNN.add_argument('--surrogate', type=str, default=f'{_surrogate}')
    parser_SNN.add_argument("--T", type=int, default=T)
    
    parser_SNN.add_argument("--return_firing_rate", default=True)
    
    return parser_SNN


def extracting_script_parser():
    
    parser = argparse.ArgumentParser(description="Featuremap Extractor")
    parser = universal_extracting_parser(parser)
    parser = training_strategy_parser(parser)
    
    parser.add_argument("--FSA_root", type=str, default="/home/acxyle-workstation/Downloads/FSA")
    parser.add_argument("--FSA_dir", type=str, default='Resnet/Resnet')
    parser.add_argument("--FSA_config", type=str, default='Resnet18_C2k_fold_/runs/Resnet18_C2k_fold_0')
    
    sub_parser = parser.add_subparsers(dest="command", help='Sub-command help')
    
    # --- ANN
    parser_ANN = sub_parser.add_parser('ANN', help='pathway for ANN')
    parser_ANN = ANN_extracting_parser(parser_ANN)
    
    # --- SNN
    parser_SNN = sub_parser.add_parser('SNN', help='pathway for SNN')
    parser_SNN = SNN_extracting_parser(parser_SNN)
    
    args = parser.parse_args()

    return args


# ----------------------------------------------------------------------------------------------------------------------
class SP_Extractor_ANN(SP_Trainer_ANN):
    
    def __init__(self, args, **kwargs) -> None:
        
        super().__init__(args, **kwargs)
        
        if not next(self.model.parameters()).device == torch.device(args.device):
            self.model.to(args.device)
        
        model_weight = os.path.join(args.FSA_root, args.FSA_dir, f'FSA {args.FSA_config}', args.model_weight)
        
        self.load_weight(model_weight)
        

    def extract(self, args, **kwargs):

        layers_info_generator = get_layers_info_generator_ANN(args, **kwargs)

        self.layers, self.units, self.shapes = get_layers_info(layers_info_generator, 'an')
        
        # --- obtains the feature map
        self.hook_registration()
        self.evaluate(args)     
        self.features_transformation()
        
        self.features_check()
        
        self.features_save(args)

        
    def features_check(self, ) -> None:
        
        for idx, u in enumerate(self.units):
            
            assert self.features[idx].shape[1] == u, 'Detected abnormal shape, please check transform() of dataset'
    
    
    def features_save(self, args) -> None:
        
        self.save_path = os.path.join(args.FSA_root, args.FSA_dir, f'FSA {args.FSA_config}/Features')
        
        os.makedirs(self.save_path, exist_ok=True)
        
        for idx, _layer in tqdm(enumerate(self.layers), 'Saving Feature', total=len(self.layers)):
            
            utils_.dump(self.features[idx], os.path.join(self.save_path, f'{_layer}.pkl'), verbose=False)
            
    
    def hook_fn(self, module, inputs, outputs) -> None:
        
        self.feature_single_layer.append(outputs.detach().cpu().reshape(args.batch_size, -1))
    

    def hook_registration(self, ) -> None:

        self.feature_single_layer = []
        self.handles = []
        
        for _, _m in self.model.named_modules():
            if isinstance(_m, torch.nn.ReLU):
                handle = _m.register_forward_hook(self.hook_fn)
                self.handles.append(handle)

    
    def features_transformation(self, ) -> None:

        num_layers = len(self.features[0])
        self.features = [torch.stack([self.features[i_idx][l_idx] for i_idx in range(500)], dim=0).reshape(500, -1).numpy() for l_idx in tqdm(range(num_layers), desc='Transforming')]
             

    def evaluate(self, args, verbose=True) -> None:

        # -----
        training_utils.set_deterministic()
        self.model.eval()
       
        top1 = training_utils.AverageMeter()
        top5 = training_utils.AverageMeter()
        _loss = training_utils.AverageMeter()
        
        with torch.inference_mode():
            
            self.features = []
            
            for i, (image, target) in tqdm(enumerate(self.data_loader_val), desc='Extracting', total=len(self.data_loader_val)):
                
                image = image.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                image = self.preprocess_test_sample(args, image)

                output = self.process_model_output(args, self.model(image))
                loss = self.criterion(output, target)

                acc1, acc5 = self.cal_acc1_acc5(output, target)
                batch_size = target.shape[0]

                functional.reset_net(self.model)
                
                top1.update(acc1.item(), batch_size)
                top5.update(acc5.item(), batch_size)
                _loss.update(loss.item(), batch_size)
                
                # --- features
                self.features.append(self.feature_single_layer)
                self.feature_single_layer = []
            
            for handle in self.handles:
                handle.remove()

        if verbose:
            print(f'Validation -> acc@1: {top1.avg:.3f}, acc@5: {top5.avg:.3f}, loss: {_loss.avg:.5f}')
            

# ----------------------------------------------------------------------------------------------------------------------
class SP_Extractor_SNN(SP_Trainer_SNN):

    def __init__(self, args, **kwargs) -> None:
        
        super().__init__(args, **kwargs)
        
        if not next(self.model.parameters()).device == torch.device(args.device):
            self.model.to(args.device)
        
        model_weight = os.path.join(args.FSA_root, args.FSA_dir, f'FSA {args.FSA_config}', args.model_weight)
        
        self.load_weight(model_weight)
        

    def extract(self, args, **kwargs):

        layers_info_generator = get_layers_info_generator_SNN(args, **kwargs)

        self.layers, self.units, self.shapes = get_layers_info(layers_info_generator, 'sn')
        
        target_module = neuron.__dict__[f'{args.neuron}Node']

        # --- obtains the feature map
        self.hook_registration(target_module=target_module)
        self.evaluate(args)     
        self.features_transformation()
        
        self.features_check()
        
        self.features_save(args)
        
        
    def features_check(self, ) -> None:
        
        for idx, u in enumerate(self.units):
            
            assert self.features[idx].shape[1] == u, 'Detected abnormal shape, please check transform() of dataset'
    
    
    def features_save(self, args) -> None:
        
        self.save_path = os.path.join(args.FSA_root, args.FSA_dir, f'FSA {args.FSA_config}/Features')
        
        os.makedirs(self.save_path, exist_ok=True)
        
        for idx, _layer in tqdm(enumerate(self.layers), 'Saving Feature', total=len(self.layers)):
            
            utils_.dump(self.features[idx], os.path.join(self.save_path, f'{_layer}.pkl'), verbose=False)
            

    def hook_fn(self, module, inputs, outputs, return_firing_rate=True) -> None:
        
        if return_firing_rate:
            self.feature_single_layer.append(torch.mean(outputs.detach().cpu(), dim=0)) 
        else:
            self.feature_single_layer.append(outputs.detach().cpu())
    

    def hook_registration(self, target_module=None) -> None:
            
        assert target_module is not None
        
        self.feature_single_layer = []
        self.handles = []
        
        for _, _m in self.model.named_modules():
            
            if isinstance(_m, target_module):
                
                handle = _m.register_forward_hook(partial(self.hook_fn, return_firing_rate=args.return_firing_rate))
                self.handles.append(handle)

    
    def features_transformation(self, return_firing_rate=True) -> None:
        
        num_samples = len(self.features)
        num_layers = len(self.features[0])
        
        if return_firing_rate:
            
            self.features = [torch.stack([self.features[i_idx][l_idx] for i_idx in range(num_samples)], dim=0).reshape(num_samples, -1).numpy() for l_idx in tqdm(range(num_layers), desc='Transforming')]
            
        else:     # --- this may RAM consuming
            
            T = self.features[0][0].shape[0]
            assert args.T == T
            
            self.features = [torch.stack([self.features[i_idx][l_idx] for i_idx in range(num_samples)], dim=1).reshape(T, num_samples, -1).numpy() for l_idx in tqdm(range(num_layers), desc='Transforming')]
            

    def evaluate(self, args, verbose=True) -> None:

        # -----
        training_utils.set_deterministic()
        self.model.eval()
       
        top1 = training_utils.AverageMeter()
        top5 = training_utils.AverageMeter()
        _loss = training_utils.AverageMeter()
        
        with torch.inference_mode():
            
            self.features = []
            
            for i, (image, target) in tqdm(enumerate(self.data_loader_val), desc='Extracting', total=len(self.data_loader_val)):
                
                image = image.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                image = self.preprocess_test_sample(args, image)

                output = self.process_model_output(args, self.model(image))
                loss = self.criterion(output, target)

                acc1, acc5 = self.cal_acc1_acc5(output, target)
                batch_size = target.shape[0]

                functional.reset_net(self.model)
                
                top1.update(acc1.item(), batch_size)
                top5.update(acc5.item(), batch_size)
                _loss.update(loss.item(), batch_size)
                
                # --- features
                self.features.append(self.feature_single_layer)
                self.feature_single_layer = []
            
            for handle in self.handles:
                handle.remove()

        if verbose:
            print(f'Validation -> acc@1: {top1.avg:.3f}, acc@5: {top5.avg:.3f}, loss: {_loss.avg:.5f}')
            
            
# ----------------------------------------------------------------------------------------------------------------------
def get_layers_info(layers_info_generator, target_element='an') -> None:
    
    layers, units, shapes = layers_info_generator.get_layer_names_and_units_and_shapes()
    
    layers, units, shapes = zip(*[(l, u, s) for l, u, s in zip(layers, units, shapes) if target_element in l])
    
    utils_.describe_model(layers, units, shapes)
    
    return layers, units, shapes


def get_layers_info_generator_ANN(args, **kwargs):

    if 'vgg' in args.model:
        layers_info_generator = utils_.VGG_layers_info_generator(model=args.model, **kwargs)
    elif 'resnet' in args.model:
        layers_info_generator = utils_.Resnet_layers_info_generator(model=args.model, **kwargs)
    else:
        raise ValueError

    return layers_info_generator


def get_layers_info_generator_SNN(args, **kwargs):

    if 'vgg' in args.model:
        layers_info_generator = utils_.SVGG_layers_info_generator(model=args.model, **kwargs)
    elif 'resnet' in args.model and 'spiking' in args.model:
        layers_info_generator = utils_.SResnet_layers_info_generator(model=args.model, **kwargs)
    elif 'resnet' in args.model and 'sew' in args.model:
        layers_info_generator = utils_.SEWResnet_layers_info_generator(model=args.model, **kwargs)
    else:
        raise ValueError

    return layers_info_generator



# ======================================================================================================================
if __name__ =="__main__":
    
    args = extracting_script_parser()
    
    if args.command == 'ANN':
        extractor = SP_Extractor_ANN(args, shuffle=False)
    elif args.command == 'SNN':
        extractor = SP_Extractor_SNN(args, shuffle=False)

    extractor.extract(args)
    
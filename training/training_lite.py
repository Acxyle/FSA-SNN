#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:59:47 2023

@author: fangwei123456
@modified: axcyle
    
    1. no tensorboard
    2. no resume
    3. no ddp

"""

# --- python
import os
import random
import argparse
import datetime
import numpy as np
from collections import OrderedDict
#from tqdm import tqdm

# --- pytorch
import torch
import torch.nn as nn
import torchvision
from torch.utils.data.dataloader import default_collate

# --- spikingjelly
from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model.tv_ref_classify import transforms, utils

# --- local
from . import models
from . import training_utils


# ----------------------------------------------------------------------------------------------------------------------
def universal_training_parser(parser):
    """ basic training config from spikingjelly training script """
    
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers")
    
    parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="images per gpu")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=1e-2, type=float, help="initial learning rate")     
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight_decay", default=1e-5, type=float, metavar="W", help="weight decay", dest="weight_decay")
    parser.add_argument("--norm_weight_decay", default=None, type=float, help="weight decay for Normalization layers")
    
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    
    return parser


def training_strategy_parser(parser):
    """ tricks used for data augmentation and other training startegies """
    
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="label smoothing", dest="label_smoothing")
    
    parser.add_argument("--mixup_alpha", default=0.2, type=float, help="mixup alpha")
    parser.add_argument("--cutmix_alpha", default=0.2, type=float, help="cutmix alpha")
    
    parser.add_argument("--lr_scheduler", default="cosa", type=str, help="the lr scheduler")
    parser.add_argument("--lr_warmup_epochs", default=5, type=int, help="the number of epochs to warmup")
    parser.add_argument("--lr_warmup_method", default="linear", type=str, help="the warmup method")
    parser.add_argument("--lr_warmup_decay", default=0.01, type=float, help="the decay for lr")
    
    parser.add_argument("--auto_augment", default='ta_wide', type=str, help="auto augment policy")
    parser.add_argument("--random_erase", default=0.1, type=float, help="random erasing probability")

    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method")

    parser.add_argument("--val_resize_size", default=232, type=int, help="the resize size used for validation")
    parser.add_argument("--val_crop_size", default=224, type=int, help="the central crop size used for validation")
    parser.add_argument("--train_crop_size", default=176, type=int, help="the random crop size used for training")
    
    parser.add_argument("--seed", default=2020, type=int, help="the random seed")

    parser.add_argument("--disable_pinmemory", action="store_true", help="not use pin memory in dataloader")
    parser.add_argument("--disable_amp", default=True, help="not use automatic mixed precision training")
    
    return parser
    

def training_parser(add_help=True):

    parser = argparse.ArgumentParser(description="SpikingJelly Classification Training", add_help=add_help)
    parser = universal_training_parser(parser)
    parser = training_strategy_parser(parser)
    
    return parser


# ----------------------------------------------------------------------------------------------------------------------
class SP_Trainer_lite():

    def __init__(self, args, **kwargs):

        print(args)
        
        # --- prepare for the training config
        self.device = torch.device(args.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        # ---
        if args.disable_amp:
            self.scaler = None
        else:
            self.scaler = torch.cuda.amp.GradScaler()     # --- default way

        # ----- prepare for the dataset
        self.prepare_datasets(args, **kwargs)
        self.prepare_dataloaders(args)
        
        # --- prepare for the model
        self.load_model(args)
        self.model.to(self.device)
        
        # ---
        if 'output_dir' in args:
            self.save_path = args.output_dir
            os.makedirs(self.save_path, exist_ok=True)


    def train(self, args, val=True, save=True, verbose=True) -> None:
        
        # ---
        training_utils.set_deterministic(args.seed)

        # --- prepare for training arguments
        optimizer = self.set_optimizer(args, self.model)
        lr_scheduler = self.set_lr_scheduler(args, optimizer)

        # -----
        max_test_acc1 = -1.
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for epoch in range(args.start_epoch, args.epochs):     # for every epoch
            
            train_acc1, train_acc5, train_loss = self.train_one_epoch(optimizer, epoch, args, verbose=verbose)

            lr_scheduler.step()
            
            if val:
                
                test_acc1, test_acc5, test_loss = self.evaluate(args)

                if test_acc1 > max_test_acc1:
                    max_test_acc1 = test_acc1
                
                if save:
                    utils.save_on_master(self.model, os.path.join(self.save_path, "checkpoint_max_test_acc1.pth"))
             
            if verbose:
                print(f'{current_time} Epoch [{epoch}] -> acc@1: {train_acc1:.3f}, acc@5: {train_acc5:.3f}, loss: {train_loss:.5f}')
                print(optimizer.state_dict()['param_groups'][0]['lr'])


    def train_one_epoch(self, optimizer, epoch, args, verbose=False):
        
        self.model.train()

        top1 = training_utils.AverageMeter()
        top5 = training_utils.AverageMeter()
        _loss = training_utils.AverageMeter()

        for i, (image, target) in enumerate(self.data_loader):

            image = image.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            with torch.amp.autocast(args.device, enabled=self.scaler is not None):

                image = self.preprocess_train_sample(args, image)
                output = self.process_model_output(args, self.model(image))
                loss = self.criterion(output, target)

            optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            functional.reset_net(self.model)

            acc1, acc5 = self.cal_acc1_acc5(output, target)
            batch_size = target.shape[0]
            
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            _loss.update(loss.item(), batch_size)
        
        return top1.avg, top5.avg, _loss.avg


    def evaluate(self, args, verbose=True):
        
        self.model.eval()
       
        top1 = training_utils.AverageMeter()
        top5 = training_utils.AverageMeter()
        _loss = training_utils.AverageMeter()
        
        with torch.inference_mode():
            
            for i, (image, target) in enumerate(self.data_loader_val):
                
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

        if verbose:
            print(f'Validation -> acc@1: {top1.avg:.3f}, acc@5: {top5.avg:.3f}, loss: {_loss.avg:.5f}')
     
        return top1.avg, top5.avg, _loss.avg

    
    def prepare_datasets(self, args, verbose=False, **kwargs):

        if args.hierarchy == 'tv':
            self.dataset_train, self.dataset_val = self.prepare_datasets_tv(args, verbose=verbose, **kwargs)
        elif args.hierarchy == 'cls':
            self.dataset_train, self.dataset_val = self.prepare_datasets_cls(args, verbose=verbose, **kwargs)
        else:
            raise ValueError
    
    @staticmethod
    def prepare_datasets_tv(args, verbose=False, **kwargs):
        
        dataset_train, dataset_val = training_utils.prepare_datasets_tv(args, verbose=verbose, **kwargs)
            
        return dataset_train, dataset_val
    
    
    @staticmethod
    def prepare_datasets_cls(args, verbose=False, **kwargs):
        
        dataset_train, dataset_val = training_utils.prepare_datasets_cls(args, split_ratio=args.split_ratio, verbose=verbose, **kwargs)
        
        return dataset_train, dataset_val
    
    
    def prepare_dataloaders(self, args):
        
        self.loader_g = torch.Generator()
        self.loader_g.manual_seed(args.seed)
        
        if len(self.dataset_train) != 0:     # --- when need to load empty dataset
            self.train_sampler = torch.utils.data.RandomSampler(self.dataset_train, generator=self.loader_g)
        else:
            self.train_sampler = torch.utils.data.SequentialSampler(self.dataset_train)
            
        self.val_sampler = torch.utils.data.SequentialSampler(self.dataset_val)
        
        collate_fn = None
        
        if args.hierarchy == 'tv':
            self.prepare_dataloaders_tv(args)
        elif args.hierarchy == 'cls':
            self.prepare_dataloader_cls(args)
        else:
            raise ValueError
            
        mixup_transforms = []
        if args.mixup_alpha > 0.0:
            mixup_transforms.append(transforms.RandomMixup(self.num_classes, p=1.0, alpha=args.mixup_alpha))
        if args.cutmix_alpha > 0.0:
            mixup_transforms.append(transforms.RandomCutmix(self.num_classes, p=1.0, alpha=args.cutmix_alpha))
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
            collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731
            
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=args.batch_size,
            sampler=self.train_sampler,
            num_workers=args.workers,
            pin_memory= not args.disable_pinmemory,
            collate_fn=collate_fn,
            worker_init_fn=_seed_worker
        )

        self.data_loader_val = torch.utils.data.DataLoader(
            self.dataset_val, 
            batch_size=args.batch_size, 
            sampler=self.val_sampler, 
            num_workers=args.workers, 
            pin_memory= not args.disable_pinmemory,
            worker_init_fn=_seed_worker
        )
    
    
    def prepare_dataloaders_tv(self, args):
        
        self.num_classes = len(self.dataset_train.classes)
        
        
    def prepare_dataloader_cls(self, args):
        
        self.num_classes = len(self.dataset_train.dataset.classes)


    @staticmethod
    def set_optimizer(args, model):
        
        return training_utils.set_optimizer(args, model)
    
    
    @staticmethod
    def set_lr_scheduler(args, optimizer):

        return training_utils.set_lr_scheduler(args, optimizer)    
    

    @staticmethod
    def cal_acc1_acc5(output, target):
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        return acc1, acc5
    
    
    def load_weight(self, weight_path):
        
        params = torch.load(weight_path)     # --- can manually set weights_only = True if OrderedDict
        
        if isinstance(params, OrderedDict):
            self.model.load_state_dict(params)
        
        elif isinstance(params, dict):     # --- assume the weight is saved by spikingjelly hierarchy
            self.model.load_state_dict(params['model'])
            
        else:
            raise ValueError


# ----------------------------------------------------------------------------------------------------------------------
class SP_Trainer_ANN(SP_Trainer_lite):
    
    def __init__(self, args, **kwargs):
        
        super().__init__(args, **kwargs)
    
    def preprocess_train_sample(self, args, x: torch.Tensor):
        return x

    def preprocess_test_sample(self, args, x: torch.Tensor):
        return x

    def process_model_output(self, args, y: torch.Tensor):
        return y

    def load_model(self, args):
        
        # --- local
        if args.model in models.ANN.vgg.__all__:
            self.model = models.ANN.vgg.__dict__[args.model](num_classes=args.num_classes)
        
        elif args.model in models.ANN.resnet.__all__:
            self.model = models.ANN.resnet.__dict__[args.model](num_classes=args.num_classes)
        
        elif args.model in models.ANN.resnet_identity_conv.__all__:
            self.model = models.ANN.resnet_identity_conv.__dict__[args.model](num_classes=args.num_classes)
        
        # --- torchvision
        elif args.model in torchvision.models.vision_transformer.__all__:
            self.model = torchvision.models.vision_transformer.__dict__[args.model](num_classes=args.num_classes)
        
        else:
            raise ValueError
            
            
# ----------------------------------------------------------------------------------------------------------------------
class SP_Trainer_SNN(SP_Trainer_lite):
    
    def __init__(self, args, **kwargs):
        
        super().__init__(args, **kwargs)
    
    def preprocess_train_sample(self, args, x: torch.Tensor):
        return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)     # [N, C, H, W] -> [T, N, C, H, W]

    def preprocess_test_sample(self, args, x: torch.Tensor):
        return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)     # [N, C, H, W] -> [T, N, C, H, W]

    def process_model_output(self, args, y: torch.Tensor):
        return y.mean(0)     # return firing rate

    def load_model(self, args):
        
        _neuron = neuron.__dict__[f'{args.neuron}Node']
        _surrogate = surrogate.__dict__[args.surrogate]()
        
        # --- SNN
        if args.model in models.SNN.spiking_resnet.__all__:
            self.model = models.SNN.spiking_resnet.__dict__[args.model](
                                                        num_classes=args.num_classes,
                                                        spiking_neuron=_neuron, 
                                                        surrogate_function=_surrogate, 
                                                        detach_reset=False, 
                                                        zero_init_residual=True)     # --- comment: cumbersome
            
        # --- deprecated ---
        elif args.model in models.SNN.tdBN_spiking_resnet.__all__:
            self.model = models.SNN.tdBN_spiking_resnet.__dict__[args.model](
                                                        alpha=1.,
                                                        v_threshold=1.,
                                                        num_classes=args.num_classes,
                                                        spiking_neuron=_neuron, 
                                                        surrogate_function=_surrogate, 
                                                        detach_reset=False, 
                                                        zero_init_residual=False)     # --- comment: seems mutually exclusive with N(0,1)?
        
        elif args.model in models.SNN.sew_resnet.__all__:
            self.model = models.SNN.sew_resnet.__dict__[args.model](
                                                    num_classes=args.num_classes,
                                                    spiking_neuron=_neuron,
                                                    surrogate_function=_surrogate, 
                                                    detach_reset=True, 
                                                    cnf='ADD')
        
        elif args.model in models.SNN.spiking_vgg.__all__:
            self.model = models.SNN.spiking_vgg.__dict__[args.model](
                                                     num_classes=args.num_classes,
                                                     spiking_neuron=_neuron,
                                                     surrogate_function=_surrogate, 
                                                     detach_reset=True)
            
        else:
            raise ValueError
            
        # ---
        functional.set_step_mode(self.model, step_mode='m')


def _seed_worker(worker_id):

    worker_seed = torch.initial_seed() % int(np.power(2, 32))     # --- set float for windows OS

    np.random.seed(worker_seed)
    random.seed(worker_seed)
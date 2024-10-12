#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 04:13:27 2023

@author: fangwei123456
@modified: acxyle

    contains: 
        1. dataset
        2. dataloader
        3. optimizer
        4. scheduler

"""

# --- python
import os
import random
import numpy as np

# --- pytorch
import torch
import torchvision
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import random_split

# --- spikingjelly
from spikingjelly.activation_based.model.tv_ref_classify import presets


# ----------------------------------------------------------------------------------------------------------------------
def prepare_datasets_tv(args, verbose=False):
    """ tv, train & val, assume the directory's hierarchy is like ImageNet folders """
    
    assert 'train' in os.listdir(args.data_path) and 'val' in os.listdir(args.data_path), "invalid data structure"
    
    traindir = os.path.join(args.data_path, "train")
    valdir = os.path.join(args.data_path, "val")
    
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)

    auto_augment_policy = getattr(args, "auto_augment", None)     # -> 0.2
    random_erase_prob = getattr(args, "random_erase", 0.0)     # -> 0.2
    
    dataset = torchvision.datasets.ImageFolder(
                                            root = traindir,
                                            transform = presets.ClassificationPresetTrain(
                                                                crop_size=train_crop_size,
                                                                interpolation=interpolation,
                                                                auto_augment_policy=auto_augment_policy,
                                                                random_erase_prob=random_erase_prob,
                                                                                        ),
                                            )
     
    dataset_test = torchvision.datasets.ImageFolder(
                                                    root = valdir, 
                                                    transform = presets.ClassificationPresetEval(
                                                                        crop_size=val_crop_size, 
                                                                        resize_size=val_resize_size, 
                                                                        interpolation=interpolation
                                                                                                ),
                                                    )
        
    if verbose:
        print(dataset)
        print(dataset_test)
    
    return dataset, dataset_test


def prepare_datasets_cls(args, split_ratio=0.8, verbose=False, shuffle=True):
    """ cls, classes, assume the directory's hierarchy is like subclasses folders """
    dataset = torchvision.datasets.ImageFolder(root = args.data_path, 
                                               transform = presets.ClassificationPresetEval(
                                                            resize_size=args.val_resize_size, 
                                                            crop_size=args.val_crop_size, 
                                                            interpolation=InterpolationMode(args.interpolation)
                                                                                        ),)
    if shuffle:
        dataset_train, dataset_val = random_split(dataset, [split_ratio, 1 - split_ratio])
    else:
        num_samples = len(dataset)
        train_indices = np.arange(int(num_samples*split_ratio))
        val_indices = np.arange(len(train_indices), num_samples)
        
        dataset_train = torch.utils.data.Subset(dataset, train_indices)
        dataset_val = torch.utils.data.Subset(dataset, val_indices)
        
    return dataset_train, dataset_val


def get_dataloader_single(args, dataset, num_workers=1):
    
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=1, 
                                             shuffle=False, 
                                             num_workers=num_workers, 
                                             pin_memory=False, 
                                             worker_init_fn=seed_worker)
    
    return dataloader


def set_optimizer(args, model):
    
    if args.norm_weight_decay is None:
        parameters = model.parameters()
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]
    
    optimizer = torch.optim.SGD(
        parameters,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
   
    return optimizer


def set_lr_scheduler(args, optimizer):

    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.lr_warmup_epochs)
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs])

    return lr_scheduler


def set_deterministic(_seed_=2020):
    """
        "A handful of CUDA operations are nondeterministic if the CUDA version is 10.2 or greater, unless the environment
        variable 'CUBLAS_WORKSPACE_CONFIG=:4096:8' or 'CUBLAS_WORKSPACE_CONFIG=:16:8' is set. See the CUDA documentation 
        for more details: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility"
    """
    
    random.seed(_seed_)
    np.random.seed(_seed_)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG (Random Number Generator) for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'      # ←
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    
    worker_seed = torch.initial_seed() % int(np.power(2, 32))     # this needs to be changed for Win OS due to int limit

    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AverageMeter():  
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



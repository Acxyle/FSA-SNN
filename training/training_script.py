#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 22:39:36 2023

@author: fangwei123456
@modified: acxyle

"""

# --- python basic
import os

# ---local
from training_lite import SP_Trainer_ANN, SP_Trainer_SNN, training_parser


# ----------------------------------------------------------------------------------------------------------------------
def training_script_parser():
    
    # --- basic config for NN training
    parser = training_parser()
    
    # --- env config
    parser.add_argument("--data_dir", type=str, default='/home/acxyle-workstation/Dataset')
    parser.add_argument("--dataset", type=str, default='CelebA_fold_0')
    parser.add_argument("--hierarchy", type=str, default='tv')
    parser.add_argument("--num_classes", type=int, default=2622)
   
    sub_parser = parser.add_subparsers(dest="command", help='Sub-command help')
    
    # --- ANN config
    parser_ANN = sub_parser.add_parser('ANN', help='training for ANN')
    parser_ANN.add_argument("-m", "--model", type=str, default='vgg16')
    
    # --- SNN config
    parser_SNN = sub_parser.add_parser('SNN', help='training for SNN')
    parser_SNN.add_argument("-m", "--model", type=str, default='spiking_vgg16_bn')
    parser_SNN.add_argument("--step_mode", type=str, default='m')
    parser_SNN.add_argument('--neuron', type=str, default='IF')
    parser_SNN.add_argument('--surrogate', type=str, default='ATan')
    parser_SNN.add_argument("--T", type=int, default=4)
    
    args = parser.parse_args()
    args.data_path = os.path.join(args.data_dir, args.dataset)
    
    if args.command == 'ANN':
        args.output_dir = os.path.join(f"./logs_{args.model}_{args.dataset}")
    elif args.command == 'SNN':
        args.output_dir = os.path.join(f"./logs_{args.model}_{args.neuron}_{args.surrogate}_T{args.T}_{args.dataset}")
    else:
        raise ValueError
    
    return args


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    args = training_script_parser()
    
    # -----
    if args.command == 'ANN':
        trainer = SP_Trainer_ANN(args)
    elif args.command == 'SNN':
        trainer = SP_Trainer_SNN(args)

    trainer.train(args)
                    




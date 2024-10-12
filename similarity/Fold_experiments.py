#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:56:06 2024

@author: acxyle
"""

import os
import argparse

import FSA_ANOVA, FSA_Encode, FSA_RSA, FSA_CKA, FSA_SVM

import sys
sys.path.append('../')
import utils_


# ======================================================================================================================
def get_args_parser(fold_idx=0):
    parser = argparse.ArgumentParser(description="Face Selectivity Analysis", add_help=True)
    
    parser.add_argument("--num_classes", type=int, default=50, help="set the number of classes")
    parser.add_argument("--num_samples", type=int, default=10, help="set the sample number of each class")
    parser.add_argument("--alpha", type=float, default=0.01, help='assign the alpha value for ANOVA')
    
    parser.add_argument("--FSA_root", type=str, default="/home/acxyle-workstation/Downloads/FSA-ImageNet", help="root directory for features and neurons")
    
    parser.add_argument("--target_type", type=str, default='act')
    
    parser.add_argument("--FSA_dir", type=str, default='VGG')
    parser.add_argument("--FSA_config", type=str, default='Baseline_ImageNet1k')
    parser.add_argument("--fold_idx", type=int, default=f'{fold_idx}')

    parser.add_argument("--model", type=str, default='vgg16')     
    
    return parser.parse_args()


# FIXME
class Multi_Model_Analysis():
    
    def __init__(self, args, num_folds=5, **kwargs):

        self.num_folds = num_folds
        self.root = os.path.join(args.FSA_root, args.FSA_dir, f'FSA {args.FSA_config}')
        
        self.model_structure = args.FSA_config.replace('C2k_fold_', '')
        
        _, self.layers, self.neurons, self.shapes = utils_.get_layers_and_units(args.model, 'act')
        
        # 
        self.used_unit_types = [
                                'qualified', 'a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne', 'non_anova', 
                                'a_s', 'a_m', 'high_selective', 'low_selective', 'selective', 'non_selective'
                                ]
        
        self.used_id_nums = [50, 10]
        
        self._configs = {
            'num_folds': self.num_folds,
            'root': self.root,
            'layers': self.layers,
            'neurons': self.neurons
            }
        
        # --- 
        self.Encode_folds()
        self.RSA_folds()
        self.CKA_folds()
        self.SVM_folds()
        
        
    def ANOVA_folds(self, **kwargs):
    
        FSA_ANOVA.FSA_ANOVA_folds(**self._configs)()
        
        
    def Encode_folds(self, **kwargs):
        
        FSA_Encode_folds = FSA_Encode.FSA_Encode_folds(**self._configs)
        FSA_Encode_folds(**kwargs)
        
        
    def RSA_folds(self, **kwargs):

        FSA_RSA.RSA_Monkey_folds(**self._configs)()
        
        RSA_Human_folds = FSA_RSA.RSA_Human_folds(**self._configs)
        for used_unit_type in self.used_unit_types:
            for used_id_num in self.used_id_nums:
                RSA_Human_folds(used_unit_type=used_unit_type, used_id_num=used_id_num)
        
        for used_id_num in self.used_id_nums:
            RSA_Human_folds.process_all_used_unit_results(used_id_num=used_id_num, used_unit_types=self.used_unit_types)
        
    
    def CKA_folds(self, **kwargs):
        
        FSA_CKA.CKA_Similarity_Monkey_folds(**self._configs)()
        
        CKA_Human_folds = FSA_CKA.CKA_Similarity_Human_folds(**self._configs)
        for used_unit_type in self.used_unit_types:
            for used_id_num in self.used_id_nums:
                CKA_Human_folds(used_unit_type=used_unit_type, used_id_num=used_id_num)
        
        for used_id_num in self.used_id_nums:
            CKA_Human_folds.process_all_used_unit_results(used_id_num=used_id_num, used_unit_types=self.used_unit_types)
            
    
    def SVM_folds(self, **kwargs):
        
        used_unit_types = [
                           'a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne',
                           'a_s', 'a_m',
                           'qualified', 
                           'non_anova', 
                           'selective', 'high_selective', 'low_selective', 'non_selective'
                           ]
        
        FSA_SVM.FSA_SVM_folds(**self._configs)(used_unit_types=used_unit_types)
        
if __name__ == "__main__":
    
    utils_.formatted_print('Face Selectivity Analysis Experiment...')
    
    args = get_args_parser()
    Multi_Model_Analysis(args)
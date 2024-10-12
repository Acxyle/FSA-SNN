#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:35:00 2023

@author: acxyle

    ...
    
    
"""

# --- python
import os
import time
import argparse

# --- local
import similarity
import utils_


# ======================================================================================================================
def universal_similarity_parser(fold_idx=0):
    parser = argparse.ArgumentParser(description="FSA Ver 5.1", add_help=True)
    
    parser.add_argument("--num_classes", type=int, default=50, help="the number of classes")
    parser.add_argument("--num_samples", type=int, default=10, help="the number of samples of each class")
    parser.add_argument("--alpha", type=float, default=0.01, help='the significance level for ANOVA test')
    
    parser.add_argument("--target_type", type=str, default='act')
    
    parser.add_argument("--FSA_root", type=str, default="/home/acxyle-workstation/Downloads/FSA")
    parser.add_argument("--FSA_dir", type=str, default='Resnet/Resnet')
    parser.add_argument("--FSA_config", type=str, default='Resnet18_C2k_fold_/runs/Resnet18_C2k_fold_0')
    parser.add_argument("--FSA_weight", type=str, default='pth_c50/checkpoint_max_test_acc1.pth')
    
    parser.add_argument("--fold_idx", type=int, default=f'{fold_idx}')

    parser.add_argument("--model", type=str, default='resnet18')     
    
    return parser.parse_args()


# ----------------------------------------------------------------------------------------------------------------------
class Face_Selectivity_Analyzer():
    """ this class only process with one model, exclude the folds experiments """

    def __init__(self, args, **kwargs) -> None:
        
        self.start_time = time.time()
        
        # --- init
        self.used_types_Similarity = ['qualified', 'a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne', 'non_anova']
        
        self.FSA_folder = os.path.join(args.FSA_root, args.FSA_dir, f'FSA {args.FSA_config}')
        
        # -----
        layers_info_generator, target_element = get_layers_info_generator_NN(args.model, **kwargs)

        self.layers, self.units, self.shapes = get_layers_info(layers_info_generator, target_element)
        
    
    def selectivity_analysis_script(self, **kwargs) -> None:
        
        start_time = time.time()
        
        # ---
        self.neuron_selection_anova(**kwargs)
        
        self.neuron_selection_encode(**kwargs)
        
        self.neuron_population_responses(**kwargs)
        
        self.neuron_population_SVM(**kwargs)
        
        self.neuron_population_RSA(**kwargs)
        
        self.neuron_population_CKA(**kwargs)
        
        # --- 
        end_time = time.time()
        elapsed = end_time - start_time
        
        utils_.formatted_print(f"All results are saved in {os.path.join(self.FSA_folder, 'Analysis')}")
        utils_.formatted_print('Elapsed Time: {}:{:0>2}:{:0>2} '.format(int(elapsed/3600), int((elapsed%3600)/60), int((elapsed%3600)%60)))
        utils_.formatted_print('Experiment Done.')    

    

    def neuron_selection_anova(self, **kwargs) -> None:
        
        FSA_ANOVA_analyzer = similarity.FSA_ANOVA(root=self.FSA_folder, 
                                                  layers=self.layers, 
                                                  units=self.units, 
                                                  alpha=args.alpha, 
                                                  num_classes=args.num_classes, 
                                                  num_samples=args.num_samples)

        FSA_ANOVA_analyzer.execute(**kwargs)
        
        
    def neuron_selection_encode(self, **kwargs) -> None:
        
        Encode_analyzer = similarity.FSA_Encode(root=self.FSA_folder, 
                                                layers=self.layers, 
                                                units=self.units)

        Encode_analyzer.calculation_Encode(**kwargs)
        Encode_analyzer.plot_Encode_pct_bar_chart(**kwargs)
        Encode_analyzer.plot_Encode_freq(**kwargs)
        
        
    def neuron_population_responses(self, **kwargs) -> None:
        
        Responses_analyzer = similarity.FSA_Responses(root=self.FSA_folder, 
                                                      layers=self.layers, 
                                                      units=self.units)

        Responses_analyzer.plot_unit_responses()
        Responses_analyzer.plot_stacked_responses(self.used_types_Similarity)
        Responses_analyzer.plot_responses_PDF()
        Responses_analyzer.plot_Feature_Intensity()
        
        
    def neuron_population_SVM(self, **kwargs) -> None:
        
        SVM_analyzer = similarity.FSA_SVM(root=self.FSA_folder, 
                                          layers=self.layers, 
                                          units=self.units)
        
        SVM_analyzer.process_SVM(**kwargs)

        
    def neuron_population_RSA(self, **kwargs) -> None:
        
        RSA_monkey_analyzer = similarity.RSA_Monkey(root=self.FSA_folder, 
                                                    layers=self.layers, 
                                                    units=self.units)
        
        RSA_monkey_analyzer(first_corr='pearson', second_corr='spearman', **kwargs)
                
        # ---
        RSA_human_analyzer = similarity.RSA_Human(root=self.FSA_folder, 
                                                  layers=self.layers, 
                                                  units=self.units)
        
        for used_unit_type in self.used_types_Similarity:
            for used_id_num in [args.num_classes, args.num_samples]:
                RSA_human_analyzer(used_unit_type=used_unit_type, used_id_num=used_id_num)

        for used_id_num in [args.num_classes, args.num_samples]:
            RSA_human_analyzer.process_all_used_unit_results(used_id_num=used_id_num, used_unit_types=self.used_types_Similarity)
        
        
    def neuron_population_CKA(self, **kwargs) -> None:
        
        CKA_monkey_analyzer = similarity.CKA_Monkey(root=self.FSA_folder, 
                                                    layers=self.layers, 
                                                    units=self.units)
        
        CKA_monkey_analyzer(**kwargs)
        
        # ---
        CKA_human_analyzer = similarity.CKA_Human(root=self.FSA_folder, 
                                                  layers=self.layers, 
                                                  units=self.units)
        
        for used_unit_type in self.used_types_Similarity:
            for used_id_num in [args.num_classes, args.num_samples]:
                CKA_human_analyzer(used_unit_type=used_unit_type, used_id_num=used_id_num)
                
        for used_id_num in [args.num_classes, args.num_samples]:
            CKA_human_analyzer.process_all_used_unit_results(used_id_num=used_id_num, used_unit_types=self.used_types_Similarity)
       

def get_layers_info(layers_info_generator, target_element='an') -> None:
    
    layers, units, shapes = layers_info_generator.get_layer_names_and_units_and_shapes()
    layers, units, shapes = zip(*[(l, u, s) for l, u, s in zip(layers, units, shapes) if target_element in l])
    
    utils_.formatted_print(f'Listing model [{args.FSA_config}]')
    utils_.describe_model(layers, units, shapes)
    
    return layers, units, shapes


def get_layers_info_generator_NN(model_name, **kwargs):
    
    if 'spiking' not in model_name and 'sew' not in model_name:
        return get_layers_info_generator_ANN(model_name, **kwargs), 'an'
    else:
        return get_layers_info_generator_SNN(model_name, **kwargs), 'sn'
        

def get_layers_info_generator_ANN(model_name, **kwargs):

    if 'vgg' in model_name:
        layers_info_generator = utils_.VGG_layers_info_generator(model=model_name, **kwargs)
    elif 'resnet' in model_name:
        layers_info_generator = utils_.Resnet_layers_info_generator(model=model_name, **kwargs)
    else:
        raise ValueError

    return layers_info_generator


def get_layers_info_generator_SNN(model_name, **kwargs):

    if 'vgg' in model_name:
        layers_info_generator = utils_.SVGG_layers_info_generator(model=model_name, **kwargs)
    elif 'resnet' in model_name and 'spiking' in model_name:
        layers_info_generator = utils_.SResnet_layers_info_generator(model=model_name, **kwargs)
    elif 'resnet' in model_name and 'sew' in model_name:
        layers_info_generator = utils_.SEWResnet_layers_info_generator(model=model_name, **kwargs)
    else:
        raise ValueError

    return layers_info_generator


if __name__ == "__main__":
    
    utils_.formatted_print('Face Selectivity Analysis Experiment...')
    
    args = universal_similarity_parser()
    print(args)
    
    FSA_analyzer = Face_Selectivity_Analyzer(args)
    
    FSA_analyzer.selectivity_analysis_script()
    

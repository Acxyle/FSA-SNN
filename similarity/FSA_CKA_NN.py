#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:32:48 2024

@author: acxyle-workstation
"""

import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from joblib import Parallel, delayed
import scipy

from statsmodels.stats.multitest import multipletests
import itertools

import sys
sys.path.append('../')
import utils_
from utils_ import utils_similarity

from bio_records_process.monkey_feature_process import monkey_feature_process
from bio_records_process.human_feature_process import human_feature_process

from FSA_DRG import FSA_Gram
from FSA_Encode import FSA_Responses



# ----------------------------------------------------------------------------------------------------------------------
#FIXME
class CKA_base():
    
    def __init__(self, used_unit_types, **kwargs):
        
        self.used_unit_types = used_unit_types
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
        
        ...
        
        
    def __call__(self, **kwargs):
        
        # --- 1.
        save_path = os.path.join(self.CKA_root, f'CKA {self.N1_structure} v.s. {self.N2_structure}.pkl')
        
        if os.path.exists(save_path):
            
            cka_dict = utils_.load(save_path)
            
        else:
            
            cka_dict = self.calculation_CKA(**kwargs)
        
            utils_.dump(cka_dict, save_path)
        
        # --- 2.
        for k, v in cka_dict.items():
            
            self.plot_CKA_comprehensive(v, k, **kwargs)
            
        # --- 3.
        fig, ax = plt.subplots()
        
        self.plot_diag_CKA(fig, ax, cka_dict, **kwargs)
        
        ax.legend()
        #ax.set_xticks(np.arange(len(self.N1_layers)))
        #ax.set_xticklabels(self.N1_layers, rotation='vertical')
        ax.set_ylim(0,1.1)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        ax.set_title(f'CKA diag {self.N1_structure} v.s. {self.N2_structure}')
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.CKA_root, f'CKA diag {self.N1_structure} v.s. {self.N2_structure}.svg'), bbox_inches='tight')
        
        plt.close()
        
    
    def plot_diag_CKA(self, fig, ax, cka_dict, used_unit_type=None, color=None, label=None, **kwargs):
        
        if used_unit_type is None:
            
            for k, v in cka_dict.items():
                
                ax.plot(np.diag(v), label=k)
        
        else:
            
            ax.plot(np.diag(cka_dict[used_unit_type]), color=color, label=label)
        
        
    def calculation_CKA(self, **kwargs):
        
        product_list = list(itertools.product(self.N1_layers, self.N2_layers))
        
        return {_type: np.array([utils_similarity.cka(self.N1_G_dict[_[0]][_type], self.N2_G_dict[_[1]][_type]) for _ in product_list]).reshape(len(self.N1_layers), len(self.N2_layers)) for _type in tqdm(self.used_unit_types)}
        
    
    def plot_CKA_comprehensive(self, cka_results, _type, intensity=False, layer_ticks=True, **kwargs):
        
        if not intensity:
            
            fig, ax = plt.subplots()
           
            # === 1
            ax = plt.gcf().add_axes([0.5, 0.5, 0.5, 0.5])
            img = ax.imshow(cka_results, origin='lower', cmap='magma', aspect='auto')     # vmin=0.2, vmax=1
            
            ax.set_title(_type)
            ax.set_xlabel(f'{self.N2_structure}')
            ax.set_ylabel(f'{self.N1_structure}')
            ax.set_xticks([])
            ax.set_yticks([])
            
            fig.savefig(os.path.join(self.CKA_root, f'{_type}.svg'), bbox_inches='tight')
            plt.close()
            
        else:
            
            log_Gram_dict_N1 = utils_.load(os.path.join(self.N1_root, 'Analysis/Gram/Figures/log_Gram_dict.pkl'), verbose=False)
            log_Gram_dict_N2 = utils_.load(os.path.join(self.N2_root, 'Analysis/Gram/Figures/log_Gram_dict.pkl'), verbose=False)
            
            log_Gram_N1 = log_Gram_dict_N1[_type]
            log_Gram_N2 = log_Gram_dict_N2[_type]
            
            Intensity_dict_N1 = utils_.load(os.path.join(self.N1_root, 'Analysis/Encode/Responses/Intensity/Intensity.pkl'), verbose=False)
            units_pct_N1 = utils_.load(os.path.join(self.N1_root, 'Analysis/Encode/Responses/Intensity/units_pct.pkl'), verbose=False)
            
            Intensity_dict_N2 = utils_.load(os.path.join(self.N2_root, 'Analysis/Encode/Responses/Intensity/Intensity.pkl'), verbose=False)
            units_pct_N2 = utils_.load(os.path.join(self.N2_root, 'Analysis/Encode/Responses/Intensity/units_pct.pkl'), verbose=False)
            
            fig = plt.figure(figsize=(10, 10))
           
            ax_1 = plt.gcf().add_axes([0.5, 0.5, 0.5, 0.5])
            
            cka_results[cka_results==0] = np.nan
            
            img = ax_1.imshow(cka_results, origin='lower', cmap='magma', aspect='auto', vmin=0., vmax=1)     # vmin=0., vmax=1
            
            ax_1.set_xticks([])
            ax_1.set_yticks([])
            
            ax_2 = plt.gcf().add_axes([0., 0.5, 0.24, 0.5])
            FSA_Responses.plot_Feature_Intensity_single(ax_2, self.N1_layers, Intensity_dict_N1, _type, units_pct_N1, direction='vertical')
            
            if layer_ticks:
                ax_2.set_yticks(np.arange(len(self.N1_layers)))
                ax_2.set_yticklabels(self.N1_layers)
            else:
                ax_2.set_yticks([])
                
            ax_2.set_ylabel(f'{self.N1_structure}', fontsize=18)
            
            ax_3 = plt.gcf().add_axes([0.25, 0.5, 0.24, 0.5])
            self.plot_log_Gram_intensity_single(fig, ax_3, self.N1_layers, _type, log_Gram_N1, direction='vertical', text=False)
            
            ax_4 = plt.gcf().add_axes([0.5, 0.25, 0.5, 0.24])
            self.plot_log_Gram_intensity_single(fig, ax_4, self.N2_layers, _type, log_Gram_N2, direction='horizontal', text=False)
            
            ax_5 = plt.gcf().add_axes([0.5, 0., 0.5, 0.24])
            FSA_Responses.plot_Feature_Intensity_single(ax_5, self.N2_layers, Intensity_dict_N2, _type, units_pct_N2, direction='horizontal')
            
            if layer_ticks:
                ax_5.set_xticks(np.arange(len(self.N2_layers)))
                ax_5.set_xticklabels(self.N2_layers, rotation='vertical')
            else:
                ax_5.set_xticks([])
                
            ax_5.set_xlabel(f'{self.N2_structure}', fontsize=18)
            
            c_ax1 = fig.add_axes([1.05, 0.1, 0.03, 0.8])
            c_b1 = fig.colorbar(img, cax=c_ax1)
            c_b1.ax.tick_params(labelsize=16)
            
            # ---
            legend_lines = [
                Line2D([0], [0], color='blue', linestyle='-', linewidth=3, label='log_Gram_value'),
                Line2D([0], [0], marker='o', markersize=8, color='red', linewidth=2, label='zero_pct'),
                Line2D([0], [0], marker='d', markersize=8, markeredgecolor='coral', color='coral', linestyle='--', linewidth=2, label='units_pct'),
                Line2D([0], [0], linewidth=2, label='units_value')
            ]
            
            fig.legend(handles=legend_lines, loc='lower left', bbox_to_anchor=(0.125, 0.2))
            
            fig.suptitle(f'{_type}', y=1.075, fontsize=24)
            #fig.tight_layout()

            fig.savefig(os.path.join(self.CKA_root, f'{_type} with intensity.svg'), bbox_inches='tight')
            plt.close()
            


# ----------------------------------------------------------------------------------------------------------------------
#FIXME --- the process is complicated
class CKA_Comparison(FSA_Gram, CKA_base):
    
    def __init__(self, N1_root, N1_model, N2_root, N2_model, used_unit_types=['qualified', 'selective', 'non_selective'], **kwargs):
        
        CKA_base.__init__(self, used_unit_types=used_unit_types, **kwargs)
        
        self.N1_root = N1_root
        self.N2_root = N2_root
        
        def _load_folds(nn_root, _model, _type='act',  norm=True, **kwargs):
            
            nn_grams = utils_.load(os.path.join(nn_root, f'Analysis/Gram/Gram_linear_norm_{norm}.pkl'))
            
            _, nn_layers, nn_neurons, _ = utils_.get_layers_and_units(_model, _type)
            
            nn_grams_dict = {layer: {_: nn_grams[layer][_] for _ in used_unit_types} for layer in nn_layers}
            
            nn_structure = nn_root.split('/')[-1].split(' ')[-1].replace('ATan_', '').replace('_C2k_fold_0', '')
            
            return nn_layers, nn_neurons, nn_grams_dict, nn_structure
        
        self.N1_layers, _, self.N1_G_dict, self.N1_structure = _load_folds(N1_root, N1_model)
        self.N2_layers, _, self.N2_G_dict, self.N2_structure = _load_folds(N2_root, N2_model)
        
        self.CKA_root = os.path.join(N1_root, f'Analysis/CKA/v.s. {self.N2_structure}')
        utils_.make_dir(self.CKA_root)
        #self.N2_root = os.path.join(N2_root, 'Analysis')
    
    

# ----------------------------------------------------------------------------------------------------------------------
class CKA_Comparison_folds(FSA_Gram, CKA_base):
    
    def __init__(self, N1_root, N1_model, N2_root, N2_model, used_unit_types=['qualified', 'strong_selective', 'weak_selective', 'non_selective'], num_folds=5, **kwargs):
        
        CKA_base.__init__(self, used_unit_types=used_unit_types, **kwargs)
        
        def _load_folds(nn_root, _model, _type='act', **kwargs):
            
            nn_grams = utils_.load(os.path.join(nn_root, 'CKA/Grams_linear.pkl'))
            
            nn_layers, nn_neurons, _ = utils_.get_layers_and_units(_model, _type)
            
            nn_grams_dict = {layer: {_: np.mean([nn_grams[fold_idx][layer][_] for fold_idx in range(num_folds)], axis=0) for _ in self.used_unit_types} for layer in nn_layers}
            
            nn_structure = nn_root.split('/')[-1].split(' ')[-1]
            
            return nn_layers, nn_neurons, nn_grams_dict, nn_structure

        self.N1_layers, _, self.N1_G_dict, self.N1_structure = _load_folds(N1_root, N1_model)
        self.N2_layers, _, self.N2_G_dict, self.N2_structure = _load_folds(N2_root, N2_model)
        
        self.CKA_root = os.path.join(N1_root, f'CKA/v.s. {self.N2_structure}')
        utils_.make_dir(self.CKA_root)
        #self.N2_root = N2_root
        

# ======================================================================================================================
# local debug
if __name__ == '__main__':
    
    FSA_root = '/home/acxyle-workstation/Downloads/FSA'
    
    used_unit_types = ['qualified', 'sensitive', 'non_sensitive', 'selective', 'strong_selective', 'weak_selective', 'non_selective', 's_non_encode']
    
    # --- 3.
    CKA_Comparison(
        N1_root=os.path.join(FSA_root, 'VGG/VGG/FSA Baseline'), N1_model='vgg16',
        N2_root=os.path.join(FSA_root, 'VGG/A2S_VGG/FSA A2S_Baseline(T64)'), N2_model='spiking_vgg16',
        used_unit_types=used_unit_types
        )(intensity=True, layer_ticks=False)
    

    #FIXME
# =============================================================================
#     # --- 4.
#     ANN_vs_SNN = CKA_Comparison_folds(
#         N1_root=os.path.join(root_dir, 'Face Identity VGG16bn_fold_'), N1_model='vgg16_bn',
#         N2_root=os.path.join(root_dir, 'Face Identity SpikingVGG16bn_LIF_T16_CelebA2622_fold_'), N2_model='spiking_vgg16_bn',
#         used_unit_types=used_unit_types)
#     ANN_vs_SNN()
# =============================================================================
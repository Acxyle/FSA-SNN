#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 22:02:44 2023

@author: Jinge Wang, Runnan Cao
@modified: acxyle

    refer to: https://github.com/JingeW/ID_selective
              https://osf.io/824s7/
    
    DRG: Dimensional redution, Representational similarity matrix, Gram matrix
    
"""


import os
import math
import warnings
import logging
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA

import sys
sys.path.append('../')
import utils_

from utils_ import utils_similarity
from similarity.FSA_Encode import FSA_Encode


# ----------------------------------------------------------------------------------------------------------------------
class FSA_DR(FSA_Encode):
    """
        ...
        
        TSNE only
    """
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.dest_DR = os.path.join(self.dest, 'Dimensional Reduction')
        utils_.make_dir(self.dest_DR)
         
    
    #def DR_PCA(self, ):
        ...
    
    
    def DR_TSNE(self, used_unit_types=['qualified', 'strong_selective', 'weak_selective', 'non_selective'], start_layer_idx=-5, sub_width=6, sub_height=6, **kwargs):
        """
            ...
            key=tsne_coordinate has been removed, please manually normalize if want to visualize
        """
        
        utils_.formatted_print('Executing selectivity_analysis_Tsne...')

        self.save_path_DR = os.path.join(self.dest_DR, 'TSNE')
        utils_.make_dir(self.save_path_DR)
        
        self.Sort_dict = self.load_Sort_dict()
        self.Sort_dict = self.calculation_Sort_dict(used_unit_types) if used_unit_types is not None else self.Sort_dict

        # --- 1. calculation
        TSNE_dict = self.calculation_TSNE(used_unit_types=used_unit_types, start_layer_idx=start_layer_idx, **kwargs)
        
        # --- 2. plot
        label = np.repeat(np.arange(self.num_classes)+1, self.num_samples)
        
        valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if not item[1].startswith('not')])
        markers = valid_markers + valid_markers[:self.num_classes - len(valid_markers)]
        
        fig, ax = plt.subplots(np.abs(start_layer_idx), len(used_unit_types), figsize=(len(used_unit_types)*sub_width, np.abs(start_layer_idx)*sub_height), dpi=100)
        
        self.plot_TSNE(fig, ax, TSNE_dict, label, markers, **kwargs)
        
        fig.savefig(os.path.join(self.save_path_DR, 'TSNE.svg'))
        plt.close()
        
        
    def calculation_TSNE(self, used_unit_types=['qualified', 'strong_selective', 'weak_selective', 'non_selective'], start_layer_idx=-5, **kwargs):

        save_path = os.path.join(self.save_path_DR, 'TSNE_dict.pkl')
        
        if os.path.exists(save_path):
            
            TSNE_dict = utils_.load(save_path)
        
        else:

            # ---
            TSNE_dict = {}
            
            for layer in self.layers[start_layer_idx:]:
                
                feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), **kwargs)

                TSNE_dict[layer] = {k: calculation_TSNE(feature[:, mask], **kwargs) for k, mask in self.Sort_dict[layer].items()}
                
            utils_.dump(TSNE_dict, save_path)
            
        return TSNE_dict
        

    # -----
    def plot_TSNE(self, fig, ax, TSNE_dict, label, markers, num_classes=50, num_samples=10, **kwargs):   
        
        def _plot_scatter(ax, idx, tsne, **kwargs):
            
            tsne = tsne.T
            
            try:

                tsne_y = tsne[1] if tsne.shape[0] == 2 else np.zeros_like(tsne[0])
                
                ax.scatter(tsne[0], tsne_y, label.reshape(num_classes, num_samples)[idx], marker=markers[idx])
                    
            except AttributeError as e:
                
                if "'NoneType' object has no attribute 'shape'" in str(e):
                    pass
                else:
                    raise
                    

        for row_idx, (layer, tsne_dict) in enumerate(TSNE_dict.items()):
            
            for col_idx, (k, v) in enumerate(tsne_dict.items()):
            
                if v is not None:
                    
                    tsne_x_min = np.min(v[:,0])
                    tsne_x_max = np.max(v[:,0])
                    tsne_y_min = np.min(v[:,1])
                    tsne_y_max = np.max(v[:,1])
                        
                    w_radius = tsne_x_max - tsne_x_min
                    h_radius = tsne_y_max - tsne_y_min
                    
                    for idx, v_ in enumerate(v.reshape(num_classes, num_samples, -1)):     # for each class
                        _plot_scatter(ax[row_idx, col_idx], idx, v_, **kwargs)
                        
                    ax[row_idx, col_idx].set_xlim((tsne_x_min-0.025*w_radius, tsne_x_min+1.025*w_radius))
                    ax[row_idx, col_idx].set_ylim((tsne_y_min-0.025*h_radius, tsne_y_min+1.025*h_radius))
                    
                    ax[row_idx, col_idx].vlines(0, tsne_x_min-0.5*w_radius, tsne_x_min+1.5*w_radius, colors='gray',  linestyles='--', linewidth=2.0)
                    ax[row_idx, col_idx].hlines(0, tsne_y_min-0.5*h_radius, tsne_y_min+1.5*h_radius, colors='gray',  linestyles='--', linewidth=2.0)
                    
                    pct = self.Sort_dict[layer][k].size/self.neurons[self.layers.index(layer)] *100
                    
                    ax[row_idx, col_idx].set_title(f'{layer} {k}\n {self.Sort_dict[layer][k].size}/{self.neurons[self.layers.index(layer)]} ({pct:.2f}%)')
                    ax[row_idx, col_idx].grid(False)
        
        fig.suptitle(f'{self.model_structure}', y=0.995, fontsize=30)
        plt.tight_layout()


def calculation_perplexity(mask, num_classes=50, num_samples=10, **kwargs):
    """
        this function use the smaller value of the number of features and number of total samples as the perplexity
    """

    mask = len(mask) if isinstance(mask, list) else mask
 
    return min(math.sqrt(mask), num_classes*num_samples-1) if mask > 0 else 1.


def calculation_TSNE(input: np.array, **kwargs):
    """
        ...
        b) a commonly used way is to reduce the dimension firstly by PCA before the tSNE, the disadvantage is the 
    dimensions after PCA can not exceeds min(n_classes, n_features))
        c) according to TSNE authors (Maaten and Hinton), they suggested to try different values of perplexity, to 
    see the trade-off between local and glocal relationships
        ...
    """

    # --- method 1, set a threshold for data size
    # ...
    # --- method 2, use PCA to reduce all feature as (500,500)
    #test_value = int(self.num_classes*self.num_samples)     
    #if input[:, mask].shape[1] > test_value:     
    #    np_log = math.ceil(test_value*(math.log(len(mask)/test_value)+1.))
    #    pca = PCA(n_components=min(test_value, np_log))
    #    x = pca.fit_transform(input[:, mask])
    #    tsne = TSNE(perplexity=perplexity, n_jobs=-1).fit_transform(x)    
    # --- method 3, manually change the SWAP for large data
    # ...

    if input.size == 0 or np.std(input) == 0.:
        return None
    
    if input.shape[1] == 1:
        return np.repeat(input, 2, axis=1)
    
    perplexity = calculation_perplexity(input.shape[1])
    
    return TSNE(perplexity=perplexity, n_jobs=-1).fit_transform(input)
    

# ----------------------------------------------------------------------------------------------------------------------
class FSA_DSM(FSA_Encode):
    """
       ...
    """
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.dest_DSM = os.path.join(self.dest, 'DSM')
        utils_.make_dir(self.dest_DSM)


    def process_DSM(self, metric='pearson', plot=False, **kwargs):
        """
            ...
        """
        
        utils_.formatted_print(f'Executing DSM {metric} of {self.model_structure}')

        # ----- 
        DSM_dict = self.calculation_DSM(metric, **kwargs)
        
        # ----- 
        self.plot_DSM(metric, DSM_dict, **kwargs)
           
  
    def calculation_DSM(self, metric='pearson', used_unit_types=None, **kwargs) -> dict:
        """
            ...
        """
        
        used_unit_types = self.basic_types_display + self.advanced_types_display + ['a_s', 'a_m'] if used_unit_types is None else used_unit_types

        save_path = os.path.join(self.dest_DSM, f'{metric}.pkl')
        
        if os.path.exists(save_path):
            
            DSM_dict = utils_.load(save_path, verbose=False)
            
        else:
            
            self.Sort_dict = self.calculation_Sort_dict(used_unit_types)
            
            DSM_dict = {}    

            for layer in tqdm(self.layers, desc=f'{self.model_structure} DSM({metric})'):     # for each layer

                feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), verbose=False, **kwargs)     # (500, num_samples)
                feature = np.mean(feature.reshape(self.num_classes, self.num_samples, -1), axis=1)     # (50, num_samples)
                
                pl = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(utils_similarity.DSM_calculation)(feature[:, self.Sort_dict[layer][k].astype(int)], metric, **kwargs) for k in used_unit_types)
                
                DSM_dict[layer] = {k: pl[idx] for idx, k in enumerate(used_unit_types)}
                
            utils_.dump(DSM_dict, save_path, verbose=True)

        return DSM_dict
    

    def plot_DSM(self, metric, DSM_dict, used_unit_types, vlim:tuple=None, cmap='turbo', **kwargs) -> None:

        # ----- not applicable for all metrics
        metric_dict_ = {layer:{k: v if v is not None else None for k,v in dsm_dict.items()} for layer, dsm_dict in DSM_dict.items()}     # assemble all types of all layers
        metric_dict_pool = np.concatenate([_ for _ in [np.concatenate([v for k,v in dsm_dict.items() if v is not None]).reshape(-1) for layer, dsm_dict in metric_dict_.items()]])   # in case of inhomogeneous shape
        metric_dict_pool = np.nan_to_num(metric_dict_pool, 0)
        
        vlim = (np.percentile(metric_dict_pool, 5), np.percentile(metric_dict_pool, 95)) if vlim is None else vlim

        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        fig, ax = plt.subplots(len(self.layers), len(used_unit_types), figsize=(3*len(used_unit_types), 3*len(self.layers)))

        for row_idx, (layer, dsm_dict) in enumerate(DSM_dict.items()):     # for each layer
            
            for col_idx, k in enumerate(used_unit_types):     # for each type of cells
                
                if row_idx == 0: ax[row_idx, col_idx].set_title(k)
                if col_idx == 0: ax[row_idx, col_idx].set_ylabel(layer)
                
                if (DSM:=dsm_dict[k]) is not None:
                    
                    ax[row_idx, col_idx].imshow(DSM, origin='lower', aspect='auto', vmin=vlim[0], vmax=vlim[1], cmap=cmap)
                    ax[row_idx, col_idx].set_xlabel(f"{self.Sort_dict[layer][k].size/(self.neurons[self.layers.index(layer)])*100:.2f}%")
                    
                    cax = fig.add_axes([1.01, 0.1, 0.01, 0.75])
                    norm = plt.Normalize(vmin=vlim[0], vmax=vlim[1])
                    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                    
                else:
                    
                    ax[row_idx, col_idx].axis('off')
                        
                ax[row_idx, col_idx].set_xticks([])
                ax[row_idx, col_idx].set_yticks([])
                
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.1, hspace=0.1)
        
        fig.suptitle(f'{self.model_structure} | {metric}', y=0.995, fontsize=50)
         
        fig.tight_layout()
        fig.savefig(os.path.join(self.dest_DSM, f'{self.model_structure}.png'), bbox_inches='tight')
        
        plt.close()

    
# ----------------------------------------------------------------------------------------------------------------------
class FSA_Gram(FSA_Encode):

    def __init__(self, used_unit_types=None, **kwargs):
        
        super().__init__(**kwargs)
        
        used_unit_types = self.basic_types_display + self.advanced_types_display + ['a_s', 'a_m'] if used_unit_types is None else used_unit_types

        self.dest_Gram = os.path.join(self.dest, 'Gram')
        utils_.make_dir(self.dest_Gram)
       
        
    def calculation_Gram(self, kernel='linear', normalize=True, **kwargs):
        
        
        if kernel == 'rbf' and 'threshold' in kwargs:
            save_path = os.path.join(self.dest_Gram, f"Gram_{kernel}_{kwargs['threshold']}_norm_{normalize}.pkl")
        elif kernel == 'linear':
            save_path = os.path.join(self.dest_Gram, f"Gram_{kernel}_norm_{normalize}.pkl")
        else:
            raise ValueError
        
        if os.path.exists(save_path):
            
            Gram_dict = utils_.load(save_path, verbose=False)
            
        else:
            
            self.Sort_dict = self.load_Sort_dict()
            self.Sort_dict = self.calculation_Sort_dict(self.used_unit_types) if self.used_unit_types is not None else self.Sort_dict
            
            def _calculation_Gram(layer, normalize, **kwargs):
                
                feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), normalize=normalize, verbose=False, **kwargs)     # (500, num_samples)
                feature = np.mean(feature.reshape(self.num_classes, self.num_samples, -1), axis=1)     # (50, num_samples)
                
                # --- 
                if kernel == 'linear':
                    gram = utils_similarity.gram_linear
                elif kernel =='rbf':
                    gram = utils_similarity.gram_rbf
                    
                # ---
                pl = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(gram)(feature[:, self.Sort_dict[layer][k].astype(int)], **kwargs) for k in self.used_unit_types)

                metric_type_dict = {k: pl[idx] for idx, k in enumerate(self.used_unit_types)}

                return metric_type_dict
            
            utils_.formatted_print(f'Calculating NN_unit_Gram of {self.model_structure}...')
            
            Gram_dict = {_:_calculation_Gram(_, normalize, **kwargs) for _ in tqdm(self.layers, desc='NN Gram')}
        
            utils_.dump(Gram_dict, save_path)
            
        return Gram_dict
    
    
    def plot_Gram(self, ):
        """
            plot the Gram
        """
        
        ...
        
    
    def calculation_Gram_intensity(self, kernel, **kwargs):
        
        utils_.make_dir(save_root:=os.path.join(self.dest_Gram, 'Figures'))
        
        save_path = os.path.join(save_root, 'log_Gram_dict.pkl')
        
        if os.path.exists(save_path):
            
            log_Gram_dict = utils_.load(save_path)
        
        else:
        
            Gram_dict = self.calculation_Gram(normalize=True)
            Gram_dict = {k: [Gram_dict[_][k] for _ in self.layers] for k in self.used_unit_types}
            
            log_Gram_dict = {}
            
            for k,v in Gram_dict.items():
                
                log_mean = np.zeros(len(self.layers))
                log_std = log_mean.copy()
                zero_pct = log_mean.copy()
                
                for idx, _ in enumerate(v):
                    
                    log_values = np.log(_[_!=0])/np.log(10)
                    log_mean[idx] = np.mean(log_values)
                    log_std[idx] = np.std(log_values)
                    
                    zero_pct[idx] = np.sum(_==0)/_.size*100
                
                log_Gram_dict[k] = {
                    'log_mean': log_mean,
                    'log_std': log_std,
                    'zero_pct': zero_pct
                    }
                
            utils_.dump(log_Gram_dict, save_path)
            
        return log_Gram_dict
    

    def plot_Gram_intensity(self, kernel='linear', **kwargs):
        """
            plot the log intensity for different types of units
        """
        
        utils_.make_dir(save_root:=os.path.join(self.dest_Gram, 'Figures'))
        
        log_Gram_dict = self.calculation_Gram_intensity(kernel, **kwargs)
        
        for k,v in log_Gram_dict.items():
            
            fig, ax = plt.subplots(figsize=(10, 3))
            
            self.plot_log_Gram_intensity_single(fig, ax, self.layers, k, v)
            
            fig.savefig(os.path.join(save_root, f'{self.model_structure} {k}.svg'), bbox_inches='tight')
            plt.close()
        
        ...
        
    @staticmethod
    def plot_log_Gram_intensity_single(fig, ax, layers, used_unit_type, log_Gram_stats, direction='horizontal', text=True, **kwargs):
        
        log_mean, log_std, zero_pct = log_Gram_stats['log_mean'], log_Gram_stats['log_std'], log_Gram_stats['zero_pct']
        
        x = np.arange(len(layers))
        
        if direction == 'horizontal':
        
            ax.plot(x, log_mean, color='blue', label='Gram (log)')
            ax.fill_between(x, log_mean-log_std, log_mean+log_std, color='blue', alpha=0.5)
            ax.set_ylabel("Gram Values", color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            ax.set_xticks([])
            
            ax.hlines(0, 0, len(layers), colors='skyblue', linestyle='--', alpha=0.5)
            ax.set_xlim([0, len(layers)-1])
            
            ax2 = ax.twinx()
            ax2.plot(x, zero_pct, label='0%', color='red', marker='.')
            ax2.set_ylim([0, 105])
            ax2.set_ylabel("%", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            if text:
                ax.set_title(f"Gram Intensity {used_unit_type}")
                fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.1))
            
        elif direction == 'vertical':
            
            ax.plot(log_mean, x, color='blue')
            ax.fill_betweenx(x, log_mean-log_std, log_mean+log_std, color='blue', alpha=0.5)
            ax.set_xlabel("Gram Values", color='blue')
            ax.tick_params(axis='x', labelcolor='blue')
            ax.set_yticks([])
            
            ax.vlines(0, 0, len(layers), colors='skyblue', linestyle='--', alpha=0.5)
            ax.set_ylim([0, len(layers)-1])
            ax.invert_xaxis()
            
            ax2 = ax.twiny()
            ax2.plot(zero_pct, x, color='red', marker='.')
            
            ax2.set_xlim([0, 105])
            ax2.set_xlabel("%", color='red')
            ax2.tick_params(axis='x', labelcolor='red')
            ax2.invert_xaxis()
            
            
 
# ======================================================================================================================
if __name__ == '__main__':

    FSA_root = '/home/acxyle-workstation/Downloads/FSA'
    FSA_dir = 'VGG/SpikingVGG'
    model_depth = 16
    T = 8
    FSA_config = f'SpikingVGG{model_depth}bn_IF_ATan_T4_C2k_fold_'
    FSA_model =  f'spiking_vgg{model_depth}_bn'
    
    _, layers, neurons, _ = utils_.get_layers_and_units(FSA_model, 'act')
    
    # -----
    #DR_analyzer = FSA_DR(root=os.path.join(root_dir, 'Face Identity Baseline'), layers=layers, neurons=neurons)
    #DR_analyzer.DR_TSNE()
    
    for _ in range(4,5):
        
        root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}/-_Single Models/FSA {FSA_config}{_}')
        #root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
    
        DSM_analyzer = FSA_DSM(root=root, layers=layers, neurons=neurons)
        DSM_analyzer.calculation_DSM()
    
        Gram_analyzer = FSA_Gram(root=root, layers=layers, neurons=neurons)
        Gram_analyzer.calculation_Gram()
        Gram_analyzer.plot_Gram_intensity()
    
    #Gram_analyzer.plot_Gram_intensity(kernel='linear')
    #for threshold in [0.5, 1.0, 2.0, 10.0]:
    #    Gram_analyzer.calculation_Gram(kernel='rbf', threshold=threshold)
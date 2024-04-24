#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 22:02:44 2023

@author: acxyle

    FSA: Face Similarity Analysis
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

import utils_
from utils_ import utils_similarity
from FSA_Encode import FSA_Encode

import sys
sys.path.append('../')
import models_


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


    def process_DSM(self, metric, used_unit_types=['qualified', 'selective', 'strong_selective', 'weak_selective', 'non_selective'], plot:bool=False, **kwargs):
        """
            ...
        """
        
        utils_.formatted_print(f'Executing DSM {metric} of {self.model_structure}')

        # ----- 
        DSM_dict = self.calculation_DSM(metric, used_unit_types, **kwargs)
        
        # ----- 
        self.plot_DSM(metric, DSM_dict, used_unit_types, **kwargs)
           
  
    def calculation_DSM(self, metric, used_unit_types=['qualified', 'selective', 'strong_selective', 'weak_selective', 'non_selective'], **kwargs):
        """
            ...
        """

        utils_.make_dir(metric_folder:=os.path.join(self.dest_DSM, f'{metric}'))
        
        self.Sort_dict = self.load_Sort_dict()
        self.Sort_dict = self.calculation_Sort_dict(used_unit_types) if used_unit_types is not None else self.Sort_dict
        
        dict_path = os.path.join(metric_folder, f'{metric} {used_unit_types}.pkl')
        
        if os.path.exists(dict_path):
            
            DSM_dict = utils_.load(dict_path, verbose=False)
            
        else:
            
            # ----- 
            DSM_dict = {}     # use a dict to store the info of each layer

            for layer in tqdm(self.layers, desc=f'NN {metric} DSM'):     # for each layer

                feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), verbose=False, **kwargs)     # (500, num_samples)
                feature = np.mean(feature.reshape(self.num_classes, self.num_samples, -1), axis=1)     # (50, num_samples)
                
                pl = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(utils_similarity.DSM_calculation)(metric, feature[:, self.Sort_dict[layer][k].astype(int)], **kwargs) for k in used_unit_types)
                
                DSM_dict[layer] = {k: pl[idx] for idx, k in enumerate(used_unit_types)}
                
            utils_.dump(DSM_dict, dict_path, verbose=True)

        return DSM_dict
    

    def plot_DSM(self, metric, DSM_dict, used_unit_types, vlim:tuple=None, cmap='turbo', **kwargs):

        # ----- not applicable for all metrics
        metric_dict_ = {layer:{k: v if v is not None else None for k,v in dsm_dict.items()} for layer, dsm_dict in DSM_dict.items()}     # assemble all types of all layers
        metric_dict_pool = np.concatenate([_ for _ in [np.concatenate([v for k,v in dsm_dict.items() if v is not None]).reshape(-1) for layer, dsm_dict in metric_dict_.items()]])   # in case of inhomogeneous shape
        metric_dict_pool = np.nan_to_num(metric_dict_pool, 0)
        
        vlim = (np.percentile(metric_dict_pool, 5), np.percentile(metric_dict_pool, 95)) if vlim is None else vlim

        utils_.make_dir(metric_folder:=os.path.join(self.dest_DSM, f'{metric}'))
        
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
        fig.savefig(os.path.join(metric_folder, f'{self.model_structure}.png'), bbox_inches='tight')
        
        plt.close()

    
# ----------------------------------------------------------------------------------------------------------------------
class FSA_Gram(FSA_Encode):

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.dest_Gram = os.path.join(self.dest, 'Gram')
        utils_.make_dir(self.dest_Gram)
       
        
    def calculation_Gram(self, kernel='linear', used_unit_types=['qualified', 'selective', 'strong_selective', 'weak_selective', 'non_selective'], normalize=False, **kwargs):
        """
            [notice] unlike RSA, the normalize = True or False will influence the Gram
        """
        
        self.Sort_dict = self.load_Sort_dict()
        self.Sort_dict = self.calculation_Sort_dict(used_unit_types) if used_unit_types is not None else self.Sort_dict
        
        if kernel == 'rbf' and 'threshold' in kwargs:
            save_path = os.path.join(self.dest_Gram, f"Gram_{kernel}_{kwargs['threshold']}_norm_{normalize}.pkl")
        elif kernel == 'linear':
            save_path = os.path.join(self.dest_Gram, f"Gram_{kernel}_norm_{normalize}.pkl")
        else:
            raise ValueError(f'[Coderror] Invalid kernel [{kernel}]')
        
        if os.path.exists(save_path):
            
            Gram_dict = utils_.load(save_path)
            
        else:
            
            def _calculation_Gram(layer, normalize, **kwargs):
                
                feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), normalize=normalize, verbose=False, **kwargs)     # (500, num_samples)
                feature = np.mean(feature.reshape(self.num_classes, self.num_samples, -1), axis=1)     # (50, num_samples)
                
                # --- 
                if kernel == 'linear':
                    gram = utils_similarity.gram_linear
                elif kernel =='rbf':
                    gram = utils_similarity.gram_rbf
                    
                # ---
                pl = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(gram)(feature[:, self.Sort_dict[layer][k].astype(int)], **kwargs) for k in used_unit_types)

                metric_type_dict = {k: pl[idx] for idx, k in enumerate(used_unit_types)}

                return metric_type_dict
            
            utils_.formatted_print(f'Calculating NN_unit_Gram of {self.model_structure}...')
            
            Gram_dict = {_:_calculation_Gram(_, normalize, **kwargs) for _ in tqdm(self.layers, desc='NN Gram')}
        
            utils_.dump(Gram_dict, save_path)
            
        return Gram_dict
        
 
# ======================================================================================================================
if __name__ == '__main__':

    layers, neurons, shapes = utils_.get_layers_and_units('vgg16', target_layers='act')
    root_dir = '/home/acxyle-workstation/Downloads/'

    # -----
    #DR_analyzer = FSA_DR(root=os.path.join(root_dir, 'Face Identity Baseline'), layers=layers, neurons=neurons)
    #DR_analyzer.DR_TSNE()
    
    #DSM_analyzer = FSA_DSM(root=os.path.join(root_dir, 'Face Identity Baseline'), layers=layers, neurons=neurons)
    #DSM_analyzer.process_DSM(metric='pearson')
    
    Gram_analyzer = FSA_Gram(root=os.path.join(root_dir, 'Face Identity Baseline'), layers=layers, neurons=neurons)
    Gram_analyzer.calculation_Gram(kernel='linear', normalize=True)
    #for threshold in [0.5, 1.0, 2.0, 10.0]:
    #    Gram_analyzer.calculation_Gram(kernel='rbf', threshold=threshold)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:53:33 2023

@author: Jinge Wang, Runnan Cao
@modified: acxyle

    refer to: https://github.com/JingeW/ID_selective
              https://osf.io/824s7/
    

"""

# --- python
import os
import math
import warnings
#import logging
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# --- local
import utils_


# ----------------------------------------------------------------------------------------------------------------------
__all__ = ["FSA_ANOVA", "FSA_ANOVA_folds", "FSA_ANOVA_Comparison"]

plt.rcParams.update({'font.size': 18})    
plt.rcParams.update({"font.family": "Times New Roman"})


# ----------------------------------------------------------------------------------------------------------------------
class FSA_ANOVA():
    """ the 'Features' contains all processed firing rate for img, not spike train """
    
    def __init__(self, root='./FSA Baseline', layers=None, units=None, num_classes=50, num_samples=10, alpha=0.01, **kwargs):
        
        self.root = os.path.join(root, 'Features')     # <- folder for feature maps, which should be generated in advance
        self.dest = os.path.join(root, 'Analysis')    # <- folder for analysis results
        utils_.make_dir(self.dest)
        
        self.dest_ANOVA = os.path.join(self.dest, 'ANOVA')
        utils_.make_dir(self.dest_ANOVA)
        
        self.layers = layers
        self.units = units
        
        self._check_layers()
        
        # ---
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_samples = num_samples
        
        self.model_structure = root.split('/')[-1].split(' ')[-1]     # in current code, the 'root' file should list those information in structural name, arbitrary
    
    
    def _check_layers(self, ):     # --- running check for units inside the loop by default
        
        pkls_set = {os.path.splitext(_)[0] for _ in os.listdir(self.root)}
        layers_set = set(self.layers)
        
        assert pkls_set == layers_set, "The layers and features are mismatch"
    
    def execute(self, **kwargs):
        
        # --- 1. 
        self.calculation_ANOVA(**kwargs)
        
        ratio_dict = self.calculation_ANOVA_pct(**kwargs)
        
        # --- 2.
        fig, ax = plt.subplots(figsize=(math.floor(len(self.layers)/1.6), 6))
        
        title = f"Sensitive pct {self.model_structure}"
        
        self.plot_ANOVA_pct(fig, ax, ratio_dict, title=title, plot_bar=True, **kwargs)
        
        fig.savefig(os.path.join(self.dest_ANOVA, f'{title}.svg'), bbox_inches='tight')
        plt.close()
    

    def calculation_ANOVA(self, normalize=True, sort=True, num_workers=-1, **kwargs):
        """
            normalize: if True, normalize the feature map
            sort: if True, sort the featuremap from lexicographic order (pytorch) into natural order
            num_workers: parallel workers for joblib
        """
        
        utils_.formatted_print('Executing calculation_ANOVA')
        
        idces_path = os.path.join(self.dest_ANOVA, 'ANOVA_idces.pkl')
        stats_path = os.path.join(self.dest_ANOVA, 'ANOVA_stats.pkl')
        
        if os.path.exists(idces_path) and os.path.exists(stats_path):
            self.ANOVA_idces = self.load_ANOVA_idces()
            self.ANOVA_stats = self.load_ANOVA_stats()
        
        else:
            self.ANOVA_idces = {}
            self.ANOVA_stats = {}     # <- p_values
            
            for idx, layer in enumerate(self.layers):     # for each layer
    
                feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), normalize=normalize, sort=sort, verbose=False, **kwargs)

                if feature.shape[0] != self.num_classes*self.num_samples or feature.shape[1] != self.units[idx]:     # running check
                    raise AssertionError('[Coderror] feature.shape[0] ({}) != self.num_classes*self.num_samples ({},{}) or feature.shape[1] ({}) != self.units[idx] ({})'.format(feature.shape[0], self.num_classes, self.num_samples, feature.shape[1], self.units[idx]))
                
                # ----- joblib
                pl = Parallel(n_jobs=num_workers)(delayed(one_way_ANOVA)(feature[:, i]) for i in tqdm(range(feature.shape[1]), desc=f'ANOVA [{layer}]'))
    
                neuron_idx = np.array([idx for idx, p in enumerate(pl) if p < self.alpha])
            
                self.ANOVA_stats[layer] = pl
                self.ANOVA_idces[layer] = neuron_idx
            
            utils_.dump(self.ANOVA_idces, idces_path)
            utils_.dump(self.ANOVA_stats, stats_path)
            
            utils_.formatted_print('ANOVA results have been saved in {}'.format(self.dest_ANOVA))
            
            
    def calculation_ANOVA_pct(self, ANOVA_path=None, **kwargs):
        
        ratio_path = os.path.join(self.dest_ANOVA, 'ratio.pkl') if ANOVA_path == None else ANOVA_path
        
        if os.path.exists(ratio_path):
            
            ratio_dict = utils_.load(ratio_path, verbose=False)
            
        else:
            
            self.ANOVA_idces = self.load_ANOVA_idces()
            
            ratio_dict = {layer: self.ANOVA_idces[layer].size/self.units[idx]*100 for idx, layer in enumerate(self.layers)}
                  
            utils_.dump(ratio_dict, ratio_path)     
            
        return ratio_dict
            
            
    def plot_ANOVA_pct(self, fig, ax, ratio_dict, title='sensitive ratio', line_color=None, plot_bar=False, **kwargs):      
        
        utils_.formatted_print('Executing plot_ANOVA_pct...')
        
        # -----
        _, pcts = zip(*ratio_dict.items())
        
        # -----
        if line_color is None:
            line_color = 'red'
        
        if plot_bar:
            colors = color_column(self.layers)
            plot_ANOVA_pct(ax, self.layers, pcts, bar_colors=colors, line_color=line_color, linewidth=1.5, **kwargs)
        
        else:
            plot_ANOVA_pct(ax, self.layers, pcts, line_color=line_color, linewidth=1.5, **kwargs)
            
        ax.set_title(title)
        ax.set_xticks(np.arange(len(self.layers)))
        ax.set_xticklabels(self.layers, rotation='vertical')
        ax.set_ylabel('percentage (%)')
        ax.set_ylim([0, 100])
        ax.legend()
        
    
    def load_ANOVA_idces(self, ANOVA_idces_path=None):
        if not ANOVA_idces_path:
            ANOVA_idces_path = os.path.join(self.dest_ANOVA, 'ANOVA_idces.pkl')
        return utils_.load(ANOVA_idces_path)
        
    
    def load_ANOVA_stats(self, ANOVA_stats_path=None):
        if not ANOVA_stats_path:
            ANOVA_stats_path = os.path.join(self.dest_ANOVA, 'ANOVA_stats.pkl')
        return utils_.load(ANOVA_stats_path)
    
            
# ----------------------------------------------------------------------------------------------------------------------
def one_way_ANOVA(input, num_classes=50, num_samples=10, **kwargs):
    """
        if all values are 0, this will return 'nan' F_value and 'nan' p_value, nan values will be filtered out in following 
        selection with threshold 0.01
    """
    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore')
        
        d = list(input.reshape(num_classes, num_samples))
        p = stats.f_oneway(*d)[1]     # [0] for F-value, [1] for p-value

    return p


# ----------------------------------------------------------------------------------------------------------------------
def plot_ANOVA_pct(ax, layers, pcts, bar_colors=None, line_color=None, linewidth=2.5, label=None, **kwargs):
    
    if bar_colors is not None:
        ax.bar(layers, pcts, color=bar_colors, width=0.5)
     
    if label == None:
        label='sensitive units'
        
    ax.plot(np.arange(len(layers)), pcts, color=line_color, linestyle='-', linewidth=linewidth, alpha=1, label=label)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
    
     
# ----------------------------------------------------------------------------------------------------------------------
def color_column(layers, constant_colors=False, **kwargs):
    """ randomly generate colors """
    
    layers_t = []
    color = []
    
    for layer in layers:
        layers_t.append(layer.split('_')[0])
    layers_t = list(set(layers_t))

    if not constant_colors:
        for item in range(len(layers_t)):
            color.append((np.random.random(), np.random.random(), np.random.random()))
    else:
        assert len(layers_t) == 5
        color = ['teal', 'red', 'orange', 'lightskyblue', 'tomato']     # ['bn', 'fc', 'avgpool', 'activation', 'conv']
    
    layers_c_dict = {}
    for i in range(len(layers_t)):
        layers_c_dict[layers_t[i]] = color[i]     

    layers_color_list = []
        
    for layer in layers:
        layers_color_list.append(layers_c_dict[layer.split('_')[0]])

    return layers_color_list


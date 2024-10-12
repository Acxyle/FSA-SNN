#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 05:22:54 2024

@author: acxyle-workstation
"""

# --- python
import os
import scipy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#from matplotlib.lines import Line2D

# --- local
import utils_
from .FSA_Encode import FSA_Encode

# ----------------------------------------------------------------------------------------------------------------------
__all__ = ["FSA_SVM", "FSA_SVM_folds", "FSA_SVM_Comparison"]

plt.rcParams.update({'font.size': 18})    
plt.rcParams.update({"font.family": "Times New Roman"})


# ----------------------------------------------------------------------------------------------------------------------
class FSA_SVM(FSA_Encode):
    """ the default SVM kernel is RBF """
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.dest_SVM = os.path.join(self.dest, 'SVM')
        utils_.make_dir(self.dest_SVM)
        
        ...
        
    
    def process_SVM(self, **kwargs):
        
        # ----- calculation
        SVM_results = self.calculation_SVM(**kwargs)
        
        # ----- plot
        fig, ax = plt.subplots(figsize=(10,6))
        
        self.plot_SVM(ax, SVM_results, **kwargs)
        
        ax.set_title(title:=f'SVM {self.model_structure}')
        fig.savefig(os.path.join(self.dest_SVM, f'{title}.svg'), bbox_inches='tight')
        plt.close()
    
    
    def calculation_SVM(self, used_unit_types=None, **kwargs):
        """
            ...
        """
        
        utils_.formatted_print(f'computing SVM {self.model_structure}...')
        
        if used_unit_types == None:
            
            used_unit_types = self.basic_types_display + self.advanced_types_display + ['a_s', 'a_m']
            
        if os.path.exists(SVM_path:=os.path.join(self.dest_SVM, f'SVM {self.model_structure}.pkl')):
            
            SVM_results = utils_.load(SVM_path)
            
        else:
            
            # --- init
            self.Sort_dict = self.load_Sort_dict()
            Sort_dict = self.calculation_Sort_dict(used_unit_types)
            
            SVM_results = {}
            
            for layer in tqdm(self.layers, desc=f'SVM {self.model_structure}'):
                
                # --- depends
                feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), verbose=False, **kwargs)

                # ---
                SVM_results[layer] = {k: calculation_SVM(feature[:, v], np.repeat(np.arange(self.num_classes), self.num_samples)) for k,v in Sort_dict[layer].items()}
            
            SVM_results = {_: np.array([v[_] for k,v in SVM_results.items()]) for _ in used_unit_types}
            
            utils_.dump(SVM_results, SVM_path, verbose=False)

        return SVM_results
            
    
    def plot_SVM(self, ax, SVM_results, color=None, label=None, ncol=2, smooth=True, text=False, **kwargs):
        """
            ...
        """
        
        # --- init
        SVM_type_conifg = self.plot_Encode_config
        types_to_plot = ['qualified', 'a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne', 'non_anova']

        # --- all
        for k, v in SVM_results.items():
            
            if k in types_to_plot:
            
                plot_config = SVM_type_conifg.loc[k]
                
                if smooth:
                    SVM_results = scipy.ndimage.gaussian_filter(SVM_results[k], sigma=1)
                else:
                    SVM_results = SVM_results[k]
                
                if color is None and label is None:
                    _color = plot_config['color']
                    ax.plot(SVM_results, color=_color, linestyle=plot_config['linestyle'], label=k)
                else:
                    ax.plot(SVM_results, color=color, linestyle=plot_config['linestyle'], label=label)
            
        # -----
        if text:
            ax.set_xticks(np.arange(len(self.layers)))
            ax.set_xticklabels(self.layers, rotation='vertical')
        else:
            ax.set_xticks([0, len(self.layers)-1])
            ax.set_xticklabels([0, 1])
        
        ax.set_xlim([0, len(self.layers)-1])
        ax.set_ylim([0, 100])
        ax.set_yticks(np.arange(1, 109, 10))
        ax.set_yticklabels(np.arange(0, 109, 10))
        
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        ax.legend(ncol=ncol, framealpha=0.5)
        

def calculation_SVM(input, label, **kwargs):
    """ no default k-fold implemented here, change the internal dataset division if needed """
    
    return utils_.SVM_classification(input, label, test_size=0.2, random_state=42, **kwargs) if input.size != 0 else 0.


# ----------------------------------------------------------------------------------------------------------------------
class FSA_SVM_folds(FSA_SVM):
    """
        ...
    """
    
    def __init__(self, num_folds=5, root=None, **kwargs):
        super().__init__(root=root, **kwargs)
        
        self.root = root
        
        self.num_folds = num_folds
        
        ...
    
    
    def __call__(self, **kwargs):
        
        # ---
        SVM_folds = self.calculation_SVM_folds(**kwargs)
        
        # ---
        fig, ax = plt.subplots(figsize=(len(self.layers)/2, 4))
        
        self.plot_SVM_folds(fig, ax, SVM_folds, **kwargs)
        
        title = f"SVM {self.model_structure.replace('_fold_', '')}"
        #ax.set_title(title=title)
        fig.savefig(os.path.join(self.dest_SVM, f'{title}.svg'), bbox_inches='tight')
        
        plt.close()
    
    
    def calculation_SVM_folds(self, used_unit_types=None, SVM_path=None, **kwargs):
        
        if used_unit_types == None:
            used_unit_types = ['qualified', 'a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne', 'non_anova']
 
        SVM_folds_path = os.path.join(self.dest_SVM, f'SVM_folds {used_unit_types}.pkl') if SVM_path == None else SVM_path              
        
        if os.path.exists(SVM_folds_path):
            
            SVM_folds = utils_.load(SVM_folds_path, verbose=False)
        
        else:
            
            SVM_folds = self.collect_SVM_folds(used_unit_types, **kwargs)

            SVM_folds = {k: np.array([SVM_folds[fold_idx][k] for fold_idx in range(self.num_folds)]) for k in used_unit_types} 
            SVM_folds = {stat: {k: getattr(np, stat)(v, axis=0) for k, v in SVM_folds.items()} for stat in ['mean', 'std']}

            utils_.dump(SVM_folds, SVM_folds_path)
        
        return SVM_folds
    
    
    def collect_SVM_folds(self, used_unit_types, SVM_path=None, **kwargs):
        
        SVM_folds = {}

        for fold_idx in np.arange(self.num_folds):
            
            _FSA_config = self.root.split('/')[-1]
            
            SVM_folds[fold_idx] = utils_.load(os.path.join(self.root, f"-_Single Models/{_FSA_config}{fold_idx}/Analysis/SVM/SVM {used_unit_types}.pkl"), verbose=False)

        return SVM_folds
        
        
    def plot_SVM_folds(self, fig, ax, SVM_folds, color=None, label=None, ncol=2, used_unit_types=None, smooth=True, text=False, **kwargs):
        
        SVM_type_conifg = self.plot_Encode_config
        #types_to_plot = ['qualified', 'strong_selective', 'weak_selective', 's_non_encode', 'non_sensitive']
        types_to_plot = ['qualified', 'a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne', 'non_anova']
        
        for k, v in SVM_folds['mean'].items():
            
            if k in types_to_plot:
            
                plot_config = SVM_type_conifg.loc[k]
                
                if smooth:
                    means = scipy.ndimage.gaussian_filter(v, sigma=1)
                    stds = scipy.ndimage.gaussian_filter(SVM_folds['std'][k], sigma=1)
                else:
                    means = v
                    stds = SVM_folds['std'][k]
            
                if color is None and label is None:
                    _color = plot_config['color']
                    ax.plot(means, color=_color, linestyle=plot_config['linestyle'], label=k)
                else:
                    ax.plot(means, color=color, linestyle=plot_config['linestyle'], label=label)
    
                ax.fill_between(np.arange(len(self.layers)), means-stds, means+stds, edgecolor=None, facecolor=utils_.lighten_color(utils_.color_to_hex(_color), 20), alpha=0.5, **kwargs)

        # -----
        if text:
            ax.set_xticks(np.arange(len(self.layers)))
            ax.set_xticklabels(self.layers, rotation='vertical')
        else:
            ax.set_xticks([0, len(self.layers)-1])
            ax.set_xticklabels([0, 1])
        
        ax.set_ylim([0, 10])
        ax.set_yticks(np.arange(0, 109, 10))
        ax.set_yticklabels(np.arange(0, 109, 10))
        
        ax.set_xlim([0, len(self.layers)-1])
        
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        #ax.legend(ncol=ncol, framealpha=0.5)
        

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
        

# ----------------------------------------------------------------------------------------------------------------------
class FSA_SVM_Comparison(FSA_SVM_folds):
    """
        unlike ANOVA and Encode, SVM results depend on specific computation thus load a complete dict then extract values
    """
    
    def __init__(self, **kwargs):
        
        # ---
        self.color_pool = ['blue', 'green', 'red', 'purple', 'orange', 'chocolate']     # manually change the pool
        
    def __call__(self, roots_and_models, used_unit_types, **kwargs):
        """
            single unit type now
        """
        
        assert len(used_unit_types) == 1

        dummy_types = ['qualified', 'strong_selective', 'weak_selective', 'non_selective']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        title = []
        
        for idx, (root, model) in enumerate(roots_and_models):
            
            super().__init__(root=roots_and_models[idx][0], **kwargs)
            self.layers, self.units, _ = utils_.get_layers_and_units(roots_and_models[idx][1], target_layers='act')
            
            if 'fold' in root:
                
                ratio_dict = self.calculation_SVM_folds(used_unit_types=dummy_types, SVM_path=os.path.join(root, f'Analysis/Encode/SVM/SVM_folds {dummy_types}.pkl'))
                
                ratio_dict = {_: {used_unit_types[0]: ratio_dict[_][used_unit_types[0]]} for _ in ['acc', 'std']}
                
                _label = root.split('/')[-1].split(' ')[-1].replace('_fold_', '').replace('_CelebA2622', '')
                title.append(_label)
                
                self.plot_SVM_folds(fig, ax, ratio_dict, used_unit_types=used_unit_types, color=self.color_pool[idx], label=_label)
                
                ...
                
            else:
                
                self.Sort_dict = self.load_Sort_dict()
                units_pct = self.calculation_units_pct(dummy_types, **kwargs)
                ratio_dict = self.calculation_curve_dict(units_pct, Encode_path=os.path.join(root, f'Analysis/Encode/ratio_curve_dict {dummy_types}.pkl'), **kwargs)
                
                _label=root.split('/')[-1].split(' ')[-1]
                title.append(_label)
                
                self.plot_units_pct(fig, ax, self.layers, ratio_dict, color=self.color_pool[idx], label=_label)
                ...


        ax.set_title(title:=f'{used_unit_types} SVM acc '+' v.s '.join(title))
        #ax.set_title(title:=f'{used_unit_types} SVM acc ANN v.s SNN ')
        
        # --- setting
        ...
        # ---
        
        fig.tight_layout()
        fig.savefig(os.path.join(roots_and_models[0][0], f'Analysis/Encode/SVM/Comparison {title}.svg'))
        
        plt.close()
        
        ...
        

# ======================================================================================================================
if __name__ == "__main__":
    
    FSA_root = '/home/acxyle-workstation/Downloads/FSA'
    FSA_dir = 'VGG/SpikingVGG'
    model_depth = 16
    T = 4
    FSA_config = f'SpikingVGG{model_depth}bn_IF_ATan_T4_C2k_fold_'
    FSA_model =  f'spiking_vgg{model_depth}_bn'
    
    used_unit_types1 = ['qualified', 'sensitive', 'non_sensitive', 'selective', 'strong_selective', 'weak_selective', 'non_selective', 's_non_encode']
    used_unit_types2 = ['s_si', 's_wsi', 's_mi', 's_wmi', 'non_selective']
    used_unit_types = used_unit_types1 + used_unit_types2
    
    ttk1 = ['qualified', 'anova', 'non_anova', 'selective', 'high_selective', 'low_selective', 'non_selective', 'a_ne']
    ttk2 = ['a_hs', 'a_ls', 'a_hm', 'a_lm', 'non_selective']
    ttk = ttk1 + ttk2
    
    # ----- (1). Encode
    _, layers, units, shapes = utils_.get_layers_and_units(FSA_model, 'act')
    
    #root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
    #selectivity_analyzer = FSA_SVM(root=root, layers=layers, units=units)
    #selectivity_analyzer.process_SVM(used_unit_types=used_unit_types)
    
# =============================================================================
#     for fold_idx in range(5):
#         
#         root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}/-_Single Models/FSA {FSA_config}{fold_idx}')
#         #root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
#         
#         tmp1 = utils_.load(os.path.join(root, f'Analysis/SVM/SVM {used_unit_types}.pkl'))
#         #tmp2 = utils_.load(os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}/Analysis/SVM/SVM {ttk2}.pkl'))
#         #tmp3 = {**tmp1, **tmp2}
#         
#         tmp3 = {}
#         for _ in range(len(ttk)):
#             tmp3[ttk[_]] = tmp1[used_unit_types[_]]
#         
#         utils_.dump(tmp3, os.path.join(root, f'Analysis/SVM/SVM {ttk}.pkl'))
#         
#         #os.remove(os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}/-_Single Models/FSA {FSA_config}{fold_idx}/Analysis/SVM/SVM {ttk1}.pkl'))
#         #os.remove(os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}/-_Single Models/FSA {FSA_config}{fold_idx}/Analysis/SVM/SVM {ttk2}.pkl'))
#         
#         selectivity_analyzer = FSA_SVM(root=root, layers=layers, units=units)
#         selectivity_analyzer.process_SVM(used_unit_types=ttk)
# =============================================================================
        
    # ----- (3). SVM
    # --- 1. Folds
    root=os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
    FSA_SVM_folds(num_folds=5, root=root, layers=layers, units=units)(used_unit_types=ttk)
    
    # --- 2. Multi Models Comparison
    #FSA_SVM_Comparison()(roots_and_models)

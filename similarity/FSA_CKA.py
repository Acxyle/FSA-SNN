#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:47:40 2024

@author: acxyle

"""

# --- python
import os
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from joblib import Parallel, delayed

import scipy
from statsmodels.stats.multitest import multipletests

# --- local
import utils_
from utils_ import utils_similarity

from bio_records_process.monkey_feature_process import monkey_feature_process
from bio_records_process.human_feature_process import human_feature_process

from similarity.FSA_DRG import FSA_Gram


# ----------------------------------------------------------------------------------------------------------------------
__all__ = ["CKA_Monkey", "CKA_Monkey_folds", 
           "CKA_Human", "CKA_Human_folds",
           
           "CKA_Monkey_Comparison",
           "CKA_Human_Comparison"]

plt.rcParams.update({'font.size': 16})    
plt.rcParams.update({"font.family": "Times New Roman"})

# ----------------------------------------------------------------------------------------------------------------------
class CKA_Similarity_base():
    
    def __init__(self, ):
        
        self.ts
        
        self.dest_primate
        self.layers
        
        self.primate_DM
        self.primate_DM_perm
        
        # --- if use permutation every time, those two can be ignored, 5-fold experiment is suggested
        # --- empirically, the max fluctuation of mean scores between experiments could be ±0.03 with num_perm = 1000
        self.primate_DM_temporal
        self.primate_DM_temporal_perm
    
    
    def calculation_CKA_Similarity(self, primate=None, kernel='linear', alpha=0.05, FDR_method='fdr_bh', used_unit_type='qualified', used_id_num=50, **kwargs):
        """ calculation and save the CKA results for monkey """
        
        utils_.make_dir(dest_primate:=os.path.join(self.dest_CKA,  f'{primate}'))
        
        if primate == 'Monkey':
 
            self.dest_primate = dest_primate
            utils_.make_dir(self.dest_primate)
            
        elif primate == 'Human':
            
            self.dest_primate = os.path.join(dest_primate, used_unit_type, str(used_id_num))
            utils_.make_dir(self.dest_primate)
            
        else:
            
            raise ValueError
        
        if kernel == 'rbf' and 'threshold' in kwargs:
            save_path = os.path.join(self.dest_primate, f"CKA_results_{kernel}_{kwargs['threshold']}.pkl")
        elif kernel == 'linear':
            save_path = os.path.join(self.dest_primate, f"CKA_results_{kernel}.pkl")
        else:
            raise ValueError
        
        if os.path.exists(save_path):
            
            cka_dict = utils_.load(save_path, verbose=False)
            
        else:
            
            def _calculation_CKA(layer, **kwargs):    

                corr_coef = utils_similarity.cka(self.primate_Gram, self.NN_Gram_dict[layer], **kwargs)
                corr_coef_perm = np.array([utils_similarity.cka(_, self.NN_Gram_dict[layer], **kwargs) for _ in self.primate_Gram_perm])
                
                if np.isnan(corr_coef):
                    p_perm = np.nan
                else:
                    p_perm = np.mean(corr_coef_perm > corr_coef)     # equal to: np.sum(corr_coef_perm > corr_coef)/num_perm,
                
                # --- temporal
                corr_coef_temporal = utils_similarity.cka_temporal(self.primate_Gram_temporal, self.NN_Gram_dict[layer], **kwargs)     # (time_steps, )
                corr_coef_temporal_perm = np.array([utils_similarity.cka_temporal( _, self.NN_Gram_dict[layer], **kwargs) for _ in self.primate_Gram_temporal_perm])
                
                p_perm_temporal = np.array([np.mean(corr_coef_temporal_perm[:, _] > corr_coef_temporal[_]) if not np.isnan(corr_coef_temporal[_]) else np.nan for _ in range(len(corr_coef_temporal))])
                
                # ---
                return {
                    'corr_coef': corr_coef,
                    'corr_coef_perm': corr_coef_perm,
                    'p_perm': p_perm,     
                    
                    'corr_coef_temporal': corr_coef_temporal,
                    'corr_coef_temporal_perm': corr_coef_temporal_perm,
                    'p_perm_temporal': p_perm_temporal
                    }
            
            #for layer in self.layers[-2:]:
            #    _calculation_CKA(layer, **kwargs)
            
            pl = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(_calculation_CKA)(layer, **kwargs) for layer in tqdm(self.layers, desc=f'CKA {primate}'))
            
            pl_k = ['corr_coef', 'corr_coef_perm', 'p_perm', 'corr_coef_temporal', 'corr_coef_temporal_perm', 'p_perm_temporal']
        
            assert set(pl[0].keys()) == set(pl_k)
            
            similarity, similarity_perm, similarity_p, similarity_temporal, similarity_temporal_perm, similarity_temporal_p = [np.array([_[__] for _ in pl]) for __ in pl_k]
            
            # --- static
            (sig_FDR, p_FDR, alpha_Sadik, alpha_Bonf) = multipletests(similarity_p, alpha=alpha, method=FDR_method)    # FDR (flase discovery rate) correction
            sig_Bonf = p_FDR<alpha_Bonf
            
            # --- temporal
            p_temporal_FDR = np.zeros((len(self.layers), self.primate_Gram_temporal.shape[0]))     # (num_layers, num_time_steps)
            sig_temporal_FDR, sig_temporal_Bonf =  np.zeros_like(p_temporal_FDR), np.zeros_like(p_temporal_FDR)
            
            for _ in range(len(self.layers)):
                
                (sig_temporal_FDR[_, :], p_temporal_FDR[_, :], alpha_Sadik_temporal, alpha_Bonf_temporal) = multipletests(similarity_temporal_p[_, :], alpha=alpha, method=FDR_method)      # FDR
                sig_temporal_Bonf[_, :] = p_temporal_FDR[_, :]<alpha_Bonf_temporal     # Bonf correction
            
            # --- seal results
            cka_dict = {
                'similarity': similarity,
                'similarity_perm': similarity_perm,
                'similarity_p': similarity_p,
                
                'similarity_temporal': similarity_temporal,
                'similarity_temporal_perm': similarity_temporal_perm,
                'similarity_temporal_p': similarity_temporal_p,
                
                'p_FDR': p_FDR,
                'sig_FDR': sig_FDR,
                'sig_Bonf': sig_Bonf,
                
                'p_temporal_FDR': p_temporal_FDR,
                'sig_temporal_FDR': sig_temporal_FDR,
                'sig_temporal_Bonf': sig_temporal_Bonf,
                }
                
            utils_.dump(cka_dict, save_path, verbose=False)

        return cka_dict
    
    
    def plot_CKA_comprehensive(self, ax, CKA_dict, EC='sig_FDR', stats=True, ticks=None, **kwargs):
        """
            ...
        """
        
        similarity = CKA_dict['similarity']
        similarity_mask = CKA_dict[EC]
        similarity_perm = CKA_dict['similarity_perm']
        
        similarity_std = CKA_dict['similarity_std']  if 'similarity_std' in CKA_dict.keys() else None
        
        plot_CKA_comprehensive(ax, similarity, similarity_std=similarity_std, similarity_mask=similarity_mask, similarity_perm=similarity_perm, **kwargs)
        
        if ticks:
            ax.set_xticks(np.arange(len(self.layers)))
            ax.set_xticklabels(self.layers, rotation='vertical')
        else:
            ax.set_xlim([0, len(self.layers)-1])
            ax.set_xticks([0, len(self.layers)-1])
            ax.set_xticklabels([0, 1])
            
        if stats:
            utils_similarity.fake_legend_describe_numpy(ax, similarity, similarity_mask, **kwargs)


    def plot_CKA_temporal_comprehensive(self, fig, ax, CKA_dict, EC='sig_temporal_Bonf', vlim=None, stats=True, ticks=None, **kwargs):
        """
            ...
        """

        similarity = CKA_dict['similarity_temporal']
        similarity_mask = CKA_dict[EC]
    
        plot_CKA_temporal_comprehensive(fig, ax, similarity, similarity_mask=similarity_mask, **kwargs)
        
        ax.set_xlabel('Time (ms)', fontsize=18)
        ax.set_ylabel('Layers', fontsize=18)
        ax.tick_params(axis='both', labelsize=12)
        
        if ticks:
            ax.set_yticks(np.arange(len(self.layers)))
            ax.set_yticklabels(self.layers)
        else:
            ax.set_yticks([0, len(self.layers)-1])
            ax.set_yticklabels([0, 1])
        
        if stats:
            utils_similarity.fake_legend_describe_numpy(ax, CKA_dict['similarity_temporal'], CKA_dict[EC].astype(bool), **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
class CKA_Monkey(monkey_feature_process, FSA_Gram, CKA_Similarity_base):
    """
        ...
        
        CKA results is not invariant to normalization process      
    """
    
    def __init__(self,  **kwargs):
        
        monkey_feature_process.__init__(self, **kwargs)
        FSA_Gram.__init__(self, **kwargs)
 
        self.dest_CKA = os.path.join(self.dest, 'CKA')
        
        
    def __call__(self, **kwargs):
        
        cka_dict = self.calculation_CKA_Monkey( **kwargs)
        
        self.plot_CKA_Monkey(cka_dict, **kwargs)
        
        
    def calculation_CKA_Monkey(self, **kwargs):
    
        # --- monkey init
        self.primate_Gram, self.primate_Gram_temporal, self.primate_Gram_perm, self.primate_Gram_temporal_perm = self.calculation_Gram_perm_monkey(**kwargs)
        
        # --- NN init --- monkey only use the entire cells/units
        self.NN_Gram_dict = {k:v['qualified'] for k,v in self.calculation_Gram(**kwargs).items()}
        
        # ----- calculation
        cka_dict = self.calculation_CKA_Similarity(primate='Monkey', **kwargs)
        
        return cka_dict


    def plot_CKA_Monkey(self, cka_dict, kernel='linear', **kwargs):
        """
            ...
        """
        
        # --- init
        if kernel == 'rbf' and 'threshold' in kwargs:
            title_static = f"CKA {self.model_structure} {kernel} {kwargs['threshold']}"
            title_temporal = f"CKA temporal {self.model_structure} {kernel} {kwargs['threshold']}"
        elif kernel == 'linear':
            title_static = f'CKA {self.model_structure} {kernel}'
            title_temporal = f'CKA temporal {self.model_structure} {kernel}'
        else:
            raise ValueError
        
        # 1. static
        fig, ax = plt.subplots(figsize=(10,6))
        
        self.plot_CKA_comprehensive(ax, cka_dict, **kwargs)
        ax.set_title(title_static)
        
        fig.tight_layout(pad=1)
        fig.savefig(os.path.join(self.dest_primate, f'{title_static}.svg'), bbox_inches='tight')   
        plt.close()
        
        # 2. temporal
        fig, ax = plt.subplots(figsize=(10, 6))
        
        self.plot_CKA_temporal_comprehensive(fig, ax, cka_dict, **kwargs)
        ax.set_title(title_temporal)
        
        ax.set_xticks([0, 5, 10, 15, 20, 25])
        ax.set_xticklabels([-50, 0, 50, 100, 150, 200])
        
        fig.savefig(os.path.join(self.dest_primate, f'{title_temporal}.svg'), bbox_inches='tight')
        plt.close()
        

# ----------------------------------------------------------------------------------------------------------------------
class CKA_Human(human_feature_process, FSA_Gram, CKA_Similarity_base):
    """
        ...
    """
    
    def __init__(self, **kwargs):
        
        # ---
        human_feature_process.__init__(self, **kwargs)
        FSA_Gram.__init__(self, **kwargs)
        
        self.dest_CKA = os.path.join(self.dest, 'CKA')
        self.save_root_primate = os.path.join(self.dest_CKA, 'Human')
        utils_.make_dir(self.save_root_primate)
        
    
    def __call__(self, **kwargs):
        
        # --- additional parameters

        cka_dict = self.calculation_CKA_Human(**kwargs)
        
        self.plot_CKA_Human(cka_dict, **kwargs)
    
    
    def calculation_CKA_Human(self, kernel='linear', used_unit_type='qualified', used_id_num=50, **kwargs):
        
        # --- init
        utils_.formatted_print(f'CKA | {self.model_structure} | {kernel} | {used_unit_type} | {used_id_num}')
        
        utils_.make_dir(save_root_cell_type:=os.path.join(self.save_root_primate, used_unit_type))
        
        self.dest_primate = os.path.join(save_root_cell_type, str(used_id_num))
        utils_.make_dir(self.dest_primate)

        used_id = self.calculation_subIDs(used_id_num)
        NN_Gram_dict = self.calculation_Gram(kernel=kernel, **kwargs)
        
        if used_unit_type == 'legacy':
            Gram, Gram_temporal, Gram_perm, Gram_temporal_perm = self.calculation_Gram_perm_human(kernel, used_unit_type='selective', used_id_num=used_id_num, **kwargs)
            self.NN_Gram_dict = {_: NN_Gram_dict[_]['strong_selective'][np.ix_(used_id, used_id)] for _ in NN_Gram_dict.keys()}
        else:     
            Gram, Gram_temporal, Gram_perm, Gram_temporal_perm = self.calculation_Gram_perm_human(kernel, used_unit_type=used_unit_type, used_id_num=used_id_num,**kwargs)
            self.NN_Gram_dict = {_: np.nan_to_num(NN_Gram_dict[_][used_unit_type][np.ix_(used_id, used_id)]) for _ in NN_Gram_dict.keys()}
            
        self.primate_Gram = Gram
        self.primate_Gram_temporal = np.array([_ for _ in Gram_temporal])
        self.primate_Gram_perm = np.array([_ for _ in Gram_perm])
        self.primate_Gram_temporal_perm = np.array([np.array([__ for __ in _]) for _ in Gram_temporal_perm])
        
        # ----- calculation
        cka_dict = self.calculation_CKA_Similarity(kernel=kernel, used_unit_type=used_unit_type, used_id_num=used_id_num, primate='Human', **kwargs)
        
        return cka_dict
        
    
    def plot_CKA_Human(self, cka_dict, kernel='linear', used_unit_type='qualified', used_id_num=50, **kwargs):
        
        # --- init
        if kernel == 'rbf' and 'threshold' in kwargs:
            title_static = f"CKA {self.model_structure} {used_unit_type} {used_id_num} {kernel} {kwargs['threshold']}"
            title_temporal = f"CKA T {self.model_structure} {used_unit_type} {used_id_num} {kernel} {kwargs['threshold']}"
        elif kernel == 'linear':
            title_static = f'CKA {self.model_structure} {used_unit_type} {used_id_num} {kernel}'
            title_temporal = f'CKA T {self.model_structure} {used_unit_type} {used_id_num} {kernel}'
        else:
            raise ValueError
        
        # 1. static
        fig, ax = plt.subplots(figsize=(10,6))
        
        self.plot_CKA_comprehensive(ax, cka_dict, **kwargs)
        ax.set_title(title_static)
        
        fig.tight_layout(pad=1)
        fig.savefig(os.path.join(self.dest_primate, f'{title_static}.svg'), bbox_inches='tight')   
        plt.close()
        
        # 2. temporal
        fig, ax = plt.subplots(figsize=(10, 6))
        
        self.plot_CKA_temporal_comprehensive(fig, ax, cka_dict, **kwargs)
        ax.set_title(title_temporal)
        
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.set_xticklabels([-250, 0, 250, 500, 750, 1000, 1250])
        
        fig.savefig(os.path.join(self.dest_primate, f'{title_temporal}.svg'), bbox_inches='tight')
        plt.close()

    
    def process_all_used_unit_results(self, used_id_num=50, **kwargs):
        
        CKA_types_dict = self.collect_all_used_unit_results(used_id_num, **kwargs)
        
        # --- static
        fig, ax = plt.subplots(figsize=(len(self.layers)/2, 4))
        
        self.plot_collect_all_used_unit_results(fig, ax, CKA_types_dict, used_id_num)

    
    def collect_all_used_unit_results(self, used_id_num=50, used_unit_types=None, **kwargs):
        
        CKA_types_dict = {}
        
        for used_unit_type in used_unit_types:
        
            CKA_types_dict[used_unit_type] = self.calculation_CKA_Similarity(used_unit_type=used_unit_type, used_id_num=used_id_num, primate='Human', **kwargs)
        
        return CKA_types_dict
    
    
    def plot_collect_all_used_unit_results(self, fig, ax, CKA_types_dict, used_id_num=50, used_unit_types=None, text=False, **kwargs):
        
        for k, v in CKA_types_dict.items():
            
            similarity = np.nan_to_num(v['similarity'])
            similarity_std = np.nan_to_num(v['similarity_std']) if 'similarity_std' in v else None
            
            color = self.plot_Encode_config.loc[k]['color']
            label = self.plot_Encode_config.loc[k]['label']
            linestyle = self.plot_Encode_config.loc[k]['linestyle']
            
            plot_CKA(ax, similarity, similarity_std=similarity_std, color=color, linestyle=linestyle, label=label, smooth=True)
        
        if text:
            ax.legend()
            ax.set_title(f'{self.model_structure} used_id_num: {used_id_num}')
        
        ax.hlines(0, 0, len(self.layers)-1, color='gray', linestyle='--', alpha=0.25)
        
        ax.set_xticks([0, len(self.layers)-1])
        ax.set_xticklabels([0, 1])
        ax.set_xlim([0, len(self.layers)-1])
        #ax.set_ylim([-0.025, 0.4])
        
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        
        #colors = ['#000000', '#0000FF', '#00BFFF', '#FF4500',]
        #labels = ['All', 'a_hs', 'a_ls', 'a_hm']
        #legend_patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
        #ax.legend(handles=legend_patches, ncol=4, loc='upper left', bbox_to_anchor=(0., -0.075), handletextpad=0.5, columnspacing=1.95)

        fig.tight_layout(pad=1)
        fig.savefig(os.path.join(self.dest_CKA, f'{self.model_structure} CKA results types {used_id_num}.svg'), bbox_inches='tight')
        plt.close()


# ----------------------------------------------------------------------------------------------------------------------
def plot_CKA_comprehensive(ax, similarity, used_id_num=None, debiased=True, **kwargs):
    """
        ...
    """
    
    plot_CKA(ax, similarity, **kwargs)

    # --- common plot setting
    ax.set_ylabel("Spearman's $\\rho$")
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    
    ...


def plot_CKA(ax, similarity, similarity_std=None, similarity_mask=None, similarity_perm=None, color=None, smooth=True, **kwargs):
    
    # -- init
    color = 'blue' if color is None else color

    plot_x = np.arange(len(similarity))
    
    if smooth:
        similarity = scipy.ndimage.gaussian_filter(similarity, sigma=1)
        if similarity_std is not None:
            similarity_std = scipy.ndimage.gaussian_filter(similarity_std, sigma=1)
        if similarity_perm is not None:
            similarity_perm = scipy.ndimage.gaussian_filter(similarity_perm, sigma=1)
        
    # --- 1. CKA scores
    ax.plot(similarity, color=color, **kwargs)
    
    # --- 2. FDR scores
    if similarity_mask is not None:
        
        assert len(similarity) == len(similarity_mask)

        for idx, _ in enumerate(similarity_mask):
             if not _:   
                 ax.scatter(idx, similarity[idx], facecolors='none', edgecolors=color)     # hollow circle
             else:
                 ax.scatter(idx, similarity[idx], facecolors=color, edgecolors=color)     # solid circle
    
    # --- 2. std for folds results
    if similarity_std is not None:
        
        ax.fill_between(plot_x, similarity-similarity_std, similarity+similarity_std, edgecolor=None, facecolor=utils_.lighten_color(utils_.color_to_hex(color), 100), alpha=0.75)
        
    # --- 3. error area
    if similarity_perm is not None:
        
        if similarity_perm.ndim == 2:     # permutation results
        
            perm_mean = np.mean(similarity_perm, axis=1)  
            perm_std = np.std(similarity_perm, axis=1)  
            
            ax.fill_between(plot_x, perm_mean-2*perm_std, perm_mean+2*perm_std, color='lightgray', edgecolor='none', alpha=0.5)
            ax.plot(plot_x, perm_mean, color='dimgray')
        
        elif similarity_perm.ndim == 1:     # mean value of permutation results
            
            ax.plot(plot_x, similarity_perm, color='dimgray')


def plot_CKA_temporal_comprehensive(fig, ax, similarity, used_id_num=None, debiased=True, **kwargs):
    
    # --- init
    assert similarity.ndim == 2
        
    # ---
    img = plot_CKA_temporal(ax, similarity, **kwargs)

    c_b2 = fig.colorbar(img, cax=fig.add_axes([0.91, 0.125, 0.03, 0.75]))
    c_b2.ax.tick_params(labelsize=16)


def plot_CKA_temporal(ax, similarity, similarity_mask=None, mask_type='shadow', **kwargs):
    
    # ---
    img = ax.imshow(similarity, aspect='auto', **kwargs)
    
    if similarity_mask is not None:
        
        if mask_type == 'shadow':     # [notice] this will expand the region for visualization
            similarity_mask_ = scipy.ndimage.gaussian_filter(similarity_mask, sigma=1, radius=2)
            mask_1 = similarity_mask_.copy()
            mask_1[mask_1>0.] = 1.     # plt will take np.nan as transparent
            ax.contour(mask_1, levels=[0.5], origin='lower', cmap='autumn', linewidths=3)
            mask_1 = 1-mask_1
            mask_1[mask_1==0.] = np.nan
            ax.imshow(mask_1, aspect='auto', cmap='gray', alpha=0.5)
        elif mask_type == 'stars':
            y, x = np.where(similarity_mask == True)
            ax.scatter(x, y, marker='*', c='red', s=100)
        else:
            raise ValueError
    
    return img


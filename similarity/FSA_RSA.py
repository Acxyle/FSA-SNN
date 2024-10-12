#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:47:12 2023

@author: Jinge Wang, Runnan Cao

    refer to: https://github.com/JingeW/ID_selective
              https://osf.io/824s7/
    
@modified: acxyle

    task: this code need to be fixed
    
"""


import os
import warnings
import logging
import numpy as np
import itertools

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from joblib import Parallel, delayed
import scipy

from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind
from statsmodels.stats.multitest import multipletests

from similarity.FSA_DRG import FSA_DSM

import utils_
from utils_ import utils_similarity

from bio_records_process.monkey_feature_process import monkey_feature_process
from bio_records_process.human_feature_process import human_feature_process


# ----------------------------------------------------------------------------------------------------------------------
__all__ = ["RSA_Monkey", "RSA_Monkey_folds", 
           "RSA_Human", "RSA_Human_folds",
           
           "RSA_Monkey_Comparison",
           "Monkey_similarity_scores_comparison", 
           "Human_similarity_scores_comparison"]

plt.rcParams.update({'font.size': 16})    
plt.rcParams.update({"font.family": "Times New Roman"})


# ======================================================================================================================
class RSA_Base():
    """ choose RSA firt_corr-second_scorr"""
    
    def __init__(self, **kwargs):
        """ those attributes should be defined by subclass """
        
        self.ts
        
        self.dest_primate
        self.layers
        
        self.primate_DM
        self.primate_DM_perm
        
        # --- if use permutation every time, those two can be ignored
        # --- empirically, the max fluctuation of mean scores between experiments could be ±0.03 with num_perm = 1000
        self.primate_DM_temporal
        self.primate_DM_temporal_perm
        
        
    def calculation_RSA(self, first_corr='pearson', second_corr='spearman', used_unit_type='qualified', used_id_num=50, primate=None, alpha=0.05, FDR_method='fdr_bh', **kwargs):
        """
            input:
                - first_corr: select from 'euclidean', 'pearson', 'spearman', 'mahalanobis', 'concordance'
                - second_corr: select from 'pearson', 'spearman', 'condordance'
                - alpha: significant level for FDR based on permutation test

            return:
                ...
        """
        
        utils_.make_dir(dest_primate:=os.path.join(self.dest_RSA,  f'{primate}'))
        
        if primate == 'Monkey':
 
            self.dest_primate = os.path.join(dest_primate, f'{first_corr}')
            utils_.make_dir(self.dest_primate)
            
            save_path = os.path.join(self.dest_primate, f'RSA_results_{first_corr}_{second_corr}.pkl')
            
        elif primate == 'Human':
            
            assert used_unit_type != None and used_id_num != None
            
            self.dest_primate = os.path.join(dest_primate, f'{first_corr}/{second_corr}', used_unit_type, str(used_id_num))
            utils_.make_dir(self.dest_primate)
            
            save_path = os.path.join(self.dest_primate, f'RSA_results_{first_corr}_{second_corr}_{used_unit_type}_{used_id_num}.pkl')
        
        else:
            
            raise ValueError
        
        if os.path.exists(save_path):
            
            RSA_dict = utils_.load(save_path, verbose=False)
            
        else:
            
            def _calculation_RSA(_layer, _second_corr='spearman', **kwargs):    

                # --- init, NN_DSM_v
                NN_DM = _vectorize_check(self.NN_DM_dict[_layer])
                
                if np.isnan(NN_DM).all():
                    NN_DM = np.full_like(self.primate_DM, np.nan)
                
                assert self.primate_DM.shape == NN_DM.shape
                
                # --- init, corr_func
                corr_func = _corr(_second_corr)
                
                # ----- static
                corr_coef = calculation_RSA(corr_func, self.primate_DM, NN_DM)
                corr_coef_perm = np.array([calculation_RSA(corr_func, _, NN_DM) for _ in self.primate_DM_perm])     # (1000,)
                
                # ----- temporal
                corr_coef_temporal = calculation_RSA_temporal(corr_func, self.primate_DM_temporal, NN_DM)     # (time_steps, )
                corr_coef_temporal_perm = np.array([calculation_RSA_temporal(corr_func, _, NN_DM) for _ in self.primate_DM_temporal_perm])     # (num_perm, time_steps)

                return {
                    'corr_coef': corr_coef,
                    'corr_coef_perm': corr_coef_perm,
                    'p_perm': np.mean(corr_coef_perm > corr_coef),     # equal to: np.sum(corr_coef_perm > corr_coef)/num_perm,
                    
                    'corr_coef_temporal': corr_coef_temporal,
                    'corr_coef_temporal_perm': corr_coef_temporal_perm,
                    'p_perm_temporal': np.array([np.mean(corr_coef_temporal_perm[:, _] > corr_coef_temporal[_]) for _ in range(len(corr_coef_temporal))])
                    }
            
            # -----
            pl = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(_calculation_RSA)(layer, second_corr=second_corr, **kwargs) for layer in tqdm(self.layers, desc='RSA'))

            # -----
            pl_k = ['corr_coef', 'corr_coef_perm', 'p_perm', 'corr_coef_temporal', 'corr_coef_temporal_perm', 'p_perm_temporal']
        
            similarity, similarity_perm, similarity_p, similarity_temporal, similarity_temporal_perm, similarity_temporal_p = [np.array([_[__] for _ in pl]) for __ in pl_k]
            
            # --- static
            (sig_FDR, p_FDR, alpha_Sadik, alpha_Bonf) = multipletests(similarity_p, alpha=alpha, method=FDR_method)    # FDR (flase discovery rate) correction
            sig_Bonf = p_FDR<alpha_Bonf
            
            # --- temporal
            p_temporal_FDR = np.zeros((len(self.layers), self.primate_DM_temporal.shape[0]))     # (num_layers, num_time_steps)
            sig_temporal_FDR, sig_temporal_Bonf =  np.zeros_like(p_temporal_FDR), np.zeros_like(p_temporal_FDR)
            
            for _ in range(len(self.layers)):
                
                (sig_temporal_FDR[_, :], p_temporal_FDR[_, :], alpha_Sadik_temporal, alpha_Bonf_temporal) = multipletests(similarity_temporal_p[_, :], alpha=alpha, method=FDR_method)      # FDR
                sig_temporal_Bonf[_, :] = p_temporal_FDR[_, :]<alpha_Bonf_temporal     # Bonf correction
            
            # --- seal results
            RSA_dict = {
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
            
            utils_.dump(RSA_dict, save_path, verbose=False)
        
        return RSA_dict
    
    
    def plot_RSA_comprehensive(self, ax, RSA_dict, EC='sig_FDR', stats=True, ticks=None, **kwargs):
        """ EC: Error Control Measurement """

        similarity = RSA_dict['similarity']
        similarity_mask = RSA_dict[EC]
        similarity_perm = RSA_dict['similarity_perm']
        
        similarity_std = RSA_dict['similarity_std']  if 'similarity_std' in RSA_dict.keys() else None
        
        plot_RSA_comprehensive(ax, similarity, similarity_std=similarity_std, similarity_mask=similarity_mask, similarity_perm=similarity_perm, **kwargs)
        
        if ticks:
            ax.set_xticks(np.arange(len(self.layers)))
            ax.set_xticklabels(self.layers, rotation='vertical')
        else:
            ax.set_xlim([0, len(self.layers)-1])
            ax.set_xticks([0, len(self.layers)-1])
            ax.set_xticklabels([0, 1])
            
        if stats:
            utils_similarity.fake_legend_describe_numpy(ax, similarity, similarity_mask, **kwargs)

            
    def plot_RSA_temporal_comprehensive(self, fig, ax, RSA_dict, EC='sig_temporal_Bonf', vlim=None, stats=True, ticks=None, **kwargs):

        # --- depackaging
        #extent = [self.ts.min()-5, self.ts.max()+5, -0.5, RSA_dict['similarity_temporal'].shape[0]-0.5]
        
        similarity = RSA_dict['similarity_temporal']
        similarity_mask = RSA_dict[EC]
    
        plot_RSA_temporal_comprehensive(fig, ax, similarity, similarity_mask=similarity_mask, **kwargs)
        
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
            utils_similarity.fake_legend_describe_numpy(ax, RSA_dict['similarity_temporal'], RSA_dict[EC].astype(bool), **kwargs)
            

# ----------------------------------------------------------------------------------------------------------------------
class RSA_Monkey(monkey_feature_process, FSA_DSM, RSA_Base):
    """
        ...
    """
    
    def __init__(self, seed=6, **kwargs):
        
        # --- init
        monkey_feature_process.__init__(self, seed=seed)
        FSA_DSM.__init__(self, **kwargs)
        
        self.dest_RSA = os.path.join(self.dest, 'RSA')

        
    def __call__(self, first_corr='pearson', second_corr='spearman', **kwargs):
        """
            input:
                ...
                      
        """
        utils_.formatted_print(f'RSA Monkey | {first_corr} | {second_corr}')
        
        # --- monkey init
        self.primate_DM, self.primate_DM_temporal, self.primate_DM_perm, self.primate_DM_temporal_perm = self.calculation_DSM_perm_monkey(first_corr=first_corr, **kwargs)
        
        self.primate_DM = _vectorize_check(self.primate_DM)
        self.primate_DM_temporal = np.array([_vectorize_check(_) for _ in self.primate_DM_temporal])
        self.primate_DM_perm = np.array([_vectorize_check(_) for _ in self.primate_DM_perm])
        self.primate_DM_temporal_perm = np.array([np.array([_vectorize_check(__) for __ in _]) for _ in self.primate_DM_temporal_perm])
        
        # --- NN init
        self.NN_DM_dict = {k:v['qualified'] for k,v in self.calculation_DSM(first_corr, vectorize=False, **kwargs).items()}   # layer - cell_type

        # ----- 1. RSA calculation
        RSA_dict = self.calculation_RSA(first_corr, second_corr, primate='Monkey', **kwargs)
        
        # ----- 2. plot
        # --- 2.1 static
        fig, ax = plt.subplots(figsize=(10,6))
        
        self.plot_RSA_comprehensive(ax, RSA_dict, **kwargs)
        title = f'RSA Score {self.model_structure} {first_corr} {second_corr}'
        ax.set_title(f'{title}')

        fig.tight_layout(pad=1)
        fig.savefig(os.path.join(self.dest_primate, f'{title}.svg'), bbox_inches='tight')
        plt.close()
        
        # --- 2.2 temporal
        fig, ax = plt.subplots(figsize=(10,6))
        
        self.plot_RSA_temporal_comprehensive(fig, ax, RSA_dict, **kwargs)
        title=f'RSA Score temporal {self.model_structure} {first_corr} {second_corr}'
        ax.set_title(f'{title}')

        fig.savefig(os.path.join(self.dest_primate, f'{title}.svg'), bbox_inches='tight')
        plt.close()
        
        # --- 3. example correlation
        self.plot_correlation_example(RSA_dict['similarity'], first_corr=first_corr, second_corr=second_corr)
        

    #FIXME --- legacy
    def plot_correlation_example(self, similarity, first_corr='pearson', second_corr='spearman', neuron_type='qualified', attach_psth:bool=False):
        """
            this function plot with fig definition and ax addition
        """
        
        # plot correlation for sample layer 
        layer = self.layers[np.argmax(similarity)]     # find the layer with strongest similarity score
        NN_DM_v = _vectorize_check(self.NN_DM_dict[layer])
        
        if not attach_psth:
            
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # plot sample PSTH - they namually set the target time point is 90, same results for below 2 methods
            #bestTimeFR = np.mean(self.meanPSTHID[:, np.where(self.psthTime == 90)[0][0], :], axis=0)
            bestTimeFR = np.mean(self.meanPSTHID[:, self.psthTime>60, :], axis=(0,1))
            
            most_active_cell = np.argmax(bestTimeFR)

            # plot corr example
            r, p, _ = self.plot_corr_2d(self.primate_DM, NN_DM_v, 'blue', ax, 'Spearman')
            ax.set_xlabel('Monkey Pairwise Distance')
            ax.set_ylabel('Network Pairwise Distance')
            ax.set_title(f'r:{r:.3f}, p:{p:.3e}')
        
        else:
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # plot sample PSTH - they namually set the target time point is 90, same results for below 2 methods
            bestTimeFR = np.mean(self.meanPSTHID[:, self.psthTime>60, :], axis=(0,1))
            
            most_active_cell = np.argmax(bestTimeFR)
            
            axes[0].imshow(self.meanPSTHID[:, :, most_active_cell], extent=[self.ts[0], self.ts[-1], 1, 50], aspect='auto')
            axes[0].set_xlabel('Time(ms)')
            axes[0].set_ylabel('ID')
            axes[0].set_title(f'Monkey IT Neuron {most_active_cell}')
            axes[0].tick_params(labelsize=12)
            
            # plot corr example
            r, p, _ = self.plot_corr_2d(self.primate_DM, NN_DM_v, 'b', axes[1], 'Spearman')
            axes[1].set_xlabel('Monkey Pairwise Distance')
            axes[1].set_ylabel('Network Pairwise Distance')
            axes[1].set_title(f'r:{r:.3f}, p:{p:.3e}')
        
        title = f'Monkey - {self.model_structure} {layer} polyfit {first_corr} {second_corr}'
        fig.suptitle(f'{title}', fontsize=20, y=1.)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dest_primate, f'{title}.svg'))
        plt.close()
                      
    
    #FIXME --- legacy
    def plot_corr_2d(self, A, B, color='blue', ax=None, criteria='Pearson'):
        
        if criteria == 'Pearson':  # no tested
            corr_func = pearsonr
        elif criteria == 'Spearman':
            corr_func = spearmanr
        elif criteria == 'Kendalltau':  # no tested
            corr_func = kendalltau
        else:
            raise ValueError('[Coderror] Unknown correlation type')
    
        ind = np.where(~np.isnan(A) & ~np.isnan(B))[0]
    
        if ind.size == 0:
            raise ValueError('[Coderror] All NaN values')
    
        r, p = corr_func(A[ind], B[ind])
    
        title = f'r={r:.5f} p={p:.3e}'
    
        if ax is not None and isinstance(ax, matplotlib.axes.Axes):
            
            ax.plot(A[ind], B[ind], color=color, linestyle='none', marker='.', linewidth=2, markersize=2)
            (k_, p_) = np.polyfit(A[ind], B[ind], 1)     # polynomial fitting, degree=1

            x_ = np.array([np.min(A), np.max(A)])
            y_ = x_*k_ + p_
            
            ax.plot(x_, y_,color='red', linewidth=2)
            ax.axis('tight')
    
        return r, p, title                                  


# ----------------------------------------------------------------------------------------------------------------------
class RSA_Human(human_feature_process, FSA_DSM, RSA_Base):
    """
        ...
    """
    
    def __init__(self, seed=6, **kwargs):
        
        human_feature_process.__init__(self, seed=seed)
        FSA_DSM.__init__(self, **kwargs)
        
        self.dest_RSA = os.path.join(self.dest, 'RSA')
        utils_.make_dir(self.dest_RSA)
        
        self.save_root_primate = os.path.join(self.dest_RSA, 'Human')
        utils_.make_dir(self.save_root_primate)
        
        
    def __call__(self, first_corr='pearson', second_corr='spearman', used_unit_type='qualified', used_id_num=50, **kwargs):
        """
            ...
        """
        # --- 
        utils_.formatted_print(f'RSA Human | {self.model_structure} | {first_corr} | {second_corr} | {used_unit_type} | {used_id_num}')
        
        # --- init
        used_id = self.calculation_subIDs(used_id_num)
        
        NN_DM_dict = self.calculation_DSM(first_corr, **kwargs)     # by default, it contains ['qualified', 'selective', 'strong_selective', 'weak_selective', 'non_selective']
        
        DM, DM_temporal, DM_perm, DM_temporal_perm = self.calculation_DSM_perm_human(first_corr, used_unit_type=used_unit_type, used_id_num=used_id_num, **kwargs)
        self.NN_DM_dict = {_: _vectorize_check(NN_DM_dict[_][used_unit_type][np.ix_(used_id, used_id)]) if ~np.isnan(NN_DM_dict[_][used_unit_type]).all() else np.nan for _ in NN_DM_dict.keys()}
            
        # ---
        self.primate_DM = _vectorize_check(DM)
        self.primate_DM_temporal = np.array([_vectorize_check(_) for _ in DM_temporal])
        
        self.primate_DM_perm = np.array([_vectorize_check(_) for _ in DM_perm])
        self.primate_DM_temporal_perm = np.array([np.array([_vectorize_check(__) for __ in _]) for _ in DM_temporal_perm])
        
        # --- RSA calculation
        RSA_dict = self.calculation_RSA(first_corr=first_corr, second_corr=second_corr, used_unit_type=used_unit_type, used_id_num=used_id_num, primate='Human', **kwargs)
        
        # --- plot
        # --- 2.1 static
        fig, ax = plt.subplots(figsize=(10, 6))
        
        self.plot_RSA_comprehensive(ax, RSA_dict, **kwargs)
        title=f'RSA Score {self.model_structure} {first_corr} {second_corr} {used_unit_type} {used_id_num}'
        ax.set_title(f'{title}')

        fig.tight_layout(pad=1)
        fig.savefig(os.path.join(self.dest_primate, f'{title}.svg'), bbox_inches='tight')
        plt.close()
        
        # --- 2.2 temporal
        fig, ax = plt.subplots(figsize=(np.array(RSA_dict['similarity_temporal'].T.shape)/3.7))
        
        self.plot_RSA_temporal_comprehensive(fig, ax, RSA_dict, **kwargs)
        title=f'RSA Score temporal {self.model_structure} {first_corr} {second_corr} {used_unit_type} {used_id_num}'
        ax.set_title(f'{title}', fontsize=16)
        
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.set_xticklabels([-250, 0, 250, 500, 750, 1000, 1250])
        
        fig.savefig(os.path.join(self.dest_primate, f'{title}.svg'), bbox_inches='tight')     
        plt.close()
    
    
    def process_all_used_unit_results(self, used_id_num=50, **kwargs):
        
        RSA_types_dict = self.collect_all_used_unit_results(used_id_num, **kwargs)
        
        # --- static
        fig, ax = plt.subplots(figsize=(len(self.layers)/2, 4))
        
        self.plot_collect_all_used_unit_results(fig, ax, RSA_types_dict, used_id_num)
        
    
    def collect_all_used_unit_results(self, used_id_num=50, used_unit_types=None, **kwargs):
        
        RSA_types_dict = {}
        
        for used_unit_type in used_unit_types:
        
            RSA_types_dict[used_unit_type] = self.calculation_RSA(used_unit_type=used_unit_type, used_id_num=used_id_num, primate='Human', **kwargs)
        
        return RSA_types_dict
    
    def plot_collect_all_used_unit_results(self, fig, ax, RSA_types_dict, used_id_num=50, used_unit_types=None, text=False, **kwargs):
        
        for k, v in RSA_types_dict.items():
            
            similarity = np.nan_to_num(v['similarity'])
            similarity_std = np.nan_to_num(v['similarity_std']) if 'similarity_std' in v else None
            
            color = self.plot_Encode_config.loc[k]['color']
            label = self.plot_Encode_config.loc[k]['label']
            linestyle = self.plot_Encode_config.loc[k]['linestyle']
            
            plot_RSA(ax, similarity, similarity_std=similarity_std, color=color, linestyle=linestyle, label=label, smooth=True)
        
        if text:
            ax.legend()
            ax.set_title(f'{self.model_structure} used_id_num: {used_id_num}')
        
        ax.hlines(0, 0, len(self.layers)-1, color='gray', linestyle='--', alpha=0.25)
        
        ax.set_xlim([0, len(self.layers)-1])
        ax.set_xticks([0, len(self.layers)-1])
        ax.set_xticklabels([0, 1])

        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        
        fig.tight_layout(pad=1)
        fig.savefig(os.path.join(self.dest_RSA, f'{self.model_structure} RSA results types {used_id_num}.svg'), bbox_inches='tight')
        plt.close()


# ----------------------------------------------------------------------------------------------------------------------
def calculation_RSA(corr_func, primate_DM, NN_DM, **kwargs):
    return corr_func(primate_DM, NN_DM, **kwargs)


def calculation_RSA_temporal(corr_func, primate_DM_temporal, NN_DM, **kwargs):
    # input shape: Bio - (time_steps, corr_vector), NN - (corr_vector, )
    return np.array([calculation_RSA(corr_func, _, NN_DM, **kwargs) for _ in primate_DM_temporal])      # (time_steps, )


def plot_RSA(ax, similarity, similarity_std=None, similarity_mask=None, similarity_perm=None, color=None, smooth=True, used_unit_types=None, **kwargs):
    
    # -- init
    color = 'blue' if color is None else color

    plot_x = np.arange(len(similarity))
    
    if smooth:
        similarity = scipy.ndimage.gaussian_filter(similarity, sigma=1)
        if similarity_std is not None:
            similarity_std = scipy.ndimage.gaussian_filter(similarity_std, sigma=1)
        if similarity_perm is not None:
            similarity_perm = scipy.ndimage.gaussian_filter(similarity_perm, sigma=1)
        
    # --- 1. RSA scores
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
        
        ax.fill_between(plot_x, similarity-similarity_std, similarity+similarity_std, edgecolor=None, facecolor=utils_.lighten_color(utils_.color_to_hex(color), 100), alpha=0.5)
        
    # --- 3. error area
    if similarity_perm is not None:
        
        if similarity_perm.ndim == 2:     # permutation results
        
            perm_mean = np.mean(similarity_perm, axis=1)  
            perm_std = np.std(similarity_perm, axis=1)  
            
            ax.fill_between(plot_x, perm_mean-2*perm_std, perm_mean+2*perm_std, color='lightgray', edgecolor='none', alpha=0.5)
            ax.plot(plot_x, perm_mean, color='dimgray')
        
        elif similarity_perm.ndim == 1:     # mean value of permutation results
            
            ax.plot(plot_x, similarity_perm, color='dimgray')


def plot_RSA_temporal_comprehensive(fig, ax, similarity, **kwargs):
    
    # --- init
    assert similarity.ndim == 2
        
    # ---
    img = plot_RSA_temporal(ax, similarity, **kwargs)

    c_b2 = fig.colorbar(img, cax=fig.add_axes([0.91, 0.125, 0.03, 0.75]))
    c_b2.ax.tick_params(labelsize=16)


def plot_RSA_temporal(ax, similarity, similarity_mask=None, mask_type='shadow', used_unit_types=None, **kwargs):
    
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

# ----------------------------------------------------------------------------------------------------------------------
#FIXME --- seems need to add abnormal detection? because ns cells/units always generate weird output 
def _vectorize_check(input:np.ndarray):
    
    if np.isnan(input).all() or input.ndim == 1:
        pass
    elif input.ndim == 2:
        input = utils_similarity.RSM_vectorize(input)     # (50,50) -> (1225,)
    else:
        raise ValueError('invalid input')
        
    return input


def _corr(second_corr):
    
    corr_func_map = {
        'spearman': _spearmanr,
        'pearson': _pearson,
        'concordance': _ccc
        }
    
    if second_corr in corr_func_map:
        return corr_func_map[second_corr]
    else:
        raise ValueError('[Coderror] invalid second_corr')


def _spearmanr(x, y):
    """
        x: primate
        y: NN
    """
    
    if np.unique(y).size < 2 or np.any(np.isnan(y)):
        return np.nan
    else:
        return spearmanr(x, y, nan_policy='omit').statistic


def _pearson(x, y):
    return np.corrcoef(x, y)[0, 1]


def _ccc(x, y):
    return utils_similarity._ccc(x, y)


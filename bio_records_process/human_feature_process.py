#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 13:35:41 2023

@author: Runnan Cao
@modified: acxyle

    - refer to: https://osf.io/824s7/
    - 
    
"""

# --- python
import os
import warnings
import numpy as np
from tqdm import tqdm
from matplotlib import gridspec
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# --- stats
import scipy.stats as stats
from scipy.stats import gaussian_kde
#from scipy.stats import norm, skew, lognorm, kstest
#from scipy.spatial.distance import pdist, squareform
from scipy.integrate import quad, IntegrationWarning
from sklearn.manifold import TSNE

# --- local
import sys
sys.path.append('../')     # if run this code as script
import utils_

from .human_raw_data_process import human_raw_data_process
from .primate_feature_process import primate_feature_process

# --- legacy
from similarity import Selectivity_Analysis_Feature


# ======================================================================================================================
class human_feature_process(human_raw_data_process, primate_feature_process):

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 18})
 

    def calculation_DSM_human(self, first_corr='pearson', used_unit_type='qualified', used_id_num=50, **kwargs):

        # --- 
        utils_.make_dir(save_root:=os.path.join(self.human_neuron_stats, 'Corr'))
   
        # --- init
        used_cells = self.calculation_Sort()[used_unit_type]
        
        if used_cells.size == 0:
            
            return None, None
        
        else:
            
            save_path = os.path.join(save_root, f'DM_{first_corr}_{used_unit_type}_{len(used_cells)}.pkl')
    
            if os.path.exists(save_path):
                
                (DM, DM_temporal) = utils_.load(save_path, verbose=False)

            else:
            
                FR_id, psth_id = self.calculation_FM()
                
                FR_id = FR_id[:, used_cells]
                psth_id = np.array([_[:, used_cells] for _ in psth_id])
    
                # ---
                DM, DM_temporal = self.calculation_1st_stats('DSM', FR_id, psth_id, first_corr=first_corr, **kwargs)
                
                utils_.dump((DM, DM_temporal), save_path, verbose=False)     # (50, 50)
            
            # ---
            used_ids = self.calculation_subIDs(used_id_num)
            
            DM = DM[np.ix_(used_ids, used_ids)]
            DM_temporal = np.array([_[np.ix_(used_ids, used_ids)] for _ in DM_temporal])
            
            return DM, DM_temporal
    
    
    def calculation_DSM_perm_human(self, first_corr='pearson', **kwargs):

        DM, DM_temporal = self.calculation_DSM_human(first_corr, **kwargs)
        
        DM_perm, DM_temporal_perm = self.calculation_1st_stats_perm(DM, DM_temporal, **kwargs)
            
        return DM, DM_temporal, DM_perm, DM_temporal_perm
    
    
    def calculation_Gram_human(self, kernel='linear', used_unit_type='qualified', used_id_num=50, permutation=True, num_perm=1000, save=True, **kwargs):

        
        utils_.make_dir(save_root:=os.path.join(self.human_neuron_stats, 'Gram'))
        
        used_cells = self.calculation_Sort()[used_unit_type]
        
        if kernel == 'rbf' and 'threshold' in kwargs:
            save_path = os.path.join(save_root,  f"CKA_results_{kernel}_{kwargs['threshold']}_{used_unit_type}_{len(used_cells)}.pkl")
        elif kernel == 'linear':
            save_path = os.path.join(save_root, f"CKA_results_{kernel}_{used_unit_type}_{len(used_cells)}.pkl")
        else:
            raise ValueError

        if os.path.exists(save_path):
            
            (Gram, Gram_temporal) = utils_.load(save_path, verbose=False)
            
        else:
            
            FR_id, psth_id = self.calculation_FM()
            
            FR_id = FR_id[:, used_cells]
            psth_id = np.array([_[:, used_cells] for _ in psth_id])
            
            Gram, Gram_temporal = self.calculation_1st_stats('Gram', FR_id, psth_id, kernel=kernel, **kwargs)

            utils_.dump((Gram, Gram_temporal), save_path, verbose=False)
            
        # ---
        used_ids = self.calculation_subIDs(used_id_num)
        
        Gram = Gram[np.ix_(used_ids, used_ids)]
        Gram_temporal = np.array([_[np.ix_(used_ids, used_ids)] for _ in Gram_temporal])
            
        return Gram, Gram_temporal
    
    
    def calculation_Gram_perm_human(self, kernel='linear', **kwargs):

        Gram, Gram_temporal = self.calculation_Gram_human(kernel, **kwargs)
        
        Gram_perm, Gram_temporal_perm = self.calculation_1st_stats_perm(Gram, Gram_temporal, **kwargs)
            
        return Gram, Gram_temporal, Gram_perm, Gram_temporal_perm
        
    
    def plot_FR_PDF(self, init:float=0.15, **kwargs):
        """ this function does not remove the responses of repeated images """
        
        FR_stats = self.calculation_SortedFR()
        
        feature = [FR_stats[_]['spike_count_0_2000'] for _ in range(len(FR_stats))]     
        feature = np.array([np.mean(_[~np.isnan(_)]) for _ in feature])     # (2082,)
        
        # --- min-max normalization
        scaling_factor = np.max(feature)-np.min(feature)
        init_th = init/scaling_factor
        feature = feature/scaling_factor     # (0, 1)
        
        fig = self.plot_PDF(feature=feature, init_threshold=init_th, scaling_factor=scaling_factor)
        plt.tight_layout()
        fig.savefig(os.path.join(self.human_neuron_stats, 'neuron PDF.svg'))
        plt.close()
    
    
    @staticmethod
    def plot_PDF(model_structure='human MTL', target='cell', feature=None, init_threshold=None, scaling_factor=1., layer='', unit_type='', **kwargs) -> None:
        """ this function plots the log gaussian hist and PDF of human cell """
        
        warnings.filterwarnings('ignore', category=IntegrationWarning)     # [caution]
        
        if feature.ndim == 1:
            feature = feature
        elif feature.ndim ==2 and feature.shape[0]==500:     # assume the shape is (num_samples, num_features)
            feature = np.mean(feature, axis=0)
        else:
            raise ValueError
        
        # -----
        def _hist(feature):
            
            kde = gaussian_kde(feature)
            feature_mean = np.mean(feature)
            feature_std = np.std(feature)
            feature_radius = max(np.max(feature) - feature_mean, feature_mean - np.min(feature))
            x = np.linspace(feature_mean - feature_radius, feature_mean + feature_radius, 1000)
            y_kde = kde(x)
            
            return kde, x, y_kde, feature_mean, feature_std, feature_radius
        
        def _plot_hist(ax, feature, title, plot_fitted_PDF=False, plot_legend=False, init_threshold=None):
            
            kde, x, y_kde, feature_mean, feature_std, feature_radius = _hist(feature)
            
            # ---
            (hist_pct, hist_x, _) = ax.hist(feature, bins=100, density=True)
            ylim_max = max(np.ceil(np.max(hist_pct) / 10) * 10, 1.5 * np.max(hist_pct)) if np.max(hist_pct) > 5 else np.max(hist_pct) * 1.5
            ax.set_ylim([0, ylim_max])
            ax.set_title(title, fontsize=24)
            ylim_max_auto = ax.get_ylim()[1]
            
            # ---
            if plot_fitted_PDF:
                ax.vlines(np.mean(feature), 0, ylim_max_auto, color='red', label='mean')
                y_norm = stats.norm.pdf(x, feature_mean, feature_std)
                ax.plot(x, y_kde, linestyle='--', linewidth=2, color='orange', label='gaussian_kde')
                ax.plot(x, y_norm, linestyle='--', linewidth=2, color='red', label='gaussian_fit')
            
            # ---
            if plot_legend:
                for i in range(3):
                    pct_below = quad(kde, -np.inf, feature_mean - (i + 1) * feature_std)[0] * 100
                    pct_above = quad(kde, feature_mean + (i + 1) * feature_std, np.inf)[0] * 100
                    ax.vlines(feature_mean - (i + 1) * feature_std, 0, ylim_max_auto, linestyle='dotted', color='gold', label=f'p < mean-{i+1}std: {pct_below:.2f}%')
                    ax.vlines(feature_mean + (i + 1) * feature_std, 0, ylim_max_auto, linestyle='dotted', color='purple', label=f'p > mean+{i+1}std: {pct_above:.2f}%')
                ax.legend(framealpha=0.5)
                ax.set_xlim([feature_mean - feature_radius, feature_mean + feature_radius])
        
            # ---
            if init_threshold:
                pct_init = quad(kde, -np.inf, np.log10(init_threshold))[0] * 100
                ax.vlines(np.log10(init_threshold), 0, ylim_max_auto, linestyle='dotted', color='red', label=f'manual value of {init_threshold:.2f} ({pct_init:.2f}%)')
                ax.fill_between(x, y_kde, where=(x < np.log10(init_threshold)), color='gray', alpha=0.5)
  
            return kde, x, y_kde, feature_mean, feature_std, feature_radius
        
        # -----
        suptitle = f"{model_structure} {layer} {unit_type} {target} PDF"

        if feature.size < 2:
            fig, ax = plt.subplots(1, 2, figsize=(20,10))
            return fig
        elif np.any(feature<0):     # -> details for original data, simple for log data
            # --- original
            fig, ax = plt.subplots(1, 2, figsize=(20,10))
            _plot_hist(ax[0], feature, 'hist of original data', True, True, init_threshold=init_threshold)
            # --- log
            feature_log = np.log10(feature[feature>0])
            _plot_hist(ax[1], feature_log, f'log10 hist and gaussian kde exclude <=0 ({(feature.size-feature_log.size)/feature.size*100:.2f}%)', True, init_threshold=init_threshold)
            fig.suptitle(suptitle, y=0.975, fontsize=28)
        else:   # -> simple for original data, details for log data
            fig, ax = plt.subplots(1, 2, figsize=(20,10))
            _plot_hist(ax[0], feature, 'hist of original data', True, init_threshold=init_threshold)
            # --- log
            feature_log = np.log10(feature[feature>0])
            _plot_hist(ax[1], feature_log, f'log10 hist and gaussian kde excluse <=0 ({(feature.size-feature_log.size)/feature.size*100:.2f}%)', True, True, init_threshold=init_threshold)
            fig.suptitle(suptitle, y=0.975, fontsize=28)
            
        return fig
    

    def plot_piechart(self, transparent=True):

        cell_stats = self.calculation_subIDs()

        bio_pct = cell_stats['cell_types_dict']
        
        tmp = [bio_pct[_] for _ in bio_pct.keys() if 'non' in _]
        tmp = [__ for _ in tmp for __ in _]
        
        bio_pct_new = {}

        for _ in bio_pct.keys():
            if 'non' not in _:
                bio_pct_new.update({_: bio_pct[_]})
        
        bio_pct_new.update({'non_encode': np.array(tmp)})
        
        values = [len(bio_pct_new[_]) for _ in bio_pct_new.keys()]
        labels = [f's_si ({values[0]})', f's_wsi ({values[1]})', f's_mi ({values[2]})', f's_wmi ({values[3]})', f'n_e ({values[4]})']
        colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']
        explode = np.array([0.5, 0.1, 0.5, 0.1, 0.])
        
        title = f'{np.sum(values)} Cells'

        fig, ax = plt.subplots(figsize=(10,6))
        utils_.plot_pie_chart(fig, ax, values, labels, title, colors, explode)
        fig.savefig(os.path.join(self.human_neuron_stats, 'bio_pct_pie_chart.svg'), transparent=transparent)
        plt.close()
        
        
    def plot_stacked_responses(self, num_types=5):
 
        # --- load cell_stats
        cell_stats = self.calculation_subIDs()
        idx_dict = cell_stats['cell_types_dict']
        
        # --- load feature
        meanFR_dict = self.human_cell_SortFR(data_type='default')
        meanFR = meanFR_dict['meanFR']

        feature = meanFR.T     
        feature[np.isnan(feature)] = 0
        
        # ---
        if num_types == 5:
           
            idx_dict = {
                's_si': idx_dict['sensitive_si'],
                's_wsi': idx_dict['sensitive_wsi'],
                
                's_mi': idx_dict['sensitive_mi'],
                's_wmi': idx_dict['sensitive_wmi'],
                
                'n_e': np.concatenate((idx_dict['sensitive_non_encode'], 
                        idx_dict['non_sensitive_si'], 
                        idx_dict['non_sensitive_wsi'], 
                        idx_dict['non_sensitive_mi'], 
                        idx_dict['non_sensitive_wmi'], 
                        idx_dict['non_sensitive_non_encode'])).astype(np.int64)
                }
        
            # init the canvas
            fig, ax = plt.subplots(figsize=(26, 6))
            gs_main = gridspec.GridSpec(1, 5, figure=fig)
    
            plot_single(fig, gs_main, 5, idx_dict, feature, 50, 10)
        
        
        elif num_types == 10:

            idx_dict = {
                's_si': idx_dict['sensitive_si'],
                's_wsi': idx_dict['sensitive_wsi'],
                
                's_mi': idx_dict['sensitive_mi'],
                's_wmi': idx_dict['sensitive_wmi'],
                
                's_non_encode': idx_dict['sensitive_non_encode'],
                
                'ns_si': idx_dict['non_sensitive_si'],
                'ns_wsi': idx_dict['non_sensitive_wsi'],
                
                'ns_mi': idx_dict['non_sensitive_mi'],
                'ns_wmi': idx_dict['non_sensitive_wmi'],
                
                'ns_non_encode': idx_dict['non_sensitive_non_encode']
                }
        
            # init the canvas
            fig, ax = plt.subplots(figsize=(26,10))
            gs_main = gridspec.GridSpec(2, 5, figure=fig)
    
            plot_single(fig, gs_main, 5, idx_dict, feature, 50, 10)
        
        # -----
        ax.axis('off')
        ax.plot([],[],color='blue', linestyle='--', label='mean')
        ax.plot([],[],color='teal', linestyle='--', label='ref')
        ax.plot([],[],color='red', linestyle='--', label='threshold')
        
        fig.suptitle('Human MTL Neuron Responses for Human Faces', y=0.97, fontsize=20)
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, bbox_transform=plt.gcf().transFigure)
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.human_neuron_stats, f'Human MTL Neuron Responses {num_types} types.svg'), bbox_inches='tight')
        plt.close()

    
    # ------------------------------------------------------------------------------------------------------------------
    # FIXME --- legacy --- not in use
    def human_DR(self, NN_folder, layer='neuron_2'):
        
        coor_name = ''.join([NN_folder.split(' ')[-1], ' ', layer])
        tsne_dict = utils_.load(f'/home/acxyle-workstation/Downloads/{NN_folder}/Analysis/Dimension_Reduction/TSNE/tsne_all.pkl')
        
        # --- default: all
        tsne = tsne_dict[layer]['tsne_dict']['all']
        
        self.human_DR_single(coor_name, tsne)
        
    # FIXME --- need to simplify --- legacy --- not in use
    def human_DR_single(self, coor_name:str=None, tsne:np.array=None):
        """
            this function generates the coordinates based on the 
        """
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        id_labels = np.arange(1, 51)
        img_labels = np.array([np.array([_]*10) for _ in id_labels]).reshape(-1)
        
        meanFR_dict = self.human_cell_SortFR()
        cell_stats = self.calculation_subIDs()
        
        meanFR = meanFR_dict['meanFR']
        qualified_cells = meanFR_dict['qualified_cells']
        
        self.DR_save_folder = os.path.join(self.human_neuron_stats, 'DR results')
        utils_.make_dir(self.DR_save_folder)
        
        self.tsne_folder = os.path.join(self.DR_save_folder, 'TSNE')
        utils_.make_dir(self.tsne_folder)
        
        # -----
        if coor_name is None and tsne is None:
            
            meanFR_ = np.nan_to_num(meanFR[qualified_cells, :])
            
            perplexity = np.min([np.sqrt(len(qualified_cells)), 50*10-1])
            
            # --- local coordinates
            tsne = TSNE(perplexity=perplexity).fit_transform(meanFR_.T)
            
            coor_name = 'human_coor'

        else:
            
            assert isinstance(tsne, np.ndarray) and coor_name is not None
            
        self.tsne_save_folder = os.path.join(self.tsne_folder, coor_name)
        utils_.make_dir(self.tsne_save_folder)
            
        # ----- p_values
        # --- init
        DR_sub_type = 'all'
        layer = 'neuron_2'
        sq = 0.035
        
        feature = np.nan_to_num(self.human_cell_SortFR()['meanFR']).T
        
        save_path = os.path.join(self.tsne_save_folder, f'{layer}_{DR_sub_type}_sq{sq}.pkl')
        
        if os.path.exists(save_path):
            
            results = utils_.pickle_load_tqdm(save_path)
            
        else:
            
            kernel_size, kernel_sigma = Selectivity_Analysis_Feature.Selectivity_Analysis_Feature.get_kernel_size(tsne)
            gaussian_kernel = Selectivity_Analysis_Feature.Selectivity_Analysis_Feature.gausskernel(kernel_size, kernel_sigma)
            
            # --- calculate p values
            p = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(Selectivity_Analysis_Feature.calculate_density_perm_p)(tsne, feature[:, i], num_perm=1000, kernel=gaussian_kernel) for i in tqdm(np.arange(feature.shape[1]), desc=f'{DR_sub_type}'))
            
            # --- wrap results and save
            results = {
                       'layer': layer, 
                       'tsne': tsne,     
                       'DR_sub_type': DR_sub_type,
                       'p': p, 
                       'sigma_scaling_factor': sq, 
                       'kernel_size': kernel_size, 
                       'kernel_sigma': kernel_sigma, 
                       'kernel': gaussian_kernel,
                       }
            
            utils_.pickle_dump(save_path, results)
            
        # ----- feature regions
        # --- init
        p_values = results['p']
        gaussian_kernel = results['kernel']
        tsne = results['tsne']
        maskFactor = 0.1
        cluster_size_scaling_factor=0.025
        alpha=0.01

        # ---
        save_path = os.path.join(self.tsne_save_folder, f'{layer}_{DR_sub_type}_unit_stats.pkl')
        
        if os.path.exists(save_path):
            
            results = utils_.pickle_load_tqdm(save_path)
            
        else:
        
            density_map, convolved_density_map = Selectivity_Analysis_Feature.calculate_convolved_density_map(tsne, None, gaussian_kernel)
            
            # --- remove corners and edges with too sparse dots
            mask = convolved_density_map >= (maskFactor*np.mean(convolved_density_map))
 
            # --- init
            reversed_sort_dict = {value: [key for key, vals in cell_stats['cell_types_dict'].items() if value in vals][0] for key_list in cell_stats['cell_types_dict'].values() for value in key_list}
            reversed_sort_dict = {_: reversed_sort_dict[_] for _ in sorted(reversed_sort_dict.keys())}
            
            cluster_size_threshold = mask.size*cluster_size_scaling_factor
            
            # --- Sequential, for test
            pl = {}
            
            units = list(reversed_sort_dict.keys())
            for unit in tqdm(units, desc='Sequential region selection'):
                
                results = Selectivity_Analysis_Feature.feature_region_selection_single_unit(

                                                          tsne, 
                                                          
                                                          {unit: reversed_sort_dict[unit]},
                                                          p_values[unit], 
                                                          cell_stats['encode_id'][unit],      # ---
                                                               
                                                          mask, 
                                                          cluster_size_threshold, 
                                                          
                                                          img_labels,
                                                          )
                                            
                pl[unit] = results
        
            feature_selective_stats = {_: pl[_]['feature_selective_unit'] for _ in units if pl[_] is not None and len(pl[_]['feature_selective_unit']) != 0}
            tmp_pool = [___ for __ in [feature_selective_stats[_] for _ in feature_selective_stats.keys()] for ___ in __]
            tmp_pool_new = [_.split('encode_')[-1] for _ in tmp_pool]
            
            feature_component_stats = {_:pl[_]['feature_component_dict'] for _ in units if pl[_] is not None and len(pl[_]['feature_component_dict']) != 0}
    
            # feature_unit_sorting
            feature_units = np.array(list(feature_component_stats.keys()))
            feature_selective_units = np.array(list(feature_selective_stats.keys()))
            feature_non_selective_units = np.setdiff1d(feature_units, feature_selective_units)
            
            # ---
            feature_strong_sensitive_mi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'strong_encode_sensitive_mi' in feature_selective_stats[_]])
            feature_weak_sensitive_mi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'weak_encode_sensitive_mi' in feature_selective_stats[_] and 'strong_encode_sensitive_mi' not in feature_selective_stats[_]])
            feature_merged_sensitive_mi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'merged_encode_sensitive_mi' in feature_selective_stats[_] and 'strong_encode_sensitive_mi' not in feature_selective_stats[_] and 'weak_encode_sensitive_mi' not in feature_selective_stats[_]])
            
            feature_strong_sensitive_wmi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'strong_encode_sensitive_wmi' in feature_selective_stats[_]])
            feature_weak_sensitive_wmi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'weak_encode_sensitive_wmi' in feature_selective_stats[_] and 'strong_encode_sensitive_wmi' not in feature_selective_stats[_]])
            feature_merged_sensitive_wmi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'merged_encode_sensitive_wmi' in feature_selective_stats[_] and 'strong_encode_sensitive_wmi' not in feature_selective_stats[_] and 'weak_encode_sensitive_wmi' not in feature_selective_stats[_]])
            
            # ---
            feature_strong_sensitive_si_idx = np.array([_ for _ in feature_selective_stats.keys() if 'strong_encode_sensitive_si' in feature_selective_stats[_]])
            feature_weak_sensitive_si_idx = np.array([_ for _ in feature_selective_stats.keys() if 'weak_encode_sensitive_si' in feature_selective_stats[_] and 'strong_encode_sensitive_si' not in feature_selective_stats[_]])
            feature_merged_sensitive_si_idx = np.array([_ for _ in feature_selective_stats.keys() if 'merged_encode_sensitive_si' in feature_selective_stats[_] and 'strong_encode_sensitive_si' not in feature_selective_stats[_] and 'weak_encode_sensitive_si' not in feature_selective_stats[_]])
            
            feature_strong_sensitive_wsi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'strong_encode_sensitive_wsi' in feature_selective_stats[_]])
            feature_weak_sensitive_wsi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'weak_encode_sensitive_wsi' in feature_selective_stats[_] and 'strong_encode_sensitive_wsi' not in feature_selective_stats[_]])
            feature_merged_sensitive_wsi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'merged_encode_sensitive_wsi' in feature_selective_stats[_] and 'strong_encode_sensitive_wsi' not in feature_selective_stats[_] and 'weak_encode_sensitive_wsi' not in feature_selective_stats[_]])
            
            feature_unit_sorting_dict = {
                'feature_non_selective_units': feature_non_selective_units,
                
                'feature_strong_sensitive_mi_idx': feature_strong_sensitive_mi_idx,
                'feature_weak_sensitive_mi_idx': feature_weak_sensitive_mi_idx,
                'feature_merged_sensitive_mi_idx': feature_merged_sensitive_mi_idx,
                
                'feature_strong_sensitive_wmi_idx': feature_strong_sensitive_wmi_idx,
                'feature_weak_sensitive_wmi_idx': feature_weak_sensitive_wmi_idx,
                'feature_merged_sensitive_wmi_idx': feature_merged_sensitive_wmi_idx,
                
                'feature_strong_sensitive_si_idx': feature_strong_sensitive_si_idx,
                'feature_weak_sensitive_si_idx': feature_weak_sensitive_si_idx,
                'feature_merged_sensitive_si_idx': feature_merged_sensitive_si_idx,
                
                'feature_strong_sensitive_wsi_idx': feature_strong_sensitive_wsi_idx,
                'feature_weak_sensitive_wsi_idx': feature_weak_sensitive_wsi_idx,
                'feature_merged_sensitive_wsi_idx': feature_merged_sensitive_wsi_idx
                }
            
            # -----
            results = {
                'original_results': pl,
                
                'preliminary_p_masks': {_:pl[_]['preliminary_p_mask'] for _ in units if pl[_] is not None},
                'qualified_p_masks': {_:pl[_]['qualified_p_mask'] for _ in units if pl[_] is not None},
                
                'feature_selective_stats': feature_selective_stats,
                
                'feature_component_stats': feature_component_stats,
                
                'feature_unit_sorting_dict': feature_unit_sorting_dict
                }
            
            utils_.pickle_dump(save_path, results)
            
        # -----
        pl = results['original_results']
        feature_unit_sorting_dict = results['feature_unit_sorting_dict']
        feature_component_stats = results['feature_component_stats']
        
        self.single_unit_folder = os.path.join(self.tsne_save_folder, 'Single Unit Plot')
        utils_.make_dir(self.single_unit_folder)
        
        colors = [plt.get_cmap('jet', 50)(i) for i in range(50)]
        
        for plot_type in feature_unit_sorting_dict.keys():     # foe each type
        
            plot_types_idces = feature_unit_sorting_dict[plot_type]
            
            if len(plot_types_idces) != 0:
        
                for unit in plot_types_idces:
                
                    Selectivity_Analysis_Feature.Selectivity_Analysis_Feature.plot_region_based_coding_single_unit(unit, tsne, feature, feature_component_stats[unit], 
                                                     layer='neuron_1', img_labels=img_labels, colors=colors, num_classes=50, plot_type_folder=self.single_unit_folder)
        
        # --------------------------------------------------------------------------------------------------------------
        self.sample_folder = os.path.join(self.tsne_save_folder, 'sample figs')
        utils_.make_dir(self.sample_folder)
        
        for type_ in cell_stats['cell_types_dict'].keys():
            
            if cell_stats['cell_types_dict'][type_].size > 0:
                
                # ---
                #cell = np.random.choice(cell_stats['cell_types_dict'][type_])
                
                # ---
                if type_ == 'sensitive_si':
                    cell = 127
                elif type_ == 'sensitive_wsi':
                    cell = 1121
                elif type_ == 'sensitive_mi':
                    cell = 56
                elif type_ == 'sensitive_wmi':
                    cell = 1287
                elif type_ == 'sensitive_non_encode':
                    cell = 998
                elif type_ == 'non_sensitive_wsi':
                    cell = 163
                elif type_ == 'non_sensitive_wmi':
                    cell = 1618
                elif type_ == 'non_sensitive_non_encode':
                    cell = 1525
            
                FR = meanFR[cell, :]
                
                # -----
                encoded_ids = np.append(cell_stats['encode_id'][cell]['encode'], cell_stats['encode_id'][cell]['weak_encode']).astype(int)
                
                fig, ax = plt.subplots(figsize=(10,10))
                
                DR_scatter(ax, tsne, img_labels, FR, encoded_ids)
                
                ax.set_title(f'Human Cells DR(TSNE) | Coordinates from: {coor_name} | Unit: {cell} | Type: {type_}')

                fig.tight_layout()
                fig.savefig(os.path.join(self.sample_folder, f'{type_} {cell}.png'))
                plt.close()



# ======================================================================================================================
def plot_single(fig, gs_main, num_types, idx_dict, feature, num_classes, num_samples, layer=None, percentile=99.):
    """ [notice] no auto-adjust for figure size, the proper figsize must be manually appointed """
    
    colorpool_jet = plt.get_cmap('jet', 50)
    colors = [colorpool_jet(i) for i in range(50)]
    
    tqdm_bar = tqdm(total=num_types, desc=f'{layer}')
    
    y_lim_min = np.min(feature)
    
    #y_lim_max = np.max(feature)     # this will extremely extend the radius by outliers
    y_lim_max = np.percentile(feature, percentile)

    y_lin_range = y_lim_max - y_lim_max
    
    num_cols = gs_main.ncols
    num_rows = gs_main.nrows
    
    for i in range(num_rows):
        for j in range(num_cols):
            
            unit_type = list(idx_dict.keys())[i*num_cols+j]
            
            gs_sub = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[4, 1], subplot_spec=gs_main[i, j])

            ax_left = fig.add_subplot(gs_sub[0])
            ax_right = fig.add_subplot(gs_sub[1])
            
            if (i+j) != 0:
                ax_left.set_xticks([])
                ax_left.set_yticks([])

            if idx_dict[unit_type].size == 0:
                ax_left.set_title(unit_type + ' [0.00%]')
                ax_right.set_title('th')

            else:
                feature_test = feature[:, idx_dict[unit_type]]     # (500, num_units)

                feature_test_mean = np.mean(feature_test.reshape(num_classes, num_samples, -1), axis=1)     # (50, num_units)
                
                num_units = len(idx_dict[unit_type])
                
                # -----
                x = np.tile(np.arange(num_classes), num_units)     # (0,1,...,49,0,1,...)
                y = feature_test_mean.T.reshape(-1)     # every 50 ids for unit by unit
                
                c = np.tile(np.array(colors), [num_units, 1])

                # -----
                ax_left.scatter(x, y, color=c, alpha=0.75, marker='.', s=1)     # use small size to replace adjustable alpha
                # -----
                
                #pct = num_units/feature.shape[1]*100
                pct = num_units/1577*100
                
                ax_left.set_title(unit_type + f' [{pct:.2f}%]')
                # -----
                
                # ----- stats: mean firing rate for each id
                values = feature_test_mean.reshape(-1)    # (50*num_units)
                
                plot_single_subsubplot(ax_left, ax_right, values, color='blue')
                
                # ----- stats: threshold (mean+2std of all 500 values)
                values = np.mean(feature_test, axis=0) + 2*np.std(feature_test, axis=0)     # (num_units,)

                plot_single_subsubplot(ax_left, ax_right, values, linestyle='dotted', color='red')
                
                # ----- stats: ref (mean+2std of all 50 mean values)
                values = np.mean(feature_test_mean, axis=0) + 2*np.std(feature_test_mean, axis=0)     # (num_units,)
                
                plot_single_subsubplot(ax_left, ax_right, values, linestyle='dotted', color='teal')
                
                # -----
                scaling_factor = 0.1
                
                ax_left.set_ylim([y_lim_min - y_lin_range*scaling_factor, y_lim_max + y_lin_range*scaling_factor])
                ax_right.set_ylim([y_lim_min - y_lin_range*scaling_factor, y_lim_max + y_lin_range*scaling_factor])
                ax_right.set_title('PDF')
                ax_right.set_yticks([])
                
            tqdm_bar.update(1)


def plot_single_subsubplot(ax_left, ax_right, values, linestyle=None, color=None, scaling_factor=0.1):
    
    if np.std(values) == 0:
        pass
    else:
        kde = gaussian_kde(values)
        
        min_values = np.min(values)
        max_values = np.max(values)
        
        values_range = max_values - min_values
        
        x_vals = np.linspace(min_values - scaling_factor*values_range, max_values + scaling_factor*values_range, 101)
        y_vals = kde(x_vals)
        ax_right.plot(y_vals, x_vals, linestyle=linestyle, color=color)
    
        y_vals_max = np.max(y_vals)
        
        if len(y_peak:=np.where(y_vals==y_vals_max)[0]) == 1:
            x_vals_max = x_vals[y_peak.item()]
        else:
            x_vals_max = x_vals[y_peak[0]]
        
        ax_left.hlines(x_vals_max, 0, 50, colors=color, alpha=0.75, linestyle='--')
        ax_right.hlines(x_vals_max, np.min(y_vals), np.max(y_vals), colors=color, alpha=0.75, linestyle='--')


def DR_scatter(ax, tsne, img_labels, weights, encoded_ids):
    
    x = tsne[:, 0] - np.min(tsne[:, 0])
    y = tsne[:, 1] - np.min(tsne[:, 1])
    
    colors = [plt.get_cmap('jet', 50)(i) for i in range(50)]
    
    if np.sum(weights) == 0 or np.sum(weights!=0) ==1:
        for gg in range(1,51):  # this can be changed to different types of id
            current_scatter = ax.scatter(x[img_labels == gg], y[img_labels == gg], s=5, color=colors[gg-1], alpha=0.5)
            
    else:
        
        size_weight = weights / max(weights)     # [notice] can not divide by 0 if all values are 0
        sizes = np.ones(500) * 15 * (1 + 20 * size_weight)
        
        for gg in range(1,51):  # this can be changed to different types of id
            current_scatter = ax.scatter(x[img_labels == gg], y[img_labels == gg], s=sizes[img_labels==gg], color=colors[gg-1], alpha=0.5)
        
        # -----
        if len(encoded_ids) > 0:
        
            handles_featured = []
            labels = []
            
            for gg in encoded_ids:
                current_scatter = ax.scatter(x[img_labels == gg], y[img_labels == gg], s=sizes[img_labels == gg], color=colors[gg-1], alpha=0.7)
                handles_featured.append(current_scatter)
                labels.append(f'{gg}')
            
            ax.add_artist(ax.legend(handles=handles_featured, labels=labels, framealpha=0.5))


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    human_record_process = human_feature_process()
    
    human_record_process.plot_stacked_responses()
    human_record_process.plot_FR_PDF()
    
    human_record_process.calculation_DSM_perm_human(used_id_num=10, used_unit_type='qualified')
    human_record_process.calculation_Gram_perm_human(used_unit_type='qualified')
        
    human_record_process.plot_piechart()

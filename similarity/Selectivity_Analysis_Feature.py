#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:44:28 2023

@author: Jinge Wang, Runnan Cao

    refer to: https://github.com/JingeW/ID_selective
              https://osf.io/824s7/
    
@modified: acxyle

    Task: (1) make the code clear and precise; (2) make the code computation and plot separate
    
    Status: under construction
    
    Update: basic function achieved, need to upgrade: (1) code clean; (2) parallel calculation

"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import logging

from tqdm import tqdm

import numpy as np
import scipy.ndimage

from scipy.stats import ttest_ind, gaussian_kde
from scipy.spatial.distance import pdist
from itertools import combinations

from joblib import Parallel, delayed

import utils_


# ----------------------------------------------------------------------------------------------------------------------
class Selectivity_Analysis_Feature():
    """
        this class contains a set of processes for feature-neuron, based on the principle of 
        "Feature-based encoding of face identity by single neurons in the human amygdala and hippocampus" 
        [link: https://www.biorxiv.org/content/10.1101/2020.09.01.278283v2.full]
        
        The source code is MATLAB format and provided by Dr. Runnan Cao. This class currently is the python version of 
        the source code
        
        [task]:
            
            1. rewrite this code based on single model analysis for local needed
            2. design a comparison between models, the code will be located here or design a special file to store all 
            functions related to models comparisons
    """
    
    def __init__(self, root='./Face Identity Baseline', DR_type='TSNE',
                 num_samples=10, num_classes=50, data_name=''):
        
        assert root[-1] != '/', f"[Codinfo] root {root} should not end with '/'"

        # --- overall variables
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.DR_type = DR_type
        
        def _select_layers_and_units(model_structure):
            
            if 'spiking' in model_structure.lower() or 'sew'  in model_structure.lower():
                act = 'sn'
            else:
                act = 'an'
            
            if 'resnet18' in model_structure.lower() or 'resnet34' in model_structure.lower():
                layers = [f'L4_B2_{act}_1', f'L4_B2_{act}_2']
                neurons = [25088, 25088, 512, 50]
            elif 'resnet' in model_structure.lower():
                layers = [f'L4_B3_{act}_2', f'L4_B3_{act}_3']
                neurons = [25088, 100352, 512, 50]
            elif 'vgg' in model_structure.lower():
                layers = [f'L5_B3_{act}', f'{act}_1', f'{act}_2']
                neurons = [100352, 4096, 4096]
                
            return layers, neurons
        
        # -----
        self.root = os.path.join(root, 'Features/')     # <- folder for feature maps, which should be generated before analysis
        if not os.path.exists(self.root):
            try:
                self.root = os.path.join(root, 'Features(spike)')     # FIXME --- this should triger the branch of process of spike trains
                os.path.exists(self.root)
            except:
                raise RuntimeWarning(f'[Codwarning] can not find the root of [{self.root}]')
                
        self.dest = os.path.join(root, 'Analysis')     # <- folder for analysis results
        utils_.make_dir(self.dest)
        
        self.dest_Feature = os.path.join(self.dest, 'Feature')
        utils_.make_dir(self.dest_Feature)
        
        self.feature_list = [os.path.join(self.root, _) for _ in sorted(os.listdir(self.root)) if 'pkl' in _]     # feature .pkl list
        
        self.model_structure = root.split('/')[-1].split(' ')[-1]
        
        self.layers, self.neurons = _select_layers_and_units(self.model_structure)
        
        # --- legacy, local variables
        self.taskInstruction = 'CelebA'
        
        if self.taskInstruction == 'ImageNet':
            self.nSD = 1.8
            self.scaling_factor = 0.021
            self.maskFactor = 0.5
        elif self.taskInstruction == 'CoCo':
            self.nSD = 1.5
            self.maskFactor = 0.3
        elif self.taskInstruction == 'CelebA':
            self.nSD = 4
            self.scaling_factor = 0.035
            self.maskFactor = 0.1
            
    # ------------------------------------------------------------------------------------------------------------------
    # FIXME --- need to simplify and upgrade
    def feature_analysis(self, ):
        """
            [notice] this design relies on RAM space
        """
        # -----
        self.dest_Encode = os.path.join(self.dest, 'Encode')
        
        self.id_label = utils_.lexicographic_order(self.num_classes)
        self.img_labels = np.array([[self.id_label[_]]*10 for _ in range(50)]).reshape(-1)
        
        if not hasattr(self, 'Sort_dict'):
            self.Sort_dict = utils_.load(os.path.join(self.dest_Encode, 'Sort_dict.pkl'))

        if not hasattr(self, 'Encode_dict'):
            self.Encode_dict = utils_.load(os.path.join(self.dest_Encode, 'Encode_dict.pkl'))
            self.Encode_dict = {_:self.Encode_dict[_] for _ in self.layers}
            
        # --- depends 
        self.DR_dicts = utils_.load(os.path.join(self.dest, f'Dimension_Reduction/{self.DR_type}/{self.DR_type.lower()}_all.pkl'), verbose=False)
        utils_._print(f"len(self.DR_dicts.keys()) is '{len(self.DR_dicts.keys())}'")
        
        # ---            
        for (self.layer, self.neuron) in zip(self.layers, self.neurons):     # for each layer
            
            # --- data preparation
            self.layer_folder = os.path.join(self.dest_Feature, f'{self.layer}')
            utils_.make_dir(self.layer_folder)
            
            # --- encode and correction
            encode_dict = self.Encode_dict[self.layer]
            self.encode_dict = {_: {'encode': self.id_label[encode_dict[_]['encode']-1], 'weak_encode': self.id_label[encode_dict[_]['weak_encode']-1] } for _ in encode_dict.keys()}
            self.sort_dict = self.Sort_dict[self.layer]
            
            self.feature = utils_.load_feature(os.path.join(self.root, self.layer+'.pkl'), verbose=False)
  
            # --- the used coordinates
            for DR_coordinate in [
                                'all',
                                #'s_si', 's_wsi', 's_mi', 's_wmi', 's_non_encode', 
                                #'ns_si', 'ns_wsi', 'ns_mi', 'ns_wmi', 'ns_non_encode'
                                ]:
                
                # --- [test version] size judgement
                DR_dicts=f'{self.DR_type.lower()}_dict'
                reduced_features = self.DR_dicts[self.layer][DR_dicts][DR_coordinate]
                max_distance = pdist([[np.min(reduced_features[:, 0]), np.min(reduced_features[:,1])], [np.max(reduced_features[:, 0]), np.max(reduced_features[:,1])]], 'euclidean')
                
                if 9 < max_distance and max_distance < 100:
                    
                    utils_._print(f'Processing [{DR_coordinate}]...')
                    
                    self.layer_folder_DR_coordinate = os.path.join(self.layer_folder, f'coordinate_{DR_coordinate}')
                    utils_.make_dir(self.layer_folder_DR_coordinate)
                
                    unit_type_and_reduced_feature = (DR_coordinate, self.DR_dicts[self.layer][DR_dicts][DR_coordinate])
                    
                    self.feature_analysis_single_layer(unit_type_and_reduced_feature)
                
                else:
                    
                    utils_._print(f"The diagnal distance of coordinate space '{DR_coordinate}' is smaller than 9 or greater than 100.", '[Codwarning]')
        
        utils_._print('Feature analysis completed')
        
        
    # ------------------------------------------------------------------------------------------------------------------
    def feature_analysis_single_layer(self, unit_type_and_reduced_feature):    
        """
            [task] divide calculation and plot
        """
        
        # ----- init
        ...
        
        # ----- 1.1. calculate p_values for permutation test
        p_results = self.generate_p_values(unit_type_and_reduced_feature)
        
        # ----- 1.2. calculate feature units
        feature_unit_results = self.feature_region_selection(unit_type_and_reduced_feature, p_results)
        
        # --- legacy design
        #from scipy.stats import binom
        #bionorP = 1 - binom.cdf(len(feature_unit_results['feature_component_stats']), len(units), 0.05)
        
        # ----- 2. feature plot
        self.feature_analysis_plot(unit_type_and_reduced_feature, feature_unit_results)
        
        
    # ------------------------------------------------------------------------------------------------------------------
    # FIXME --- need to upgrade and simplify
    def feature_analysis_plot(self, unit_type_and_reduced_feature, feature_unit_results):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        # ----- 1. single unit
        self.single_unit_folder = os.path.join(self.layer_folder_DR_coordinate, 'Single Unit Plot')
        utils_.make_dir(self.single_unit_folder)
        
        self.feature_coding_plot(unit_type_and_reduced_feature, feature_unit_results)
        
        # ----- 2. population level
        self.population_folder = os.path.join(self.layer_folder_DR_coordinate, 'Population Plot')
        utils_.make_dir(self.population_folder)
        
        feature_unit_sorting_dict = feature_unit_results['feature_unit_sorting_dict']
        plot_types_dict = {_: feature_unit_sorting_dict[_] for _ in feature_unit_sorting_dict.keys() if feature_unit_sorting_dict[_].size != 0} 
       
        # ----- feature units comparisons
        # --- 2.1 size
        self.feature_folder = os.path.join(self.population_folder, 'in Feature')
        utils_.make_dir(self.feature_folder)
        
        self.size_folder = os.path.join(self.feature_folder, 'Size')
        utils_.make_dir(self.size_folder)
        
        for target_cluster_type in ['max']:
            
            feature_cluster_stats, region_stats = self.population_feature_size(unit_type_and_reduced_feature, plot_types_dict, feature_unit_results, target_cluster_type)
            
            # --- feature_unit analysis
            self.plot_overlapped_receptive_field(unit_type_and_reduced_feature, feature_cluster_stats, target_cluster_type)
            
            self.plot_sizes_box(unit_type_and_reduced_feature, feature_cluster_stats, target_cluster_type)
            
            # --- overall_unit analysis
            original_types_dict, region_types_dict = self.region_units_analysis(feature_cluster_stats, region_stats, )

            self.region_folder = os.path.join(self.population_folder, 'in Region')
            utils_.make_dir(self.region_folder)
            
            self.region_size_folder = os.path.join(self.region_folder, 'Size')
            utils_.make_dir(self.region_size_folder)
            
            for type_dict in [original_types_dict, region_types_dict]:
                
                self.plot_overlapped_receptive_field_regions(unit_type_and_reduced_feature, feature_unit_results, type_dict, target_cluster_type)

                self.plot_sizes_box_region(unit_type_and_reduced_feature, feature_unit_results, type_dict, target_cluster_type)
            

        # --- 2.2. distance
        in_feature_values = self.plot_distance_figures(unit_type_and_reduced_feature, plot_types_dict, feature_unit_results)
        
        # --- overall_unit analysis

        self.region_distance_folder = os.path.join(self.region_folder, 'Distance')
        utils_.make_dir(self.region_distance_folder)
        
        encode_types = ['merged']
        
        for type_dict in [original_types_dict, region_types_dict]: 
        
            for encode_type in encode_types:
                
                    distance_stats = self.calculate_distance(unit_type_and_reduced_feature, feature_unit_results, encode_type=encode_type, in_region=False)  
                    
                    groups = {_: [__ for __ in type_dict[_][0].keys() if __ in distance_stats['distance_dict'].keys()] for _ in type_dict.keys()}
                    
                    in_region_values = self.plot_distance_box_region(unit_type_and_reduced_feature, groups, distance_stats, interested_type_list=['fmi'])
                
        rank_stats = {'in_feature': {
            'size': feature_cluster_stats,
            'distance': in_feature_values
            },
        
        'in_region': {
            'size': region_types_dict,
            'distance': in_region_values
            }}
        
        # ---
        rank_stats_path = os.path.join(self.layer_folder_DR_coordinate, 'rank_stats.pkl')
        
        if not os.path.exists(rank_stats_path):
            utils_.dump(rank_stats, rank_stats_path)    
            

    def plot_distance_box_region(self, unit_type_and_reduced_feature, groups, distance_stats, interested_type_list=['feature_strong_s_mi_idx']):
        
        DR_coordinate = unit_type_and_reduced_feature[0]
        
        distance_dict = distance_stats['distance_dict']
        in_region = distance_stats['in_region']
        encode_type = distance_stats['encode_type']
        
        group_values = [ [distance_dict[__] for __ in groups[_] if __ in distance_dict.keys()] for _ in groups.keys()]
        
        fig, ax = plt.subplots(figsize=(10,10))
        
        flierprops = dict(marker='.', markerfacecolor='red', markersize=1, linestyle='none', markeredgecolor='red')
        ax.boxplot(group_values, notch=True, flierprops=flierprops)
        
        # Formatting plot
        ax.set_xticks(np.arange(1, len(groups)+1))
        
        group_ticks = [f'{_} ({len(group_values[idx])})' for idx, _ in enumerate(groups.keys())]
        
        ax.set_xticklabels(group_ticks, rotation=90)
        
        ax.set_ylabel('Normalized Distance')
        ax.set_title(f'Layer: {self.layer} | Coordinates from: {DR_coordinate} | Encode type: {encode_type} | In region: {in_region} | Types {len(groups)} | Distances')
        ax.grid(axis='y', linestyle='--')
        
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        group_idcs = np.arange(len(groups.keys()))
        groups_combination = list(combinations(group_idcs, 2))
        
        # ---
        if interested_type_list != None:
            if len(interested_type_list) == 1:
                interested_type = interested_type_list[0]
                groups_combination = [_ for _ in groups_combination if list(groups.keys()).index(interested_type) in _]
            else:
                raise RuntimeError(f'[Coderror] Do not support multiple [{interested_type}] in current code')
        
        # ---
        p_value_list = []
        
        for idx, _ in enumerate(groups_combination):
            ttest_stat, ttest_p = ttest_ind(group_values[_[0]], group_values[_[1]])
            p_value_list.append(ttest_p)
            groups_combination[idx] = np.array(_) + 1
        
        utils_.sigstar(groups_combination, p_value_list, ax)
        
        # ---
        plt.savefig(os.path.join(self.region_distance_folder, f'{self.layer}_distance_{encode_type}_{in_region} {len(groups)} types.png'), bbox_inches='tight')
        plt.savefig(os.path.join(self.region_distance_folder, f'{self.layer}_distance_{encode_type}_{in_region} {len(groups)} types.eps'), bbox_inches='tight')
        plt.close()
        
        return group_values
    
    # ------------------------------------------------------------------------------------------------------------------
    def plot_sizes_box_region(self, unit_type_and_reduced_feature, feature_unit_results, original_types_dict, target_cluster_type):
        
        interested_type_list = ['fmi']
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        flierprops = dict(marker='.', markerfacecolor='red', markersize=1, linestyle='none', markeredgecolor='red')
        
        feature_cluster_size_groups = [list(_[0].values()) for _ in list(original_types_dict.values())]
        ax.boxplot(feature_cluster_size_groups, notch=True, flierprops=flierprops)
        
        group_ticks = [f'{_} ({len(feature_cluster_size_groups[idx])})' for idx, _ in enumerate(original_types_dict.keys())]
        
        ax.set_xticklabels(group_ticks, rotation=90)
        
        ax.set_title(f"Layer: {self.layer} ({len(feature_unit_results['preliminary_p_masks'])}/{self.neuron}) | Coordinates from: {unit_type_and_reduced_feature[0]} | Types {len(original_types_dict)} | Region size ({target_cluster_type}) distribution")
        ax.set_ylabel('Percentage of Feature Space(%)')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(axis='y', linestyle='--')
        
        group_idcs = np.arange(len(original_types_dict.keys()))
        groups = list(combinations(group_idcs, 2))
        
        if interested_type_list != None:
            if len(interested_type_list) == 1:
                interested_type = interested_type_list[0]
                groups = [_ for _ in groups if list(original_types_dict.keys()).index(interested_type) in _]
            else:
                raise RuntimeError(f'[Coderror] Do not support multiple [{interested_type}] in current code')
        
        p_value_list = []
        
        for idx, _ in enumerate(groups):
            ttest_stat, ttest_p = ttest_ind(feature_cluster_size_groups[_[0]], feature_cluster_size_groups[_[1]])
            p_value_list.append(ttest_p)
            groups[idx] = np.array(_) + 1
            
        utils_.sigstar(groups, p_value_list, ax)
        
        fig.savefig(os.path.join(self.region_size_folder, f'{self.layer}_{target_cluster_type}_size {len(original_types_dict)} types.png'), bbox_inches='tight')
        fig.savefig(os.path.join(self.region_size_folder, f'{self.layer}_{target_cluster_type}_size {len(original_types_dict)} types.eps'), bbox_inches='tight')
        plt.close()
    
    
    def plot_overlapped_receptive_field_regions(self, unit_type_and_reduced_feature, feature_unit_results, original_types_dict, target_cluster_type):
        
        fig, ax = plt.subplots(1, len(original_types_dict), figsize=(3*len(original_types_dict)+3, 4))       

        vmin = np.min([np.min(original_types_dict[_][1]) for _ in original_types_dict.keys()])
        vmax = np.max([np.max(original_types_dict[_][1]) for _ in original_types_dict.keys()])
        
        for idx, _ in enumerate(original_types_dict.keys()):  # for each unit
        
            if isinstance(original_types_dict[_][1], int):
                pass
            else:
                cax = ax[idx].imshow(original_types_dict[_][1], aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
            
            avg_intensity = original_types_dict[_][1]/(vmax-vmin)
            
            ax[idx].set_title(f"units: {len(original_types_dict[_][0])} /{len(feature_unit_results['preliminary_p_masks'])} | avg intensity: {np.mean(avg_intensity[avg_intensity>0]):.3f}")

            ax[idx].set_xlabel(_)
            ax[idx].tick_params(axis='both', which='major', labelsize=12)
            
            if idx != 0:
                ax[idx].set_xticks([])
                ax[idx].set_yticks([])
        
        fig.colorbar(cax, cax=fig.add_axes([1.02, 0.15, 0.01, 0.7]))     # [left, bottom, width, height]
        fig.suptitle(f'Layer: {self.layer} ({self.neuron}) | Coordinates from: {unit_type_and_reduced_feature[0]} | Types {len(original_types_dict)} | Overlapped regions ({target_cluster_type})')
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            #fig.tight_layout(rect=[0, 0, 0.9, 1])
            fig.tight_layout()
            
            fig.savefig(os.path.join(self.region_size_folder, f'{self.layer}_covered_area {len(original_types_dict)} types ({target_cluster_type}).png'), bbox_inches='tight')
            fig.savefig(os.path.join(self.region_size_folder, f'{self.layer}_covered_area {len(original_types_dict)} types ({target_cluster_type}).eps'), bbox_inches='tight')
            plt.close()
    
    
    # FIXME --- not all models have below types
    def region_units_analysis(self, feature_cluster_stats, region_stats, ):
        si = region_stats['s_si']
        
        try:
            fmi = feature_cluster_stats['feature_strong_s_mi_idx']
        except:
            fmi = ({}, 0)
        
        tmp = {_:feature_cluster_stats[_][0] for _ in feature_cluster_stats.keys() if 'weak_s_mi' in _ or 'merged_s_mi' in _}
        combined_dict = {k: v for sub_dict in tmp.values() for k, v in sub_dict.items()}
        
        tmp_2 = {_:feature_cluster_stats[_][1] for _ in feature_cluster_stats.keys() if 'weak_s_mi' in _ or 'merged_s_mi' in _}
        tmp2 = np.sum(np.array(list(tmp_2.values())), axis=0)
        
        nfmi = (
            {
            **combined_dict,
            **region_stats['s_mi_others'][0]
            }, 
            
            tmp2 + 
            region_stats['s_mi_others'][1]
            )
        
        # ---
        original_types_dict = {
            'fmi': fmi,
            'nfmi': nfmi,
            'si': si
            }
        
        # ---
        mi_others = region_stats['s_mi_others']
        
        try:
            fwsi = feature_cluster_stats['feature_weak_s_si_idx']    
        except:
            fwsi = ({}, 0)
        try:
            fmsi = feature_cluster_stats['feature_merged_s_si_idx']     
        except:
            fmsi = ({}, 0)
            
        si_others = {_:si[0][_] for _ in si[0].keys() if _ not in fwsi[0].keys() and _ not in fmsi[0].keys()}     # 909
        si_others_maps = si[1] - fwsi[1] - fmsi[1]
        
        si_others = (si_others, si_others_maps)
        
        # --- feature_non_selective(excluded) + feature_weak_s_smi_idx + others(have no feature_resions)
        try:
            fns = feature_cluster_stats['feature_non_selective_units']    
        except:
            fns = ({}, 0)
        try:
            fwwmi = feature_cluster_stats['feature_weak_s_wmi_idx']
        except:
            fwwmi = ({}, 0)
        
        others_f = (
            {**{_:fns[0][_] for _ in fns[0].keys() if _ not in mi_others[0].keys() and _ not in si[0].keys()},
            **{_:fwwmi[0][_] for _ in fwwmi[0].keys() if _ not in mi_others[0].keys() and _ not in si[0].keys()}},
            fns[1]+fwwmi[1]     # 不是非常严格
                    )
        others_nf = region_stats['others_']     # this should come from regions_stats
        
        others = (
            {**others_f[0], **others_nf[0]},
            others_f[1]+others_nf[1]
            )
        
        region_types_dict = {
            'fmi': fmi,     # those 4 are all mi units
            'mi_others': mi_others,
            
            'fwsi': fwsi,     # those 3 are all si units - can not use 'si' because different calculation rules if want to collect the stats
            'fmsi': fmsi,
            'si_others': si_others,
            
            'others': others
            }
        
        if 'feature_weak_s_mi_idx' in feature_cluster_stats.keys():
            region_types_dict['fwmi'] = feature_cluster_stats['feature_weak_s_mi_idx']
        if 'feature_merged_s_mi_idx' in feature_cluster_stats.keys():
            region_types_dict['fmmi'] = feature_cluster_stats['feature_merged_s_mi_idx']
        
        region_types_dict = {_:region_types_dict[_] for _ in sorted(region_types_dict)}
        
        return original_types_dict, region_types_dict
    # ------------------------------------------------------------------------------------------------------------------
    
    
    # -----
    def plot_overlapped_receptive_field(self, unit_type_and_reduced_feature, feature_cluster_stats, target_cluster_type):
        """
            [question] looks like mark the pct of covered range is not a reasonable value...
        """
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        DR_coordinate = unit_type_and_reduced_feature[0]
        
        if not len(feature_cluster_stats) > 0:
            
            return
            
        else:
        
            fig, ax = plt.subplots(1, len(feature_cluster_stats), figsize=(3*len(feature_cluster_stats)+3, 4))       
            
            if len(feature_cluster_stats) == 1:
                
                ax = np.array([ax])
            
            vmin = np.min([np.min(feature_cluster_stats[_][1]) for _ in feature_cluster_stats.keys()])
            vmax = np.max([np.max(feature_cluster_stats[_][1]) for _ in feature_cluster_stats.keys()])
            
            for idx, _ in enumerate(feature_cluster_stats.keys()):  # for each unit

                cax = ax[idx].imshow(feature_cluster_stats[_][1], aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
                
                avg_intensity = feature_cluster_stats[_][1]/(vmax-vmin)
                
                ax[idx].set_title(f'units pct: {len(feature_cluster_stats[_][0])/self.neuron*100:.2f}% | avg intensity: {np.mean(avg_intensity[avg_intensity>0]):.3f}')
                #ax[idx].set_title(f'{np.sum(feature_cluster_stats[_][1]>0)/feature_cluster_stats[_][1].size*100:.2f}%')
                
                ax[idx].set_xlabel(_)
                ax[idx].tick_params(axis='both', which='major', labelsize=12)
                
                if idx != 0:
                    ax[idx].set_xticks([])
                    ax[idx].set_yticks([])
            
            fig.colorbar(cax, cax=fig.add_axes([1.02, 0.15, 0.01, 0.7]))     # [left, bottom, width, height]
            fig.suptitle(f'Layer: {self.layer} | Coordinates from: {DR_coordinate} | Overlapped regions ({target_cluster_type})')
            
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                
                #fig.tight_layout(rect=[0, 0, 0.9, 1])
                fig.tight_layout()
                
                fig.savefig(os.path.join(self.size_folder, f'{self.layer}_covered_area ({target_cluster_type}).png'), bbox_inches='tight')
                fig.savefig(os.path.join(self.size_folder, f'{self.layer}_covered_area ({target_cluster_type}).eps'), bbox_inches='tight')
                plt.close()
            
            # --- fourier analysis
            #f = np.fft.fft2(feature_cluster_stats['feature_weak_s_mi_idx'][1])
            #fshift = np.fft.fftshift(f)
            #magnitude_spectrum = 20*np.log(np.abs(fshift))
        
        
    # -----
    def plot_sizes_box(self, unit_type_and_reduced_feature, feature_cluster_sizes:dict, target_cluster_type='max', interested_type_list=['feature_strong_s_mi_idx']) -> None:
        """
            the input is a list in which each element contains numbers of area sizes of each unit of certain unit type,
            if the interested_type==None, then it will plot all combinations, be careful when too many types.
        """
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        DR_coordinate = unit_type_and_reduced_feature[0]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        flierprops = dict(marker='.', markerfacecolor='red', markersize=1, linestyle='none', markeredgecolor='red')
        
        feature_cluster_size_groups = [list(_[0].values()) for _ in list(feature_cluster_sizes.values())]
        
        ax.boxplot(feature_cluster_size_groups, notch=True, flierprops=flierprops)
        
        group_ticks = [f'{_} ({len(feature_cluster_size_groups[idx])})' for idx, _ in enumerate(feature_cluster_sizes.keys())]
        
        ax.set_xticklabels(group_ticks, rotation=90)
        
        ax.set_title(f'Layer: {self.layer} | Coordinates from: {DR_coordinate} | Region size ({target_cluster_type}) distribution')
        ax.set_ylabel('Percentage of Feature Space(%)')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(axis='y', linestyle='--')
        
        # -----
        group_idcs = np.arange(len(feature_cluster_sizes.keys()))
        groups = list(combinations(group_idcs, 2))
        
        if interested_type_list != None:
            if len(interested_type_list) == 1:
                
                interested_type = interested_type_list[0]
                
                if interested_type in feature_cluster_sizes.keys():
                
                    groups = [_ for _ in groups if list(feature_cluster_sizes.keys()).index(interested_type) in _]
                    
                    p_value_list = []
                    
                    for idx, _ in enumerate(groups):
                        ttest_stat, ttest_p = ttest_ind(feature_cluster_size_groups[_[0]], feature_cluster_size_groups[_[1]])
                        p_value_list.append(ttest_p)
                        groups[idx] = np.array(_) + 1
                    
                    if len(groups) != 0:
                        utils_.sigstar(groups, p_value_list, ax)
                    
                else:
                    
                    print(f'[Codinfo] [{interested_type}] not in current reduced_feature coordinates')
                
                fig.savefig(os.path.join(self.size_folder, f'{self.layer}_{target_cluster_type}_size.png'), bbox_inches='tight')
                fig.savefig(os.path.join(self.size_folder, f'{self.layer}_{target_cluster_type}_size.eps'), bbox_inches='tight')
                plt.close()
            
            else:
                raise RuntimeError(f'[Coderror] Do not support multiple [{interested_type}] in current code')
        
    # -----
    def plot_distance_figures(self, unit_type_and_reduced_feature, plot_types_dict, feature_unit_results):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        self.distance_folder = os.path.join(self.feature_folder, 'Distance')
        utils_.make_dir(self.distance_folder)
        
        encode_types = ['merged']
        in_regions = [False]
        
        for encode_type in encode_types:
            
            for in_region in in_regions:
        
                distance_stats = self.calculate_distance(unit_type_and_reduced_feature, feature_unit_results, encode_type=encode_type, in_region=in_region)  
                
                groups = {_:plot_types_dict[_] for _ in plot_types_dict.keys() if len(plot_types_dict[_]) != 0}
                
                group_values = self.plot_distance_box(unit_type_and_reduced_feature, groups, distance_stats)
        
        return group_values
        

    def plot_distance_box(self, unit_type_and_reduced_feature, groups, distance_stats, interested_type_list=['feature_strong_s_mi_idx']):
        
        DR_coordinate = unit_type_and_reduced_feature[0]
        
        distance_dict = distance_stats['distance_dict']
        in_region = distance_stats['in_region']
        encode_type = distance_stats['encode_type']
        
        group_values = [ [distance_dict[__] for __ in groups[_] if __ in distance_dict.keys()] for _ in groups.keys()]
        
        flierprops = dict(marker='.', markerfacecolor='red', markersize=1, linestyle='none', markeredgecolor='red')
        
        fig, ax = plt.subplots(figsize=(10,10))
        
        ax.boxplot(group_values, notch=True, flierprops=flierprops)
        
        # Formatting plot
        ax.set_xticks(np.arange(1, len(groups)+1))
        
        group_ticks = [f'{_} ({len(group_values[idx])})' for idx, _ in enumerate(groups.keys())]
        
        ax.set_xticklabels(group_ticks, rotation=90)
        
        ax.set_ylabel('Normalized Distance')
        ax.set_title(f'Layer: {self.layer} | Coordinates from: {DR_coordinate} | Encode type: {encode_type} | In region: {in_region} | Distances')
        ax.grid(axis='y', linestyle='--')
        
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        group_idcs = np.arange(len(groups.keys()))
        groups_combination = list(combinations(group_idcs, 2))
        
        # ---
        if interested_type_list != None:
            if len(interested_type_list) == 1:
                interested_type = interested_type_list[0]
                if interested_type in distance_dict.keys():
                    groups_combination = [_ for _ in groups_combination if list(groups.keys()).index(interested_type) in _]
                    
                    # ---
                    p_value_list = []
                    
                    for idx, _ in enumerate(groups_combination):
                        ttest_stat, ttest_p = ttest_ind(group_values[_[0]], group_values[_[1]])
                        p_value_list.append(ttest_p)
                        groups_combination[idx] = np.array(_) + 1
                    
                    utils_.sigstar(groups_combination, p_value_list, ax)
                    
                else:
                    
                    print(f'[Codinfo] [{interested_type}] not in current reduced_feature coordinates')
                    
                # ---
                plt.savefig(os.path.join(self.distance_folder, f'{self.layer}_distance_{encode_type}_{in_region}.png'), bbox_inches='tight')
                plt.savefig(os.path.join(self.distance_folder, f'{self.layer}_distance_{encode_type}_{in_region}.eps'), bbox_inches='tight')
                plt.close()
                    
            else:
                raise RuntimeError(f'[Coderror] Do not support multiple [{interested_type}] in current code')
        
        return group_values
    
    # --- legacy design
    def plot_distance_bar(self, ax, groups, distance_dict, save_folder, layer, interested_type_list=['feature_strong_s_mi_idx']):
        
        stats_mean = []
        stats_std = []
        
        for _ in groups.keys():
            tmp = [distance_dict[__] for __ in groups[_] if __ in distance_dict.keys()]
            stats_mean.append(np.mean(tmp))
            stats_std.append(np.std(tmp))

        ax.bar(np.arange(1, len(groups)+1), stats_mean, width=0.5)
        
        ax.errorbar(np.arange(1, len(groups)+1), stats_mean, yerr=stats_std, fmt='.', capsize=8, linewidth=2, color='black')

        ax.set_xticks(np.arange(1, len(groups)+1))
        ax.set_xticklabels(list(groups.keys()), rotation=90)
        
        ax.set_ylabel('Normalized Distance')
        ax.set_title(f'{layer}')
        
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        plt.savefig(os.path.join(save_folder, f'{layer}.png'), bbox_inches='tight')
        plt.savefig(os.path.join(save_folder, f'{layer}.eps'), bbox_inches='tight')
        plt.close()
        

    #FIXME --- update this code for new version analysis 
    def calculate_distance(self, unit_type_and_reduced_feature, feature_unit_results, encode_type='merged', in_region=False):      # try to make a figure to illustrate this?
        """
            [question] why use encoded_id rather than featured_ids? - so use this dim of 'feature' units to analysis of the 'selective' units
            
            
            default version is 'feature irrelevant'
            
            add one more condition to only calculate the distance inside the region? but looks like this will make all distance similiar?
            
        """
        (DR_coordinate, reduced_feature) = unit_type_and_reduced_feature
        
        feature_component_stats = feature_unit_results['feature_component_stats']
        
        def _encode_type(unit_, encode_type):
            
            if encode_type == 'merged':
                
                return np.append(self.encode_dict[unit_]['encode'], self.encode_dict[unit_]['weak_encode'])
            
            elif encode_type == 'encode' or encode_type == 'weak_encode':
                
                return self.encode_dict[unit_][encode_type]
            
            else:
                raise ValueError(f'[Codinfo] {encode_type} is invalid')
        
        # --- init
        max_distance = pdist([[np.min(reduced_feature[:, 0]), np.min(reduced_feature[:, 1])], [np.max(reduced_feature[:, 0]), np.max(reduced_feature[:, 1])]], 'euclidean').item()     # normalization factor

        distance_dict = {}

        for unit in tqdm(np.arange(self.neuron), desc=f'encode_type: {encode_type} in_region: {in_region}'):
            
            encode_id = _encode_type(unit, encode_type)     # obtain the encoded ID   [question] why not featured ID?
            
            # --- default distance
            if not in_region:
            
                if encode_id.size <= 1:
                    
                    pass
                
                elif encode_id.size > 1:     # check if this is a valid encoded unit/unit_typeron
                
                    encoded_img = [np.where(self.img_labels==encode_id[i])[0] for i in range(encode_id.size)]
                    tsne_img_avg = np.mean(reduced_feature[encoded_img], axis=1)
                    
                    point_wise_pairs = list(combinations(np.arange(tsne_img_avg.shape[0]), 2))
                    
                    tmp_distance = np.zeros(len(point_wise_pairs))
                    for idx, i in enumerate(point_wise_pairs):
                        tmp_distance[idx] = pdist([tsne_img_avg[i[0]], tsne_img_avg[i[1]]], 'euclidean').item()
                    
                    distance_dict[unit] = np.mean(tmp_distance)/max_distance
                    
            # --- add condition of feature region
            else:
                
                if not unit in feature_component_stats.keys():     # unit is not a feature unit
                    
                    pass
                
                else:
                
                    components = feature_component_stats[unit]
                    
                    distance_list = []
                    
                    for component_idx in components.keys():
                        
                        component = components[component_idx]
                        
                        if component['feature_selective_type'] is None:
                        
                            pass
                            
                        else:
                            
                            encode_id_ = np.intersect1d(encode_id, component['sig_id'])
                            
                            if encode_id_.size <= 1:
                                
                                pass

                            else:
                                
                                encoded_img = [np.where(self.img_labels==encode_id_[i])[0] for i in range(encode_id_.size)]
                                tsne_img_avg = np.mean(reduced_feature[encoded_img], axis=1)
                                
                                point_wise_pairs = list(combinations(np.arange(tsne_img_avg.shape[0]), 2))
                                
                                tmp_distance = np.zeros(len(point_wise_pairs))
                                for idx, i in enumerate(point_wise_pairs):
                                    tmp_distance[idx] = pdist([tsne_img_avg[i[0]], tsne_img_avg[i[1]]], 'euclidean').item()
        
                                distance_list.append(np.mean(tmp_distance)/max_distance)
                                
                    if len(distance_list) == 0:
                        
                        pass
                    
                    else:
                        
                        distance_dict[unit] = np.mean(distance_list)

        # --- normed
        distance_stats = {
                          'distance_dict': distance_dict,
                          'encode_type': encode_type,
                          'in_region': in_region
                          }

        return distance_stats
    

    # ------------------------------------------------------------------------------------------------------------------
    def feature_coding_plot(self, unit_type_and_reduced_feature, feature_unit_results, k=10):
        
        """
            test version
        """
        (DR_unit_type, DR_feature_map) = unit_type_and_reduced_feature
        
        # --- init
        feature_unit_sorting_dict = feature_unit_results['feature_unit_sorting_dict']
        feature_component_stats = feature_unit_results['feature_component_stats']
        
        print('[Codinfo] Executing feature unit plotting...')
        
        colorpool_jet = plt.get_cmap('jet', 50)
        self.colors = [colorpool_jet(i) for i in range(50)]
       
        plot_types_dict = {_: np.random.choice(feature_unit_sorting_dict[_], np.min([10, len(feature_unit_sorting_dict[_])])) for _ in feature_unit_sorting_dict.keys()}
        
        for plot_type in plot_types_dict.keys():     # foe each type
            
            plot_types_idces = plot_types_dict[plot_type]
            
            if len(plot_types_idces) != 0:
                
                self.plot_type_folder = os.path.join(self.single_unit_folder, plot_type)
                utils_.make_dir(self.plot_type_folder)
                
                self.plot_region_based_coding(plot_type, plot_types_idces, DR_feature_map, feature_component_stats)
            
    
    # ------------------------------------------------------------------------------------------------------------------
    def plot_region_based_coding(self, plot_type, plot_types_idces, DR_feature_map, feature_component_stats):
        
        #Parallel(n_jobs=int(os.cpu_count()/2))(delayed(self.plot_region_based_coding_single_unit)(unit_idx, DR_feature_map, feature, feature_component_stats[unit_idx], self.layer, self.img_labels, self.colors, self.num_classes, self.plot_type_folder) for unit_idx in tqdm(plot_types_idces, desc=f'{plot_type}'))
        
        for unit_idx in tqdm(plot_types_idces, desc=f'{plot_type}'):
            
            self.plot_region_based_coding_single_unit(unit_idx, DR_feature_map, self.feature[:, unit_idx].astype(float), feature_component_stats[unit_idx], self.layer, self.img_labels, self.colors, self.num_classes, self.plot_type_folder)

    # FIXME --- 
    @staticmethod
    def plot_region_based_coding_single_unit(unit_idx, DR_feature_map, FR, feature_component, layer=None, img_labels=None, colors=None, num_classes=None, plot_type_folder=None):
        
        """
            Task:    
                need to simplify and upgrade
        """
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        # --- init
        x = DR_feature_map[:, 0] - np.min(DR_feature_map[:, 0])
        y = DR_feature_map[:, 1] - np.min(DR_feature_map[:, 1])
        
        # --- plot
        fig = plt.figure(figsize=(18, 9))
        #plt.annotate(f'FC_6 Unit: {unit}', (0.5, 0.98), xycoords='axes fraction', ha='center', fontsize=14, bbox=dict(boxstyle="square", ec="none", fc="white"))
        
        # === 1
        ax_1 = plt.gcf().add_axes([0.05, 0.1, 0.2, 0.8])
        Selectivity_Analysis_Feature.plot_distance_boxplot(ax_1, FR, img_labels, colors, num_classes)
        
        # === 2 --- construction...
        ax_2 = plt.gcf().add_axes([0.3, 0.1, 0.4, 0.8])
        Selectivity_Analysis_Feature.plot_scatter_with_contour(ax_2, FR, x, y, unit_idx, feature_component, img_labels, colors)
        
        # === 3
        ax_3_upper = plt.gcf().add_axes([0.75, 0.55, 0.175, 0.35])
        ax_3_lower = plt.gcf().add_axes([0.75, 0.1, 0.175, 0.35])
        pdfxy = Selectivity_Analysis_Feature.kde_2d_v3(x, y, weights=FR)
        pdfPerm = Selectivity_Analysis_Feature.kde_2d_perm(x, y, weights=FR)
        vmin, vmax = Selectivity_Analysis_Feature.plot_kde(ax_3_upper, ax_3_lower, pdfxy, pdfPerm)     # [question] maerge the value range from all units?
        
        cmap = plt.get_cmap("viridis")
        norm_ = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_)
        sm.set_array([])  # Just a dummy array
        cbar_ax = fig.add_axes([0.95, 0.1, 0.0125, 0.8])
        fig.colorbar(sm, cax=cbar_ax)
        #cbar = plt.colorbar(cax1, ax=axes, orientation='vertical', fraction=0.02, pad=0.06)
        
        fig.suptitle(f'{layer} Unit: {unit_idx}', y=0.95, fontsize=16)
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            plt.savefig(os.path.join(plot_type_folder, f'{plot_type_folder.split("/")[-1]}_{layer}_{unit_idx}.png'), bbox_inches='tight')
            plt.savefig(os.path.join(plot_type_folder, f'{plot_type_folder.split("/")[-1]}_{layer}_{unit_idx}.eps'), bbox_inches='tight')
            plt.close()
    
    
    #FIXME - all function should use the same box_plot/bar_plot/scatter_plot in this series of codes
    @staticmethod
    def plot_distance_boxplot(ax, FR, img_labels=None, colors=None, num_classes=None):
        """
            this function use the correct label to plot the response
        """
        # ----- boxplot
        #bp = ax.boxplot([FR[self.img_labels == i] for i in range(1,51)], vert=False, patch_artist=True, sym='+')
        
        #for patch, color in zip(bp['boxes'], self.colors):
        #    patch.set_facecolor(color)
        #    patch.set_alpha(0.8)
        #    patch.set(edgecolor='none')
        
        #mean_list = np.mean(np.array([FR[self.img_labels == i] for i in range(1,51)]), axis=1)

        #encoded_idx = np.where(mean_list > threshold)[0]
        #non_encoded_idx = np.setdiff1d(np.arange(50), encoded_idx)

        #ax.scatter(mean_list[encoded_idx], encoded_idx+1, color='red', linewidth=0, alpha=0.5, label=r'$\overline{x}>V_{th}$', zorder=2)
        #ax.scatter(mean_list[non_encoded_idx], non_encoded_idx+1, color='blue', linewidth=0, alpha=0.5, label=r'$\overline{x}<V_{th}$', zorder=2)

        #for idx, _ in enumerate(mean_list):
        #    if idx in encoded_idx:
        #        ax.hlines(idx+1, 0, _, colors='orange', linestyles='--', linewidth=2.0, alpha=0.5)
        #    else:
        #        ax.hlines(idx+1, 0, _, colors='blue', linestyles='--', linewidth=2.0, alpha=0.5)

        #ax.set_yticks(range(1, 51))
        #ax.set_ylim([0,51])
        #ax.set_yticklabels([str(i) for i in range(1, 51)])
        #ax.set_xlabel('Response')

        # ----- scatter
        threshold = np.mean(FR) + 2*np.std(FR)
        ax.vlines(threshold, 0, 52, colors='red', linestyles='-', linewidth=1.0, alpha=0.75)
        
        ref = np.mean(FR) + 2*np.std(np.mean(FR.reshape(50, 10), axis=1))
        ax.vlines(ref, 0, 52, colors='teal', linestyles='-', linewidth=1.0, alpha=0.75)
        
        x = np.array([[_]*10 for _ in range(1,51)])
        
        FR_new = np.array([FR[img_labels == i] for i in range(1,51)])
        test_feature_mean = np.mean(FR_new, axis=1)
        y = FR_new.reshape(-1)
        
        c = np.repeat(np.array(colors), 10, axis=0)
        
        ax.scatter(y, x, color=c, s=10)
        ax.scatter(test_feature_mean, np.arange(1,51), color=colors, marker='d')
        for _ in range(num_classes):
            ax.hlines(_+1, np.min(y), test_feature_mean[_], linestyle='--')
        
        ax.set_yticks(range(1, 51))
        ax.set_ylim([0,51])
        ax.set_yticklabels([str(i) for i in range(1, 51)])
        ax.set_xlabel('Response')
        #ax.legend(framealpha=0.75)

        #ax.set_ylim([y_lim_min, y_lim_max])
        
    # FIXME - consider to remove 'best'?
    @staticmethod
    def plot_scatter_with_contour(ax, FR, x, y, unit_idx, feature_components, img_labels=None, colors=None):
        
        """
            Drawing and annotating each connected region
            
        """
        
        def _location_bank():
            
            location_bank = [
                'best',
                'upper left',
                'lower right',
                'upper right',
                'lower left',
                'center left',
                'center right',
                'lower center',
                'right center'
                ]
            
            return location_bank
        
        def _connect_diagonals(data):

            connected_data = data.copy()
            rows, cols = data.shape
            for i in range(rows - 1):
                for j in range(cols - 1):
                    # Check for diagonal connectivity and fill in the gaps
                    if data[i, j] and not data[i, j + 1] and not data[i + 1, j] and data[i + 1, j + 1]:
                        connected_data[i, j + 1] = connected_data[i + 1, j] = True
                    elif not data[i, j] and data[i, j + 1] and data[i + 1, j] and not data[i + 1, j + 1]:
                        connected_data[i, j] = connected_data[i + 1, j + 1] = True
                        
            return connected_data
        
        def _contour_color_and_label(feature_unit_type):
            
            if feature_unit_type is None:
                return 'gray', 'NE'
            elif 'strong' in feature_unit_type:
                return 'red', 'strong'
            elif 'weak' in feature_unit_type:
                return 'teal', 'weak'
            elif 'merged' in feature_unit_type:
                return 'cyan', 'merged'
            else:
                raise ValueError(f'[Codinfo] [{feature_unit_type}] invalid input')
        
        if np.sum(FR) == 0 or np.sum(FR!=0) ==1:
            for gg in range(1,51):  # this can be changed to different types of id
                current_scatter = ax.scatter(x[img_labels == gg], y[img_labels == gg], s=5, color=colors[gg-1], alpha=0.5)
                
        else:
            
            size_weight = FR / max(FR)     # [notice] can not divide by 0 if all values are 0
            sizes = np.ones(500) * 15 * (1 + 20 * size_weight)
            
            # ---
            sig_id_list = []
            for component_idx in feature_components.keys():
                
                component = feature_components[component_idx]
                sig_id_list = np.append(sig_id_list, component['sig_id'])
            
            # --- non_featured_points
            not_featured_id_idx = np.setdiff1d(np.arange(50)+1, sig_id_list)
            
            handles_not_featured = []
            for gg in not_featured_id_idx:  # this can be changed to different types of id
                current_scatter = ax.scatter(x[img_labels == gg], y[img_labels == gg], s=sizes[img_labels == gg], color=colors[gg-1], alpha=0.5)
                handles_not_featured.append(current_scatter)
            
            # --- featured_points
            #with warnings.catch_warnings():
            #    warnings.simplefilter("ignore")
            for idx, component_idx in enumerate(feature_components.keys(), 0):
                
                component = feature_components[component_idx]
                sig_id = component['sig_id']
                
                # -----
                handles_featured = []
                labels = []
                
                for gg in sig_id:
                    current_scatter = ax.scatter(x[img_labels == gg], y[img_labels == gg], s=sizes[img_labels == gg], color=colors[gg-1], alpha=0.7)
                    handles_featured.append(current_scatter)
                    labels.append(f'{gg}')
                
                ax.add_artist(ax.legend(handles=handles_featured, labels=labels, loc=_location_bank()[idx], framealpha=0.5))
                # -----
                
                # -----
                tsne_map = empty_density_map(y, x)
                for y_, x_ in component['region_location'].T:
                    tsne_map[y_, x_]=1
                    
                contour_color, contour_label = _contour_color_and_label(component['feature_selective_type'])
                
                contour_lines = ax.contour(scipy.ndimage.gaussian_filter(_connect_diagonals(tsne_map).astype(float), sigma=1), [0.5], colors=contour_color)
                ax.clabel(contour_lines, inline=True, fontsize=8, fmt={0.5: contour_label})
                for line in contour_lines.collections:
                    line.set_linestyle('-') 
                    line.set_linewidth(2)    

            ax.set_xlabel('TSNE Dimension 1')
            ax.set_ylabel('TSNE Dimension 2')
        
    @staticmethod
    def plot_kde(ax_up, ax_down, pdfxy, pdfPerm):
        
        vmin = np.min([np.min(pdfxy), np.min(pdfPerm)])
        vmax = np.max([np.max(pdfxy), np.max(pdfPerm)])
        
        ax_up.imshow(pdfxy, origin='lower', vmin=vmin, vmax=vmax)
        ax_up.set_xticks([])
        ax_up.set_yticks([])
        ax_up.set_xlabel('Feature Dimension 1')
        ax_up.set_ylabel('Feature Dimension 2')
        ax_up.set_title('observed density map')
        
        ax_down.imshow(pdfPerm, origin='lower', vmin=vmin, vmax=vmax)
        ax_down.set_xticks([])
        ax_down.set_yticks([])
        ax_down.set_xlabel('Feature Dimension 1')
        ax_down.set_ylabel('Feature Dimension 2')
        ax_down.set_title('mean permuted density map')
        
        return vmin, vmax

    @staticmethod
    def kde_2d_perm(x, y, band_width=None, weights=None, num_perm=1000):
 
        pdfPerm = []
    
        for ii in range(num_perm):
            permed_weights = weights[np.random.permutation(len(weights))]
            pdf_xy = Selectivity_Analysis_Feature.kde_2d_v3(x, y, band_width=band_width, weights=permed_weights)
            pdfPerm.append(pdf_xy)

        pdfPerm = np.mean(np.array(pdfPerm), axis=0)

        return pdfPerm
          
    @staticmethod
    def kde_2d_v3(x, y, band_width=None, weights=None, plot_scale=100):
        
        pdfx = Selectivity_Analysis_Feature.ksdensity(x, band_width, weights)
        pdfy = Selectivity_Analysis_Feature.ksdensity(y, band_width, weights)
        
        pdfx = pdfx(np.linspace(min(x), max(x), plot_scale))
        pdfy = pdfy(np.linspace(min(y), max(y), plot_scale))

        pdfx, pdfy = np.meshgrid(pdfx, pdfy)
        pdfxy = pdfx * pdfy
        
        return pdfxy
        
    @staticmethod
    def ksdensity(data, band_width=None, weights=None):
        if weights is not None and np.sum(weights) != 0 and len(np.where(weights!=0)[0])!=1:
            
            def _delta_pdf(input:np.ndarray) -> np.ndarray:
                return np.unique(input)
            
            try:
                ksdensity = gaussian_kde(data, bw_method=band_width, weights=weights)
            except:
                
                if len(np.unique(data)) == 1:
                    #raise RuntimeWarning('[Codinfo] detected the input is not able to calculate kde, use delta instead')
                    ksdensity = _delta_pdf
                else:
                    if np.sum(data*weights) == 0:    
                        #raise RuntimeWarning('[Codinfo] deteriorate to without weights(FR)')
                        ksdensity = gaussian_kde(data, bw_method=band_width)
                    else:
                        raise RuntimeError('[Codinfo] unknown error')
                    
        else:
            ksdensity = gaussian_kde(data, bw_method=band_width, weights=None)
        return ksdensity
    
    # ------------------------------------------------------------------------------------------------------------------
    def population_feature_size(self, unit_type_and_reduced_feature, unit_types_dict, feature_unit_results, target_cluster_type='max'):
        """
            this function calculates (1) the average cluster size and (2) overlapped size for one type of units --- FIXME
        """
        (DR_coordinate, DR_feature_map) = unit_type_and_reduced_feature
        
        x = DR_feature_map[:, 0] - np.min(DR_feature_map[:, 0])
        y = DR_feature_map[:, 1] - np.min(DR_feature_map[:, 1])
        
        empty_mask = empty_density_map(y, x)
    
        # --- feature_unit analysis
        feature_cluster_stats = {}
        
        list_pixels = feature_unit_results['feature_component_stats']
        
        tqdm_bar = tqdm(total=len(unit_types_dict.keys()), desc=f'Region Size | target_cluster: {target_cluster_type}')
        
        for unit_type in unit_types_dict.keys():     # for each type of units --- this will only consider the 'feature' unit
        
            unit_idx = unit_types_dict[unit_type]
            
            if len(unit_idx) != 0:
    
                unit_cluster_size_dict = {}
                overlapped_pixels = empty_mask.copy()
                
                for unit in unit_idx:     # for each unit
                
                    self.single_unit_cluster_size_calculation(target_cluster_type, unit, unit_cluster_size_dict, list_pixels[unit], empty_mask, overlapped_pixels, unit_type)
                    
                feature_cluster_stats[unit_type] = (unit_cluster_size_dict, overlapped_pixels)
                
            tqdm_bar.update(1)
            
        # --- non_feature_unit analysis
        
        region_stats = {}
        
        preliminary_p_masks = feature_unit_results['preliminary_p_masks']
        
        s_si = self.sort_dict['advanced_type']['s_si']
        
        s_si_regions = {_:preliminary_p_masks[_] for _ in s_si if _ in preliminary_p_masks.keys()}
        
        #FIXME --- remove the feature_si
        
        s_si_regions_sizes_dict = {}
        empty_mask = empty_density_map(y, x)
        
        for _ in s_si_regions.keys():
            
            labeled_regions, num_components = scipy.ndimage.label(s_si_regions[_], structure=scipy.ndimage.generate_binary_structure(2,2)) 
            
            region_sizes = []
            
            for __ in np.arange(1, num_components+1):
                
                component_size = np.sum(labeled_regions==__)
                region_sizes.append(component_size/labeled_regions.size*100)
                
                for (y_, x_) in zip(*np.where(labeled_regions==__)):     
                    empty_mask[y_,  x_] += 1
            
            s_si_regions_sizes_dict[_] = np.max(region_sizes)
        
        region_stats['s_si'] = (s_si_regions_sizes_dict, empty_mask)
        
        # ---
        # s_mi = fmi + nfmi = fmi + f_strong_mi + f_weak_mi + f_merged_mi + others
        # [example: Baseline neuron_1] 229 = 108 + 2 + 14 + 105
        
        s_mi = self.sort_dict['advanced_type']['s_mi']
        
        tmp_2 = [__ for group in [unit_types_dict[_] for _ in unit_types_dict.keys() if 's_mi' in _] for __ in group]
        
        s_mi_others_regions = {_: preliminary_p_masks[_] for _ in s_mi if _ in preliminary_p_masks.keys() and _ not in tmp_2}
        
        s_mi_others_regions_sizes = {}
        empty_mask = empty_density_map(y, x)
        
        for _ in s_mi_others_regions.keys():
            
            labeled_regions, num_components = scipy.ndimage.label(s_mi_others_regions[_], structure=scipy.ndimage.generate_binary_structure(2,2)) 
            
            region_sizes = []
            
            for __ in np.arange(1, num_components+1):
                
                component_size = np.sum(labeled_regions==__)
                region_sizes.append(component_size/labeled_regions.size*100)
                
                for (y_, x_) in zip(*np.where(labeled_regions==__)):     
                    empty_mask[y_,  x_] += 1
            
            s_mi_others_regions_sizes[_] = np.max(region_sizes)
        
        region_stats['s_mi_others'] = (s_mi_others_regions_sizes, empty_mask)
        
        # ---
        # others: those units are not mi and not si, which also not feature units but have preliminary regions
        # 1st, find those units
        
        others_pool = list(preliminary_p_masks.keys())
        feature_units_idx = list(feature_unit_results['feature_component_stats'].keys())
        others_pool = [_ for _ in others_pool if _ not in feature_units_idx+list(s_si)+list(s_mi)]
        others_regions = {_: preliminary_p_masks[_] for _ in others_pool}
        
        others_regions_sizes = {}
        empty_mask = empty_density_map(y, x)
        
        for _ in others_regions.keys():
            
            labeled_regions, num_components = scipy.ndimage.label(others_regions[_], structure=scipy.ndimage.generate_binary_structure(2,2)) 
            
            region_sizes = []
            
            for __ in np.arange(1, num_components+1):
                
                component_size = np.sum(labeled_regions==__)
                region_sizes.append(component_size/labeled_regions.size*100)
                
                for (y_, x_) in zip(*np.where(labeled_regions==__)):     
                    empty_mask[y_,  x_] += 1
            
            others_regions_sizes[_] = np.max(region_sizes)
        
        region_stats['others_'] = (others_regions_sizes, empty_mask)
        
        
        print('6')
        
    
        return feature_cluster_stats, region_stats
             
    
    def single_unit_cluster_size_calculation(self, target_cluster_type, unit, unit_cluster_size_dict, unit_clusters, empty_mask, overlapped_pixels, unit_type):
        #FIXME
        """
            add a judgement to revome weak/merged/NE cluster for strong unit although the size is the biggest
        """
        
        if len(unit_clusters) != 0:      # prevent unit  without useful cluster
        
            if target_cluster_type == 'max':     # only count the biggest cluster
                
                region_size_pool_dict = {}
            
                for component_idx in unit_clusters.keys():
                    component = unit_clusters[component_idx]
                    
                    # --- only consider the major cluster for given unit
                    if 'non' in unit_type:
                        region_size = component['region_size']
                        region_size_pool_dict[component_idx] = region_size
                    
                    elif 'strong' in unit_type:
                        if component['feature_selective_type'] is not None and 'strong' in component['feature_selective_type']:
                            region_size = component['region_size']
                            region_size_pool_dict[component_idx] = region_size
                        
                    elif 'weak' in unit_type:
                        if component['feature_selective_type'] is not None and 'weak' in component['feature_selective_type']:
                            region_size = component['region_size']
                            region_size_pool_dict[component_idx] = region_size
                    
                    elif 'merged' in unit_type:
                        if component['feature_selective_type'] is not None and 'merged' in component['feature_selective_type']:
                            region_size = component['region_size']
                            region_size_pool_dict[component_idx] = region_size
                    else:
                        raise ValueError(f'[Codinfo] {unit_type} is invalid')
            
                sorted_dict = sorted(region_size_pool_dict.items(), key=lambda item: item[1], reverse=True)
                sorted_dict = dict(sorted_dict)
            
                (component_idx_of_max_size, max_region_size) = list(sorted_dict.items())[0]
                
                unit_cluster_size_dict[unit] = max_region_size/empty_mask.size*100     # [unit_idx][biggest_cluster_idx][cluster_size]
                
                for y_, x_ in unit_clusters[component_idx_of_max_size]['region_location'].T:     
                    overlapped_pixels[y_,  x_] += 1
            
            elif target_cluster_type == 'all':     # count all qualified clusters, include all clusters for this unit
                
                tmp_region_size = 0.
            
                for component_idx in unit_clusters.keys():
                    component = unit_clusters[component_idx]
                    
                    tmp_region_size += component['region_size']/empty_mask.size*100
                    
                    for y_, x_ in unit_clusters[component_idx]['region_location'].T:     
                        overlapped_pixels[y_,  x_] += 1
                    
                unit_cluster_size_dict[unit] = tmp_region_size
        else:
            
            unit_cluster_size_dict[unit] = 0.
    
    
    def feature_region_selection(self, 
                                 unit_type_and_reduced_feature,
                                 p_results, 
                                 cluster_size_scaling_factor=0.025, alpha=0.01):
        """
            input:
                
                p_results: results of self.generate_p_values()
                
            parameters:
                
                self.maskFactor: scaling factor
                self.sort_dict: unit type
                self.neuron: number of units
                self.encode_dict: encode ids
                
            call functions:
                
                calculate_convolved_density_map()
                feature_region_selection_single_unit()
                
            
        """
        # --- init
        DR_coordinate = unit_type_and_reduced_feature[0]
        
        p_values = p_results['p']
        gaussian_kernel = p_results['kernel']
        reduced_feature = p_results[f'{self.DR_type.lower()}']

        # ---
        file_path = os.path.join(self.layer_folder_DR_coordinate, f'tsne_{DR_coordinate}_unit_stats.pkl')
        
        if os.path.exists(file_path):
            
            results = utils_.load(file_path)
            
        else:
        
            density_map, convolved_density_map = calculate_convolved_density_map(reduced_feature, weights=None, kernel=gaussian_kernel)
            
            # --- remove corners and edges with too sparse dots
            mask = convolved_density_map >= (self.maskFactor*np.mean(convolved_density_map))
 
            # --- init
            reversed_sort_dict = {value: [key for key, vals in self.sort_dict['advanced_type'].items() if value in vals][0] for key_list in self.sort_dict['advanced_type'].values() for value in key_list}
            reversed_sort_dict = {_: reversed_sort_dict[_] for _ in sorted(reversed_sort_dict.keys())}
            
            # FIXME upgrade this section --- currently the 'encode_id' is only considering 'selective' units

            cluster_size_threshold = mask.size * cluster_size_scaling_factor
            
            # --- Sequential, for test
            pl = []

            for unit in tqdm(units:=np.arange(self.neuron), desc='Sequential region selection'):
                
                results = feature_region_selection_single_unit(

                                                          reduced_feature,     # (500, 2)
                                                          
                                                          unit, 
                                                          reversed_sort_dict[unit],
                                                          
                                                          p_values[unit], 
                                                          self.encode_dict[unit], 
                                                               
                                                          mask, 
                                                          cluster_size_threshold, 
                                                          
                                                          self.img_labels,
                                                          )
                                            
                pl.append(results)
            
            # FIXME -- need to upgraded
            feature_selective_stats = {_: pl[_]['feature_selective_type'] for _ in units if pl[_] is not None and len(pl[_]['feature_selective_type']) != 0}
            
            #tmp_pool = [___ for __ in [feature_selective_stats[_] for _ in feature_selective_stats.keys()] for ___ in __]
            #tmp_pool_new = [_.split('encode_')[-1] for _ in tmp_pool]
            
            feature_component_stats = {_:pl[_]['feature_component_dict'] for _ in units if pl[_] is not None and len(pl[_]['feature_component_dict']) != 0}

            # feature_unit_sorting
            feature_units = np.array(list(feature_component_stats.keys()))
            feature_selective_units = np.array(list(feature_selective_stats.keys()))
            feature_non_selective_units = np.setdiff1d(feature_units, feature_selective_units)
            
            # ---
            feature_strong_s_mi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'strong_encode_s_mi' in feature_selective_stats[_]])
            feature_weak_s_mi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'weak_encode_s_mi' in feature_selective_stats[_] and 'strong_encode_s_mi' not in feature_selective_stats[_]])
            feature_merged_s_mi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'merged_encode_s_mi' in feature_selective_stats[_] and 'strong_encode_s_mi' not in feature_selective_stats[_] and 'weak_encode_s_mi' not in feature_selective_stats[_]])
            
            feature_strong_s_wmi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'strong_encode_s_wmi' in feature_selective_stats[_]])
            feature_weak_s_wmi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'weak_encode_s_wmi' in feature_selective_stats[_] and 'strong_encode_s_wmi' not in feature_selective_stats[_]])
            feature_merged_s_wmi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'merged_encode_s_wmi' in feature_selective_stats[_] and 'strong_encode_s_wmi' not in feature_selective_stats[_] and 'weak_encode_s_wmi' not in feature_selective_stats[_]])
            
            # ---
            feature_strong_s_si_idx = np.array([_ for _ in feature_selective_stats.keys() if 'strong_encode_s_si' in feature_selective_stats[_]])
            feature_weak_s_si_idx = np.array([_ for _ in feature_selective_stats.keys() if 'weak_encode_s_si' in feature_selective_stats[_] and 'strong_encode_s_si' not in feature_selective_stats[_]])
            feature_merged_s_si_idx = np.array([_ for _ in feature_selective_stats.keys() if 'merged_encode_s_si' in feature_selective_stats[_] and 'strong_encode_s_si' not in feature_selective_stats[_] and 'weak_encode_s_si' not in feature_selective_stats[_]])
            
            feature_strong_s_wsi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'strong_encode_s_wsi' in feature_selective_stats[_]])
            feature_weak_s_wsi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'weak_encode_s_wsi' in feature_selective_stats[_] and 'strong_encode_s_wsi' not in feature_selective_stats[_]])
            feature_merged_s_wsi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'merged_encode_s_wsi' in feature_selective_stats[_] and 'strong_encode_s_wsi' not in feature_selective_stats[_] and 'weak_encode_s_wsi' not in feature_selective_stats[_]])
            
            feature_unit_sorting_dict = {
                'feature_non_selective_units': feature_non_selective_units,
                
                'feature_strong_s_mi_idx': feature_strong_s_mi_idx,
                'feature_weak_s_mi_idx': feature_weak_s_mi_idx,
                'feature_merged_s_mi_idx': feature_merged_s_mi_idx,
                
                'feature_strong_s_wmi_idx': feature_strong_s_wmi_idx,
                'feature_weak_s_wmi_idx': feature_weak_s_wmi_idx,
                'feature_merged_s_wmi_idx': feature_merged_s_wmi_idx,
                
                'feature_strong_s_si_idx': feature_strong_s_si_idx,
                'feature_weak_s_si_idx': feature_weak_s_si_idx,
                'feature_merged_s_si_idx': feature_merged_s_si_idx,
                
                'feature_strong_s_wsi_idx': feature_strong_s_wsi_idx,
                'feature_weak_s_wsi_idx': feature_weak_s_wsi_idx,
                'feature_merged_s_wsi_idx': feature_merged_s_wsi_idx
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
            
            utils_.dump(results, file_path)
        
        return  results


    # ------------------------------------------------------------------------------------------------------------------
    def generate_p_values(self, unit_type_and_reduced_feature):
        """
            this function is the parallel executor of calculate_density_perm() to obtain p values for all units
            
        """
        
        # --- init
        DR_coordinate = unit_type_and_reduced_feature[0]
        
        file_path = os.path.join(self.layer_folder_DR_coordinate, f'{self.layer}_{DR_coordinate}_sq{self.scaling_factor}.pkl')
        
        if os.path.exists(file_path):
            
            results = utils_.load(file_path)
            
        else:
            reduced_feature = unit_type_and_reduced_feature[1]
            
            kernel_size, kernel_sigma = self.get_kernel_size(reduced_feature, self.scaling_factor)
            gaussian_kernel = self.gausskernel(kernel_size, kernel_sigma)
            
            # --- calculate p values
            p = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(calculate_density_perm_p)(reduced_feature, self.feature[:, i], gaussian_kernel) for i in tqdm(np.arange(self.neuron), desc=f'{DR_coordinate}'))
            
            # --- wrap results and save
            results = {
                       'layer': self.layer, 
                       'scaling_factor': self.scaling_factor, 
                       
                       'DR_coordinate': DR_coordinate,
                       f'{self.DR_type.lower()}': reduced_feature,    
                       'p': p, 
                       
                       'kernel_size': kernel_size, 
                       'kernel_sigma': kernel_sigma, 
                       'kernel': gaussian_kernel,
                       }
            
            utils_.dump(results, file_path)
            
        return results
        
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_kernel_size(reduced_feature:np.array, sq:float=0.035):
        """
            this function calculate the kernel_size and sigma for the following calculation on p values
            
            inputs:
                
                reduced_feature: (num_samples, num_dims=2);
          
            parameters:
                
                sq: an empirical scale factor to decide the sigma of the gaussian filter. default to be 0.035, (close to sd = 4, used in previous expriments)
                
                2*3*kernel_sigma is a commonly used factor to represent the 2-tailed gaussian kernel with 3 std
            
            In fact, the function here is not the same one described in the paper. 
                (1) The method here calculate kernel_sigma and kernel_size based on 'the number of 
                connected components', and use a scaling factor: sq=0.035 to 
                decide the value of kernel_sigma(sigma, which decides the speed of decrease)
                -> kernel_sigma(sigma) = sq*num_component, KS = 2*radius(3*sigma)+1
                (2) The method described in paper is based on 'the size of feature map',
                and use another scaling factor: ff1=0.2(in self.feature_coding_plot()) to
                decide the kernel size -> KS = ff1*[ylim,xlim]
            
        """
        # --- normalization
        density_map = calculate_2d_density_map(reduced_feature)
        labeled, num_component = scipy.ndimage.label(density_map, structure=scipy.ndimage.generate_binary_structure(2,2))  
        
        kernel_sigma = num_component * sq
        
        # decide the kernel size
        kernel_size_y = int(np.floor(2*3*kernel_sigma + 1))
        kernel_size_x = int(np.floor(kernel_size_y*density_map.shape[1]/density_map.shape[0]))
        
        kernel_size = [kernel_size_y, kernel_size_x]
    
        return kernel_size, kernel_sigma
    
    @staticmethod
    def gausskernel(R, S):
        """
            Creates a discretized N-dimensional Gaussian kernel.
            R: kernel size (pixels in one side)
            S: standard deviation
            [note] in current use, the R is a 2-D vector and S is a scalar
        """
    
        # Check Inputs
        R = np.asarray(R)
        S = np.asarray(S)
        D = R.size
        D2 = S.size
    
        if ((D > 1 and R.ndim != 1) or (D2 > 1 and S.ndim != 1)):
            raise ValueError('Matrix arguments are not supported.')
    
        if ((D > 1 and D2 > 1) and (D != D2)):
            raise ValueError('R and S must have same number of elements (unless one is scalar).')
    
        # Force bins/sigmas 
        if (D2 > D):  
            D = D2  
            R = R * np.ones(D) 
    
        # To be same length
        if (D > D2): 
            S = S * np.ones(D)  
    
        # And force row vectors
        R = R.flatten()
        S = S.flatten()
    
        # Make the Kernel
        kernel = None
        
        RR = 2*R+1
        for k in range(D):
           
            grid = np.arange(-R[k], R[k] + 1)
            gauss = np.exp(-grid**2 / (2 * S[k]**2))
            gauss = gauss / np.sum(gauss)  # normalization
    
            # [note] this function is only valid for current 2D kernel, higher dimension needs more experiments
            if (k == 0):
                kernel = gauss
            else:
                Dpast = [1] * k
                expand = np.tile(gauss.reshape(1,-1), [*RR[0:k], 1])
                kernel = np.tile(kernel.reshape(-1,1), [*Dpast, RR[k]]) * expand
    
        return kernel


# ----------------------------------------------------------------------------------------------------------------------
def feature_region_selection_single_unit(
                                         reduced_feature, 
                                         
                                         unit, 
                                         unit_type,
                                         
                                         p, 
                                         encode_id, 
                                              
                                         mask, 
                                         cluster_size_threshold, 
                                         
                                         img_labels,
                                         
                                         alpha=0.01, 
                                         num_ids_threshold=2, 
                                         num_imgs_threshold=5
                                         ):
    """
        1 dependent conditions:
            
            0. existing connected area of the p_value probability map
    
        2 conditions to determine a feature unit:
            
            1. the size of region (cluster)
            2. the imgs (default 5) and ids (default: 2) contained in the region
            
        1 more condition to determine a feature_selective unit:
            
            3. the intersection with encoded_ids (default: >=2)
            
        return:
            
            a dict contains preliminary regions information and qualified regions information. key ['preliminary_p_mask']
        contains the results of fundamental conditions, used to conduct analysis of size/distance between feature and non_feature
        units (no size threshold for non_feature unit according to source paper)
        
    """
    
    # --- init
    x = reduced_feature[:, 0] - np.min(reduced_feature[:, 0])
    y = reduced_feature[:, 1] - np.min(reduced_feature[:, 1])
    
    # --- condition 0, perm p
    p_regions = (p<alpha)*mask
    p_regions_init = p_regions.copy()

    labeled_p_regions, num_components = scipy.ndimage.label(p_regions, structure=scipy.ndimage.generate_binary_structure(2,2)) 
    
    if num_components == 0:     # if the unit does not have connected component
        
        return None
    
    else:     # if the unit has connected component
       
        # --- init
        feature_component_dict = {}
        feature_unit=False

        for i in range(1, num_components+1):     # for each component
            
            component = (labeled_p_regions == i)
            
            # --- condition 1, size
            if component.sum() < cluster_size_threshold:     
                p_regions[component] = 0
                labeled_p_regions[component] = 0
            
            else:
                
                tmp_sig_img = np.array([_ for _ in range(reduced_feature.shape[0]) if labeled_p_regions[round(y[_]), round(x[_])] == i])
                
                #FIXME
                if len(tmp_sig_img) == 0:
                    return None
                
                tmp_sig_id = np.unique(img_labels[tmp_sig_img])
                
                # --- condition 2, num 
                #FIXME, can make this conditional statement more stringent?
                if len(tmp_sig_id) < num_ids_threshold or len(tmp_sig_img) < num_imgs_threshold: 
                    
                    p_regions[component] = 0
                    labeled_p_regions[component] = 0
                    
                else:
                    
                    feature_unit = True    # if one region is qualified, then the unit is a feature unit 
                    
                    feature_component_dict[i] = {
                        'sig_img': tmp_sig_img,
                        'sig_id': tmp_sig_id,
                        'region_size': np.sum(labeled_p_regions==i),
                        'region_location': np.array(np.where(labeled_p_regions==i)),
                        'feature_selective_type': None
                        }
                    
                    # --- condition 3, intersection
                    if not np.intersect1d(tmp_sig_id, np.append(encode_id['encode'], encode_id['weak_encode'])).size > 1:
                        
                        p_regions[component] = 0
                        labeled_p_regions[component] = 0
                        
                    else:
                        
                        if np.intersect1d(tmp_sig_id, encode_id['encode']).size > 1:     # strong_encode si or mi
                        
                            feature_component_dict[i].update({'feature_selective_type': f'strong_encode_{unit_type}'})
                            
                        elif np.intersect1d(tmp_sig_id, encode_id['weak_encode']).size > 1:     # weak_encode si, wsi, mi, wmi
                        
                            feature_component_dict[i].update({'feature_selective_type': f'weak_encode_{unit_type}'})
                        
                        else:     # merged_encode
                            
                            feature_component_dict[i].update({'feature_selective_type': f'merged_encode_{unit_type}'})
                        
        feature_unit_type_pool = list(set([feature_component_dict[_]['feature_selective_type'] for _ in feature_component_dict.keys() if feature_component_dict[_]['feature_selective_type'] is not None]))
        
        results = {
            'preliminary_p_mask': p_regions_init,
            'qualified_p_mask': p_regions,
            
            'feature_component_dict': feature_component_dict,
            'feature_unit': feature_unit,
            
            'feature_selective_type': feature_unit_type_pool,
            'encode_id': encode_id
            }

        return results

# ----------------------------------------------------------------------------------------------------------------------
def calculate_density_perm_p(reduced_feature, weights=None, kernel=None, num_perm=1000):
    """
        this function is a warp of calculate_density_perm(), extract the p_values only
        
        in current process script, this is used for parallel process
    """
    
    p, *_ = calculate_density_perm(reduced_feature, weights, kernel, num_perm)
    
    return p
    

def calculate_density_perm(reduced_feature, weights=None, kernel=None, num_perm=1000):
    """
        this function is an advanced wrap of calculate_convolved_density_map(), added permutation test to generate a p_value
        
        input:
            reduced_feature: reduced_feature 2D feature with shape (num_samples, 2)
            
            weights: original weights to indicate the size (radius) of each dot in the reduced_feature coordinate
            
            num_perm:
                
            kernel:
        
        return:
            p: p values of permutation test with the same shape of density_map
            
            perm_density_maps: density_maps of each permutation
            
            perm_convolved_density_maps: convolved_density_maps of each permutation
    """
    if weights is not None and np.sum(weights) != 0 and np.sum(weights!=0) > 1:     # expel unqualified units
        
        # --- calculate the original density map
        density_map, convolved_density_map = calculate_convolved_density_map(reduced_feature, weights, kernel)
        
        # --- init
        permutation_stats = np.zeros(density_map.shape)
        perm_density_maps = np.ones((num_perm, *density_map.shape))
        perm_convolved_density_maps = np.ones((num_perm, *density_map.shape))
    
        # --- permutation test
        for _ in range(num_perm):

            perm_density_map, perm_convolved_density_map = calculate_convolved_density_map(reduced_feature, weights[np.random.permutation(len(weights))], kernel)
            
            permutation_stats += perm_convolved_density_map > convolved_density_map
            
            perm_density_maps[_] = perm_density_map
            perm_convolved_density_maps[_] = perm_convolved_density_map
    
        p = permutation_stats / num_perm
    
    else:
        
        density_map, convolved_density_map = calculate_convolved_density_map(reduced_feature, None, kernel)
        
        perm_density_maps = np.ones((num_perm, *density_map.shape))
        perm_convolved_density_maps = np.ones((num_perm, *density_map.shape))
        
        # init with 1
        p = np.ones((int(np.ceil(np.max(reduced_feature[:,1]-np.min(reduced_feature[:,1]))+1)), int(np.ceil(np.max(reduced_feature[:,0]-np.min(reduced_feature[:,0]))+1))))      
            
    return p, perm_density_maps, perm_convolved_density_maps


def calculate_convolved_density_map(reduced_feature, weights=None, kernel=None):
    """
        this function is a simple wrap of calculate_2d_density_map(), added the smoothed (convolved) density map
        
        input:
            
            reduced_feature: reduced_feature 2D feature with shape (num_samples, 2)
            
            weights: original weights to indicate the size (radius) of each dot in the reduced_feature coordinate
            
            kernel: 2D kernel used for smooth (convolve) process
            
        return:
            
            density_map: point density map
            
            convolved_density_map: smoothed (convolved) density map with a 2D kernel
            
    """
    
    density_map = calculate_2d_density_map(reduced_feature, weights)
    convolved_density_map = scipy.ndimage.convolve(density_map, kernel, mode='constant')

    return density_map, convolved_density_map


def calculate_2d_density_map(reduced_feature, weights=None):
    """
        this function is the most basic process to calculate the 2D density map. It creates an empty grid first, then count
        the number of dots around each grid point. Finally it generates a 'point density map' which illustrates the spatial
        distribution of data points.
        
        input:
            
            reduced_feature: reduced_feature 2D feature with shape (num_samples, 2)
            
            weights: original weights to indicate the size (radius) of each dot in the reduced_feature coordinate. If weights are not
        None, the grid accumulates with 1 for each dot; othwise the grid accumulates with wight (firing rate) for each dot.
            
        return:
            
            density_map: point density map, if the weights is None, it will generate a density map without 'weights'
            
    """
    
    # --- init weights if no weights are assigned
    if weights is None:
        weights = np.ones(500)
    
    # --- create an empty density map
    x = reduced_feature[:, 0] - np.min(reduced_feature[:, 0])
    y = reduced_feature[:, 1] - np.min(reduced_feature[:, 1])
    
    density_map = empty_density_map(y, x)
    
    # --- calculte the point density map
    for i in range(len(x)):
        density_map[round(y[i]), round(x[i])] += weights[i]

    return density_map

def empty_density_map(y, x):
    
    return np.zeros((np.ceil(np.max(y)+1).astype(int), np.ceil(np.max(x)+1).astype(int)))

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    model_name = 'vgg16_bn'
    
    #model_ = vgg.__dict__[model_name](num_classes=50)
    #layers, neurons, shapes = utils_.generate_vgg_layers_list_ann(model_, model_name)
    
    layers = ['neuron_2']
    neurons  = [4096]
    
    root_dir = '/home/acxyle-workstation/Downloads'
    
    for folder in [
                   'Face Identity VGG16bn', 
                   'Face Identity SpikingVGG16bn_IF_T4_CelebA2622', 
                   'Face Identity SpikingVGG16bn_LIF_T4_CelebA2622', 
                   'Face Identity SpikingVGG16bn_LIF_T4_vggface', 
                   'Face Identity SpikingVGG16bn_LIF_T16_CelebA2622'
                   ]:
        
        print(f'[Codinfo] Folder {folder}')
        
        selectivity_feature_analyzer = Selectivity_Analysis_Feature(
                    root=os.path.join(root_dir, folder), layers=layers, neurons=neurons
                    )
        
        selectivity_feature_analyzer.feature_analysis('TSNE')

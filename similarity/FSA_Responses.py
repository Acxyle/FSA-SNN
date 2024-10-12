#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:13:51 2024

@author: Jinge Wang, Runnan Cao
@modified: acxyle

    refer to: https://github.com/JingeW/ID_selective
              https://osf.io/824s7/
    

"""


# --- python
import os
import numpy as np
from tqdm import tqdm
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed

#import pandas as pd
#from collections import Counter
#from scipy.interpolate import interp1d

# --- local
import utils_
from .FSA_Encode import FSA_Encode
from bio_records_process.human_feature_process import human_feature_process



# ----------------------------------------------------------------------------------------------------------------------
__all__ = ["FSA_Responses", "FSA_Responses_folds"]


# ----------------------------------------------------------------------------------------------------------------------
class FSA_Responses(FSA_Encode):
    """
        ...
    """
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.dest_Responses = os.path.join(self.dest_Encode, 'Responses')
        utils_.make_dir(self.dest_Responses)
        
        self.Sort_dict = self.load_Sort_dict(**kwargs)
        
        ...
    
    
    def calculation_Feature_Intensity(self, used_unit_types, **kwargs):
        
        self.dest_Intensity = os.path.join(self.dest_Responses, 'Intensity')
        utils_.make_dir(self.dest_Intensity)
        
        save_path = os.path.join(self.dest_Intensity, 'Intensity.pkl')
        save_path_units_pct = os.path.join(self.dest_Intensity, 'units_pct.pkl')
        
        # ---
        if os.path.exists(save_path):
            
            Intensity_dict = utils_.load(save_path)
            
        else:
        
            def _single_layer_process(layer, sort_dict):
                
                feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), verbose=False)
                feature = np.mean(feature.reshape(self.num_classes, self.num_samples, -1), axis=1)     # (50, num_samples)
                
                mean = {}
                std = {}
                log_mean = {}
                log_std = {}
                zero_pct = {}
                
                for k, v in sort_dict.items():
                
                    subfeature = feature[:, v]    
                
                    mean[k] = np.mean(subfeature)
                    std[k] =np.std(subfeature)
                    log_mean[k] = np.mean(np.log(subfeature[subfeature!=0])/np.log(10))
                    log_std[k] = np.std(np.log(subfeature[subfeature!=0])/np.log(10))
                    zero_pct[k] = np.sum(subfeature==0)/subfeature.size*100
                
                I_dict = {
                    'mean': mean,
                    'std': std,
                    'log_mean': log_mean,
                    'log_std': log_std,
                    'zero_pct': zero_pct
                    }
                
                return I_dict
        
            self.Sort_dict = self.load_Sort_dict()
            Sort_dict = self.calculation_Sort_dict(used_unit_types, **kwargs)
            
            Intensity_dict = {}
            
            pl = Parallel(n_jobs=15)(delayed(_single_layer_process)(layer, Sort_dict[layer]) for layer in self.layers)
            
            Intensity_dict = {k: pl[idx] for idx, k in enumerate(self.layers)}
            Intensity_dict = {k: {__: [Intensity_dict[_][__][k] for _ in self.layers] for __ in ['mean', 'std', 'log_mean', 'log_std', 'zero_pct']} for k in used_unit_types}
            
            utils_.dump(Intensity_dict, save_path)
            
        # ---
        if os.path.exists(save_path_units_pct):
            
            units_pct = utils_.load(save_path_units_pct)
            
        else:
            
            units_pct = self.calculation_units_pct(used_unit_types, **kwargs)
            utils_.dump(units_pct, save_path_units_pct)
            
        return Intensity_dict, units_pct
    
    @staticmethod
    def plot_Feature_Intensity_single(ax, layers, Intensity_dict, used_unit_type, units_pct, direction='horizontal'):
        
        x = np.arange(len(layers))
        
        if direction == 'horizontal':
        
            ax.plot(x, Intensity_dict[used_unit_type]['mean'], label='Values')
            ax.fill_between(x, np.array(Intensity_dict[used_unit_type]['mean'])-np.array(Intensity_dict[used_unit_type]['std']), np.array(Intensity_dict[used_unit_type]['mean'])+np.array(Intensity_dict[used_unit_type]['std']), alpha=0.5)
            
            ax.set_ylabel("Firing Rates", color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            ax.set_ylim(bottom=0)
            ax.set_xlim([0, len(layers)-1])
            
            ax2 = ax.twinx()
            ax2.plot(Intensity_dict[used_unit_type]['zero_pct'], color='red', marker='.', label='% 0')
            ax2.set_ylim([0, 105])
            ax2.set_ylabel("%", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
    
            ax2.plot(units_pct[used_unit_type], linestyle='--', color='coral', marker='d', label='% Units')
            ax2.grid(True, axis='y', linestyle='--', linewidth=0.5, color='red', alpha=0.5)
        
        elif direction == 'vertical':
            
            ax.plot(Intensity_dict[used_unit_type]['mean'], x)
            ax.fill_betweenx(x, np.array(Intensity_dict[used_unit_type]['mean'])-np.array(Intensity_dict[used_unit_type]['std']), np.array(Intensity_dict[used_unit_type]['mean'])+np.array(Intensity_dict[used_unit_type]['std']), alpha=0.5)
            
            ax.set_xlabel("Firing Rates", color='blue')
            ax.tick_params(axis='x', labelcolor='blue')
            ax.set_xlim(left=0)
            ax.set_ylim([0, len(layers)-1])
            ax.invert_xaxis()
            
            ax2 = ax.twiny()
            ax2.plot(Intensity_dict[used_unit_type]['zero_pct'], x, color='red', marker='.')
            ax2.set_xlim([0, 105])
            ax2.set_xlabel("%", color='red')
            ax2.tick_params(axis='x', labelcolor='red')
            ax2.plot(units_pct[used_unit_type], x, linestyle='--', color='coral', marker='d')
            ax2.grid(True, axis='x', linestyle='--', linewidth=0.5, color='red', alpha=0.5)
            ax2.invert_xaxis()
    
    
    def plot_Feature_Intensity(self, used_unit_types=None, **kwargs):
        
        used_unit_types = self.advanced_types_display if used_unit_types == None else used_unit_types
        
        self.dest_Intensity = os.path.join(self.dest_Responses, 'Intensity')
        utils_.make_dir(self.dest_Intensity)
        
        Intensity_dict, units_pct = self.calculation_Feature_Intensity(used_unit_types)

        for used_unit_type in used_unit_types:

            fig, ax = plt.subplots(figsize=(10, 3))

            self.plot_Feature_Intensity_single(ax, self.layers, Intensity_dict, used_unit_type, units_pct, direction='horizontal')
            
            ax.set_title(used_unit_type)
            fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.9), framealpha=0.5)
            
            fig.savefig(os.path.join(self.dest_Intensity, f'{self.model_structure} {used_unit_type}.svg'), bbox_inches='tight')
            plt.close()

    
    
    def plot_unit_responses(self, random_select_units=10, start_layer_idx=-5, cmap='jet', **kwargs):
        """
            threshold: r'$V_{th}=\bar{x}+2\sqrt{\frac{1}{500}\sum(x_i-\bar{x})^2}$'
            ref: r'$ref=\bar{x}+2\sqrt{\frac{1}{50}\sum(x_i-\bar{x_i})^2}, \bar{x_i} = \frac{1}{10}\sum{x_i}$'
            
        """
        # --- init ---
        utils_.make_dir(fig_folder:=os.path.join(self.dest_Responses, 'Single Units'))

        colors = [plt.get_cmap(cmap, self.num_classes)(i) for i in range(self.num_classes)]
        
        # ---
        def _plot_unit_responses_layer(unit_idx, input, **kwargs):

            local_means, global_mean, threshold, ref = calculation_unit_responses(input, **kwargs)
            
            # -----
            fig, ax = plt.subplots(figsize=(6, 6))
            
            plot_unit_responses(ax, input, local_means, colors=colors, vmin=vmin, vmax=vmax, **kwargs)

            # ---
            ax.hlines(0., 0, self.num_classes-1, colors='gray', linestyle='-')
            ax.hlines(threshold, 0, self.num_classes-1, colors='red', linestyle='--', label='threshold')
            ax.hlines(ref, 0, 49, colors='teal', linestyle='--', label='ref')
            #ax.set_ylim([vmin, vmax])
            
            # ---
            handles, labels = ax.get_legend_handles_labels()
            
            mean = Line2D([0], [0], marker='d', markersize=5, markeredgecolor='none', color='gray', linestyle='--', linewidth=1)
            median = Line2D([0], [0], marker='_', markersize=5, color='orange', linewidth=0)
            outlier = Line2D([0], [0], marker='+', markersize=5, markeredgecolor='gray', linewidth=0)

            handles.extend([mean, median, outlier])
            labels.extend(['mean', 'median', 'outlier'])
            
            ax.legend(handles, labels, framealpha=0.75)
            
            # ---
            fig.suptitle(f'[{layer}] unit: {unit_idx}', y=0.975)
            fig.tight_layout()
            
            fig.savefig(os.path.join(type_layer_fig_folder, f'{cell_type}_{unit_idx}.svg'))
            plt.close()
        
        # ---
        for layer in self.layers[start_layer_idx:]:
             
            utils_.make_dir(layer_fig_folder:=os.path.join(fig_folder, f'{layer}'))
            feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), normalize=True, sort=True, verbose=False, **kwargs)
            
            vmin = np.min(feature)
            vmax = np.max(feature)
            
            for cell_type, v  in tqdm(self.Sort_dict[layer].items(), desc=f'{layer}'):     # for each type
                
                utils_.make_dir(type_layer_fig_folder:=os.path.join(layer_fig_folder, f'{cell_type}'))
            
                if v.size != 0:
 
                    if v.size > random_select_units:
                        
                        v = np.random.choice(v, random_select_units)

                    Parallel(n_jobs=random_select_units)(delayed(_plot_unit_responses_layer)(unit_idx, feature[:, unit_idx], **kwargs) for unit_idx in v)  
                    

    def plot_stacked_responses(self, used_unit_types:list[str], start_layer_idx=-5, **kwargs) -> None:
        """ RAM consuming """
                
        assert start_layer_idx < 0, f'start_layer_idx {start_layer_idx} must be negative in current design'

        utils_.formatted_print(f'Executing plot_stacked_responses... | {used_unit_types} | num_layers: {np.abs(start_layer_idx)}')
        
        num_types = len(used_unit_types)
        
        # ---
        utils_.make_dir(fig_folder:=os.path.join(self.dest_Responses, 'Stacked Responses'))
        utils_.make_dir(type_fig_folder:=os.path.join(fig_folder, str(num_types)))
        
        # ---
        Sort_dict = self.calculation_Sort_dict(used_unit_types)
        
        # ---
        length = num_types*5+1
        
        figsize = (length, 6) if num_types != 10 else (26, 10)
        gs_rows, gs_cols = (1, num_types) if num_types != 10 else (2, 5)
        
        for layer in self.layers[start_layer_idx:]:
            
            fig, ax = plt.subplots(figsize=figsize)
            gs_main = gridspec.GridSpec(gs_rows, gs_cols, figure=fig)
        
            feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), normalize=True, sort=True, verbose=False, **kwargs)
            plot_stacked_responses(fig, gs_main, layer, Sort_dict[layer], feature)
        
            ax.axis('off')
            ax.plot([], [], color='blue', label='mean')
            ax.plot([], [], color='teal', linestyle='--', label='ref')
            ax.plot([], [], color='red', linestyle='--', label='threshold')
        
            fig.suptitle(f'{layer} [{self.model_structure}]', y=0.97, fontsize=20)
            fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, bbox_transform=plt.gcf().transFigure)
        
            fig.tight_layout()
            fig.savefig(os.path.join(type_fig_folder, f'{layer} num_types {num_types}.png'), bbox_inches='tight')
            plt.close()

            #mem = psutil.virtual_memory()
            #swap = psutil.swap_memory()
            #print(f'\nCPU+SWAP used: {(mem.used+swap.used)/1024/1024/1024:.3f} / 256')
            #print(f'\nCPU used pct: {mem.percent} %')
            #print(sys.getrefcount(self.plot_stacked_responses_single_layer))
        

    def plot_responses_PDF(self, used_unit_types:list[str]=None, start_layer_idx=-5, **kwargs):
        """ this funtion is equivalent to the PDF plot of self.plot_stacked_responses() """
        
        utils_.make_dir(fig_folder:=os.path.join(self.dest_Responses, 'Responses PDF'))
        
        used_unit_types = self.advanced_types_display if used_unit_types is None else used_unit_types
        Sort_dict = self.calculation_Sort_dict(used_unit_types)
        
        for unit_type in used_unit_types:

            utils_.make_dir(save_path:=os.path.join(fig_folder, unit_type))

            for layer in tqdm(self.layers[start_layer_idx:], desc=f'{unit_type} PDF'):
    
                #warnings.simplefilter(action='ignore')
                #logging.getLogger('matplotlib').setLevel(logging.ERROR)
                
                feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), verbose=False, **kwargs)[:, Sort_dict[layer][unit_type]]
                
                fig = human_feature_process.plot_PDF(self.model_structure, 'unit', feature, unit_type=unit_type, **kwargs)
                
                fig.tight_layout()
                fig.savefig(os.path.join(save_path, f'{layer}.svg'))
                
                plt.close()
        
    
    # --- legacy
    def plot_pct_pie_chart(self, used_unit_types=['a_hs', 'a_ls', 'a_hm', 'a_lm', 'non_selective'], start_layer_idx=-5, **kwargs):
        
        utils_.make_dir(save_path:=os.path.join(self.dest_Encode, 'Pie Chart'))
        
        Sort_dict = self.calculation_Sort_dict(used_unit_types)
        
        for layer in tqdm(self.layers[start_layer_idx:], desc='Pir Chart'):
            
            sort_dict = Sort_dict[layer]
            pcts = [_.size/self.units[self.layers.index(layer)]*100 for idx, _ in enumerate(sort_dict.values())]
            
            # ---
            labels = [f'{_}' for idx, _ in enumerate(used_unit_types)]
            colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']
            explode = [0.5 * (1. - (value - min(pcts)) / (max(pcts) - min(pcts))) for value in pcts]
            
            # ---
            fig, ax = plt.subplots(figsize=(10,6))
            utils_.plot_pie_chart(fig, ax, pcts, labels, title=f'{self.units[self.layers.index(layer)]} Units', colors=colors, explode=explode, **kwargs)
            fig.savefig(os.path.join(save_path, f'{layer}_pct_pie_chart.svg'), transparent=True)
            plt.close()
        
    


def plot_unit_responses(ax, input, local_means, colors=None, num_classes=50, num_samples=10, **kwargs):
    """
        ...
    """
 
    # ---
    boxes = ax.boxplot(list(input.reshape(num_classes, num_samples)), patch_artist=True, sym='+', positions=np.arange(50))
    
    for i, _ in enumerate(boxes['boxes']):
        _.set(color=colors[i], alpha=0.5)
        
    for i, _ in enumerate(boxes['fliers']):
        _.set(marker='+', markerfacecolor=colors[i], markeredgecolor=colors[i], markersize=5, alpha=0.75)
    
    # ---
    ax.scatter(np.repeat(np.arange(num_classes), num_samples), input, color=np.repeat(np.array(colors), num_samples, axis=0), s=8, alpha=0.5, edgecolor='none')
    ax.scatter(np.arange(num_classes), local_means, color=colors, marker='d', s=12)
    
    for _ in range(num_classes):
        ax.vlines(_, np.min(input), local_means[_], linestyle='--', alpha=0.75)
    
    ax.set_xticks(ticks:=[0, 9, 19, 29, 39, 49])
    ax.set_xticklabels(ticks)

         
# ----------------------------------------------------------------------------------------------------------------------
def plot_stacked_responses(fig, gs_main, layer, sort_dict, feature, num_classes=50, num_samples=10, scaling_factor=0.1, **kwargs):
    """
        ...
        
        manually set fig size
        
    """
    
    def _plot_stacked_responses_single(input, vmin=0., vmax=1., color=None, linestyle=None, scaling_factor=0.1, **kwargs):
        
        if np.std(input) != 0:

            v_radius = vmax - vmin
            
            dummy_x = np.linspace(vmin-scaling_factor*v_radius, vmax+scaling_factor*v_radius, 101)
            dummy_y = gaussian_kde(input)(dummy_x)
            
            ax_right.plot(dummy_y, dummy_x, linestyle=linestyle, color=color)
            
            if len(y_peak:=np.where(dummy_y==np.max(dummy_y))[0]) == 1:
                x_vals_max = dummy_x[y_peak.item()]
            else:
                x_vals_max = dummy_x[y_peak[0]]
            
            ax_left.hlines(x_vals_max, 0, 50, colors=color, alpha=0.75, linestyle='--')
            ax_right.hlines(x_vals_max, np.min(dummy_y), np.max(dummy_y), colors=color, alpha=0.75, linestyle='--')
        
 
    # ---
    colors = [plt.get_cmap('jet', 50)(i) for i in range(50)]

    tqdm_bar = tqdm(total=len(sort_dict.keys()), desc=f'{layer}')

    y_min = np.min(feature)
    y_max = np.percentile(feature, 99.) if np.max(feature) > 1. else 1.

    y_lin_range = y_max - y_min
    
    for i in range(num_rows:=gs_main.nrows):
        for j in range(num_cols:=gs_main.ncols):
            
            unit_type = list(sort_dict.keys())[i*num_cols+j]
            
            gs_sub = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[4, 1], subplot_spec=gs_main[i, j])

            ax_left = fig.add_subplot(gs_sub[0])
            ax_right = fig.add_subplot(gs_sub[1])
            ax_right.set_yticks([])
            
            if (i+j) != 0:
                ax_left.set_xticks([])
                ax_left.set_yticks([])

            if (num_units:=sort_dict[unit_type].size) == 0:
                ax_left.set_title(f'{unit_type} [0.00%]')
                ax_right.set_title('')

            else:
                
                local_means, global_mean, threshold, ref = [np.array(_) for _ in zip(*[calculation_unit_responses(_, **kwargs) for _ in feature[:, sort_dict[unit_type]].T])]

                # ---
                ax_left.scatter(np.tile(np.arange(num_classes), num_units), local_means.reshape(-1), color=np.tile(np.array(colors), [num_units, 1]), alpha=0.1, marker='.', s=1)
                ax_left.set_title(f'{unit_type} [{num_units/feature.shape[1]*100:.2f}%]')
                
                # ---
                _plot_stacked_responses_single(local_means.reshape(-1), vmin=y_min, vmax=y_max, color='blue', **kwargs)
                _plot_stacked_responses_single(threshold, vmin=y_min, vmax=y_max, color='red', linestyle='dotted', **kwargs)
                _plot_stacked_responses_single(ref, vmin=y_min, vmax=y_max, color='teal', linestyle='dotted', **kwargs)
                
                # ---
                ax_left.set_ylim([y_min-y_lin_range*scaling_factor, y_max+y_lin_range*scaling_factor])
                ax_right.set_ylim([y_min-y_lin_range*scaling_factor, y_max+y_lin_range*scaling_factor])
                ax_right.set_title('PDF')
                
            tqdm_bar.update(1)


def calculation_Encode(input, **kwargs):
    """
        ...
    """
    
    local_means, global_mean, threshold, ref = calculation_unit_responses(input, **kwargs)
    
    encode = np.where(local_means>threshold)[0]     # '>' prevent all 0
    weak_encode = np.setdiff1d(np.where(local_means>ref)[0], encode)
    
    return {'encode': encode, 'weak_encode': weak_encode}


def calculation_unit_responses(input, num_classes=50, num_samples=10, n=2, **kwargs):

    global_mean = np.mean(input)    
    local_means = np.mean(input.reshape(num_classes, num_samples), axis=1)

    threshold = global_mean + n*np.std(input)     # total variance
    ref = global_mean + n*np.std(local_means)     # between-group variance

    return local_means, global_mean, threshold, ref



class FSA_Responses_folds(FSA_Responses):

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.dest_Responses = os.path.join(self.dest_Encode, 'Responses')
        utils_.make_dir(self.dest_Responses)
        
        self.plot_stacked_responses_folds()
    
    def plot_stacked_responses_folds(self, **kwargs):
    
        print('6')

# ======================================================================================================================
if __name__ == "__main__":
    
    FSA_root = '/home/acxyle-workstation/Downloads/FSA'
    FSA_dir = 'VGG/SpikingVGG'
    model_depth = 16
    T = 8
    FSA_config = f'SpikingVGG16bn_IF_ATan_T4_C2k_fold_'
    FSA_model =  f'spiking_vgg16_bn'
    
    used_unit_types = ['high_selective', 'low_selective', 'a_ne', 'non_anova']
    
    _, layers, units, shapes = utils_.get_layers_and_units(FSA_model, 'act')
    
# =============================================================================
#     root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
#     selectivity_analyzer = FSA_Responses(root=root, layers=layers, units=units)
#     #selectivity_analyzer.plot_unit_responses()
#     selectivity_analyzer.plot_stacked_responses(used_unit_types)
#     #selectivity_analyzer.plot_responses_PDF()
#     #selectivity_analyzer.plot_pct_pie_chart()
#     #selectivity_analyzer.plot_Feature_Intensity()
# =============================================================================
    
    
    for _ in range(1):
        
        root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}/-_Single Models/FSA {FSA_config}{_}')
        selectivity_analyzer = FSA_Responses(root=root, layers=layers, units=units)
        selectivity_analyzer.plot_unit_responses()
        #selectivity_analyzer.plot_stacked_responses(used_unit_types)
        #selectivity_analyzer.plot_responses_PDF()
        #selectivity_analyzer.plot_pct_pie_chart()
        #selectivity_analyzer.plot_Feature_Intensity()
    
# =============================================================================
#     root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
#     selectivity_analyzer = FSA_Responses_folds(root=root, layers=layers, units=units)
# =============================================================================
    

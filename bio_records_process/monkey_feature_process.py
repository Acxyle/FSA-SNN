#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 13:35:41 2023

@author: Runnan Cao
@modified: acxyle

    refer to: https://osf.io/824s7/
    
"""

# --- python
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
#from tqdm import tqdm

# --- local
import sys
sys.path.append('../')     # if run this code as script
import utils_
#from utils_ import _bio_cells, utils_similarity

from .primate_feature_process import primate_feature_process

# ======================================================================================================================
class monkey_feature_process(primate_feature_process):
    """ Unlike human cell data, no data process here due to the Monkey data is a well processed dataset """
    
    def __init__(self, primate='Monkey', seed=6, local_data_root='/home/acxyle-workstation/Downloads/Bio Neuron Data', **kwargs):
        """
            this function determines the time range of interest [-50, 200] from the original time range [-100, 380], 
            follow original Matlab code
        """
        super().__init__(seed=seed, **kwargs)
        
        self.bio_root = os.path.join(local_data_root, primate)
        self.ts = np.arange(-50, 201, 10)

        self.mat_to_py()
        

    def mat_to_py(self, ):
        """
            this function converts the original .mat file to a python dict
            The **img** FR and PSTH are not under the natural order but indices of displayed imgs
        """
        
        data_path = os.path.join(self.bio_root, 'data.pkl')
        
        # -----
        if os.path.exists(data_path):
            
            data = utils_.load(data_path, verbose=False)
            
            self.FR_countAll = data['FR_count_all']
            self.FR_countBase = data['FR_count_base']
            self.FR_countVis = data['FR_count_vis']
            
            self.meanFR = data['meanFR']
            self.meanBase = data['meanBase']
            self.meanGray = data['meanGray']
            self.meanVis = data['meanVis']
            
            self.psthTime = data['psthTime']
            self.meanPSTH = data['meanPSTH']
            self.meanPSTHID = data['meanPSTHID']
        
        else:
            
            # convert .mat to python dict
            monkey_neuron_data_path = os.path.join(self.bio_root, 'Original Data/IT_FR_CA_Range70-180.mat')     # processed monkey neural data
            monkey_neuron_data = sio.loadmat(monkey_neuron_data_path)

            monkey_dict_keys = [i for i in monkey_neuron_data.keys() if '__' not in i]
            monkey_dict = {_:monkey_neuron_data[_] for _ in monkey_dict_keys}     # rebuild the dict to store monkey IT MUA data
            
            # -----
            self.FR = monkey_dict['FR']
            
            # ---
            self.FR_countAll = self.FR['countAll'][0][0]     
            self.FR_countBase = self.FR['countBase'][0][0]
            self.FR_countVis = self.FR['countVis'][0][0]
            
            # ---
            self.meanFR = monkey_dict['meanFR']     # (53, 500)
            self.meanBase = monkey_dict['meanBase']     # (53, 500)
            self.meanGray = monkey_dict['meanGray'].reshape(-1)     # (53, )
            self.meanVis = monkey_dict['meanVis']     # (53, 500)
            
            # ---
            self.psthTime = monkey_dict['psthTime'].reshape(-1)     # (49,)
            self.meanPSTH = monkey_dict['meanPSTH']     # (500, 49, 53), [disordered img idx, time steps, channels], normalized
            self.meanPSTHID = monkey_dict['meanPSTHID']     # (50, 49, 53), [id idx, time steps, channels], normalized
            
            data = {
                'FR_count_all': self.FR_countAll,
                'FR_count_base': self.FR_countBase,
                'FR_count_vis': self.FR_countVis,
                
                'meanFR': self.meanFR,
                'meanBase': self.meanBase,
                'meanGray': self.meanGray,
                'meanVis': self.meanVis,
                
                'psthTime': self.psthTime,
                'meanPSTH': self.meanPSTH,
                'meanPSTHID': self.meanPSTHID
                }
            
            utils_.dump(data, data_path)
    
    
    def calculation_feature(self, time_bin=10):
        """
            this function calculates the firing rate and PSTH of monkey neuronal data
        
            return:
                feature dict of 'FR_id' and 'psth_id' of natural order
        """
        
        file_path = os.path.join(self.bio_root, 'features.pkl')
        
        if os.path.exists(file_path):
            
            feature_dict = utils_.load(file_path, verbose=False)
            
            FR_id = feature_dict['FR_id']
            psth_id = feature_dict['psth_id']
        
        else:
        
            label = sio.loadmat(os.path.join(self.bio_root, 'Original Data/Label.mat'))['label'].reshape(-1)
            
            # ----- FR
            sacling_factor = self.meanGray
            FR_id = np.array([np.mean(self.meanFR[:, np.where(label==_)[0]], axis=1)/sacling_factor for _ in range(1, 51)])     # (50, 53)
            
            # ----- PSTH
            if time_bin == 10:
                used_psth = self.meanPSTH[:, [np.where(self.psthTime==_)[0][0] for _ in self.ts], :]
                
            else:
                used_psth = np.zeros((self.meanPSTH.shape[0], len(self.ts), self.meanPSTH.shape[2]))     # (500, 26, 53) (img, time, unit)
                for idx, tt in enumerate(self.ts): 
                    used_psth[:, idx, :] = np.mean(self.meanPSTH[:, np.where(((tt-time_bin/2)<=self.psthTime) & (self.psthTime<=(tt+time_bin/2)))[0], :], axis=1)
            
            # ---
            used_psth_id = np.array([np.mean(used_psth[np.where(label==_)[0], :, :], axis=0) for _ in  range(1, 51)])     
            used_psth_id = np.transpose(used_psth_id, (1,0,2))     # (time, ID, unit)
            
            scaling_factor = np.mean(self.meanBase, axis=1)
            psth_id = np.array([np.array([used_psth_id[i, j, :]/scaling_factor for j in range(50)]) for i in range(used_psth.shape[1])])     # (26, 50, 53)
            
            # ---
            feature_dict = {
                'FR_id': FR_id,
                'psth_id': psth_id
                }
            
            utils_.dump(feature_dict, file_path)
            
        return FR_id, psth_id
        
    
    def calculation_DSM_monkey(self, first_corr='pearson', vectorize=False, verbose=False, **kwargs):
        
        utils_.make_dir(save_root:=os.path.join(self.bio_root, 'DSM'))
        
        save_path = os.path.join(save_root, f'Monkey_DSM_{first_corr}.pkl')
        
        if os.path.exists(save_path):
            
            (DM, DM_temporal) = utils_.load(save_path, verbose=verbose)
            
        else:
            
            FR_id, psth_id = self.calculation_feature()
            DM, DM_temporal = self.calculation_1st_stats('DSM', FR_id, psth_id, first_corr=first_corr, **kwargs)
            
            utils_.dump((DM, DM_temporal), save_path, verbose=verbose)
        
        return DM, DM_temporal
        
    
    def calculation_DSM_perm_monkey(self, first_corr='pearson', vectorize=False, **kwargs):

        DM, DM_temporal = self.calculation_DSM_monkey(first_corr, **kwargs)
        
        DM_perm, DM_temporal_perm = self.calculation_1st_stats_perm(DM, DM_temporal, **kwargs)
        
        return DM, DM_temporal, DM_perm, DM_temporal_perm
        
    
    def calculation_Gram_monkey(self, kernel='linear', **kwargs):

        utils_.make_dir(os.path.join(self.bio_root, 'Gram'))
        
        if kernel == 'rbf' and 'threshold' in kwargs:
            save_path = os.path.join(self.bio_root, 'Gram', f"CKA_results_{kernel}_{kwargs['threshold']}.pkl")
        elif kernel == 'linear':
            save_path = os.path.join(self.bio_root, 'Gram', f"CKA_results_{kernel}.pkl")
        else:
            raise ValueError
            
        if os.path.exists(save_path):
            
            (Gram, Gram_temporal) = utils_.load(save_path, verbose=False)
            
        else:

            FR_id, psth_id = self.calculation_feature()
            
            Gram, Gram_temporal = self.calculation_1st_stats('Gram', FR_id, psth_id, kernel=kernel, **kwargs)

            utils_.dump((Gram, Gram_temporal), save_path, verbose=False)
            
        return Gram, Gram_temporal
    
    
    def calculation_Gram_perm_monkey(self, kernel='linear', **kwargs):

        Gram, Gram_temporal = self.calculation_Gram_monkey(kernel, **kwargs)
        
        Gram_perm, Gram_temporal_perm = self.calculation_1st_stats_perm(Gram, Gram_temporal, **kwargs)
            
        return Gram, Gram_temporal, Gram_perm, Gram_temporal_perm
    
    
    def plto_example(self, average=True):
        """
            this function plot the sample responses of monkey IT cells
            
            Parameter:
                average: if True, plot the average responses across channels, otherwise plot the channel with strongest
                reponses, i.e. channel 51 (0-based)
        """
        
        # --- normalized
        scaling_factor = np.mean(self.meanBase, axis=1)     # (53,)

        meanPSTHIDNorm = self.meanPSTHID/scaling_factor     # [id idx, time steps, channels]
        
        if average:
            meanPSTHIDNorm = np.mean(meanPSTHIDNorm, axis=2)     # [time, id]
            title = 'Monkey normalized channel-average PSTH'
        else:
            target_channel = np.argmax(np.mean(meanPSTHIDNorm, axis=(0, 1)))
            meanPSTHIDNorm = meanPSTHIDNorm[..., target_channel]
            title = f'Monkey normalized channel {target_channel} PSTH'
        
        # -----
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 18})
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        plot_PSTH(fig, ax, meanPSTHIDNorm, title, self.psthTime, -50, 200)
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.bio_root, f'{title}.svg'))
        plt.close()


# ======================================================================================================================    
def plot_PSTH(fig, ax, PSTH, title=None, time_point=None, time_start=None, time_end=None):
    """
        ...
    """
    img = ax.imshow(PSTH, aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(img)
    
    if time_point is not None:
        
        ax.vlines(np.where(time_point==time_start)[0][0], 0, PSTH.shape[0], linestyle='--', color='red', linewidth=3)
        ax.vlines(np.where(time_point==time_end)[0][0], 0, PSTH.shape[0], linestyle='--', color='red', linewidth=3)
        
        loc = np.arange(np.where(time_point==time_start)[0][0], np.where(time_point==time_end)[0][0])
        loc = np.append(loc, max(loc)+1)
        
        loc = np.where(time_point%50==0)[0]
        ax.set_xticks(loc, time_point[loc], fontsize=14)
        
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel('ID',fontsize=20)
    
    ax.set_ylim([0, PSTH.shape[0]-1])
    ax.set_title(title, fontsize=24)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    monkey_record_process = monkey_feature_process()
    
    monkey_record_process.mat_to_py()
    monkey_record_process.plto_example()
    monkey_record_process.calculation_feature()
    
    monkey_record_process.calculation_DSM_monkey('spearman')
    monkey_record_process.calculation_DSM_perm_monkey('pearson')
    
    monkey_record_process.calculation_Gram_monkey(kernel='linear')
    monkey_record_process.calculation_Gram_perm_monkey()
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 20:06:27 2023

@author: acxyle

    refer to: https://www.biorxiv.org/content/10.1101/2020.09.01.278283v2.abstract
    
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def get_session_idces():
    """ Those are manually selected files. The CelebA/FBI/NavFace indicates the used datasets """
    sessions = [
            'p6WV_CelebA_Sess1',
            'p6WV_CelebA_Sess2',
            'p7WV_CelebA_Sess1',
            'p7WV_CelebA_Sess2',
            'p7WV_CelebA_Sess3',
            'p7WV_CelebA_Sess4',
            'p9WV_CelebA_Sess1',
            'p9WV_CelebA_Sess2',
            'p9WV_CelebA_Sess3',
            'p9WV_CelebA_Sess4',
            'p10WV_CelebA_S2_FBI_S2',
            'p10WV_CelebA_Sess3',
            'p10WV_Loc2_S1_CelebA_S1_FBI_S1',
            'p11WV_CelebA_S1_FBI_S1_Loc2_S1',
            'p11WV_CelebA_S2_FBI_S2_Loc2_S2',
            'p11WV_CelebA_S3_FBI_S3_Loc2_S3',
            'p11WV_CelebA_S4_FBI_S4_Loc2_S4',
            'p11WV_CelebA_Sess5',
            'p13WV_CelebA_Sess1',
            'p14WV_CelebA_S1_FBI_S1',
            'p14WV_CelebA_S2_FBI_S2',
            'p14WV_CelebA_S3_FBI_S3',
            'p14WV_CelebA_S4_FBI_S4',
            'p15WV_CelebA_S1_FBI_S1',
            'p15WV_CelebA_S2_FBI_S2',
            'p16WV_CelebA_S1',
            'p16WV_CelebA_S2_NavFace_S1',
            'p16WV_CelebA_S3_NavFace_S3',
            'p16WV_CelebA_S4_NavObj_S2',
            'p16WV_CelebA_S5_FBI_S1_NavFace_S4',
            'p16WV_CelebA_S6_NavFace_S5',
            'p18WV_CelebA_S1_FBI_S1',
            'p18WV_CelebA_S2_NavFace_S1',
            'p18WV_CelebA_S3_NavFace_S2',
            'p18WV_CelebA_S4',
            'p19WV_CelebA_S1_NavFace_S1',
            'p19WV_CelebA_S2',
            'p20WV_CelebA_S1_NavFace_S1',
            'p20WV_CelebA_S2_NavFace_S2',
            'p20WV_CelebA_S3_FBI_S1']
    
    return sessions


def plotSpikeRasterMain(ax, spikes, colors, spikeheight=3, spikewidth=3, start_time=0, end_time=2000):
    """
        some of the hyper paramaters are designed only for this experiments, needed to be changed in future use
        
        this function simply plot spike (vertical lines) based on actual time. please refer to spikingjelly tutorial
        for ax.eventplot() of plt if dsicrete time
    """

    if len(spikes) != 0:
        for idx, _ in enumerate(spikes):     # for each trial
            if _[3] is not np.nan and _[3].size != 0:

                n = _[3].size
                
                ax.plot(
                    np.vstack((_[3], _[3])), np.vstack(((idx - spikeheight/2)*np.ones(n), (idx + spikeheight/2)*np.ones(n))), 
                    linewidth=spikewidth, color=colors[_[0]-1]
                    )


def getTimestampsOfBubbles(timestampsOfCell, periods_and_infos, session_idx, adjust_idx_dict, id_img_idces_dict):
    """
        prepare bubbles trials for plotting of raster (with color info)
        periods_and_infos: list of periods (each a list of trials)     [ order in experiment | trial start time | trial end time | img label | ID ]
        
        urut/nov09
        
        timestampsOfCell: disordered img sequence
        
        return: spikes_to_plot (ID, img_idx, num_imgs, spikes)
    """

    spikes_to_plot = []

    for inds in periods_and_infos:     # for each group
      
        trialTimestamps = getRelativeTimestamps(timestampsOfCell, inds)     # timestamps for one ID,  | ID | img_idx | img_num | timestamps |
        
        # ----- intergrity check
        if session_idx < 10: 
            img_idces = np.array([adjust_idx_dict[_] for _ in sorted([_[1] for _ in trialTimestamps])])     # img_idces correction
        else:
            img_idces = np.array(sorted([_[1] for _ in trialTimestamps]))
            
        # --- integrity check. ID must be one
        ID = np.unique([trialTimestamps[_][0] for _ in range(len(trialTimestamps))]).item()

        entire_imgs = id_img_idces_dict[ID-1]
        
        if img_idces.shape != entire_imgs.shape:
            missed_imgs = np.setdiff1d(entire_imgs, img_idces)
        
            for _ in missed_imgs:
                trialTimestamps.append([ID, _, np.nan, np.nan])
            
        # --- sort based on label, this order is also the order in each folder
        trialTimestamps = sorted(trialTimestamps, key=lambda x:x[1])
        
        # ----- add into all
        for _ in trialTimestamps:
            spikes_to_plot.append(_)
    
    return spikes_to_plot
    

def getRelativeTimestamps(timestampsOfCell, periods):
    """
        reference timestamps to beginning of the trial
        periods: each row is one trial. 3 columns: trial nr, from, to.
        returns a cell array; each item contains the timestamps of one trial. the number of trials is equal to the number of rows in periods.
        
        urut/dec07
        
        modified: acxyle/Oct17,2023
    """
    
    trialTimestamps = []
    
    for i in range(periods.shape[0]):     # for each trial
        
        inds = np.where((periods[i, 1] <= timestampsOfCell) & (timestampsOfCell <= periods[i, 2]))[0]
        
        # | ID | img_idx | img_num | timestamps |
        trialTimestamps.append([periods[i, 4].astype(int), periods[i, 3].astype(int), periods.shape[0], (timestampsOfCell[inds] - periods[i, 1])/1000])
        
    return trialTimestamps


# ---
def _plot_spike_raster(ax, spikes_to_plot, colors, spikeheight=5, spikewidth=3, text=True):
    
    plotSpikeRasterMain(ax, spikes=spikes_to_plot, colors=colors, spikeheight=spikeheight, spikewidth=spikewidth)
    
    ax.set_facecolor('#F5F5F5')
    
    if text:
    
        ax.vlines(500, -1, 501, linestyle='-', alpha=0.75, color='gray', label='image on')
        ax.vlines(1500, -1, 501, linestyle='--', alpha=0.75, color='gray', label='image off')
        
        ax.vlines(750, -1, 501, linestyle='-', alpha=0.75, color='tomato', label='count on')
        ax.vlines(1750, -1, 501, linestyle='--', alpha=0.75, color='tomato', label='count off')
        
        ax.set_xlim([0, 2000])
        
        ax.set_xticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
        ax.set_xticklabels([-500, -250, 0, 250, 500, 750, 1000, 1250, 1500], fontsize=20)
        
        ax.set_ylim([- spikeheight/2, 500 + spikeheight/2])
    
        ax.tick_params(axis='both', labelsize=28, width=2, length=5, labelcolor='black', labelbottom=True)
        ax.set_xlabel('Time (ms)', fontsize=30)
        ax.set_ylabel('Images (10 imgs/ID)', fontsize=30)

        ax.set_title('Raster Plot', fontsize=32)
        ax.legend(fontsize=28, loc='upper right')


def _plot_bar_chart(ax, FR_ID, colors):
    
    # ----- stats
    FR_ID_all = np.array([__ for _ in FR_ID for __ in _])
    FR_ID_all_mean = np.mean(FR_ID_all)
    FR_ID_std = np.std(FR_ID_all)
    
    FR_ID_mean = np.array([np.mean(_) for _ in FR_ID])
    FR_ID_ref = np.std(FR_ID_mean)
    
    # -----
    ax.barh(np.arange(50), FR_ID_mean, color=colors)
    
    sem_list = [np.std(FR_ID[_])/np.sqrt(len(FR_ID[_])) for _ in range(len(FR_ID))]
    
    for idx, _ in enumerate(FR_ID):
        
        ax.scatter(FR_ID_mean[idx] + sem_list[idx], idx, color='black', marker='d')
        ax.scatter(FR_ID_mean[idx] - sem_list[idx], idx, color='black', marker='d')

        ax.hlines(idx, FR_ID_mean[idx] - sem_list[idx], FR_ID_mean[idx] + sem_list[idx], linestyle='--', color='black')
    
    ax.vlines(FR_ID_all_mean+2*FR_ID_std, -1, 50, linestyle='--', alpha=0.75, color='red', label='threshold')
    ax.vlines(FR_ID_all_mean+2*FR_ID_ref, -1, 50, linestyle='--', alpha=0.75, color='teal', label='ref')
    
    ax.tick_params(axis='both', labelsize=20, width=2, length=5, labelcolor='black', labelbottom=True)
    ax.set_xlabel('Firing Rate (Hz)', fontsize=24)
    ax.set_ylabel('ID', fontsize=24)
    #ax.set_yticks([])
    
    ax.set_xlim([0, np.max([np.max(FR_ID_mean) + np.max(sem_list), FR_ID_all_mean+2*FR_ID_std])*1.2])
    ax.set_ylim([-0.5, 49.5])

    ax.legend(fontsize=24)
    ax.set_title('Mean Firing Rate [750ms - 1750ms] (with SE)', fontsize=28)


def _plot_psth(fig, ax, PSTH_ID):
    
    PSTH_ID_ = np.array([np.mean(_, axis=0) for _ in PSTH_ID])
    
    fig_psth = ax.imshow(PSTH_ID_, origin='lower', aspect='auto', cmap='turbo')
    
    ax.set_xticks([0,5,10,15,20,25,30])
    ax.set_xticklabels([-250, 0, 250, 500, 750, 1000, 1250], fontsize=20)
    ax.tick_params(axis='both', labelsize=20, width=2, length=5, labelcolor='black', labelbottom=True)
    ax.set_xlabel('Time (ms) [250ms - 2000ms]', fontsize=24)
    ax.set_ylabel('ID', fontsize=24)
    #ax.set_yticks([])
    ax.set_title('PSTH [window: 50ms, step: 250ms]', fontsize=28)
    
    cbar_ax = fig.add_axes([0.6125, 0.05, 0.01, 0.325])  # [left, bottom, width, height]
    cbar = fig.colorbar(fig_psth, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=24)


def _plot_table(ax, cell_stats, cell_idx, FR_stats, FR_tmp):
    
    all_neurons_of_this_session = cell_stats['cell_attr'][cell_idx]['all_neurons_of_this_session']
    all_neurons_of_this_patient = cell_stats['cell_attr'][cell_idx]['all_neurons_of_this_patient']
    
    FR_all = [__ for _ in range(2082) for __ in FR_stats[_]['spike_count']]
    FR_session = [__ for _ in all_neurons_of_this_session for __ in FR_stats[_]['spike_count']]
    FR_patient = [__ for _ in all_neurons_of_this_patient for __ in FR_stats[_]['spike_count']]
    
    data = [
        [int(np.max(FR_tmp)), int(np.max(FR_session)), int(np.max(FR_patient)), int(np.max(FR_all))],
        [int(np.min(FR_tmp)), int(np.min(FR_session)), int(np.min(FR_patient)), int(np.min(FR_all))],
        [int(np.median(FR_tmp)), int(np.median(FR_session)), int(np.median(FR_patient)), int(np.median(FR_all))],
        [np.mean(FR_tmp).round(2), np.mean(FR_session).round(2), np.mean(FR_patient).round(2), np.mean(FR_all).round(2)],
        [np.std(FR_tmp).round(2), np.std(FR_session).round(2), np.std(FR_patient).round(2), np.std(FR_all).round(2)]
        ]
    
    # ----- 
    col_labels = ['cell', f'session\n({len(all_neurons_of_this_session)})', f'patient\n({len(all_neurons_of_this_patient)})', 'all cells\n(2082)']
    row_labels = ['max', 'min', 'median', 'mean', 'std']
    
    ccolors = plt.cm.BuPu(np.linspace(0, 0.5, len(col_labels)))
    rcolors = plt.cm.BuPu(np.linspace(0, 0.5, len(row_labels)))


    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=col_labels, rowLabels=row_labels, 
            cellLoc='center', rowLoc='center', colColours=ccolors, rowColours=rcolors, loc='center',
            colWidths=[0.25]*4, 
            bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(26)
    
    ax.set_title('Statistic (Firing Rate)', fontsize=28)


def _compare_beh_and_behm(behavior):
    """
        compare the beh_matlab and beh_python
        [Oct 5, 2023] beh_python is identical with beh_matlab
    """
    # --- convert beh_matlab to python friendly structure
    beh_m = sio.loadmat('/home/acxyle-workstation/Downloads/Bio_Neuron_Data/Human/osfstorage-archive/SingleNeuron/FiringRate/Original Data/SortedFR_CelebA.mat')['beh'].reshape(-1)
    beh_m = {_: beh_m[_] for _ in beh_m.dtype.names}
    beh_m = [{__: beh_m[__][_].reshape(-1) for __ in behavior[0].keys()} for _ in range(40)]
    
    for i in range(40):
        beh_m[i]['iT'] = {_: np.array([np.nan if ___.size == 0 else ___.item() for ___ in beh_m[i]['iT'][_]]) for _ in beh_m[i]['iT'].dtype.names}
        
    for i in range(40):
        beh_m[i]['T'] = {_: beh_m[i]['T'][_][0].reshape(-1).item() if beh_m[i]['T'][_][0].size == 1 else ['T'][_][0].reshape(-1) for _ in beh_m[i]['T'].dtype.names}
    
    for i in range(40):
        beh_m[i]['back_id'] = beh_m[i]['back_id'] - 1
    
    # --- compare each segment
    for i in range(40):
        for key in behavior[i].keys():
            if isinstance(behavior[i][key], np.ndarray):
                if not np.all(behavior[i][key][~np.isnan(behavior[i][key])] == beh_m[i][key][~np.isnan(beh_m[i][key])]):
                    print(i, key, np.all(behavior[i][key][~np.isnan(behavior[i][key])] == beh_m[i][key][~np.isnan(beh_m[i][key])]))
            if isinstance(behavior[i][key], dict):
                if 'iT' in key:
                    for key_ in behavior[i][key].keys():
                        if not np.all( behavior[i][key][key_][~np.isnan(behavior[i][key][key_])] == beh_m[i][key][key_][~np.isnan(beh_m[i][key][key_])] ):
                            print(i, key, key_, np.all( behavior[i][key][key_][~np.isnan(behavior[i][key][key_])] == beh_m[i][key][key_][~np.isnan(beh_m[i][key][key_])] ))
                elif 'T' in key:
                    for key_ in behavior[i][key].keys():
                        if not behavior[i][key][key_]==beh_m[i][key][key_]:
                            print(i, key, key_, behavior[i][key][key_]==beh_m[i][key][key_])


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 13:35:41 2023

@author: Runnan Cao
@modified: acxyle

    refer to: https://osf.io/824s7/

"""

# --- python
import numpy as np

# --- local
import sys
sys.path.append('../')
from utils_ import utils_similarity


# ======================================================================================================================
class primate_feature_process():
    
    def __init__(self, seed=6, **kwargs):

        #np.random.seed(seed)     # FIXME
        ...

    def calculation_1st_stats(self, metric, FR, PSTH, **kwargs):
        
        _1st_stats = self._calculation_1st_stats(metric, FR, **kwargs)
        
        _1st_stats_temporal = self._calculation_1st_stats_temporal(metric, PSTH, **kwargs)
        
        return _1st_stats, _1st_stats_temporal
    
    
    def calculation_1st_stats_perm(self, _1st_stats, _1st_stats_temporal, num_perm=1000, seed=666, **kwargs):
        
        num_samples = _1st_stats.shape[0]
        num_steps = _1st_stats_temporal.shape[0]
        
        np.random.seed(seed)     # re-initialize to make the permutation constant
        perm_indces = [np.random.permutation(num_samples) for _ in range(num_perm)]
        
        _1st_stats_perm = np.array([_1st_stats[np.ix_(_, _)] for _ in perm_indces])     # (1000, 50, 50)
        _1st_stats_temporal_perm = np.array([np.array([_1st_stats_temporal[t][np.ix_(_, _)] for t in range(num_steps)]) for _ in perm_indces])     # (1000, 26, 50, 50)

        return _1st_stats_perm, _1st_stats_temporal_perm
    
    
    def _calculation_1st_stats(self, metric, feature, **kwargs):
        
        if metric == 'DSM':
            return _calculation_DSM(feature, **kwargs)
        elif metric == 'Gram':
            return _calculation_Gram(feature, **kwargs)
        else:
            raise ValueError
    
    
    def _calculation_1st_stats_temporal(self, metric, feature, **kwargs):
        
        assert feature.ndim == 3      # (time_steps, num_samples, num_features)

        return np.array([self._calculation_1st_stats(metric, feature[_, :, :], **kwargs) for _ in range(feature.shape[0])])
        

    
# ----------------------------------------------------------------------------------------------------------------------
def _calculation_DSM(feature, **kwargs):

    return utils_similarity.DSM_calculation(feature, **kwargs)


def _calculation_Gram(feature, kernel='linear', **kwargs):
    
    if kernel == 'linear':
        gram = utils_similarity.gram_linear
    elif kernel =='rbf':
        gram = utils_similarity.gram_rbf
    else:
        raise ValueError
    
    return gram(feature, **kwargs)
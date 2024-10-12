#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 23:31:53 2023

@author: acxyle

    the entire code needs to be simplfied and upgrade


"""

import scipy.stats as stats
import numpy as np

from matplotlib.lines import Line2D

from scipy.spatial.distance import pdist, squareform

from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind
import itertools
from scipy.spatial.distance import mahalanobis


# ----------------------------------------------------------------------------------------------------------------------
#FIXME ---
def _ccc(x, y):
    
    cor = np.corrcoef(x, y)[0][1]

    mean_true = np.mean(x)
    mean_pred = np.mean(y)
    
    var_true = np.var(x)
    var_pred = np.var(y)
    
    sd_true = np.std(x)
    sd_pred = np.std(y)
    
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    ccc = numerator / denominator
    
    return ccc


def DSM_calculation(feature, metric='pearson', vectorize=False, num_calsses=50, **kwargs):
    """
        'bicor': Biweight midcorrelation; 
        'percbend': Percentage bend correlation; 
        'shepherd': Shepherd's pi correlation; 
        'skipped': Skipped correlation
    """
    
    if 'euclidean' in metric.lower():
        
        if (mask:=_size_and_mask_check(feature)) is not None:

            feature = feature[:, mask]     # remove the cell/unit with nan values
            eu_distance = pdist(feature, 'euclidean')     
            
            if not vectorize:
                return squareform(eu_distance)
            else:
                return eu_distance
        else:
            #print('detacted abnormal value')
            #return np.zeros((num_calsses, num_calsses))
            raise ValueError

    elif 'pearson' in metric.lower():

        if (mask:=_size_and_mask_check(feature)) is not None:
            
            similarity = np.ma.corrcoef(np.ma.masked_invalid(feature)).data     # filter out Nan or Inf
                
            return RSM_process(similarity, vectorize, 'arctanh')  
        
        else:
            return np.zeros((num_calsses, num_calsses))
            #raise ValueError
   
    elif 'spearman':
        
        return RSM_process(spearmanr(feature, axis=1, nan_policy='omit').statistic, vectorize)
    
    elif 'mahalanobis':
        
        if (mask:=_size_and_mask_check(feature)) is not None:
            
            mahal_matrix = _mahalanobis(feature[:, mask])
            
            return RSM_process(mahal_matrix, vectorize, 'standardization')
        
        else:
            raise ValueError
        
    elif 'concordance':
        
        if (mask:=_size_and_mask_check(feature)) is not None:
            
            feature = feature[:, mask]
            
            num_sampels = feature.shape[0]
            
            product_list = list(itertools.product(np.arange(num_sampels),np.arange(num_sampels)))
            
            ccc_matrix = np.full((num_sampels, num_sampels), np.nan)
            
            for _ in product_list:
                ccc_matrix[_[0], _[1]] = _ccc(feature[_[0]], feature[_[1]])
            
            return RSM_process(ccc_matrix, vectorize, 'standardization')
        
        else:
            raise ValueError
            
    else:
        raise ValueError
    

def RSM_process(RSM:np.ndarray, vectorize:bool=False, post_process:bool=None, **kwargs):
    """
        ...
    """
    # --- [default] transfer the RSM to DSM by linear subtraction, spearmanr() is invariant to linear tranformation
    if post_process == None:
        DSM = 1 - RSM    # [-1, 1] -> [0, 2] if RSM is [-1, 1]
    
    # --- [lagacy] 
    elif post_process == 'arctanh': 
        RSM = np.arctanh(RSM-1e-8)    # input: [-1e-8, 1-1e-8]
        DSM = 1 - RSM     # (-inf, +inf)
        
    # --- **fisher_z**
    elif post_process == 'fisher_z': 
        RSM = RSM-1e-8
        RSM = np.log((1+RSM)/(1-RSM))/2
        DSM = np.max(RSM) - RSM     # [0, +inf)
    
    # --- [standardization / z-score transformation]
    elif post_process == 'standardization':
        RSM = (RSM-np.mean(RSM))/(np.std(RSM)+1e-8)
        DSM = np.max(RSM) - RSM
        
    elif post_process == 'normalization':
        RSM = (RSM-np.min(RSM))/(np.max(RSM)-np.min(RSM))
        DSM = np.max(RSM) - RSM
 
    elif post_process == 'square':
        RSM = RSM**2
        DSM = np.max(RSM) - RSM
        
    elif post_process == 'log':
        RSM = RSM + 1
        RSM = np.log(RSM)/np.log(np.e)
        DSM = np.max(RSM) - RSM
        
    elif post_process == 'yeo-johnson':
        RSM, used_lambda = stats.yeojohnson(RSM)
        DSM = np.max(RSM) - RSM
        
    # -----
    if vectorize:
        return RSM_vectorize(DSM)
    else:
        return DSM
    

def RSM_vectorize(RSM:np.ndarray, k=1, m=None):
    
    return RSM[np.triu_indices(RSM.shape[0], k, m)]


def _mahalanobis(input):
    
    (num_sampels, num_features) = input.shape
    
    product_list = list(itertools.product(np.arange(num_sampels),np.arange(num_sampels)))
    
    cm = np.cov(input, rowvar=False) + np.eye(num_features)*1e-8
    icm = np.linalg.inv(cm)
    
    mahal_matrix = np.full((num_sampels, num_sampels), np.nan)
    
    for _ in product_list:
        mahal_matrix[_[0], _[1]] = mahalanobis(input[_[0]], input[_[1]], icm)
    
    return mahal_matrix


def _size_and_mask_check(input):
    """
        ...
    """
    if input.size == 0:     # size check
        return None
    mask = ~np.any(np.isnan(input), axis=0)
    if mask.size == 0:     
        raise ValueError('detected all feature vector contains NaN')
        #return None
    return mask


# ----------------------------------------------------------------------------------------------------------------------
def gram_linear(x, **kwargs):

    return x.dot(x.T)


def gram_rbf(x, threshold=1.0, **kwargs):

    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def cka(gram_x, gram_y, debiased=True):
    
    if (gram_x.size == 1 and gram_x == 0) or (gram_y.size == 1 and gram_y == 0):     # if feature is empty
        return np.nan

    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())
    
    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    
    if normalization_x == 0 or normalization_y == 0:     # if feature is all zero
        return np.nan
    else:     
        cka_score = scaled_hsic / (normalization_x * normalization_y)
        
        if cka_score  > 0.:
            return np.min([1., cka_score])
        else:     # if score < 0 when unbiased == True
            #return np.nan
            return 0.


def center_gram(gram, unbiased=True):

    if not np.allclose(gram, gram.T, rtol=1e-06, atol=1e-05):
      raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()
    
    if unbiased:
      
      n = gram.shape[0]
      np.fill_diagonal(gram, 0)
      means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
      means -= np.sum(means) / (2 * (n - 1))
      gram -= means[:, None]
      gram -= means[None, :]
      np.fill_diagonal(gram, 0)
    else:
      means = np.mean(gram, 0, dtype=np.float64)
      means -= np.mean(means) / 2
      gram -= means[:, None]
      gram -= means[None, :]
    
    return gram


def cka_temporal(primate_Gram_temporal, NN_Gram, **kwargs):
    # input shape: Bio - (time_steps, corr_matrix), NN - (corr_matrix,)
    return np.array([cka(_, NN_Gram, **kwargs) for _ in primate_Gram_temporal])      # (time_steps, )


# ----------------------------------------------------------------------------------------------------------------------
def describe_numpy(input:np.array=None):
    """
        this function uses a dict format to print the information of a numpy array like Pandas describe
    """
    
    if input is None or input[~np.isnan(input)].size == 0:
        #utils_.formatted_print('received input is None or full of nan values, returned None')
        return None
    
    array_stats = {
        "count_valid": input[~np.isnan(input)].size,
        "count_all": input.size,
        "mean": np.mean(input[~np.isnan(input)]),
        "std": np.std(input[~np.isnan(input)]),
        "min": np.min(input[~np.isnan(input)]),
        "25%": np.percentile(input[~np.isnan(input)], 25),
        "50% (median)": np.median(input[~np.isnan(input)]),
        "75%": np.percentile(input[~np.isnan(input)], 75),
        "max": np.max(input[~np.isnan(input)])
    }
    
    return array_stats


def fake_legend_describe_numpy(ax, input, mask=None, **kwargs):
    
    similarity_interest = input[mask]
    
    similarity_stats_dict = describe_numpy(similarity_interest)
    
    if similarity_stats_dict is None:
        
        fake_legend_stats_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='orange', markersize=5, markeredgecolor='orange') for _ in range(8)]
        
        fake_legend_stats_labels = [
            f"count: {similarity_interest.size}/{input.size}",
            "mean: - ",
            "std: - ",
            "min: - ",
            "25%: - ",
            "50%: - ",
            "75%: - ",
            "max: - "
            ]

        fake_legend = ax.legend(fake_legend_stats_handles, fake_legend_stats_labels, framealpha=0.25, ncol=2, handlelength=0, borderpad=0, labelspacing=0)
        ax.add_artist(fake_legend)
        
    else:
    
        fake_legend_stats_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='orange', markersize=5, markeredgecolor='orange') for _ in range(8)]
        
        fake_legend_stats_labels = [
            f"count: {similarity_interest.size}/{input.size}",
            f"mean: {similarity_stats_dict['mean']:.3f}",
            f"std: {similarity_stats_dict['std']:.3f}",
            f"min: {similarity_stats_dict['min']:.3f}",
            f"25%: {similarity_stats_dict['25%']:.3f}",
            f"50%: {similarity_stats_dict['50% (median)']:.3f}",
            f"75%: {similarity_stats_dict['75%']:.3f}",
            f"max: {similarity_stats_dict['max']:.3f}"
            ]
        
        fake_legend = ax.legend(fake_legend_stats_handles, fake_legend_stats_labels, framealpha=0.25, ncol=2, handlelength=0, borderpad=0, labelspacing=0)
        ax.add_artist(fake_legend)
    
    return fake_legend


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:49:00 2024

@author: Grasshlw, ahwillia

    refer to: 
        https://github.com/Grasshlw/SNN-Neural-Similarity-Static
        https://github.com/ahwillia/netrep?tab=readme-ov-file

@modified: acxyle


"""
from tqdm import tqdm
import os
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

import sys
sys.path.append('../')
import utils_

from bio_records_process.monkey_feature_process import monkey_feature_process
from bio_records_process.human_feature_process import human_feature_process
from FSA_Encode import FSA_Encode

# ======================================================================================================================
class Similarity_metric_base(monkey_feature_process, human_feature_process, FSA_Encode):
    
    def __init__(self, primate='monkey', **kwargs):
        
        self.primate = primate
        
        self.FR_id, self.psth_id = self._obtain_primate_data(**kwargs)
        
        self.NN_feature_dict = self._obtain_NN_data(**kwargs)
        
        ...
    
    def _obtain_primate_data(self, used_cell_type='qualified', used_id_num=50, **kwargs):
        
        if self.primate == 'monkey':

            monkey_feature_process.__init__(self, **kwargs)
            
            FR_id, psth_id = self.calculation_feature()

        elif self.primate == 'human':

            human_feature_process.__init__(self, **kwargs)
            used_cells = self.calculation_Sort()[used_cell_type]
            used_ids = self.calculation_subIDs(used_id_num)
            
            FR_id, psth_id = self.calculation_FM()
            
            FR_id = FR_id[np.ix_(used_ids, used_cells)]
            psth_id = np.array([_[np.ix_(used_ids, used_cells)] for _ in psth_id])
        
        else:
            raise ValueError
            
        return FR_id, psth_id

    
    def _obtain_NN_data(self, used_cell_type='qualified', used_id_num=50, fr=True, **kwargs):
        
        FSA_Encode.__init__(self, **kwargs)

        self.used_unit_types = ['qualified', 'selective', 'strong_selective', 'weak_selective', 'non_selective']
        
        self.Sort_dict = self.load_Sort_dict()
        self.Sort_dict = self.calculation_Sort_dict(self.used_unit_types) if self.used_unit_types is not None else self.Sort_dict
        
        pl = Parallel(n_jobs=-1)(delayed(self._load_NN_feature)(layer, used_cell_type, used_id_num, fr) for layer in self.layers)
        
        #pl = []
        #for layer in self.layers:
        #    pl.append(self._load_NN_feature(layer, used_cell_type, used_id_num, fr))
            
        NN_feature_dict = {self.layers[_]: pl[_] for _ in range(len(self.layers))}
        
        return NN_feature_dict
    
    
    
    def _load_NN_feature(self, layer, used_cell_type='qualified', used_id_num=50, fr=True, **kwargs):
        
        feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), verbose=False, **kwargs)     # (500, num_samples)
        
        if self.primate == 'human':
            used_ids = self.calculation_subIDs(used_id_num)
        else:
            used_ids = np.arange(50)
        
        if used_cell_type != 'qualified':
        
            feature = feature[:, self.Sort_dict[layer][used_cell_type].astype(int)]
        
        if fr:
            feature = np.mean(feature.reshape(self.num_classes, self.num_samples, -1), axis=1)     # (50, num_samples)
            feature = feature[used_ids, :]
        else:
            raise ValueError
        
        return feature
    
    
    def _obtain_DR_model(self, **kwargs):
        
        if self.DR == "PCA":
            red_model = PCA(n_components=self.dims, random_state=self.seed)
            red_neural = PCA(n_components=self.dims, random_state=self.seed)
        elif self.DR == "TSVD":
            red_model = TruncatedSVD(n_components=self.dims, random_state=self.seed)
            red_neural = TruncatedSVD(n_components=self.dims, random_state=self.seed)
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction}")
        
        return red_model, red_neural
    
    
    def _calculation(self, **kwargs):
        
        ...
    
    def _calculation_layer(self, **kwargs):
        
        ...
    
        
    def _process(self, **kwargs):
        
        ...
        
        
# -----
class Similarity_CCA(Similarity_metric_base):
    
    def __init__(self, primate='monkey', used_unit_type='qualified', used_id_num=50, DR='TSVD', dims=40, seed=2020, **kwargs):
        
        super().__init__(primate=primate, used_cell_type=used_unit_type, used_id_num=used_id_num, **kwargs)
        
        np.random.seed(seed)
        self.seed = seed
        self.DR = DR
        self.dims = dims
        
        self.dest_CCA = os.path.join(self.dest, 'CCA')
        utils_.make_dir(self.dest_CCA)
        
        self.dest_primate = os.path.join(self.dest_CCA, primate)
        utils_.make_dir(self.dest_primate)
        
        if primate == 'human':
            self.dest_primate = os.path.join(self.dest_primate, used_unit_type, str(used_id_num))
            utils_.make_dir(self.dest_primate)
        
        self.calculation_CCA()
        
        
    def calculation_CCA(self, used_unit_type='qualified', used_id_num=50, **kwargs):
        
        if self.primate == 'human':
            save_path = os.path.join(self.dest_primate, f'CCA_results_{used_unit_type}_{used_id_num}.pkl')
        elif self.primate == 'monkey':
            save_path = os.path.join(self.dest_primate, 'CCA_results.pkl')
        
        if os.path.exists(save_path):
        
            results = utils_.load(save_path, verbose=False)
        
        else:
            CCA_results_dict = []
            #CCA_results_temporal_dict = []
            
            for layer in tqdm(self.layers):
                
                CCA_results_dict.append(self._calculation_CCA(self.FR_id, self.NN_feature_dict[layer]))
                
                #pl = Parallel(n_jobs=-1)(delayed(self._calculation_CCA)(_, self.NN_feature_dict[layer]) for _ in self.psth_id)
                #CCA_results_temporal_dict.append(np.array(pl))
            
            results = {
                'similarity': np.array(CCA_results_dict),
                #'similarity_temporal': np.array(CCA_results_temporal_dict)
                }
            
            utils_.dump(results, save_path, verbose=False)
            
        return results
        
    
    def _calculation_CCA(self, neural_data, model_data, **kwargs):
        """
            calculate CCA for each layer
        """
        red_model, red_neural = self._obtain_DR_model()
        
        if (self.DR is not None) and self.dims < model_data.shape[1]:
            red_model.fit(model_data)
            model_lowd = red_model.transform(model_data)
        else:
            model_lowd = model_data.copy()
            
        if (self.DR is not None) and self.dims < neural_data.shape[1]:
            red_neural.fit(neural_data)
            neural_lowd = red_neural.transform(neural_data)
        else:
            neural_lowd = neural_data.copy()
        
        model_lowd = model_lowd.transpose((1, 0))
        neural_lowd = neural_lowd.transpose((1, 0))

        s, _, _ = _cca(model_lowd, neural_lowd)
        
        return np.mean(s)
    
        ...
            
 
def _cca(x, y):
    def matrix_sqrt(m):
        w, v = np.linalg.eigh(m)
        w_sqrt = np.sqrt(np.abs(w))
        return np.dot(v, np.dot(np.diag(w_sqrt), np.conj(v).T))

    x_num = x.shape[0]
    y_num = y.shape[0]

    covariance = np.cov(x, y)
    cov_xx = covariance[:x_num, :x_num]
    cov_xy = covariance[:x_num, x_num:]
    cov_yx = covariance[x_num:, :x_num]
    cov_yy = covariance[x_num:, x_num:]

    x_max = np.max(np.abs(cov_xx))
    y_max = np.max(np.abs(cov_yy))
    cov_xx /= x_max
    cov_yy /= y_max
    cov_xy /= np.sqrt(x_max * y_max)
    cov_yx /= np.sqrt(x_max * y_max)

    cov_xx_inv = np.linalg.pinv(cov_xx)
    cov_yy_inv = np.linalg.pinv(cov_yy)

    cov_xx_sqrt_inv = matrix_sqrt(cov_xx_inv)
    cov_yy_sqrt_inv = matrix_sqrt(cov_yy_inv)

    M = np.dot(cov_xx_sqrt_inv, np.dot(cov_xy, cov_yy_sqrt_inv))

    u, s, v = np.linalg.svd(M)
    s = np.abs(s)

    x_ = np.dot(np.dot(u.T, cov_xx_sqrt_inv), x)
    y_ = np.dot(np.dot(v, cov_yy_sqrt_inv), y)

    return s, x_, y_
    
    
# -----
class Similarity_Reg(Similarity_metric_base):
    
    # interesting, DR will significantly influence the similarity
    # no DR is preferred for local dataset
    def __init__(self, primate='monkey', used_unit_type='qualified', used_id_num=50, DR=None, dims=40, seed=2020, **kwargs):
        
        super().__init__(primate=primate, used_cell_type=used_unit_type, used_id_num=used_id_num, **kwargs)
        
        np.random.seed(seed)
        self.seed = seed
        self.DR = DR
        if dims < used_id_num:
            self.dims = dims
        else:
            self.dims = used_id_num
        self.splits = -1
        
        self.dest_Reg = os.path.join(self.dest, 'Reg')
        utils_.make_dir(self.dest_Reg)
        
        self.dest_primate = os.path.join(self.dest_Reg, primate)
        utils_.make_dir(self.dest_primate)
        
        if primate == 'human':
            self.dest_primate = os.path.join(self.dest_primate, used_unit_type, str(used_id_num))
            utils_.make_dir(self.dest_primate)
        
        self.calculation_Reg(used_unit_type=used_unit_type, used_id_num=used_id_num, **kwargs)
    
        
    def calculation_Reg(self, used_unit_type='qualified', used_id_num=50, **kwargs):
        
        if self.primate == 'human':
            save_path = os.path.join(self.dest_primate, f'Reg_results_{used_unit_type}_{used_id_num}.pkl')
        elif self.primate == 'monkey':
            save_path = os.path.join(self.dest_primate, 'Reg_results.pkl')
        
        if os.path.exists(save_path):
        
            results = utils_.load(save_path, verbose=False)
        
        else:
            Reg_results_dict = []
            Reg_results_temporal_dict = []
            
            for layer in tqdm(self.layers):
                
                Reg_results_dict.append(self._calculation_Reg(self.FR_id, self.NN_feature_dict[layer]))
                
                pl = Parallel(n_jobs=-1)(delayed(self._calculation_Reg)(_, self.NN_feature_dict[layer]) for _ in self.psth_id)
                Reg_results_temporal_dict.append(np.array(pl))
            
            results = {
                'similarity': np.array(Reg_results_dict),
                'similarity_temporal': np.array(Reg_results_temporal_dict)
                }
            
            utils_.dump(results, save_path, verbose=False)
            
        return results
        
    
    def _calculation_Reg(self, neural_data, model_data, **kwargs):
        
        if (self.DR != None) and (self.dims < model_data.shape[1]):
            red_model, _ = self._obtain_DR_model()

            red_model.fit(model_data)
            model_lowd = red_model.transform(model_data)
        else:
            model_lowd = model_data.copy()
            
        num_classes = model_data.shape[0]
        neural_pred = np.zeros(neural_data.shape)
        if self.splits == -1:
            kf = KFold(n_splits=num_classes, shuffle=True, random_state=self.seed)
        else:
            kf = KFold(n_splits=self.splits, shuffle=True, random_state=self.seed)
        for train_index, test_index in kf.split(model_lowd):
            model_lowd_train = model_lowd[train_index]
            model_lowd_test = model_lowd[test_index]
            neural_train = neural_data[train_index]
            neural_test = neural_data[test_index]

            reg = Ridge(alpha=1.0)
            reg.fit(model_lowd_train, neural_train)
            neural_pred[test_index] = reg.predict(model_lowd_test)

        r = _pcc(neural_pred, neural_data)
        return np.mean(r)


def _pcc(x,y):
    x_mean = np.mean(x, axis=0, keepdims=True)
    y_mean = np.mean(y, axis=0, keepdims=True)
    
    x_center = x - x_mean
    y_center = y - y_mean
    
    r = np.sum(x_center * y_center, axis=0) / np.sqrt(np.sum(x_center * x_center, axis=0) * np.sum(y_center * y_center, axis=0))
    return r


# -----
# FIXME --- this is not GSM now but the linear function if Procrustes Distance
class Similarity_GSM(Similarity_metric_base):
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
    def calculation_GSM(self, **kwargs):
        
        import netrep
        from netrep.metrics import LinearMetric

        # Rotationally invariant metric (fully regularized).
        proc_metric = LinearMetric(alpha=1.0, center_columns=True)
        proc_metric.fit(NN, Bio)
        dist = proc_metric.score(NN, Bio)



# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    FSA_root = '/home/acxyle-workstation/Downloads/FSA'
    FSA_dir = 'VGG/SpikingVGG'
    FSA_config = 'SpikingVGG16bn_IF_ATan_T4_C2k_fold_'
    FSA_model = 'spiking_vgg16_bn'
    
    _, layers, neurons, _ = utils_.get_layers_and_units(FSA_model, 'act')
    
    for _ in [0]:
        
        root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}/-_Single Models/FSA {FSA_config}{_}')
        
        #root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
        
        test_analyzer = Similarity_Reg(root=root, layers=layers, neurons=neurons, primate='monkey', used_unit_type='qualified', used_id_num=50)
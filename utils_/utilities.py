#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:12:39 2023

@author: acxyle
"""

import os
#import random
import datetime
import subprocess
from tqdm import tqdm
import numpy as np
#import cv2

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch

#from spikingjelly import visualizing
from spikingjelly.activation_based import tensor_cache
from spikingjelly.activation_based.model.tv_ref_classify import utils
#from spikingjelly.activation_based import surrogate, neuron, functional, layer


# ----------------------------------------------------------------------------------------------------------------------
#TODO ----- need to find a solution to save small pieces and merge them into one single feature map
def spikes_to_frs(feature, select_ratio=0, **kwargs):
    """
        in this function, the feature is compressed by sipikingjelly module
        if zlib is used, the compressed file may greater than the original one when the data is diverse
        
        refer to: https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based_en/ann2snn.html
    """
    assert feature[0][0].dtype==torch.uint8, "the data must be compressed 'bool spikes'."

    target_T_idx = int(feature[0][2][1]*select_ratio)

    feature = [np.mean(tensor_cache.bool_spike_to_float(*_).type(torch.uint8).numpy()[-target_T_idx:, :], axis=0) for _ in tqdm(feature, desc='bool_spikes -> frs')]
    feature = np.array(feature).astype(np.float32)

    return feature


def bool_spikes_to_spikes(feature, data_type='ndarray'):
    assert feature[0][0].dtype==torch.uint8, "the data must be compressed 'bool spikes'."
    
    if data_type == 'tensor':
        feature = [tensor_cache.bool_spike_to_float(*_).type(torch.uint8) for _ in tqdm(feature, desc='bool_spikes - > spikes')]
        feature = torch.stack(feature)
        
    elif data_type == 'ndarray':
        
        feature = [tensor_cache.bool_spike_to_float(*_).type(torch.uint8).numpy() for _ in tqdm(feature, desc='bool_spikes - > spikes')]
        feature = np.array(feature)
    
    else:
        raise ValueError
        
    return feature


def _file_system_type(path):
    result = subprocess.check_output(['df', '-T', path], encoding='utf-8')
    lines = result.splitlines()
    return lines[1].split()[1]


def _is_binary(input):
    return np.all((input==0)|(input==1))


def formatted_print(content, message_type='[Codinfo]', symbol='-', padding=2, border_length=None):
    """
        ...
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row_data = [current_time, message_type, content]
    formatted_row = "|".join("{:<{}}".format(' '*padding + item, len(item)+padding*2) for item in row_data)
    
    border = symbol * (len(formatted_row) + 2)if border_length else symbol * 20

    print(formatted_row)


# -----
def make_dir(path):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

# -----
def cal_acc1_acc5(output, target):
    acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
    return acc1, acc5
    
# ----- SVM
# FIXME --- add k-fold entrance
def SVM_classification(matrix, label, test_size=0.2, random_state=42):
    """
        1. default kernel='rbf', non-linear. 
        2. train_test_split() splits train data and test data randomly, not like n_fold experiments
        
        "It should be noticed the performance of svm.SVC() can be highly sensitive 
        to the choice of kernel, used what kernel depends on the data and the 
        research problem, sounds like can use GridSearchCV from sklearn.model_selection
        to perform a grid search for best combination for a model, but perhaps time
        consuming and computation intensive."
    """
    
    matrix_train, matrix_test, label_train, label_test = train_test_split(matrix, label, test_size=test_size, random_state=random_state)

    clf = svm.SVC()     # .SVC() .LinearSVC() .NuSVC() ... 
    
    if matrix_train.shape[1] == 0:
      acc = 0.
    else:
      clf.fit(matrix_train, label_train)
      predicted = clf.predict(matrix_test)
      acc = accuracy_score(label_test, predicted)*100
      
    return acc


def makeLabels(num_samples, num_classes):  # generate a label list
    label = []
    for i in range(num_classes):
        label += [i + 1] * num_samples
    return label


def describe_model(layers, units, shapes, idx=None):
    
    if idx is None:
        layers_info = [list(pair) for pair in zip(list(np.arange(len(layers)+1)), layers, units, shapes)]
    else:
        layers_info = [list(pair) for pair in zip(idx, layers, units, shapes)]
    
    max_widths = [max(len(str(row[i])) for row in layers_info)+2 for i in range(len(layers_info[0]))]
    
    print("|".join("{:<{}}".format(header, max_widths[i]) for i, header in enumerate(["No.", "layer", "units", "shapes"])))
    print("-" * sum(max_widths))
    
    for row in layers_info:
        print("|".join("{:<{}}".format(str(item), max_widths[i]) for i, item in enumerate(row)))

        
    print("-" * sum(max_widths))



    
    
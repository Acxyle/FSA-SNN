#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 22:07:14 2024

@author: acxyle
"""


import pickle
import gzip
import joblib
import json

import os
import numpy as np

from tqdm import tqdm


__all__ = [
    'dump', 'load', 'load_feature',
    'restore_order', 'lexicographic_order'
    ]


# ----------------------------------------------------------------------------------------------------------------------
# FIXME --- test version, create a visible progress bar for loading and saving (with problems)
class tqdm_file_object:
    def __init__(self, file_path, mode='rb', verbose=True, estimated_size=None):
        self.file = open(file_path, mode)
        self.mode = mode
        self.verbose = verbose

        # init progress bar
        if 'r' in mode:
            self.length = os.fstat(self.file.fileno()).st_size
            if self.verbose:
                self.tqdm = tqdm(total=self.length, unit='B', unit_scale=True, desc=f'Loading {file_path}')
        elif 'w' in mode or 'a' in mode:
            if self.verbose:
                self.tqdm = tqdm(total=estimated_size, unit='B', unit_scale=True, desc=f'Saving {file_path}')


    def read(self, size=-1):
        if 'r' not in self.mode:
            raise NotImplementedError("read() not implemented on file opened in write mode")
        data = self.file.read(size)
        if self.verbose:
            self.tqdm.update(len(data))
        return data
    
    def readline(self, size=-1):
        data = self.file.readline(size)
        if self.verbose:
            self.tqdm.update(len(data))
        return data

    def write(self, data):
        if 'w' not in self.mode and 'a' not in self.mode:
            raise NotImplementedError("write() not implemented on file opened in read mode")
        
        if isinstance(data, pickle.PickleBuffer):     # PickleBuffer, for writing
            data = bytes(data)
        
        self.file.write(data)
        if self.verbose:
            self.tqdm.update(len(data))
            self.tqdm.refresh()    # Refresh the progress bar display
        os.fsync(self.file.fileno())    # Ensure data is written to disk

    def close(self):
        if self.verbose:
            self.tqdm.close()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


# ----------------------------------------------------------------------------------------------------------------------
def dump(file, file_path, cmd='wb', tool='pickle', protocol=-1, verbose=True, **kwargs):
    
    assert cmd in ['w', 'wb', 'w+', 'wb+']
    
    if os.path.exists(file_path):
        print(f'path {file_path} exists, OVERWRITING...')
    
    if tool == 'joblib':
        
        joblib.dump(file, file_path, compress=0, protocol=protocol)     # see joblib.dump() for more details
    
    elif tool == 'gzip':
        
        assert os.path.splitext(file_path)[-1] in ['.gz', '.tar.gz', '.xz', '.tar.xz']
        
        with gzip.open(file_path, cmd) as f:
            
            pickle.dump(file, f, protocol=protocol)
        
    elif tool == 'pickle':
        
        assert os.path.splitext(file_path)[-1] in ['.pkl', '.pickle']
        
        estimated_size = len(pickle.dumps(file, protocol=protocol))
        
        with tqdm_file_object(file_path, cmd, verbose, estimated_size=estimated_size) as f:
            
            pickle.dump(file, f, protocol=protocol)
          
    elif tool == 'json':
        
        assert os.path.splitext(file_path)[-1] == '.json'
        
        with open(file_path, cmd) as f:
            
            json.dump(file, f, indent=5)  

    else:

        raise ValueError(f"Invalid tool: {tool}. Choose from 'pickle', 'gzip', 'joblib', 'json'.")



# ----------------------------------------------------------------------------------------------------------------------
def load(file_path, cmd='rb', tool='pickle', verbose=True, **kwargs):
    
    assert cmd in ['r', 'rb', 'r+', 'rb+']
    
    if not os.path.exists(file_path):
        raise ValueError(f'[Coderror] can not find the file {file_path}')
        
    if tool == 'gzip':
        assert os.path.splitext(file_path)[-1] in ['.gz', '.tar.gz', '.xz', '.tar.xz']
        with gzip.open(file_path, cmd) as f:
            loaded_file = pickle.load(f)
        
    elif tool == 'pickle':
        assert os.path.splitext(file_path)[-1] in ['.pkl', '.pickle']
        with tqdm_file_object(file_path, cmd, verbose) as f:
            loaded_file = pickle.load(f)
        
    elif tool == 'joblib':
        assert os.path.splitext(file_path)[-1] in ['.pkl', '.pickle', '.joblib']
        loaded_file = joblib.load(file_path)
        
    elif tool == 'json':
        assert os.path.splitext(file_path)[-1] == '.json'
        with open(file_path, cmd) as f:
            loaded_file = json.load(f)

    else:
        raise ValueError(f"Invalid tool: {tool}. Choose from 'pickle', 'gzip', 'joblib', 'json'.")
    
    return loaded_file


# -----
def load_feature(file_path, normalize=True, sort=True, num_classes=50, num_samples=10, **kwargs):
    """
        ...
    """
    
    feature = load(file_path, **kwargs)
    
    if normalize:     # min-max normalize -> [0,1], not standardize -> N(0,1)
        
        feature = (feature-np.min(feature))/(np.max(feature)-np.min(feature))     # (500, num_features)
    
    if sort:
        
        feature = restore_order(feature, num_classes, num_samples)     # (50, num_units)
    
    return feature


# ---
def restore_order(input, num_classes=50, num_samples=10, **kwargs):
    """
        ...
    """
    
    if input.shape[0] == num_classes:
    
        return input[np.argsort(lexicographic_order(num_classes))]
    
    elif input.shape[0] == num_classes*num_samples:
        
        return input[np.argsort(lexicographic_order(num_classes, num_samples), kind='stable')]


# -----
def lexicographic_order(num_classes, num_samples=None):
    """
        ...
    """
    
    id_order = np.arange(1, num_classes+1).astype(str)
    id_order_idx = np.argsort(id_order)
    id_order_lexical = id_order[id_order_idx].astype(int)
    
    if num_samples is not None:
        return np.repeat(id_order_lexical, num_samples)
    else:
        return id_order_lexical
    
    
# -----

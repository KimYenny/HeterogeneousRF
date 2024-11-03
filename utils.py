# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 2022

@author: Ye-eun Kim
"""
import numpy as np

def check_input(*args):
    arg_ = []
    for arg in args:
        if not isinstance(arg, np.ndarray):
             arg = np.array(arg)
        arg_.append(arg)
    if len(arg_) == 1:
        return arg_[0]
    else:
        return arg_
    
def get_classes_dict(y_str):
    
    classes_ = np.unique(y_str)
    classes_.sort()
    
    classes_dict = {}
    for c_idx, c in enumerate(classes_):
        classes_dict[c] = c_idx
        
    return classes_dict

def y_str2num(y_str, classes_dict):
    """
    y_str : np.array
    """
    
    y_num = [classes_dict[y_val] for y_val in y_str]
    y_num = np.array(y_num, dtype = np.int64)
    
    return y_num

def y_num2str(y_num, classes_dict):
    """
    y_num : np.array
    """
    
    keys = list(classes_dict.keys())
    y_str = [keys[int(y_val)] for y_val in y_num]
    y_str = np.array(y_str)
    
    return y_str

def voting(x):
    lab, cnt = np.unique(x,return_counts=True)
    return lab[np.argmax(cnt)]

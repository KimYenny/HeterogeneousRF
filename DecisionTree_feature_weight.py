# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 2022

@author: Ye-eun Kim
"""

import numpy as np
from numpy.random import choice
from math import sqrt, inf
from numba import njit
from utils import check_input, get_classes_dict, y_str2num, y_num2str

class _Tree():
    def __init__(self,
                 X, y,
                 n_random_features,
                 max_depth = None,
                 weights = None,
                 classes_ = None):
        
        
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        
        self.classes_ = classes_
        
        self.value = []
        self.predict_value = []
        
        self.depth = []
        self.max_depth = max_depth
            
        self.node_count = 1
            
        self.feature = []
        self.threshold = []
        self.children_left = []
        self.children_right = []
            
        self.weights = np.ones(shape = self.p) / self.p if weights is None else weights
        
        self.n_random_features = n_random_features
        if n_random_features is None:
            if self.calsses_ is None:
                #regression
                self.n_random_features = int(self.p/3)
            else:
                #classification
                self.n_random_features = int(sqrt(self.p))
                
                
    def __check_split_criterion(self,y,depth):
        
         isNotPure = len(np.unique(y)) > 1
         
         isLessThanMaxDepth = True
         if self.max_depth is not None:
             isLessThanMaxDepth = depth < self.max_depth
        
         return isNotPure & isLessThanMaxDepth
     
        
    def apply(self):
        
        queue = [(self.X.copy(), self.y.copy(),0)]
        while len(queue) > 0:
            _X, __y, _depth = queue.pop(0)
            
            best_x = None
            gf = -inf
            
            # value and predict value
            if self.classes_ is not None:
                value = np.unique(np.append(self.classes_,__y),
                                  return_counts=True)
                tbl = value[1] - 1
                        
                self.value.append(tbl)
                self.predict_value.append(self.classes_[tbl.argmax()])
                
            else:
                self.value.append(__y)
                self.predict_value.append(np.mean(__y))
            
            
            if self.__check_split_criterion(__y,_depth):
                random_feature = choice(np.arange(self.p),
                                        size = self.n_random_features,
                                        replace=False,
                                        p=self.weights)
                __X = _X[:,random_feature]
                
                # fine best split
                if self.classes_ is not None:
                    best_x, best_split, gf = greedySearch_cls(np.array(__X, dtype = np.float64),
                                                              np.array(__y))
                else:
                    best_x, best_split, gf = greedySearch_reg(np.array(__X, dtype = np.float64),
                                                              np.array(__y, dtype = np.float64))
                    
                
                        
            # selcet final best_x
            if best_x is None:
                self.feature.append(-2)
                self.threshold.append(-2)
                self.children_left.append(-1)
                self.children_right.append(-1)
                self.depth.append(_depth)
                
            else:
                self.feature.append(random_feature[best_x])
                self.threshold.append(best_split)
                
                left_child_idx = max(self.children_right)+1 if len(self.children_right)>0 else 1
                self.children_left.append(left_child_idx)
                self.children_right.append(left_child_idx+1)
                self.depth.append(_depth)
                
                # split
                x = __X[:,best_x]
                left_idx = x < best_split
                
                queue.append((_X[left_idx,:], __y[left_idx], _depth+1))
                queue.append((_X[~left_idx,:], __y[~left_idx],_depth+1))
    
                # update attributes
                self.node_count += 2
                
            
    def predict(self, X):
        
        y_pred = []
        for x_idx in range(X.shape[0]):
            x = X[x_idx,:]
            y_pred.append(self.__predict(x))
            
        return np.array(y_pred)
        
    def __predict(self, x):
        
        node_idx = 0
        best_x = self.feature[node_idx]
        best_split = self.threshold[node_idx]
        
        while best_x >= 0:
            isLeft = x[best_x] < best_split
            node_idx = self.children_left[node_idx] if isLeft else self.children_right[node_idx]
            best_x = self.feature[node_idx]
            best_split = self.threshold[node_idx]
        
        return self.predict_value[node_idx]
            
    def predict_proba(self,X):
        y_predproba = []
        for x_idx in range(X.shape[0]):
            x = X[x_idx,:]
            y_predproba.append(self.__predict_proba(x))
            
        return y_predproba
    
    def __predict_proba(self, x):
        """
        X : np.array
        pred_y_proba : np.array
        """
        node_idx = 0
        best_x = self.feature[node_idx]
        best_split = self.threshold[node_idx]
        
        while best_x >= 0:
            isLeft = x[best_x] < best_split
            node_idx = self.children_left[node_idx] if isLeft else self.children_right[node_idx]
            best_x = self.feature[node_idx]
            best_split = self.threshold[node_idx]
        
        tbl = self.value[node_idx]
        return tbl / sum(tbl)
    
      
class __DecisionTree():
    def __init__(self, max_depth = None, n_random_features = None):
        self.max_depth = max_depth  
        self.n_random_features = n_random_features
    
    def get_depth(self):
            
        if self.tree_ is None:
            print("The tree is not trained")
        else:
            return max(self.tree_.depth)
            
    def predict(self, X):
        X = check_input(X)
            
        if self.tree_ is None:
            print("The tree is not trained")
        else:
            y_pred = self.tree_.predict(X)
            return y_pred
        
    def get_feature_depth(self):
        if self.tree_ is None:
            print("The tree is not trained")
        else:
            features = self.tree_.feature
            depths = self.tree_.depth
            
            max_node_depth = max(depths)
            depth_b = np.full(shape=(self.p,), fill_value = np.nan)
            for node_idx, feature in enumerate(features):
                if feature >= 0 and np.isnan(depth_b[feature]):
                    depth_b[feature] = depths[node_idx]
                
            return depth_b, max_node_depth
        
class DecisionTreeClassifier(__DecisionTree):
    
    def __init__(self, 
                 max_depth=None,
                 n_random_features=None):
        
        super().__init__()
        self.n_random_features = n_random_features
        self.max_depth = max_depth
        
    def set_classes(self, classes_dict, classes = None):
        self.classes_dict = classes_dict
        if classes is None:
            self.classes_ = np.array(list(self.classes_dict.values()))
            self.classes_.sort()
        else:
            self.classes_ = classes
    
    def fit(self, X, y, weights=None):
        
        if self.classes_dict is None:
            self.X, self.y_str = check_input(X,y)
            
            classes_dict = get_classes_dict(self.y_str)
            self.set_classes(classes_dict)
            self.y_num = y_str2num(self.y, self.classes_dict)
            
        else:
            self.X, self.y_num = check_input(X,y)
    
        self.n, self.p = self.X.shape
        
        if self.n_random_features is None:
            self.n_random_features = int(sqrt(self.X.shape[1]))
                
        self.tree_ = _Tree(X=self.X,
                           y=self.y_num,
                           n_random_features=self.n_random_features,
                           weights = weights,
                           max_depth=self.max_depth,
                           classes_=self.classes_)
        self.tree_.apply()
        
        return self
    
    def predict(self,X):
        predicted_values = super().predict(X)
        
        return y_num2str(predicted_values,self.classes_dict)
    
    
    def predict_proba(self, X):
        
        X = check_input(X)
            
        if self.tree_ is None:
            print("The tree is not trained")
        else:
            y_prop = self.tree_.predict_proba(X)
        return y_prop
        
        
class DecisionTreeRegressor(__DecisionTree):
    
    def __init__(self, max_depth = None, n_random_features = None):
        
        super().__init__()
        
        self.n_random_features = n_random_features
        self.max_depth = max_depth
        
    def fit(self, X, y, weights = None):
        self.X, self.y = check_input(X,y)
        
        self.n, self.p = X.shape
       
        if self.n_random_features is None:
            self.n_random_features = int(max(X.shape[1]//3,1))
                
        self.tree_ = _Tree(X = self.X,
                           y = self.y,
                           n_random_features = self.n_random_features,
                           weights = weights,
                           max_depth=self.max_depth)
        self.tree_.apply()
        
        return self


@njit
def greedySearch_cls(X, y):
    """
    X, y : np.array()
    criterion : the function
    """
    
    def measurement(vector):
        p2 = []
        n = vector.size
        
        for v in np.unique(vector):
            p = (vector == v).sum() / n 
            p2.append(p**2)
        
        p2 = np.array(p2, dtype = np.float64)
        
        return 1 - p2.sum()
    
    root_imp = measurement(y)
    
    best_goodness = -inf
    best_x = None
    best_split = None
        
    # search for best split
    for col_idx in range(X.shape[1]):
        
        x = X[:,col_idx]
        x_unique = np.unique(x)
        x_unique.sort()
                
        splits = []
        for x_idx in range(1,len(x_unique)):
            splits.append((x_unique[x_idx-1] + x_unique[x_idx]) / 2)
        
        for s in splits:
            left = y[x < s]
            right = y[x >= s]
            
            left_imp = measurement(left)
            right_imp = measurement(right)
            
            weighted_imp = (len(left)*left_imp + len(right)*right_imp) / len(y)
            
            goodness = root_imp - weighted_imp
            
            if best_goodness < goodness:
                
                best_goodness = goodness
                best_x = col_idx
                best_split = s
        
    return best_x, best_split, best_goodness

@njit
def greedySearch_reg(X, y):
    """
    X, y : np.array(float64)
    criterion : the function
    """
    
    def measurement(vector):
        diff = vector - np.mean(vector)
        return np.sqrt(np.mean(diff**2))
    
    root_rmse = measurement(y)
    
    best_goodness = -inf
    best_x = None
    best_split = None
        
    # search for best split
    for col_idx in range(X.shape[1]):
        
        x = X[:,col_idx]
        x_unique = np.unique(x)
        x_unique.sort()
                
        splits = []
        for x_idx in range(1,len(x_unique)):
            splits.append((x_unique[x_idx-1] + x_unique[x_idx]) / 2)
        
        for s in splits:
            left = y[x < s]
            right = y[x >= s]
            
            left_rmse = measurement(left)
            right_rmse = measurement(right)
            
            weighted_rmse = (len(left)*left_rmse + len(right)*right_rmse) / len(y)
            
            goodness = root_rmse - weighted_rmse
            
            if best_goodness < goodness:
                
                best_goodness = goodness
                best_x = col_idx
                best_split = s

    return best_x, best_split, best_goodness
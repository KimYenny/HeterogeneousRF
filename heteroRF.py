# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 2022

@author: Ye-eun Kim
"""

from abc import ABC, abstractmethod
from DecisionTree_feature_weight import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
from scipy import stats
import warnings
from utils import *
from tqdm import trange, tqdm

warnings.filterwarnings('ignore')

class __HeterogeneousRandomForest():
        
    def __init__(self, base_estimator,
                 n_estimators,
                 n_random_features,
                 alpha, beta,
                 max_depth,
                 calcul_variable_importance,
                 verbose):
        
        self.base_estimator = base_estimator
        
        self.n_estimators = n_estimators
        
        self.max_depth = max_depth
        self.n_random_features = n_random_features
        
        self.alpha = alpha
        self.beta = beta
        
        self.calcul_variable_importance = calcul_variable_importance
        
        self.verbose = verbose
        
    @abstractmethod        
    def performance(self, y, y_hat):
        pass 
    
    def fit(self, X, y,
            classes_dict=None,
            classes=None):
        
        self.X, self.y  = X, y
        self.n, self.p = self.X.shape
        
        self.estimators_ = []
        self._oob_index = []
        
        self.feature_importances_ = None
        
        # in regression, they are None
        self.classes_dict = classes_dict
        self.classes = classes
            
        # initialize feature_depth
        self.feature_depth = np.full(shape=(self.n_estimators, self.p),
                                     fill_value=-1)
        self.cumulative_feature_depth = np.full(shape=(self.n_estimators, self.p),
                                                fill_value=-1.0)
        # Training
        
        for b in (trange(self.n_estimators, desc='Training', leave=True)
                  if self.verbose 
                  else range(self.n_estimators)):
            
            # define base learner
            _model_b = self.base_estimator()
            if isinstance(_model_b, DecisionTreeClassifier):
                _model_b.set_classes(classes_dict, classes)
            
            # get bootstrap sample
            bts_idx = np.random.choice(self.n, size = self.n, replace = True)
            _X = self.X[bts_idx,:]
            _y = self.y[bts_idx]
            
            # out of bag sample
            self._oob_index.append(list(set(np.arange(self.n)) - set(bts_idx)))
            
            # random feature
            if self.alpha == 0.0 or self.beta == 0:
                weights = np.ones(self.p)/self.p    
            else:
                weights = self._get_feature_weights(b)
            
            # fit base learner : nodewise random subspace DT
            _model_b.fit(_X, _y, weights)
            
            # save the base learner
            self.estimators_.append(_model_b)
            
            # update feature depth
            if not(self.alpha == 0.0 or self.beta == 0):
                self._update_feature_depth(_model_b, b)
                self._update_cumulative_feature_depth(b)
                
        # calculate variable importance
        if self.calcul_variable_importance:
            self.__variable_importance()
        
        return self
            
    def _get_feature_weights(self, b):
        
        if b == 0:
            # equal weights
            weights = 1/self.p * np.ones(shape = self.p)
            
        else:
            weights = self.cumulative_feature_depth[b-1,:]
            weights /= np.sum(weights)
            
        return weights
    
    def _update_feature_depth(self, mdl, b):
        
        depth_b, max_node_depth = mdl.get_feature_depth()
        
        if np.mean(np.isnan(depth_b)) == 1:
            max_node_depth = 0
        
        depth_b[np.isnan(depth_b)] = max_node_depth + self.beta
        
        self.feature_depth[b,:] = depth_b
    
    def _update_cumulative_feature_depth(self, b):
        # b= 0,1,2,...
        
        if b > 0:
            # cumulative term
            alpha = self.alpha**np.arange(b,-1,step=-1)
            alpha = alpha.reshape((1,len(alpha)))
            fd = self.feature_depth[:(b+1),:]
            cum = np.matmul(alpha, fd)
            
            self.cumulative_feature_depth[b,:] = cum[0]
            
        else: # b == 0
            self.cumulative_feature_depth[b,:] = self.feature_depth[b,:]
            
    def __variable_importance(self):
        vi = []
        
        for _model, __oob_index in (tqdm(zip(self.estimators_,self._oob_index),
                                         desc='Feature Importance', 
                                         leave=True) 
                                    if self.verbose 
                                    else zip(self.estimators_,self._oob_index)):
        
            _oob_X = self.X[__oob_index,:]
            _oob_y = self.y[__oob_index]
            
            if self.classes_dict is not None:
                _oob_y = y_num2str(_oob_y, self.classes_dict)
                
            _e_b = self.performance(_oob_y, _model.predict(_oob_X))
            
            _d_b = []
            for col_idx in range(self.p):
                _oob_X_p = _oob_X.copy()
                _oob_X_p[:,col_idx] = np.random.choice(_oob_X_p[:,col_idx],
                                                     size=_oob_X_p.shape[0],
                                                     replace=False)
                _p = self.performance(_oob_y, _model.predict(_oob_X_p))
                
                if self.classes_dict is None:
                    _d_b.append(_p - _e_b)
                else:
                    _d_b.append(_e_b - _p)
            
            vi.append(_d_b)
            
            
        vi = np.array(vi)
        vi = vi.mean(axis=0) / vi.std(axis=0)
        self.feature_importances_ = vi
    
    def get_feature_importance(self):
        if self.feature_importances_ is None:
            self.__variable_importance()
        else:
            return self.feature_importances_
        
class HeterogeneousRandomForestClassifier(__HeterogeneousRandomForest):
    
    def __init__(self, base_estimator=DecisionTreeClassifier,
                 n_estimators=100, 
                 n_random_features=None, max_depth=None,
                 alpha=0.5, beta=1,
                 calcul_variable_importance=False,
                 verbose=True):
        
        super().__init__(n_estimators=n_estimators,
                         n_random_features=n_random_features, 
                         base_estimator=base_estimator,
                         max_depth=None,
                         alpha=alpha, beta=beta,
                         calcul_variable_importance=calcul_variable_importance,
                         verbose=verbose)
    def performance(self, y, y_hat):
        return np.mean(y==y_hat)
    
    def fit(self, X, y):
        self.X, self.y_str = check_input(X,y)
        
        self.classes_dict = get_classes_dict(self.y_str)
        self.y_num = y_str2num(self.y_str, self.classes_dict)
        self.classes = np.array(list(self.classes_dict.values()))
        
        return super().fit(self.X, self.y_num,
                           self.classes_dict, self.classes)
    
        
    def predict(self, X):
        
        X = check_input(X)
            
        predicted_values = []
        
        for mdl in self.estimators_:
            predicted_values.append(mdl.predict(X))
        
        predicted_values = np.array(predicted_values) # shape = (n_estimator, n_data)
        
        return np.apply_along_axis(voting, 0, predicted_values)
    
    def predict_proba(self, X):
        
        X = check_input(X)

        predicted_probs = np.array(self.estimators_[0].predict_proba(X))
        
        for mdl in self.estimators_[1:]:
            predicted_probs += np.array(mdl.predict_proba(X))
        
        predicted_probs /= self.n_estimators
        
        return predicted_probs
    
        
class HeterogeneousRandomForestRegressor(__HeterogeneousRandomForest):
    
    def __init__(self, 
                 n_estimators = 100, base_estimator = DecisionTreeRegressor,
                 n_random_features = None, max_depth = None,
                 alpha = 0.5, beta = 1,
                 calcul_variable_importance = False,
                 verbose=True):
        
        super().__init__(n_estimators=n_estimators,
                         n_random_features=n_random_features, 
                         base_estimator=base_estimator,
                         max_depth=None,
                         alpha=alpha, beta=beta,
                         calcul_variable_importance=calcul_variable_importance,
                         verbose=verbose)
        
    def performance(self, y, y_hat):
        return np.mean((y-y_hat)**2)
        
        
    def fit(self, X, y):
        self.X, self.y = check_input(X,y)
        
        return super().fit(self.X, self.y)
        
    def predict(self, X):     
        
        X = check_input(X)

        predicted_values = []
        
        for mdl in self.estimators_:
            predicted_values.append(mdl.predict(X))
        
        predicted_values = np.array(predicted_values) # shape = (n_estimator, n_data)
        predicted_values = np.mean(predicted_values,axis=0)
        
        return predicted_values
    
        
# Heterogeneous Random Forests
Official code for Hegerogeneous Random Forest
  
## Classification Example
  ```Python
  from heteroRF import HeterogeneousRandomForestClassifier
  from sklearn.datasets import load_breast_cancer
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score, roc_auc_score
  
  ### load sample data
  datasets = load_breast_cancer()
  
  X = datasets['data']
  y = datasets['target_names'][datasets['target']]
  n,_ = X.shape
  X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=round(0.3*n))
  
  ### Training
  HRF = HeterogeneousRandomForestClassifier(calcul_variable_importance=True)
  HRF.fit(X_train,y_train)
  
  ### Evaluation 
  y_pred = HRF.predict(X_test)
  print(accuracy_score(y_test, y_pred))
  
  y_predprob = HRF.predict_proba(X_test)
  print(HRF.classes_dict)
  print(roc_auc_score(y_test == 'malignant',
                      y_score=y_predprob[:,1]))
  
  print(HRF.feature_importances_)
  ```
  
## Regression Example
  ```Python
  from heteroRF import HeterogeneousRandomForestRegressor
  from sklearn.datasets import load_diabetes
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import mean_squared_error, r2_score
  
  ### load sample data
  datasets = load_diabetes()
  
  X = datasets['data']
  y = datasets['target']
  n,_ = X.shape
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = round(0.3*n))
  
  ### Training
  HRF = HeterogeneousRandomForestRegressor(calcul_variable_importance=True)
  HRF.fit(X_train,y_train)
  
  ### Evaluation
  y_pred = HRF.predict(X_test)
  print(mean_squared_error(y_test, y_pred))
  print(r2_score(y_test, y_pred))
  
  print(HRF.feature_importances_)
```

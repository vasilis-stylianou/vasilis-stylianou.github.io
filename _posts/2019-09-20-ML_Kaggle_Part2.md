---
title: "Machine Learning (Kaggle): Part II"
date: 2019-09-20
tags: [machine learning, sklearn, keras, XGBoost, Neural Networks]
excerpt: sklearn, keras, XGBoost, Neural Networks
---



```python
import numpy as np
import pandas as pd
from fraud_modeller import *
from sklearn.metrics import roc_auc_score

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
```

# Load data


```python
from pickleObjects import *
```


```python
path = './Data/'

X_train, X_test, y_train, y_test = loadObjects(path+'X_train'),loadObjects(path+'X_test'),loadObjects(path+'y_train'),loadObjects(path+'y_test')

```

    Object loaded!
    Object loaded!
    Object loaded!
    Object loaded!



```python
np.bincount(y_train.astype(int))
```




    array([455902,  16530])




```python
np.bincount(y_test.astype(int))
```




    array([113975,   4133])



# Scale Data


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
```


```python
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
```

# A. Machine Learning Models

## 1. Naive Bayes


```python
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
```


```python
clf = GaussianNB()
clf.fit(X_train_scaled, y_train.astype(int))
y_pred=clf.predict_proba(X_test_scaled)[:, 1]
clf_roc=roc_auc_score(y_test.astype(int),y_pred)
clf_roc
```

## 2. Logistic Regression


```python
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs')


LR = Pipeline([('LR', clf)])

penalties = ['l1', 'l2']
Cs = np.logspace(-4, 1, 20)
solvers = ['liblinear']

parameters = [{'LR__penalty':penalty, 'LR__C': c, 'LR__solver':solver}
              for penalty in penalties
              for c in Cs
              for solver in solvers]

print('Number of parameters: ' , len(parameters))
```


```python
model = ModelTuner(pipeline=LR,parameters=parameters[:10], X_train=X_train_scaled,
                    y_train=y_train.astype(int), X_valid=X_test_scaled,
                    y_valid=y_test.astype(int), eval_metric=roc_auc_score)
```


```python
model.best_performance
```

## 3. SVM


```python
from sklearn import svm

clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
            max_iter=-1, probability=True, random_state=0, shrinking=True,
            tol=0.001, verbose=False)


SVM = Pipeline([('SVM', clf)])

Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
kernels = ['rbf','poly','linear']


parameters = [{'SVM__gamma':gamma, 'SVM__C': c, 'SVM__kernel':kernel}
              for gamma in gammas
              for c in Cs
              for kernel in kernels]

print('Number of parameters: ' , len(parameters))
```


```python
model = ModelTuner(pipeline=SVM,parameters=parameters[:5], X_train=X_train_scaled,
                    y_train=y_train.astype(int), X_valid=X_test_scaled,
                    y_valid=y_test.astype(int), eval_metric=roc_auc_score)
```


```python
model.best_performance
```

## 4. Random forests


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)


RF = Pipeline([('RF', clf)])

bootstraps = [True, False]
max_depths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
max_features = ['auto', 'sqrt']
min_samples_leafs = [1, 2, 4]
min_samples_splits = [2, 5, 10]
n_estimators = np.arange(10,500,50)
criteria = ['gini','entropy']


parameters = [{'RF__bootstrap': bootstrap,
               'RF__max_depth': depth,
               'RF__max_features': feat,
               'RF__min_samples_leaf': leaf,
               'RF__min_samples_split': split,
               'RF__n_estimators': estimators,
               'RF__criterion': criterion}

              for bootstrap in bootstraps
              for depth in max_depths
              for feat in max_features
              for leaf in min_samples_leafs
              for split in min_samples_splits
              for estimators in n_estimators
              for criterion in criteria]

print('Number of parameters: ' , len(parameters))
```


```python
#RANDOM SAMPLING
random_inds = np.random.RandomState(20).randint(0,len(parameters),size=100)

parameters = np.array(parameters)
random_params = parameters[random_inds]

print('Number of random parameters: ' , len(random_params))
```


```python
model = ModelTuner(pipeline=RF,parameters=random_params, X_train=X_train_scaled,
                    y_train=y_train.astype(int), X_valid=X_test_scaled,
                    y_valid=y_test.astype(int), eval_metric=roc_auc_score)
```


```python
model.best_performance
```

## 5. XGBOOST


```python
from xgboost import XGBClassifier
```


```python
clf = XGBClassifier(
             learning_rate =0.1,
             n_estimators=1000,
             max_depth=5,
             min_child_weight=1,
             gamma=0,
             subsample=0.8,
             colsample_bytree=0.8,
             objective= 'binary:logistic',
             nthread=10,
             scale_pos_weight=1,
             seed=27)


XGB = Pipeline([('XGB', clf)])


learning_rates = [0.05, 0.10, 0.15]
max_depths = [6, 8, 10, 12, 15]
min_child_weights = [ 1, 3, 5, 7 ]
gammas = [ 0.1, 0.2 , 0.3, 0.4 ]
colsample_bytrees = [ 0.3, 0.4, 0.5 , 0.7 ]
n_estimators = np.arange(100,300,50)



parameters = [{  'XGB__learning_rate'    : learning_rate,
                 'XGB__max_depth'        : depth,
                 'XGB__min_child_weight' : min_child_weight,
                 'XGB__gamma'            : gamma,
                 'XGB__colsample_bytree' : colsample_bytree,
                 'XGB__n_estimators' : estimators}

              for learning_rate in learning_rates
              for depth in max_depths
              for min_child_weight in min_child_weights
              for gamma in gammas
              for colsample_bytree in colsample_bytrees
              for estimators in n_estimators]

print('Number of parameters: ' , len(parameters))
```

    Number of parameters:  3840



```python
#RANDOM SAMPLING
random_inds = np.unique(np.random.RandomState(20)\
                .randint(0,len(parameters),size=20))

parameters = np.array(parameters)
random_params = parameters[random_inds]

print('Number of random parameters: ' , len(random_params))
```

    Number of random parameters:  20



```python
#indices = np.random.RandomState(30).randint(0,X_train_scaled.shape[0],size=10000)
```


```python
model = ModelTuner(pipeline=XGB,parameters=random_params,
                   X_train=X_train_scaled,
                    y_train=y_train.astype(int),
                    X_valid=X_test_scaled,
                    y_valid=y_test.astype(int), eval_metric=roc_auc_score)
```

    100%|██████████| 20/20 [44:10<00:00, 161.56s/it]



```python
model.best_performance
```




    0.9573498270146494




```python
model.best_model
```




    Pipeline(memory=None,
         steps=[('XGB', XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=0.5, gamma=0.1,
           learning_rate=0.1, max_delta_step=0, max_depth=15,
           min_child_weight=5, missing=None, n_estimators=150, n_jobs=1,
           nthread=10, objective='binary:logistic', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=27, silent=None,
           subsample=0.8, verbosity=1))])




```python
X = np.concatenate((X_train_scaled,X_test_scaled),axis=0)
y = np.concatenate((y_train.astype(int),y_test.astype(int)),axis=0)
```


```python
model.best_model.set_params(**model.best_params).fit(X,y)
```




    Pipeline(memory=None,
         steps=[('XGB', XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=0.5, gamma=0.1,
           learning_rate=0.1, max_delta_step=0, max_depth=15,
           min_child_weight=5, missing=None, n_estimators=150, n_jobs=1,
           nthread=10, objective='binary:logistic', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=27, silent=None,
           subsample=0.8, verbosity=1))])




```python
X_test_sub = loadObjects(path+'X_test_comp')
```

    Object loaded!



```python
X_test_sub_scaled = scaler.transform(X_test_sub)
```


```python
y_preds = model.best_model.predict_proba(X_test_sub_scaled)[:,1]
```

# Submission file


```python
df_test_sub = pd.read_csv('./Data/test_transaction.csv',
                       usecols = ['TransactionID'])
```


```python
df_test_sub['isFraud'] = y_preds
```


```python
df_test_sub.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionID</th>
      <th>isFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3663549</td>
      <td>0.003449</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3663550</td>
      <td>0.005749</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3663551</td>
      <td>0.011596</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test_sub.shape
```




    (506691, 2)




```python
df_test_sub.to_csv('./Data/codenames_sub_3.csv',index=False)
```


# B. Deep Neural Networks


```python
from sklearn.metrics import roc_auc_score
```


```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
```

    Using TensorFlow backend.



```python
X_train_scaled.shape[1]
```




    306




```python
model = Sequential()

model.add(Dense(units=150, activation='relu', input_dim=X_train_scaled.shape[1]))
model.add(Dropout(0.1))
model.add(Dense(units=74, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=36, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(units=2, activation='softmax'))
```


```python
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=0.001))
```


```python
# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model.h5',
                             monitor='val_loss', save_best_only=True)]

# Train neural network
history = model.fit(X_train_scaled, # Features
                      y_train, # Target vector
                      epochs=100, # Number of epochs
                      callbacks=callbacks, # Early stopping
                      verbose=1, # Print description after each epoch
                      batch_size=128, # Number of observations per batch
                      validation_data=(X_test_scaled, y_test)) # Data for evaluation
```

    Train on 472432 samples, validate on 118108 samples
    Epoch 1/100
    472432/472432 [==============================] - 15s 31us/step - loss: 0.1284 - val_loss: 0.1167
    Epoch 2/100
    472432/472432 [==============================] - 15s 31us/step - loss: 0.1175 - val_loss: 0.1141
    Epoch 3/100
    472432/472432 [==============================] - 15s 32us/step - loss: 0.1131 - val_loss: 0.1098
    Epoch 4/100
    472432/472432 [==============================] - 15s 31us/step - loss: 0.1106 - val_loss: 0.1075
    Epoch 5/100
    472432/472432 [==============================] - 14s 30us/step - loss: 0.1082 - val_loss: 0.1063
    Epoch 6/100
    472432/472432 [==============================] - 15s 31us/step - loss: 0.1065 - val_loss: 0.1049
    Epoch 7/100
    472432/472432 [==============================] - 14s 30us/step - loss: 0.1050 - val_loss: 0.1027
    Epoch 8/100
    472432/472432 [==============================] - 15s 32us/step - loss: 0.1040 - val_loss: 0.1021
    Epoch 9/100
    472432/472432 [==============================] - 14s 30us/step - loss: 0.1032 - val_loss: 0.1016
    Epoch 10/100
    472432/472432 [==============================] - 15s 33us/step - loss: 0.1020 - val_loss: 0.1007
    Epoch 11/100
    472432/472432 [==============================] - 14s 30us/step - loss: 0.1014 - val_loss: 0.1017
    Epoch 12/100
    472432/472432 [==============================] - 19s 41us/step - loss: 0.1008 - val_loss: 0.1012



```python
classes = history.model.predict(X_test_scaled, batch_size=128)
```


```python
roc_auc_score(y_test,classes)
```




    0.9371390379340748

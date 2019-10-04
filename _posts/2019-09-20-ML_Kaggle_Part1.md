---
title: "Machine Learning (Kaggle): Part I"
date: 2019-09-20
tags: [machine learning, data science, pandas, numpy]
excerpt: Machine Learning, Feature Engineering, pandas, numpy
---


# Libraries


```python
import numpy as np
import pandas as pd
from fraud_pre_proc import *
from fraud_feat_engineering import *


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")

```

# Tuning knobs

### 1. Data Cleaning
A. Number of low NaN-rate features: nans_rate_cut_off (parameter) <br>
B.
- Numerical_categorical split: min_categories (parameter)
- Methods to fill NaNs: Vasilis or Papes  <br>

### 2. Feature Engineering
A. Which Datetime Feats to add: select manually <br>
B1. Which Card-Address Interaction Feats to add: : select manually <br>
B2. Which Card-Address-Datetime Interaction Feats to add:<br>
$\quad$ i) period_feats (list)<br>
$\quad$ ii) card_addr_feats (list)<br>
C. Which Aggregated TransAmt Feats to add: select manually <br>
D. Which Frequency Feats to add: select manually <br>

### 3. Preprocessing and Feature Selection
Numerical_categorical split: min_categories (parameter)<br>
A. Number of highly correlated features: corr_cut_off (parameter) <br>
B. Method of treating categorical feats: how = {'dummies','label_enc'} <br>

### 4. Stratified Split
- Stratified split parameters: frac, n_splits<br>
- (PCA)

# 0. Import data


```python
#trans data
df_train_trans = import_data('./Data/train_transaction.csv',nrows=20000)
df_train_trans.head(3);
df_train_trans.shape
```

    Memory usage of dataframe is 60.12 MB --> 16.77 MB (Decreased by 72.1%)





    (20000, 394)




```python
#id data
df_train_id = import_data('./Data/train_identity.csv',nrows=1000)
df_train_id.head(3);
df_train_id.shape
```

    Memory usage of dataframe is 0.31 MB --> 0.18 MB (Decreased by 42.7%)





    (1000, 41)



## Join transactions and ids

All TransactionID's from df_train_id are present in df_train_trans<br>
BUT 76% of all TransactionID's of df_train_trans are missing from df_train_id<br>
$\implies$ perform a LEFT merge of df_train_id on df_train_trans


```python
# df_all=pd.merge(df_train_trans,df_train_id,on='TransactionID',how='left')
# df_all.shape
```


```python
df_all=df_train_trans
```


```python
all_vars = TableDescriptor(df_all,'All_data','isFraud')
```

# 1. Data Cleaning

### A. Filter features by NaNs_rate


```python
#Select features with low NaN rate
low_nan_vars = getCompletedVars(all_vars,nans_rate_cut_off=0.1)
```

    Selected features: 114/394



```python
cols_to_drop = [var.name for var in all_vars.variables if var not in low_nan_vars]
df_all = df_all.drop(cols_to_drop,axis=1)
```

### B. Fill NaNs

Numerical features: fill NaNs with mean value <br>
Categorical features: fill NaNs with most frequent value


```python
#split data to numerical/categorical
numerical_vars,categorical_vars= numerical_categorical_split(low_nan_vars,min_categories=30)
```

    No of numerical features: 56
    No of categorical features: 58



```python
fill_nans(df_all,numerical_vars,feat_type='numerical')
fill_nans(df_all,categorical_vars,feat_type='categorical')
```




    "NaNs have been filled in with column's most frequent value."




```python
print_null_cols(df_all)
```




    'No null columns.'



# 2. Feature Engineering

### A. Datetime Features


```python
period_feats=addDatetimeFeats(df_all)
```

    Datetime-like features added to dataframe:

    ['month', 'week', 'yearday', 'hour', 'weekday', 'day']



```python
# TransactionDT col is redundant
df_all = df_all.drop('TransactionDT',axis=1)
```

### B. Interaction Features


```python
#B.1 Add Interaction Features by ADDING the values of card_ and addr_ columns
card_addr_interactions = addCardAddressInteractionFeats(df_all)
```

    Interaction features added to dataframe:

    ['card12', 'card1235', 'card1235_addr12']



```python
#card_addr_feats = card_addr_interactions + ['card1','card2','card3','card5']
card_addr_feats = ['card12']
```


```python
#B.2 Add interaction features by ADDING the values of card_addr_feats and period_feats
#   and computing value frequencies
addDatetimeInteractionFeats(df_all, cols=card_addr_feats, period_cols=period_feats);
```

    Interaction features added to dataframe: 6

    ['card12_month', 'card12_week', 'card12_yearday', 'card12_hour', 'card12_weekday', 'card12_day']


### C. Aggregated Features


```python
# Add aggregated features by grouping-by card_addr_feats and computing the mean & STD of 'TransactionAmt'
addAggTransAmtFeats(df_all,cols=card_addr_feats);
```

    Aggregated TransactionAmt features added to dataframe: 2

    ['TransAmt_card12_mean', 'TransAmt_card12_std']


### D. Indicator/Frequency Features


```python
try_cols = card_addr_feats + \
        ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
        'D1','D2','D3','D4','D5','D6','D7','D8',
        'addr1','addr2',
        'dist1','dist2',
        'P_emaildomain', 'R_emaildomain',
        'DeviceInfo','DeviceType',
        'id_30','id_33']
```


```python
# Add indicator features by computing the value frequencies of try_cols
addFrequencyFeats(df_all,cols=try_cols);
```

    Frequency features added to dataframe: 18

    ['card12_freq', 'C1_freq', 'C2_freq', 'C3_freq', 'C4_freq', 'C5_freq', 'C6_freq', 'C7_freq', 'C8_freq', 'C9_freq', 'C10_freq', 'C11_freq', 'C12_freq', 'C13_freq', 'C14_freq', 'D1_freq', 'addr1_freq', 'addr2_freq']



```python
print_null_cols(df_all)
```




    'No null columns.'



# 3. Preprocessing and Feature Selection


```python
all_vars = TableDescriptor(df_all,'All_data','isFraud')
```

```python
numerical_vars,categorical_vars= numerical_categorical_split(all_vars.variables,min_categories=30)
```

    No of numerical features: 69
    No of categorical features: 75


## A. Filter Features by Correlation to target


```python
#numerical_vars = getCorrelatedFeatures(numerical_vars,corr_cut_off=0.000)
```


```python
categorical_vars = getCorrelatedFeatures(categorical_vars,corr_cut_off=0.1)
```

    Selected features: 8/75


## B. Convert categorical data to Dummies or Codes


```python
categorical_cols = [var.name for var in categorical_vars if var.name not in ['TransactionID','isFraud'] ]
numerical_vars = [var for var in numerical_vars if var.name!= 'TransactionDT']
all_cols = [var.name for var in numerical_vars] + [var.name for var in categorical_vars]
```


```python
df_all=to_categorical(df=df_all[all_cols],cat_cols=categorical_cols,how='dummies')
df_all.shape
```




    (20000, 16552)




```python
print_null_cols(df_all)
```




    'No null columns.'



# 4. Stratified Split training and validation data


```python
x_cols = [col for col in df_all.columns.tolist() if col not in ['TransactionID','isFraud']]
y_col = 'isFraud'
```


```python
#define X and y

X, y = df_all.loc[:,x_cols].values, df_all.loc[:,y_col].values

X_train, X_test, y_train, y_test = getStratifiedTrainTestSplit(X,y,frac=0.2,n_splits=1,
                                                                random_state=0)
```


```python
#df's shapes

for i in [X_train, X_test, y_train, y_test]:
    print(i.shape)
```

    (16000, 16550)
    (4000, 16550)
    (16000,)
    (4000,)


## Save analyzed data


```python
from pickleObjects import *
```


```python
path = './Data/'

dumpObjects(X_train,path+'X_train')
dumpObjects(y_train,path+'y_train')
dumpObjects(X_test,path+'X_test')
dumpObjects(y_test,path+'y_test')
```

    Object saved!
    Object saved!
    Object saved!
    Object saved!


### PCA


```python
# X_train = np.hstack((X_train,PCAT.rec_error(X_train_scaled).reshape(-1,1)))
# X_test = np.hstack((X_test,PCAT.rec_error(X_test_scaled).reshape(-1,1)))
```

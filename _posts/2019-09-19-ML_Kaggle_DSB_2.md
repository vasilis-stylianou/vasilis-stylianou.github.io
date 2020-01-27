---
title: "PBS KIDS Educational App (Kaggle): Part II"
date: 2019-09-19
tags: [machine learning, lightgbm, pandas, numpy]
excerpt: Built a LightGBM multi-classifier to assess the learning process of kids after engaging with educational media
---

# Intro:

Recall that in the previous [post](https://vasilis-stylianou.github.io/ML_Kaggle_DSB_1/) I set up the problem of predicting the accuracy of a user completing an in-app assessment, for the Kaggle competition: [2019 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2019). In particular, I split this problem into four steps:

1. Data Exploration
2. Feature Engineering
3. Feature Selection and Preprocessing
4. Model Training/Evaluation/Selection

and covered the first two steps. In this post I discuss the last two steps. 

[**Github Code**](https://github.com/vasilis-stylianou/Data-Science/tree/master/Projects/Kaggle_IEEE_Fraud) 

# Libraries


```python
import warnings
warnings.simplefilter('ignore')
```


```python
import numpy as np # linear algebra
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb
```

# Import Data


```python
test = pd.read_csv('./data/test.csv')
train = pd.read_csv('./data/train.csv')
train_labels = pd.read_csv('./data/train_labels.csv')
```


```python
train_labels.head()
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
      <th>game_session</th>
      <th>installation_id</th>
      <th>title</th>
      <th>num_correct</th>
      <th>num_incorrect</th>
      <th>accuracy</th>
      <th>accuracy_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>6bdf9623adc94d89</td>
      <td>0006a69f</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>1</td>
      <td>77b8ee947eb84b4e</td>
      <td>0006a69f</td>
      <td>Bird Measurer (Assessment)</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>901acc108f55a5a1</td>
      <td>0006a69f</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>9501794defd84e4d</td>
      <td>0006a69f</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>1</td>
      <td>1</td>
      <td>0.5</td>
      <td>2</td>
    </tr>
    <tr>
      <td>4</td>
      <td>a9ef3ecb3d1acc6a</td>
      <td>0006a69f</td>
      <td>Bird Measurer (Assessment)</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



# 3. Feature Engineering/Selection and Pre-processing

## 3.1 Feature Engineering

```python
def extract_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek  
    return df
    
def get_object_columns(df, columns):
    df = df.groupby(['installation_id', columns])['event_id'].count().reset_index()
    df = df.pivot_table(index = 'installation_id', columns = [columns], values = 'event_id')
    df.columns = list(df.columns)
    df.fillna(0, inplace = True)
    return df

def get_numeric_columns(df, column):
    df = df.groupby('installation_id').agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std']})
    df.fillna(0, inplace = True)
    df.columns = [f'{column}_mean', f'{column}_sum', f'{column}_min', f'{column}_max', f'{column}_std']
    return df

def get_numeric_columns_add(df, agg_column, column):
    df = df.groupby(['installation_id', agg_column]).agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std']}).reset_index()
    df = df.pivot_table(index = 'installation_id', columns = [agg_column], values = [col for col in df.columns if col not in ['installation_id', 'type']])
    df.fillna(0, inplace = True)
    df.columns = list(df.columns)
    return df

def perform_features_engineering(train_df, test_df, train_labels_df):
    print(f'Perform features engineering')
    numerical_columns = ['game_time']
    categorical_columns = ['type', 'world']

    comp_train_df = pd.DataFrame({'installation_id': train_df['installation_id'].unique()})
    comp_train_df.set_index('installation_id', inplace = True)
    comp_test_df = pd.DataFrame({'installation_id': test_df['installation_id'].unique()})
    comp_test_df.set_index('installation_id', inplace = True)

    test_df = extract_time_features(test_df)
    train_df = extract_time_features(train_df)

    for i in numerical_columns:
        comp_train_df = comp_train_df.merge(get_numeric_columns(train_df, i), left_index = True, right_index = True)
        comp_test_df = comp_test_df.merge(get_numeric_columns(test_df, i), left_index = True, right_index = True)
    
    for i in categorical_columns:
        comp_train_df = comp_train_df.merge(get_object_columns(train_df, i), left_index = True, right_index = True)
        comp_test_df = comp_test_df.merge(get_object_columns(test_df, i), left_index = True, right_index = True)
    
    for i in categorical_columns:
        for j in numerical_columns:
            comp_train_df = comp_train_df.merge(get_numeric_columns_add(train_df, i, j), left_index = True, right_index = True)
            comp_test_df = comp_test_df.merge(get_numeric_columns_add(test_df, i, j), left_index = True, right_index = True)
    
    
    comp_train_df.reset_index(inplace = True)
    comp_test_df.reset_index(inplace = True)
    
    print('Our training set have {} rows and {} columns'.format(comp_train_df.shape[0], comp_train_df.shape[1]))

    # get the mode of the title
    labels_map = dict(train_labels_df.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0]))
    # merge target
    labels = train_labels_df[['installation_id', 'title', 'accuracy_group']]
    # replace title with the mode
    labels['title'] = labels['title'].map(labels_map)
    # get title from the test set
    comp_test_df['title'] = test_df.groupby('installation_id').last()['title'].map(labels_map).reset_index(drop = True)
    # join train with labels
    comp_train_df = labels.merge(comp_train_df, on = 'installation_id', how = 'left')
    print('We have {} training rows'.format(comp_train_df.shape[0]))
    
    return comp_train_df, comp_test_df
```


```python
train_df, test_df = perform_features_engineering(train, test, train_labels)
del train, test, train_labels; gc.collect()
```

    Perform features engineering
    Our training set have 17000 rows and 54 columns
    We have 17690 training rows





    0



# 3.2 Data Pre-processing


```python
x_cols = [col for col in train_df.columns if col not in ['installation_id', 'accuracy_group']]
X, y, X_test= train_df.loc[:,x_cols].values, train_df['accuracy_group'].values, test_df.loc[:,x_cols].values
test_sub = test_df[['installation_id']]
```


```python
X_train, X_val, y_train, y_val = train_test_split(X,y)
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)
```

# 4. Training & Evaluation


```python
params = {'n_estimators':2000,
          'boosting_type': 'gbdt',
          'objective': 'regression',
          'metric': 'rmse',
          'subsample': 0.75,
          'subsample_freq': 1,
          'learning_rate': 0.04,
          'feature_fraction': 0.9,
          'max_depth': 15,
          'lambda_l1': 1,  
          'lambda_l2': 1,
          'verbose': 100,
          'early_stopping_rounds': 100
         }
```


```python
num_round = 1000

clf = lgb.train(params,
                train_data,
                num_round,
                valid_sets=[train_data, val_data],
                )
```

    [1]	training's rmse: 1.24717	valid_1's rmse: 1.22921
    Training until validation scores don't improve for 100 rounds
    [2]	training's rmse: 1.23349	valid_1's rmse: 1.21633
    [3]	training's rmse: 1.22063	valid_1's rmse: 1.20397
    [4]	training's rmse: 1.20858	valid_1's rmse: 1.19239
    [5]	training's rmse: 1.19723	valid_1's rmse: 1.18179
    [6]	training's rmse: 1.18668	valid_1's rmse: 1.17179
    [7]	training's rmse: 1.177	valid_1's rmse: 1.16272
    [8]	training's rmse: 1.16768	valid_1's rmse: 1.15398
    [9]	training's rmse: 1.15879	valid_1's rmse: 1.14575
    [10]	training's rmse: 1.15045	valid_1's rmse: 1.13844
    [11]	training's rmse: 1.14273	valid_1's rmse: 1.13157
    [12]	training's rmse: 1.13548	valid_1's rmse: 1.12504
    [13]	training's rmse: 1.12833	valid_1's rmse: 1.11849
    [14]	training's rmse: 1.12178	valid_1's rmse: 1.11276
    [15]	training's rmse: 1.11565	valid_1's rmse: 1.1074
    [16]	training's rmse: 1.1126	valid_1's rmse: 1.10538
    [17]	training's rmse: 1.10706	valid_1's rmse: 1.10066
    [18]	training's rmse: 1.10182	valid_1's rmse: 1.09612
    [19]	training's rmse: 1.09665	valid_1's rmse: 1.09159
    [20]	training's rmse: 1.09194	valid_1's rmse: 1.08751
    [21]	training's rmse: 1.08739	valid_1's rmse: 1.08368
    [22]	training's rmse: 1.08491	valid_1's rmse: 1.08218
    [23]	training's rmse: 1.0807	valid_1's rmse: 1.07863
    [24]	training's rmse: 1.07657	valid_1's rmse: 1.0753
    [25]	training's rmse: 1.07266	valid_1's rmse: 1.07229
    [26]	training's rmse: 1.06886	valid_1's rmse: 1.06943
    [27]	training's rmse: 1.06533	valid_1's rmse: 1.06689
    [28]	training's rmse: 1.06212	valid_1's rmse: 1.06443
    [29]	training's rmse: 1.05898	valid_1's rmse: 1.06178
    [30]	training's rmse: 1.05674	valid_1's rmse: 1.06026
    [31]	training's rmse: 1.05378	valid_1's rmse: 1.05816
    [32]	training's rmse: 1.05099	valid_1's rmse: 1.05596
    [33]	training's rmse: 1.04825	valid_1's rmse: 1.05397
    [34]	training's rmse: 1.04556	valid_1's rmse: 1.05208
    [35]	training's rmse: 1.04308	valid_1's rmse: 1.05037
    [36]	training's rmse: 1.04066	valid_1's rmse: 1.04875
    [37]	training's rmse: 1.03827	valid_1's rmse: 1.0474
    [38]	training's rmse: 1.03604	valid_1's rmse: 1.04595
    [39]	training's rmse: 1.03424	valid_1's rmse: 1.0448
    [40]	training's rmse: 1.03206	valid_1's rmse: 1.04336
    [41]	training's rmse: 1.03008	valid_1's rmse: 1.04218
    [42]	training's rmse: 1.02818	valid_1's rmse: 1.041
    [43]	training's rmse: 1.02633	valid_1's rmse: 1.04011
    [44]	training's rmse: 1.02448	valid_1's rmse: 1.03899
    [45]	training's rmse: 1.0227	valid_1's rmse: 1.03797
    [46]	training's rmse: 1.02107	valid_1's rmse: 1.03721
    [47]	training's rmse: 1.01947	valid_1's rmse: 1.03626
    [48]	training's rmse: 1.01805	valid_1's rmse: 1.0357
    [49]	training's rmse: 1.01659	valid_1's rmse: 1.03482
    [50]	training's rmse: 1.01512	valid_1's rmse: 1.03411
    [51]	training's rmse: 1.01365	valid_1's rmse: 1.03321
    [52]	training's rmse: 1.01237	valid_1's rmse: 1.03227
    [53]	training's rmse: 1.01094	valid_1's rmse: 1.03134
    [54]	training's rmse: 1.0096	valid_1's rmse: 1.03065
    [55]	training's rmse: 1.00834	valid_1's rmse: 1.02978
    [56]	training's rmse: 1.00711	valid_1's rmse: 1.02916
    [57]	training's rmse: 1.00583	valid_1's rmse: 1.02856
    [58]	training's rmse: 1.00446	valid_1's rmse: 1.02801
    [59]	training's rmse: 1.00336	valid_1's rmse: 1.02743
    [60]	training's rmse: 1.00229	valid_1's rmse: 1.02696
    [61]	training's rmse: 1.00116	valid_1's rmse: 1.02637
    [62]	training's rmse: 1.00011	valid_1's rmse: 1.02618
    [63]	training's rmse: 0.99881	valid_1's rmse: 1.02552
    [64]	training's rmse: 0.99783	valid_1's rmse: 1.02509
    [65]	training's rmse: 0.996634	valid_1's rmse: 1.02458
    [66]	training's rmse: 0.995486	valid_1's rmse: 1.02405
    [67]	training's rmse: 0.994434	valid_1's rmse: 1.02345
    [68]	training's rmse: 0.993481	valid_1's rmse: 1.02321
    [69]	training's rmse: 0.992184	valid_1's rmse: 1.02239
    [70]	training's rmse: 0.991295	valid_1's rmse: 1.02216
    [71]	training's rmse: 0.990178	valid_1's rmse: 1.02168
    [72]	training's rmse: 0.98907	valid_1's rmse: 1.02107
    [73]	training's rmse: 0.988151	valid_1's rmse: 1.02078
    [74]	training's rmse: 0.987343	valid_1's rmse: 1.02041
    [75]	training's rmse: 0.986434	valid_1's rmse: 1.01999
    [76]	training's rmse: 0.985485	valid_1's rmse: 1.01981
    [77]	training's rmse: 0.984629	valid_1's rmse: 1.01957
    [78]	training's rmse: 0.983809	valid_1's rmse: 1.01953
    [79]	training's rmse: 0.982942	valid_1's rmse: 1.0192
    [80]	training's rmse: 0.982092	valid_1's rmse: 1.01907
    [81]	training's rmse: 0.98105	valid_1's rmse: 1.01864
    [82]	training's rmse: 0.980063	valid_1's rmse: 1.01817
    [83]	training's rmse: 0.979249	valid_1's rmse: 1.01789
    [84]	training's rmse: 0.978346	valid_1's rmse: 1.01752
    [85]	training's rmse: 0.977496	valid_1's rmse: 1.01724
    [86]	training's rmse: 0.976551	valid_1's rmse: 1.01674
    [87]	training's rmse: 0.975737	valid_1's rmse: 1.01634
    [88]	training's rmse: 0.974938	valid_1's rmse: 1.01621
    [89]	training's rmse: 0.974252	valid_1's rmse: 1.01603
    [90]	training's rmse: 0.973317	valid_1's rmse: 1.01581
    [91]	training's rmse: 0.972486	valid_1's rmse: 1.01554
    [92]	training's rmse: 0.971528	valid_1's rmse: 1.01511
    [93]	training's rmse: 0.970531	valid_1's rmse: 1.01484
    [94]	training's rmse: 0.969596	valid_1's rmse: 1.0147
    [95]	training's rmse: 0.968708	valid_1's rmse: 1.01447
    [96]	training's rmse: 0.96802	valid_1's rmse: 1.01432
    [97]	training's rmse: 0.967175	valid_1's rmse: 1.01396
    [98]	training's rmse: 0.966402	valid_1's rmse: 1.01363
    [99]	training's rmse: 0.965709	valid_1's rmse: 1.01338
    [100]	training's rmse: 0.964846	valid_1's rmse: 1.01286
    [101]	training's rmse: 0.964005	valid_1's rmse: 1.01256
    [102]	training's rmse: 0.96325	valid_1's rmse: 1.01241
    [103]	training's rmse: 0.962458	valid_1's rmse: 1.01217
    [104]	training's rmse: 0.961548	valid_1's rmse: 1.01158
    [105]	training's rmse: 0.960806	valid_1's rmse: 1.01145
    [106]	training's rmse: 0.960025	valid_1's rmse: 1.01123
    [107]	training's rmse: 0.959293	valid_1's rmse: 1.01096
    [108]	training's rmse: 0.958531	valid_1's rmse: 1.01079
    [109]	training's rmse: 0.957774	valid_1's rmse: 1.0107
    [110]	training's rmse: 0.957017	valid_1's rmse: 1.01054
    [111]	training's rmse: 0.956309	valid_1's rmse: 1.01041
    [112]	training's rmse: 0.955541	valid_1's rmse: 1.01033
    [113]	training's rmse: 0.954823	valid_1's rmse: 1.01002
    [114]	training's rmse: 0.954114	valid_1's rmse: 1.00982
    [115]	training's rmse: 0.953407	valid_1's rmse: 1.00953
    [116]	training's rmse: 0.952623	valid_1's rmse: 1.0095
    [117]	training's rmse: 0.951939	valid_1's rmse: 1.00936
    [118]	training's rmse: 0.951251	valid_1's rmse: 1.00916
    [119]	training's rmse: 0.950519	valid_1's rmse: 1.00904
    [120]	training's rmse: 0.949771	valid_1's rmse: 1.00877
    [121]	training's rmse: 0.949012	valid_1's rmse: 1.00861
    [122]	training's rmse: 0.948297	valid_1's rmse: 1.0085
    [123]	training's rmse: 0.947622	valid_1's rmse: 1.00844
    [124]	training's rmse: 0.946824	valid_1's rmse: 1.00824
    [125]	training's rmse: 0.946006	valid_1's rmse: 1.00793
    [126]	training's rmse: 0.945328	valid_1's rmse: 1.00794
    [127]	training's rmse: 0.944694	valid_1's rmse: 1.00796
    [128]	training's rmse: 0.944032	valid_1's rmse: 1.00772
    [129]	training's rmse: 0.943335	valid_1's rmse: 1.0074
    [130]	training's rmse: 0.942676	valid_1's rmse: 1.0072
    [131]	training's rmse: 0.942037	valid_1's rmse: 1.00707
    [132]	training's rmse: 0.941402	valid_1's rmse: 1.00701
    [133]	training's rmse: 0.940878	valid_1's rmse: 1.00706
    [134]	training's rmse: 0.940244	valid_1's rmse: 1.00692
    [135]	training's rmse: 0.939576	valid_1's rmse: 1.0068
    [136]	training's rmse: 0.939007	valid_1's rmse: 1.00668
    [137]	training's rmse: 0.938361	valid_1's rmse: 1.00651
    [138]	training's rmse: 0.937804	valid_1's rmse: 1.00646
    [139]	training's rmse: 0.937105	valid_1's rmse: 1.00622
    [140]	training's rmse: 0.93646	valid_1's rmse: 1.00612
    [141]	training's rmse: 0.935878	valid_1's rmse: 1.00608
    [142]	training's rmse: 0.93522	valid_1's rmse: 1.006
    [143]	training's rmse: 0.934551	valid_1's rmse: 1.00573
    [144]	training's rmse: 0.933947	valid_1's rmse: 1.00559
    [145]	training's rmse: 0.933454	valid_1's rmse: 1.00552
    [146]	training's rmse: 0.932836	valid_1's rmse: 1.00544
    [147]	training's rmse: 0.932252	valid_1's rmse: 1.00535
    [148]	training's rmse: 0.931638	valid_1's rmse: 1.00521
    [149]	training's rmse: 0.931075	valid_1's rmse: 1.00521
    [150]	training's rmse: 0.930521	valid_1's rmse: 1.00508
    [151]	training's rmse: 0.93	valid_1's rmse: 1.00496
    [152]	training's rmse: 0.929357	valid_1's rmse: 1.0046
    [153]	training's rmse: 0.928774	valid_1's rmse: 1.00455
    [154]	training's rmse: 0.928273	valid_1's rmse: 1.00454
    [155]	training's rmse: 0.927688	valid_1's rmse: 1.00447
    [156]	training's rmse: 0.927095	valid_1's rmse: 1.00436
    [157]	training's rmse: 0.926587	valid_1's rmse: 1.00428
    [158]	training's rmse: 0.926057	valid_1's rmse: 1.00405
    [159]	training's rmse: 0.925434	valid_1's rmse: 1.00389
    [160]	training's rmse: 0.924938	valid_1's rmse: 1.00379
    [161]	training's rmse: 0.924455	valid_1's rmse: 1.00382
    [162]	training's rmse: 0.92384	valid_1's rmse: 1.00348
    [163]	training's rmse: 0.923344	valid_1's rmse: 1.0034
    [164]	training's rmse: 0.922868	valid_1's rmse: 1.00319
    [165]	training's rmse: 0.922383	valid_1's rmse: 1.00318
    [166]	training's rmse: 0.921824	valid_1's rmse: 1.00298
    [167]	training's rmse: 0.921272	valid_1's rmse: 1.00288
    [168]	training's rmse: 0.920691	valid_1's rmse: 1.00281
    [169]	training's rmse: 0.920219	valid_1's rmse: 1.00282
    [170]	training's rmse: 0.919808	valid_1's rmse: 1.00263
    [171]	training's rmse: 0.919327	valid_1's rmse: 1.00259
    [172]	training's rmse: 0.918829	valid_1's rmse: 1.00238
    [173]	training's rmse: 0.918388	valid_1's rmse: 1.00218
    [174]	training's rmse: 0.917962	valid_1's rmse: 1.00198
    [175]	training's rmse: 0.917406	valid_1's rmse: 1.00198
    [176]	training's rmse: 0.916976	valid_1's rmse: 1.00191
    [177]	training's rmse: 0.916515	valid_1's rmse: 1.00186
    [178]	training's rmse: 0.916105	valid_1's rmse: 1.00191
    [179]	training's rmse: 0.915645	valid_1's rmse: 1.00178
    [180]	training's rmse: 0.91523	valid_1's rmse: 1.00158
    [181]	training's rmse: 0.914807	valid_1's rmse: 1.00149
    [182]	training's rmse: 0.914389	valid_1's rmse: 1.00149
    [183]	training's rmse: 0.913944	valid_1's rmse: 1.0013
    [184]	training's rmse: 0.913497	valid_1's rmse: 1.00134
    [185]	training's rmse: 0.913038	valid_1's rmse: 1.00128
    [186]	training's rmse: 0.912658	valid_1's rmse: 1.0012
    [187]	training's rmse: 0.912285	valid_1's rmse: 1.00124
    [188]	training's rmse: 0.911823	valid_1's rmse: 1.001
    [189]	training's rmse: 0.911411	valid_1's rmse: 1.00102
    [190]	training's rmse: 0.910984	valid_1's rmse: 1.00107
    [191]	training's rmse: 0.910634	valid_1's rmse: 1.00107
    [192]	training's rmse: 0.91018	valid_1's rmse: 1.00101
    [193]	training's rmse: 0.909803	valid_1's rmse: 1.00109
    [194]	training's rmse: 0.909255	valid_1's rmse: 1.00073
    [195]	training's rmse: 0.908784	valid_1's rmse: 1.00065
    [196]	training's rmse: 0.908358	valid_1's rmse: 1.00051
    [197]	training's rmse: 0.907988	valid_1's rmse: 1.00041
    [198]	training's rmse: 0.90758	valid_1's rmse: 1.00034
    [199]	training's rmse: 0.907087	valid_1's rmse: 1.00019
    [200]	training's rmse: 0.906743	valid_1's rmse: 1.00013
    [201]	training's rmse: 0.906382	valid_1's rmse: 1.00003
    [202]	training's rmse: 0.906005	valid_1's rmse: 1.00011
    [203]	training's rmse: 0.905549	valid_1's rmse: 1.00004
    [204]	training's rmse: 0.905204	valid_1's rmse: 0.999967
    [205]	training's rmse: 0.904819	valid_1's rmse: 0.999825
    [206]	training's rmse: 0.904492	valid_1's rmse: 0.999783
    [207]	training's rmse: 0.904167	valid_1's rmse: 0.999867
    [208]	training's rmse: 0.90372	valid_1's rmse: 0.999857
    [209]	training's rmse: 0.903247	valid_1's rmse: 0.999892
    [210]	training's rmse: 0.902904	valid_1's rmse: 0.99991
    [211]	training's rmse: 0.902437	valid_1's rmse: 0.999847
    [212]	training's rmse: 0.902017	valid_1's rmse: 1.00002
    [213]	training's rmse: 0.901562	valid_1's rmse: 1.00012
    [214]	training's rmse: 0.9012	valid_1's rmse: 0.999877
    [215]	training's rmse: 0.900758	valid_1's rmse: 0.999895
    [216]	training's rmse: 0.900341	valid_1's rmse: 0.999814
    [217]	training's rmse: 0.90004	valid_1's rmse: 0.999736
    [218]	training's rmse: 0.899634	valid_1's rmse: 0.999529
    [219]	training's rmse: 0.899209	valid_1's rmse: 0.99962
    [220]	training's rmse: 0.898863	valid_1's rmse: 0.999537
    [221]	training's rmse: 0.898518	valid_1's rmse: 0.999556
    [222]	training's rmse: 0.898134	valid_1's rmse: 0.999435
    [223]	training's rmse: 0.897717	valid_1's rmse: 0.99944
    [224]	training's rmse: 0.897357	valid_1's rmse: 0.999511
    [225]	training's rmse: 0.897009	valid_1's rmse: 0.999549
    [226]	training's rmse: 0.896694	valid_1's rmse: 0.999488
    [227]	training's rmse: 0.896317	valid_1's rmse: 0.999418
    [228]	training's rmse: 0.89592	valid_1's rmse: 0.999376
    [229]	training's rmse: 0.895684	valid_1's rmse: 0.999353
    [230]	training's rmse: 0.895328	valid_1's rmse: 0.999271
    [231]	training's rmse: 0.895005	valid_1's rmse: 0.999369
    [232]	training's rmse: 0.894616	valid_1's rmse: 0.999322
    [233]	training's rmse: 0.894312	valid_1's rmse: 0.999303
    [234]	training's rmse: 0.893935	valid_1's rmse: 0.999221
    [235]	training's rmse: 0.893577	valid_1's rmse: 0.999303
    [236]	training's rmse: 0.893253	valid_1's rmse: 0.99933
    [237]	training's rmse: 0.892933	valid_1's rmse: 0.999327
    [238]	training's rmse: 0.892551	valid_1's rmse: 0.999361
    [239]	training's rmse: 0.892139	valid_1's rmse: 0.999391
    [240]	training's rmse: 0.891783	valid_1's rmse: 0.999477
    [241]	training's rmse: 0.891342	valid_1's rmse: 0.99942
    [242]	training's rmse: 0.890988	valid_1's rmse: 0.999332
    [243]	training's rmse: 0.890652	valid_1's rmse: 0.999298
    [244]	training's rmse: 0.890391	valid_1's rmse: 0.999325
    [245]	training's rmse: 0.890001	valid_1's rmse: 0.999128
    [246]	training's rmse: 0.889654	valid_1's rmse: 0.999053
    [247]	training's rmse: 0.889352	valid_1's rmse: 0.999155
    [248]	training's rmse: 0.888997	valid_1's rmse: 0.999066
    [249]	training's rmse: 0.888719	valid_1's rmse: 0.998986
    [250]	training's rmse: 0.888457	valid_1's rmse: 0.998916
    [251]	training's rmse: 0.88816	valid_1's rmse: 0.998819
    [252]	training's rmse: 0.887825	valid_1's rmse: 0.998737
    [253]	training's rmse: 0.887556	valid_1's rmse: 0.998674
    [254]	training's rmse: 0.88727	valid_1's rmse: 0.998694
    [255]	training's rmse: 0.886955	valid_1's rmse: 0.99862
    [256]	training's rmse: 0.88663	valid_1's rmse: 0.99862
    [257]	training's rmse: 0.886302	valid_1's rmse: 0.998627
    [258]	training's rmse: 0.88602	valid_1's rmse: 0.998555
    [259]	training's rmse: 0.885704	valid_1's rmse: 0.998667
    [260]	training's rmse: 0.885417	valid_1's rmse: 0.998681
    [261]	training's rmse: 0.885133	valid_1's rmse: 0.998477
    [262]	training's rmse: 0.884803	valid_1's rmse: 0.998411
    [263]	training's rmse: 0.884573	valid_1's rmse: 0.998505
    [264]	training's rmse: 0.884338	valid_1's rmse: 0.998619
    [265]	training's rmse: 0.884079	valid_1's rmse: 0.998537
    [266]	training's rmse: 0.883751	valid_1's rmse: 0.998573
    [267]	training's rmse: 0.883393	valid_1's rmse: 0.998695
    [268]	training's rmse: 0.883108	valid_1's rmse: 0.998586
    [269]	training's rmse: 0.882865	valid_1's rmse: 0.998622
    [270]	training's rmse: 0.882633	valid_1's rmse: 0.998663
    [271]	training's rmse: 0.88234	valid_1's rmse: 0.998608
    [272]	training's rmse: 0.88204	valid_1's rmse: 0.998609
    [273]	training's rmse: 0.88171	valid_1's rmse: 0.998623
    [274]	training's rmse: 0.881402	valid_1's rmse: 0.998474
    [275]	training's rmse: 0.881081	valid_1's rmse: 0.998401
    [276]	training's rmse: 0.880756	valid_1's rmse: 0.998341
    [277]	training's rmse: 0.880499	valid_1's rmse: 0.998376
    [278]	training's rmse: 0.880158	valid_1's rmse: 0.998348
    [279]	training's rmse: 0.87993	valid_1's rmse: 0.99842
    [280]	training's rmse: 0.879571	valid_1's rmse: 0.998484
    [281]	training's rmse: 0.879258	valid_1's rmse: 0.998447
    [282]	training's rmse: 0.879072	valid_1's rmse: 0.99836
    [283]	training's rmse: 0.878808	valid_1's rmse: 0.99841
    [284]	training's rmse: 0.87855	valid_1's rmse: 0.998384
    [285]	training's rmse: 0.878244	valid_1's rmse: 0.998277
    [286]	training's rmse: 0.878062	valid_1's rmse: 0.998357
    [287]	training's rmse: 0.87773	valid_1's rmse: 0.998322
    [288]	training's rmse: 0.877441	valid_1's rmse: 0.998202
    [289]	training's rmse: 0.87714	valid_1's rmse: 0.998302
    [290]	training's rmse: 0.876817	valid_1's rmse: 0.998226
    [291]	training's rmse: 0.876563	valid_1's rmse: 0.998063
    [292]	training's rmse: 0.876293	valid_1's rmse: 0.998167
    [293]	training's rmse: 0.875993	valid_1's rmse: 0.998132
    [294]	training's rmse: 0.875813	valid_1's rmse: 0.99806
    [295]	training's rmse: 0.875564	valid_1's rmse: 0.998047
    [296]	training's rmse: 0.875294	valid_1's rmse: 0.998088
    [297]	training's rmse: 0.875	valid_1's rmse: 0.998122
    [298]	training's rmse: 0.874794	valid_1's rmse: 0.998168
    [299]	training's rmse: 0.874523	valid_1's rmse: 0.998054
    [300]	training's rmse: 0.874305	valid_1's rmse: 0.998128
    [301]	training's rmse: 0.874075	valid_1's rmse: 0.998204
    [302]	training's rmse: 0.873802	valid_1's rmse: 0.99811
    [303]	training's rmse: 0.873519	valid_1's rmse: 0.99802
    [304]	training's rmse: 0.873268	valid_1's rmse: 0.997972
    [305]	training's rmse: 0.873049	valid_1's rmse: 0.997953
    [306]	training's rmse: 0.872792	valid_1's rmse: 0.997943
    [307]	training's rmse: 0.872547	valid_1's rmse: 0.998026
    [308]	training's rmse: 0.872289	valid_1's rmse: 0.998086
    [309]	training's rmse: 0.871974	valid_1's rmse: 0.997997
    [310]	training's rmse: 0.871748	valid_1's rmse: 0.997915
    [311]	training's rmse: 0.871459	valid_1's rmse: 0.99788
    [312]	training's rmse: 0.87124	valid_1's rmse: 0.997989
    [313]	training's rmse: 0.871067	valid_1's rmse: 0.998083
    [314]	training's rmse: 0.870833	valid_1's rmse: 0.99817
    [315]	training's rmse: 0.870577	valid_1's rmse: 0.998232
    [316]	training's rmse: 0.870278	valid_1's rmse: 0.99824
    [317]	training's rmse: 0.870087	valid_1's rmse: 0.99835
    [318]	training's rmse: 0.869867	valid_1's rmse: 0.998313
    [319]	training's rmse: 0.86963	valid_1's rmse: 0.998317
    [320]	training's rmse: 0.869443	valid_1's rmse: 0.998332
    [321]	training's rmse: 0.869259	valid_1's rmse: 0.998265
    [322]	training's rmse: 0.869015	valid_1's rmse: 0.998228
    [323]	training's rmse: 0.868815	valid_1's rmse: 0.998219
    [324]	training's rmse: 0.868584	valid_1's rmse: 0.998194
    [325]	training's rmse: 0.868389	valid_1's rmse: 0.998243
    [326]	training's rmse: 0.868171	valid_1's rmse: 0.998203
    [327]	training's rmse: 0.867947	valid_1's rmse: 0.998254
    [328]	training's rmse: 0.867715	valid_1's rmse: 0.998211
    [329]	training's rmse: 0.867464	valid_1's rmse: 0.99817
    [330]	training's rmse: 0.867171	valid_1's rmse: 0.998234
    [331]	training's rmse: 0.866947	valid_1's rmse: 0.998149
    [332]	training's rmse: 0.866776	valid_1's rmse: 0.998038
    [333]	training's rmse: 0.866515	valid_1's rmse: 0.998123
    [334]	training's rmse: 0.866273	valid_1's rmse: 0.99813
    [335]	training's rmse: 0.86604	valid_1's rmse: 0.998062
    [336]	training's rmse: 0.865815	valid_1's rmse: 0.998063
    [337]	training's rmse: 0.865521	valid_1's rmse: 0.998049
    [338]	training's rmse: 0.865318	valid_1's rmse: 0.998169
    [339]	training's rmse: 0.865108	valid_1's rmse: 0.998237
    [340]	training's rmse: 0.864794	valid_1's rmse: 0.998259
    [341]	training's rmse: 0.864556	valid_1's rmse: 0.99828
    [342]	training's rmse: 0.864344	valid_1's rmse: 0.99829
    [343]	training's rmse: 0.864166	valid_1's rmse: 0.998367
    [344]	training's rmse: 0.863987	valid_1's rmse: 0.998442
    [345]	training's rmse: 0.863771	valid_1's rmse: 0.998423
    [346]	training's rmse: 0.863582	valid_1's rmse: 0.9984
    [347]	training's rmse: 0.863416	valid_1's rmse: 0.998434
    [348]	training's rmse: 0.863146	valid_1's rmse: 0.998486
    [349]	training's rmse: 0.862946	valid_1's rmse: 0.99849
    [350]	training's rmse: 0.862711	valid_1's rmse: 0.998556
    [351]	training's rmse: 0.862496	valid_1's rmse: 0.998491
    [352]	training's rmse: 0.862275	valid_1's rmse: 0.9986
    [353]	training's rmse: 0.862101	valid_1's rmse: 0.998678
    [354]	training's rmse: 0.861848	valid_1's rmse: 0.998633
    [355]	training's rmse: 0.861574	valid_1's rmse: 0.998597
    [356]	training's rmse: 0.861365	valid_1's rmse: 0.998635
    [357]	training's rmse: 0.861196	valid_1's rmse: 0.998628
    [358]	training's rmse: 0.86082	valid_1's rmse: 0.998723
    [359]	training's rmse: 0.860644	valid_1's rmse: 0.99881
    [360]	training's rmse: 0.860446	valid_1's rmse: 0.998716
    [361]	training's rmse: 0.860312	valid_1's rmse: 0.99862
    [362]	training's rmse: 0.860143	valid_1's rmse: 0.998598
    [363]	training's rmse: 0.859776	valid_1's rmse: 0.998628
    [364]	training's rmse: 0.859542	valid_1's rmse: 0.998581
    [365]	training's rmse: 0.859329	valid_1's rmse: 0.998599
    [366]	training's rmse: 0.85912	valid_1's rmse: 0.99867
    [367]	training's rmse: 0.858969	valid_1's rmse: 0.998582
    [368]	training's rmse: 0.85868	valid_1's rmse: 0.998585
    [369]	training's rmse: 0.858517	valid_1's rmse: 0.998749
    [370]	training's rmse: 0.858286	valid_1's rmse: 0.99868
    [371]	training's rmse: 0.858079	valid_1's rmse: 0.998618
    [372]	training's rmse: 0.857943	valid_1's rmse: 0.998551
    [373]	training's rmse: 0.857714	valid_1's rmse: 0.998676
    [374]	training's rmse: 0.857456	valid_1's rmse: 0.998677
    [375]	training's rmse: 0.857176	valid_1's rmse: 0.998709
    [376]	training's rmse: 0.857012	valid_1's rmse: 0.998795
    [377]	training's rmse: 0.856865	valid_1's rmse: 0.998784
    [378]	training's rmse: 0.856682	valid_1's rmse: 0.99879
    [379]	training's rmse: 0.856479	valid_1's rmse: 0.998748
    [380]	training's rmse: 0.8563	valid_1's rmse: 0.998791
    [381]	training's rmse: 0.85611	valid_1's rmse: 0.998734
    [382]	training's rmse: 0.855965	valid_1's rmse: 0.998851
    [383]	training's rmse: 0.85575	valid_1's rmse: 0.998823
    [384]	training's rmse: 0.855621	valid_1's rmse: 0.998893
    [385]	training's rmse: 0.85529	valid_1's rmse: 0.99885
    [386]	training's rmse: 0.855147	valid_1's rmse: 0.998885
    [387]	training's rmse: 0.854914	valid_1's rmse: 0.998791
    [388]	training's rmse: 0.854728	valid_1's rmse: 0.998774
    [389]	training's rmse: 0.854541	valid_1's rmse: 0.998831
    [390]	training's rmse: 0.854371	valid_1's rmse: 0.998854
    [391]	training's rmse: 0.854271	valid_1's rmse: 0.998838
    [392]	training's rmse: 0.854131	valid_1's rmse: 0.998849
    [393]	training's rmse: 0.853916	valid_1's rmse: 0.998808
    [394]	training's rmse: 0.853569	valid_1's rmse: 0.998863
    [395]	training's rmse: 0.853418	valid_1's rmse: 0.998891
    [396]	training's rmse: 0.853284	valid_1's rmse: 0.998906
    [397]	training's rmse: 0.853196	valid_1's rmse: 0.998903
    [398]	training's rmse: 0.853017	valid_1's rmse: 0.998825
    [399]	training's rmse: 0.852853	valid_1's rmse: 0.998786
    [400]	training's rmse: 0.852681	valid_1's rmse: 0.998849
    [401]	training's rmse: 0.852414	valid_1's rmse: 0.998896
    [402]	training's rmse: 0.852268	valid_1's rmse: 0.998976
    [403]	training's rmse: 0.852132	valid_1's rmse: 0.99905
    [404]	training's rmse: 0.851923	valid_1's rmse: 0.999106
    [405]	training's rmse: 0.851703	valid_1's rmse: 0.999201
    [406]	training's rmse: 0.851542	valid_1's rmse: 0.999243
    [407]	training's rmse: 0.851411	valid_1's rmse: 0.999244
    [408]	training's rmse: 0.851214	valid_1's rmse: 0.999287
    [409]	training's rmse: 0.851027	valid_1's rmse: 0.999306
    [410]	training's rmse: 0.850894	valid_1's rmse: 0.999381
    [411]	training's rmse: 0.850712	valid_1's rmse: 0.99936
    Early stopping, best iteration is:
    [311]	training's rmse: 0.871459	valid_1's rmse: 0.99788



```python
y_pred = clf.predict(X_val)
cohen_kappa_score(y_val, np.round(y_pred,0), weights= 'quadratic')
```




    0.5246363303586942




```python
test_pred = clf.predict(X_val)
test_sub['accuracy_group'] = pd.Series(test_pred)
```

## Save predictions


```python
test_sub.to_csv('submission.csv', index=False)
```

# Conclusions
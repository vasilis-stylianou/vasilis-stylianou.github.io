---
title: "PBS KIDS Educational App (Kaggle): Part I"
date: 2019-09-19
tags: [machine learning, lightgbm, pandas, numpy]
excerpt: Built a LightGBM multi-classifier to assess the learning process of kids after engaging with educational media
---

# Intro:
The following work was done for the Kaggle competition: [2019 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2019). Contestants were provided with four datasets, from **PBS KIDS Measure Up! app**:
- train.csv - the training set
- train_labels.csv - the ground truth labels for the training set
- specs.csv -  the specification of the various event types
- test.csv - the test set 

and were asked to predict the level of accuracy of a user completing an in-app assessment. 

Submissions were evaluated on based on the quadratic weighted kappa, which measures the agreement between two outcomes. 

# Index:
1. Data Exploration
2. Feature Engineering
3. Feature Selection and Preprocessing
4. Model Training/Evaluation/Selection

In this post I discuss steps 1-2.

[**Github Code**](https://github.com/vasilis-stylianou/Data-Science/tree/master/Projects/Kaggle_IEEE_Fraud) 

# Step 1: Data Cleaning


# Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json 
import os
from pprint import pprint
sns.set_style("whitegrid")
my_pal = sns.color_palette(n_colors=10)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
```

# A. Load Data (a sample)


```python
def load_and_process_data(path, nrows=1000000, timestamp=False):
    
    # load data
    if timestamp:
        df = pd.read_csv(path, nrows=nrows, parse_dates=['timestamp'])
    else:
        df = pd.read_csv(path, nrows=nrows)
    
    # convert utf-16-like ids to integers
    try:
        df['installation_id'] = df['installation_id'].apply(lambda x: int(x,16))
    except:
        pass

    # convert milliseconds to mins
    try:
        df['game_time'] = df['game_time'].apply(lambda x: round(x/(1000*60),2))
    except:
        pass
    
    df = df.set_index(['installation_id','game_session'])
    
    return df
```

### A.1 Train


```python
train = load_and_process_data('./data/train.csv', nrows=1000000, timestamp=True)
train.head()
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
      <th></th>
      <th>event_id</th>
      <th>timestamp</th>
      <th>event_data</th>
      <th>event_count</th>
      <th>event_code</th>
      <th>game_time</th>
      <th>title</th>
      <th>type</th>
      <th>world</th>
    </tr>
    <tr>
      <th>installation_id</th>
      <th>game_session</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="top">125199</td>
      <td>45bb1e1b6b50c07b</td>
      <td>27253bdc</td>
      <td>2019-09-06 17:53:46.937000+00:00</td>
      <td>{"event_code": 2000, "event_count": 1}</td>
      <td>1</td>
      <td>2000</td>
      <td>0.00</td>
      <td>Welcome to Lost Lagoon!</td>
      <td>Clip</td>
      <td>NONE</td>
    </tr>
    <tr>
      <td>17eeb7f223665f53</td>
      <td>27253bdc</td>
      <td>2019-09-06 17:54:17.519000+00:00</td>
      <td>{"event_code": 2000, "event_count": 1}</td>
      <td>1</td>
      <td>2000</td>
      <td>0.00</td>
      <td>Magma Peak - Level 1</td>
      <td>Clip</td>
      <td>MAGMAPEAK</td>
    </tr>
    <tr>
      <td>0848ef14a8dc6892</td>
      <td>77261ab5</td>
      <td>2019-09-06 17:54:56.302000+00:00</td>
      <td>{"version":"1.0","event_count":1,"game_time":0...</td>
      <td>1</td>
      <td>2000</td>
      <td>0.00</td>
      <td>Sandcastle Builder (Activity)</td>
      <td>Activity</td>
      <td>MAGMAPEAK</td>
    </tr>
    <tr>
      <td>0848ef14a8dc6892</td>
      <td>b2dba42b</td>
      <td>2019-09-06 17:54:56.387000+00:00</td>
      <td>{"description":"Let's build a sandcastle! Firs...</td>
      <td>2</td>
      <td>3010</td>
      <td>0.00</td>
      <td>Sandcastle Builder (Activity)</td>
      <td>Activity</td>
      <td>MAGMAPEAK</td>
    </tr>
    <tr>
      <td>0848ef14a8dc6892</td>
      <td>1bb5fbdb</td>
      <td>2019-09-06 17:55:03.253000+00:00</td>
      <td>{"description":"Let's build a sandcastle! Firs...</td>
      <td>3</td>
      <td>3110</td>
      <td>0.12</td>
      <td>Sandcastle Builder (Activity)</td>
      <td>Activity</td>
      <td>MAGMAPEAK</td>
    </tr>
  </tbody>
</table>
</div>



### Remove Noisy Data

Noise: game_time =0  and incorrect 

### A.2 Labels


```python
train_labels = load_and_process_data('./data/train_labels.csv', nrows=1000000)
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
      <th></th>
      <th>title</th>
      <th>num_correct</th>
      <th>num_incorrect</th>
      <th>accuracy</th>
      <th>accuracy_group</th>
    </tr>
    <tr>
      <th>installation_id</th>
      <th>game_session</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="top">435871</td>
      <td>6bdf9623adc94d89</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>Bird Measurer (Assessment)</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>901acc108f55a5a1</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>9501794defd84e4d</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>1</td>
      <td>1</td>
      <td>0.5</td>
      <td>2</td>
    </tr>
    <tr>
      <td>a9ef3ecb3d1acc6a</td>
      <td>Bird Measurer (Assessment)</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### A3. Merge all Assessment train data


```python
assessments = pd.merge(train[train.type == 'Assessment'], train_labels, 
                       left_index=True, right_index=True, how='inner')

# clean df
assessments = assessments.drop(['title_y','type'],axis=1)\
                         .rename({'title_x':'title'},axis=1)
```


```python
assessments.head(100)
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
      <th></th>
      <th>event_id</th>
      <th>timestamp</th>
      <th>event_data</th>
      <th>event_count</th>
      <th>event_code</th>
      <th>game_time</th>
      <th>title</th>
      <th>world</th>
      <th>num_correct</th>
      <th>num_incorrect</th>
      <th>accuracy</th>
      <th>accuracy_group</th>
    </tr>
    <tr>
      <th>installation_id</th>
      <th>game_session</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="100" valign="top">435871</td>
      <td>6bdf9623adc94d89</td>
      <td>3bfd1a65</td>
      <td>2019-08-06 05:37:50.020000+00:00</td>
      <td>{"version":"1.0","event_count":1,"game_time":0...</td>
      <td>1</td>
      <td>2000</td>
      <td>0.00</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>db02c830</td>
      <td>2019-08-06 05:37:50.078000+00:00</td>
      <td>{"event_count":2,"game_time":77,"event_code":2...</td>
      <td>2</td>
      <td>2025</td>
      <td>0.00</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>a1e4395d</td>
      <td>2019-08-06 05:37:50.082000+00:00</td>
      <td>{"description":"Pull three mushrooms out of th...</td>
      <td>3</td>
      <td>3010</td>
      <td>0.00</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>7da34a02</td>
      <td>2019-08-06 05:37:52.799000+00:00</td>
      <td>{"coordinates":{"x":199,"y":484,"stage_width":...</td>
      <td>4</td>
      <td>4070</td>
      <td>0.05</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>28ed704e</td>
      <td>2019-08-06 05:37:53.631000+00:00</td>
      <td>{"height":1,"coordinates":{"x":171,"y":519,"st...</td>
      <td>5</td>
      <td>4025</td>
      <td>0.06</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>a52b92d5</td>
      <td>2019-08-06 05:37:53.632000+00:00</td>
      <td>{"description":"Pull three mushrooms out of th...</td>
      <td>6</td>
      <td>3110</td>
      <td>0.06</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>9d29771f</td>
      <td>2019-08-06 05:37:53.635000+00:00</td>
      <td>{"description":"That's one!","identifier":"Dot...</td>
      <td>7</td>
      <td>3021</td>
      <td>0.06</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>c74f40cd</td>
      <td>2019-08-06 05:37:54.253000+00:00</td>
      <td>{"description":"That's one!","identifier":"Dot...</td>
      <td>8</td>
      <td>3121</td>
      <td>0.07</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>28ed704e</td>
      <td>2019-08-06 05:37:54.930000+00:00</td>
      <td>{"height":2,"coordinates":{"x":496,"y":502,"st...</td>
      <td>9</td>
      <td>4025</td>
      <td>0.08</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>9d29771f</td>
      <td>2019-08-06 05:37:54.934000+00:00</td>
      <td>{"description":"two...","identifier":"Dot_Two"...</td>
      <td>10</td>
      <td>3021</td>
      <td>0.08</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>c74f40cd</td>
      <td>2019-08-06 05:37:55.464000+00:00</td>
      <td>{"description":"two...","identifier":"Dot_Two"...</td>
      <td>11</td>
      <td>3121</td>
      <td>0.09</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>28ed704e</td>
      <td>2019-08-06 05:37:56+00:00</td>
      <td>{"height":4,"coordinates":{"x":327,"y":543,"st...</td>
      <td>12</td>
      <td>4025</td>
      <td>0.10</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>9d29771f</td>
      <td>2019-08-06 05:37:56.003000+00:00</td>
      <td>{"description":"and three!","identifier":"Dot_...</td>
      <td>13</td>
      <td>3021</td>
      <td>0.10</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>3dfd4aa4</td>
      <td>2019-08-06 05:37:56.693000+00:00</td>
      <td>{"event_count":16,"game_time":6692,"event_code...</td>
      <td>16</td>
      <td>2020</td>
      <td>0.11</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>83c6c409</td>
      <td>2019-08-06 05:37:56.693000+00:00</td>
      <td>{"duration":6615,"event_count":15,"game_time":...</td>
      <td>15</td>
      <td>2035</td>
      <td>0.11</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>c74f40cd</td>
      <td>2019-08-06 05:37:56.693000+00:00</td>
      <td>{"description":"and three!","identifier":"Dot_...</td>
      <td>14</td>
      <td>3121</td>
      <td>0.11</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>a1e4395d</td>
      <td>2019-08-06 05:37:56.697000+00:00</td>
      <td>{"description":"Now order these mushrooms by h...</td>
      <td>17</td>
      <td>3010</td>
      <td>0.11</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>7da34a02</td>
      <td>2019-08-06 05:37:58.885000+00:00</td>
      <td>{"coordinates":{"x":323,"y":534,"stage_width":...</td>
      <td>18</td>
      <td>4070</td>
      <td>0.15</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>7da34a02</td>
      <td>2019-08-06 05:37:59.759000+00:00</td>
      <td>{"coordinates":{"x":333,"y":551,"stage_width":...</td>
      <td>19</td>
      <td>4070</td>
      <td>0.16</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>7da34a02</td>
      <td>2019-08-06 05:38:00.706000+00:00</td>
      <td>{"coordinates":{"x":328,"y":534,"stage_width":...</td>
      <td>20</td>
      <td>4070</td>
      <td>0.18</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>a52b92d5</td>
      <td>2019-08-06 05:38:00.716000+00:00</td>
      <td>{"description":"Now order these mushrooms by h...</td>
      <td>21</td>
      <td>3110</td>
      <td>0.18</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>7da34a02</td>
      <td>2019-08-06 05:38:01.621000+00:00</td>
      <td>{"coordinates":{"x":290,"y":536,"stage_width":...</td>
      <td>22</td>
      <td>4070</td>
      <td>0.19</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>fbaf3456</td>
      <td>2019-08-06 05:38:02.273000+00:00</td>
      <td>{"height":4,"stumps":[0,0,0],"coordinates":{"x...</td>
      <td>23</td>
      <td>4030</td>
      <td>0.20</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>5f0eb72c</td>
      <td>2019-08-06 05:38:03.621000+00:00</td>
      <td>{"height":4,"destination":"right","stumps":[0,...</td>
      <td>24</td>
      <td>4020</td>
      <td>0.23</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>fbaf3456</td>
      <td>2019-08-06 05:38:04.420000+00:00</td>
      <td>{"height":2,"stumps":[0,0,4],"coordinates":{"x...</td>
      <td>25</td>
      <td>4030</td>
      <td>0.24</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>5f0eb72c</td>
      <td>2019-08-06 05:38:05.265000+00:00</td>
      <td>{"height":2,"destination":"middle","stumps":[0...</td>
      <td>26</td>
      <td>4020</td>
      <td>0.25</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>fbaf3456</td>
      <td>2019-08-06 05:38:05.766000+00:00</td>
      <td>{"height":1,"stumps":[0,2,4],"coordinates":{"x...</td>
      <td>27</td>
      <td>4030</td>
      <td>0.26</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>5f0eb72c</td>
      <td>2019-08-06 05:38:06.823000+00:00</td>
      <td>{"height":1,"destination":"left","stumps":[1,2...</td>
      <td>28</td>
      <td>4020</td>
      <td>0.28</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>a1e4395d</td>
      <td>2019-08-06 05:38:06.826000+00:00</td>
      <td>{"description":"Okay, when you want to check y...</td>
      <td>29</td>
      <td>3010</td>
      <td>0.28</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>25fa8af4</td>
      <td>2019-08-06 05:38:08.036000+00:00</td>
      <td>{"correct":true,"stumps":[1,2,4],"event_count"...</td>
      <td>30</td>
      <td>4100</td>
      <td>0.30</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>a52b92d5</td>
      <td>2019-08-06 05:38:08.039000+00:00</td>
      <td>{"description":"Okay, when you want to check y...</td>
      <td>31</td>
      <td>3110</td>
      <td>0.30</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>9d29771f</td>
      <td>2019-08-06 05:38:08.040000+00:00</td>
      <td>{"description":"Alright! This one is the littl...</td>
      <td>32</td>
      <td>3021</td>
      <td>0.30</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>c74f40cd</td>
      <td>2019-08-06 05:38:12.223000+00:00</td>
      <td>{"description":"Alright! This one is the littl...</td>
      <td>33</td>
      <td>3121</td>
      <td>0.37</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>6c930e6e</td>
      <td>2019-08-06 05:38:12.836000+00:00</td>
      <td>{"duration":16126,"misses":0,"event_count":34,...</td>
      <td>34</td>
      <td>2030</td>
      <td>0.38</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>6bdf9623adc94d89</td>
      <td>a5be6304</td>
      <td>2019-08-06 05:38:16.835000+00:00</td>
      <td>{"session_duration":26827,"exit_type":"game_co...</td>
      <td>35</td>
      <td>2010</td>
      <td>0.45</td>
      <td>Mushroom Sorter (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>f56e0afc</td>
      <td>2019-08-06 05:35:19.167000+00:00</td>
      <td>{"version":"1.0","event_count":1,"game_time":0...</td>
      <td>1</td>
      <td>2000</td>
      <td>0.00</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>ec138c1c</td>
      <td>2019-08-06 05:35:19.174000+00:00</td>
      <td>{"stage_number":1,"event_count":2,"game_time":...</td>
      <td>2</td>
      <td>2020</td>
      <td>0.00</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>1375ccb7</td>
      <td>2019-08-06 05:35:19.177000+00:00</td>
      <td>{"description":"Use the caterpillars to measur...</td>
      <td>3</td>
      <td>3010</td>
      <td>0.00</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>bdf49a58</td>
      <td>2019-08-06 05:35:23.654000+00:00</td>
      <td>{"description":"Use the caterpillars to measur...</td>
      <td>4</td>
      <td>3110</td>
      <td>0.08</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:35:24.263000+00:00</td>
      <td>{"hat":0,"caterpillar":"left","coordinates":{"...</td>
      <td>5</td>
      <td>4030</td>
      <td>0.09</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:35:26.351000+00:00</td>
      <td>{"height":11,"bird_height":4,"correct":false,"...</td>
      <td>6</td>
      <td>4025</td>
      <td>0.12</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:35:26.911000+00:00</td>
      <td>{"hat":0,"caterpillar":"middle","coordinates":...</td>
      <td>7</td>
      <td>4030</td>
      <td>0.13</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:35:28.466000+00:00</td>
      <td>{"height":11,"bird_height":8,"correct":false,"...</td>
      <td>8</td>
      <td>4025</td>
      <td>0.16</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:35:29.778000+00:00</td>
      <td>{"hat":0,"caterpillar":"right","coordinates":{...</td>
      <td>9</td>
      <td>4030</td>
      <td>0.18</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:35:30.999000+00:00</td>
      <td>{"height":11,"bird_height":5,"correct":false,"...</td>
      <td>10</td>
      <td>4025</td>
      <td>0.20</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>1375ccb7</td>
      <td>2019-08-06 05:35:31.002000+00:00</td>
      <td>{"description":"When you want to check to see ...</td>
      <td>11</td>
      <td>3010</td>
      <td>0.20</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:35:31.931000+00:00</td>
      <td>{"hat":0,"caterpillar":"right","coordinates":{...</td>
      <td>12</td>
      <td>4030</td>
      <td>0.21</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:35:32.555000+00:00</td>
      <td>{"height":11,"bird_height":5,"correct":false,"...</td>
      <td>13</td>
      <td>4025</td>
      <td>0.22</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:35:32.881000+00:00</td>
      <td>{"hat":0,"caterpillar":"right","coordinates":{...</td>
      <td>14</td>
      <td>4030</td>
      <td>0.23</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>bdf49a58</td>
      <td>2019-08-06 05:35:33.345000+00:00</td>
      <td>{"description":"When you want to check to see ...</td>
      <td>15</td>
      <td>3110</td>
      <td>0.24</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:35:33.520000+00:00</td>
      <td>{"height":11,"bird_height":5,"correct":false,"...</td>
      <td>16</td>
      <td>4025</td>
      <td>0.24</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:35:34.733000+00:00</td>
      <td>{"hat":0,"caterpillar":"middle","coordinates":...</td>
      <td>17</td>
      <td>4030</td>
      <td>0.26</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:35:37.002000+00:00</td>
      <td>{"height":1,"bird_height":8,"correct":false,"c...</td>
      <td>18</td>
      <td>4025</td>
      <td>0.30</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:35:38.461000+00:00</td>
      <td>{"hat":0,"caterpillar":"middle","coordinates":...</td>
      <td>19</td>
      <td>4030</td>
      <td>0.32</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:35:39.328000+00:00</td>
      <td>{"height":11,"bird_height":8,"correct":false,"...</td>
      <td>20</td>
      <td>4025</td>
      <td>0.34</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:35:41.223000+00:00</td>
      <td>{"hat":0,"caterpillar":"middle","coordinates":...</td>
      <td>21</td>
      <td>4030</td>
      <td>0.37</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:35:41.984000+00:00</td>
      <td>{"height":5,"bird_height":8,"correct":false,"c...</td>
      <td>22</td>
      <td>4025</td>
      <td>0.38</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:35:43.700000+00:00</td>
      <td>{"hat":0,"caterpillar":"right","coordinates":{...</td>
      <td>23</td>
      <td>4030</td>
      <td>0.41</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:35:44.799000+00:00</td>
      <td>{"height":7,"bird_height":5,"correct":false,"c...</td>
      <td>24</td>
      <td>4025</td>
      <td>0.43</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:35:47.512000+00:00</td>
      <td>{"hat":0,"caterpillar":"middle","coordinates":...</td>
      <td>25</td>
      <td>4030</td>
      <td>0.47</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:35:48.604000+00:00</td>
      <td>{"height":8,"bird_height":8,"correct":true,"ca...</td>
      <td>26</td>
      <td>4025</td>
      <td>0.49</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:35:52.707000+00:00</td>
      <td>{"hat":0,"caterpillar":"right","coordinates":{...</td>
      <td>27</td>
      <td>4030</td>
      <td>0.56</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:35:53.290000+00:00</td>
      <td>{"height":3,"bird_height":5,"correct":false,"c...</td>
      <td>28</td>
      <td>4025</td>
      <td>0.57</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>17113b36</td>
      <td>2019-08-06 05:35:54.898000+00:00</td>
      <td>{"correct":false,"caterpillars":[11,8,3],"even...</td>
      <td>29</td>
      <td>4110</td>
      <td>0.60</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>ad2fc29c</td>
      <td>2019-08-06 05:35:54.904000+00:00</td>
      <td>{"description":"Whoops. This caterpillar is to...</td>
      <td>30</td>
      <td>3020</td>
      <td>0.60</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>e37a2b78</td>
      <td>2019-08-06 05:35:57.819000+00:00</td>
      <td>{"description":"Whoops. This caterpillar is to...</td>
      <td>31</td>
      <td>3120</td>
      <td>0.64</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:35:59.789000+00:00</td>
      <td>{"hat":0,"caterpillar":"right","coordinates":{...</td>
      <td>32</td>
      <td>4030</td>
      <td>0.68</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:36:00.572000+00:00</td>
      <td>{"height":11,"bird_height":5,"correct":false,"...</td>
      <td>33</td>
      <td>4025</td>
      <td>0.69</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>a16a373e</td>
      <td>2019-08-06 05:36:01.424000+00:00</td>
      <td>{"coordinates":{"x":541,"y":131,"stage_width":...</td>
      <td>34</td>
      <td>4070</td>
      <td>0.70</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>17113b36</td>
      <td>2019-08-06 05:36:01.927000+00:00</td>
      <td>{"correct":false,"caterpillars":[11,8,11],"eve...</td>
      <td>35</td>
      <td>4110</td>
      <td>0.71</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>ad2fc29c</td>
      <td>2019-08-06 05:36:01.934000+00:00</td>
      <td>{"description":"Uh oh. This caterpillar is too...</td>
      <td>36</td>
      <td>3020</td>
      <td>0.71</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:36:04.378000+00:00</td>
      <td>{"hat":0,"caterpillar":"right","coordinates":{...</td>
      <td>37</td>
      <td>4030</td>
      <td>0.75</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>e37a2b78</td>
      <td>2019-08-06 05:36:04.723000+00:00</td>
      <td>{"description":"Uh oh. This caterpillar is too...</td>
      <td>38</td>
      <td>3120</td>
      <td>0.76</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:36:05.099000+00:00</td>
      <td>{"height":5,"bird_height":5,"correct":true,"ca...</td>
      <td>39</td>
      <td>4025</td>
      <td>0.77</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>17113b36</td>
      <td>2019-08-06 05:36:06.512000+00:00</td>
      <td>{"correct":false,"caterpillars":[11,8,5],"even...</td>
      <td>40</td>
      <td>4110</td>
      <td>0.79</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>ad2fc29c</td>
      <td>2019-08-06 05:36:06.513000+00:00</td>
      <td>{"description":"Uh oh. This caterpillar is too...</td>
      <td>41</td>
      <td>3020</td>
      <td>0.79</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:36:07.717000+00:00</td>
      <td>{"hat":0,"caterpillar":"right","coordinates":{...</td>
      <td>42</td>
      <td>4030</td>
      <td>0.81</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:36:08.516000+00:00</td>
      <td>{"height":7,"bird_height":5,"correct":false,"c...</td>
      <td>43</td>
      <td>4025</td>
      <td>0.82</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>e37a2b78</td>
      <td>2019-08-06 05:36:09.270000+00:00</td>
      <td>{"description":"Uh oh. This caterpillar is too...</td>
      <td>44</td>
      <td>3120</td>
      <td>0.84</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>17113b36</td>
      <td>2019-08-06 05:36:09.739000+00:00</td>
      <td>{"correct":false,"caterpillars":[11,8,7],"even...</td>
      <td>45</td>
      <td>4110</td>
      <td>0.84</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>ad2fc29c</td>
      <td>2019-08-06 05:36:09.740000+00:00</td>
      <td>{"description":"Uh oh. This caterpillar is too...</td>
      <td>46</td>
      <td>3020</td>
      <td>0.84</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>e37a2b78</td>
      <td>2019-08-06 05:36:12.499000+00:00</td>
      <td>{"description":"Uh oh. This caterpillar is too...</td>
      <td>47</td>
      <td>3120</td>
      <td>0.89</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:36:12.858000+00:00</td>
      <td>{"hat":0,"caterpillar":"right","coordinates":{...</td>
      <td>48</td>
      <td>4030</td>
      <td>0.90</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:36:13.474000+00:00</td>
      <td>{"height":4,"bird_height":5,"correct":false,"c...</td>
      <td>49</td>
      <td>4025</td>
      <td>0.91</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>17113b36</td>
      <td>2019-08-06 05:36:13.951000+00:00</td>
      <td>{"correct":false,"caterpillars":[11,8,4],"even...</td>
      <td>50</td>
      <td>4110</td>
      <td>0.91</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>ad2fc29c</td>
      <td>2019-08-06 05:36:13.953000+00:00</td>
      <td>{"description":"Uh oh. This caterpillar is too...</td>
      <td>51</td>
      <td>3020</td>
      <td>0.91</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>e37a2b78</td>
      <td>2019-08-06 05:36:16.710000+00:00</td>
      <td>{"description":"Uh oh. This caterpillar is too...</td>
      <td>52</td>
      <td>3120</td>
      <td>0.96</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>17113b36</td>
      <td>2019-08-06 05:36:17.407000+00:00</td>
      <td>{"correct":false,"caterpillars":[11,8,4],"even...</td>
      <td>53</td>
      <td>4110</td>
      <td>0.97</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>ad2fc29c</td>
      <td>2019-08-06 05:36:17.410000+00:00</td>
      <td>{"description":"Whoops. This caterpillar is to...</td>
      <td>54</td>
      <td>3020</td>
      <td>0.97</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:36:19.154000+00:00</td>
      <td>{"hat":0,"caterpillar":"right","coordinates":{...</td>
      <td>55</td>
      <td>4030</td>
      <td>1.00</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:36:19.661000+00:00</td>
      <td>{"height":2,"bird_height":5,"correct":false,"c...</td>
      <td>56</td>
      <td>4025</td>
      <td>1.01</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>e37a2b78</td>
      <td>2019-08-06 05:36:20.292000+00:00</td>
      <td>{"description":"Whoops. This caterpillar is to...</td>
      <td>57</td>
      <td>3120</td>
      <td>1.02</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>17113b36</td>
      <td>2019-08-06 05:36:21.390000+00:00</td>
      <td>{"correct":false,"caterpillars":[11,8,2],"even...</td>
      <td>58</td>
      <td>4110</td>
      <td>1.04</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>ad2fc29c</td>
      <td>2019-08-06 05:36:21.391000+00:00</td>
      <td>{"description":"Uh oh. This caterpillar is too...</td>
      <td>59</td>
      <td>3020</td>
      <td>1.04</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>e37a2b78</td>
      <td>2019-08-06 05:36:24.154000+00:00</td>
      <td>{"description":"Uh oh. This caterpillar is too...</td>
      <td>60</td>
      <td>3120</td>
      <td>1.08</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>51102b85</td>
      <td>2019-08-06 05:36:25.256000+00:00</td>
      <td>{"hat":0,"caterpillar":"right","coordinates":{...</td>
      <td>61</td>
      <td>4030</td>
      <td>1.10</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>4a4c3d21</td>
      <td>2019-08-06 05:36:25.792000+00:00</td>
      <td>{"height":1,"bird_height":5,"correct":false,"c...</td>
      <td>62</td>
      <td>4025</td>
      <td>1.11</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>17113b36</td>
      <td>2019-08-06 05:36:26.296000+00:00</td>
      <td>{"correct":false,"caterpillars":[11,8,1],"even...</td>
      <td>63</td>
      <td>4110</td>
      <td>1.12</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>ad2fc29c</td>
      <td>2019-08-06 05:36:26.299000+00:00</td>
      <td>{"description":"Whoops. This caterpillar is to...</td>
      <td>64</td>
      <td>3020</td>
      <td>1.12</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>e37a2b78</td>
      <td>2019-08-06 05:36:29.181000+00:00</td>
      <td>{"description":"Whoops. This caterpillar is to...</td>
      <td>65</td>
      <td>3120</td>
      <td>1.17</td>
      <td>Bird Measurer (Assessment)</td>
      <td>TREETOPCITY</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# B. Data Exploration/Visualization

## B1. Bad data

Observe that the accuracy (or equivalently the num_correct) is the same for all events generated in the same assessmnet session:


```python
temp_df = assessments.groupby(['installation_id','game_session'])[['event_count','accuracy']]\
                     .agg({'event_count':'count', 'accuracy':'sum'})
temp_df.head()

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
      <th></th>
      <th>event_count</th>
      <th>accuracy</th>
    </tr>
    <tr>
      <th>installation_id</th>
      <th>game_session</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="top">435871</td>
      <td>6bdf9623adc94d89</td>
      <td>35</td>
      <td>35.0</td>
    </tr>
    <tr>
      <td>77b8ee947eb84b4e</td>
      <td>87</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>901acc108f55a5a1</td>
      <td>48</td>
      <td>48.0</td>
    </tr>
    <tr>
      <td>9501794defd84e4d</td>
      <td>42</td>
      <td>21.0</td>
    </tr>
    <tr>
      <td>a9ef3ecb3d1acc6a</td>
      <td>32</td>
      <td>32.0</td>
    </tr>
  </tbody>
</table>
</div>



Observe that there are no assessment sessions with event_count=1:


```python
np.sum(temp_df.event_count[temp_df.event_count==1])
```




    0



On the other hand, the whole ```train``` dataset has a lot of sessions with event_count=1:


```python
print('Fraction of single-event sessions in train: {0}/{1} = {2}%'\
      .format(
              np.sum(train.event_count[train.event_count==1]),
              len(train),
              round(np.sum(train.event_count[train.event_count==1])/len(train)*100,2)
             )
     )
```

    Fraction of single-event sessions in train: 28338/1000000 = 2.83%



```python
del temp_df
```

Note: game sessions with a single event have an event_code = 2000 which represents the start of the session.

## <font color='green'> IGNORE: game sessions with a single event</font> 


```python
def filter_out_bad_sess(df):
    """Method to select only game sessions with more than one event."""
    
    temp_df = df.groupby(['installation_id','game_session'])[['event_count']].count()

    return df.loc[temp_df[temp_df.event_count>1].index,:]
```


```python
train = filter_out_bad_sess(train)
```

## B2. Visualize cols and accuracies

There are 10 columns:


```python
train.columns.tolist() + ['game_session']
```




    ['event_id',
     'timestamp',
     'event_data',
     'event_count',
     'event_code',
     'game_time',
     'title',
     'type',
     'world',
     'game_session']



Let's start our data exploration with the categorical data.

## <font color='blue'>Categorical Data<font>

### 1) Title


```python
title_val_counts = train['title'].value_counts().sort_values()
_ = title_val_counts.plot(kind='barh',figsize=(20,10))
_ = plt.gcf().suptitle("Title Frequencies", size = 18)
```


![png](../images/kaggle_dsb_figs/output_32_0.png)



```python
pct_corrects = lambda g: round(g.sum()/g.count(),3)
assess_title = assessments.groupby(['title'])[['num_correct']]\
                          .agg(['count','sum', pct_corrects])
```


```python
# Clean df
assess_title.columns = ['num_assessments', 'num_correct', 'pct_correct']
assess_title['pct_incorrect'] =  1 - assess_title.pct_correct
assess_title.index = [title.split('(')[0].strip() for title in assess_title.index]
```


```python
assess_title.loc[:,['pct_correct','pct_incorrect']]\
            .plot.barh(stacked=True,figsize=(10, 6), title='Correct Assessment (by title)')
plt.gca().legend(frameon=True,loc=(1.05,0.9))
plt.gca().axvline(x=0.5, ymin=0, ymax=1, color='black')

del assess_title
```


![png](../images/kaggle_dsb_figs/output_35_0.png)


## <font color='green'> Good cols: ```'title'```</font> 

QUESTION: What pct of activities get completed and reach the assessment phase?

### 2) World


```python
world_val_counts = train['world'].value_counts().sort_values()
_ = world_val_counts.plot(kind='bar',figsize=(8,5), color=my_pal[0])
_ = plt.gcf().suptitle("World Frequencies", size = 18)
ax = plt.gca()
_ = ax.set_xticklabels(ax.get_xticklabels(),rotation=0)
```


![png](output_39_0.png)



```python
assess_world = assessments.groupby(['world'])[['num_correct']]\
                          .agg(['count','sum', pct_corrects])

# Clean df
assess_world.columns = ['num_assessments', 'num_correct', 'pct_correct']
assess_world['pct_incorrect'] =  1 - assess_world.pct_correct
assess_world.index = [title.split('(')[0].strip() for title in assess_world.index]

assess_world.loc[:,['pct_correct','pct_incorrect']]\
            .plot.barh(stacked=True,figsize=(10, 6), title='Correct Assessment (by title)')
plt.gca().legend(frameon=True,loc=(1.05,0.9))
plt.gca().axvline(x=0.5, ymin=0, ymax=1, color='black')

del assess_world
```


![png](../images/kaggle_dsb_figs/output_40_0.png)


## <font color='green'> Good cols: ```'world'```  </font> 

### 3) Timestamps


```python
def plot_assess_vs_timestamps(df, use_col):
    df = df.groupby(use_col)[['num_correct']]\
           .agg(['count','sum', pct_corrects])

    # Clean df
    df.columns = ['num_assessments', 'num_correct', 'pct_correct']
    df['pct_incorrect'] =  1 - df.pct_correct
    
    # Plot accuracies
    df.loc[:,['pct_correct','pct_incorrect']]\
                .plot.barh(stacked=True,figsize=(10, 6), title=f'Correct Assessment (by {use_col})')
    plt.gca().legend(frameon=True,loc=(1.05,0.9))
    plt.gca().axvline(x=0.5, ymin=0, ymax=1, color='black')
    
    
    return "STD of pct_correct for " + use_col + ": {:.2f}".format(df['pct_correct'].std())
```


```python
assessments['hour'] = assessments['timestamp'].dt.hour
assessments['day'] = assessments['timestamp'].dt.day
assessments['weekday'] = assessments['timestamp'].dt.weekday
assessments['month'] = assessments['timestamp'].dt.month
```


```python
plot_assess_vs_timestamps(assessments, use_col='day')
```




    'STD of pct_correct for day: 0.09'




![png](../images/kaggle_dsb_figs/output_45_1.png)


The feature 'hour' has the highest STD (0.14) among: 'day' (0.09), 'weekday' (0.03), 'month' (0.03)

## <font color='green'> Good cols: ```'hour'``` and perhaps ```'day'``` </font> 
### <font color='green'> ADD New cols!</font> 


```python
train['hour'] = train.timestamp.dt.hour
train['day'] = train.timestamp.dt.day
```

## 4) Type


```python
for game_type in set(train.type):
    print(game_type)
    pprint(list(set(train[train.type==game_type].title)))
```

    Game
    ['Crystals Rule',
     'Chow Time',
     'Happy Camel',
     'Leaf Leader',
     'Bubble Bath',
     'Dino Dive',
     'All Star Sorting',
     'Air Show',
     'Pan Balance',
     'Dino Drink',
     'Scrub-A-Dub']
    Assessment
    ['Mushroom Sorter (Assessment)',
     'Cauldron Filler (Assessment)',
     'Bird Measurer (Assessment)',
     'Chest Sorter (Assessment)',
     'Cart Balancer (Assessment)']
    Activity
    ['Chicken Balancer (Activity)',
     'Bottle Filler (Activity)',
     'Egg Dropper (Activity)',
     'Watering Hole (Activity)',
     'Sandcastle Builder (Activity)',
     'Flower Waterer (Activity)',
     'Bug Measurer (Activity)',
     'Fireworks (Activity)']


## <font color='red'>No new information in ```type```! <font>

## 5) Event_code 


```python
print("Fraction of unique event codes: {0}/{1}"\
      .format(len(set(train.event_code)-set(assessments.event_code)),
              len(set(train.event_code)))
     )
```

    Fraction of unique event codes: 20/42



```python
assess_ec = assessments.groupby(['event_code'])[['num_correct']]\
                          .agg(['count','sum', pct_corrects])

# Clean df
assess_ec.columns = ['num_assessments', 'num_correct', 'pct_correct']
assess_ec['pct_incorrect'] =  1 - assess_ec.pct_correct

assess_ec.loc[:,['pct_correct','pct_incorrect']]\
            .plot.barh(stacked=True,figsize=(10, 6), title='Correct Assessment (by event_code)')
plt.gca().legend(frameon=True,loc=(1.05,0.9))
plt.gca().axvline(x=0.5, ymin=0, ymax=1, color='black')

del assess_ec
```


![png](../images/kaggle_dsb_figs/output_54_0.png)


## <font color='green'> Good cols: ```'event_code'``` - values in assessments!!! </font> 

## 6) Event_id


```python
print("Fraction of unique Event IDs: {0}/{1}"\
      .format(len(set(train.event_id)-set(assessments.event_id)),
              len(set(train.event_id)))
     )
```

    Fraction of unique Event IDs: 273/366



```python
assess_ed = assessments.groupby('event_id')[['num_correct']]\
                       .agg(['count','sum', pct_corrects])

# Clean df
assess_ed.columns = ['num_assessments', 'num_correct', 'pct_correct']
assess_ed['pct_incorrect'] =  1 - assess_ed.pct_correct
assess_ed.index = [title.split('(')[0].strip() for title in assess_ed.index]

assess_ed.loc[:,['pct_correct']]\
         .plot(kind='hist' ,bins=30,edgecolor='white',
               figsize=(15,6),color=my_pal[1])

plt.gca().axvline(x=0.5, ymin=0, ymax=1, color='black')
plt.gcf().suptitle('Distribution of Event IDs vs % accuracy',fontsize = 18)


del assess_ed
```


![png](../images/kaggle_dsb_figs/output_58_0.png)


## <font color='blue'>Continuous Data<font>

## 7) Game_session 


```python
def no_unique_sessions(df, cut_off=1):
    
    return len(df[df.event_count>cut_off].reset_index()['game_session'].unique())
    
```


```python
cut_off = 1
uniq_sess_assess = no_unique_sessions(assessments)
uniq_sess_train = no_unique_sessions(train, cut_off=cut_off)
pct_fraction = round(uniq_sess_assess/uniq_sess_train*100,2)

print(f"""Fraction of unique assessment sessions vs train(cut_off): {uniq_sess_assess}/{uniq_sess_train} = {pct_fraction}%""")

```

    Fraction of unique assessment sessions vs train(cut_off): 1798/10994 = 16.35%


## 8) Game_time


```python
assessments.groupby(['installation_id','title'])[['game_time']].sum().unstack()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">game_time</th>
    </tr>
    <tr>
      <th>title</th>
      <th>Bird Measurer (Assessment)</th>
      <th>Cart Balancer (Assessment)</th>
      <th>Cauldron Filler (Assessment)</th>
      <th>Chest Sorter (Assessment)</th>
      <th>Mushroom Sorter (Assessment)</th>
    </tr>
    <tr>
      <th>installation_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>435871</td>
      <td>78.52</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.77</td>
    </tr>
    <tr>
      <td>442770</td>
      <td>366.12</td>
      <td>NaN</td>
      <td>3.10</td>
      <td>NaN</td>
      <td>82.30</td>
    </tr>
    <tr>
      <td>1218646</td>
      <td>13.40</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1904336</td>
      <td>25.14</td>
      <td>2.09</td>
      <td>NaN</td>
      <td>8.53</td>
      <td>31.48</td>
    </tr>
    <tr>
      <td>2252647</td>
      <td>8.40</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2595525</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>199.90</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2996195</td>
      <td>143.47</td>
      <td>11.62</td>
      <td>7.70</td>
      <td>62.32</td>
      <td>24.65</td>
    </tr>
    <tr>
      <td>3371696</td>
      <td>16.20</td>
      <td>14.04</td>
      <td>10.81</td>
      <td>32.10</td>
      <td>8.95</td>
    </tr>
    <tr>
      <td>4989073</td>
      <td>5.23</td>
      <td>NaN</td>
      <td>13.38</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>6505523</td>
      <td>NaN</td>
      <td>6.76</td>
      <td>33.82</td>
      <td>49.94</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>6716296</td>
      <td>NaN</td>
      <td>83.40</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>6885427</td>
      <td>150.83</td>
      <td>NaN</td>
      <td>21.96</td>
      <td>17.07</td>
      <td>53.04</td>
    </tr>
    <tr>
      <td>10542059</td>
      <td>NaN</td>
      <td>835.94</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>10828131</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.12</td>
    </tr>
    <tr>
      <td>11343246</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22.82</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>12179686</td>
      <td>NaN</td>
      <td>1.98</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>13563777</td>
      <td>NaN</td>
      <td>10.30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>14774898</td>
      <td>22.83</td>
      <td>10.90</td>
      <td>37.37</td>
      <td>19.23</td>
      <td>9.83</td>
    </tr>
    <tr>
      <td>15021759</td>
      <td>82.45</td>
      <td>3.60</td>
      <td>22.32</td>
      <td>95.42</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>16418433</td>
      <td>55.45</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.51</td>
      <td>10.70</td>
    </tr>
    <tr>
      <td>16541110</td>
      <td>53.72</td>
      <td>34.89</td>
      <td>28.05</td>
      <td>97.67</td>
      <td>11.76</td>
    </tr>
    <tr>
      <td>17547733</td>
      <td>39.21</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>48.65</td>
    </tr>
    <tr>
      <td>17960722</td>
      <td>10.88</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>22268247</td>
      <td>52.53</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>22404486</td>
      <td>NaN</td>
      <td>18.01</td>
      <td>NaN</td>
      <td>96.75</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>22509236</td>
      <td>NaN</td>
      <td>9.69</td>
      <td>NaN</td>
      <td>32.65</td>
      <td>24.87</td>
    </tr>
    <tr>
      <td>22553105</td>
      <td>NaN</td>
      <td>9.74</td>
      <td>6.22</td>
      <td>87.63</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>23128005</td>
      <td>NaN</td>
      <td>42.89</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>25317668</td>
      <td>60.30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>33.24</td>
    </tr>
    <tr>
      <td>29218592</td>
      <td>55.89</td>
      <td>14.86</td>
      <td>NaN</td>
      <td>77.12</td>
      <td>6.36</td>
    </tr>
    <tr>
      <td>30168051</td>
      <td>NaN</td>
      <td>7.58</td>
      <td>5.74</td>
      <td>8.49</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>30794229</td>
      <td>17.47</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>32970498</td>
      <td>77.83</td>
      <td>1.48</td>
      <td>15.17</td>
      <td>69.73</td>
      <td>8.34</td>
    </tr>
    <tr>
      <td>33203391</td>
      <td>16.17</td>
      <td>8.52</td>
      <td>NaN</td>
      <td>36.96</td>
      <td>85.83</td>
    </tr>
    <tr>
      <td>35750827</td>
      <td>38.04</td>
      <td>191.26</td>
      <td>53.88</td>
      <td>57.37</td>
      <td>62.94</td>
    </tr>
    <tr>
      <td>37093018</td>
      <td>NaN</td>
      <td>2.14</td>
      <td>18.74</td>
      <td>77.95</td>
      <td>42.15</td>
    </tr>
    <tr>
      <td>37505995</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.62</td>
    </tr>
    <tr>
      <td>38341382</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21.21</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>39859689</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.35</td>
      <td>59.71</td>
      <td>10.42</td>
    </tr>
    <tr>
      <td>40777523</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.83</td>
    </tr>
    <tr>
      <td>41630796</td>
      <td>NaN</td>
      <td>5.46</td>
      <td>4.09</td>
      <td>11.19</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>42071743</td>
      <td>60.05</td>
      <td>5.80</td>
      <td>19.80</td>
      <td>35.17</td>
      <td>13.82</td>
    </tr>
    <tr>
      <td>42425794</td>
      <td>NaN</td>
      <td>11.14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>43232884</td>
      <td>74.59</td>
      <td>5.65</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22.32</td>
    </tr>
    <tr>
      <td>44716383</td>
      <td>369.05</td>
      <td>51.77</td>
      <td>13.47</td>
      <td>26.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>45218484</td>
      <td>21.78</td>
      <td>14.89</td>
      <td>6.55</td>
      <td>16.08</td>
      <td>17.97</td>
    </tr>
    <tr>
      <td>46757308</td>
      <td>39.29</td>
      <td>84.41</td>
      <td>27.09</td>
      <td>74.94</td>
      <td>18.11</td>
    </tr>
    <tr>
      <td>47043726</td>
      <td>281.89</td>
      <td>24.53</td>
      <td>13.78</td>
      <td>NaN</td>
      <td>9.17</td>
    </tr>
    <tr>
      <td>47054845</td>
      <td>104.32</td>
      <td>3.50</td>
      <td>38.72</td>
      <td>17.20</td>
      <td>49.66</td>
    </tr>
    <tr>
      <td>48721731</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.79</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>52097976</td>
      <td>3.58</td>
      <td>93.01</td>
      <td>20.16</td>
      <td>30.21</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>53017743</td>
      <td>51.45</td>
      <td>192.98</td>
      <td>29.68</td>
      <td>22.88</td>
      <td>66.52</td>
    </tr>
    <tr>
      <td>54533657</td>
      <td>53.19</td>
      <td>40.82</td>
      <td>14.10</td>
      <td>32.24</td>
      <td>244.40</td>
    </tr>
    <tr>
      <td>55971345</td>
      <td>NaN</td>
      <td>7.80</td>
      <td>NaN</td>
      <td>6.63</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>60126850</td>
      <td>26.92</td>
      <td>2.22</td>
      <td>8.74</td>
      <td>7.50</td>
      <td>44.37</td>
    </tr>
    <tr>
      <td>60705479</td>
      <td>NaN</td>
      <td>9.36</td>
      <td>NaN</td>
      <td>27.73</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>61190150</td>
      <td>NaN</td>
      <td>21.81</td>
      <td>34.20</td>
      <td>73.65</td>
      <td>11.75</td>
    </tr>
    <tr>
      <td>62344165</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>106.39</td>
    </tr>
    <tr>
      <td>62858595</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>164.83</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>63281549</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.85</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>63784333</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>136.04</td>
      <td>9.61</td>
    </tr>
    <tr>
      <td>64052057</td>
      <td>17.48</td>
      <td>42.51</td>
      <td>6.02</td>
      <td>6.53</td>
      <td>21.02</td>
    </tr>
    <tr>
      <td>65242981</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.06</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>66621504</td>
      <td>NaN</td>
      <td>6.23</td>
      <td>NaN</td>
      <td>14.94</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>68009406</td>
      <td>15.80</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11.18</td>
    </tr>
    <tr>
      <td>68398666</td>
      <td>NaN</td>
      <td>7.07</td>
      <td>20.58</td>
      <td>28.92</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>69472177</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.24</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>69853695</td>
      <td>NaN</td>
      <td>183.43</td>
      <td>110.24</td>
      <td>NaN</td>
      <td>12.35</td>
    </tr>
    <tr>
      <td>71204658</td>
      <td>143.09</td>
      <td>6.20</td>
      <td>21.53</td>
      <td>23.66</td>
      <td>19.94</td>
    </tr>
    <tr>
      <td>73491865</td>
      <td>116.54</td>
      <td>2.38</td>
      <td>14.18</td>
      <td>43.83</td>
      <td>20.48</td>
    </tr>
    <tr>
      <td>73544898</td>
      <td>NaN</td>
      <td>154.11</td>
      <td>34.81</td>
      <td>NaN</td>
      <td>37.89</td>
    </tr>
    <tr>
      <td>75018306</td>
      <td>78.10</td>
      <td>12.32</td>
      <td>16.21</td>
      <td>33.75</td>
      <td>26.55</td>
    </tr>
    <tr>
      <td>76954607</td>
      <td>64.63</td>
      <td>26.39</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>77952885</td>
      <td>12.83</td>
      <td>1.42</td>
      <td>4.96</td>
      <td>73.31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>77959805</td>
      <td>9.32</td>
      <td>12.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>109.28</td>
    </tr>
    <tr>
      <td>80089853</td>
      <td>128.55</td>
      <td>8.60</td>
      <td>33.45</td>
      <td>89.53</td>
      <td>108.93</td>
    </tr>
    <tr>
      <td>81231602</td>
      <td>2.92</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.82</td>
    </tr>
    <tr>
      <td>81668782</td>
      <td>22.50</td>
      <td>3.15</td>
      <td>6.45</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>81948971</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>38.97</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>83021711</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.51</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>83407430</td>
      <td>NaN</td>
      <td>6.44</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>83529839</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.72</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>83619426</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>42.28</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>84177929</td>
      <td>182.80</td>
      <td>15.21</td>
      <td>28.12</td>
      <td>NaN</td>
      <td>89.03</td>
    </tr>
    <tr>
      <td>84897028</td>
      <td>NaN</td>
      <td>31.67</td>
      <td>28.16</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>85036929</td>
      <td>53.49</td>
      <td>103.10</td>
      <td>21.56</td>
      <td>911.81</td>
      <td>12.85</td>
    </tr>
    <tr>
      <td>85431492</td>
      <td>101.18</td>
      <td>10.74</td>
      <td>57.01</td>
      <td>1766.37</td>
      <td>31.12</td>
    </tr>
    <tr>
      <td>85706149</td>
      <td>NaN</td>
      <td>139.90</td>
      <td>5.97</td>
      <td>268.54</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>86220607</td>
      <td>NaN</td>
      <td>33.38</td>
      <td>4.72</td>
      <td>36.05</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>87071422</td>
      <td>81.39</td>
      <td>1.80</td>
      <td>15.01</td>
      <td>17.25</td>
      <td>8.79</td>
    </tr>
    <tr>
      <td>87974739</td>
      <td>NaN</td>
      <td>19.20</td>
      <td>4.39</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>92498238</td>
      <td>121.09</td>
      <td>80.53</td>
      <td>29.21</td>
      <td>24.03</td>
      <td>47.68</td>
    </tr>
    <tr>
      <td>94613089</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.93</td>
      <td>11.71</td>
      <td>29.69</td>
    </tr>
    <tr>
      <td>95069641</td>
      <td>NaN</td>
      <td>42.24</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>95525029</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.66</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>96245699</td>
      <td>17.12</td>
      <td>4.97</td>
      <td>NaN</td>
      <td>9.32</td>
      <td>5.26</td>
    </tr>
    <tr>
      <td>101183416</td>
      <td>24.55</td>
      <td>57.56</td>
      <td>5.72</td>
      <td>5.88</td>
      <td>16.58</td>
    </tr>
    <tr>
      <td>104057552</td>
      <td>84.54</td>
      <td>1.51</td>
      <td>23.86</td>
      <td>61.33</td>
      <td>18.81</td>
    </tr>
    <tr>
      <td>104072703</td>
      <td>NaN</td>
      <td>17.13</td>
      <td>18.34</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>104210994</td>
      <td>NaN</td>
      <td>1.57</td>
      <td>19.02</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>104742402</td>
      <td>1543.98</td>
      <td>393.46</td>
      <td>163.09</td>
      <td>936.49</td>
      <td>216.31</td>
    </tr>
    <tr>
      <td>105010250</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.44</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>105078037</td>
      <td>34.58</td>
      <td>NaN</td>
      <td>14.73</td>
      <td>NaN</td>
      <td>5.46</td>
    </tr>
    <tr>
      <td>105653440</td>
      <td>NaN</td>
      <td>58.51</td>
      <td>41.97</td>
      <td>NaN</td>
      <td>8.72</td>
    </tr>
    <tr>
      <td>109594399</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>43.86</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>110125329</td>
      <td>NaN</td>
      <td>6.27</td>
      <td>5.05</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>111599264</td>
      <td>20.62</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>40.54</td>
    </tr>
    <tr>
      <td>114458645</td>
      <td>NaN</td>
      <td>4.49</td>
      <td>18.83</td>
      <td>42.47</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>115778531</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.33</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>116465812</td>
      <td>54.73</td>
      <td>NaN</td>
      <td>10.24</td>
      <td>NaN</td>
      <td>6.87</td>
    </tr>
    <tr>
      <td>120202668</td>
      <td>221.73</td>
      <td>117.27</td>
      <td>84.84</td>
      <td>154.01</td>
      <td>17.32</td>
    </tr>
    <tr>
      <td>122374181</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.11</td>
    </tr>
    <tr>
      <td>123517666</td>
      <td>285.23</td>
      <td>19.56</td>
      <td>NaN</td>
      <td>84.44</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>124455131</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.30</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>124875698</td>
      <td>26.56</td>
      <td>NaN</td>
      <td>17.39</td>
      <td>NaN</td>
      <td>12.24</td>
    </tr>
    <tr>
      <td>124981832</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.11</td>
    </tr>
    <tr>
      <td>125516956</td>
      <td>NaN</td>
      <td>27.67</td>
      <td>NaN</td>
      <td>7.57</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>126227536</td>
      <td>30.39</td>
      <td>1.79</td>
      <td>9.43</td>
      <td>65.83</td>
      <td>14.70</td>
    </tr>
    <tr>
      <td>127016711</td>
      <td>43.22</td>
      <td>NaN</td>
      <td>2.16</td>
      <td>13.89</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>127090846</td>
      <td>32.65</td>
      <td>NaN</td>
      <td>5.19</td>
      <td>NaN</td>
      <td>10.21</td>
    </tr>
    <tr>
      <td>128866967</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>178.17</td>
      <td>NaN</td>
      <td>9.93</td>
    </tr>
    <tr>
      <td>129235991</td>
      <td>9.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.01</td>
    </tr>
    <tr>
      <td>129925710</td>
      <td>23.80</td>
      <td>4.94</td>
      <td>NaN</td>
      <td>8.39</td>
      <td>6.35</td>
    </tr>
    <tr>
      <td>130176008</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>158.84</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>133148658</td>
      <td>NaN</td>
      <td>8.72</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>134620418</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.07</td>
    </tr>
    <tr>
      <td>135239305</td>
      <td>NaN</td>
      <td>26.10</td>
      <td>NaN</td>
      <td>397.13</td>
      <td>23.69</td>
    </tr>
    <tr>
      <td>140827042</td>
      <td>22.90</td>
      <td>23.56</td>
      <td>5.21</td>
      <td>10.63</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>143437678</td>
      <td>24.23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.30</td>
    </tr>
    <tr>
      <td>144210952</td>
      <td>227.26</td>
      <td>145.30</td>
      <td>3091.45</td>
      <td>160.86</td>
      <td>1325.39</td>
    </tr>
    <tr>
      <td>144994902</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.83</td>
    </tr>
    <tr>
      <td>145129804</td>
      <td>NaN</td>
      <td>2.85</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.27</td>
    </tr>
    <tr>
      <td>148179796</td>
      <td>17.20</td>
      <td>32.98</td>
      <td>12.54</td>
      <td>33.43</td>
      <td>9.14</td>
    </tr>
    <tr>
      <td>148839632</td>
      <td>NaN</td>
      <td>9.79</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>148873116</td>
      <td>NaN</td>
      <td>3.83</td>
      <td>NaN</td>
      <td>30.67</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>149275878</td>
      <td>NaN</td>
      <td>8.94</td>
      <td>NaN</td>
      <td>784.65</td>
      <td>10.64</td>
    </tr>
    <tr>
      <td>149873966</td>
      <td>NaN</td>
      <td>2.75</td>
      <td>NaN</td>
      <td>158.09</td>
      <td>27.46</td>
    </tr>
    <tr>
      <td>152916456</td>
      <td>NaN</td>
      <td>1.43</td>
      <td>10.95</td>
      <td>75.92</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>153059497</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.42</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>153467579</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22.88</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>153522610</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>108.69</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>153706856</td>
      <td>580.18</td>
      <td>125.88</td>
      <td>180.96</td>
      <td>232.90</td>
      <td>130.31</td>
    </tr>
    <tr>
      <td>154708203</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21.46</td>
    </tr>
    <tr>
      <td>161102520</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.65</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>161621895</td>
      <td>NaN</td>
      <td>10.19</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>163313038</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.07</td>
      <td>NaN</td>
      <td>18.94</td>
    </tr>
    <tr>
      <td>164935072</td>
      <td>45.78</td>
      <td>33.41</td>
      <td>9.04</td>
      <td>14.30</td>
      <td>17.08</td>
    </tr>
    <tr>
      <td>167748617</td>
      <td>NaN</td>
      <td>13.24</td>
      <td>4.82</td>
      <td>8.43</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>167816183</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.31</td>
    </tr>
    <tr>
      <td>168384458</td>
      <td>15.74</td>
      <td>15.64</td>
      <td>4.98</td>
      <td>29.56</td>
      <td>15.51</td>
    </tr>
    <tr>
      <td>168497528</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.98</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>170650177</td>
      <td>NaN</td>
      <td>5.07</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.18</td>
    </tr>
    <tr>
      <td>170997512</td>
      <td>120.12</td>
      <td>58.14</td>
      <td>34.86</td>
      <td>24.21</td>
      <td>26.63</td>
    </tr>
    <tr>
      <td>171469918</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.15</td>
    </tr>
    <tr>
      <td>172348953</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.31</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>177262668</td>
      <td>474.99</td>
      <td>108.05</td>
      <td>236.96</td>
      <td>62.85</td>
      <td>74.54</td>
    </tr>
    <tr>
      <td>179542587</td>
      <td>NaN</td>
      <td>45.46</td>
      <td>25.46</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>185622262</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.23</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>187689187</td>
      <td>25.09</td>
      <td>NaN</td>
      <td>35.01</td>
      <td>60.41</td>
      <td>8.87</td>
    </tr>
    <tr>
      <td>189921909</td>
      <td>19.10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.61</td>
    </tr>
    <tr>
      <td>190138760</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>71.77</td>
    </tr>
    <tr>
      <td>191719653</td>
      <td>21.62</td>
      <td>36.71</td>
      <td>NaN</td>
      <td>10.98</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>192598301</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.20</td>
    </tr>
    <tr>
      <td>192983703</td>
      <td>26.36</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.87</td>
    </tr>
    <tr>
      <td>194545066</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.40</td>
    </tr>
    <tr>
      <td>194900734</td>
      <td>NaN</td>
      <td>11.39</td>
      <td>23.03</td>
      <td>NaN</td>
      <td>34.18</td>
    </tr>
    <tr>
      <td>196328319</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.26</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>196638448</td>
      <td>NaN</td>
      <td>4.42</td>
      <td>NaN</td>
      <td>30.11</td>
      <td>23.26</td>
    </tr>
    <tr>
      <td>196997924</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.27</td>
    </tr>
    <tr>
      <td>199496302</td>
      <td>10.51</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.80</td>
    </tr>
    <tr>
      <td>199766464</td>
      <td>NaN</td>
      <td>19.67</td>
      <td>5.85</td>
      <td>30.46</td>
      <td>67.92</td>
    </tr>
    <tr>
      <td>200104091</td>
      <td>21.73</td>
      <td>NaN</td>
      <td>6.49</td>
      <td>13.91</td>
      <td>18.96</td>
    </tr>
    <tr>
      <td>200890985</td>
      <td>NaN</td>
      <td>2.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>202499750</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.74</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>206986067</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.67</td>
      <td>74.87</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>212467522</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>52.57</td>
    </tr>
    <tr>
      <td>212894168</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.05</td>
    </tr>
    <tr>
      <td>213213766</td>
      <td>15.29</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.69</td>
    </tr>
    <tr>
      <td>214029586</td>
      <td>NaN</td>
      <td>2.57</td>
      <td>NaN</td>
      <td>33.54</td>
      <td>7.85</td>
    </tr>
    <tr>
      <td>214082119</td>
      <td>16.53</td>
      <td>7.87</td>
      <td>12.37</td>
      <td>80.02</td>
      <td>9.96</td>
    </tr>
    <tr>
      <td>214982529</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.52</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>216752406</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.73</td>
    </tr>
    <tr>
      <td>216797987</td>
      <td>NaN</td>
      <td>1237.60</td>
      <td>2.56</td>
      <td>10.93</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>217227836</td>
      <td>101.12</td>
      <td>9.71</td>
      <td>59.30</td>
      <td>51.00</td>
      <td>26.66</td>
    </tr>
    <tr>
      <td>218754435</td>
      <td>26.74</td>
      <td>NaN</td>
      <td>13.70</td>
      <td>53.66</td>
      <td>6.72</td>
    </tr>
    <tr>
      <td>220398112</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.96</td>
    </tr>
    <tr>
      <td>224003587</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.27</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>224008798</td>
      <td>415.87</td>
      <td>79.53</td>
      <td>94.83</td>
      <td>96.68</td>
      <td>56.07</td>
    </tr>
    <tr>
      <td>226139830</td>
      <td>413.48</td>
      <td>67.64</td>
      <td>71.16</td>
      <td>102.13</td>
      <td>91.61</td>
    </tr>
    <tr>
      <td>227335874</td>
      <td>140.90</td>
      <td>12.69</td>
      <td>9.32</td>
      <td>NaN</td>
      <td>22.05</td>
    </tr>
    <tr>
      <td>228393170</td>
      <td>NaN</td>
      <td>2.10</td>
      <td>15.79</td>
      <td>26.37</td>
      <td>12.64</td>
    </tr>
    <tr>
      <td>229867697</td>
      <td>19.86</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.11</td>
    </tr>
    <tr>
      <td>230281147</td>
      <td>33.68</td>
      <td>16.96</td>
      <td>23.96</td>
      <td>21.28</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>231581190</td>
      <td>NaN</td>
      <td>2.58</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>232107973</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>38.08</td>
      <td>NaN</td>
      <td>55.91</td>
    </tr>
    <tr>
      <td>235075597</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.20</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>235320177</td>
      <td>NaN</td>
      <td>190.46</td>
      <td>37.06</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>235550290</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.08</td>
    </tr>
    <tr>
      <td>236259055</td>
      <td>183.15</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20.69</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>238156429</td>
      <td>NaN</td>
      <td>86.73</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>242017138</td>
      <td>13.37</td>
      <td>12.13</td>
      <td>9.36</td>
      <td>14.27</td>
      <td>34.61</td>
    </tr>
    <tr>
      <td>242199361</td>
      <td>NaN</td>
      <td>6.41</td>
      <td>6.32</td>
      <td>9.96</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>242277852</td>
      <td>230.17</td>
      <td>15.36</td>
      <td>131.62</td>
      <td>56.82</td>
      <td>100.33</td>
    </tr>
    <tr>
      <td>242821744</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>200.17</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>245765162</td>
      <td>76.36</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.39</td>
    </tr>
    <tr>
      <td>248063880</td>
      <td>73.10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>249073277</td>
      <td>31.42</td>
      <td>4.63</td>
      <td>NaN</td>
      <td>18.03</td>
      <td>25.70</td>
    </tr>
    <tr>
      <td>250962553</td>
      <td>11.73</td>
      <td>NaN</td>
      <td>7.19</td>
      <td>NaN</td>
      <td>6.91</td>
    </tr>
    <tr>
      <td>251013805</td>
      <td>NaN</td>
      <td>4.56</td>
      <td>NaN</td>
      <td>159.85</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>252174650</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.61</td>
      <td>NaN</td>
      <td>30.15</td>
    </tr>
    <tr>
      <td>252622660</td>
      <td>472.88</td>
      <td>16.47</td>
      <td>45.42</td>
      <td>134.69</td>
      <td>212.77</td>
    </tr>
    <tr>
      <td>252829728</td>
      <td>20.46</td>
      <td>15.83</td>
      <td>16.49</td>
      <td>100.18</td>
      <td>8.96</td>
    </tr>
    <tr>
      <td>254682036</td>
      <td>NaN</td>
      <td>7.02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>255004820</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.23</td>
      <td>NaN</td>
      <td>17.14</td>
    </tr>
    <tr>
      <td>255195144</td>
      <td>127.45</td>
      <td>NaN</td>
      <td>14.72</td>
      <td>NaN</td>
      <td>8.88</td>
    </tr>
    <tr>
      <td>256372870</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.43</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>257330872</td>
      <td>NaN</td>
      <td>70.27</td>
      <td>22.56</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>257634555</td>
      <td>73.47</td>
      <td>19.95</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.76</td>
    </tr>
    <tr>
      <td>262319136</td>
      <td>12.83</td>
      <td>6.54</td>
      <td>NaN</td>
      <td>20.87</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>262401502</td>
      <td>NaN</td>
      <td>5.37</td>
      <td>NaN</td>
      <td>75.59</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>262554871</td>
      <td>NaN</td>
      <td>19.77</td>
      <td>NaN</td>
      <td>79.76</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>263997382</td>
      <td>89.01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.94</td>
    </tr>
    <tr>
      <td>266167798</td>
      <td>52.74</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>268547657</td>
      <td>22.84</td>
      <td>NaN</td>
      <td>12.22</td>
      <td>NaN</td>
      <td>21.91</td>
    </tr>
    <tr>
      <td>269793452</td>
      <td>NaN</td>
      <td>18.89</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>270049044</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.04</td>
    </tr>
    <tr>
      <td>270681648</td>
      <td>42.72</td>
      <td>690.64</td>
      <td>26.20</td>
      <td>190.71</td>
      <td>19.87</td>
    </tr>
    <tr>
      <td>271807826</td>
      <td>NaN</td>
      <td>15.31</td>
      <td>46.33</td>
      <td>3.22</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>274895920</td>
      <td>2.36</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>275988630</td>
      <td>98.30</td>
      <td>5.75</td>
      <td>21.16</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>276600280</td>
      <td>3.16</td>
      <td>NaN</td>
      <td>35.23</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>276721417</td>
      <td>NaN</td>
      <td>4.31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>277230761</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>245.30</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>277993595</td>
      <td>100.58</td>
      <td>1.71</td>
      <td>7.14</td>
      <td>18.52</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>278444632</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.73</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>278547406</td>
      <td>34.47</td>
      <td>14.87</td>
      <td>16.74</td>
      <td>254.09</td>
      <td>8.54</td>
    </tr>
    <tr>
      <td>279726728</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>124.63</td>
    </tr>
    <tr>
      <td>280516976</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.66</td>
      <td>NaN</td>
      <td>22.64</td>
    </tr>
    <tr>
      <td>280583486</td>
      <td>NaN</td>
      <td>46.68</td>
      <td>NaN</td>
      <td>22.54</td>
      <td>29.02</td>
    </tr>
    <tr>
      <td>284306016</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20.66</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>284671698</td>
      <td>NaN</td>
      <td>1.97</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>285551360</td>
      <td>10.56</td>
      <td>6.84</td>
      <td>6.50</td>
      <td>19.86</td>
      <td>13.50</td>
    </tr>
    <tr>
      <td>286107200</td>
      <td>NaN</td>
      <td>7.15</td>
      <td>13.07</td>
      <td>26.18</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>287566264</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.98</td>
    </tr>
    <tr>
      <td>288701968</td>
      <td>392.55</td>
      <td>3.38</td>
      <td>NaN</td>
      <td>110.81</td>
      <td>28.58</td>
    </tr>
    <tr>
      <td>289304141</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.43</td>
      <td>NaN</td>
      <td>27.26</td>
    </tr>
    <tr>
      <td>290088476</td>
      <td>NaN</td>
      <td>13.51</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>290508392</td>
      <td>21.08</td>
      <td>7.16</td>
      <td>13.78</td>
      <td>48.54</td>
      <td>11.16</td>
    </tr>
    <tr>
      <td>291280844</td>
      <td>130.22</td>
      <td>14.29</td>
      <td>NaN</td>
      <td>53.92</td>
      <td>32.88</td>
    </tr>
    <tr>
      <td>293232723</td>
      <td>48.66</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>294321756</td>
      <td>NaN</td>
      <td>222.61</td>
      <td>NaN</td>
      <td>135.40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>294824225</td>
      <td>NaN</td>
      <td>114.44</td>
      <td>23.61</td>
      <td>251.05</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>295098230</td>
      <td>4.70</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.06</td>
    </tr>
    <tr>
      <td>297514538</td>
      <td>13.28</td>
      <td>18.68</td>
      <td>22.00</td>
      <td>80.92</td>
      <td>18.13</td>
    </tr>
    <tr>
      <td>297568758</td>
      <td>246.28</td>
      <td>18.21</td>
      <td>NaN</td>
      <td>281.86</td>
      <td>51.02</td>
    </tr>
    <tr>
      <td>297658779</td>
      <td>59.04</td>
      <td>10.13</td>
      <td>NaN</td>
      <td>27.10</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>297924326</td>
      <td>NaN</td>
      <td>7.97</td>
      <td>NaN</td>
      <td>11.52</td>
      <td>9.21</td>
    </tr>
    <tr>
      <td>298301734</td>
      <td>28.25</td>
      <td>35.41</td>
      <td>11.69</td>
      <td>81.49</td>
      <td>14.59</td>
    </tr>
    <tr>
      <td>300715607</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.52</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>302402353</td>
      <td>380.39</td>
      <td>NaN</td>
      <td>12.07</td>
      <td>NaN</td>
      <td>7.07</td>
    </tr>
    <tr>
      <td>302996410</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.99</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>305491966</td>
      <td>NaN</td>
      <td>32.63</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>305807521</td>
      <td>350.28</td>
      <td>64.97</td>
      <td>333.71</td>
      <td>256.10</td>
      <td>278.39</td>
    </tr>
    <tr>
      <td>306966525</td>
      <td>76.45</td>
      <td>27.92</td>
      <td>13.23</td>
      <td>49.14</td>
      <td>20.59</td>
    </tr>
    <tr>
      <td>310011451</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>96.29</td>
      <td>NaN</td>
      <td>49.68</td>
    </tr>
    <tr>
      <td>310934389</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.27</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>311212217</td>
      <td>55.58</td>
      <td>2.86</td>
      <td>NaN</td>
      <td>5.62</td>
      <td>29.94</td>
    </tr>
    <tr>
      <td>312140185</td>
      <td>13.74</td>
      <td>8.61</td>
      <td>6.71</td>
      <td>62.60</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>313107294</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>66.87</td>
    </tr>
    <tr>
      <td>313124092</td>
      <td>185.04</td>
      <td>15.05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.67</td>
    </tr>
    <tr>
      <td>314048916</td>
      <td>94.64</td>
      <td>58.76</td>
      <td>87.19</td>
      <td>78.77</td>
      <td>23.15</td>
    </tr>
    <tr>
      <td>315148871</td>
      <td>129.62</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.21</td>
    </tr>
    <tr>
      <td>315439869</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.77</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>316219335</td>
      <td>17.39</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>131.01</td>
      <td>7.34</td>
    </tr>
    <tr>
      <td>319422608</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.99</td>
      <td>NaN</td>
      <td>44.15</td>
    </tr>
    <tr>
      <td>320556311</td>
      <td>21.94</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.91</td>
    </tr>
    <tr>
      <td>322296085</td>
      <td>103.94</td>
      <td>106.22</td>
      <td>11.56</td>
      <td>13.81</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>323327967</td>
      <td>14.58</td>
      <td>15.47</td>
      <td>11.34</td>
      <td>21.39</td>
      <td>47.13</td>
    </tr>
    <tr>
      <td>324029935</td>
      <td>30.12</td>
      <td>26.16</td>
      <td>NaN</td>
      <td>13.15</td>
      <td>11.10</td>
    </tr>
    <tr>
      <td>325410202</td>
      <td>114.17</td>
      <td>7.17</td>
      <td>9.93</td>
      <td>NaN</td>
      <td>39.42</td>
    </tr>
    <tr>
      <td>325436369</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.25</td>
    </tr>
    <tr>
      <td>327079023</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.49</td>
      <td>NaN</td>
      <td>14.63</td>
    </tr>
    <tr>
      <td>327532551</td>
      <td>54.37</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.37</td>
    </tr>
    <tr>
      <td>328241980</td>
      <td>963.76</td>
      <td>28.52</td>
      <td>NaN</td>
      <td>72.27</td>
      <td>90.28</td>
    </tr>
    <tr>
      <td>329073920</td>
      <td>9.46</td>
      <td>45.38</td>
      <td>14.42</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>331507462</td>
      <td>NaN</td>
      <td>18.34</td>
      <td>NaN</td>
      <td>21.49</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>333772141</td>
      <td>21.93</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>132.03</td>
    </tr>
    <tr>
      <td>333923208</td>
      <td>61.93</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>334845115</td>
      <td>NaN</td>
      <td>48.69</td>
      <td>NaN</td>
      <td>90.74</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>336628043</td>
      <td>2.73</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>867.17</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>336867215</td>
      <td>31.28</td>
      <td>11.01</td>
      <td>17.49</td>
      <td>54.66</td>
      <td>12.28</td>
    </tr>
    <tr>
      <td>337689551</td>
      <td>33.02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.82</td>
    </tr>
    <tr>
      <td>338793195</td>
      <td>NaN</td>
      <td>97.54</td>
      <td>NaN</td>
      <td>189.09</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>339081310</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20.55</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>340545413</td>
      <td>15.21</td>
      <td>NaN</td>
      <td>8.75</td>
      <td>NaN</td>
      <td>19.13</td>
    </tr>
    <tr>
      <td>343101975</td>
      <td>19.40</td>
      <td>2.09</td>
      <td>4.82</td>
      <td>7.33</td>
      <td>8.70</td>
    </tr>
    <tr>
      <td>343651168</td>
      <td>123.83</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.04</td>
    </tr>
    <tr>
      <td>344426364</td>
      <td>15.67</td>
      <td>105.89</td>
      <td>32.10</td>
      <td>10.10</td>
      <td>15.66</td>
    </tr>
    <tr>
      <td>347260004</td>
      <td>129.37</td>
      <td>45.56</td>
      <td>32.97</td>
      <td>87.05</td>
      <td>29.03</td>
    </tr>
    <tr>
      <td>348184437</td>
      <td>53.97</td>
      <td>1.92</td>
      <td>7.57</td>
      <td>22.85</td>
      <td>28.04</td>
    </tr>
    <tr>
      <td>348464122</td>
      <td>92.29</td>
      <td>49.81</td>
      <td>43.56</td>
      <td>37.83</td>
      <td>51.16</td>
    </tr>
    <tr>
      <td>350469054</td>
      <td>108.19</td>
      <td>8.10</td>
      <td>22.49</td>
      <td>84.30</td>
      <td>19.04</td>
    </tr>
    <tr>
      <td>350619874</td>
      <td>NaN</td>
      <td>25.46</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>350710551</td>
      <td>NaN</td>
      <td>4.91</td>
      <td>6.66</td>
      <td>256.21</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>352321237</td>
      <td>NaN</td>
      <td>7.32</td>
      <td>38.42</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>356324220</td>
      <td>NaN</td>
      <td>16.22</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>357854297</td>
      <td>21.72</td>
      <td>1.52</td>
      <td>5.04</td>
      <td>NaN</td>
      <td>26.36</td>
    </tr>
    <tr>
      <td>360209609</td>
      <td>NaN</td>
      <td>23.89</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>362620330</td>
      <td>NaN</td>
      <td>66.81</td>
      <td>NaN</td>
      <td>18.32</td>
      <td>38.88</td>
    </tr>
    <tr>
      <td>364399287</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.30</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>365737009</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32.21</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>366754393</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.04</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>369286588</td>
      <td>NaN</td>
      <td>4.22</td>
      <td>98.88</td>
      <td>13.62</td>
      <td>48.37</td>
    </tr>
    <tr>
      <td>370923228</td>
      <td>36.05</td>
      <td>507.72</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>58.37</td>
    </tr>
    <tr>
      <td>371296877</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.61</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>371681940</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.60</td>
      <td>NaN</td>
      <td>29.67</td>
    </tr>
    <tr>
      <td>372517985</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.92</td>
    </tr>
    <tr>
      <td>374995552</td>
      <td>31.22</td>
      <td>2.90</td>
      <td>7.85</td>
      <td>37.12</td>
      <td>192.19</td>
    </tr>
    <tr>
      <td>377356859</td>
      <td>33.65</td>
      <td>1.96</td>
      <td>9.66</td>
      <td>116.41</td>
      <td>13.04</td>
    </tr>
    <tr>
      <td>378598818</td>
      <td>NaN</td>
      <td>8.89</td>
      <td>35.65</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>379701338</td>
      <td>25.04</td>
      <td>2.84</td>
      <td>10.41</td>
      <td>NaN</td>
      <td>43.79</td>
    </tr>
    <tr>
      <td>379748121</td>
      <td>46.36</td>
      <td>NaN</td>
      <td>8.96</td>
      <td>104.27</td>
      <td>6.96</td>
    </tr>
    <tr>
      <td>381266657</td>
      <td>20.54</td>
      <td>14.71</td>
      <td>35.51</td>
      <td>23.77</td>
      <td>23.80</td>
    </tr>
    <tr>
      <td>383054822</td>
      <td>32.96</td>
      <td>5.45</td>
      <td>3.49</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>383661443</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.50</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>383808598</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.78</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>383973423</td>
      <td>81.13</td>
      <td>NaN</td>
      <td>74.71</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>384241357</td>
      <td>34.32</td>
      <td>8.57</td>
      <td>14.48</td>
      <td>32.06</td>
      <td>7.57</td>
    </tr>
    <tr>
      <td>384534025</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>64.82</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>384863779</td>
      <td>NaN</td>
      <td>5.15</td>
      <td>NaN</td>
      <td>6.73</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>387022940</td>
      <td>12.79</td>
      <td>10.85</td>
      <td>24.78</td>
      <td>12.12</td>
      <td>15.98</td>
    </tr>
    <tr>
      <td>388935889</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.77</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>389377620</td>
      <td>17.26</td>
      <td>NaN</td>
      <td>5.59</td>
      <td>NaN</td>
      <td>5.51</td>
    </tr>
    <tr>
      <td>391305209</td>
      <td>NaN</td>
      <td>181.22</td>
      <td>12.73</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## 9) Event_count


```python
assess_ec = assessments.groupby('event_count')[['num_correct']]\
                       .agg(['count','sum', pct_corrects])

# Clean df
assess_ec.columns = ['num_assessments', 'num_correct', 'pct_correct']
assess_ec['pct_incorrect'] =  1 - assess_ec.pct_correct

assess_ec.loc[:,['pct_correct']]\
         .plot(kind='hist' ,bins=30,edgecolor='white',
               figsize=(15,6),color=my_pal[1])

plt.gca().axvline(x=0.5, ymin=0, ymax=1, color='black')
plt.gcf().suptitle('Distribution of Event Counts vs % accuracy',fontsize = 18)


del assess_ec
```


![png](../images/kaggle_dsb_figs/output_66_0.png)



```python
assessments.groupby(['installation_id'])[['event_count']].sum().unstack()
```




                 installation_id
    event_count  435871               7065
                 442770              27186
                 1218646               820
                 1904336              4635
                 2252647               703
                 2595525              9180
                 2996195             13586
                 3371696              5177
                 4989073              1249
                 6505523              4879
                 6716296              4699
                 6885427             16509
                 10542059             6177
                 10828131              820
                 11343246             1540
                 12179686              153
                 13563777              465
                 14774898             7058
                 15021759            13090
                 16418433             3679
                 16541110            16634
                 17547733             7071
                 17960722              465
                 22268247             2016
                 22404486             7666
                 22509236             2728
                 22553105             6026
                 23128005             2080
                 25317668             5254
                 29218592             8366
                 30168051             1607
                 30794229             1711
                 32970498             9122
                 33203391             4774
                 35750827            35730
                 37093018             6708
                 37505995             1683
                 38341382             1431
                 39859689             6160
                 40777523              820
                 41630796             1498
                 42071743            10987
                 42425794              699
                 43232884             6044
                 44716383            21763
                 45218484             5002
                 46757308            15841
                 47043726             4879
                 47054845            11031
                 48721731             2404
                 52097976             9446
                 53017743            16366
                 54533657            13841
                 55971345             1156
                 60126850             6266
                 60705479             2101
                 61190150             6702
                 62344165             5505
                 62858595             9817
                 63281549              946
                 63784333             9598
                 64052057             7967
                 65242981              351
                 66621504             1296
                 68009406             1491
                 68398666             4232
                 69472177              496
                 69853695            14071
                 71204658            13735
                 73491865            12851
                 73544898            18358
                 75018306             9748
                 76954607             4234
                 77952885             7087
                 77959805             9647
                 80089853            19898
                 81231602             1221
                 81668782             1825
                 81948971             3486
                 83021711             1771
                 83407430              291
                 83529839              528
                 83619426             3081
                 84177929            22326
                 84897028             2266
                 85036929            83351
                 85431492            33592
                 85706149            14642
                 86220607             4904
                 87071422             5249
                 87974739             1348
                 92498238            18668
                 94613089             4015
                 95069641             2434
                 95525029              595
                 96245699             3023
                 101183416            7123
                 104057552           13546
                 104072703            1726
                 104210994            1582
                 104742402          139427
                 105010250             703
                 105078037            3388
                 105653440            6269
                 109594399            3240
                 110125329             812
                 111599264            3684
                 114458645            3260
                 115778531             171
                 116465812            4036
                 120202668           38937
                 122374181             780
                 123517666           26742
                 124455131             406
                 124875698            3767
                 124981832             780
                 125516956            1702
                 126227536            9335
                 127016711            3187
                 127090846            3742
                 128866967            5226
                 129235991            2436
                 129925710            1943
                 130176008           12561
                 133148658             496
                 134620418            1128
                 135239305           32797
                 140827042            3879
                 143437678            1941
                 144210952          448618
                 144994902             741
                 145129804            1010
                 148179796            7650
                 148839632             561
                 148873116            2838
                 149275878           37315
                 149873966            8992
                 152916456            5902
                 153059497             435
                 153467579            1275
                 153522610            4656
                 153706856           90975
                 154708203            1953
                 161102520             378
                 161621895             627
                 163313038            2061
                 164935072            8738
                 167748617            1612
                 167816183             666
                 168384458            4683
                 168497528            1225
                 170650177            1401
                 170997512           18418
                 171469918             741
                 172348953             465
                 177262668           45302
                 179542587            4107
                 185622262             378
                 187689187            6784
                 189921909            1870
                 190138760            6391
                 191719653            4817
                 192598301            1959
                 192983703            1620
                 194545066            3160
                 194900734            3686
                 196328319             435
                 196638448            2402
                 196997924            1758
                 199496302            2320
                 199766464            9670
                 200104091            2458
                 200890985              91
                 202499750             528
                 206986067            3291
                 212467522            4753
                 212894168             595
                 213213766            1123
                 214029586            2544
                 214082119            7710
                 214982529             780
                 216752406             703
                 216797987           14244
                 217227836           14124
                 218754435            7986
                 220398112             861
                 224003587             253
                 224008798           50505
                 226139830           47274
                 227335874            9729
                 228393170            4492
                 229867697            2856
                 230281147            4742
                 231581190              91
                 232107973            5833
                 235075597             528
                 235320177           10041
                 235550290             820
                 236259055           10591
                 238156429            1681
                 242017138            5312
                 242199361            1381
                 242277852           32129
                 242821744           12090
                 245765162            5060
                 248063880            2850
                 249073277            5369
                 250962553            2017
                 251013805            8914
                 252174650            2991
                 252622660           49249
                 252829728           10221
                 254682036             820
                 255004820            1708
                 255195144            7682
                 256372870            2926
                 257330872            6780
                 257634555            3854
                 262319136            2299
                 262401502            2545
                 262554871            7636
                 263997382            4984
                 266167798            2926
                 268547657            3459
                 269793452            1035
                 270049044             666
                 270681648           20067
                 271807826            7128
                 274895920             351
                 275988630            8284
                 276600280            2350
                 276721417             325
                 277230761           15051
                 277993595            8711
                 278444632            2016
                 278547406           16432
                 279726728            2145
                 280516976            3072
                 280583486            6585
                 284306016             993
                 284671698              91
                 285551360            4107
                 286107200            3502
                 287566264             528
                 288701968           23137
                 289304141            3034
                 290088476            1035
                 290508392            6027
                 291280844           16246
                 293232723            2926
                 294321756           23693
                 294824225           19394
                 295098230            1406
                 297514538            8184
                 297568758           31150
                 297658779            4264
                 297924326            2280
                 298301734           12139
                 300715607            1176
                 302402353           23002
                 302996410             435
                 305491966            1908
                 305807521           70420
                 306966525            9054
                 310011451            8639
                 310934389             435
                 311212217            6268
                 312140185            5684
                 313107294            3160
                 313124092            8920
                 314048916           17274
                 315148871            7008
                 315439869             649
                 316219335            9562
                 319422608            3951
                 320556311            1521
                 322296085            9548
                 323327967            6096
                 324029935            7710
                 325410202            8808
                 325436369            1225
                 327079023            1721
                 327532551            4591
                 328241980           69521
                 329073920            2712
                 331507462            1766
                 333772141            9304
                 333923208            5253
                 334845115            6451
                 336628043            5397
                 336867215           10235
                 337689551            2671
                 338793195           20085
                 339081310            1275
                 340545413            3539
                 343101975            2559
                 343651168            8206
                 344426364            8586
                 347260004           15992
                 348184437            8059
                 348464122           18638
                 350469054           16019
                 350619874            1156
                 350710551           19398
                 352321237            3385
                 356324220             820
                 357854297            2693
                 360209609            1225
                 362620330            5289
                 364399287             300
                 365737009            1516
                 366754393             561
                 369286588           17403
                 370923228           13591
                 371296877             561
                 371681940            2584
                 372517985             780
                 374995552            8449
                 377356859           10154
                 378598818            2958
                 379701338            4154
                 379748121            9248
                 381266657            7216
                 383054822            2242
                 383661443            1176
                 383808598             820
                 383973423            6771
                 384241357            5770
                 384534025            2556
                 384863779             900
                 387022940            5583
                 388935889            1128
                 389377620            1963
                 391305209            2424
    dtype: int64



## <font color='blue'>Meta-Data<font>

## 10) Event_data

# C. Feature Engineering

Let's first try to construct "domain-like" features which might correlate to user performance. Intuitively, I'll seek for features that somehow describe:

1. Level of game difficulty
2. User Expertise
3. User Focus 
4. Clarity of Game Instructions
5. Other

In particular, I'll use the continuous cols to construct aggregated feats.

## QUESTION: What pct of activities get completed and reach the assessment phase?

## C1. Game_time Aggregations
How much time do the app users spend:<br>
a) playing in general <br>
b) playing in a certain level (title) <br>
c) playing in a certain world <br>
d) on average per game session <br>
e) to complete a particular task <br>

### a) Net Game Time


```python
def net_time(df):
    
    return df.groupby('installation_id')[['game_time']]\
             .sum()\
             .rename({'game_time': 'net_time'}, axis=1)
```


```python
net_times = net_time(train)
```


```python
net_times.head()
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
      <th>net_time</th>
    </tr>
    <tr>
      <th>installation_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>125199</td>
      <td>4563.90</td>
    </tr>
    <tr>
      <td>280516</td>
      <td>244.44</td>
    </tr>
    <tr>
      <td>435871</td>
      <td>5225.76</td>
    </tr>
    <tr>
      <td>442770</td>
      <td>3350.47</td>
    </tr>
    <tr>
      <td>632233</td>
      <td>2736.90</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Number of ids with ZERO net_time : {0}/{1}'
      .format(np.sum(net_times['net_time']==0), 
              len(net_times))
     )
     
```

    Number of ids with ZERO net_time : 2/1119



```python
less_than = 1
print('Number of ids with net_time less than {0} min: {1}/{2}'
      .format(less_than, 
              np.sum(net_times['net_time']<less_than), 
              len(net_times))
     )
```

    Number of ids with net_time less than 1 min: 22/1119



```python
def plot_log_distribution(df, cut_off=1, title="Distribution of Log(net_time) with cut-off="):
   
    # filter out ids with net_time less than threshold (min)
    df = df[df.net_time > cut_off]
    df.apply(np.log)\
      .plot(kind='hist' ,bins=50,edgecolor='white',
            figsize=(15,6),color=my_pal[1])
    plt.gcf().suptitle(title+str(cut_off),fontsize = 18)
    plt.ylabel('num_ids', fontsize=18)
    
    return df.describe()
    
```


```python
plot_log_distribution(net_times)
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
      <th>net_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>1096.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>2456.108823</td>
    </tr>
    <tr>
      <td>std</td>
      <td>7321.960625</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.030000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>106.110000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>475.130000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>1748.597500</td>
    </tr>
    <tr>
      <td>max</td>
      <td>107002.490000</td>
    </tr>
  </tbody>
</table>
</div>




![png](../images/kaggle_dsb_figs/output_80_1.png)


## <font color='green'> Good Features: ```net_time``` </font> 

Comments: 
- perhaps use log of net_time 
- treat differently ids with zero to less than a cut_off-value of game time


### b) Game Times Per Title


```python
def plot_barh_chart(df ,col):
    
    df = df.groupby(col)['game_time']\
           .sum()\
           .sort_values()
    
    df.plot(kind='barh',figsize=(10,6))
    plt.gcf().suptitle("Net Game Time Per " + col.title(), size = 18)
    plt.gca().set_ylabel(plt.gca().get_ylabel(), visible=False) #remove ylabel
    
    return df.describe()
```


```python
plot_barh_chart(train, col='title')
```




    count        24.000000
    mean     112162.643750
    std      109094.089271
    min        7983.320000
    25%       49515.712500
    50%       77829.115000
    75%      139781.872500
    max      435304.750000
    Name: game_time, dtype: float64




![png](../images/kaggle_dsb_figs/output_84_1.png)



```python
def net_times_per_col(df, col):
    
    return train.groupby(['installation_id', col])[['game_time']]\
                .sum()\
                .rename({'game_time': 'net_time_per_' + col}, axis=1)\
                .unstack()\
                .fillna(0)
```


```python
net_times_per_col(train, col='title').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="24" halign="left">net_time_per_title</th>
    </tr>
    <tr>
      <th>title</th>
      <th>Air Show</th>
      <th>All Star Sorting</th>
      <th>Bird Measurer (Assessment)</th>
      <th>Bottle Filler (Activity)</th>
      <th>Bubble Bath</th>
      <th>Bug Measurer (Activity)</th>
      <th>Cart Balancer (Assessment)</th>
      <th>Cauldron Filler (Assessment)</th>
      <th>Chest Sorter (Assessment)</th>
      <th>Chicken Balancer (Activity)</th>
      <th>Chow Time</th>
      <th>Crystals Rule</th>
      <th>Dino Dive</th>
      <th>Dino Drink</th>
      <th>Egg Dropper (Activity)</th>
      <th>Fireworks (Activity)</th>
      <th>Flower Waterer (Activity)</th>
      <th>Happy Camel</th>
      <th>Leaf Leader</th>
      <th>Mushroom Sorter (Assessment)</th>
      <th>Pan Balance</th>
      <th>Sandcastle Builder (Activity)</th>
      <th>Scrub-A-Dub</th>
      <th>Watering Hole (Activity)</th>
    </tr>
    <tr>
      <th>installation_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>125199</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>304.14</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>495.39</td>
      <td>3764.37</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>280516</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>193.02</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>51.42</td>
    </tr>
    <tr>
      <td>435871</td>
      <td>479.98</td>
      <td>124.6</td>
      <td>78.52</td>
      <td>622.83</td>
      <td>340.05</td>
      <td>270.66</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>381.35</td>
      <td>20.70</td>
      <td>301.65</td>
      <td>0.0</td>
      <td>1495.52</td>
      <td>339.12</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>29.74</td>
      <td>0.00</td>
      <td>261.13</td>
      <td>378.24</td>
      <td>101.67</td>
    </tr>
    <tr>
      <td>442770</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>366.12</td>
      <td>843.68</td>
      <td>137.03</td>
      <td>104.85</td>
      <td>0.0</td>
      <td>3.12</td>
      <td>0.0</td>
      <td>103.87</td>
      <td>51.25</td>
      <td>0.77</td>
      <td>238.11</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>452.06</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>82.30</td>
      <td>1.23</td>
      <td>586.18</td>
      <td>371.76</td>
      <td>8.14</td>
    </tr>
    <tr>
      <td>632233</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2571.03</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>115.61</td>
      <td>0.00</td>
      <td>50.26</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='green'>  Good Features:  ```net_time_per_title```<font>

### c) Game Time per World


```python
plot_barh_chart(train, col='world')
```




    count    3.000000e+00
    mean     8.973012e+05
    std      4.259774e+05
    min      5.787693e+05
    25%      6.553723e+05
    50%      7.319752e+05
    75%      1.056567e+06
    max      1.381159e+06
    Name: game_time, dtype: float64




![png](../images/kaggle_dsb_figs/output_89_1.png)



```python
net_times_per_col(train, col='world').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">net_time_per_world</th>
    </tr>
    <tr>
      <th>world</th>
      <th>CRYSTALCAVES</th>
      <th>MAGMAPEAK</th>
      <th>TREETOPCITY</th>
    </tr>
    <tr>
      <th>installation_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>125199</td>
      <td>0.00</td>
      <td>4563.90</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>280516</td>
      <td>0.00</td>
      <td>244.44</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>435871</td>
      <td>0.00</td>
      <td>2026.27</td>
      <td>3199.49</td>
    </tr>
    <tr>
      <td>442770</td>
      <td>156.35</td>
      <td>2188.02</td>
      <td>1006.10</td>
    </tr>
    <tr>
      <td>632233</td>
      <td>0.00</td>
      <td>2736.90</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='green'>  Good Features:  ```net_time_per_world```<font>

COMMENT: Can also compute aggregates for event_counts, game_sessions, etc per title/world

### d) Avg Time Per Session


```python
def avg_time_per_sess(df):
    
    df = df.reset_index()\
           .groupby(['installation_id'])[['game_time','game_session']]\
           .agg({'game_time': 'sum',
                 'game_session': 'nunique'})
    
    df['avg_time_per_sess'] = df['game_time'] / df['game_session']
    
    return df[['avg_time_per_sess']].astype('int')

             
```


```python
avg_time_per_sess(train).head()
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
      <th>avg_time_per_sess</th>
    </tr>
    <tr>
      <th>installation_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>125199</td>
      <td>912</td>
    </tr>
    <tr>
      <td>280516</td>
      <td>122</td>
    </tr>
    <tr>
      <td>435871</td>
      <td>124</td>
    </tr>
    <tr>
      <td>442770</td>
      <td>186</td>
    </tr>
    <tr>
      <td>632233</td>
      <td>912</td>
    </tr>
  </tbody>
</table>
</div>



### e)  Time per task (Activity)


```python
train[train.type=="Activity"]
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
      <th></th>
      <th>event_id</th>
      <th>timestamp</th>
      <th>event_data</th>
      <th>event_count</th>
      <th>event_code</th>
      <th>game_time</th>
      <th>title</th>
      <th>type</th>
      <th>world</th>
      <th>hour</th>
      <th>day</th>
    </tr>
    <tr>
      <th>installation_id</th>
      <th>game_session</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="top">125199</td>
      <td>0848ef14a8dc6892</td>
      <td>77261ab5</td>
      <td>2019-09-06 17:54:56.302000+00:00</td>
      <td>{"version":"1.0","event_count":1,"game_time":0...</td>
      <td>1</td>
      <td>2000</td>
      <td>0.00</td>
      <td>Sandcastle Builder (Activity)</td>
      <td>Activity</td>
      <td>MAGMAPEAK</td>
      <td>17</td>
      <td>6</td>
    </tr>
    <tr>
      <td>0848ef14a8dc6892</td>
      <td>b2dba42b</td>
      <td>2019-09-06 17:54:56.387000+00:00</td>
      <td>{"description":"Let's build a sandcastle! Firs...</td>
      <td>2</td>
      <td>3010</td>
      <td>0.00</td>
      <td>Sandcastle Builder (Activity)</td>
      <td>Activity</td>
      <td>MAGMAPEAK</td>
      <td>17</td>
      <td>6</td>
    </tr>
    <tr>
      <td>0848ef14a8dc6892</td>
      <td>1bb5fbdb</td>
      <td>2019-09-06 17:55:03.253000+00:00</td>
      <td>{"description":"Let's build a sandcastle! Firs...</td>
      <td>3</td>
      <td>3110</td>
      <td>0.12</td>
      <td>Sandcastle Builder (Activity)</td>
      <td>Activity</td>
      <td>MAGMAPEAK</td>
      <td>17</td>
      <td>6</td>
    </tr>
    <tr>
      <td>0848ef14a8dc6892</td>
      <td>1325467d</td>
      <td>2019-09-06 17:55:06.279000+00:00</td>
      <td>{"coordinates":{"x":583,"y":605,"stage_width":...</td>
      <td>4</td>
      <td>4070</td>
      <td>0.17</td>
      <td>Sandcastle Builder (Activity)</td>
      <td>Activity</td>
      <td>MAGMAPEAK</td>
      <td>17</td>
      <td>6</td>
    </tr>
    <tr>
      <td>0848ef14a8dc6892</td>
      <td>1325467d</td>
      <td>2019-09-06 17:55:06.913000+00:00</td>
      <td>{"coordinates":{"x":601,"y":570,"stage_width":...</td>
      <td>5</td>
      <td>4070</td>
      <td>0.18</td>
      <td>Sandcastle Builder (Activity)</td>
      <td>Activity</td>
      <td>MAGMAPEAK</td>
      <td>17</td>
      <td>6</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td rowspan="5" valign="top">394216680</td>
      <td>e12e5328ddf03935</td>
      <td>4bb2f698</td>
      <td>2019-08-21 22:52:36.868000+00:00</td>
      <td>{"coordinates":{"x":721,"y":385,"stage_width":...</td>
      <td>52</td>
      <td>4070</td>
      <td>0.88</td>
      <td>Chicken Balancer (Activity)</td>
      <td>Activity</td>
      <td>CRYSTALCAVES</td>
      <td>22</td>
      <td>21</td>
    </tr>
    <tr>
      <td>e12e5328ddf03935</td>
      <td>56bcd38d</td>
      <td>2019-08-21 22:52:37.729000+00:00</td>
      <td>{"object":"pig","layout":{"left":{"chickens":1...</td>
      <td>53</td>
      <td>4030</td>
      <td>0.90</td>
      <td>Chicken Balancer (Activity)</td>
      <td>Activity</td>
      <td>CRYSTALCAVES</td>
      <td>22</td>
      <td>21</td>
    </tr>
    <tr>
      <td>e12e5328ddf03935</td>
      <td>46cd75b4</td>
      <td>2019-08-21 22:52:38.740000+00:00</td>
      <td>{"side":"right","layout":{"left":{"chickens":1...</td>
      <td>54</td>
      <td>4022</td>
      <td>0.92</td>
      <td>Chicken Balancer (Activity)</td>
      <td>Activity</td>
      <td>CRYSTALCAVES</td>
      <td>22</td>
      <td>21</td>
    </tr>
    <tr>
      <td>e12e5328ddf03935</td>
      <td>56bcd38d</td>
      <td>2019-08-21 22:52:42.710000+00:00</td>
      <td>{"object":"pig","layout":{"left":{"chickens":1...</td>
      <td>55</td>
      <td>4030</td>
      <td>0.98</td>
      <td>Chicken Balancer (Activity)</td>
      <td>Activity</td>
      <td>CRYSTALCAVES</td>
      <td>22</td>
      <td>21</td>
    </tr>
    <tr>
      <td>e12e5328ddf03935</td>
      <td>46cd75b4</td>
      <td>2019-08-21 22:52:44.306000+00:00</td>
      <td>{"side":"right","layout":{"left":{"chickens":1...</td>
      <td>56</td>
      <td>4022</td>
      <td>1.01</td>
      <td>Chicken Balancer (Activity)</td>
      <td>Activity</td>
      <td>CRYSTALCAVES</td>
      <td>22</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
<p>399640 rows × 11 columns</p>
</div>



## C2. Game_session Aggregations

a) time of the day user signed in <br>
b) how much time spent per session <br>
c) boredom by playing too much?

### a) Count Sessions Per Col


```python
def count_sess_per_col(df, col):
    
    return df.reset_index()\
             .groupby(['installation_id', col])[['game_session']]\
             .nunique()\
             .rename({'game_session': 'no_sess_per_' + col}, axis=1)\
             .unstack()\
             .fillna(0)
    

```


```python
count_sess_per_col(train, col='hour').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="24" halign="left">no_sess_per_hour</th>
    </tr>
    <tr>
      <th>hour</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
    </tr>
    <tr>
      <th>installation_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>125199</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>280516</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>435871</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>442770</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>632233</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
count_sess_per_col(train, col='day').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="31" halign="left">no_sess_per_day</th>
    </tr>
    <tr>
      <th>day</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
    </tr>
    <tr>
      <th>installation_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>125199</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>280516</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>435871</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>442770</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>632233</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
count_sess_per_col(train, col='world').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">no_sess_per_world</th>
    </tr>
    <tr>
      <th>world</th>
      <th>CRYSTALCAVES</th>
      <th>MAGMAPEAK</th>
      <th>TREETOPCITY</th>
    </tr>
    <tr>
      <th>installation_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>125199</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>280516</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>435871</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>442770</td>
      <td>3.0</td>
      <td>10.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>632233</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
count_sess_per_col(train, col='title').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="24" halign="left">no_sess_per_title</th>
    </tr>
    <tr>
      <th>title</th>
      <th>Air Show</th>
      <th>All Star Sorting</th>
      <th>Bird Measurer (Assessment)</th>
      <th>Bottle Filler (Activity)</th>
      <th>Bubble Bath</th>
      <th>Bug Measurer (Activity)</th>
      <th>Cart Balancer (Assessment)</th>
      <th>Cauldron Filler (Assessment)</th>
      <th>Chest Sorter (Assessment)</th>
      <th>Chicken Balancer (Activity)</th>
      <th>Chow Time</th>
      <th>Crystals Rule</th>
      <th>Dino Dive</th>
      <th>Dino Drink</th>
      <th>Egg Dropper (Activity)</th>
      <th>Fireworks (Activity)</th>
      <th>Flower Waterer (Activity)</th>
      <th>Happy Camel</th>
      <th>Leaf Leader</th>
      <th>Mushroom Sorter (Assessment)</th>
      <th>Pan Balance</th>
      <th>Sandcastle Builder (Activity)</th>
      <th>Scrub-A-Dub</th>
      <th>Watering Hole (Activity)</th>
    </tr>
    <tr>
      <th>installation_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>125199</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>280516</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>435871</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>442770</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>632233</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='green'>  Good Features:  ```no_sess_per_col```<font>
```col = ['hour', 'day', 'title', 'world']```

### b) 

## <font color='green'>  Good Features:  ```time_per_sess, events_per_sess, avg_time_per_sess```<font>

### c) How often do they sign in AND play? And Why?


```python
# collect "long" game sessions, i.e. sessions with more than one event
long_game_sess = time_per_sess[time_per_sess.event_count > 1].index

# select the beginning of all game sessions, i.e. the sign-in date
# and then filter out "short" sessions 
time_diff = train[train.event_count == 1].loc[long_game_sess, ['timestamp']]
time_diff['date'] = time_diff['timestamp'].dt.date

# count the number of days that a user has signed in (good sing-in's since long sessions)
time_diff = time_diff.groupby(['installation_id'])[['date']].nunique()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-55-1bf77fc06cc8> in <module>
          1 # collect "long" game sessions, i.e. sessions with more than one event
    ----> 2 long_game_sess = time_per_sess[time_per_sess.event_count > 1].index
          3 
          4 # select the beginning of all game sessions, i.e. the sign-in date
          5 # and then filter out "short" sessions


    NameError: name 'time_per_sess' is not defined



```python
all_accuracies = assessments[assessments.game_time>0][['num_correct']].groupby('installation_id')[['num_correct']]\
                            .agg(['count','sum', pct_corrects])

# Clean df
all_accuracies.columns = ['num_assessments', 'num_correct', 'pct_correct']
all_accuracies['pct_incorrect'] =  1 - all_accuracies.pct_correct
```


```python
multiple_signins = all_accuracies.reindex(time_diff[time_diff.date>1].index).dropna()
single_signins = all_accuracies.reindex(time_diff[time_diff.date==1].index).dropna()
```


```python
multiple_signins[multiple_signins.num_correct>0].describe().applymap(lambda x: round(x,2))
```


```python
single_signins[single_signins.num_correct>0].describe().applymap(lambda x: round(x,2))
```


```python
multiple_signins[multiple_signins.num_correct>0]['pct_correct'].hist(bins=20)
```


```python
single_signins[single_signins.num_correct>0]['pct_correct'].hist(bins=20)
```

Comment: users with single sign-ins have quite higher accuracy than users with multiple sing-ins. Perhaps single-sing-ins users need less time to advance as players and pass the different tasks. Or perhaps they spent time doing only the easy tasks and then skipped the difficult tasks or gave up.

Note: some players might have signed in from different devices (i.e. with different ids)

## <font color='green'>  Good Features:  ```user sign-ins```<font>

## C3. Event_code Aggregations


```python
def count_codes_per_col(df, col):
    
    return df.reset_index()\
             .groupby(['installation_id', col])[['event_code']]\
             .nunique()\
             .rename({'event_code': 'no_codes_per_' + col}, axis=1)\
             .unstack()\
             .fillna(0)
```


```python
count_codes_per_col(train, 'world').head()
```


```python
count_codes_per_col(train, 'title')
```


```python
count_codes_per_col(train, 'hour')
```

## 4. Clarity of game instructions

Game Clarity:<br>
a) rate hints based on user performance <br>
b) check if the user was doing something weird: from coords (e.g. false negatives) <br>
c) check if the user was doing something weird to familiarize himself/herself with the game (e.g. at the beginnig might press randomly)

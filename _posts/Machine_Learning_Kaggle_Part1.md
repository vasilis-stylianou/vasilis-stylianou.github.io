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

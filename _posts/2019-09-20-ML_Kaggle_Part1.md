---
title: "Machine Learning (Kaggle): Part I"
date: 2019-09-20
tags: [machine learning, Imputation, PCA, OOP, Feature Engineering, pandas, numpy]
excerpt: OOP, Imputation, Feature Engineering, PCA
---

# Intro:
The following work was done for the Kaggle competition -  [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection). Contestants were provided with four datasets, from Vesta Corporation:
- train_{transaction, identity}.csv - the training set
- test_{transaction, identity}.csv - the test set

and were asked to predict the probability of a card transaction being fraudulent. Submissions were evaluated on area under the ROC curve between the predicted probability (of the test set) and the observed target.

# Index:
1. Data Cleaning
2. Feature Engineering
3. Feature Selection and Preprocessing
4. Model Training/Evaluation/Selection

In this post I discuss steps 1-3.

Note: By the completion of this project the code was distributed among a number of python scripts and jupyter notebooks. In this post I have concatenated everything together but for the sake of brevity I left out some of the utility scripts, and I refer the reader to my github repository for more details.

# Step 1: Data Cleaning

## Libraries
```python
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
```

## Utility script for data cleaning and preprocessing

It is very useful to create a utility script for data analysis as this will help us keep our code clean and neat. For example, we can create a class called ```Variable``` to capture all the important statistical properties and attributes of each feature (the terms features/columns/variables are used interchangeably in this post):

```python
class Variable:
    """Class to describe a variable"""

    def __init__(self,data_series,col_name,label=None):
        """Method to initiate object.
        Inputs:
            data_series: pandas series Variable
            col_name: string with column's col_name
            label: data series of prediction
        """
        self.name = col_name
        self.num_cats = len(data_series.unique()) #number of unique values in column
        self.type = str(data_series.dtype)
        self.num_nans = np.sum(data_series.isnull()) #number of NaNs in column
        self.nans_rate = self.num_nans/len(data_series) #percent of NaNs in column
        self.fraud_nans = np.sum(data_series[label==1].isnull()) #number of Fraud NaNs in column
        self.fraud_nans_rate = self.fraud_nans/np.sum(label) #pct of Fraud NaNs in column

        if str(data_series.dtype) == 'object':
            try:
                self.corr = self.correlation_ratio(data_series.values,label.values)
            except:
                self.corr = 'Invalid variable'
                pass
        else:
            self.average = data_series.dropna().values.mean()
            self.std = data_series.dropna().values.std()
            try:
                self.corr = pearsonr(data_series, label)[0]
            except:
                self.corr = 'Invalid variable'
                pass
```
With the above class in hand, we can also define another class, called ```TableDescriptor```, to create a list of ```Variable``` objects for the columns of an input dataframe:

```python
class TableDescriptor:
    """Class to automatically describe every variable in a df"""

    def __init__(self,df,df_name,label_name=None):
        """Method to intiate object.
        Inputs:
            df: pandas dataframe
            df_name: string with df's col_name
            label_name: series name
        """
        if label_name == None:
            self.label = None
        else:
            self.label = df[label_name]
        self.variables = [Variable(df[col],col,
                                    label=self.label) for col in tqdm(df.columns)]
```
From now on, we refer the reader to the python script ```fraud_pre_proc.py``` for all the user-defined classes (and methods) which help for data cleaning and preprocessing.

```python
from fraud_pre_proc import *
```

# Import data

## 1.1.1 Transactions

```python
df_train_trans = import_data('./Data/train_transaction.csv')
df_test_trans = import_data('./Data/test_transaction.csv')
```
Here ```import_data``` is a method in ```fraud_pre_proc.py``` which creates a pandas dataframe and reduces its memory usage by ~60%.

```python
df_train_trans.head(3)
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
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>ProductCD</th>
      <th>card1</th>
      <th>card2</th>
      <th>card3</th>
      <th>card4</th>
      <th>card5</th>
      <th>card6</th>
      <th>addr1</th>
      <th>addr2</th>
      <th>dist1</th>
      <th>dist2</th>
      <th>P_emaildomain</th>
      <th>R_emaildomain</th>
      <th>C1</th>
      <th>C2</th>
      <th>C3</th>
      <th>C4</th>
      <th>C5</th>
      <th>C6</th>
      <th>C7</th>
      <th>C8</th>
      <th>C9</th>
      <th>C10</th>
      <th>C11</th>
      <th>C12</th>
      <th>C13</th>
      <th>C14</th>
      <th>D1</th>
      <th>D2</th>
      <th>D3</th>
      <th>D4</th>
      <th>D5</th>
      <th>D6</th>
      <th>D7</th>
      <th>D8</th>
      <th>D9</th>
      <th>D10</th>
      <th>D11</th>
      <th>D12</th>
      <th>D13</th>
      <th>D14</th>
      <th>D15</th>
      <th>M1</th>
      <th>M2</th>
      <th>M3</th>
      <th>M4</th>
      <th>M5</th>
      <th>M6</th>
      <th>M7</th>
      <th>M8</th>
      <th>M9</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>V29</th>
      <th>V30</th>
      <th>V31</th>
      <th>V32</th>
      <th>V33</th>
      <th>V34</th>
      <th>V35</th>
      <th>V36</th>
      <th>V37</th>
      <th>V38</th>
      <th>V39</th>
      <th>V40</th>
      <th>V41</th>
      <th>V42</th>
      <th>V43</th>
      <th>V44</th>
      <th>V45</th>
      <th>V46</th>
      <th>V47</th>
      <th>V48</th>
      <th>V49</th>
      <th>V50</th>
      <th>V51</th>
      <th>V52</th>
      <th>V53</th>
      <th>V54</th>
      <th>V55</th>
      <th>V56</th>
      <th>V57</th>
      <th>V58</th>
      <th>V59</th>
      <th>V60</th>
      <th>V61</th>
      <th>V62</th>
      <th>V63</th>
      <th>V64</th>
      <th>V65</th>
      <th>V66</th>
      <th>V67</th>
      <th>V68</th>
      <th>V69</th>
      <th>V70</th>
      <th>V71</th>
      <th>V72</th>
      <th>V73</th>
      <th>V74</th>
      <th>V75</th>
      <th>V76</th>
      <th>V77</th>
      <th>V78</th>
      <th>V79</th>
      <th>V80</th>
      <th>V81</th>
      <th>V82</th>
      <th>V83</th>
      <th>V84</th>
      <th>V85</th>
      <th>V86</th>
      <th>V87</th>
      <th>V88</th>
      <th>V89</th>
      <th>V90</th>
      <th>V91</th>
      <th>V92</th>
      <th>V93</th>
      <th>V94</th>
      <th>V95</th>
      <th>V96</th>
      <th>V97</th>
      <th>V98</th>
      <th>V99</th>
      <th>V100</th>
      <th>V101</th>
      <th>V102</th>
      <th>V103</th>
      <th>V104</th>
      <th>V105</th>
      <th>V106</th>
      <th>V107</th>
      <th>V108</th>
      <th>V109</th>
      <th>V110</th>
      <th>V111</th>
      <th>V112</th>
      <th>V113</th>
      <th>V114</th>
      <th>V115</th>
      <th>V116</th>
      <th>V117</th>
      <th>V118</th>
      <th>V119</th>
      <th>V120</th>
      <th>V121</th>
      <th>V122</th>
      <th>V123</th>
      <th>V124</th>
      <th>V125</th>
      <th>V126</th>
      <th>V127</th>
      <th>V128</th>
      <th>V129</th>
      <th>V130</th>
      <th>V131</th>
      <th>V132</th>
      <th>V133</th>
      <th>V134</th>
      <th>V135</th>
      <th>V136</th>
      <th>V137</th>
      <th>V138</th>
      <th>V139</th>
      <th>V140</th>
      <th>V141</th>
      <th>V142</th>
      <th>V143</th>
      <th>V144</th>
      <th>V145</th>
      <th>V146</th>
      <th>V147</th>
      <th>V148</th>
      <th>V149</th>
      <th>V150</th>
      <th>V151</th>
      <th>V152</th>
      <th>V153</th>
      <th>V154</th>
      <th>V155</th>
      <th>V156</th>
      <th>V157</th>
      <th>V158</th>
      <th>V159</th>
      <th>V160</th>
      <th>V161</th>
      <th>V162</th>
      <th>V163</th>
      <th>V164</th>
      <th>V165</th>
      <th>V166</th>
      <th>V167</th>
      <th>V168</th>
      <th>V169</th>
      <th>V170</th>
      <th>V171</th>
      <th>V172</th>
      <th>V173</th>
      <th>V174</th>
      <th>V175</th>
      <th>V176</th>
      <th>V177</th>
      <th>V178</th>
      <th>V179</th>
      <th>V180</th>
      <th>V181</th>
      <th>V182</th>
      <th>V183</th>
      <th>V184</th>
      <th>V185</th>
      <th>V186</th>
      <th>V187</th>
      <th>V188</th>
      <th>V189</th>
      <th>V190</th>
      <th>V191</th>
      <th>V192</th>
      <th>V193</th>
      <th>V194</th>
      <th>V195</th>
      <th>V196</th>
      <th>V197</th>
      <th>V198</th>
      <th>V199</th>
      <th>V200</th>
      <th>V201</th>
      <th>V202</th>
      <th>V203</th>
      <th>V204</th>
      <th>V205</th>
      <th>V206</th>
      <th>V207</th>
      <th>V208</th>
      <th>V209</th>
      <th>V210</th>
      <th>V211</th>
      <th>V212</th>
      <th>V213</th>
      <th>V214</th>
      <th>V215</th>
      <th>V216</th>
      <th>V217</th>
      <th>V218</th>
      <th>V219</th>
      <th>V220</th>
      <th>V221</th>
      <th>V222</th>
      <th>V223</th>
      <th>V224</th>
      <th>V225</th>
      <th>V226</th>
      <th>V227</th>
      <th>V228</th>
      <th>V229</th>
      <th>V230</th>
      <th>V231</th>
      <th>V232</th>
      <th>V233</th>
      <th>V234</th>
      <th>V235</th>
      <th>V236</th>
      <th>V237</th>
      <th>V238</th>
      <th>V239</th>
      <th>V240</th>
      <th>V241</th>
      <th>V242</th>
      <th>V243</th>
      <th>V244</th>
      <th>V245</th>
      <th>V246</th>
      <th>V247</th>
      <th>V248</th>
      <th>V249</th>
      <th>V250</th>
      <th>V251</th>
      <th>V252</th>
      <th>V253</th>
      <th>V254</th>
      <th>V255</th>
      <th>V256</th>
      <th>V257</th>
      <th>V258</th>
      <th>V259</th>
      <th>V260</th>
      <th>V261</th>
      <th>V262</th>
      <th>V263</th>
      <th>V264</th>
      <th>V265</th>
      <th>V266</th>
      <th>V267</th>
      <th>V268</th>
      <th>V269</th>
      <th>V270</th>
      <th>V271</th>
      <th>V272</th>
      <th>V273</th>
      <th>V274</th>
      <th>V275</th>
      <th>V276</th>
      <th>V277</th>
      <th>V278</th>
      <th>V279</th>
      <th>V280</th>
      <th>V281</th>
      <th>V282</th>
      <th>V283</th>
      <th>V284</th>
      <th>V285</th>
      <th>V286</th>
      <th>V287</th>
      <th>V288</th>
      <th>V289</th>
      <th>V290</th>
      <th>V291</th>
      <th>V292</th>
      <th>V293</th>
      <th>V294</th>
      <th>V295</th>
      <th>V296</th>
      <th>V297</th>
      <th>V298</th>
      <th>V299</th>
      <th>V300</th>
      <th>V301</th>
      <th>V302</th>
      <th>V303</th>
      <th>V304</th>
      <th>V305</th>
      <th>V306</th>
      <th>V307</th>
      <th>V308</th>
      <th>V309</th>
      <th>V310</th>
      <th>V311</th>
      <th>V312</th>
      <th>V313</th>
      <th>V314</th>
      <th>V315</th>
      <th>V316</th>
      <th>V317</th>
      <th>V318</th>
      <th>V319</th>
      <th>V320</th>
      <th>V321</th>
      <th>V322</th>
      <th>V323</th>
      <th>V324</th>
      <th>V325</th>
      <th>V326</th>
      <th>V327</th>
      <th>V328</th>
      <th>V329</th>
      <th>V330</th>
      <th>V331</th>
      <th>V332</th>
      <th>V333</th>
      <th>V334</th>
      <th>V335</th>
      <th>V336</th>
      <th>V337</th>
      <th>V338</th>
      <th>V339</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2987000</td>
      <td>0</td>
      <td>86400</td>
      <td>68.5</td>
      <td>W</td>
      <td>13926</td>
      <td>NaN</td>
      <td>150.0</td>
      <td>discover</td>
      <td>142.0</td>
      <td>credit</td>
      <td>315.0</td>
      <td>87.0</td>
      <td>19.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>T</td>
      <td>T</td>
      <td>T</td>
      <td>M2</td>
      <td>F</td>
      <td>T</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2987001</td>
      <td>0</td>
      <td>86401</td>
      <td>29.0</td>
      <td>W</td>
      <td>2755</td>
      <td>404.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>102.0</td>
      <td>credit</td>
      <td>325.0</td>
      <td>87.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>gmail.com</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>M0</td>
      <td>T</td>
      <td>T</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2987002</td>
      <td>0</td>
      <td>86469</td>
      <td>59.0</td>
      <td>W</td>
      <td>4663</td>
      <td>490.0</td>
      <td>150.0</td>
      <td>visa</td>
      <td>166.0</td>
      <td>debit</td>
      <td>330.0</td>
      <td>87.0</td>
      <td>287.0</td>
      <td>NaN</td>
      <td>outlook.com</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>315.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>315.0</td>
      <td>T</td>
      <td>T</td>
      <td>T</td>
      <td>M0</td>
      <td>F</td>
      <td>F</td>
      <td>F</td>
      <td>F</td>
      <td>F</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


```python
trans = TableDescriptor(df_train_trans,'Transaction_df','isFraud')
```

    100%|██████████| 394/394 [00:13<00:00, 34.89it/s]



```python
len(trans.variables)
```




    394



## NaNs_rate
Let us select all variables with less than 90% of NaN values. The method ```getCompletedVars``` returns a list of all such variables.

```python
low_nan_vars=getCompletedVars(trans,nans_rate_cut_off = 0.1)
```

    Selected features: 112/394


Let us also use the method ```numerical_categorical_split``` to split the variables into two lists:
- numerical_vars: a list of numerical variables with a number of unique values higher than 300
- categorical_vars: a list of string variables or numerical variables with a number of unique values less than or equal to 300.

```python
numerical_vars,categorical_vars= numerical_categorical_split(low_nan_vars,min_categories=300)
```

    No of numerical features: 57
    No of categorical features: 55


## 1.1.2 Multiple Clustering Imputation Method
Evidently, the data consists of a lot of missing values and in order to proceed we need to adopt an imputation method to fill these values. A quick way to achieve this would be to fill the missing values of each numerical/categorical column with the average/most frequent value of the column, excluding all NaNs.
A more sophisticated way would be to find for each row with missing values, a cluster of the most similar rows, compute each column's average or most frequent value (restricted to the cluster's rows), and then impute the missing values of the row under consideration.

The authors of this [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6371325/) has developed such a methodology - called Multiple Clustering Imputation Process (MCIP) - and published their algorithm [here](https://github.com/panas89/multipleClusteringImputation). The code has been appropriately adapted and saved in our github repository as ```mcip.py```.

Before we begin the imputation let us increase the number of available non-NaN data by concatenating the train and test sets together.

```python
num_cols_trans = [var.name for var in numerical_vars if var.name not in ['TransactionID','isFraud']]
cat_cols_trans = [var.name for var in categorical_vars if var.name not in ['TransactionID','isFraud']]
cols_trans = num_cols_trans + cat_cols_trans
```


```python
all_cols_trans = ['TransactionID']+ cols_trans

df_train_test_trans = df_train_trans[all_cols_trans].append(df_test_trans[all_cols_trans],
                                              ignore_index=True)
```


```python
df_train_test_trans.shape
```




    (1097231, 111)


## Some Technical Prerequisites
```python
#numpy array of nulls
bool_nulls_train = df_train_test_trans.loc[:,cat_cols_trans].isnull()

#replacing nulls with 'NaN' so that it can be used with the encoder
for col in cat_cols_trans:
    df_train_test_trans[col] = df_train_test_trans[col].astype('str')

df_train_test_cats = df_train_test_trans.loc[:,cat_cols_trans].copy()
df_train_test_cats[df_train_test_cats.isnull()] = 'NaN'
df_train_test_trans.loc[:,cat_cols_trans] = df_train_test_cats.copy()
```
```python
from sklearn.preprocessing import LabelEncoder

#label encoding strings of categorical variables
lb_make = LabelEncoder()
for col in cat_cols_trans:
    lb_make.fit(df_train_test_trans[col].dropna())
    df_train_test_trans[col] = lb_make.transform(df_train_test_trans[col])

#replacing categorical varibales with 'NaN' value to nulls
df_train_test_cats = df_train_test_trans.loc[:,cat_cols_trans].copy()
df_train_test_cats[bool_nulls_train] = np.nan
df_train_test_trans.loc[:,cat_cols_trans] = df_train_test_cats.copy()
```

```python
df_train_test_trans[cols_trans].head(3)
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
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>card1</th>
      <th>card2</th>
      <th>C1</th>
      <th>C2</th>
      <th>C4</th>
      <th>C5</th>
      <th>C6</th>
      <th>C7</th>
      <th>C8</th>
      <th>C10</th>
      <th>C11</th>
      <th>C12</th>
      <th>C13</th>
      <th>C14</th>
      <th>D1</th>
      <th>V95</th>
      <th>V96</th>
      <th>V97</th>
      <th>V101</th>
      <th>V102</th>
      <th>V103</th>
      <th>V126</th>
      <th>V127</th>
      <th>V128</th>
      <th>V129</th>
      <th>V130</th>
      <th>V131</th>
      <th>V132</th>
      <th>V133</th>
      <th>V134</th>
      <th>V135</th>
      <th>V136</th>
      <th>V137</th>
      <th>V279</th>
      <th>V280</th>
      <th>V293</th>
      <th>V294</th>
      <th>V295</th>
      <th>V306</th>
      <th>V307</th>
      <th>V308</th>
      <th>V309</th>
      <th>V310</th>
      <th>V311</th>
      <th>V312</th>
      <th>V313</th>
      <th>V314</th>
      <th>V315</th>
      <th>V316</th>
      <th>V317</th>
      <th>V318</th>
      <th>V319</th>
      <th>V320</th>
      <th>V321</th>
      <th>ProductCD</th>
      <th>card3</th>
      <th>card4</th>
      <th>card5</th>
      <th>card6</th>
      <th>C3</th>
      <th>C9</th>
      <th>V98</th>
      <th>V99</th>
      <th>V100</th>
      <th>V104</th>
      <th>V105</th>
      <th>V106</th>
      <th>V107</th>
      <th>V108</th>
      <th>V109</th>
      <th>V110</th>
      <th>V111</th>
      <th>V112</th>
      <th>V113</th>
      <th>V114</th>
      <th>V115</th>
      <th>V116</th>
      <th>V117</th>
      <th>V118</th>
      <th>V119</th>
      <th>V120</th>
      <th>V121</th>
      <th>V122</th>
      <th>V123</th>
      <th>V124</th>
      <th>V125</th>
      <th>V281</th>
      <th>V282</th>
      <th>V283</th>
      <th>V284</th>
      <th>V285</th>
      <th>V286</th>
      <th>V287</th>
      <th>V288</th>
      <th>V289</th>
      <th>V290</th>
      <th>V291</th>
      <th>V292</th>
      <th>V296</th>
      <th>V297</th>
      <th>V298</th>
      <th>V299</th>
      <th>V300</th>
      <th>V301</th>
      <th>V302</th>
      <th>V303</th>
      <th>V304</th>
      <th>V305</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>86400</td>
      <td>68.5</td>
      <td>13926</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
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
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>50.0</td>
      <td>1.0</td>
      <td>42.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <th>1</th>
      <td>86401</td>
      <td>29.0</td>
      <td>2755</td>
      <td>404.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>50.0</td>
      <td>2.0</td>
      <td>2.0</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <th>2</th>
      <td>86469</td>
      <td>59.0</td>
      <td>4663</td>
      <td>490.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>50.0</td>
      <td>4.0</td>
      <td>66.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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



## Starting The Imputation Process


```python
from mcip import *
```


```python
temp_df = pd.DataFrame(columns=df_train_test_trans.columns)

for chunk in tqdm_notebook(np.array_split(df_train_test_trans, 1000)):
    df_trans_compl = replaceMissValMCIP(df_train_test_trans,
                                          chunk.reset_index(drop=True),
                                          cols=cols_trans,frac=0.001,
                                          categorical=cat_cols_trans,
                                          continuous=num_cols_trans)

    temp_df = temp_df.append(df_trans_compl,ignore_index=True)
```
```python
#checking if nulls have been placed
df_train_test_trans[cols_trans].isnull().sum()[:5]
```




    TransactionDT         0
    TransactionAmt        0
    card1                 0
    card2             17587
    C1                    3
    dtype: int64




```python
#checking if cols were imputed
temp_df[cols_trans].isnull().sum()[:5]
```




    TransactionDT     0
    TransactionAmt    0
    card1             0
    card2             0
    C1                0
    dtype: int64




```python
temp_df[cols_trans].head(3)
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
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>card1</th>
      <th>card2</th>
      <th>C1</th>
      <th>C2</th>
      <th>C4</th>
      <th>C5</th>
      <th>C6</th>
      <th>C7</th>
      <th>C8</th>
      <th>C10</th>
      <th>C11</th>
      <th>C12</th>
      <th>C13</th>
      <th>C14</th>
      <th>D1</th>
      <th>V95</th>
      <th>V96</th>
      <th>V97</th>
      <th>V101</th>
      <th>V102</th>
      <th>V103</th>
      <th>V126</th>
      <th>V127</th>
      <th>V128</th>
      <th>V129</th>
      <th>V130</th>
      <th>V131</th>
      <th>V132</th>
      <th>V133</th>
      <th>V134</th>
      <th>V135</th>
      <th>V136</th>
      <th>V137</th>
      <th>V279</th>
      <th>V280</th>
      <th>V293</th>
      <th>V294</th>
      <th>V295</th>
      <th>V306</th>
      <th>V307</th>
      <th>V308</th>
      <th>V309</th>
      <th>V310</th>
      <th>V311</th>
      <th>V312</th>
      <th>V313</th>
      <th>V314</th>
      <th>V315</th>
      <th>V316</th>
      <th>V317</th>
      <th>V318</th>
      <th>V319</th>
      <th>V320</th>
      <th>V321</th>
      <th>ProductCD</th>
      <th>card3</th>
      <th>card4</th>
      <th>card5</th>
      <th>card6</th>
      <th>C3</th>
      <th>C9</th>
      <th>V98</th>
      <th>V99</th>
      <th>V100</th>
      <th>V104</th>
      <th>V105</th>
      <th>V106</th>
      <th>V107</th>
      <th>V108</th>
      <th>V109</th>
      <th>V110</th>
      <th>V111</th>
      <th>V112</th>
      <th>V113</th>
      <th>V114</th>
      <th>V115</th>
      <th>V116</th>
      <th>V117</th>
      <th>V118</th>
      <th>V119</th>
      <th>V120</th>
      <th>V121</th>
      <th>V122</th>
      <th>V123</th>
      <th>V124</th>
      <th>V125</th>
      <th>V281</th>
      <th>V282</th>
      <th>V283</th>
      <th>V284</th>
      <th>V285</th>
      <th>V286</th>
      <th>V287</th>
      <th>V288</th>
      <th>V289</th>
      <th>V290</th>
      <th>V291</th>
      <th>V292</th>
      <th>V296</th>
      <th>V297</th>
      <th>V298</th>
      <th>V299</th>
      <th>V300</th>
      <th>V301</th>
      <th>V302</th>
      <th>V303</th>
      <th>V304</th>
      <th>V305</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>86400.0</td>
      <td>68.5</td>
      <td>13926.0</td>
      <td>357.13153</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
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
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>50.0</td>
      <td>1.0</td>
      <td>42.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <th>1</th>
      <td>86401.0</td>
      <td>29.0</td>
      <td>2755.0</td>
      <td>404.00000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>50.0</td>
      <td>2.0</td>
      <td>2.0</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <th>2</th>
      <td>86469.0</td>
      <td>59.0</td>
      <td>4663.0</td>
      <td>490.00000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>50.0</td>
      <td>4.0</td>
      <td>66.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
#replace temp_df with df_train_test_trans
df_train_test_trans = temp_df
```

## 1.2.1 IDs


```python
df_train_ids = pd.import_data('./Data/train_identity.csv')
df_test_ids = pd.import_data('./Data/test_identity.csv')
```


```python
df_train_ids['isFraud'] = [1]*len(df_train_ids)
df_train_ids.head(3)
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
      <th>id_01</th>
      <th>id_02</th>
      <th>id_03</th>
      <th>id_04</th>
      <th>id_05</th>
      <th>id_06</th>
      <th>id_07</th>
      <th>id_08</th>
      <th>id_09</th>
      <th>id_10</th>
      <th>id_11</th>
      <th>id_12</th>
      <th>id_13</th>
      <th>id_14</th>
      <th>id_15</th>
      <th>id_16</th>
      <th>id_17</th>
      <th>id_18</th>
      <th>id_19</th>
      <th>id_20</th>
      <th>id_21</th>
      <th>id_22</th>
      <th>id_23</th>
      <th>id_24</th>
      <th>id_25</th>
      <th>id_26</th>
      <th>id_27</th>
      <th>id_28</th>
      <th>id_29</th>
      <th>id_30</th>
      <th>id_31</th>
      <th>id_32</th>
      <th>id_33</th>
      <th>id_34</th>
      <th>id_35</th>
      <th>id_36</th>
      <th>id_37</th>
      <th>id_38</th>
      <th>DeviceType</th>
      <th>DeviceInfo</th>
      <th>isFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2987004</td>
      <td>0.0</td>
      <td>70787.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>NaN</td>
      <td>-480.0</td>
      <td>New</td>
      <td>NotFound</td>
      <td>166.0</td>
      <td>NaN</td>
      <td>542.0</td>
      <td>144.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>New</td>
      <td>NotFound</td>
      <td>Android 7.0</td>
      <td>samsung browser 6.2</td>
      <td>32.0</td>
      <td>2220x1080</td>
      <td>match_status:2</td>
      <td>T</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>mobile</td>
      <td>SAMSUNG SM-G892A Build/NRD90M</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2987008</td>
      <td>-5.0</td>
      <td>98945.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>-5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>49.0</td>
      <td>-300.0</td>
      <td>New</td>
      <td>NotFound</td>
      <td>166.0</td>
      <td>NaN</td>
      <td>621.0</td>
      <td>500.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>New</td>
      <td>NotFound</td>
      <td>iOS 11.1.2</td>
      <td>mobile safari 11.0</td>
      <td>32.0</td>
      <td>1334x750</td>
      <td>match_status:1</td>
      <td>T</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>mobile</td>
      <td>iOS Device</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2987010</td>
      <td>-5.0</td>
      <td>191631.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>52.0</td>
      <td>NaN</td>
      <td>Found</td>
      <td>Found</td>
      <td>121.0</td>
      <td>NaN</td>
      <td>410.0</td>
      <td>142.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Found</td>
      <td>Found</td>
      <td>NaN</td>
      <td>chrome 62.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>desktop</td>
      <td>Windows</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


```python
ids = TableDescriptor(df_train_ids,'Transaction_df','isFraud')
```


    100%|██████████| 42/42 [00:00<00:00, 42.39it/s]



```python
len(trans.variables)
```




    42




## NaNs_rate

```python
low_nan_vars=getCompletedVars(ids,nans_rate_cut_off = 0.025)
```

    Selected features: 14/42


```python
numerical_vars,categorical_vars= numerical_categorical_split(low_nan_vars,min_categories=300)
```

    No of numerical features: 3
    No of categorical features: 11


```python
cols_ids = [var.name for var in low_nan_vars if var.name != 'isFraud']
all_cols_ids = ['TransactionID']+ cols_ids
df_train_test_ids = df_train_ids[all_cols_ids].append(df_test_ids[all_cols_ids], ignore_index=True)
df_train_test_ids.shape
```

    (286140, 14)


```python
num_cols_ids = [var.name for var in numerical_vars if var.name not in ['TransactionID','isFraud']]
cat_cols_ids = [var.name for var in categorical_vars if var.name not in ['TransactionID','isFraud']]
```

## 1.2.2 Join Transactions & IDs
It is advantageous to join the IDs dataset with the imputed transactions dataset, because any complete data augmentation will result in a much more accurate clustering method for all those incomplete rows in  ```df_train_test_ids```.

Let us perform a left join of the two datasets since ~76% of the TransactionID's in ```df_train_test_trans``` are missing from ```df_train_test_ids```.
```python
df_all=pd.merge(df_train_test_trans,df_train_test_ids,on='TransactionID',how='left')
df_all.shape
```
```python
cat_cols = cat_cols_trans + cat_cols_ids
num_cols = num_cols_trans + num_cols_ids
cols = cols_trans + cols_ids
```

## 1.2.3 Imputing Missing Values of IDs

```python
#numpy array of nulls
bool_nulls = df_all.loc[:,cat_cols].isnull()

#making cat vars to string to be saved as objects
for col in cat_cols:
    df_all[col] = df_all[col].astype('str')

#replacing nulls with 'NaN' so that it can be used with the encoder
df_all_cats = df_all.loc[:,cat_cols].copy()
df_all_cats[df_all_cats.isnull()] = 'NaN'
df_all.loc[:,cat_cols] = df_all_cats.copy()

#label encoding strings of categorical variables
lb_make = LabelEncoder()
for col in cat_cols:
    lb_make.fit(df_all[col].dropna())
    df_all[col] = lb_make.transform(df_all[col])

#replacing categorical varibales with 'NaN' value to nulls
df_all_cats = df_all.loc[:,cat_cols].copy()
df_all_cats[bool_nulls] = np.nan
df_all.loc[:,cat_cols] = df_all_cats.copy()
```

```python
temp_df= pd.DataFrame(columns=df_all.columns)

for chunk in tqdm_notebook(np.array_split(df_all, 200)):
    df_all_compl = replaceMissValMCIP(df_all,
                                    chunk.reset_index(drop=True),frac=0.001,
                                    cols=cols,categorical=cat_cols,
                                    continuous=num_cols)

    temp_df = temp_df.append(df_all_compl,ignore_index=True)

temp_df[cols].head(3)
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
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>card1</th>
      <th>card2</th>
      <th>C1</th>
      <th>C2</th>
      <th>C4</th>
      <th>C5</th>
      <th>C6</th>
      <th>C7</th>
      <th>C8</th>
      <th>C10</th>
      <th>C11</th>
      <th>C12</th>
      <th>C13</th>
      <th>C14</th>
      <th>D1</th>
      <th>V95</th>
      <th>V96</th>
      <th>V97</th>
      <th>V101</th>
      <th>V102</th>
      <th>V103</th>
      <th>V126</th>
      <th>V127</th>
      <th>V128</th>
      <th>V129</th>
      <th>V130</th>
      <th>V131</th>
      <th>V132</th>
      <th>V133</th>
      <th>V134</th>
      <th>V135</th>
      <th>V136</th>
      <th>V137</th>
      <th>V279</th>
      <th>V280</th>
      <th>V293</th>
      <th>V294</th>
      <th>V295</th>
      <th>V306</th>
      <th>V307</th>
      <th>V308</th>
      <th>V309</th>
      <th>V310</th>
      <th>V311</th>
      <th>V312</th>
      <th>V313</th>
      <th>V314</th>
      <th>V315</th>
      <th>V316</th>
      <th>V317</th>
      <th>V318</th>
      <th>V319</th>
      <th>V320</th>
      <th>V321</th>
      <th>ProductCD</th>
      <th>card3</th>
      <th>card4</th>
      <th>card5</th>
      <th>card6</th>
      <th>C3</th>
      <th>C9</th>
      <th>V98</th>
      <th>V99</th>
      <th>V100</th>
      <th>V104</th>
      <th>V105</th>
      <th>V106</th>
      <th>V107</th>
      <th>V108</th>
      <th>V109</th>
      <th>V110</th>
      <th>V111</th>
      <th>V112</th>
      <th>V113</th>
      <th>V114</th>
      <th>V115</th>
      <th>V116</th>
      <th>V117</th>
      <th>V118</th>
      <th>V119</th>
      <th>V120</th>
      <th>V121</th>
      <th>V122</th>
      <th>V123</th>
      <th>V124</th>
      <th>V125</th>
      <th>V281</th>
      <th>V282</th>
      <th>V283</th>
      <th>V284</th>
      <th>V285</th>
      <th>V286</th>
      <th>V287</th>
      <th>V288</th>
      <th>V289</th>
      <th>V290</th>
      <th>V291</th>
      <th>V292</th>
      <th>V296</th>
      <th>V297</th>
      <th>V298</th>
      <th>V299</th>
      <th>V300</th>
      <th>V301</th>
      <th>V302</th>
      <th>V303</th>
      <th>V304</th>
      <th>V305</th>
      <th>pca_error</th>
      <th>TransactionID</th>
      <th>id_01</th>
      <th>id_02</th>
      <th>id_11</th>
      <th>id_12</th>
      <th>id_15</th>
      <th>id_28</th>
      <th>id_29</th>
      <th>id_35</th>
      <th>id_36</th>
      <th>id_37</th>
      <th>id_38</th>
      <th>DeviceType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>86400.0</td>
      <td>68.5</td>
      <td>13926.0</td>
      <td>357.13153</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
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
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>79.0</td>
      <td>1.0</td>
      <td>75.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>0.002857</td>
      <td>2987000.0</td>
      <td>41.0</td>
      <td>206475.041812</td>
      <td>99.687317</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>86401.0</td>
      <td>29.0</td>
      <td>2755.0</td>
      <td>404.00000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>79.0</td>
      <td>2.0</td>
      <td>50.0</td>
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
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>0.000613</td>
      <td>2987001.0</td>
      <td>41.0</td>
      <td>206475.041812</td>
      <td>99.687317</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>86469.0</td>
      <td>59.0</td>
      <td>4663.0</td>
      <td>490.00000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>79.0</td>
      <td>3.0</td>
      <td>101.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>0.000534</td>
      <td>2987002.0</td>
      <td>41.0</td>
      <td>206475.041812</td>
      <td>99.687317</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
#filter data of interest
df_all = temp_df[cols].copy()
```

# Step 2: Feature Engineering

Next we would like to extend our set of features by engineering some new ones. This process will allow us to isolate some key information from the existing features, and even highlight various patterns.
All feature-engineering methods used in this post are contained in the python script  ```fraud_feat_engineering.py```.
```python
from fraud_feat_engineering import *
```



## 2.1 Datetime Features
The fist set of engineered features are extracted from the timedelta's in the column called ```TransactionDT```.
```python
period_feats=addDatetimeFeats(df_all)
period_feats
```
    ['month','week','yearday','hour','weekday','day']

The ```addDatetimeFeats``` method converts the timedelta's of ```TransactionDT``` to datetime objects and adds datetime-like columns to ```df_all``` whose names are returned in a python list.

## 2.2 Interaction Features
The second type of features is obtained by adding the values of ```card_``` and ```addr_``` columns.
```python
card_addr_feats = addCardAddressInteractionFeats(df_all)
card_addr_feats
```
    ['card12','card1235','card1235_addr12']

We can also add interactions among the ```card_``` columns and ```period_feats```.
```python
card_addr_all = card_addr_feats + ['card1','card2','card3','card5']
card_date_feats = addDatetimeInteractionFeats(df_all, cols=card_addr_all, period_cols=period_feats);
card_date_feats;
```

## 2.3 Aggregated Features
Another type of features we can add are aggregations of the ```TransactionAmt``` column by grouping-by with ```card_addr_feats``` and computing the mean and std.
```python
agg_feats = addAggTransAmtFeats(df_all,cols=card_addr_feats);
```

### 2.4. Indicator/Frequency Features
At last we can add features which contain the value counts of the input columns
```python
try_cols = card_addr_feats \
          + ['C{}'.format(i) for i in range(1,15)] \
          + ['D{}'.format(i) for i in range(1,9)] \
          + ['addr1','addr2','dist1','dist2'] \
          + ['P_emaildomain', 'R_emaildomain'] \
          + ['DeviceInfo','DeviceType'] \
          + ['id_30','id_33']
freq_feats = addFrequencyFeats(df_all,cols=try_cols);
```

# Step 3: Preprocessing and Feature Selection
To begin our preprocessing we need to combine the target column ```isFraud``` with the train data ```df_train```.

```python
# Import labels
df_fraud = pd.read_csv('./Data/train_transaction.csv',usecols = ['isFraud'])

# Select only train data
df_train = df_all.iloc[:len(df_fraud),:].copy()

# Add isFraud col to train data
df_train['isFraud'] = df_fraud['isFraud'].values
```

## 3.1 Filter Features by Correlation to target
To this end, we would like to filter out all those features which are weakly correlated to the target variables in ```isFraud```. We will use the ```getCorrelatedFeatures``` method in ```fraud_pre_proc.py```.
```python
# Generate a list of Variable objects
all_vars = TableDescriptor(df_train,'All_data','isFraud')
```

```python
# Convert numerical cols to numerical vars
numerical_vars = [var for var in all_vars.variables if var.name in num_cols+new_cols]

# Select high-correlated num vars
num_vars = getCorrelatedFeatures(numerical_vars,corr_cut_off=0.005)

# Create a list of high-correlated num cols
new_num_cols = [var.name for var in num_vars]
```
    Selected features: 41/60

```python
# Convert categorical cols to cat vars
categorical_vars = [var for var in all_vars.variables if var.name in cat_cols]

# Select high-correlated cat vars
cat_vars = getCorrelatedFeatures(categorical_vars,corr_cut_off=0.1)

# Create a list of high-correlated cat cols
new_cat_cols = [var.name for var in cat_vars]
```
    Selected features: 18/42
    
## 3.2. Convert categorical data to Dummies or Codes


```python
all_cols = new_cat_cols + new_num_cols
```


```python
#all_cols.remove('TransactionDT')
```


```python
df_all = to_categorical(df=df_all[all_cols],cat_cols=new_cat_cols,how='dummies')
df_all.shape
```


```python
print_null_cols(df_all)
```

## 3.3 PCA


```python
# # feature scalingmodel.best_performance
num_cols.remove('TransactionDT')
X_train = df_all[new_num_cols].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)

X_train_scaled=scaler.transform(X_train)


PCAT = PCATransformer(X_train_scaled)
```


```python
df_all['pca_error'] = PCAT.rec_error(X_train_scaled).reshape(-1,1)
```

# 3.4. Stratified Split training and validation data


```python
df_train_dummy = df_all.iloc[:len(df_fraud),:].copy()
```


```python
x_cols = [col for col in df_train_dummy.columns.tolist() if col not in ['TransactionID','isFraud']]
y_col = 'isFraud'
```


```python
#define X and y of df_train

X, y = df_train_dummy.loc[:,x_cols].values, df_train.loc[:,y_col].values

X_train, X_test, y_train, y_test = getStratifiedTrainTestSplit(X,y,frac=0.2,n_splits=1,
                                                                random_state=0)
```


```python
#df's shapes

for i in [X_train, X_test, y_train, y_test]:
    print(i.shape)
```


```python
df_test = df_all.loc[len(df_fraud):,x_cols].copy()
df_test.shape
```

## 3.5 Save Analyzed Data


```python
from pickleObjects import *
```


```python
path = './Data/'

dumpObjects(X_train,path+'X_train')
dumpObjects(y_train,path+'y_train')
dumpObjects(X_test,path+'X_test')
dumpObjects(y_test,path+'y_test')
dumpObjects(df_test.values,path+'X_test_submission')
```


# Summary: The Tuning Knobs

## Step 1: Data Cleaning
- Set the cut-off value ```nans_rate_cut_off``` for the percent of NaNs in a column.
- Set the min value ```min_categories``` of unique values in a columns, used for classifying a numerical-type column as categorical.

## Step 2: Feature Engineering
Select manually:

2.1 Which Datetime Feats to add?

2.2 Which Card-Address and/or Card-Address-Datetime Interactions to add?

2.3 Which Aggregated TransactionAmt Feats to add?

2.4 Which Frequency Feats to add?

## Step 3: Preprocessing and Feature Selection
Numerical_categorical split: min_categories (parameter)
3.1 Number of highly correlated features: corr_cut_off (parameter)
3.2 Method of treating categorical feats: how = {'dummies','label_enc'}
3.3 (PCA)
3.4. Stratified split parameters: frac, n_splits

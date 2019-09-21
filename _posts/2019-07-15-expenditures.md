---
title: "Data Aggregations: Expenditures"
date: 2019-07-15
tags: [data analysis, data science, pandas, numpy]
excerpt: Data Analysis, Data Science, pandas, numpy
---

# Analyzing House Office Expenditure Data

Members of Congress and Congressional offices receive an annual budget to spend on staff, supplies, transportation, and other expenses. Each quarter, representatives report the recipients of their expenditures. ProPublica complies these reports into research-ready CSV files and publishes them [here](https://projects.propublica.org/represent/expenditures). We will study the detailed (not summary) data.

Note: There is an updated version of the 2015Q2 file in the zip archive; use this and discard the original. For convenience rename this file to "2015Q2-house-disburse-detail.csv".


```python
import pandas as pd
import numpy as np
```


```python
#Generate file paths and store them in dict called paths
paths={}
for year in range(2010,2018):
    for quarter in range(1,5):
        key='{}Q{}'.format(year,quarter)
        path='{}-house-disburse-detail.csv'.format(key)
        paths[key]=path
for key in ['2009Q3','2009Q4','2018Q1']:   
    path='{}-house-disburse-detail.csv'.format(key)
    paths[key]=path
paths;
```


```python
total_memory=0
for path in paths.values():
    df_memory=pd.read_csv(path,engine='python').memory_usage(deep=True).sum()
    total_memory+=df_memory

print("Total Memory Usage = {:.2f}GB".format(total_memory/1024**3))   

```

    Total Memory Usage = 3.15GB


Let's investigate the data in more detail to see how we should handle memory usage in the following tasks.


```python
#View a sample of data
sample=pd.read_csv(paths['2011Q2'],engine='python')
sample.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 112855 entries, 0 to 112854
    Data columns (total 15 columns):
    BIOGUIDE_ID      92390 non-null object
    OFFICE           112855 non-null object
    QUARTER          112855 non-null object
    CATEGORY         112855 non-null object
    DATE             100229 non-null object
    PAYEE            98342 non-null object
    START DATE       112855 non-null object
    END DATE         112855 non-null object
    PURPOSE          112855 non-null object
    AMOUNT           112855 non-null object
    YEAR             112855 non-null object
    TRANSCODE        100229 non-null object
    TRANSCODELONG    100229 non-null object
    RECORDID         100229 non-null object
    RECIP (orig.)    98342 non-null object
    dtypes: object(15)
    memory usage: 12.9+ MB



```python
sample.head()
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
      <th>BIOGUIDE_ID</th>
      <th>OFFICE</th>
      <th>QUARTER</th>
      <th>CATEGORY</th>
      <th>DATE</th>
      <th>PAYEE</th>
      <th>START DATE</th>
      <th>END DATE</th>
      <th>PURPOSE</th>
      <th>AMOUNT</th>
      <th>YEAR</th>
      <th>TRANSCODE</th>
      <th>TRANSCODELONG</th>
      <th>RECORDID</th>
      <th>RECIP (orig.)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>OFFICE OF THE SPEAKER</td>
      <td>2011Q2</td>
      <td>PERSONNEL COMPENSATION</td>
      <td>NaN</td>
      <td>CASSIDY, ED</td>
      <td>04/01/11</td>
      <td>06/30/11</td>
      <td>DIRECTOR OF HOUSE OPERATIONS</td>
      <td>42,000.00</td>
      <td>FISCAL YEAR 2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CASSIDY, ED</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>OFFICE OF THE SPEAKER</td>
      <td>2011Q2</td>
      <td>PERSONNEL COMPENSATION</td>
      <td>NaN</td>
      <td>GREEN, JO-MARIE S</td>
      <td>04/01/11</td>
      <td>06/30/11</td>
      <td>GEN COUNSEL &amp; CHIEF OF LEG OPS</td>
      <td>42,999.99</td>
      <td>FISCAL YEAR 2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>GREEN, JO-MARIE S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>OFFICE OF THE SPEAKER</td>
      <td>2011Q2</td>
      <td>PERSONNEL COMPENSATION</td>
      <td>NaN</td>
      <td>JACKSON,BARRY S</td>
      <td>04/01/11</td>
      <td>06/30/11</td>
      <td>CHIEF OF STAFF</td>
      <td>43,125.00</td>
      <td>FISCAL YEAR 2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>JACKSON,BARRY S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>OFFICE OF THE SPEAKER</td>
      <td>2011Q2</td>
      <td>PERSONNEL COMPENSATION</td>
      <td>NaN</td>
      <td>PIERSON, JAY</td>
      <td>04/01/11</td>
      <td>06/30/11</td>
      <td>FLOOR ASSISTANT</td>
      <td>42,099.99</td>
      <td>FISCAL YEAR 2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PIERSON, JAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>OFFICE OF THE SPEAKER</td>
      <td>2011Q2</td>
      <td>PERSONNEL COMPENSATION</td>
      <td>NaN</td>
      <td>PORTER, EMILY S</td>
      <td>04/01/11</td>
      <td>06/30/11</td>
      <td>ASST TO THE SPEAKER FOR POLICY</td>
      <td>27,500.01</td>
      <td>FISCAL YEAR 2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PORTER, EMILY S</td>
    </tr>
  </tbody>
</table>
</div>



One can convince oneself that some columns are redundant since either their info is captured by other more informative columns or they don't contain any info at all.  


```python
sample.columns.tolist();
```

### Columns:
#### Informative:
'BIOGUIDE_ID'<br>
'OFFICE'<br>
'CATEGORY'<br>
'PAYEE'<br>
'START DATE'<br>
'END DATE'<br>
'PURPOSE'<br>
'AMOUNT'

#### Redundant:
'DATE<br>
'QUARTER <br>
'YEAR' <br>
'TRANSCODE'<br>
'TRANSCODELONG'<br>
'RECORDID'<br>
'RECIP (orig.)'


## Useful Functions

A quick inspection of the "AMOUNT", "START DATE" and "END DATE" columns reveals that the corresponding values are not in the right format. In particular, some of the values in the "AMOUNT" column are in string format, and similarly we the values in the date's columns. It is therefore convenient to define the following two conversion functions which will use thoroughly throughout our data analysis.


```python
def amount_numeric_converter(df):
    """
    Convert column "AMOUNTS" to numerical data
    Input: a dataframe df
    Return: the data frame column "AMOUNTS" converted to float
    """
    if df.AMOUNT.dtype =='float64': pass
    else: df.AMOUNT=df.AMOUNT.str.replace(',','').astype('float')

    return df.AMOUNT

```


```python
def date_converter(df,date_col):
    """
    Convert string dates to timestamps
    Input: dataframe (df) and a list of date columns to be converted (data_col)
    Return: df with timestamp-date columns
    """
    for col in date_col:
        df[col]=df[col].replace('   ',np.nan).fillna(method='ffill')
        try:
            df[col]=pd.to_datetime(df[col],format='%m/%d/%y')
        except ValueError:
            pass
        #some files have dates in a different string format
        try:
            df[col]=pd.to_datetime(df[col],format='%m/%d/%Y')
        except ValueError:
            print(path)

    return df


```

### 1. What is the total of all the payments in the dataset?


```python
def total_payment():
    total_payment=0
    for path in paths.values():
        df=pd.read_csv(path,engine='python',usecols=['AMOUNT'])
        payment=amount_numeric_converter(df).sum()
        total_payment += payment

    return total_payment

```


```python
print("Total of all Payments = ${:,.2f}".format(total_payment()))   
```

    Total of all Payments = $13,660,703,793.31


### 2. Define the 'COVERAGE PERIOD' for each payment as the difference (in days) between 'END DATE' and 'START DATE'. What is the standard deviation in 'COVERAGE PERIOD'? Consider only payments with strictly positive amounts.


```python
def std_coverage_period():
    cov_per_data=np.empty((0,))
    for path in paths.values():
        df=pd.read_csv(path,engine='python',usecols=['START DATE','END DATE','AMOUNT'])
        df['AMOUNT']=amount_numeric_converter(df)
        df=df[df['AMOUNT']>0].drop(['AMOUNT'],axis=1) #select only data with strictly positive amounts
        df=date_converter(df,['START DATE','END DATE'])
        df['COVERAGE_PERIOD']=(df.iloc[:,1]-df.iloc[:,0]).dt.days #compute time difference of End/Start dates
        cov_per_data=np.append(cov_per_data, df.COVERAGE_PERIOD.get_values())  
    return np.std(cov_per_data)

```


```python
print("STD of Coverage Period = {:,.2f} days".format(std_coverage_period()))
```

    STD of Coverage Period = 61.41 days


### 3. What was the average annual expenditure with a 'START DATE' date between Jan 1, 2010 and Dec 31, 2016 (inclusive)? Consider only payments with strictly positive amounts.


```python
def avg_annual_expenditure():
    annual_expenditures=[]
    for path in paths.values():
        df=pd.read_csv(path,engine='python',usecols=['START DATE','AMOUNT'])

        df=date_converter(df,['START DATE'])
        df['START_YEAR']=df['START DATE'].dt.year

        #select only data from 2010-2016
        df=df[df['START_YEAR']<2017].drop(['START DATE'],axis=1)
        df=df[df['START_YEAR']>2009]

        #select only data with strictly positive amounts
        df['AMOUNT']=amount_numeric_converter(df)
        df=df[df['AMOUNT']>0]

        #find net amount per year per dataframe
        annual_expenditures.append(df.groupby('START_YEAR').sum())

        #compute net annual expenditures
        net_annual_expenditures=pd.concat(annual_expenditures).groupby(level=0).sum()


    return net_annual_expenditures.mean(axis=0)[0]


```


```python
print("Avg Annual Expenditure (2010-2016) = ${:,.2f}".format(avg_annual_expenditure()))   
```

    Avg Annual Expenditure (2010-2016) = $1,230,258,512.37


### 4. Find the 'OFFICE' with the highest total expenditures with a 'START DATE' in 2016. For this office, find the 'PURPOSE' that accounts for the highest total expenditures. What fraction of the total expenditures (all records, all offices) with a 'START DATE' in 2016 do these expenditures amount to?


```python
#Step1: Compute expenditures of ALL offices in 2016
def office_expenditures():
    offices_expenditures=[]
    for path in paths.values():
        df=pd.read_csv(path,engine='python',usecols=['START DATE','AMOUNT','OFFICE'])

        #select only data from 2016
        df=date_converter(df,['START DATE'])
        df['START_YEAR']=df['START DATE'].dt.year
        df=df[df['START_YEAR']==2016]

        df['AMOUNT']=amount_numeric_converter(df)

        offices_expenditures.append(df.groupby('OFFICE')['AMOUNT'].sum())

    return pd.concat(offices_expenditures).groupby(level=0).sum()

office_expenditures=office_expenditures()
```


```python
#Step2: Find office w/ max total expenditures in 2016
print("Office with max expenditures in 2016 = {}".format(office_expenditures.idxmax()))
print("Amount of corresponding office = ${:,.2f}".format(office_expenditures.sum(axis=0)))
```

    Office with max expenditures in 2016 = GOVERNMENT CONTRIBUTIONS
    Amount of corresponding office = $1,236,836,563.11



```python
#Step3: Find purpose w/ max total expenditures for office w/ max expenditures
def purpose_max_expenditures():
    purpose_expenditures=[]
    for path in paths.values():
        df=pd.read_csv(path,engine='python',usecols=['START DATE','AMOUNT','OFFICE','PURPOSE'])

        #Select only data for office = 'GOVERNMENT CONTRIBUTIONS'
        df=df[df.OFFICE == 'GOVERNMENT CONTRIBUTIONS'].drop(['OFFICE'],axis=1)

        #select only data from 2016
        df=date_converter(df,['START DATE'])
        df['START_YEAR']=df['START DATE'].dt.year
        df=df[df['START_YEAR']==2016]

        df['AMOUNT']=amount_numeric_converter(df)

        purpose_expenditures.append(df.groupby('PURPOSE')['AMOUNT'].sum())

        series_purp_exp=pd.concat(purpose_expenditures).groupby(level=0)\
                                                       .sum()\
                                                       .sort_values(ascending=False)

    return series_purp_exp.index[0],series_purp_exp[0] #return both name and amount of max_purpose

purpose_max_expenditures=purpose_max_expenditures()
```


```python
print("Purpose of max expenditures in 2016 = {}".format(purpose_max_expenditures[0]))
print("Amount of corresponding purpose = ${:,.2f}".format(purpose_max_expenditures[1]))
```

    Purpose of max expenditures in 2016 = FERS
    Amount of corresponding purpose = $81,451,623.46



```python
#Step 4: Compute fraction of total expenditure
print("Fraction of 'Max Purpose' of total expenditures = {}"\
      .format(purpose_max_expenditures[1]/(office_expenditures.sum(axis=0))))
```

    Fraction of 'Max Purpose' of total expenditures = 0.06585479916213956


### 5. What was the highest average staff salary among all representatives in 2016? Assume staff sizes is equal to the number of unique payees in the 'PERSONNEL COMPENSATION' category for each representative.


```python
#Step1: Compute a list of all staff salaries
def staff_salaries():
    list_staff_salaries=[]
    for path in paths.values():
        df=pd.read_csv(path,engine='python',usecols=['BIOGUIDE_ID','START DATE','AMOUNT','PAYEE','CATEGORY'])

        #filter data by category ('PERSONNEL COMPENSATION')
        df=df[df.CATEGORY=='PERSONNEL COMPENSATION'].drop(['CATEGORY'],axis=1)

        #filter data by year (2016)
        df=date_converter(df,['START DATE'])
        df=df[df['START DATE'].dt.year==2016].drop(['START DATE'],axis=1)

        if df.empty: pass
        else:
            #drop NaN
            df=df.dropna(subset=['BIOGUIDE_ID','PAYEE'])

            #Compute payees's salaries from each rep, per df
            df['AMOUNT']=amount_numeric_converter(df)
            df=df.groupby(['BIOGUIDE_ID','PAYEE']).sum()

            list_staff_salaries.append(df)

    return list_staff_salaries


list_staff_salaries=staff_salaries()
list_staff_salaries;
```


```python
#Step2: Find the highest average staff salary among all representatives in 2016
def max_avg_staff_salary():
    df_total=list_staff_salaries[0]
    for i,df in enumerate(list_staff_salaries[1:]):
        df.columns=['AMOUNT{}'.format(i+1)]
        df_total=pd.merge(df_total,df,left_index=True,right_index=True,how='outer')
    series_salaries=df_total.sum(axis=1).groupby(level=0).mean()

    return series_salaries.max()

```


```python
print("Highest avg staff salary in 2016 = ${:,.2f}".format(max_avg_staff_salary()))
```

    Highest avg staff salary in 2016 = $34,755.23

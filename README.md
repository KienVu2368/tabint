
# tabular data interpretation and explaination

## Pre processing


```python
df = pd.read_csv('df_sample.csv')
df_proc, y, pp_outp = tabular_proc(df, 'TARGET', [fill_na(), app_cat(), dummies()])
```

Unify class for pre processing class. 


```python
class cls(TBPreProc):
    @staticmethod
    def func(df, pp_outp, na_dict = None):
        ...
        return df
```

For example, fill_na class


```python
class fill_na(TBPreProc):
    @staticmethod
    def func(df, pp_outp, na_dict = None):
        na_dict = {} if na_dict is None else na_dict.copy()
        na_dict_initial = na_dict.copy()
        for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
        if len(na_dict_initial.keys()) > 0:
            df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
        pp_outp['na_dict'] = na_dict
        return df
```

## Dataset

Dataset can be built by split method of SKlearn


```python
ds = TBDataset.from_SKSplit(df_proc, y, cons, cats, ratio = 0.2)
```

Or by split method of tabint. This method will try to keep the same distribution of categories between training set and validation set.


```python
ds = TBDataset.from_TBSplit(df_proc, y, cons, cats, ratio = 0.2)
```

Dataset class contain training set, validation set and test set.

## Learner

Learner class unify training method from sklearn model


```python
learner = LGBLearner()
params = {'task': 'train', 'objective': 'binary', 'metric':'binary_logloss'}
learner.fit(params, *ds.trn, *ds.val)
```

    Training until validation scores don't improve for 100 rounds.
    [100]	training's binary_logloss: 0.23814	valid_1's binary_logloss: 0.247746
    Did not meet early stopping. Best iteration is:
    [100]	training's binary_logloss: 0.23814	valid_1's binary_logloss: 0.247746

LGBM model

```python
learner = SKLearner(RandomForestClassifier())
learner.fit(*ds.trn, *ds.val)
```

    trn accuracy:  0.985715911677669
    val accuracy:  0.9176300343072695

and XGB model (WIP)

## Feature correlation

tabint use đenogram for easy to see and pick features with high correlation

```python
ddg = Dendogram.from_df(ds.x_trn)
```


```python
ddg.plot()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/KienVu2368/tabint/master/docs/feature_correlation.png" />
</p>

## Feature importance


tabint use [permutation importance](http://explained.ai/rf-importance/index.html). Each column or group of columns in validation set in dataset will be permute to calculate the importance.


```python
group_cols = [['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY'], ['FLAG_OWN_CAR_N', 'OWN_CAR_AGE_na']]
```


```python
impt = Importance.from_Learner(learner, ds, group_cols)
```


```python
impt.plot()
```


<p align="center">
  <img src="https://raw.githubusercontent.com/KienVu2368/tabint/master/docs/feature_importance.png" />
</p>


We can easily get the most importance feature by method in Importance class


```python
impt.top_features(24)
```


```
['EXT_SOURCE_3', 'EXT_SOURCE_2', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'EXT_SOURCE_1', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'NAME_EDUCATION_TYPE_Higher education', 'DAYS_ID_PUBLISH', 'NAME_CONTRACT_TYPE_Cash loans', 'NAME_FAMILY_STATUS_Married', 'CODE_GENDER_F', 'EXT_SOURCE_1_na', 'CODE_GENDER_M', 'OWN_CAR_AGE', 'FLAG_OWN_CAR_N', 'OWN_CAR_AGE_na', 'NAME_EDUCATION_TYPE_Secondary / secondary special', 'DAYS_LAST_PHONE_CHANGE', 'NAME_INCOME_TYPE_Working', 'ORGANIZATION_TYPE', 'OCCUPATION_TYPE', 'FLAG_DOCUMENT_16', 'REG_CITY_NOT_LIVE_CITY', 'FLAG_WORK_PHONE', 'NAME_FAMILY_STATUS_Widow']
```



Dataset class have method that can simultaneously edit training set, validation set and test set.

Drop method can drop one or many columns in training set, validation set and test set.


```python
ds.drop('DAYS_LAST_PHONE_CHANGE_na')
```

Or if we need to keep only importance columns that we found above. Just use keep method from dataset.


```python
impt_features = impt.top_features(24)
ds.keep(impt_features)
```

Dataset class in tabint also can simultaneously apply a funciton to training set, validation set and test set

```python
ds.apply('DAYS_BIRTH', lambda df: -df['DAYS_BIRTH']/365)
```

Or we can pass many transformation function at once.

```python
tfs =  {'drop 1': ['AMT_REQ_CREDIT_BUREAU_HOUR_na', 'AMT_REQ_CREDIT_BUREAU_YEAR_na'],
    
        'apply':{'DAYS_BIRTH': lambda df: -df['DAYS_BIRTH']/365,
                 'DAYS_EMPLOYED': lambda df: -df['DAYS_EMPLOYED']/365,
                 'NEW_EXT_SOURCES_MEAN': lambda df: df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1, skipna=True),
                 'NEW_EXT_SOURCES_GEO': lambda df: (df['EXT_SOURCE_1']*df['EXT_SOURCE_2']*df['EXT_SOURCE_3'])**(1/3),
                 'AMT_CREDIT/AMT_GOODS_PRICE': lambda df: df['AMT_CREDIT']/df['AMT_GOODS_PRICE'],
                 'AMT_CREDIT/AMT_CREDIT': lambda df: df['AMT_CREDIT']/df['AMT_CREDIT'],
                 'DAYS_EMPLOYED/DAYS_BIRTH': lambda df: df['DAYS_EMPLOYED']/df['DAYS_BIRTH'],
                 'DAYS_BIRTH*EXT_SOURCE_1_na': lambda df: df['DAYS_BIRTH']*df['EXT_SOURCE_1_na']},
    
        'drop 2': ['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE']}

ds.transform(tfs)
```
## Model performance

### Classification problem
#### Receiver operating characteristic

```python
roc = ReceiverOperatingCharacteristic.from_learner(learner, *ds.val)
roc.plot()
```
<p align="center">
  <img src="/media/zero/dropbox/Dropbox/Data science project/tabint/docs/roc.png" />
</p>

#### Probability distribution

```python
kde = KernelDensityEstimation.from_learner(learner, *ds.val)
kde.plot()
```
<p align="center">
  <img src="/media/zero/dropbox/Dropbox/Data science project/tabint/docs/prob_dist.png" />
</p>

#### Precision and Recall

```python
pr = PrecisionRecall.from_series(db.valid_ds.y, preds)
pr.plot()
```
<p align="center">
  <img src="/media/zero/dropbox/Dropbox/Data science project/tabint/docs/prescision_n_recall.png" />
</p>

### Regression problem

#### Actual vs Predict
```python
avp = actual_vs_predict.from_learner(learner, ds)
avp.plot(hue = 'Height')
```
<p align="center">
  <img src="/media/zero/dropbox/Dropbox/Data science project/tabint/docs/actual_vs_predict.png" />
</p>


## Interpretation and explaination

### Partial dependence

tabint use [PDPbox library](https://github.com/SauceCat/PDPbox) to visualize partial dependence.


```python
pdp = PartialDependence.from_Learner(learner, ds)
```

#### info target plot

```python
pdp.info_target_plot('EXT_SOURCE_3')
```


<p align="center">
  <img src="https://raw.githubusercontent.com/KienVu2368/tabint/master/docs/pdp_target_plot.png" />
</p>


We can see result as table

```python
pdp.info_target_data()
```

#### isolate plot

```python
pdp.isolate_plot('EXT_SOURCE_3')
```


<p align="center">
  <img src="https://raw.githubusercontent.com/KienVu2368/tabint/master/docs/PDP_plot.png" />
</p>


### Tree interpreter


```python
Tf = Traterfall.from_SKTree(learner, ds.x_trn, 3)
```


```python
Tf.plot(formatting = "$ {:,.3f}")
```

<p align="center">
  <img src="https://raw.githubusercontent.com/KienVu2368/tabint/master/docs/Traterfall.png" />
</p>

We can see and filter result table

```python
Tf.data.pos(5)
Tf.data.neg(5)
```

### SHAP

tabint visual SHAP values from [SHAP library](https://github.com/slundberg/shap). SHAP library use red and blue for default color. tabint change these color to green and blue for easy to see and consistence with pdpbox library.


```python
Shap = SHAP.from_Tree(learner, ds)
```

#### force plot

```python
Shap.one_force_plot(3)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/KienVu2368/tabint/master/docs/shap_force_plot.png" />
</p>

And we can see table result also.

```python
Shap.one_force_data.pos(5)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column name</th>
      <th>Column value</th>
      <th>Shap value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EXT_SOURCE_3</td>
      <td>0.253963</td>
      <td>0.720332</td>
    </tr>
    <tr>
      <th>5</th>
      <td>EXT_SOURCE_1</td>
      <td>0.290425</td>
      <td>0.185534</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AMT_GOODS_PRICE</td>
      <td>247500.000000</td>
      <td>0.127218</td>
    </tr>
    <tr>
      <th>24</th>
      <td>REG_CITY_NOT_LIVE_CITY</td>
      <td>1.000000</td>
      <td>0.071174</td>
    </tr>
    <tr>
      <th>9</th>
      <td>DAYS_ID_PUBLISH</td>
      <td>-520.000000</td>
      <td>0.052655</td>
    </tr>
  </tbody>
</table>
</div>




```python
Shap.one_force_data.neg(5)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column name</th>
      <th>Column value</th>
      <th>Shap value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>AMT_CREDIT</td>
      <td>267102.0</td>
      <td>-0.260664</td>
    </tr>
    <tr>
      <th>15</th>
      <td>OWN_CAR_AGE</td>
      <td>3.0</td>
      <td>-0.128298</td>
    </tr>
    <tr>
      <th>16</th>
      <td>FLAG_OWN_CAR_N</td>
      <td>0.0</td>
      <td>-0.068803</td>
    </tr>
    <tr>
      <th>13</th>
      <td>EXT_SOURCE_1_na</td>
      <td>0.0</td>
      <td>-0.043089</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DAYS_BIRTH</td>
      <td>-11185.0</td>
      <td>-0.032210</td>
    </tr>
  </tbody>
</table>
</div>


#### dependence plot

```python
Shap.dependence_plot('EXT_SOURCE_2')
```

<p align="center">
  <img src="https://raw.githubusercontent.com/KienVu2368/tabint/master/docs/Shap.png" />
</p>

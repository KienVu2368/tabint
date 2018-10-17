
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

Dataset can be build by use split method by SKlearn


```python
ds= TBDataset.from_SKSplit(df_proc, y, cons, cats, ratio = 0.2)
```

Or by split method by tabint. This method will try to keep distribution of categories between training set and validation set


```python
ds= TBDataset.from_TBSplit(df_proc, y, cons, cats, ratio = 0.2)
```

Dataset class contain training set, validation set and test set.

## Learner

Learner class unify training method from sklearn model, LGBM model and XGB model


```python
learner = LGBLearner()
params = {'task': 'train', 'objective': 'binary', 'metric':'binary_logloss'}
learner.fit(params, *ds.trn, *ds.val)
```

    Training until validation scores don't improve for 100 rounds.
    [100]	training's binary_logloss: 0.23814	valid_1's binary_logloss: 0.247746
    Did not meet early stopping. Best iteration is:
    [100]	training's binary_logloss: 0.23814	valid_1's binary_logloss: 0.247746



```python
learner = SKLearner(RandomForestClassifier())
learner.fit(*ds.trn, *ds.val)
```

    trn accuracy:  0.985715911677669
    val accuracy:  0.9176300343072695


## Feature correlation


```python
ddg = Dendogram.from_df(ds.x_trn)
```


```python
ddg.plot()
```


![png](output_19_0.png)


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


![png](output_24_0.png)


We can easily get the most importance feature by method in Importance class


```python
impt.top_features(24)
```




['EXT_SOURCE_3', 'EXT_SOURCE_2', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'EXT_SOURCE_1', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'NAME_EDUCATION_TYPE_Higher education', 'DAYS_ID_PUBLISH', 'NAME_CONTRACT_TYPE_Cash loans', 'NAME_FAMILY_STATUS_Married', 'CODE_GENDER_F', 'EXT_SOURCE_1_na', 'CODE_GENDER_M', 'OWN_CAR_AGE', 'FLAG_OWN_CAR_N', 'OWN_CAR_AGE_na', 'NAME_EDUCATION_TYPE_Secondary / secondary special', 'DAYS_LAST_PHONE_CHANGE', 'NAME_INCOME_TYPE_Working', 'ORGANIZATION_TYPE', 'OCCUPATION_TYPE', 'FLAG_DOCUMENT_16', 'REG_CITY_NOT_LIVE_CITY', 'FLAG_WORK_PHONE', 'NAME_FAMILY_STATUS_Widow']



Dataset class have method that can edit training set, validation set and test set.

Drop method can drop one or many columns in training set, validation set and test set.


```python
ds.drop('DAYS_LAST_PHONE_CHANGE_na')
```

Or if we need to keep only importance columns that we found above. Just use keep method from dataset.


```python
impt_features = impt.top_features(24)
ds.keep(impt_features)
```


```python
ds.features
```




    Index(['EXT_SOURCE_3', 'EXT_SOURCE_2', 'AMT_CREDIT', 'AMT_GOODS_PRICE',
           'AMT_ANNUITY', 'EXT_SOURCE_1', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
           'NAME_EDUCATION_TYPE_Higher education', 'DAYS_ID_PUBLISH',
           'NAME_CONTRACT_TYPE_Cash loans', 'NAME_FAMILY_STATUS_Married',
           'CODE_GENDER_F', 'EXT_SOURCE_1_na', 'CODE_GENDER_M', 'OWN_CAR_AGE',
           'FLAG_OWN_CAR_N', 'OWN_CAR_AGE_na',
           'NAME_EDUCATION_TYPE_Secondary / secondary special',
           'DAYS_LAST_PHONE_CHANGE', 'NAME_INCOME_TYPE_Working',
           'ORGANIZATION_TYPE', 'OCCUPATION_TYPE', 'FLAG_DOCUMENT_16',
           'REG_CITY_NOT_LIVE_CITY', 'FLAG_WORK_PHONE',
           'NAME_FAMILY_STATUS_Widow'],
          dtype='object')



You can read source code of dataset class for more usefull method.

## Interpretation and explaination

### Partial dependence

tabint use [PDPbox library](https://github.com/SauceCat/PDPbox) to visualize partial dependence.


```python
pdp = PartialDependence.from_Learner(learner, ds)
```


```python
pdp.info_target('EXT_SOURCE_3')
```


![png](output_38_0.png)



```python
pdp.isolate('EXT_SOURCE_3')
```


![png](output_39_0.png)


### Tree interpreter


```python
Tf = Traterfall.from_SKTree(learner, ds.x_trn, 3)
```


```python
Tf.plot(formatting = "$ {:,.3f}")
```

![png](output_42_1.png)


### SHAP

tabint visual SHAP values from [SHAP library](https://github.com/slundberg/shap). SHAP library use red and blue for default color. tabint change these color to green and blue for easy to see and consistence with pdpbox library.


```python
Shap = SHAP.from_Tree(learner, ds)
```


```python
Shap.force_plot_one(3)
```





```python
Shap.force_value_one.pos(5)
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
Shap.force_value_one.neg(5)
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




```python
Shap.dependence_plot('EXT_SOURCE_2')
```


![png](output_49_0.png)


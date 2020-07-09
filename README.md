# Welcom to Tabint
> NB: this is on development process, many things we want to develop but have not yet done. If you want to contribute please feel free to do so.


## Installing

```git clone https://github.com/KienVu2368/tabint
cd tabint
conda env create -f environment.yml
conda activate tabint```

## Pre-processing

```
import pandas as pd
df = pd.read_csv('df_sample.csv')
df_proc, y, pp_outp = tabular_proc(df, 'TARGET', [fill_na(), app_cat(), dummies()])
```

Unify class for pre processing class.

```
class cls(TBPreProc):
    @staticmethod
    def func(df, pp_outp, na_dict = None):
        ...
        return df
```

For example, fill_na class

```
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

Dataset class contain training set, validation set and test set.

Dataset can be built by split method of SKlearn

```
ds = TBDataset.from_SKSplit(df_proc, y, cons, cats, ratio = 0.2)
```

Or by split method of tabint. This method will try to keep the same distribution of categorie variables between training set and validation set.

```
ds = TBDataset.from_TBSplit(df_proc, y, cons, cats, ratio = 0.2)
```

Dataset class have method that can simultaneously edit training set, validation set and test set.

Drop method can drop one or many columns in training set, validation set and test set.

```
ds.drop('DAYS_LAST_PHONE_CHANGE_na')
```

Or if we need to keep only importance columns that we found above. Just use keep method from dataset.

```
mpt_features = impt.top_features(24)
ds.keep(impt_features)
```

Dataset class in tabint also can simultaneously apply a funciton to training set, validation set and test set

```
ds.apply('DAYS_BIRTH', lambda df: -df['DAYS_BIRTH']/365)
```

Or we can pass many transformation function at once.

```
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

## Learner

Learner class unify training method from sklearn model

```
learner = LGBLearner()
params = {'task': 'train', 'objective': 'binary', 'metric':'binary_logloss'}
learner.fit(params, *ds.trn, *ds.val)
```

LGBM model

```
learner = SKLearner(RandomForestClassifier())
learner.fit(*ds.trn, *ds.val)
```

and XGB model (WIP)

## Feature correlation

tabint use đenogram for easy to see and pick features with high correlation

```
ddg = Dendogram.from_df(ds.x_trn)
```

```
ddg.plot()
```

## Feature importance

tabint use [permutation importance](https://explained.ai/rf-importance/index.html). Each column or group of columns in validation set in dataset will be permute to calculate the importance.

```
group_cols = [['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY'], ['FLAG_OWN_CAR_N', 'OWN_CAR_AGE_na']]
```

```
impt = Importance.from_Learner(learner, ds, group_cols)
```

```
impt.plot()
```

We can easily get the most importance feature by method in Importance class

```
impt.top_features(24)
```

## Model performance

### Classification problem

#### Receiver operating characteristic

```
roc = ReceiverOperatingCharacteristic.from_learner(learner, ds)
roc.plot()
```

#### Probability distribution

```
kde = KernelDensityEstimation.from_learner(learner, ds)
kde.plot()
```

#### Precision and Recall

```
pr = PrecisionRecall.from_series(y_true, y_pred)
pr.plot()
```

### Regression problem

#### Actual vs Predict

```
avp = actual_vs_predict.from_learner(learner, ds)
avp.plot(hue = 'Height')
```

## Interpretation and explaination

### Partial dependence

tabint use [PDPbox](https://github.com/SauceCat/PDPbox) library to visualize partial dependence.

```
pdp = PartialDependence.from_Learner(learner, ds)
```

### info target plot

```
pdp.info_target_plot('EXT_SOURCE_3')
```

We can see result as table

```
pdp.info_target_data()
```

### isolate plot

```
pdp.isolate_plot('EXT_SOURCE_3')
```

### Tree interpreter

```
Tf = Traterfall.from_SKTree(learner, ds.x_trn, 3)
```

```
Tf.plot(formatting = "$ {:,.3f}")
```

We can see and filter result table

```
Tf.data.pos(5)
Tf.data.neg(5)
```

### SHAP

tabint visual SHAP values from [SHAP](https://github.com/slundberg/shap) library. SHAP library use red and blue for default color. tabint change these color to green and blue for easy to see and consistence with pdpbox library.

```
Shap = SHAP.from_Tree(learner, ds)
```

#### force plot

```
Shap.one_force_plot(3)
```

And we can see table result also.

```
Shap.one_force_data.pos(5)
```

```
Shap.one_force_data.neg(5)
```

#### dependence plot

```
Shap.dependence_plot('EXT_SOURCE_2')
```

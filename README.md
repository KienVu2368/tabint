# tabular data interpretation and explaination

## Pre processing
```
app_train = pd.read_csv('application_train.csv')

app_train_proc, y, pp_outp = tabular_proc(app_train, 'TARGET', [fill_na(), app_cat(), dummies()])
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

Stratified split between category variables
```
ds= TBDataset.from_TBSplit(app_train_proc, y, cons, cats)
```


Dataset class contain training set, validation set and test set.
```
ds.x_trn, ds.x_val, ds.x_tst
```

Dataset contain convenient method to change dataset after run model.


## Learner


## Feature

### Feature clustering

### Feature importance

# Interpretation

## Partial importance

## SHAP

# Tree interpretation



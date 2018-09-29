from .utils import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

 
#imbalance data??? http://www.chioka.in/class-imbalance-problem/
#subsampling, over sampling?? http://forums.fast.ai/t/unbalanced-data-upsampling-vs-downsampling/20406/4
#simulators? opimizers
#mixup augmentation? http://forums.fast.ai/t/mixup-data-augmentation/22764

class TBDataset:
    """
    Contain train, validation, test set
    """
    def __init__(self, x_trn, y_trn, x_val, y_val, x_tst = None):
        self.x_trn, self.y_trn = x_trn, y_trn
        self.x_val, self.y_val = x_val, y_val
        self.x_tst = x_tst

    @classmethod
    def from_SklearnSplit(cls, df, y_df, ratio = 0.2, x_tst = None, **kargs):
        """
        use sklearn split function to split data
        """
        x_trn, x_val, y_trn, y_val = train_test_split(df, y_df, test_size=ratio, stratify = y_df)
        return cls(x_trn, y_trn, x_val, y_val, x_tst)

    @classmethod
    def from_TBSplit(cls, df, y_df, x_tst, pct = 1, ratio = 0.2, tp = 'classification', **kargs):
        """
        split data smarter: https://medium.com/@kien.vu/d6b7a8dbaaf5
        still messi
        """
        if tp == 'classification':
            _, cats = get_cons_cats(df)
            
            tst_key = x_tst[cats].drop_duplicates().values
            tst_key = set('~'.join([str(j) for j in i]) for i in tst_key)

            df_key = df[cats].apply(lambda x: '~'.join([str(j) for j in x.values]), axis=1)
            mask = df_key.isin(tst_key)

            x_trn, y_trn = df[~mask], y_df[~mask]
            x_val_set, y_val_set = df[mask], y_df[mask]

            x_val = x_val_set.groupby(cats).apply(cls.random_choose, pct, ratio, **kargs)
            val_index = set([i[-1] for i in x_val.index.values])
            x_val.reset_index(drop=True, inplace=True)
            
            mask = x_val_set.index.isin(val_index)
            y_val = y_val_set[mask]
            x_trn, y_trn = pd.concat([x_trn, val_set[~mask]]), pd.concat([y_trn, y_val_set[~mask]])
        else:
            None
        return cls(x_trn, y_trn, x_val, y_val, x_tst)

    @staticmethod
    def random_choose(x, pct = 1, ratio = 0.2, **kargs):
        """
        static method for from_TBSplit, random choose rows from a group
        """
        n = x.shape[0] if random.randint(0,9) < pct else int(np.round(x.shape[0]*(ratio-0.04)))
        return x.sample(n=n, **kargs)

    def val_permutation(self, cols):
        """"
        permute one or many columns of validation set. For permutation importance
        """
        cols = to_list(cols)
        df = self.x_val.copy()
        for col in cols: df[col] = np.random.permutation(df[col])
        return df

    def apply(self, col, f, inplace = True, tp = 'trn'):
        """
        apply a function f for all dataset
        """
        if inplace:
            self.x_trn[col] = f(self.x_trn)
            self.x_val[col] = f(self.x_val)
            if self.x_tst is not None: self.x_tst[col] = f(self.x_tst)
        else:
            if tp == 'tst':
                df = self.x_tst.copy()
                df[col] = f(df)
                return df
            else:
                df, y_df = (self.x_trn.copy(), self.y_trn) if tp == 'trn' else (self.x_trn.copy(), self.y_trn)
                df[col] = f(df)
                return df, y_df

    def sample(self, tp = 'trn', ratio = 0.3):
        """
        get sample of dataset
        """
        if 'tst' == tp: 
            return None if self.x_tst is None else self.x_tst.sample(self.x_tst.shape[0]*ratio)
        else:
            df, y_df = (self.x_trn[col], self.y_trn) if tp == 'trn' else (self.x_trn[col], self.y_trn)
            _, df, _, y_df = train_test_split(df, y_df, test_size = ratio, stratify = y_df)
            return df, y_df

    def keep(self, col, inplace = True, tp = 'trn'):
        """
        keep columns of dataset
        """
        if inplace:
            self.x_trn = self.x_trn[col]
            self.x_val = self.x_val[col]
            if self.x_tst is not None: self.x_tst = self.x_tst[col]
        else:
            if tp == 'tst':
                return None if self.x_tst is None else self.x_tst[col]
            else:
                return (self.x_trn[col], self.y_trn) if tp == 'trn' else (self.x_val[col], self.y_val)

    def drop(self, col, inplace = True, tp = 'trn'):
        """
        drop columns of dataset
        """
        if inplace:
            self.x_trn.drop(col, axis=1, inplace = True)
            self.x_val.drop(col, axis=1, inplace = True)
            if self.x_tst is not None: self.x_tst.drop(col, axis=1, inplace = True)
        else:
            if tp == 'tst': 
                return None if self.x_tst is None else self.x_tst.drop(col, axis = 1)
            else:
                return (self.x_trn.drop(col, axis = 1), self.y_trn) if tp == 'trn' else (self.x_val.drop(col, axis = 1), self.y_val)

    def transform(self, tfs):
        for tps in tfs.keys():
            for col in tfs[tps]:
                if tps == 'apply': self.apply(col, tfs[tps][col])
                elif tps == 'drop': self.drop(col)
                elif tps == 'keep': self.keep(col)

    @property
    def features(self): return self.x_trn.columns

    @property
    def trn(self): return self.x_trn, self.y_trn

    @property
    def val(self): return self.x_val, self.y_val


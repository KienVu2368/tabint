from .utils import *
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import random

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.validation import _num_samples, check_array
from sklearn.model_selection._split import _approximate_mode, _validate_shuffle_split
from sklearn.utils import indexable, check_random_state, safe_indexing

 
#imbalance data??? http://www.chioka.in/class-imbalance-problem/
#subsampling, over sampling?? http://forums.fast.ai/t/unbalanced-data-upsampling-vs-downsampling/20406/4
#simulators? opimizers
#mixup augmentation? http://forums.fast.ai/t/mixup-data-augmentation/22764

class TBDataset:
    """
    Contain train, validation, test set
    """
    def __init__(self, x_trn, y_trn, x_val, y_val, cons, cats, x_tst = None):
        self.x_trn, self.y_trn = x_trn, y_trn
        self.x_val, self.y_val = x_val, y_val
        self.x_tst = x_tst
        self.cons, self.cats = cons, cats

    @classmethod
    def from_SklearnSplit(cls, df, y, cons, cats, ratio = 0.2, x_tst = None, **kargs):
        """
        use sklearn split function to split data
        """
        x_trn, x_val, y_trn, y_val = train_test_split(df, y, test_size=ratio, stratify = y)
        return cls(x_trn, y_trn, x_val, y_val, cons, cats, x_tst)

    @classmethod
    def from_TBSplit(cls, df, y, cons, cats, x_tst = None, pct = 0.1, ratio = 0.2, tp = 'classification', time_df = None, **kargs):
        if tp == 'classification':
            if x_tst is None:
                df = df.copy()
                df['y_'] = y
                keys = df[cats + ['y_']].apply(lambda x: '~'.join([str(j) for j in x.values]), axis=1)

                sss = split_by_cats(train_size =1-ratio, test_size=ratio)
                train, val = next(sss.split(df, keys))                
                x_trn, x_val = safe_indexing(df, train), safe_indexing(df, val)
                
                y_trn = x_trn['y_'].copy()
                y_val = x_val['y_'].copy()
                
                x_trn.drop('y_', axis=1, inplace = True)
                x_val.drop('y_', axis=1, inplace = True)
            else:
                tst_key = x_tst[cats].drop_duplicates().values
                tst_key = set('~'.join([str(j) for j in i]) for i in tst_key)

                df_key = df[cats].apply(lambda x: '~'.join([str(j) for j in x.values]), axis=1)
                mask = df_key.isin(tst_key)

                x_trn, y_trn = df[~mask], y[~mask]
                x_val_set, y_val_set = df[mask], y[mask]

                x_val = x_val_set.groupby(cats).apply(random_choose, pct, ratio, **kargs)
                val_index = set([i[-1] for i in x_val.index.values])
                x_val.reset_index(drop=True, inplace=True)
                
                mask = x_val_set.index.isin(val_index)
                y_val = y_val_set[mask]
                x_trn, y_trn = pd.concat([x_trn, val_set[~mask]]), pd.concat([y_trn, y_val_set[~mask]])
        else:
            df = df.sort_values(by=time_df, ascending=True)
            split_id = int(df.shape*(1-ratio))
            x_trn, y_trn = df[:split_id], y[:split_id]
            x_val, y_val = df[split_id:], y[split_id:]
        return cls(x_trn, y_trn, x_val, y_val, cons, cats, x_tst)

    def val_permutation(self, cols):
        """"
        permute one or many columns of validation set. For permutation importance
        """
        cols = to_iter(cols)
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
            self.add_col_to_cons_cats(col)            
        else:
            if tp == 'tst':
                df = self.x_tst.copy()
                df[col] = f(df)
                return df
            else:
                df, y = (self.x_trn.copy(), self.y_trn) if tp == 'trn' else (self.x_trn.copy(), self.y_trn)
                df[col] = f(df)
                return df, y


    def sample(self, tp = 'trn', ratio = 0.3):
        """
        get sample of dataset
        """
        if 'tst' == tp: 
            return None if self.x_tst is None else self.x_tst.sample(self.x_tst.shape[0]*ratio)
        else:
            df, y = (self.x_trn, self.y_trn) if tp == 'trn' else (self.x_val, self.y_val)
            _, df, _, y = train_test_split(df, y, test_size = ratio, stratify = y)
            return df, y

    def keep(self, col, inplace = True, tp = 'trn'):
        """
        keep columns of dataset
        """
        if inplace:
            self.x_trn = self.x_trn[col]
            self.x_val = self.x_val[col]
            if self.x_tst is not None: self.x_tst = self.x_tst[col]
            self.remove_col_from_cons_cats(col)          
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
            self.remove_col_from_cons_cats(col)
        else:
            if tp == 'tst': 
                return None if self.x_tst is None else self.x_tst.drop(col, axis = 1)
            else:
                return (self.x_trn.drop(col, axis = 1), self.y_trn) if tp == 'trn' else (self.x_val.drop(col, axis = 1), self.y_val)

    def transform(self, tfs):
        for key in tfs.keys():
            if key[:5] == 'apply': 
                for col in tfs[key]: self.apply(col, tfs[key][col])
            elif key[:4] == 'drop': self.drop(tfs[key])
            elif key[:4] == 'keep': self.keep(tfs[key])

    def remove_col_from_cons_cats(self, col):
        self.cons = [i for i in self.cons if i not in to_iter(col)]
        self.cats = [i for i in self.cats if i not in to_iter(col)]

    def add_col_to_cons_cats(self, col):
        if col not in self.cons and col not in self.cats: 
            if is_numeric_dtype(self.x_trn[col].values): self.cons.append(col)
            else: self.cats.append(col)

    @property
    def features(self): return self.x_trn.columns

    @property
    def trn(self): return self.x_trn, self.y_trn

    @property
    def val(self): return self.x_val, self.y_val


def random_choose(x, pct = 0.1, ratio = 0.2, **kargs):
    """
    static method for from_TBSplit, random choose rows from a group
    """
    n = x.shape[0] if random.uniform(0,1) <= pct else int(np.round(x.shape[0]*(ratio-0.04)))
    return x.sample(n=n, **kargs)


class ResultDF:
    def __init__(self, df, cons):
        self.result = df
        self.cons = to_iter(cons)
        self.len = df.shape[0]
    
    def __call__(self): return self.result
    
    def top(self, n=None, col=None): 
        return self.result.sort_values(by=col or self.cons, ascending=False)[:(n or self.len)]
    
    def pos(self, n=None, col=None): 
        col = col or self.cons[0]
        return self.result[self.result[col]>=0].sort_values(by=col, ascending=False)[:(n or self.len)]
    
    def neg(self, n=None, col=None):
        col = col or self.cons[0]
        return self.result[self.result[col]<=0].sort_values(by=col, ascending=True)[:(n or self.len)]




class split_by_cats(StratifiedShuffleSplit):
    def _iter_indices(self, X, y, groups=None):
        n_samples = _num_samples(X)
        y = check_array(y, ensure_2d=False, dtype=None)
        n_train, n_test = _validate_shuffle_split(n_samples, self.test_size,
                                                  self.train_size)

        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([' '.join(row.astype('str')) for row in y])

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)
        #if np.min(class_counts) < 2:
        #    raise ValueError("The least populated class in y has only 1"
        #                     " member, which is too few. The minimum"
        #                     " number of groups for any class cannot"
        #                     " be less than 2.")

        if n_train < n_classes:
            raise ValueError('The train_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (n_train, n_classes))
        if n_test < n_classes:
            raise ValueError('The test_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (n_test, n_classes))

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(np.argsort(y_indices, kind='mergesort'),
                                 np.cumsum(class_counts)[:-1])

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            # if there are ties in the class-counts, we want
            # to make sure to break them anew in each iteration
            n_i = _approximate_mode(class_counts, n_train, rng)
            class_counts_remaining = class_counts - n_i
            t_i = _approximate_mode(class_counts_remaining, n_test, rng)

            train = []
            test = []

            for i in range(n_classes):
                permutation = rng.permutation(class_counts[i])
                perm_indices_class_i = class_indices[i].take(permutation,
                                                             mode='clip')

                train.extend(perm_indices_class_i[:n_i[i]])
                test.extend(perm_indices_class_i[n_i[i]:n_i[i] + t_i[i]])

            train = rng.permutation(train)
            test = rng.permutation(test)

            yield train, test
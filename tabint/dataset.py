from .utils import *
from .pre_processing import *
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
    def from_SKSplit(cls, df, y = None, y_field = None, cats = None, cons = None, ratio = 0.2, x_tst = None, **kargs):
        """
        use sklearn split function to split data
        """
        if y is None: y = df[y_field]; df = df.drop(y_field, axis = 1) 

        stratify = None if y.dtype.name[:5] == 'float' else y
        x_trn, x_val, y_trn, y_val = train_test_split(df, y, test_size=ratio, stratify = stratify, **kargs)
        
        if cons is None and cats is not None: cons = [i for i in df.columns if i not in cats]
        if cons is not None and cats is None: cats = [i for i in df.columns if i not in cons]
        
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

    def val_permutation(self, features):
        """"
        permute one or many columns of validation set. For permutation importance
        """
        features = to_iter(features)
        df = self.x_val.copy()
        for ft in features: df[ft] = np.random.permutation(df[ft])
        return df

    def apply(self, feature, func, inplace = True, tp = 'trn'):
        """
        apply a function f for all dataset
        """
        if inplace:
            self.x_trn[feature] = func(self.x_trn)
            self.x_val[feature] = func(self.x_val)
            if self.x_tst is not None: self.x_tst[feature] = func(self.x_tst)
            self.add_col_to_cons_cats(feature)            
        else:
            if tp == 'tst':
                df = self.x_tst.copy()
                df[feature] = func(df)
                return df
            else:
                df, y = (self.x_trn.copy(), self.y_trn) if tp == 'trn' else (self.x_trn.copy(), self.y_trn)
                df[feature] = func(df)
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

    def keep(self, feature, inplace = True, tp = 'trn'):
        """
        keep columns of dataset
        """
        if inplace:
            self.x_trn = self.x_trn[feature]
            self.x_val = self.x_val[feature]
            if self.x_tst is not None: self.x_tst = self.x_tst[feature]
            self.keep_col_from_cons_cats(feature)          
        else:
            if tp == 'tst':
                return None if self.x_tst is None else self.x_tst[feature]
            else:
                return (self.x_trn[feature], self.y_trn) if tp == 'trn' else (self.x_val[feature], self.y_val)

    def drop(self, feature, inplace = True, tp = 'trn'):
        """
        drop columns of dataset
        """
        if inplace:
            self.x_trn.drop(feature, axis=1, inplace = True)
            self.x_val.drop(feature, axis=1, inplace = True)
            if self.x_tst is not None: self.x_tst.drop(feature, axis=1, inplace = True)
            self.remove_col_from_cons_cats(feature)
        else:
            if tp == 'tst': 
                return None if self.x_tst is None else self.x_tst.drop(feature, axis = 1)
            else:
                return (self.x_trn.drop(feature, axis = 1), self.y_trn) if tp == 'trn' else (self.x_val.drop(feature, axis = 1), self.y_val)

    def transform(self, tfms):
        for key in tfms.keys():
            if key[:5] == 'apply': 
                for feature in tfms[key]: self.apply(feature, tfms[key][feature])
            elif key[:4] == 'drop': self.drop(tfms[key])
            elif key[:4] == 'keep': self.keep(tfms[key])

    def remove_col_from_cons_cats(self, feature):
        self.cons = [i for i in self.cons if i not in to_iter(feature)]
        self.cats = [i for i in self.cats if i not in to_iter(feature)]

    def keep_col_from_cons_cats(self, feature):
        self.cons = [i for i in self.cons if i in to_iter(feature)]
        self.cats = [i for i in self.cats if i in to_iter(feature)]


    def add_col_to_cons_cats(self, feature):
        if feature not in self.features: 
            if is_numeric_dtype(self.x_trn[feature].values): self.cons.append(feature)
            else: self.cats.append(feature)

    def remove_outlier(self, features = None, inplace = True, tp = 'trn'):
        features = features or self.cons
        if inplace:
            self.x_trn, mask = filter_outlier(self.x_trn, features)
            self.y_trn = self.y_trn[mask]

            self.x_val, mask = filter_outlier(self.x_val, features)
            self.y_val = self.y_val[mask]

            self.x_tst = filter_outlier(self.x_tst, features)[0]
        else:
            if tp == 'tst': return filter_outlier(self.x_trn, features)[0]
            else:
                if tp == 'trn': 
                    df, mask = filter_outlier(self.x_trn, features)
                    y = self.y_trn[mask]
                else: 
                    df, mask = filter_outlier(self.x_val, features)
                    y = self.y_val[mask]
                return df, y

    @property
    def features(self): return self.x_trn.columns

    @property
    def trn(self): return self.x_trn, self.y_trn

    @property
    def n_trn(self): return self.x_trn.shape[0]

    @property
    def val(self): return self.x_val, self.y_val

    @property
    def n_val(self): return self.x_val.shape[0]


def random_choose(x, pct = 0.1, ratio = 0.2, **kargs):
    """
    static method for from_TBSplit, random choose rows from a group
    """
    n = x.shape[0] if random.uniform(0,1) <= pct else int(np.round(x.shape[0]*(ratio-0.04)))
    return x.sample(n=n, **kargs)


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
        if np.min(class_counts) < 2:
            print(ValueError("The least populated class in y has only 1"
                             " member, which is too few. The minimum"
                             " number of groups for any class cannot"
                             " be less than 2."))

        if n_train < n_classes:
            print(ValueError('The train_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (n_train, n_classes)))
        if n_test < n_classes:
            print(ValueError('The test_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (n_test, n_classes)))

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
from .utils import *
from .pre_processing import *
from .transform import *
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
    def __init__(self, x_trn, x_val, x_tst, x_tfms, y_trn, y_val, y_tfms):            
        self.x_trn, self.y_trn, self.x_tst = x_trn, y_trn, x_tst
        self.x_val, self.y_val = x_val, y_val
        self.x_tfms, self.y_tfms = x_tfms, y_tfms

    @classmethod
    def from_Split(cls, df, y = None, y_field = None, tp = '_',
                    cats = None, x_tst = None, time_feature = None, ratio = 0.2, 
                     x_tfms = None, y_tfms = None, **kargs):
        """
        use sklearn split function to split data
        """
        df = df.copy()
        if y is None: y = df[y_field]; df = df.drop(y_field, axis = 1)
            
        if tp != 'time series': x_trn, y_trn, x_val, y_val = stratify_split(df, y, x_tfms.cats, ratio)
        else: x_trn, y_trn, x_val, y_val = split_time_series(df, y, time_feature, ratio)
        
        #x_trn, x_val, x_tst, y_trn, y_val, x_tfms, y_tfms = cls.transform_data(x_trn, x_val, x_tst, y_trn, y_val, x_tfms, y_tfms)
        x_tfms, y_tfms = None, None 
        return cls(x_trn, x_val, x_tst, x_tfms, y_trn, y_val, y_tfms)
    
    @staticmethod
    def transform_data(x_trn, x_val, x_tst, y_trn, y_val, x_tfms, y_tfms):
        if x_tfms is None: x_tfms = noop_transform
        x_tfms.fit(x_trn)
        x_trn = x_tfms.transform(x_trn)
        x_val = x_tfms.transform(x_val)
        if x_tst is not None: x_tst = x_tfms.transform(x_tst)
        
        if y_tfms is None: y_tfms = noop_transform
        y_tfms.fit(y_trn)
        y_trn = y_tfms.transform(y_trn)
        y_val = y_tfms.transform(y_val)
            
        return x_trn, x_val, x_tst, y_trn, y_val, x_tfms, y_tfms
            
    def val_permutation(self, features):
        """"
        permute one or many columns of validation set. For permutation importance
        """
        features = to_iter(features)
        df = self.x_val.copy()
        for ft in features: df[ft] = np.random.permutation(df[ft])
        return df

    def apply_function(self, feature, function_dict, inplace = True, tp = 'trn'):
        """
        apply a function f for all dataset
        """
        features = to_iter(features)
        step = apply_function(function_dict).fit(self.x_trn)
        self.apply_step(step, features, inplace, tp)

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

    def select(self, features, inplace = True, tp = 'trn'):
        """
        keep columns of dataset
        """
        features = to_iter(features)
        step = select(features).fit(self.x_trn)
        self.apply_step(step, features, inplace, tp)
        
    def drop(self, feature, inplace = True, tp = 'trn'):
        """
        drop columns of dataset
        """
        features = to_iter(features)
        step = drop_features(features).fit(self.x_trn)
        self.apply_step(step, features, inplace, tp)
            
    def remove_outlier(self, features = None, inplace = True, tp = 'trn'):
        features = features or self.cons
        features = to_iter(features)
        mask_trn = self.get_mask_outlier(self.x_trn, features)
        mask_val = self.get_mask_outlier(self.x_val, features)
        if inplace:
            self.x_trn, self.y_trn = self.x_trn[mask_trn], self.y_trn[mask_trn]
            self.x_val, self.y_val = self.x_val[mask_val], self.y_val[mask_val]
        else:
            return (self.x_trn[mask_trn], self.y_trn[mask_trn]) if tp == 'trn' else (self.x_val[mask_val], self.y_val[mask_val])
    
    def get_mask_outlier(self, df, features):
        step = remove_outlier(features)
        step.fit(df)
        _ = step.transform(df)
        mask = step.mask
        return mask
    
    
    def apply_step(self, step, features, inplace, tp):
        if inplace:
            x_tfms.append(step)
            self.x_trn = step.transform(self.x_trn)
            self.x_val = step.transform(self.x_val)
            if self.x_tst is not None: self.x_tst = step.transform(self.x_tst)
            x_tfms.get_features(self.x_trn)    
        else:
            if tp == 'tst': return None if self.x_tst is None else step.transform(self.x_tst)
            else: return (step.transform(self.x_trn), self.y_trn) if tp == 'trn' else (step.transform(self.x_val), self.y_val)    
    
    @property
    def cons(self): return self.x_tfms.cons
    
    @property
    def cats(self): return self.x_tfms.cats
    
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


def stratify_split(df, y, cats, ratio):
    keys = df[cats]
    if y.dtype.name[:5] != 'float': keys = pd.concat([keys, y], axis=1)
    keys = keys.apply(lambda x: '~'.join([str(j) for j in x.values]), axis=1)

    sss = split_by_cats(train_size =1-ratio, test_size=ratio)
    train, val = next(sss.split(df, keys))                
    x_trn, x_val = safe_indexing(df, train), safe_indexing(df, val)            
    y_trn, y_val = safe_indexing(y, train), safe_indexing(y, val)
    return x_trn, y_trn, x_val, y_val


def split_time_series(df, y, time_feature, ratio):
    df = df.copy()
    df = df.sort_values(by=time_feature, ascending=True)
    split_id = int(df.shape[0]*(1-ratio))
    df.drop(time_feature, axis=1, inplace = True)
    x_trn, y_trn = df[:split_id], y[:split_id]
    x_val, y_val = df[split_id:], y[split_id:]
    return x_trn, y_trn, x_val, y_val


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
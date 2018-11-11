import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def na_rm(x): return x[~np.isnan(x)]


def to_iter(col): 
    if type(col) == np.ndarray: 
        if len(col.shape) == 1: return np.array([col])
        else: return col
    elif type(col) == str or type(col) == int: return [col]
    else: return col


def sort_asc(df): return df.sort_values(df.columns[1], ascending = True)


def sort_desc(df): return df.sort_values(df.columns[1], ascending = False)


def flat_list(l): return [item for sublist in l for item in sublist]


def df_append(df, *args):
    df_dict = {}
    for col, val in zip(df.columns, args): df_dict[col] = val
    return df.append(pd.DataFrame.from_dict(df_dict), ignore_index = True)


def numpy_sample(arr, n_sample, axis=0):
    if n_sample > arr.shape[axis]: n_sample = arr.shape[axis]
    mask = np.random.permutation(np.array(list(range(n_sample))))
    
    if axis == 0: 
        if len(arr.shape) == 1: return arr[mask]
        elif len(arr.shape) == 2: return arr[mask,:]
        elif len(arr.shape) == 3: return arr[mask,:,:]
    elif axis == 1: 
        if len(arr.shape) == 2: return arr[:,mask,:]
        elif len(arr.shape) == 3: return arr[:,mask,:]
    elif axis == 2: return arr[:,:,mask]                              


class ResultDF:
    def __init__(self, df, cons):
        self.df = df
        self.cons = to_iter(cons)
        self.len = df.shape[0]
    
    def __call__(self): return self.df
    
    def top(self, n=None, features=None): 
        return self.df.sort_values(by=features or self.cons, ascending=False)[:(n or self.len)]

    def larger_than(self, value, features=None):
        features = features or self.cons
        return self.df[self.df[features] >= value].sort_values(by=features, ascending=False)

    def smaller_than(self, value, features=None): 
        features = features or self.cons
        return self.df[self.df[features] <= value].sort_values(by=features, ascending=False)
    
    def pos(self, n=None, features=None): 
        features = features or self.cons[0]
        return self.df[self.df[features]>=0].sort_values(by=features, ascending=False)[:(n or self.len)]
    
    def neg(self, n=None, features=None):
        features = features or self.cons[0]
        return self.df[self.df[features]<=0].sort_values(by=features, ascending=True)[:(n or self.len)]


def save_pickle(fn):
    with open('feature_dict.pkl', 'wb') as f: pickle.dump(feature_dict, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(fn):
    with open('feature_dict.pkl', 'rb') as f: return pickle.load(f)
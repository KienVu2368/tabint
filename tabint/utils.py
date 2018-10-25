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
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def na_rm(x): return x[~np.isnan(x)]


def to_list(col): 
    if type(col) == np.ndarray: 
        return list(col)
    else:  
        if type(col) == str or type(col) == DataFrameMapper:
            return [col]
        else:
            return col


def sort_asc(df): return df.sort_values(df.columns[1], ascending = True)

def sort_desc(df): return df.sort_values(df.columns[1], ascending = False)

def flat_list(l): return [item for sublist in l for item in sublist]


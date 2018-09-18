import pandas as pd
import numpy as np

def na_rm(x): return x[~np.isnan(x)]


def sort_desc(df): return df.sort_values(df.columns[1], ascending = False)


def to_list(col): 
    if type(col) == np.ndarray: 
        return list(col)
    else:  
        if type(col) == str:
            return [col]
        else:
            return col


def sort_asc(df): return df.sort_values(df.columns[1], ascending = True)


def flat_list(l): return [item for sublist in l for item in sublist]


def get_cons_cats(df, max_n_cat = 30):
    cons, cats= [], []
    for name, value in df.items():
        if value.dtypes.kind == "O": cats.append(name)
        else:
            if value.nunique()<=max_n_cat: cats.append(name)
            else: cons.append(name)
    return cons, cats


def parallel(df): return None
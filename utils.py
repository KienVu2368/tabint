import pandas as pd
import numpy as np

def na_rm(x): return x[~np.isnan(x)]

def sort_desc(df): return df.sort_values(df.columns[1], ascending = False)

def to_list(col): return [col] if (type(col) != list) else col

def sort_asc(df): return df.sort_values(df.columns[1], ascending = True)

def flat_list(l): return [item for sublist in l for item in sublist]

def cat_cols(df): return list(set(df.columns) - set(df._get_numeric_data().columns))
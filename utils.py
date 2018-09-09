import pandas as pd
import numpy as np

def na_rm(x): return x[~np.isnan(x)]

def sort_desc_inplace(df, col): df.sort_values(col, ascending = False, inplace=True)

def to_list(col): return [col] if (type(col) != list) else col
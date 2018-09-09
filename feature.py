import fastai
from fastai.imports import *
from fastai.structured import *
from .utils import *
import pandas as pd
import numpy as np

def aggreate(df, params, by_col):
    '''mean, median, prod, sum, std, var, max, min, count'''
    df_agg = df[list(params.keys())+to_list(by_col)].groupby(by_col).agg(params).reset_index()
    agg_cols = ['_'.join([i for i in c if i != '']) for c in df_agg.columns.tolist()]
    df_agg.columns = agg_cols
    return df_agg

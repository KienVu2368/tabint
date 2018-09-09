from .utils import *
from .eda import *
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy as hc

def aggreate(df, params, by_col, prefix = 'AGG'):
    '''mean, median, prod, sum, std, var, max, min, count, nunique'''
    df_agg = df[list(params.keys())+to_list(by_col)].groupby(by_col).agg(params)
    df_agg.columns = ['_'.join([prefix.upper(), c[1], c[2].upper()]) for c in df_agg.columns.tolist()]
    return df_agg.reset_index()

class importance(BaseEDA):
    @classmethod
    def from_df(cls, df): return cls(sort_desc(df))
    
    def plot(self): return plot_barh(self.df)


class dendrogram:
    def __init__(self, df):
        self.df = df
        self.columns = self.df.columns
        self.z = self.cal_z(df)
        self.result = self.cal_result()
    
    def chk_ft(self, n):
        fts = self.result[:n][['col1', 'col2']].values.tolist()
        return flat_list(fts)
    
    @staticmethod
    def cal_z(df):
        corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
        corr_condensed = hc.distance.squareform(1-corr)
        return hc.linkage(corr_condensed, method='average')
        
    def cal_result(self):
        z2 = [[self.columns[int(i[0])], self.columns[int(i[1])], i[2]] for i in self.z if i[3] == 2]
        result = pd.DataFrame.from_dict({'col1': [i[0] for i in z2],
                                        'col2': [i[1] for i in z2],
                                        'dist': [i[2] for i in z2]})
        return result
        
    def plot(self):
        fig = plt.figure(figsize=(10,self.df.shape[1]//2.5))
        dendrogram = hc.dendrogram(self.z, 
                                   labels=self.columns, 
                                   orientation='left', 
                                   leaf_font_size=16)
        plt.show()
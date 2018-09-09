from .utils import *
from .plot import *
import numpy as np
import pandas as pd
import pdb

class BaseEDA:
    def __init__(self, df): self.df = df
        
    def __getitem__(self, key): return self.df.__getitem__(key)
    
    def top(self, n): return list(self.df.iloc[:,0][:n])
    
    def plot(self, df): return None


class missing(BaseEDA):    
    @classmethod
    def from_df(cls, df):
        df_miss = df.isnull().sum()/len(df)*100
        df_miss = df_miss[df_miss > 0]
        df_miss = pd.DataFrame({'column':df_miss.index, 'missing_percent':df_miss.values})
        return cls(sort_desc(df_miss))
        
    def plot(self): 
        return plot_barh(self.df)


class correlation(BaseEDA):    
    @classmethod
    def from_df(cls, df, taget):
        correlations = df.corr()[taget]
        corr_df = pd.DataFrame({'column':correlations.index, 'corr':correlations.values})
        corr_df['neg'] = corr_df['corr'] < 0
        corr_df['corr'] = abs(corr_df['corr'])
        corr_df = corr_df[corr_df['column'] != taget]
        return cls(sort_desc(corr_df))
        
    def plot(self): return plot_barh(self.df)


class histogram(BaseEDA):
    def __init__(self, df, bins): 
        self.df = df
        self.data = self.cal_hist(df, bins)
        self.bins = bins
    
    @classmethod
    def from_df(cls, df, cols, bins=20): return cls(df[to_list(cols)], bins)
    
    @staticmethod
    def cal_hist(df, bins):
        result = pd.DataFrame(columns=['col', 'division', 'count'])
        for col, value in df.items():
            count, division = np.histogram(value, bins=bins)
            division = [str(division[i]) + '~' + str(division[i+1]) for i in range(len(division)-1)]
            result = result.append(pd.DataFrame.from_dict({'col': [col]*bins, 
                                                           'division': division, 
                                                           'count':count}), 
                                   ignore_index=True)
        return result
    
    def __getitem__(self, key): return self.data.__getitem__(key)
        
    def plot(self, bins = None): 
        bins = self.bins if bins is None else bins
        return plot_hist(self.df, bins = bins)


class KernelDensityEstimation(BaseEDA):
    def __init__(self, df, tg, cols, bins): 
        self.df = df
        self.tg, self.cols = tg, cols
        self.bins = bins
        self.data = self.cal_hist()

    @classmethod
    def from_df(cls, df, tg, cols, bins = 50):
        cols = to_list(cols)
        return cls(df[cols + [tg]], tg, cols, bins)
    
    def cal_hist(self):
        result = pd.DataFrame(columns=['col', 'target', 'division', 'count'])
        self.tgt_values = self.df[self.tg].unique()
        tgts_pos = [self.df[self.tg] == tgt for tgt in self.tgt_values]
        for col, value in self.df[self.cols].items():
            for i, tgt in enumerate(self.tgt_values):
                count, division = np.histogram(na_rm(value[tgts_pos[i]]), bins=self.bins) #
                division = [str(division[i]) + '~' + str(division[i+1]) for i in range(len(division)-1)]
                result = result.append(pd.DataFrame.from_dict({'col': [col]*self.bins,
                                                               'target': [tgt]*self.bins,
                                                               'division': division, 
                                                               'count':count}), 
                                                               ignore_index=True)
        return result
    
    def __getitem__(self, key): return self.data.__getitem__(key)
        
    def plot(self, bins = None): 
        bins = self.bins if bins is None else bins
        return plot_kde(self.df, self.tg, self.tgt_values, self.cols, gridsize = bins)
from .utils import *
import numpy as np
import pandas as pd
import pdb
from plotnine import *
from.dataset import *
import matplotlib.pyplot as plt
import seaborn as sns


class BaseViz:
    def __init__(self, data): self.data = data
        
    @classmethod
    def from_df(cls, df): None

    def plot(self, df): return None


class Missing(BaseViz):
    @classmethod
    def from_df(cls, df):
        df_miss = df.isnull().sum()/len(df)*100
        df_miss = df_miss[df_miss > 0]
        df_miss = pd.DataFrame({'column':df_miss.index, 'missing percent':df_miss.values})
        return cls(ResultDF(df_miss, 'missing percent'))
    
    def plot(self): return plot_barh(self.data())


class Correlation(BaseViz):    
    @classmethod
    def from_df(cls, df, taget):
        correlations = df.corr()[taget]
        corr_df = pd.DataFrame({'column':correlations.index, 'corr':correlations.values})
        corr_df['neg'] = corr_df['corr'] < 0
        corr_df['corr'] = abs(corr_df['corr'])
        corr_df = corr_df[corr_df['column'] != taget]
        return cls(ResultDF(corr_df, 'corr'))
    
    def plot(self): return plot_barh(self.data())


class Histogram(BaseViz):
    def __init__(self, plot_df, data, bins): 
        self.plot_df, self.data, self.bins = plot_df, data, bins
    
    @classmethod
    def from_df(cls, df, cols = None, bins=20):
        plot_df = plot_df = df.copy() if cols is None else df[to_iter(cols)]
        data = cls.calculate(plot_df, bins)
        return cls(plot_df, data, bins)
    
    @staticmethod
    def calculate(df, bins):
        result = pd.DataFrame(columns=['col', 'division', 'count'])
        for col, value in df.items():
            count, division = cal_histogram(value, bins)
            data = pd_append(result, [col]*bins, division, count)
        return ResultDF(data, 'count')
    
    def __getitem__(self, key): return self.data.__getitem__(key)
    
    def plot(self, bins = None):
        return plot_hist(self.plot_df, bins = bins or self.bins)


class KernelDensityEstimation(BaseViz):
    def __init__(self, data, label_uniques, label_values, col_names, col_values, bins):
        self.data = data
        self.label_uniques, self.label_values = label_uniques, label_values
        self.col_names, self.col_values = col_names, col_values
        self.bins = bins

    @classmethod
    def from_df(cls, df, label_name, col_names, bins = 50):
        label_values = df[label_name].values
        col_values, col_names = to_iter(df[col_names].T.values), to_iter(col_names)
        label_uniques = np.unique(label_values)
        data = cls.calculate(label_uniques, label_values, col_names, col_values, bins)
        return cls(data, label_uniques, label_values, col_names, col_values, bins)

    @classmethod
    def from_series(cls, label_values, col_names, col_values, bins = 50):
        label_uniques = np.unique(label_values)
        col_names, col_values = to_iter(col_names), to_iter(col_values)
        data = cls.calculate(label_uniques, label_values, col_names, col_values, bins)
        return cls(data, label_uniques, label_values, col_names, col_values, bins)
    
    @staticmethod
    def calculate(label_uniques, label_values, col_names, col_values, bins):
        data = pd.DataFrame(columns=['col name', 'division', 'label value', 'count'])
        for col_name, col_value in zip(col_names, col_values):
            for label in label_uniques:
                count, division = cal_histogram(na_rm(col_value[label_values == label]), bins)
                data = pd_append(data, [col_name]*bins, division, [label]*bins, count)
        return data
        
    def plot(self, bins = None, **kargs):
        for col_name, col_value in zip(self.col_names, self.col_values):
            plot_kde(self.label_uniques, self.label_values, col_name, col_value, gridsize = bins or self.bins, **kargs)


def cal_histogram(value, bins):
    count, division = np.histogram(value, bins=bins)
    division = [str(division[i]) + '~' + str(division[i+1]) for i in range(len(division)-1)]
    return count, division


def change_xaxis_pos(top):
    if top:
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    else:
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False


def plot_barh(df, width = 20, height_ratio = 4):
    change_xaxis_pos(True)
    sort_asc(df).plot(x = df.columns[0],
                      kind='barh',
                      figsize=(width, df.shape[0]//height_ratio), 
                      legend=False)
    change_xaxis_pos(False)


def plot_hist(df,  bins = 20):
    plt.figure(figsize = (5, df.shape[1]*4))
    for i, col in enumerate(df.columns):
        plt.subplot(df.shape[1], 1, i+1)
        df[col].plot(kind = 'hist', edgecolor = 'k', bins = bins)
        plt.title(col)
    plt.tight_layout(h_pad = 2.5)


def plot_kde(label_uniques, label_values, col_name, col_value, figsize = None, shade = True, gridsize=100):
    if figsize is None: figsize = (5, 5)
    plt.figure(figsize = figsize)
    for label in label_uniques:
        sns.kdeplot(col_value[label_values == label], shade=shade, 
                    label = 'label: ' + str(label), gridsize = gridsize)
    plt.title('Distribution of %s by Label Value' % col_name)
    plt.xlabel('%s' % col_name); plt.ylabel('Density')
    plt.tight_layout(h_pad = 2.5)


def roc_curve_plot(fpr, tpr, roc_auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
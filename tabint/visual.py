from .utils import *
import numpy as np
import pandas as pd
import pdb
from .pre_processing import *
from .dataset import *
from .learner import *
import matplotlib.pyplot as plt
import seaborn as sns
import waterfall_chart
import graphviz
from sklearn.tree import export_graphviz
import IPython
import re
from sklearn.metrics import precision_recall_curve


def change_xaxis_pos(top):
    if top:
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    else:
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False


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
        if df_miss.shape[0] == 0: print('no missing data'); return None
        else:
            data = pd.DataFrame({'feature':df_miss.index, 'missing percent':df_miss.values})
            return cls(ResultDF(data, 'missing percent'))
    
    def plot(self): return plot_barh_from_df(self.data())


class Correlation(BaseViz):    
    @classmethod
    def from_df(cls, df, taget):
        correlations = df.corr()[taget]
        corr_df = pd.DataFrame({'feature': correlations.index, 'corr':correlations.values})
        corr_df['neg'] = corr_df['corr'] < 0
        corr_df['corr'] = abs(corr_df['corr'])
        corr_df = corr_df[corr_df['column'] != taget]
        return cls(ResultDF(corr_df, 'corr'))
    
    def plot(self): return plot_barh_from_df(self.data())


def plot_barh_from_df(df, width = 20, height_ratio = 4):
    change_xaxis_pos(True)
    sort_asc(df).plot(x = df.columns[0],
                      kind='barh',
                      figsize=(width, df.shape[0]//height_ratio), 
                      legend=False)
    change_xaxis_pos(False)


def plot_barh_from_series(features, series, figsize = None, absolute = False, pos_color = 'g', neg_color = 'r'):
    if figsize is not None: plt.figure(figsize=figsize)
    if type(series) == list: series = np.array(series)
    change_xaxis_pos(True)
    
    if not absolute: 
        argsort = np.argsort(series)
        barh = plt.barh([features[s] for s in argsort], series[argsort],color=pos_color)
        mask = series[argsort]<0
    else:
        series_absolute = np.abs(series)
        argsort = np.argsort(series_absolute)
        mask = series[argsort]<0
        barh = plt.barh([features[s] for s in argsort], series_absolute[argsort], color=pos_color)
    
    for i,m in enumerate(mask): 
        if m: barh[i].set_color(neg_color)
    
    change_xaxis_pos(False)
    plt.show()


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
        result = pd.DataFrame(columns=['feature', 'division', 'count'])
        for col, value in df.items():
            count, division = cal_histogram(value, bins)
            data = df_append(result, [col]*bins, division, count)
        return ResultDF(data, 'count')
    
    def __getitem__(self, key): return self.data.__getitem__(key)
    
    def plot(self, bins = None):
        return plot_hist(self.plot_df, bins = bins or self.bins)


def cal_histogram(value, bins):
    count, division = np.histogram(value, bins=bins)
    division = [str(division[i]) + '~' + str(division[i+1]) for i in range(len(division)-1)]
    return count, division


def plot_hist(df,  bins = 20):
    plt.figure(figsize = (5, df.shape[1]*4))
    for i, col in enumerate(df.columns):
        plt.subplot(df.shape[1], 1, i+1)
        df[col].plot(kind = 'hist', edgecolor = 'k', bins = bins)
        plt.title(col)
    plt.tight_layout(h_pad = 2.5)


class BoxnWhisker(BaseViz):
    def __init__(self, features, values, data):
        self.features, self.values = features, values
        self.data = data
        
    @classmethod
    def from_df(cls, df, features = None):
        values = df.values.T if features is None else df[to_iter(features)].values.T
        features = features or df.columns        
        return cls.from_series(features, values)
    
    @classmethod
    def from_series(cls, features, values):
        data = pd.DataFrame(columns=['feature', 'min', 'q1', 'median', 'q3', 'max'])
        for f, v in zip(features, values): 
            Min, Q1, Median, Q3, Max, _ = boxnwhisker_value(v)
            data = df_append(data, [f], [Min], [Q1], [Median], [Q3], [Max])
        return cls(to_iter(features), to_iter(values), data)

    def plot(self, orient = 'h', **kwarg): 
        for f, v in zip(self.features, self.values): plot_boxnwhisker(f, v, orient = orient, **kwarg)


def plot_boxnwhisker(feature_name, value, orient = 'h', **kwarg):
    plt.figure()
    sns.boxplot(data = value, orient = orient, **kwarg).set(ylabel=feature_name)


class KernelDensityEstimation(BaseViz):
    def __init__(self, data, label_uniques, label_values, features_name, features_value, bins):
        self.data = data
        self.label_uniques, self.label_values = label_uniques, label_values
        self.features_name, self.features_value = features_name, features_value
        self.bins = bins

    @classmethod
    def from_df(cls, df, label_name, features_name, bins = 50):
        label_values = df[label_name].values
        features_value = df[features_name].values
        return cls.from_series(features_name, features_value, label_values, bins)

    @classmethod
    def from_learner(cls, learner, x, y, bins = 50): return cls.from_series('prob', learner.predict(x), y, bins)

    @classmethod
    def from_series(cls, features_name, features_value, label_values, bins = 50):
        label_uniques = np.unique(label_values)
        features_name, features_value = to_iter(features_name), to_iter(features_value)
        data = cls.calculate(label_uniques, label_values, features_name, features_value, bins)
        return cls(ResultDF(data,'count'), label_uniques, label_values, features_name, features_value, bins)
    
    @staticmethod
    def calculate(label_uniques, label_values, features_name, features_value, bins):
        data = pd.DataFrame(columns=['feature', 'division', 'label', 'count'])
        for col_name, col_value in zip(features_name, features_value):
            for label in label_uniques:
                count, division = cal_histogram(na_rm(col_value[label_values == label]), bins)
                data = df_append(data, [col_name]*bins, division, [label]*bins, count)
        return data
        
    def plot(self, bins = None, vline = None, **kargs):
        for col_name, col_value in zip(self.features_name, self.features_value):
            plot_kde(self.label_uniques, self.label_values, col_name, col_value, gridsize = bins or self.bins, vline = vline, **kargs)


def plot_kde(label_uniques, label_values, col_name, col_value, vline = None, figsize = None, shade = True, gridsize=100):
    if figsize is None: plt.figure(figsize = figsize)
    else: plt.figure()

    for label in label_uniques: 
        sns.kdeplot(col_value[label_values == label], shade=shade, label = 'label: ' + str(label), gridsize = gridsize)
    if vline is not None: plt.axvline(vline)
    plt.title('Distribution of %s by Label Value' % col_name)
    plt.xlabel('%s' % col_name)
    plt.ylabel('Density')


class ReceiverOperatingCharacteristic:
    def __init__(self, fpr, tpr, data, roc_auc):
        self.fpr, self.tpr, self.data, self.roc_auc = fpr, tpr, data, roc_auc
        
    @classmethod
    def from_learner(cls, learner, x, y_true):
        y_pred = learner.predict(x)
        return cls.from_series(y_true, y_pred)

    @classmethod
    def from_series(cls, y_true, y_pred): 
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
        data = pd.DataFrame.from_dict({'threshold': threshold, 'tpr':tpr, 'fpr':fpr})
        roc_auc = metrics.auc(fpr, tpr)
        return cls(fpr, tpr, data, roc_auc)
    
    def plot(self): plot_roc_curve(self.fpr, self.tpr, self.roc_auc)


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


class PrecisionRecall:
    def __init__(self, precision, recall, threshold)
        self.precision,self.recall,self.threshold = precision, recall, threshold

    @classmethod
    def from_learner(cls, learner, x, y_true):
        y_pred = learner.predict(x)
        return cls.from_series(y_true, y_pred)

    @classmethod
    def from_series(cls, y_true, y_pred):
        precision, recall, threshold = precision_recall_curve(y_true, y_pred)
        return cls(precision, recall, threshold)

    def plot(self, **kwargs):
        plot_line([self.threshold]*2, [self.precision[:-1], self.recall[:-1]], ['precision', 'recall'], xlabel = "threshold", **kwargs)


def plot_line(x_series, y_series, labels, xlabel = None, ylim = None, **kwargs):
    for x_serie, y_serie, labels in zip(to_iter(x_series),to_iter(y_series),to_iter(labels)):
        plt.plot(x_serie, y_serie, , label, **kwargs)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylim is not None: plt.ylim(ylim)
    if xlim is not None: plt.xlim(xlim)
    plt.show()


def plot_waterfall(Column, contributions, rotation_value=90, threshold=0.2, sorted_value=True, **kargs):
    return waterfall_chart.plot(Column, contributions, rotation_value=rotation_value, threshold=threshold, sorted_value=sorted_value,**kargs)


def plot_SKTree(es, features, precision=0, size=10, ratio=0.6, **kargs):
    p=export_graphviz(es, out_file=None, feature_names=features, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', p)))


def plot_LGBTree(md, tree_index, figsize = (20, 8), show_info = ['split_gain'], **kargs):
    # still error
    ax = lgb.plot_tree(md, tree_index=tree_index, figsize=figsize, show_info=show_info, **kargs)
    plt.show()
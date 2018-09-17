from .utils import *
from .eda import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.cluster import hierarchy as hc
from sklearn.metrics import roc_auc_score


def aggreate(df, params, by_col, prefix = 'AGG'):
    '''mean, median, prod, sum, std, var, max, min, count, nunique, size, nanmedian
    skew, kurt, iqr'''
    df_agg = df.groupby(by_col).agg(params) #[list(params.keys())+to_list(by_col)]
    df_agg.columns = ['_'.join([prefix.upper(), c[0], c[1].upper()]) for c in df_agg.columns.tolist()]
    return df_agg.reset_index()


class dendogram:
    def __init__(self, z, result, cols):
        self.z = z
        self.result = result
        self.cols = cols
    
    def chk_ft(self, n):
        fts = self.ddg_df[:n][['col1', 'col2']].values.tolist()
        return flat_list(fts)
    
    @classmethod
    def from_df(cls, df):
        cols = df.columns
        z = cls.cal_z(df)
        result = cls.cal_result(z = z , cols = cols)
        return cls(z, result, cols)
    
    @staticmethod
    def cal_z(df):
        corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
        corr = np.where(np.isnan(corr), np.random.rand()*1e-3, corr)
        corr_condensed = hc.distance.squareform(1-corr, checks=False)
        return hc.linkage(corr_condensed, method='average')
    
    @staticmethod
    def get_name(col, i): return '---' if int(i) >= len(col) else col[int(i)]
    
    @classmethod
    def cal_result(cls, z, cols):
        result = pd.DataFrame.from_dict({'col1': [cls.get_name(cols, i[0]) for i in z],
                                         'col2': [cls.get_name(cols, i[1]) for i in z],
                                         'dist': [i[2] for i in z]})
        return result
    
    def plot(self):
        plt.figure(figsize=(10,max(self.z.shape[0]//2.6, 5)))
        hc.dendrogram(self.z, 
                      labels=self.cols, 
                      orientation='left', 
                      leaf_font_size=16)
        plt.show()


class importance():
    def __init__(self, impt_df):
        self.I = sort_desc(impt_df)
    
    @classmethod
    def from_LGBLearner(cls, learner, col_group):
        '''
        http://explained.ai/rf-importance/index.html
        '''
        y_pred = learner.predict(data = learner.ds.lgb_val.data)
        baseline = roc_auc_score(learner.ds.lgb_val.label, y_pred)
        imp, ft = [], []
        for cols in col_group:
            y_pred_permut = learner.predict(data = learner.ds.val_permutation(cols))
            permut_score = roc_auc_score(learner.ds.lgb_val.label, y_pred_permut)
            imp.append(baseline - permut_score)
            ft.append(' & '.join(to_list(cols)))
        return cls(pd.DataFrame.from_dict({'Feature': ft, 'Importance': imp}))

    def top(self, n): return [col.split(' & ') for col in self.I.Feature[:n]]

    def plot(self, **kagrs): plot_barh(self.I, **kagrs)
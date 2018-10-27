from .utils import *
from .visual import *
from .learner import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.cluster import hierarchy as hc
from sklearn.metrics import roc_auc_score

#DAE???

def aggreate(df, params, by_col, prefix = 'AGG'):
    '''
    mean, median, prod, sum, std, var, max, min, count, nunique, size, nanmedian
    skew, kurt, iqr
    '''
    df_agg = df.groupby(by_col).agg(params)
    df_agg.columns = ['_'.join([prefix.upper(), c[0], c[1].upper()]) for c in df_agg.columns.tolist()]
    return df_agg.reset_index()


class Dendogram:
    """
    plot cluster of feature, to see high correlate feature
    """
    def __init__(self, z, data, fts):
        self.z = z
        self.data = ResultDF(data, 'distance')
        self.fts = fts
    
    def chk_ft(self, n):
        fts = self.ddg_df[:n][['feature 1', 'feature 2']].values.tolist()
        return flat_list(fts)
    
    @classmethod
    def from_df(cls, df):
        fts = df.columns
        z = cls.cal_z(df)
        data = cls.cal_result(z = z , fts = fts)
        return cls(z, data, fts)
    
    @staticmethod
    def cal_z(df):
        corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
        corr = np.where(np.isnan(corr), np.random.rand()*1e-3, corr)
        corr_condensed = hc.distance.squareform(1-corr, checks=False)
        return hc.linkage(corr_condensed, method='average')
    
    @staticmethod
    def get_name(col, i): return '---' if int(i) >= len(col) else col[int(i)]
    
    @classmethod
    def cal_result(cls, z, fts):
        data = pd.DataFrame.from_dict({'feature 1': [cls.get_name(fts, i[0]) for i in z],
                                       'feature 2': [cls.get_name(fts, i[1]) for i in z],
                                       'distance': [i[2] for i in z]})
        return data
    
    def plot(self):
        plt.figure(figsize=(10,max(self.z.shape[0]//2.6, 5)))
        hc.dendrogram(self.z, 
                      labels=self.fts, 
                      orientation='left', 
                      leaf_font_size=16)
        plt.show()


class Importance:
    """
    permutation importance. See more at http://explained.ai/rf-importance/index.html
    """
    def __init__(self, data):
        self.data = data
    
    @classmethod
    def from_Learner(cls, learner, ds,  group_fts = None, score = roc_auc_score):
        #to do in parrallel??
        group_fts = group_fts + [i for i in ds.features if i not in flat_list(group_fts)] if group_fts is not None else ds.features
        y_pred = learner.predict(ds.x_val)
        baseline = score(ds.y_val, y_pred)        
        data = pd.DataFrame.from_dict({'feature': [' & '.join(to_iter(fts)) for fts in group_fts]})
        data['importance'] = data.apply(cls.cal_impt, axis = 1, learner = learner, ds = ds, baseline = baseline, score = score)
        return cls(ResultDF(data, 'importance'))

    @staticmethod
    def cal_impt(x, learner, ds, baseline, score):
        fts = x[0].split(' & ')
        y_pred_permut = learner.predict(ds.val_permutation(fts))
        permut_score = score(ds.y_val, y_pred_permut)
        return baseline - permut_score

    def top_features(self, n): return flat_list([col.split(' & ') for col in self.data.top().feature[:n]])

    def plot(self, **kagrs): plot_barh(self.data(), **kagrs)
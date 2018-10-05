from .utils import *
from .eda import *
from .learner import *
from.dataset import *
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
    def __init__(self, z, result, cols):
        self.z = z
        self.result = ResultDF(result)
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

    def group_cols(self, grp):
        return grp + [i for i in self.cols if i not in flat_list(grp)]


class Importance:
    """
    permutation importance. See more at http://explained.ai/rf-importance/index.html
    """
    def __init__(self, impt_df):
        self.I = ResultDF(impt_df)
    
    @classmethod
    def from_Learner(cls, learner, ds,  group_cols, score = roc_auc_score):
        #to do in parrallel??
        y_pred = learner.predict(ds.x_val)
        baseline = score(ds.y_val, y_pred)        
        I = pd.DataFrame.from_dict({'Feature': [' & '.join(to_list(cols)) for cols in group_cols]})
        I['Importance'] = I.apply(cls.cal_impt, axis = 1, learner = learner, ds = ds, baseline = baseline, score = score)
        return cls(I)
            
    @staticmethod
    def cal_impt(x, learner, ds, baseline, score):
        cols = x[0].split(' & ')
        y_pred_permut = learner.predict(ds.val_permutation(cols))
        permut_score = score(ds.y_val, y_pred_permut)
        return baseline - permut_score

    def top(self, n): return [col.split(' & ') for col in self.I.Feature[:n]]

    def plot(self, **kagrs): plot_barh(self.I, **kagrs)
import pandas as pd
import numpy as np
from .eda import *
from .plot import *
from .utils import *
import lightgbm as lgb
import pickle


##psedou labeling??
#denoising autoencoder?? https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
##function do drop and check feature
##time dependency check https://youtu.be/3jl2h9hSRvc?t=48m50s
##hyper parameter tuning??
##essemble method - low priority
#todo learning rate finder for learner?
#optimizer


class TBLearner:
    def __init__(self, md):
        self.md = md

    def fit(self, **kargs): self.md.fit( **kargs)

    def predict(self, df, **kargs): return self.md.predict(df, **kargs)

    def load(self, fn): self.md.load(fn)
        
    def save(self): self.md.save(fn)


class LGBLearner(TBLearner):
    def __init__(self):
        self.score = []
        
    def fit(self, params, x_trn, y_trn, x_val, y_val, 
            ctn = False, save = True, fn = 'LGB_Model.pkl', early_stopping_rounds=100, verbose_eval = 100, **kargs):
        if ctn: 
            self.load(fn)
        else:
            self.md = None
        lgb_trn, lgb_val = self.LGBDataset(x_trn, y_trn, x_val, y_val)
        self.md = lgb.train(params = params,
                            train_set = lgb_trn,
                            valid_sets = [lgb_trn, lgb_val],
                            init_model = self.md, 
                            early_stopping_rounds = early_stopping_rounds, 
                            verbose_eval = verbose_eval, **kargs)

        self.score.append(self.md.best_score)
        if save: self.save(fn)

    @staticmethod
    def LGBDataset(x_trn, y_trn, x_val, y_val):
        lgb_trn = lgb.Dataset(x_trn, y_trn)
        lgb_val = lgb.Dataset(x_val, y_val, free_raw_data=False, reference=lgb_trn)
        return lgb_trn, lgb_val
    
    def load(self, fn): 
        with open(fn, 'rb') as fin: self.md = pickle.load(fin)
        
    def save(self, fn):
        with open(fn, 'wb') as fout: pickle.dump(self.md, fout)


class XGBLearner(TBLearner):
    def __init__(self):
        None


class DTLearner(TBLearner):
    def __init__(self):
        None

class RFLearner(TBLearner):
    def __init__(self):
        None
import pandas as pd
import numpy as np
from .eda import *
from .plot import *
import lightgbm as lgb
import pickle

class BaseLearner():
    def __init__(self): None
        #model
        #data
        #crit
        #lrs learning rate
        #callback?

    def fit(self): None

    def predict(self): None
    
    def param_find(self): None
    
    @property
    def summary(self): None

    def load(self): None
        
    def save(self): None

    def interpretation(self): None


class LGBLearner():
    def __init__(self, dataset, fn = 'model.pkl', callbacks = None):
        self.ds = dataset
        self.fn = fn
        self.md = None
        self.callbacks = callbacks

    def fit(self, params, ctn = False, save = True, **kargs):
        if ctn: self.load()
        self.params = params
        self.md = lgb.train(params = self.params,
                            train_set = self.ds.lgb_trn,
                            valid_sets = [self.ds.lgb_trn, self.ds.lgb_val], #to appear both train and valid metric
                            init_model = self.md,
                            callbacks = self.callbacks,
                            **kargs)
        
        if save: self.save()
    
    def predict_test_set(self, **kargs): 
        return None if self.ds.x_tst is None else self.md.predict(self.ds.lgb_tst, **kargs)
    
    def predict(self, df, **kargs): return self.md.predict(df, **kargs)

    def load(self): 
        with open(self.fn, 'rb') as fin: self.md = pickle.load(fin)
        
    def save(self): 
        with open(self.fn, 'wb') as fout: pickle.dump(self.md, fout)


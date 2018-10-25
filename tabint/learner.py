import pandas as pd
import numpy as np
from .visual import *
from .utils import *
import lightgbm as lgb
import pickle
from sklearn import tree
from sklearn.externals import joblib
from sklearn.ensemble import *
from sklearn import metrics
from sklearn.metrics import *



##psedou labeling??
#denoising autoencoder?? https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
##time dependency check https://youtu.be/3jl2h9hSRvc?t=48m50s
##hyper parameter tuning??
##essemble method - low priority
#todo learning rate finder for learner?
#optimizer


class SKLearner:
    def __init__(self, md):
        self.md = md
        
    def fit(self, x_trn, y_trn, x_val, y_val, save = False, fn = 'SKTree', **kargs):
        self.md.fit(x_trn, y_trn, **kargs)        
        print('trn accuracy: ', self.md.score(x_trn, y_trn))
        print('val accuracy: ', self.md.score(x_val, y_val))
        if save: self.save(fn)

    def predict(self, df, **kargs): return self.md.predict(df, **kargs)

    def predict_proba(self, df, **kargs): return self.md.predict_proba(df, **kargs)

    def predict_log_proba(self, df, **kargs): return self.md.predict_log_proba(df, **kargs)

    def load(self, fn): self.md = joblib.load(fn + '.joblib')
        
    def save(self, fn): joblib.dump(self.md, fn + '.joblib')


class LGBLearner(SKLearner):
    """
    Contain model and its method: learning rate, callbacks, loss function...
    """
    def __init__(self):
        self.score = []
        
    def fit(self, params, x_trn, y_trn, x_val, y_val,
            lrts = None, callbacks = None, 
            fobj=None, feval=None,
            ctn=False, save=False, fn = 'LGB_Model', 
            early_stopping_rounds=100, verbose_eval = 100, **kargs):
        if ctn:
            self.load(fn)
        else:
            self.md = None
        lgb_trn, lgb_val = self.LGBDataset(x_trn, y_trn, x_val, y_val)
        self.md = lgb.train(params = params,
                            train_set = lgb_trn,
                            valid_sets = [lgb_trn, lgb_val],
                            init_model = self.md,
                            learning_rates = lrts,
                            callbacks = callbacks,
                            fobj = fobj,
                            feval = feval,
                            early_stopping_rounds = early_stopping_rounds, 
                            verbose_eval = verbose_eval, **kargs)

        self.score.append(self.md.best_score)
        if save: self.save(fn)

    @staticmethod
    def LGBDataset(x_trn, y_trn, x_val, y_val):
        lgb_trn = lgb.Dataset(x_trn, y_trn)
        lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_trn)
        return lgb_trn, lgb_val

    def predict(self, df, **kargs): return self.md.predict(df, num_iteration = self.md.best_iteration, **kargs)
    
    def load(self, fn):
        with open(fn + '.pkl', 'rb') as fin: self.md = pickle.load(fin)
        
    def save(self, fn):
        with open(fn + '.pkl', 'wb') as fout: pickle.dump(self.md, fout)


class XGBLearner(SKLearner):
    def __init__(self):
        None


class ReceiverOperatingCharacteristic:
    def __init__(self, fpr, tpr, result, roc_auc):
        self.fpr, self.tpr, self.result, self.roc_auc = fpr, tpr, result, roc_auc
        
    @classmethod
    def from_learner(cls, learner, x, y):
        fpr, tpr, threshold = metrics.roc_curve(y, learner.predict(x))
        result = pd.DataFrame.from_dict({'threshold': threshold, 'tpr':tpr, 'fpr':fpr})
        roc_auc = metrics.auc(fpr, tpr)
        return cls(fpr, tpr, result, roc_auc)
    
    def plot(self): plot_roc_curve(self.fpr, self.tpr, self.roc_auc)
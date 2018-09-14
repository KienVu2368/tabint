import pandas as pd
import numpy as np
from .eda import *
from .plot import *
import lightgbm
from lightgbm import LGBMClassifier
import cPickle as pickle

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


class importance(BaseEDA):
    @property
    def plot(self): return plot_barh(self.df)


class lgbm_learner(BaseLearner):
    def __init__(self): None


class interpretation():
    def __init__(self): None


class Callback:
    def on_train_begin(self): pass
    def on_batch_begin(self): pass
    def on_phase_begin(self): pass
    def on_epoch_end(self, metrics): pass
    def on_phase_end(self): pass
    def on_batch_end(self, metrics): pass
    def on_train_end(self): pass
import pandas as pd
import numpy as np
from .eda import *
from .plot import *

class BaseLearner():
    def __init__(self): None
        
    @property
    def model(self): None

    @property
    def data(self): None

    @classmethod
    def from_model_date(cls): return None

    def fit(self): None  

    def predict(self): None
    
    def param_find(self): None

    def load(self): None
        
    def save(self): None
    
    def importance(self): None

    def interpretation(self): None


class importance(BaseEDA):
    @classmethod
    def from_df(cls, df): return cls(sort_desc(df))
    
    def plot(self): return plot_barh(self.df)

class lgbm_learner(BaseLearner):
    def __init__(self): None

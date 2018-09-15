from .utils import *
from sklearn.model_selection import train_test_split
import lightgbm as lgb

class LGBDataset():
    def __init__(self, x_trn, y_trn, x_val, y_val, x_tst = None):
        '''
        https://lightgbm.readthedocs.io/en/latest/Python-API.html#data-structure-api
        https://lightgbm.readthedocs.io/en/latest/Python-Intro.html#data-interface
        '''
        self.lgb_trn = lgb.Dataset(x_trn, y_trn)
        self.lgb_val = lgb.Dataset(x_val, y_val, free_raw_data=False, reference=self.lgb_trn)
        self.lgb_tst = None if x_tst is None else lgb.Dataset(x_tst)
    
    def val_permutation(self, cols):
        cols = to_list(cols)
        df = self.lgb_val.data.copy()
        for col in cols: df[col] = np.random.permutation(df[col])
        return df
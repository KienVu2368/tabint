from .utils import *
from sklearn.model_selection import train_test_split
import lightgbm as lgb


#imbalance data???
#subsampling, over sampling??

class TBDataset:
    def __init__(self, x_trn, y_trn, x_val, y_val, x_tst = None):
        self.x_trn, self.y_trn = x_trn, y_trn
        self.x_val, self.y_val = x_val, y_val
        self.x_tst = x_tst

    @classmethod
    def from_SklearnSplit(cls, df, y_df, ratio = 0.2, x_tst = None, **kargs):
        x_trn, x_val, y_trn, y_val = train_test_split(df, y_df, test_size=ratio, stratify = y_df)
        return cls(x_trn, y_trn, x_val, y_val, x_tst)

    @classmethod
    def from_TBSplit(cls, df, y_df, x_tst, pct = 2, ratio = 0.2, **kargs):
        _, cats = get_cons_cats(df)
        
        tst_key = x_tst[cats].drop_duplicates().values
        tst_key = set('~'.join([str(j) for j in i]) for i in tst_key)

        df_key = df[cats].apply(lambda x: '~'.join([str(j) for j in x.values]), axis=1)
        mask = df_key.isin(tst_key)

        x_trn, y_trn = df[~mask], y_df[~mask]
        x_val_set, y_val_set = df[mask], y_df[mask]

        x_val = x_val_set.groupby(cats).apply(cls.random_choose, pct, ratio, **kargs)
        val_index = set([i[-1] for i in x_val.index.values])
        x_val.reset_index(drop=True, inplace=True)
        
        mask = x_val_set.index.isin(val_index)
        y_val = y_val_set[mask]
        x_trn, y_trn = pd.concat([x_trn, val_set[~mask]]), pd.concat([y_trn, y_val_set[~mask]])

        return cls(x_trn, y_trn, x_val, y_val, x_tst)

    @staticmethod
    def random_choose(x, pct = 2, ratio = 0.2, **kargs):
        n = x.shape[0] if random.randint(0,9) < pct else int(np.round(x.shape[0]*(ratio-0.06)))
        return x.sample(n=n, **kargs)

    def val_permutation(self, cols):
        cols = to_list(cols)
        df = self.x_val.copy()
        for col in cols: df[col] = np.random.permutation(df[col])
        return df

    def add(self, col, f, inplace = True, tp = 'trn'): 
        if inplace:
            for df in [self.x_trn, self.x_val, self.x_tst]: if df is not None: df[col] = f(df)
        else:
            return {'trn': self.x_trn[col], self.y_trn, 'val': self.x_val[col], self.y_val, 'tst': self.x_tst[col]}[tp]

    def sample(self, tp = 'trn', ratio = 0.3):
        if 'tst' == tp: 
            return None if self.x_tst is None else self.x_tst.sample(self.x_tst.shape[0]*ratio)
        else:
            df, y_df = {'trn': self.x_trn, self.y_trn, 'val': self.x_val, self.y_val, 'tst': self.x_tst}[tp]
            _, df, _, y_df = train_test_split(df, y_df, test_size = ratio, stratify = y_df)
            return df, y_df

    def keep(self, col, inplace = True, tp = 'trn'): 
        if inplace:
            for df in [self.x_trn, self.x_val, self.x_tst]: if df is not None: df = df[col]
        else:
            return {'trn': self.x_trn[col], self.y_trn, 'val': self.x_val[col], self.y_val, 'tst': self.x_tst[col]}[tp]

    def drop(self, col, inplace = True, tp = 'trn'): 
        if inplace:
            for df in [self.x_trn, self.x_val, self.x_tst]: if df is not None: df.drop(col, axis=1, inplace = True)
        else:
            return {'trn': self.x_trn.drop(col, axis = 1), self.y_trn, 
                    'val': self.x_val.drop(col, axis = 1), self.y_val, 
                    'tst': self.x_tst.drop(col, axis = 1)}[tp]

    def trn_n_val(self): return self.x_trn, self.y_trn, self.x_val, self.y_val

    @property
    def features(self): return self.x_trn.columns
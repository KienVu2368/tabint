from .utils import *
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

#todo use dask, numba and do things in parallel
#immutation https://www.kaggle.com/dansbecker/handling-missing-values
#use sklearn pipeline and transformner??



def tabular_proc(df, tfms, ignore_flds=None):
    res = []
    df = df.copy()
    
    if ignore_flds is not None:
        ignored_flds = df.loc[:, ignore_flds]
        df.drop(ignore_flds, axis=1, inplace=True)
    
    for f in tfms:
        out = f(df)
        if str(f)[10:].split(' ')[0] == 'dummies': 
            cats = [i for i in df.columns if i not in cons]
            res += [cats]
        if out is not None: 
            if str(f)[10:].split(' ')[0] == 'app_cat': cons = out
            res += [out]
    
    if ignore_flds is not None: df = pd.concat([ignored_flds, df], axis=1)
    
    res.insert(0,df)
    return res


class TBPreProc:
    def __init__(self, *args): self.args = args
        
    def __call__(self, df): return self.func(df, *self.args)
    
    @staticmethod
    def func(*args): None


class skip_flds(TBPreProc):
    @staticmethod
    def func(df, skip_flds):
        df = df.drop(skip_flds, axis=1, inplace=True)
        return None

class subset(TBPreProc):
    @staticmethod
    def func(df, subset):
        df = df.sample(subset).copy
        return None


class get_y(TBPreProc):
    @staticmethod
    def func(df, y_fld):
        if y_fld is None: y = None
        else:
            if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
            y = df[y_fld].values
        df.drop(y_fld, axis=1, inplace=True)
        return y


class fill_na(TBPreProc):
    @staticmethod
    def func(df, na_dict = None):
        na_dict = {} if na_dict is None else na_dict.copy()
        na_dict_initial = na_dict.copy()
        for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
        if len(na_dict_initial.keys()) > 0:
            df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
        return na_dict


def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict


class app_cat(TBPreProc):
    @staticmethod
    def func(df, max_n_cat=15):
        cons = []
        for name, value in df.items():
            if is_numeric_dtype(value) and value.dtypes != np.bool:
                if value.nunique()<=max_n_cat and not np.array_equal(value.unique(), np.array([0, 1])): 
                    df[name] = value.astype('category').cat.as_ordered()
                else: cons.append(name)
            else:
                if value.nunique()>max_n_cat: df[name] = value.astype('category').cat.codes+1; cons.append(name)
                elif value.dtypes.name == 'category': df[name] = value.cat.as_ordered()
        return cons


class dummies(TBPreProc):
    @staticmethod
    def func(df):
        df = pd.get_dummies(df, dummy_na=True)
        return None
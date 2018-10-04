import fastai
from fastai.imports import *
from fastai.structured import *
from .utils import *

#todo use dask, numba and do things in parallel
#immutation https://www.kaggle.com/dansbecker/handling-missing-values
#use sklearn pipeline and transformner??


def tabular_proc(df, y_fld=None, skip_flds=None, ignore_flds=None, 
                 do_scale=False, na_dict=None, preproc_fn=None, 
                 max_n_cat=15, subset=None, mapper=None):

    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars2(df, mapper)

    df, cons = get_cons_cats(df, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    cats = [i for i in df.columns if i not in cons]
    
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict, cons, cats]
    if do_scale: res = res + [mapper] 
    return res
    

def get_cons_cats(df, max_n_cat=15):
    cons = []
    for name, value in df.items():
        if is_numeric_dtype(value) and value.dtypes != np.bool:
            if value.nunique()<=max_n_cat and not np.array_equal(value.unique(), np.array([0, 1])): 
                df[name] = value.astype('category').cat.as_ordered()
            else: cons.append(name)
        else:
            if value.nunique()>max_n_cat: df[name] = value.astype('category').cat.codes+1; cons.append(name)
            elif value.dtypes.name == 'category': df[name] = value.cat.as_ordered()
    return df, cons


def scale_vars2(df, mapper = None, scale_fld_exc = None):
    scale_fld_exc = to_list(scale_fld_exc) if scale_fld_exc is not None else []
    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n]) and n not in scale_fld_exc]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper
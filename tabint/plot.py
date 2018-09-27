import matplotlib.pyplot as plt
import seaborn as sns
from .utils import *

def change_xaxis_pos(top):
    if top:
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    else:
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False

def plot_barh(df, width = 20, height_ratio = 4):
    change_xaxis_pos(True)
    sort_asc(df).plot(x = df.columns[0],
                      kind='barh',
                      figsize=(width, df.shape[0]//height_ratio), 
                      legend=False)
    change_xaxis_pos(False)

def plot_hist(df,  bins = 20):
    plt.figure(figsize = (5, df.shape[1]*4))
    for i, col in enumerate(df.columns):
        plt.subplot(df.shape[1], 1, i+1)
        df[col].plot(kind = 'hist', edgecolor = 'k', bins = bins)
        plt.title(col)
    plt.tight_layout(h_pad = 2.5)

def plot_kde(df, tg, tg_values, cols, gridsize=100):
    plt.figure(figsize = (5, len(cols)*4))
    for i, source in enumerate(cols):
        plt.subplot(len(cols), 1, i+1)
        for tgt in tg_values: 
            sns.kdeplot(df.loc[df[tg] == tgt, source], 
                        label = 'target = ' + str(tgt),
                        gridsize = gridsize)
        plt.title('Distribution of %s by Target Value' % source)
        plt.xlabel('%s' % source); plt.ylabel('Density')
    plt.tight_layout(h_pad = 2.5)


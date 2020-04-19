

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

cm_green = sns.light_palette("green", as_cmap=True)

def crosstab_background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
    rng = M - m
    norm = colors.Normalize(m - (rng * low),
                            M + (rng * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]


def crosstab(df, row, column, cmap=cm_green, **kagrs):
    df_crosstab = pd.crosstab(df[row], df[column], **kagrs)
    return df_crosstab.style.apply(crosstab_background_gradient, m=df_crosstab.min().min(), M=df_crosstab.max().max(), cmap=cm)
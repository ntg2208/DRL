# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from tqdm import tqdm
import yfinance as yf 
import statsmodels.tsa.stattools as ts 


#%%
with open('examinating.txt', 'r') as f:
    list_stocks = f.read().split(', ')[:-1]

# %%
def fix_len():
    # Fix to len 4000 -> 4000 - len(df): 
    os.makedirs('processing', exist_ok=True)
    for i in tqdm(list_stocks):
        df = pd.read_csv(f'csv_data/{i}.csv')
        df = df[len(df) - 4000:]
        df.to_csv(f'processing/{i}.csv')

#%%
list_stocks = sorted(list_stocks)
print(list_stocks, len(list_stocks))
#%%
maxlen = len(list_stocks)
# TODO: create 414x414 numpy array, each cell will be a 1 - cointegration value
# only upper diagonal will have value, lower diagonal will be 0
prefill = np.zeros([1, maxlen])
cointegration = np.zeros([maxlen, maxlen])

for idx, i in (enumerate(list_stocks)):
    df = pd.read_csv(f'processing/{i}.csv').fillna(0)['Close'].to_numpy()
    coints = list()
    for j in tqdm(list_stocks[idx+1:]):
        df2 = pd.read_csv(f'processing/{j}.csv').fillna(0)['Close'].to_numpy()
        coint = ts.coint(df, df2)[1]
        # print(coint)
        coints.append(1-coint)
        # break
    prefill[:, maxlen - len(coints):] = np.expand_dims(np.asarray(coints), axis=0)
    col = prefill
    prefill = np.zeros([1, maxlen])

    # print('\n')
    # print(col)
    cointegration[idx] = col

    # break



#%%
# import pickle

# np.save('cointegration.npy', cointegration)
#%%
cointegration = np.load('cointegration.npy')
#%%
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    plt.xticks(fontsize= 5)
    plt.yticks(fontsize= 5)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True, )
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

#%%
from matplotlib.pyplot import figure

figure(figsize=(120, 120), dpi=96)

num = 40

im, cbar = heatmap(cointegration[:num, :num], list_stocks[:num], list_stocks[:num], ax=ax,
                   cmap="BuGn")
plt.savefig('tmp.jpg', dpi=300)
# fig.tight_layout()
# plt.show()


#%%
# from statsmodels.tsa.vector_ar.vecm import coint_johansen
AAPL = pd.read_csv(f'csv_data/AAPL.csv')['Close'].to_numpy()
ABC = pd.read_csv(f'csv_data/ABC.csv')['Close'].to_numpy()
# print(len(AAP), len(AAPL))
result = ts.coint(AAPL, ABC)
# result=coint_johansen(pd.DataFrame([AAPL, ABC]), 0, -1)
print(result)
# print(AAPL[-1], ABC[-1])
from PIL import Image

plt.imshow(Image.open(f'plot_csv/AAPL.jpg'))
plt.show()
plt.imshow(Image.open(f'plot_csv/ABC.jpg'))
plt.show()

# %%
import numpy
import talib

close = numpy.random.random(100)
from talib import MA_Type

upper, middle, lower = talib.BBANDS(close, matype=MA_Type.T3)


# %%
print(middle, lower)

# %%




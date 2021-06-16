#%%
import os
import numpy as np 
import pandas as pd
from pandas.io.parquet import FastParquetImpl
from tqdm import tqdm
from scipy.stats import zscore

#%%
indexs = ['PG', 'FISV', 'ALL', 'IEX', 'CBRE', 'AMZN', 'BLK', 'CMA', 'HES', 'JCI' ]
input_csv = 'input.csv'
data_dir = 'CMCSA/'
index = 'CMCSA'

df = pd.read_csv(f'{data_dir}/{index}.csv', index_col=0)
df['Variation'] = df['High'] - df['Low']
df = df[['Close', 'Open', 'High', 'Low', 'Volume', 'Variation']]
# %%
close = df['Close'].to_numpy()
open = df['Open'].to_numpy()
high = df['High'].to_numpy()
low = df['Low'].to_numpy()
volumn = df['Volume'].to_numpy()
variation = df['Variation'].to_numpy()

# %%
window_size = 10

for i in range(0,4000, window_size):
    start = i
    end = i + window_size

    close[start:end] = zscore(close[start:end])
    open[start:end] = zscore(open[start:end])
    high[start:end] = zscore(high[start:end])
    low[start:end] = zscore(low[start:end])
    volumn[start:end] = zscore(volumn[start:end])
    variation[start:end] = zscore(variation[start:end])

# %%
df['Close'] = close
df['Open'] = open
df['High'] = high
df['Low'] = low
df['Volume'] = volumn
df['Variation'] = variation

df.to_csv('CMCSA.csv', index=False)
# %%

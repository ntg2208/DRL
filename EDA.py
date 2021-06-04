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

for idx, i in (enumerate(list_stocks[:1])):
    df = pd.read_csv(f'processing/{i}.csv').fillna(0)['Close'].to_numpy()
    coints = list()
    for j in tqdm(list_stocks[idx+1:]):
        df2 = pd.read_csv(f'processing/{j}.csv').fillna(0)['Close'].to_numpy()
        coint = ts.coint(df, df2)[1]
        # print(coint)
        coints.append(1-coint)
        # break
    row = prefill + np.array(coints)
    print(row)

    break



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




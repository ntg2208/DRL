#%%
import os
import numpy as np 
import pandas as pd
from pandas.io.parquet import FastParquetImpl
from tqdm import tqdm
from scipy.stats import zscore
import statsmodels.tsa.stattools as ts 
import talib
import matplotlib.pyplot as plt


#%%
indexs = ['LNT', 'DIS', 'KMX', 'DTE', 'EXR', 'EFX', 'ZBH', 'JNJ', 'PEP', 'DUK']
input_csv = 'input.csv'
data_dir = 'processing/'
chossen = 'CMCSA'
df = pd.read_csv(input_csv)
idx = df['idx'].to_numpy()

def normalization():
    df = pd.read_csv(f'{data_dir}/{index}.csv', index_col=0)
    df['Variation'] = df['High']
    df = df[['Close', 'Open', 'High', 'Low', 'Volume', 'Variation']]
    close = df['Close'].to_numpy()
    open = df['Open'].to_numpy()
    high = df['High'].to_numpy()
    low = df['Low'].to_numpy()
    volumn = df['Volume'].to_numpy()
    variation = df['Variation'].to_numpy()

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

    df['Close'] = close
    df['Open'] = open
    df['High'] = high
    df['Low'] = low
    df['Volume'] = volumn
    df['Variation'] = variation

    df.to_csv('CMCSA.csv', index=False)
# %%
def cointintegration():
    df = pd.read_csv(input_csv)
    cmcsa = pd.read_csv(f'{data_dir}/{chossen}.csv')['Close'].to_numpy()
    index = indexs[0]
    for index in indexs:
        df2 = pd.read_csv(f'{data_dir}/{index}.csv')['Close'].to_numpy()
        tmp = np.zeros(4000)
        for idx, _ in tqdm(enumerate(cmcsa)):
            if idx > 1:
                tmp[idx] = ts.coint(cmcsa[:idx+1], df2[:idx+1])[1]
            else:
                tmp[idx] = 0

        df[f'{index}'] = tmp
        df[f'{index}'] = df[f'{index}'].fillna(0) 

    df.to_csv(input_csv, index=False)
# %%
# df = pd.read_csv(f'processing/CMCSA.csv')
cmcsa = df['Close'].to_numpy()
bbands_upper, bbands_middle, bbands_lower = talib.BBANDS(cmcsa) 
#%%
df['BBANDS_upper'] = bbands_upper
df['BBANDS_middle'] = bbands_middle
df['BBANDS_lower'] = bbands_lower
df.fillna(0, inplace=True)
#%%
period = 10
o, h, l, c, v = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
df['DEMA'] = talib.DEMA(cmcsa, period)
df['EMA'] = talib.EMA(cmcsa, period)

df['SAREXT'] = talib.SAREXT(h, l, 0, 0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2)
df['SMA'] = talib.SMA(c, period)
df['TEMA'] = talib.TEMA(c, period)
df['WMA'] = talib.WMA(c, period)

df['ADXR'] = talib.ADXR(h, l, c, period)
df['APO'] = talib.APO(c, period, 2*period)

_, df['AROON_UP'] = talib.AROON(h, l, period)
df['CCI'] = talib.CCI(h, l, c, period)
df['CMO'] = talib.CMO(c, period)
df['MFI'] = talib.MFI(h, l, c, v, period)
df['MACD'], df['MACD_SIG'], df['MACD_HIST'] = talib.MACD(c, period, 2*period, period)

df['MOM'] = talib.MOM(c, period)
df['PLUS_DI'] = talib.PLUS_DI(h, l, c, period)

df['PPO'] = talib.PPO(c, period, 2*period)
df['ROC'] = talib.ROC(c, period)
df['ROCP'] = talib.ROCP(c, period)
df['RSI'] = talib.RSI(c, period)

df['SLOWK'], df['SLOWD'] = talib.STOCH(h, l, c, 5, 3, 0, 3, 0)
df['FASTK'], df['FASTD'] = talib.STOCHF(h, l, c, 5, 3, 0)
df['TRIX'] = talib.TRIX(c, period)

df['ULTOSC'] = talib.ULTOSC(h, l, c, 7, 14, 28)
df['WILLR'] = talib.WILLR(h, l, c, period)
df['AD'] = talib.AD(h, l, c, v)
df['OBV'] = talib.OBV(c, v)

df['ATR'] = talib.ATR(h, l, c, 14)
df['NATR'] = talib.NATR(h, l, c, 14)
df['HT_DCPERIOD'] = talib.HT_DCPERIOD(c)
df['HT_DCPHASE'] = talib.HT_DCPHASE(c)
df['INPHASE'], df['QUADRATURE'] = talib.HT_PHASOR(c)
df['SINE'], df['LEADSINE'] = talib.HT_SINE(c)


plt.figure(figsize=(9, 5))
plt.grid()
plt.xticks(rotation=45)

timeplot = 50
plt.plot(idx[-timeplot:], cmcsa[-timeplot:], label='CMCSA')
# plt.plot(idx[-100:], ema[-100:])
# plt.plot(idx[-100:], dema[-100:])
# plt.plot(idx[-timeplot:], bbands_upper[-timeplot:], label='bbands_upper')
# plt.plot(idx[-timeplot:], bbands_middle[-timeplot:], label='bbands_middle')
# plt.plot(idx[-timeplot:], bbands_lower[-timeplot:], label='bbands_lower')
# plt.plot(idx[-timeplot:], df['SAREXT'][-timeplot:], label='SAREXT')
# plt.plot(idx[-timeplot:], df['SMA'][-timeplot:], label='SMA')
# plt.plot(idx[-timeplot:], df['TEMA'][-timeplot:], label='TEMA')
# plt.plot(idx[-timeplot:], df['WMA'][-timeplot:], label='WMA')
# plt.plot(idx[-timeplot:], df['ADXR'][-timeplot:], label='ADXR')
plt.plot(idx[-timeplot:], df['AROON_UP'][-timeplot:], label='AROON_UP')

plt.legend()
plt.show()
plt.clf()

# plt.plot(pd.to_datetime(df['Date']), df['Open'])
# fig.autofmt_xdate()
# plt.title(f'{stock}')
# f['BBANDS_upper'] = 
# df['BBANDS_middle'] = 
# df['BBANDS_lower'] = 
# df['DEMA'] =
# df['EMA'] =
# df['SAREXT'] =
# df['SMA'] =
# df['TEMA'] =
# df['WMA'] =
# df['ADXR'] =
# df['APO'] =
# df['AROON_UP'] =
# df['CCI'] =
# df['CMO'] =
# df['MFI'] =
# df['MACD'] =
# df['MACD_SIG'] =
# df['MACD_HIST'] =
# df['MOM'] =
# df['PLUS_DI'] =
# df['PPO'] =
# df['ROC'] =
# df['ROCP'] =
# df['RSI'] =
# df['SLOWK'] =
# df['SLOWD'] =
# df['FASTK'] =
# df['FASTD'] =
# df['TRIX'] =
# df['ULTOSC'] =
# df['WILLR'] =
# df['AD'] =
# df['OBV'] =
# df['ATR'] =
# df['NATR'] =
# df['HT_DCPERIOD'] =
# df['HT_DCPHASE'] =
# df['INPHASE'] =
# df['QUADRATURE'] =
# df['SINE'] =
# df['LEADSINE']  =

# %%

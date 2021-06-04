#%%
import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from tqdm import tqdm
import yfinance as yf 


# %%
file_wiki = 'sp500_wiki.csv'
df_wiki = pd.read_csv(file_wiki)
# Symbol	Security	GICS Sector	GICS Sub-Industry	Headquarters Location	Date first added	CIK	Founded


# %%
print(df_wiki.head())
print(df_wiki['GICS Sector'].unique())


# %%
df_wiki[df_wiki['GICS Sector'] == 'Communication Services']['Symbol'].to_numpy()


# %%
stocks = df_wiki['Symbol'].to_numpy()
# print(stocks)


# %%
for stock in stocks:
    data = yf.download(stock, start='2001-01-01', end='2021-05-25')
    if len(data) > 0:
        data.to_csv(f'csv_data/{stock}.csv')
    
print('** Done **')


# %%
# plt.locator_params(axis = 'x',tight=True, nbins=4)
# def visualize()
for stock in tqdm(stocks):
    if os.path.isfile(f'csv_data/{stock}.csv'):
        df = pd.read_csv(f'csv_data/{stock}.csv')
        plt.figure(figsize=(9, 5))
        plt.grid()
        plt.xticks(rotation=45)
        plt.plot(pd.to_datetime(df['Date']), df['Close'])
        # fig.autofmt_xdate()
        plt.title(f'{stock}')
        plt.xlabel("Date")
        plt.ylabel("Close")
        plt.savefig(f'plot_csv/{stock}.jpg')
        plt.close()

print('** Done **')

#%%
from glob import glob

csvs = glob('csv_data/*.csv')
out_file = 'csv_info.csv'

names = list()
starts = list()
ends = list()
lens = list()

for file in tqdm(csvs):
    df = pd.read_csv(file)

    name = file.split('/')[-1][:-4]
    len_ = len(df)
    start = df['Date'][0]
    end = df['Date'][len_ -1]

    starts.append(start)
    ends.append(end)
    lens.append(len_)
    names.append(name)

    # print('\n')
    # print(start, end)
df = pd.DataFrame()
df['Name'] = names
df['Start'] = starts
df['End'] = ends
df['Len'] = lens

df.to_csv(out_file, index=False)
    
#%%
df = pd.read_csv(out_file)
print(df['Len'].value_counts())
print(df['Len'].unique())
#%%
list_stocks = df[df['Len'] > 4000]['Name'].to_numpy()
with open('examinating.txt', 'w') as f:
    for i in list_stocks:
        f.write(i + ', ')
# %%

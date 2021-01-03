# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 15:56:50 2020

@author: Vince
"""

import os
import pandas as pd
import numpy as np
import random
import pickle
from WindPy import w

import matplotlib.pyplot as plt
import mplfinance as mpf

os.chdir("E:/projects/sector_index_series_clustering")
from utilities_func import *

"""
########## pull raw data
w.start()
w.isconnected()

tickers =  ['882115.WI','886043.WI','886062.WI','882240.WI','886054.WI','886055.WI','886063.WI','882250.WI','882011.WI',
'886017.WI','882427.WI','882233.WI','882491.WI','882006.WI','882528.WI','882449.WI','886010.WI','886045.WI','886016.WI',
'886002.WI','886004.WI','882516.WI','882410.WI','886015.WI','886058.WI','886028.WI','886032.WI','886042.WI','886041.WI',
'886020.WI','882221.WI','886006.WI','886003.WI','882417.WI','886039.WI','882517.WI','882570.WI','886038.WI','886069.WI',
'886011.WI','882116.WI','882435.WI','886024.WI','886031.WI','886068.WI','886018.WI','886007.WI','886023.WI','886036.WI',
'886040.WI','886064.WI','886030.WI','886019.WI','886001.WI','882436.WI','886022.WI','886029.WI','882529.WI','886048.WI',
'886014.WI','886009.WI','882531.WI','886025.WI','882479.WI','882451.WI','886013.WI','886034.WI']

df=pd.DataFrame()
for ticker in tickers:
    a=w.wsd(ticker,"open,high,low,close,volume","2019-01-01","2020-09-01","PriceAdj=F",usedf=True)[1]
    a.loc[:,'Ticker']=ticker
    df=df.append(a)
df.to_pickle('./sector_index_raw.pkl')
"""
df=pd.read_pickle('./sector_index_raw.pkl')

########## slice data into N day sequence
        
sliced_data=[]   
N_days=10 
for i in df.Ticker.unique():
    sliced_data=sliced_data+slice_data(df, i, N_days)
    
#sliced_data=np.array(sliced_data)
random.shuffle(sliced_data)
pickle.dump(sliced_data, open('./sector_index_sliced.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
   
processed=[]
for i in sliced_data:
    processed.append(rescale_ohlcv(i))

processed=np.array(processed)
    
for i in range(10):
    #candlestick((df.loc[df.Ticker==tickers[i],df.columns[0:5]])[-50:-1].values)
    candlestick(sliced_data[i])
    candlestick(processed[i])
 

pickle.dump(processed, open('./sector_index_processed.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

#a=pd.read_pickle('sector_index_sliced.pkl')

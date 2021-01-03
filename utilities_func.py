# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 08:55:30 2020

@author: Vince
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

###### slice time series data into some sequences
def slice_data(df, ticker, n_seq=10):
    df1=(df.loc[df.Ticker==ticker,df.columns[0:5]]).copy()    
    
    output=[]
    for i in range(n_seq,len(df1)):
        output.append(df1.iloc[(i-n_seq):i,:].values)
        
    return output

###### rescale small sequence OHLCV data
def rescale_ohlcv(ohlcv):
    ohlcv1=ohlcv.copy()
    ohlcv1[:,0:4] = (ohlcv1[:,0:4] - ohlcv1[:,0:4].min())/(ohlcv1[:,0:4].max() - ohlcv1[:,0:4].min())
    ohlcv1[:,4] = (ohlcv1[:,4] - ohlcv1[:,4].min())/(ohlcv1[:,4].max() - ohlcv1[:,4].min())
    return ohlcv1

def ohlc2culr(df):
    arr=np.array(df)
    arr1=np.zeros((10,5),float)
    for i in range(len(arr)):
        arr1[i]=[arr[i,3],arr[i,1]-max(arr[i,0],arr[i,3]),min(arr[i,0],arr[i,3])-arr[i,2],arr[i,0]-arr[i,3],arr[i,4]]
    #scale ulr to same quantity level as c
    #cmax=np.max(arr1[:,0])
    #for i in range(1,4):
    #    if np.max(abs(arr1[:,i])!=0):
    #        arr1[:,i]=arr1[:,i]*cmax/np.max(abs(arr1[:,i]))
    return arr1

def split_features(arr):
    arr1=arr.copy()
    return arr1[:,0], arr1[:,1], arr1[:,2], arr1[:,3], arr1[:,4]

def my_dist(data,func,volume_wt=0):
    c0=[]
    u0=[]
    l0=[]
    r0=[]
    v0=[]
    for i in data:
        c,u,l,r,v = split_features(i)
        c0.append(c)
        u0.append(u)
        l0.append(l)
        r0.append(r)
        v0.append(v)
    #CULR distance are equally weighted
    d_culr = (func(np.array(c0)) + func(np.array(u0)) + func(np.array(l0)) + func(np.array(r0)))/4
    d_volume = func(np.array(v0))
    #volume distance can be over/under weight, default set to be equally weighted with price features
    dmtx = d_culr + d_volume * volume_wt
    
    return dmtx, d_volume
  
#display candlestick plots
def candlestick(ohlc,ttl="",save=False,name=1):
    
    if type(ohlc)==type(np.array(0)):
        df=pd.DataFrame(ohlc, columns=['open','high','low','close','volume'])
    elif type(ohlc)==type(pd.DataFrame()):    
        df=ohlc.copy()
        df.columns=['open','high','low','close','volume']
        
    df.set_index(pd.to_datetime(df.index),inplace=True)    
    if save==True:
        mpf.plot(df,type='candle', volume=True, title=ttl, savefig=f'obs_{name}')
    else:
        mpf.plot(df,type='candle', volume=True, title=ttl)

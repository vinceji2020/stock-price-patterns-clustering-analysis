# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:28:20 2020

@author: Vince
"""
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns

os.chdir("E:/projects/sector_index_series_clustering")
from utilities_func import *

def plot_clusterbar(label,pct=0):
    unique, counts = np.unique(label, return_counts=True)
    xxx1=pd.DataFrame({'label':unique, 'counts':counts, 'counts_pct':counts/np.sum(counts)*100}).sort_values(by=['counts'], ascending=False)
    xxx1=xxx1[xxx1.counts_pct>=pct]
    g = sns.barplot(x=xxx1.label,y=xxx1.counts_pct)
    

label=['cd_label1','cd_label2','cd_label3', #labels based on ed, cd, md with beta=0
       'cd_label4','cd_label5','cd_label6', #labels based on ed, cd, md with beta=0.25
       'volume_label1','volume_label2','volume_label3']  #labels based on ed, cd, md for volume
#which label to display, labels shown above
label_index=0
# Number of candlestick plots will be displayed
N=10
# which cluster within a label to display
n=1

data=pd.read_pickle('data_with_labels.pkl')
data0=data['x_data']
labels=data[label[label_index]]

plot_clusterbar(labels,0.5)

ngroup=data0[labels==n]
print(f'There are {len(ngroup)} elements in Label {n}')
for i in range(min(N,len(ngroup))):
    candlestick(ngroup[random.randint(0,len(ngroup)-1)],f'Group {n}')
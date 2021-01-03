# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:48:27 2020

@author: Vince
"""

import os
import numpy as np
import pandas as pd
import pickle
import random
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, manhattan_distances
from sklearn.cluster import AgglomerativeClustering

os.chdir("E:/projects/sector_index_series_clustering")
from utilities_func import *

data0=pd.read_pickle('sector_index_processed.pkl')
#### original dataset is too big, use a subset of original dataset
data0=data0[0:15000]
data1=[]
for i in data0:
    data1.append(ohlc2culr(i))
#data0 uses ohlc as features, data1 uses culr as features
data0=np.array(data0)
data1=np.array(data1)

######use culr features for better clustering result
funcs=[euclidean_distances, cosine_distances, manhattan_distances]
descrps=['euclidean_distance','cosine_distance','manhattan_distance']
wts=[0, 0.25]
#wt=0.25
for func, ddes in zip(funcs, descrps):
    for wt in wts:
        dp,dv=my_dist(data1, func, wt)
        del dv
        dmtx=dp
        print(f'{ddes}_wt_{wt}')
        K=range(0,1001,50)
        distance=[]
        for k in K:
            n=2 if k==0 else k
            print(n)
            clstmodel = AgglomerativeClustering(n_clusters=n, affinity="precomputed",linkage="average").fit(dmtx)
            tot_d=0 #total distance between each node
            tot_n=0 #total # of connections
            for i in range(n):
                #calculate average distance 
                subdmtx=dmtx[clstmodel.labels_==i]
                subdmtx=subdmtx[:,clstmodel.labels_==i]
                tot_d=tot_d+np.sum(np.triu(subdmtx))
                tot_n=tot_n+np.sum(np.triu(subdmtx)!=0)
            distance.append(tot_d/tot_n) #avearage distance between any connected two data
        wt1=int(wt*100)    
        dstns=plt.figure()
        plt.plot(K, distance, 'bx-') 
        plt.xlabel('Values of N clusters') 
        plt.ylabel(f'Average {ddes} within Cluster') 
        plt.title(f'The Elbow Method using {ddes}_{wt}(CULR)') 
        plt.show()
        dstns.savefig(f'elbow_method_{ddes}_wt_{wt1}_culr', bbox_inches='tight', pad_inches=0)
    
        result0=pd.DataFrame({'K':K,'Dist':distance})   
        result0.loc[:,'pct_chg']=result0.Dist.pct_change()*100
        result0.to_csv(f'N_{ddes}_volume_wt_{wt1}_culr.csv')  

###### pure volume distance######
funcs=[euclidean_distances, cosine_distances, manhattan_distances]
descrps=['euclidean_distance','cosine_distance','manhattan_distance']
#wt=0.25
for func, ddes in zip(funcs, descrps):
    d_p, d_v=my_dist(data1, func)
    del d_p
    dmtx=d_v
    print(f'{ddes}')
    K=range(0,1001,50)
    distance=[]
    for k in K:
        n=2 if k==0 else k
        print(n)
        clstmodel = AgglomerativeClustering(n_clusters=n, affinity="precomputed",linkage="average").fit(dmtx)
        tot_d=0 #total distance between each node
        tot_n=0 #total # of connections
        for i in range(n):
            #calculate average distance 
            subdmtx=dmtx[clstmodel.labels_==i]
            subdmtx=subdmtx[:,clstmodel.labels_==i]
            tot_d=tot_d+np.sum(np.triu(subdmtx))
            tot_n=tot_n+np.sum(np.triu(subdmtx)!=0)
        distance.append(tot_d/tot_n) #avearage distance between any connected two data
    #wt1=int(wt*100)    
    dstns=plt.figure()
    plt.plot(K, distance, 'bx-') 
    plt.xlabel('Values of N clusters') 
    plt.ylabel(f'Average {ddes} within Cluster') 
    plt.title(f'The Elbow Method using {ddes}(Volume)') 
    plt.show()
    dstns.savefig(f'elbow_method_{ddes}_volume', bbox_inches='tight', pad_inches=0)

    result0=pd.DataFrame({'K':K,'Dist':distance})   
    result0.loc[:,'pct_chg']=result0.Dist.pct_change()*100
    result0.to_csv(f'N_{ddes}_volume.csv')  

###### get labels ####################
d_p, d_v=my_dist(data1, euclidean_distances)
labela1=AgglomerativeClustering(n_clusters=350, affinity="precomputed",linkage="average").fit(d_p).labels_
vlabel1=AgglomerativeClustering(n_clusters=50, affinity="precomputed",linkage="average").fit(d_v).labels_
d_p, d_v=my_dist(data1, cosine_distances)
labela2=AgglomerativeClustering(n_clusters=250, affinity="precomputed",linkage="average").fit(d_p).labels_
vlabel2=AgglomerativeClustering(n_clusters=50, affinity="precomputed",linkage="average").fit(d_v).labels_
d_p, d_v=my_dist(data1, manhattan_distances)
labela3=AgglomerativeClustering(n_clusters=250, affinity="precomputed",linkage="average").fit(d_p).labels_
vlabel3=AgglomerativeClustering(n_clusters=50, affinity="precomputed",linkage="average").fit(d_v).labels_

d_p, d_v=my_dist(data1, euclidean_distances, 0.25)
labelb1=AgglomerativeClustering(n_clusters=350, affinity="precomputed",linkage="average").fit(d_p).labels_
d_p, d_v=my_dist(data1, cosine_distances, 0.25)
labelb2=AgglomerativeClustering(n_clusters=250, affinity="precomputed",linkage="average").fit(d_p).labels_
d_p, d_v=my_dist(data1, manhattan_distances, 0.25)
labelb3=AgglomerativeClustering(n_clusters=250, affinity="precomputed",linkage="average").fit(d_p).labels_


finaldata={'x_data':data0,'cd_label1':labela1,'cd_label2':labela2,'cd_label3':labela3,
           'cd_label4':labelb1,'cd_label5':labelb2,'cd_label6':labelb3,
           'volume_label1':vlabel1,'volume_label2':vlabel2,'volume_label3':vlabel3}

pickle.dump(finaldata, open('./data_with_labels.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def countc(label): 
    unique, counts = np.unique(label, return_counts=True)
    return sum(counts/sum(counts)>0.005)
    

def plot_clusterbar(label,pct=0):
    unique, counts = np.unique(label, return_counts=True)
    xxx1=pd.DataFrame({'label':unique, 'counts':counts, 'counts_pct':counts/np.sum(counts)*100}).sort_values(by=['counts'], ascending=False)
    xxx1=xxx1[xxx1.counts_pct>=pct]
    g = sns.barplot(x=xxx1.label,y=xxx1.counts_pct)

g=plt.figure()
plot_clusterbar(labela1,0.5)
plt.title(f'ED0 Label Counts as Pct of total Observation(pct>0.5%)') 
g.savefig(f'ed0_barplot', bbox_inches='tight', pad_inches=0)

g=plt.figure()
plot_clusterbar(labela2,0.5)
plt.title(f'CD0 Label Counts as Pct of total Observation(pct>0.5%)') 
g.savefig(f'cd0_barplot', bbox_inches='tight', pad_inches=0)

g=plt.figure()
plot_clusterbar(labela3,0.5)
plt.title(f'MD0 Label Counts as Pct of total Observation(pct>0.5%)') 
g.savefig(f'md0_barplot', bbox_inches='tight', pad_inches=0)

g=plt.figure()
plot_clusterbar(labelb1,0.5)
plt.title(f'ED25 Label Counts as Pct of total Observation(pct>0.5%)') 
g.savefig(f'ed25_barplot', bbox_inches='tight', pad_inches=0)

g=plt.figure()
plot_clusterbar(labelb2,0.5)
plt.title(f'CD25 Label Counts as Pct of total Observation(pct>0.5%)') 
g.savefig(f'cd25_barplot', bbox_inches='tight', pad_inches=0)

g=plt.figure()
plot_clusterbar(labelb3,0.5)
plt.title(f'MD25 Label Counts as Pct of total Observation(pct>0.5%)') 
g.savefig(f'md25_barplot', bbox_inches='tight', pad_inches=0)

g=plt.figure()
plot_clusterbar(vlabel1,1)
plt.title(f'Label Counts as Pct of total Observation(pct>1%)') 
g.savefig(f'edv_barplot', bbox_inches='tight', pad_inches=0)

g=plt.figure()
plot_clusterbar(vlabel2,1)
plt.title(f'Label Counts as Pct of total Observation(pct>1%)') 
g.savefig(f'cdv_barplot', bbox_inches='tight', pad_inches=0)

g=plt.figure()
plot_clusterbar(vlabel3,1)
plt.title(f'Label Counts as Pct of total Observation(pct>1%)')
g.savefig(f'mdv_barplot', bbox_inches='tight', pad_inches=0) 



d_p, d_v=my_dist(data1, cosine_distances)
label0=AgglomerativeClustering(n_clusters=1000, affinity="precomputed",linkage="average").fit(d_p).labels_
sns.countplot(label0)
unique, counts = np.unique(labelb3, return_counts=True)
xxx1=pd.DataFrame({'label':unique, 'counts':counts, 'counts_pct':counts/np.sum(counts)}).sort_values(by=['counts'], ascending=False)

####### sanity check#######
#dmtx=my_dist(data1, cosine_distances)
#clstmodel = AgglomerativeClustering(n_clusters=210, affinity="precomputed",linkage="average").fit(dmtx)
xxx=data0[labela1==39]
for i in range(10):
    candlestick(xxx[random.randint(0,len(xxx)-1)])




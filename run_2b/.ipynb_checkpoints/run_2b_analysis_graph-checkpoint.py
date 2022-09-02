%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd
import glob
import pickle
import networkx as nx
from networkx.generators.random_graphs import watts_strogatz_graph
from networkx.drawing.nx_pylab import draw_circular # unnecessary as the notebook is written, but useful if you want to visualize the graphs

from imp import reload
import sys
sys.path.insert(1,'/Users/biocomplexity/Projects/SocioCognitiveModeling/NSF-NetCogSys/Products/CAN_Model/CAN_GitRepo/IsingFitStudy/Hopfield/hop_pack')
import head as hd
import rate
import stat
reload(stat) #FOR SOME REASON NEED THIS

plt.style.use('fivethirtyeight')

grab_dir = '/Users/biocomplexity/Projects/SocioCognitiveModeling/NSF-NetCogSys/Products/CAN_Model/CAN_GitData/Hopfield/sw_data/run_2b_data/'

N = 1000
prewire = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.20, 0.40, 0.60, 1]
K = 10
replicates = range(0,10)

'''LOOP IT'''
wts_mean = np.array([])
wts_min = np.array([])

for i in prewire:
    print('i, prewire: ',i)
    
    for j in replicates:
        
        G_df = pd.read_csv(f'{grab_dir}EdgeList.N{N}.K{K}.Prewire{i}.Replicate{j}.wt',header=None,sep=' ')
        G_df.columns = ['a','b','c'] 
        G = nx.from_pandas_edgelist(G_df,'a','b','c',create_using=nx.Graph)
        G_mat = nx.to_numpy_matrix(G,weight='c')
        G_arr = np.array(G_mat)
        wts_mean = np.append(wts_mean,G_arr.sum(axis=1).mean())
        wts_min = np.append(wts_min,G_arr.sum(axis=1).min())
        


plt.scatter(np.array(range(0,len(wts_min))),wts_mean)
plt.scatter(np.array(range(0,len(wts_min))),wts_min)
plt.savefig('Mean_hi_Min_hi_draft.png',dpi=250)


'''TESTING CODE'''
G_df = pd.read_csv(f'{grab_dir}EdgeList.N{N}.K{K}.Prewire{prewire[8]}.Replicate{replicates[0]}.wt',header=None,sep=' ')
G_df.columns = ['a','b','c'] 
G = nx.from_pandas_edgelist(G_df,'a','b','c',create_using=nx.Graph)
G_mat = nx.to_numpy_matrix(G,weight='c')
G_arr = np.array(G_mat)

plt.hist(G_arr.sum(axis=1),bins=40)
G_arr.sum(axis=1).min()
G_arr.sum(axis=1).mean()

#END OF FILE
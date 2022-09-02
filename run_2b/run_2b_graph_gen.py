import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd
import glob

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

save_dir = '/Users/biocomplexity/Projects/SocioCognitiveModeling/NSF-NetCogSys/Products/CAN_Model/CAN_GitData/Hopfield/sw_data/run_2b_data/'

'''.........................
MAKE SW GRAPHS
.........................'''

'''PARAMS'''
N = 1000
prewire = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.20, 0.40, 0.60, 1]
K = 10
replicates = range(0,10)

for i in prewire:
    print('i, prewire: ',i)
    
    for j in replicates:
        filename = "N" + str(N) + ".K" + str(K) + ".Prewire" + str(i) + ".Replicate" + str(j) + ".wt"
        edge_filename = save_dir + "EdgeList." + filename

        Z = watts_strogatz_graph(N, K, 0)
        R = watts_strogatz_graph(N, K, i)
        Z.edges()==R.edges()
        #INTERSECTION
        R_int = nx.intersection(Z,R)
        R_new = nx.difference(R,Z) #IN FIRST NOT IN SECOND

        if len(R_int.edges()) > 0:
            R_int_edges_arr = np.array(R_int.edges())
            R_int_weights_arr = np.ones(len(R_int_edges_arr))
            R_int_weights_arr = R_int_weights_arr[..., None]
            R_int_weight_edge = np.append(R_int_edges_arr,R_int_weights_arr,1)
        
        if len(R_new.edges()) > 0:
            print("YES")
            R_new_edges_arr = np.array(R_new.edges())
            R_new_weights_arr = np.ones(len(R_new_edges_arr))
            R_new_weights_arr = R_new_weights_arr*-1
            R_new_weights_arr = R_new_weights_arr[..., None]
            R_new_weight_edge = np.append(R_new_edges_arr,R_new_weights_arr,1)

        if len(R_int.edges()) == 0:
            R_weight_edge_directed = R_new_weight_edge
        elif len(R_new.edges()) == 0:
            R_weight_edge_directed = R_int_weight_edge
        else:
            R_weight_edge_directed = np.append(R_int_weight_edge,R_new_weight_edge,0)

        #FOR TESTING PURPOSES SEE Z
        R_arr = np.array(R.edges())
        Z_arr = np.array(Z.edges())

        print('Z')
        print(Z_arr)
        print('len Z')
        print(len(Z_arr))
        print('R')
        print(R_arr)
        print('len R')
        print(len(R_arr))
        print('R_weight_edge_directed')
        print(R_weight_edge_directed)
        print('len R_weitght...')
        print(len(R_weight_edge_directed))

        #WORKS TO HERE
        bidir_mat_col0 = R_weight_edge_directed[:,1]
        bidir_mat_col0 = bidir_mat_col0[..., None]
        bidir_mat_col1 = R_weight_edge_directed[:,0]
        bidir_mat_col1 = bidir_mat_col1[..., None]
        bidir_mat_col2 = R_weight_edge_directed[:,2]
        bidir_mat_col2 = bidir_mat_col2[..., None]
          
        ## Appending the columns together
        bidir_mat = np.append(bidir_mat_col0, bidir_mat_col1, 1)
        bidir_mat = np.append(bidir_mat, bidir_mat_col2, 1)
          
        ## Creating the full, non-zero edgelist
        bidir_weight_edge = np.append(R_weight_edge_directed, bidir_mat, 0)
          
        ## Saving the edgelist 
        np.savetxt(edge_filename, bidir_weight_edge, delimiter = ' ', fmt='%i')

print("END EDGE LIST CREATION LOOP")

print("END OF PROGRAM")
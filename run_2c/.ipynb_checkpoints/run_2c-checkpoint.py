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

import pickle

'''NOTE, USING GRAPHS GENERATED FOR 2B here'''

grab_dir = '/Users/biocomplexity/Projects/SocioCognitiveModeling/NSF-NetCogSys/Products/CAN_Model/CAN_GitData/Hopfield/sw_data/run_2b_data/'
save_dir = '/Users/biocomplexity/Projects/SocioCognitiveModeling/NSF-NetCogSys/Products/CAN_Model/CAN_GitData/Hopfield/sw_data/run_2c_sim_out/'

'''SIMULATION'''

'''PARAMS'''
#FROM GRAPH GEN .py
N = 1000
prewire = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.20, 0.40, 0.60, 1]
K = 10
replicates = range(0,10)
#FOR SIM
#a = np.ones(N)
#U_a = a
time_steps = 4000
temperature = 0.1

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []
catch_design_mat = []
counter = 0

for i in prewire:
    print('i, prewire: ',i)
    
    for j in replicates:
        #GENERATE GRAPH
        G_df = pd.read_csv(f'{grab_dir}EdgeList.N{N}.K{K}.Prewire{i}.Replicate{j}.wt',header=None,sep=' ')
        G_df.columns = ['a','b','c'] 
        G = nx.from_pandas_edgelist(G_df,'a','b','c',create_using=nx.Graph)
        G_mat = nx.to_numpy_matrix(G,weight='c')
        G_arr = np.array(G_mat)

        #RUN SIM
        #print('PROB POSITIVE: ', i)
        W = G_arr.copy()
        T = np.zeros(N)
        U_a = rate.make_U_rate(1,N)[0,:]
        S = U_a.copy()
        S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
        catch_s_hist_l.append(S_hist)
        catch_endstate = np.vstack((catch_endstate,S))
        catch_energy = np.array([])
        for k in range(0,len(S_hist)):
            catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
        catch_energy_l.append(catch_energy)
                
        catch_design_mat.append([i,j])
        
        if counter == 1 or counter % 10 == 0:
            print('COUNTER: ',counter)
            print('WRITING INTERIM FILES:')
            np.save(f'{save_dir}catch_endstate_out_interim',catch_endstate)
            with open(f'{save_dir}catch_energy_l_out_interim','wb') as filehandle:
                pickle.dump(catch_energy_l, filehandle)
        counter+=1
                
print("DONE SIM LOOP")
print("WRITING FINAL FILES")
np.save(f'{save_dir}catch_endstate_out_final',catch_endstate)
with open(f'{save_dir}catch_energy_l_out_final','wb') as filehandle:
    pickle.dump(catch_energy_l, filehandle)
print("FILES WRITTEN")
print("END OF PROGRAM:  GO HOME!")


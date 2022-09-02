import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import subprocess
from numpy import genfromtxt
from imp import reload
import networkx as nx
import pandas as pd
''' CUSTOM PACKAGES'''
import sys
sys.path.insert(1,'/Users/biocomplexity/Projects/SocioCognitiveModeling/NSF-NetCogSys/Products/CAN_Model/CAN_GitRepo/IsingFitStudy/Hopfield/hop_pack')
import head as hd
import rate
import stat
reload(stat) #FOR SOME REASON NEED THIS
import pickle

def convert_dec_to_array(x):
    mystr = format(x, '022b') #WITH PADDING
    return np.fromiter(mystr, dtype=int)

'''ANALYSIS FOR THE ARTICLE SUBMISSION IS ANNOTATED BELOW'''

save_get_dir = '/Users/biocomplexity/Projects/SocioCognitiveModeling/NSF-NetCogSys/Products/CAN_Model/CAN_GitData/Hopfield/run_1b_sim_out/'


#GEN DATA
#IMPOR AND PROCESS WITH R
print('START R PROCESS')
subprocess.check_call(['Rscript', 'Graphs/IsingFit.r'], shell=False)
print('END R PROCESS')
#LOAD CSV TO PYTHON
#CONVERT TO W_if
from numpy import genfromtxt
W_if = genfromtxt('Graphs/graph_from_R.csv', delimiter=',')
T_if = genfromtxt('Graphs/thresholds_from_R.csv', delimiter=',')
U_if = genfromtxt('Graphs/data_from_R.csv',delimiter=',')




'''NETWORK ANALYSIS DEV'''
G = nx.from_numpy_matrix(W_if, create_using=nx.Graph) #SHOULD BE UNDIRECTD

#PLOT IT
nx.draw_networkx(G)                                                        
limits = plt.axis('on')                                                   
plt.show(G)
g_deg = nx.degree_centrality(G)
g_close = nx.closeness_centrality(G)
g_between = nx.betweenness_centrality(G)

#WEIGHT VALUES MUST BE TAKEN INTO ACCOUNT
for i in range(0,len(W_if)):
    print('node ',i,' sum wts: ',W_if[i].sum())
'''WEIGHT VALUES AS SUMS PER NODE TELL PARTIAL STORY'''


    
'''
TURN ON EACH OF THE UNITS INDEPENDENTLY + CLAMPING
'''
a = np.zeros((22,22))
np.fill_diagonal(a,1)
W = W_if.copy()
T = T_if.copy()
N = 22
temp = 0.1
replicates = 1000 
catch_endstate_clamp_sum = np.zeros((22,22))
catch_catch_endstate_clamp = []
catch_energy_l = []
catch_s_hist_l = []

for j in range(0,replicates):  #TO GENERATE SAMPLES OF THE PROCESS
    print('j',j)
    catch_overlap = np.array([]) 
    catch_ham = np.array([])
    catch_endstate_clamp = np.full(22,99)

    U_a = a
    for i in range(0,len(U_a)):
        S = U_a[i].copy()
        T_clamp = T.copy() #FIX THRESHOLD REALL HIGH FOR NODE J
        T_clamp[i] = 1000
        S[i] = 1 #FIX S FOR Node J
        S_hist = rate.sim_patt_U_prob_rate_if(S,W,T_clamp,2000,temp)
        catch_s_hist_l.append(S_hist)
        #print('ran catch_s_hist_l')
        catch_overlap = np.append(catch_overlap, hd.compute_M(S,U_a[i],N))
        catch_ham = np.append(catch_ham,np.sum(np.bitwise_xor(S.astype(int),U_a[i].astype(int))))
        catch_endstate_clamp = np.vstack((catch_endstate_clamp,S))
        catch_energy = np.array([])
        for j in range(0,len(S_hist)):
            catch_energy = np.append(catch_energy,rate.energy(S_hist[j],W,T))
        catch_energy_l.append(catch_energy)
    
        #print(' i:',i)
        
    catch_endstate_clamp_sum = catch_endstate_clamp_sum + catch_endstate_clamp[1:]
    catch_catch_endstate_clamp.append(catch_endstate_clamp)
    
print("DONE")

'''SAVE RESULTS FOR LATER
...LATEST SAVE was March 12
'''
with open(f'{save_get_dir}catch_catch_endstate_clamp_temp{temp}_replicates{replicates}_list','wb') as filehandle:                            
    pickle.dump(catch_catch_endstate_clamp, filehandle)  

with open(f'{save_get_dir}catch_energy_l_temp{temp}_replicates{replicates}_list','wb') as filehandle:                            
    pickle.dump(catch_energy_l, filehandle) 
    
with open(f'{save_get_dir}catch_s_hist_l_temp{temp}_replicates{replicates}_list','wb') as filehandle:                            
    pickle.dump(catch_s_hist_l, filehandle) 

np.save(f'{save_get_dir}catch_endstate_clamp_sum_temp{temp}_replicates{replicates}',catch_endstate_clamp_sum)


EOF

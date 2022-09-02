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

grab_dir = '/Users/biocomplexity/Projects/SocioCognitiveModeling/NSF-NetCogSys/Products/CAN_Model/CAN_GitData/Hopfield/sw_data/run_2b_sim_out/'

'''PARAMS''' 
'''FROM run_2b.py, the sim program'''
#FROM GRAPH GEN MIGHT NOT USE THEM ALL
N = 1000
prewire = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.20, 0.40, 0.60, 1]
K = 10
replicates = range(0,10)
#FOR SIM
a = np.ones(N)
U_a = a
time_steps = 4000
temperature = 0.1


'''GRAB SIM DATA'''
catch_endstate = np.load(f'{grab_dir}catch_endstate_out_final.npy')

with open(f'{grab_dir}catch_energy_l_out_final','rb') as filehandle:
    catch_energy_l = pickle.load(filehandle)   
    
'''Analysis'''
#FOR ALL SIMS
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
#FOR ONE P-REWIRE, AND ONE ARCH SHOW ALL PROB-POSITIVES
for i in range(0,11): plt.plot((-1*catch_energy_l[i])) #P-REWIRE 0, 
for i in range(90,100): plt.plot((-1*catch_energy_l[i])) #P-REWIRE 1

#ALL SIMS
hd.plot_hist(catch_endstate[1:])
#ONE P-REWIRE AND ONE ARCH SHOW ALL PROB-POSITIVES
hd.plot_hist(catch_endstate[1:12])#P-REWIRE 0
hd.plot_hist(catch_endstate[91:100]) #P REWIRE 1

'''USED FOR FIRST SUBMISSION'''
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.xlabel('Simulation Time')
plt.ylabel('Energy')
plt.savefig('EnergyOverTime.png',dpi=400,bbox_inches='tight')
#TO VERIFY THE VALUES PER BAND
for i in range(0,len(catch_energy_l)): print('i:  ',i,'  \nEnergyOverTime:  ',(-1*catch_energy_l[i]).max())
    
'''USED FOR FIRST SUBMISSION'''
plt.xlabel('State Vector Index')
plt.ylabel('Simulation Number')
hd.plot_hist(catch_endstate[1:])
plt.savefig('Endstates_AllSims.png',dpi=400,bbox_inches='tight')

'''USEFUL FOR EXPLAINING RESULTS'''
for i in range(0,101): print(i, ' ',catch_endstate[i].mean())

'''
USE CATCH_ENERGY_L last element of each list as endsate energy
CAN'T USE CATCH_ENDSTATE BC WOULD NEED TO CALL THE RIGHT WEIGHT MATRIX
'''    
#MAKE AS ARRAY
catch_energy_l_arr = -(np.array(catch_energy_l))#MAKE IT NEGATIVE HERE
catch_energy_end = catch_energy_l_arr[:,time_steps]#GRAB LAST ENERGY DURING EACH SIM

catch_energy_end_mean = np.array([])
catch_energy_end_sd = np.array([])
for i in range(0,10):
    tmp_vect = np.array([])
    for j in range(0,10):
        a = i*10
        b = a+j
        tmp_vect = np.append(tmp_vect,catch_energy_end[b])
    print(tmp_vect)
    print(tmp_vect.mean())
    print(tmp_vect.std())
    catch_energy_end_mean = np.append(catch_energy_end_mean,tmp_vect.mean())
    catch_energy_end_sd = np.append(catch_energy_end_sd,tmp_vect.std())


'''USED FOR FIRST SUBMISSION'''
#PLOT AS SCATTER
tmp1 = prewire[1:]
tmp2 = catch_energy_end_mean[1:]
plt.scatter(tmp1,tmp2,color='black',s=5)
plt.xscale('log')
plt.xlabel('Rewire Probability')
plt.ylabel('Energy')
plt.savefig('EnergyByPrewire_Scatter.png',dpi=400,bbox_inches='tight')

#PLOT AS SCATTER
tmp1 = prewire[1:]
tmp2 = catch_energy_end_sd[1:]
plt.scatter(tmp1,tmp2,color='black',s=5)
plt.xscale('log')
plt.xlabel('P-Rewire')
plt.ylabel('Energy')
plt.savefig('EnergyByPrewire_Scatter_SD.png',dpi=400,bbox_inches='tight')

'''RESULTS CONCLUSIVE'''

#END OF FILE
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

def ambig_level(x):
    catch = np.array([])
    for i in range(0,len(x)): 
        sum_a = x[i,0:4].sum()
        sum_b = x[i,4:8].sum()
        diff_ab = sum_b - sum_a
        catch = np.append(catch,diff_ab)
        
    return catch

#HAND WIRE THINGIE
def make_hand_wts(ib,ie,jb,je,wt,mat):
    '''
    ib=i begin
    ie=i end
    jb,je, see i.
    wt is pos 1 or neg 1 for the weight to assign
    '''
    for i in range(ib,ie+1):
        for j in range(jb,je+1):
            mat[i,j]=wt
            print(i,j)
            
    return mat






'''THREAD 1'''
'''JUST LOOKING AT WEIGHTS IN TWO-CLUSTERED SYSTEM'''
U_tmp1 = np.array([0,0,0,1,1,1])
U_tmp2 = np.array([1,1,1,0,0,0])

U_tmp = np.vstack((U_tmp1,U_tmp2))

W_tmp = rate.make_W_rate(U_tmp)
W_tmp.round(2)



'''THREAD 2'''
'''LOOKING AT ENERGY AND ATTRACTORS IN SIMPLE TWO-CLUSTERED SYSTEM'''

'''GENERAL SIM PARAMS'''
N = 8
time_steps = 100
temperature = 0.01


'''SIMULATIONS'''

'''SIM SET 1
---Start with fully connected and reduce systematically
---With Random bit vector as S_0
'''

'''BASIC WEIGHT CREATION'''
U1 = np.array([1,1,1,1,0,0,0,0])
U2 = np.array([0,0,0,0,1,1,1,1])
Ucomb = np.vstack((U1,U2))
W = rate.make_W_rate(Ucomb)
#W0 just ot get structure, we will hand wire
#THIS OVERWRITES W
make_hand_wts(0,3,0,3,1,W)
make_hand_wts(0,3,4,7,-1,W)
make_hand_wts(4,7,0,3,-1,W)
make_hand_wts(4,7,4,7,1,W)

for i in range(0,W.shape[0]):
    W[i,i] = 0 
'''NOW HAVE FULLY CONNECTED NET'''

catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())

'''NOW REDUCE THE NUMBER OF NEGATIVE BY 8 (the negative diagonals)'''
for i in range(0,4):
    W[i,i+4] = 0
    W[i+4,i] = 0
'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W  
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())

'''NOW REDUCE ONE MORE NEGATIVE PER ROW FOR TWO NODES IN EACH BANK'''
W[0,5] = 0
W[5,0] = 0
W[1,6] = 0
W[6,1] = 0

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())

'''REDUCE ONE MORE NEGATIVE PER ONE MORE ROW'''
W[2,7] = 0
W[7,2] = 0

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,1000):
    T = np.zeros(N)
    U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())

'''REDUCE LAST ROW  NEGATIVE PER ONE MORE ROW'''
W[3,4] = 0
W[4,3] = 0

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())

'''REDUCE SO ONLY ONE NEGATIVE PER ROW'''
W[0,6] = 0
W[6,0] = 0
W[1,7] = 0
W[7,1] = 0
W[2,4] = 0
W[4,2] = 0
W[3,5] = 0
W[5,3] = 0

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())


'''SIM SET 2
---Start with fully connected and reduce systematically
---With AllOnes S_0
'''

'''BASIC WEIGHT CREATION'''
U1 = np.array([1,1,1,1,0,0,0,0])
U2 = np.array([0,0,0,0,1,1,1,1])
Ucomb = np.vstack((U1,U2))
W = rate.make_W_rate(Ucomb)
#W0 just ot get structure, we will hand wire
#THIS OVERWRITES W
make_hand_wts(0,3,0,3,1,W)
make_hand_wts(0,3,4,7,-1,W)
make_hand_wts(4,7,0,3,-1,W)
make_hand_wts(4,7,4,7,1,W)

for i in range(0,W.shape[0]):
    W[i,i] = 0 
'''NOW HAVE FULLY CONNECTED NET'''


catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
(-1*np.array(catch_energy_l)).min()
(-1*np.array(catch_energy_l)).mean()

'''NOW REDUCE THE NUMBER OF NEGATIVE BY 8 (the negative diagonals)'''
for i in range(0,4):
    W[i,i+4] = 0
    W[i+4,i] = 0
'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W  
(-1*np.array(catch_energy_l)).min()
(-1*np.array(catch_energy_l)).mean()

'''NOW REDUCE ONE MORE NEGATIVE PER ROW FOR TWO NODES IN EACH BANK'''
W[0,5] = 0
W[5,0] = 0
W[1,6] = 0
W[6,1] = 0

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
(-1*np.array(catch_energy_l)).min()
(-1*np.array(catch_energy_l)).mean()

'''REDUCE ONE MORE NEGATIVE PER ONE MORE ROW'''
W[2,7] = 0
W[7,2] = 0

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,1000):
    T = np.zeros(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
(-1*np.array(catch_energy_l)).min()
(-1*np.array(catch_energy_l)).mean()

'''REDUCE LAST ROW  NEGATIVE PER ONE MORE ROW'''
W[3,4] = 0
W[4,3] = 0

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
(-1*np.array(catch_energy_l)).min()
(-1*np.array(catch_energy_l)).mean()

'''REDUCE SO ONLY ONE NEGATIVE PER ROW'''
W[0,6] = 0
W[6,0] = 0
W[1,7] = 0
W[7,1] = 0
W[2,4] = 0
W[4,2] = 0
W[3,5] = 0
W[5,3] = 0

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())






'''SIM SET 3 AND 4 KEEP NUMBER OF WEIGHTS EQUAL ACROSS CONDITIONS'''
'''GENERAL SIM PARAMS'''
N = 8
time_steps = 200
temperature = 0.01


'''SIMULATIONS'''
'''SIM SET 3
---Start with fully connected and reduce systematically
---With Random bit vector as S_0
'''

'''BASIC WEIGHT CREATION'''
U1 = np.array([1,1,1,1,0,0,0,0])
U2 = np.array([0,0,0,0,1,1,1,1])
Ucomb = np.vstack((U1,U2))
W = rate.make_W_rate(Ucomb)
#W0 just ot get structure, we will hand wire
#THIS OVERWRITES W
make_hand_wts(0,3,0,3,1,W)
make_hand_wts(0,3,4,7,0,W)
make_hand_wts(4,7,0,3,0,W)
make_hand_wts(4,7,4,7,1,W)

for i in range(0,W.shape[0]):
    W[i,i] = 0 

'''NOW HAVE FULLY CLUSTERED NET'''

'''A. FULLY CLUSTERED'''
'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()

'''B. REWIRE TWO UNDIRECTED (BIDIRECTED) EDGE'''
W[0,1] = 0
W[1,0] = 0
W[0,4] = -1
W[4,0] = -1
W[7,6] = 0
W[6,7] = 0
W[7,3] = -1
W[3,7] = -1
'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()


'''C. REWIRE TWO MORE UNDIRECTED (BIDIRECTED) EDGE'''
W[1,2] = 0
W[2,1] = 0
W[1,5] = -1
W[5,1] = -1
W[6,5] = 0
W[5,6] = 0
W[6,2] = -1
W[2,6] = -1

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()


'''D. REWIRE TWO MORE UNDIRECTED (BIDIRECTED) EDGE'''
W[2,0] = 0
W[0,2] = 0
W[2,5] = -1
W[5,2] = -1
W[5,7] = 0
W[7,5] = 0
W[0,7] = -1
W[7,0] = -1

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()








'''SIM SET 5 AND 6 KEEP NUMBER OF WEIGHTS EQUAL ACROSS CONDITIONS
THIS IS A REPLICATE OF SET 3 and 4, but with random bit vector 
restricted to only one side of the "necker" cube.
'''
'''GENERAL SIM PARAMS'''
N = 8
time_steps = 200
temperature = 0.01

'''TESTING NEW INPUT METHOD'''
def U_half_bits(N):
    input_N = int(N/2)
    U_half = rate.make_U_rate(1,input_N)[0,:]
    U_zeros = np.zeros(input_N)
    if np.random.binomial(1, 0.50)==1:
        U_a = np.append(U_half,U_zeros)
    else:
        U_a = np.append(U_zeros,U_half)
    
    return U_a
#THIS WORKS

#U_a = np.ones(N)

'''SIMULATIONS'''
'''SIM SET 3
---Start with fully connected and reduce systematically
---With Random bit vector as S_0
'''

'''BASIC WEIGHT CREATION'''
U1 = np.array([1,1,1,1,0,0,0,0])
U2 = np.array([0,0,0,0,1,1,1,1])
Ucomb = np.vstack((U1,U2))
W = rate.make_W_rate(Ucomb)
#W0 just ot get structure, we will hand wire
#THIS OVERWRITES W
make_hand_wts(0,3,0,3,1,W)
make_hand_wts(0,3,4,7,0,W)
make_hand_wts(4,7,0,3,0,W)
make_hand_wts(4,7,4,7,1,W)

for i in range(0,W.shape[0]):
    W[i,i] = 0 

'''NOW HAVE FULLY CLUSTERED NET'''

'''A. FULLY CLUSTERED'''
'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    U_a = U_half_bits(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()

'''B. REWIRE TWO UNDIRECTED (BIDIRECTED) EDGE'''
W[0,1] = 0
W[1,0] = 0
W[0,4] = -1
W[4,0] = -1
W[7,6] = 0
W[6,7] = 0
W[7,3] = -1
W[3,7] = -1
'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    U_a = U_half_bits(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()


'''C. REWIRE TWO MORE UNDIRECTED (BIDIRECTED) EDGE'''
W[1,2] = 0
W[2,1] = 0
W[1,5] = -1
W[5,1] = -1
W[6,5] = 0
W[5,6] = 0
W[6,2] = -1
W[2,6] = -1

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    U_a = U_half_bits(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()


'''D. REWIRE TWO MORE UNDIRECTED (BIDIRECTED) EDGE'''
W[2,0] = 0
W[0,2] = 0
W[2,5] = -1
W[5,2] = -1
W[5,7] = 0
W[7,5] = 0
W[0,7] = -1
W[7,0] = -1

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    U_a = U_half_bits(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()







'''INTERESTING TEST'''
'''SIM SET 7 AND 8 REPLICATE OF SET 3 and 4, but with zero inputs as S_0.
'''
'''GENERAL SIM PARAMS'''
N = 8
time_steps = 1000
temperature = 0.01


'''SIMULATIONS'''
'''SIM SET 3
---Start with fully connected and reduce systematically
---With Random bit vector as S_0
'''

'''BASIC WEIGHT CREATION'''
U1 = np.array([1,1,1,1,0,0,0,0])
U2 = np.array([0,0,0,0,1,1,1,1])
Ucomb = np.vstack((U1,U2))
W = rate.make_W_rate(Ucomb)
#W0 just ot get structure, we will hand wire
#THIS OVERWRITES W
make_hand_wts(0,3,0,3,1,W)
make_hand_wts(0,3,4,7,0,W)
make_hand_wts(4,7,0,3,0,W)
make_hand_wts(4,7,4,7,1,W)

for i in range(0,W.shape[0]):
    W[i,i] = 0 

'''NOW HAVE FULLY CLUSTERED NET'''

'''A. FULLY CLUSTERED'''
'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.zeros(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()

'''B. REWIRE TWO UNDIRECTED (BIDIRECTED) EDGE'''
W[0,1] = 0
W[1,0] = 0
W[0,4] = -1
W[4,0] = -1
W[7,6] = 0
W[6,7] = 0
W[7,3] = -1
W[3,7] = -1
'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.zeros(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()


'''C. REWIRE TWO MORE UNDIRECTED (BIDIRECTED) EDGE'''
W[1,2] = 0
W[2,1] = 0
W[1,5] = -1
W[5,1] = -1
W[6,5] = 0
W[5,6] = 0
W[6,2] = -1
W[2,6] = -1

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.zeros(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()


'''D. REWIRE TWO MORE UNDIRECTED (BIDIRECTED) EDGE'''
W[2,0] = 0
W[0,2] = 0
W[2,5] = -1
W[5,2] = -1
W[5,7] = 0
W[7,5] = 0
W[0,7] = -1
W[7,0] = -1

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.zeros(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()






'''GENERAL SIM PARAMS'''
N = 8
time_steps = 1000
temperature = 0.01


'''SIMULATIONS'''
'''SIM SET 9-10
---NEW CONFIGURATION, MORE BALANCED
---With Random bit vector as S_0
'''

'''BASIC WEIGHT CREATION'''
U1 = np.array([1,1,1,1,0,0,0,0])
U2 = np.array([0,0,0,0,1,1,1,1])
Ucomb = np.vstack((U1,U2))
W = rate.make_W_rate(Ucomb)
#W0 just ot get structure, we will hand wire
#THIS OVERWRITES W
make_hand_wts(0,3,0,3,1,W)
make_hand_wts(0,3,4,7,0,W)
make_hand_wts(4,7,0,3,0,W)
make_hand_wts(4,7,4,7,1,W)

for i in range(0,W.shape[0]):
    W[i,i] = 0 

'''NOW HAVE FULLY CLUSTERED NET'''

'''A. FULLY CLUSTERED'''
'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = U_half_bits(N)
    U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()

'''B. REWIRE TWO UNDIRECTED (BIDIRECTED) EDGE'''
W[0,1] = 0
W[1,0] = 0
W[0,4] = -1
W[4,0] = -1
W[4,5] = 0
W[5,4] = 0
W[1,5] = -1
W[5,1] = -1
'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = U_half_bits(N)
    U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()


'''C. REWIRE TWO MORE UNDIRECTED (BIDIRECTED) EDGE'''
W[2,3] = 0
W[3,2] = 0
W[3,7] = -1
W[7,3] = -1
W[6,7] = 0
W[7,6] = 0
W[6,2] = -1
W[2,6] = -1

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = U_half_bits(N)
    U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()


'''D. REWIRE TWO MORE UNDIRECTED (BIDIRECTED) EDGE'''
W[1,2] = 0
W[2,1] = 0
W[1,7] = -1
W[7,1] = -1
W[4,7] = 0
W[7,4] = 0
W[4,2] = -1
W[2,4] = -1

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = U_half_bits(N)
    U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()





'''GENERAL SIM PARAMS'''
N = 8
time_steps = 1000
temperature = 0.01


'''SIMULATIONS'''
'''SIM SET 11-12
---NEW CONFIGURATION, MORE BALANCED
---With Ones as vector as S_0
'''

'''BASIC WEIGHT CREATION'''
U1 = np.array([1,1,1,1,0,0,0,0])
U2 = np.array([0,0,0,0,1,1,1,1])
Ucomb = np.vstack((U1,U2))
W = rate.make_W_rate(Ucomb)
#W0 just ot get structure, we will hand wire
#THIS OVERWRITES W
make_hand_wts(0,3,0,3,1,W)
make_hand_wts(0,3,4,7,0,W)
make_hand_wts(4,7,0,3,0,W)
make_hand_wts(4,7,4,7,1,W)

for i in range(0,W.shape[0]):
    W[i,i] = 0 

'''NOW HAVE FULLY CLUSTERED NET'''

'''A. FULLY CLUSTERED'''
'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = U_half_bits(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()

'''B. REWIRE TWO UNDIRECTED (BIDIRECTED) EDGE'''
W[0,1] = 0
W[1,0] = 0
W[0,4] = -1
W[4,0] = -1
W[4,5] = 0
W[5,4] = 0
W[1,5] = -1
W[5,1] = -1
'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = U_half_bits(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()


'''C. REWIRE TWO MORE UNDIRECTED (BIDIRECTED) EDGE'''
W[2,3] = 0
W[3,2] = 0
W[3,7] = -1
W[7,3] = -1
W[6,7] = 0
W[7,6] = 0
W[6,2] = -1
W[2,6] = -1

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = U_half_bits(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()


'''D. REWIRE TWO MORE UNDIRECTED (BIDIRECTED) EDGE'''
W[1,2] = 0
W[2,1] = 0
W[1,7] = -1
W[7,1] = -1
W[4,7] = 0
W[7,4] = 0
W[4,2] = -1
W[2,4] = -1

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = U_half_bits(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()







'''GENERAL SIM PARAMS'''
N = 8
time_steps = 1000
temperature = 0.01


'''SIMULATIONS'''
'''SIM SET 13-14
---NEW CONFIGURATION, MORE BALANCED
---With Zeross as vector as S_0
'''

'''BASIC WEIGHT CREATION'''
U1 = np.array([1,1,1,1,0,0,0,0])
U2 = np.array([0,0,0,0,1,1,1,1])
Ucomb = np.vstack((U1,U2))
W = rate.make_W_rate(Ucomb)
#W0 just ot get structure, we will hand wire
#THIS OVERWRITES W
make_hand_wts(0,3,0,3,1,W)
make_hand_wts(0,3,4,7,0,W)
make_hand_wts(4,7,0,3,0,W)
make_hand_wts(4,7,4,7,1,W)

for i in range(0,W.shape[0]):
    W[i,i] = 0 

'''NOW HAVE FULLY CLUSTERED NET'''

'''A. FULLY CLUSTERED'''
'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = U_half_bits(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.zeros(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()

'''B. REWIRE TWO UNDIRECTED (BIDIRECTED) EDGE'''
W[0,1] = 0
W[1,0] = 0
W[0,4] = -1
W[4,0] = -1
W[4,5] = 0
W[5,4] = 0
W[1,5] = -1
W[5,1] = -1
'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = U_half_bits(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.zeros(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()


'''C. REWIRE TWO MORE UNDIRECTED (BIDIRECTED) EDGE'''
W[2,3] = 0
W[3,2] = 0
W[3,7] = -1
W[7,3] = -1
W[6,7] = 0
W[7,6] = 0
W[6,2] = -1
W[2,6] = -1

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = U_half_bits(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.zeros(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()


'''D. REWIRE TWO MORE UNDIRECTED (BIDIRECTED) EDGE'''
W[1,2] = 0
W[2,1] = 0
W[1,7] = -1
W[7,1] = -1
W[4,7] = 0
W[7,4] = 0
W[4,2] = -1
W[2,4] = -1

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    #U_a = U_half_bits(N)
    #U_a = rate.make_U_rate(1,N)[0,:]
    U_a = np.zeros(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
print((-1*np.array(catch_energy_l)).min(),(-1*np.array(catch_energy_l)).mean())
pd.Series(ambig_level(catch_endstate[1:])).value_counts()








'''SCRATCH '''

'''NOW ADD BACK A FEW MORE'''
W[0,5] = -1
W[5,0] = -1
W[0,7] = -1
W[7,0] = -1
W[3,4] = -1
W[4,3] = -1
'''AT THIS POINT< WITH RANDOM INPUTS, GET A VERY FEW NECKERS
WITH ALL ONES INPUT, ALL ONES OUT'''
W[2,6] = -1#FIVE NEGATIVE (BI-directional) EDGES
W[6,2] = -1
'''AT THIS POINT, WITH RANDOM INPUTS, GET LOTS OF NECKERS, SOME ALL ONES
WITH ALL ONES, GET ALL ONES OUT'''
W[3,5] = -1#SIX NEGATIVE (BI-directional) EDGES
W[5,3] = -1
'''AT THIS POINT, WITH RANDOM, ALL NECKERS (OR CLOSE), 
WITH ALL ONES< GET ALL ONES< BUT ENERGY is -12'''
W[0,4] = -1#7 NEGATIVE (BI-directional) EDGES
W[4,0] = -1

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
    T = np.zeros(N)
    U_a = rate.make_U_rate(1,N)[0,:]
    #U_a = np.ones(N)
    S = U_a.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,time_steps,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for k in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[k],W,T))
    catch_energy_l.append(catch_energy)

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W



'''NOW ADD BACK A FEW MORE'''
make_hand_wts(0,3,6,7,-1,W)
make_hand_wts(6,7,0,3,-1,W)

'''SIM'''
catch_endstate = np.full(N,99)
catch_energy_l = []
catch_s_hist_l = []

for i in range(0,100):
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

'''ANALYSIS'''
plt.subplot(2,3,1)
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]),linewidth=0.2)
plt.subplot(2,3,2)
hd.plot_hist(catch_endstate[1:])
plt.subplot(2,3,3)
plt.hist(ambig_level(catch_endstate[1:]))
W
#EOF 
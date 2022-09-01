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

t = pd.read_csv('Graphs/thresholds_from_R.csv',header=None)
plt.hist(t,bins=20)
t_m = t.mean()
t_sd = t.std()

w = pd.read_csv('Graphs/graph_from_R.csv',header=None)
plt.hist(np.array(w))
w_m = np.array(w).mean()
w_sd = np.array(w).std()

'''EXPECTED VALUE OF ON'''
d = pd.read_csv('Graphs/data_from_R.csv',header=None)
plt.hist(np.array(d))
w_m = np.array(d).mean()
w_sd = np.array(d).std()

'''CASE 1
MEANS ON ALL AND 50% on S_i
'''
catch_h_noc = np.array([])
catch_h_wc = np.array([])
catch_sig_noc = np.array([])
catch_sig_wc = np.array([])
for i in range(0,22):
    print('i: ',i)
    frac_on = i
    h_noc = w_m*frac_on + t_m
    h_wc = w_m*frac_on + t_m + w_m
    print('h: ',h_noc[0],h_wc[0])
    sig_noc = rate.sigmoid_temp(h_noc,0.001)
    sig_wc = rate.sigmoid_temp(h_wc,0.001)
    print('sigmoids: ',sig_noc[0],sig_wc[0])
    
    catch_h_noc = np.append(catch_h_noc,h_noc[0])
    catch_h_wc = np.append(catch_h_wc,h_wc[0])
    catch_sig_noc = np.append(catch_sig_noc,sig_noc[0])
    catch_sig_wc = np.append(catch_sig_wc,sig_wc[0])
    print('')
    
plt.plot(catch_h_noc)
plt.plot(catch_h_wc)

plt.plot(catch_sig_noc)
plt.plot(catch_sig_wc)

'''CASE 2, 
USING THE S_i = .61 from the data
'''
frac_on = 21*0.61
h_noc = w_m*frac_on + t_m
h_wc = w_m*frac_on + t_m + w_m
print('h: ',h_noc[0],h_wc[0])
sig_noc = rate.sigmoid_temp(h_noc,0.001)
sig_wc = rate.sigmoid_temp(h_wc,0.001)
print('sigmoids: ',sig_noc[0],sig_wc[0])
    
    catch_h_noc = np.append(catch_h_noc,h_noc[0])
    catch_h_wc = np.append(catch_h_wc,h_wc[0])
    catch_sig_noc = np.append(catch_sig_noc,sig_noc[0])
    catch_sig_wc = np.append(catch_sig_wc,sig_wc[0])
    print('')
    
plt.plot(catch_h_noc)
plt.plot(catch_h_wc)

plt.plot(catch_sig_noc)
plt.plot(catch_sig_wc)


'''NOTES:
--THE DIFFERENCE BETWEEN ONE UNIT AS ON OR OFF
IS JUST THE AVERAGE WEIGHT VALUE *MULTIPLIED BY C_I=1 OR ZERO
--MAIN CONCLUSION IS THAT THE STATE OF THE SYSTEM 
DRIVES WHETHER C_i MAKES A DIFFERENCE AND IT ONLY MAKES A DIFFERENCE
WHEN ABOUT 1/2 OF NODES ARE ON
--THIS IS THE SINGLE NODE ANALYSIS, HOW IS IT AFFECTED BY ON OF OF C_I
--WE COULD COMPUTE THE PROBABILITY THAT IT WOULD MAKE A DIFFERENCE, 
THAT IS MAKE H_t >0 to <0 or visa versa BUT NO NEED, POINT IS THAT WE CAN 
ANALYZE THE SYSTEM USING SAME INSIGHTS AS 
'''





#EOF
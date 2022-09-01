import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import subprocess
import multiprocessing
import logging
import time
import pickle
from imp import reload
import os
''' CUSTOM PACKAGES'''
import sys
#sys.path.insert(1,'/Users/biocomplexity/Projects/SocioCognitiveModeling/NSF-NetCogSys/Products/CAN_Model/CAN_GitRepo/IsingFitStudy/Hopfield/hop_pack')
import head as hd
import rate
import stat
reload(stat)

def convert_dec_to_array(x):
    mystr = format(x, '022b') #WITH PADDING
    return np.fromiter(mystr, dtype=int)


'''GET U, W, T'''
from numpy import genfromtxt
U_if = genfromtxt('Graphs/data_from_R.csv',delimiter=',')
W_if = genfromtxt('Graphs/graph_from_R.csv', delimiter=',')
T_if = genfromtxt('Graphs/thresholds_from_R.csv', delimiter=',')
T = T_if.copy()
W = W_if.copy()
U = U_if.copy()

''' ------------
ENERGY ANALYSIS
''' ------------

'''LOAD DATA'''
with open('SimOut/catch_energy_l_out_3075819','rb') as filehandle:
    catch_energy_l = pickle.load(filehandle)
    
'''GRAPH OUT OVER TIME BY SAMPLES'''
#SAMPLE
for i in random.sample(catch_energy_l,100): plt.plot((-1*i))
#MAKE GRAPH FOR USE WITH PANELS
fig, ([(ax1, ax2), (ax3, ax4)]) = plt.subplots(2,2)
for i in random.sample(catch_energy_l,10000): ax1.plot((-1*i))
for i in random.sample(catch_energy_l,10000): ax2.plot((-1*i))
for i in random.sample(catch_energy_l,10000): ax3.plot((-1*i))
for i in random.sample(catch_energy_l,10000): ax4.plot((-1*i))
#NEEDS A LITTLE WORK



'''END OF FILE'''
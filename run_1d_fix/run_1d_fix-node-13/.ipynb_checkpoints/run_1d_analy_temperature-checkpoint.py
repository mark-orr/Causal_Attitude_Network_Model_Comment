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


'''UNIQUE ANALYSIS'''
'''USED FOR TEMPERATURE'''
#PUT IN A LOOP
catch_intersection = []
catch_endstates = []
for (dirpath, dirname, filenames) in os.walk('SimOut/'):
    for filename in filenames:
        if filename[-4:] != '.npy' or filename[:18] != 'catch_endstate_out':
            continue
        print(filename)
        catch_endstate = np.load(f'SimOut/{filename}')
        #hd.plot_hist(catch_endstate[1:])

        #NUMBER OF END STATES
        catch_each_decimal = np.array([])
        for i in range(1,len(catch_endstate)):
            tmp0 = catch_endstate[i].astype(int)
            tmp1 = list(tmp0.astype(str))
            tmp2 = int(''.join(tmp1),2)
            catch_each_decimal = np.append(catch_each_decimal,tmp2)

        catch_each_decimal_U = np.array([])
        for i in range(0,len(U)):
            tmp0 = U[i].astype(int)
            tmp1 = list(tmp0.astype(str))
            tmp2 = int(''.join(tmp1),2)
            catch_each_decimal_U = np.append(catch_each_decimal_U,tmp2)
    
        uniq_endstates = np.unique(catch_each_decimal)
        uniq_us = np.unique(catch_each_decimal_U)
        #COMPARE AS SETS
        uniq_intersection = np.intersect1d(uniq_endstates,uniq_us)
        
        #CATCHES
        catch_intersection.append(uniq_intersection)
        catch_endstates.append(uniq_endstates)
        #PRINTS
        #print('Endstates: ', uniq_endstates)
        #print('Us: ', uniq_us)
        #print('Len(Endstate): ', len(uniq_endstates))
        #print('Len(Us): ', len(uniq_us))
        #print('Uniq Intersection: ', uniq_intersection)
        #print('Len(Uniq Inter): ', len(uniq_intersection))
print('END UNIQUE LOOP')
    
'''
ENDSTATE ANALYSIS
USED FOR TEMPERATURE
'''
tmp = catch_endstates
tmp = np.array(tmp)
tmp = np.hstack(tmp)
tmp.shape
uniq_catch_endstates = np.unique(tmp)
len(uniq_catch_endstates)

'''HOW UNIQUE ARE THE ENDSTATES VISUAL PLOT'''
'''THIS IS USED TO CAPTURE ENDSTATES FOR ANALYSIS OF TEMPERATURE'''
uniq_endstates_plot = np.full(22,99)
for i in range(0,len(uniq_catch_endstates)):
    x_1 = uniq_catch_endstates[i].astype(int)
    print(x_1)
    x_2 = convert_dec_to_array(x_1)
    uniq_endstates_plot = np.vstack((uniq_endstates_plot,x_2))

uniq_endstates_plot_use = uniq_endstates_plot[1:].copy()

'''
TEMPERATURE ANALYSIS
'''
n_updates = 1000
temperature = 0.1

#CATCHES
catch_s_hist_l = []
catch_energy_l = []
catch_endstate = np.full(22,99)

for i in uniq_endstates_plot_use:
    print(i)
    S = i.copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,n_updates,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for j in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[j],W,T))
    catch_energy_l.append(catch_energy)

for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]))  

'''
ALERT:  SOME STATES CHANGE
SEE THIS COMPARISON
'''
hd.plot_hist(uniq_endstates_plot_use) #FIRST ONES WE STARTEE WITH
hd.plot_hist(catch_endstate[1:])
hd.plot_hist(catch_s_hist_l[2][:100])

'''ADDON ANALYSIS: FORGOT TO RUN last index of all 1s'''
'''
TEMPERATURE ANALYSIS
'''
n_updates = 1000
temperature = 0.001

#CATCHES
catch_s_hist_l = []
catch_energy_l = []
catch_endstate = np.full(22,99)


    S = np.full(22,1)
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,n_updates,temperature)
    catch_s_hist_l.append(S_hist)
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
for j in range(0,len(S_hist)):
    catch_energy = np.append(catch_energy,rate.energy(S_hist[j],W,T))
catch_energy_l.append(catch_energy)

for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i])) 
catch_endstate[1]
most_freq_endstate = convert_dec_to_array(4194218)
catch_endstate[1]==most_freq_endstate


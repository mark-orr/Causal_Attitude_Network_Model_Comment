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


'''ENERGY ANALYSIS'''
#ENERGY
with open('SimOut/catch_energy_l_out_3075819','rb') as filehandle:
    catch_energy_l = pickle.load(filehandle)
#STRAIGN NUMBERS
for i in range(0,len(catch_energy_l[:10000])): plt.plot((-1*catch_energy_l[i]))    
#SAMPLE
for i in random.sample(catch_energy_l,1000): plt.plot((-1*i[800:900]))

'''UNIQUE ANALYSIS'''
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
        print('Endstates: ', uniq_endstates)
        #print('Us: ', uniq_us)
        #print('Len(Endstate): ', len(uniq_endstates))
        #print('Len(Us): ', len(uniq_us))
        #print('Uniq Intersection: ', uniq_intersection)
        print('Len(Uniq Inter): ', len(uniq_intersection))
print('END UNIQUE LOOP')
        
'''
INTERSECTION ANALYSIS
'''
tmp = catch_intersection
tmp = np.array(tmp)
tmp = np.hstack(tmp)
tmp.shape
uniq_catch_intersection = np.unique(tmp)
'''NINE UNIQUE'''

'''HOW UNIQUE ARE THE INTERSECTION ENDSTATES VISUAL PLOT'''
uniq_intersection_plot = np.full(22,99)
for i in range(0,len(uniq_catch_intersection)):
    x_1 = uniq_catch_intersection[i].astype(int)
    print(x_1)
    x_2 = convert_dec_to_array(x_1)
    uniq_intersection_plot = np.vstack((uniq_intersection_plot,x_2))
hd.plot_hist(uniq_intersection_plot[1:])  

'''HOW UNIQUE ARE THE INTERSCTION ENDSTATES VIA HAMMING'''
x = hd.ham_compare(np.ones(22),uniq_intersection_plot[1:]) #COMPARE TO ONES
plt.hist(x,bins=200)

'''WHAT IS THE ENERGY FOR EACH UNIQUE ENDSTATE'''
catch_energy = np.array([])
for i in range(0,len(uniq_catch_intersection)):
    x = uniq_catch_intersection[i].astype(int)
    y = convert_dec_to_array(x)
    catch_energy = np.append(catch_energy,rate.energy(y,W,T))
plt.hist(catch_energy,bins=100)



'''
ENDSTATE ANALYSIS
'''
tmp = catch_endstates
tmp = np.array(tmp)
tmp = np.hstack(tmp)
tmp.shape
uniq_catch_endstates = np.unique(tmp)
len(uniq_catch_endstates)


'''HOW UNIQUE ARE THE ENDSTATES VISUAL PLOT'''
uniq_endstates_plot = np.full(22,99)
for i in range(0,len(uniq_catch_endstates)):
    x_1 = uniq_catch_endstates[i].astype(int)
    print(x_1)
    x_2 = convert_dec_to_array(x_1)
    uniq_endstates_plot = np.vstack((uniq_endstates_plot,x_2))
hd.plot_hist(uniq_endstates_plot[1:])  

'''HOW UNIQUE ARE THE ENDSTATES VIA HAMMING'''
uniq_endstates_plot_use = uniq_endstates_plot[1:].copy()
x = hd.ham_compare(np.ones(22),uniq_endstates_plot_use) #COMPARE TO ONES
plt.hist(x,bins=200)

for i in uniq_endstates_plot_use:
    print(hd.ham_compare(i,uniq_endstates_plot_use))

'''WHAT IS THE ENERGY FOR EACH UNIQUE ENDSTATE'''
catch_energy = np.array([])
for i in range(0,len(uniq_catch_endstates)):
    x = uniq_catch_endstates[i].astype(int)
    y = convert_dec_to_array(x)
    catch_energy = np.append(catch_energy,rate.energy(y,W,T))
plt.hist(catch_energy,bins=100)


'''
WARNING, THIS SECTION REWRITES OBJECTS USED ABOVE
'''
'''
LAST NAIL IN COFFIN
INCREASE TEMP FOR 
'''
n_updates = 1000
temperature = 0.001

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

'''
FREQUENCY ANALYSIS
'''
'''UNIQUE ANALYSIS W FREQ'''
#PUT IN A LOOP
catch_intersection = []
catch_endstates = []
catch_endstates_freq = []
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
    
        uniq_endstates = np.unique(catch_each_decimal,return_counts=True)
        uniq_us = np.unique(catch_each_decimal_U)
        #COMPARE AS SETS
        uniq_intersection = np.intersect1d(uniq_endstates[0],uniq_us)
        
        #CATCHES
        catch_intersection.append(uniq_intersection)
        catch_endstates.append(uniq_endstates[0])
        catch_endstates_freq.append(uniq_endstates[1])
        #PRINTS
        print(i)
        #print('Endstates: ', uniq_endstates)
        #print('Us: ', uniq_us)
        #print('Len(Endstate): ', len(uniq_endstates))
        #print('Len(Us): ', len(uniq_us))
        #print('Uniq Intersection: ', uniq_intersection)
        #print('Len(Uniq Inter): ', len(uniq_intersection))
print('END UNIQUE+FREQ LOOP')
        
    
'''
ENDSTATE FREQ ANALYSIS
'''
#REMEMBER THESE ARE THE COLLECTIONS ACROSS 30 PROCESSES STACKED H
tmp_e = catch_endstates
tmp_e = np.array(tmp_e)
tmp_e = np.hstack(tmp_e)
tmp_e.shape


tmp_f = catch_endstates_freq
tmp_f = np.array(tmp_f)
tmp_f = np.hstack(tmp_f)
tmp_f.shape
tmp_f.sum() #GOOD

uniq_catch_endstates = np.unique(tmp_e,return_inverse=True)

'''FREQ AND SUM TOTAL OF FREQ FOR UNIQ ENDSTATES'''
sum_total = np.array([])
for i in range(0,(len(uniq_catch_endstates[0]))):
    u_index = i
    print('i, ',i)
    cond = uniq_catch_endstates[1]==u_index
    print("the endstate is: ", uniq_catch_endstates[0][u_index])
    print("freq of the endstate is: ",tmp_f[cond].sum())
    sum_total = np.append(sum_total,tmp_f[cond].sum())

sum_total.sum()


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
import pandas as pd

def convert_dec_to_array(x):
    mystr = format(x, '022b') #WITH PADDING
    return np.fromiter(mystr, dtype=int)

'''DEV'''
'''IDEA IS:
This is a loop, open all and one of the fix node files
then compute the weighted hamming and report, will do each
fix node file separatley
NEXT STEP:  MAKE THIS CODE A LOOP OVER fix-node-X, is df_9 below, 
should be very straightforward.
'''
#read in each analysis file
df_all = pd.read_csv('../run_1d/df_catch_endstate_analysis_out.csv')
df_all_use = df_all[['Endstate','Frequency','Energy']]
df_9 = pd.read_csv('../run_1d_fix-node-9/df_catch_endstate_analysis_out.csv')
df_9_use = df_9[['Endstate','Frequency','Energy']]

#MERGE
df_merged = pd.merge(df_all_use,df_9_use,left_on='Endstate',right_on='Endstate',suffixes=('_a','_b'),how='outer')

'''TEST OF WEIGHTED HAMMING'''
k_states = np.array(df_all_use.Endstate).astype(np.int64) #NEED np int64
k_states_freq = np.array(df_all_use.Frequency) #is np int64
#NORM THE FREQ
k_states_freq_norm = k_states_freq / k_states_freq.sum()

uniq_endstates_decimal = np.array(df_9_use.Endstate)
uniq_endstates_decimal_freq = np.array(df_9_use.Frequency)#is int64 as should be
#uniq_endstates_decimal = k_states
#uniq_endstates_decimal_freq = k_states_freq
'''COMPUTE WEIGHTED HAMMING'''
catch_weighted_ham_node = np.array([])
catch_min_h_vect = np.array([])
catch_min_f_vect = np.array([])
catch_hammings = np.zeros(len(uniq_endstates_decimal))
catch_freqs = np.zeros(len(uniq_endstates_decimal))
for i in k_states:
    print('')
    print('')
    print('')
    print('k state: ', i)
    k_state = convert_dec_to_array(i)
    print('k state[arr]: ',k_state)
    h_vect = np.array([])
    f_vect = np.array([])
    for k in range(0,len(uniq_endstates_decimal)):
        e_state = convert_dec_to_array(uniq_endstates_decimal[k].astype(int))
        print('e state: ',uniq_endstates_decimal[k])
        print('e state[arr]: ', e_state)
        #print('HAMMING: ', np.sum(np.bitwise_xor(k_state.astype(int),e_state.astype(int))))
        #print('FREQ: ', uniq_endstates_decimal_freq[k])
        h_vect = np.append(h_vect,np.sum(np.bitwise_xor(k_state.astype(int),e_state.astype(int))))
        print('h_vect: ',h_vect)
        f_vect = np.append(f_vect,uniq_endstates_decimal_freq[k])
        print('f_vect: ',f_vect)
    
    catch_hammings = np.vstack((catch_hammings,h_vect))
    catch_freqs = np.vstack((catch_freqs,f_vect))
    
catch_hammings_arr = catch_hammings[1:]
catch_freqs_arr = catch_freqs[1]

'''TEST'''
#TEST ARRAYS
#catch_hammings_arr = np.array([[1,3,2],[2,2,2],[3,2,2]])
#f_vect = np.array([10,20,6])
#THESE WORK AND GIVE SUM OF 30 for arr2.
'''END TEST'''

'''NEW METHOD'''
#FIRST LOOP TO FREQ DIST BY MIN HAMM PER COL
arr1 = np.zeros(catch_hammings_arr.shape)
for i in range(0,arr1.shape[1]): #looping over cols
    
    loc_min_of_col = np.where(catch_hammings_arr[:,i]==min(catch_hammings_arr[:,i]))
    list_loc = list(loc_min_of_col[0])
    arr1[list_loc,i] = catch_freqs_arr[i]/len(list_loc)
    
#SECOND LOOP TO GET FREQ DIST BY MIN HAMM PER ROW 
arr2 = np.array([]) #EACH INDEX VALUE IS A ROW IN arr1,catch_hammings_arr
arr3 = np.array([]) # ''''''
for i in range(0,arr1.shape[0]):
    loc_min_of_row = np.where(catch_hammings_arr[i,:]==min(catch_hammings_arr[i,:]))
    list_loc = list(loc_min_of_row)
    arr2 = np.append(arr2,arr1[i,list_loc].sum())
    arr3 = np.append(arr3,catch_hammings_arr[i,list_loc].mean())

'''MEAN HAMMING, WEIGHTED AND NORMALIZED BY HIT SUMS NOT NECESSARILY TOTAL SUM'''
np.dot(arr2,arr3)/arr2.sum()

'DIST NORMALIZED BY FULL DENOM'
#arr3 = arr2/36#FOR TEST 
arr4 = arr2/(2**22)
'''END NEW'''

'''COMPARE DISTS'''
pd.DataFrame([arr4,k_states_freq_norm]).round(4).T

'''NORM BOTH VECTOR DISTRIBUTIONS'''
arr4_len = np.sqrt(np.dot(arr4,arr4))
arr4_norm = arr4/arr4_len
#CHECK
np.sqrt(np.dot(arr4_norm,arr4_norm))#should equal 1

k_states_freq_norm_len = np.sqrt(np.dot(k_states_freq_norm,k_states_freq_norm))
k_states_freq_norm_norm = k_states_freq_norm/k_states_freq_norm_len
np.sqrt(np.dot(k_states_freq_norm_norm,k_states_freq_norm_norm))

'''COMPUTE DIST BETWEEN THE NORMED VECTORS'''
v1_min_v2 = k_states_freq_norm_norm - arr4_norm
v1_d_v2 = np.sqrt(np.dot(v1_min_v2,v1_min_v2))

'''NEXT TASK? SEE HEADER.'''
#EOF
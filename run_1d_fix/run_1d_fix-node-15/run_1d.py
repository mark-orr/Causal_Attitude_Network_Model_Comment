import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import subprocess
import multiprocessing
import logging
import time
import pickle
import sys
import head as hd
import rate

#IMPORT DATA
#LOAD CSV TO PYTHON
#CONVERT TO W_if
from numpy import genfromtxt
W_if = genfromtxt('Graphs/graph_from_R.csv', delimiter=',')
T_if = genfromtxt('Graphs/thresholds_from_R.csv', delimiter=',')
U_if = genfromtxt('Graphs/data_from_R.csv',delimiter=',')
print('DONE loading W, T, U')

#PARAMS SAME ACROSS SIMS
N = len(T_if) 
n_U = len(U_if);
n_updates = 1000
temperature = 0.001
W = W_if.copy()
T = T_if.copy()

'''
MAIN:
IS CALLED AS SEPARATE PARALLEL PROCESSES
CHANGE PAPAMS INSIDE, NOTHING PASSED TO IT BUT
TASK_ID FOR PARALLEL PROCESS
'''
def single_proc(task_id):
    '''
    RETURNS None
    GENERATES SETS OF SIM RESULTS, ONE SET PER INPUT 2-tuple
    task_id expects a list of 2-tuples
    '''
    print('Process Single_Proc Started: ',task_id)
    #LOOPING OVER
    init_value = task_id[0] 
    end_value = task_id[1]
    
    #CATCHES
    catch_endstate = np.full(22,99)
    catch_energy_l = []

    for i in range(init_value,end_value):

        S = np.array(list(format(i,'022b')))
        S = S.astype(int)
        T_clamp = T.copy()
        T_clamp[15] = 1000
        S[15] = 1 #FIX S FOR Node     
        S_hist = rate.sim_patt_U_prob_rate_if(S,W,T_clamp,n_updates,temperature)
    
        catch_endstate = np.vstack((catch_endstate,S))
    
        catch_energy = np.array([])
        for j in range(0,len(S_hist)):
            catch_energy = np.append(catch_energy,rate.energy(S_hist[j],W,T_clamp))
        catch_energy_l.append(catch_energy)
    
        if i == 1 or i % 10000 == 0:
            print('Subproc i: ', i)

    print('END MAIN LOOP')
    print('SAVE OUT')
    np.save(f'SimOut/catch_endstate_out_{i}',catch_endstate)

    with open(f'SimOut/catch_energy_l_out_{i}','wb') as filehandle:
        pickle.dump(catch_energy_l, filehandle)
    
    print("SINGLE PROC DONE",task_id)
    
    return None

'''
MULTIPROCESSOR
'''
def multi_proc(l_task_ids):
    '''
    RETURNS WHATEVER target=single_proc returns
    USE CASE IS THAT l_task_ids is a list of 2-tuples for which
    the first element is the beginning index and second is ending index
    for a loop over integers.  See def single_proc
    '''
    logging.debug('[multi_proc] Starts.')
    timer_start = time.time()
    l_proc = []

    for task_id in l_task_ids:
        p = multiprocessing.Process(target=single_proc,
                                    args=(task_id,),
                                    name='Proc ' + str(task_id))
        p.start()
        print("PROCESS STARTED NAMED: ", task_id)
        l_proc.append(p)
        
    while len(l_proc) > 0:
        for p in l_proc:
            if p.is_alive():
                p.join(1)
            else:
                l_proc.remove(p)
                logging.debug('[multi_proc] %s is finished.' % p.name)
    logging.debug('[multi_proc] All done in %s secs.' % str(time.time() - timer_start))
    

'''
GENERATE SIMULATION LISTS
'''
l_task_ids = []
for i in range(0,30):
    k=i+1
    fact = 139810
    print('init is:',fact*i)
    print('end is:',fact*k)
    l_task_ids.append((fact*i,fact*k))

#APPEND LAST VALUE OF LAST TUPLE
l_task_ids[29]=(l_task_ids[29][0],4194303)

'''
RUN SIMS
'''
#RUN SIMS
multi_proc(l_task_ids)

print("END OF PROGRAM")
print("GO HOME")
#EOF
